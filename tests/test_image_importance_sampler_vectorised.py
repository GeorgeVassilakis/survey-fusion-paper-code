"""The batched importance sampler must reproduce the per-exposure one exactly.

The per-exposure form drew `count` float64 uniforms per non-empty exposure, in
ascending exposure order, from one generator. The batched form draws them all
at once, so these tests pin the two properties that makes legal: torch's CPU
float64 stream is chunk-invariant, and the batched searchsorted lands every
draw on the local index the per-exposure search gave it.
"""

from __future__ import annotations

import numpy as np
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory,
    OnesModel,
    SyntheticStore,
    image_exposure,
    make_batch,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer


EXPOSURE_SHAPES = ((3, 5), (2, 7), (4, 4), (2, 2))


def _multi_exposure_trainer(*, pixel_batch_size: int = 40) -> DatumTrainer:
    device = torch.device("cpu")
    rng = np.random.default_rng(7)
    values: list[float] = []
    exposure_ids: list[int] = []
    datum_index: list[int] = []
    for exp_id, (height, width) in enumerate(EXPOSURE_SHAPES):
        n = height * width
        # A few bright pixels per exposure, so the local CDFs are far from flat
        # and a mis-set search bound would move the sampled index.
        frame = rng.gamma(0.4, 3.0, size=n)
        values.extend(float(v) for v in frame)
        exposure_ids.extend([exp_id] * n)
        datum_index.extend(range(n))
    n_image = len(values)
    batch = make_batch(
        y=torch.tensor(values, dtype=torch.float32, device=device),
        sigma=torch.ones(n_image, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(exposure_ids, dtype=torch.long, device=device),
        datum_kind=torch.zeros(n_image, dtype=torch.long, device=device),
        datum_index=torch.tensor(datum_index, dtype=torch.long, device=device),
    )
    store = SyntheticStore(
        exposures={
            exp_id: image_exposure(height=height, width=width)
            for exp_id, (height, width) in enumerate(EXPOSURE_SHAPES)
        },
        batch=batch,
        image_band_by_exposure={exp_id: "r" for exp_id in range(len(EXPOSURE_SHAPES))},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=n_image,
        n_spec=0,
    )
    return DatumTrainer(
        model=OnesModel(),
        grid=build_spatial_grid(4, 4, device),
        data=store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 4,
            "replica_samples_k": 1,
            "epochs": 1,
            "learning_rate": 1e-3,
            "pixel_batch_size": pixel_batch_size,
            "spec_batch_size": 0,
            "apply_spatial_psf": False,
            "image_importance_sampling": {"enabled": True, "max_importance_weight": 5.0},
        },
        device=device,
    )


def _serial_reference_batch(
    trainer: DatumTrainer, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-exposure sampler, transcribed: one draw call per non-empty
    exposure in ascending exposure order, concatenated exposure-major."""
    batch_size = trainer._resolve_image_batch_size()
    exposure_choice = trainer._sample_cdf(
        trainer.image_importance_exposure_cdf, batch_size, generator=generator
    )
    sampled_indices: list[torch.Tensor] = []
    sampled_weights: list[torch.Tensor] = []
    counts = torch.bincount(
        exposure_choice, minlength=len(trainer.image_importance_exposure_ids)
    )
    for choice_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        exp_id = int(trainer.image_importance_exposure_ids[int(choice_idx)])
        local_choice = trainer._sample_cdf(
            trainer.image_importance_local_cdf_by_exposure[exp_id],
            int(count),
            generator=generator,
        )
        sampled_indices.append(
            trainer.image_train_indices_by_exposure[exp_id][local_choice].to(
                dtype=torch.long, device="cpu"
            )
        )
        sampled_weights.append(
            trainer.image_importance_correction_by_exposure[exp_id][local_choice].to(
                dtype=torch.float32, device="cpu"
            )
        )
    return torch.cat(sampled_indices, dim=0), torch.cat(sampled_weights, dim=0)


def test_cpu_float64_uniform_stream_is_chunk_invariant() -> None:
    for total, splits in (
        (16, (7, 9)),
        (17, (1, 16)),
        (64, (1,) * 64),
        (10_000, (24,) * 416 + (16,)),
    ):
        whole_generator = torch.Generator(device="cpu")
        whole_generator.manual_seed(20260905)
        whole = torch.rand((total,), generator=whole_generator, dtype=torch.float64)

        chunk_generator = torch.Generator(device="cpu")
        chunk_generator.manual_seed(20260905)
        chunked = torch.cat(
            [
                torch.rand((n,), generator=chunk_generator, dtype=torch.float64)
                for n in splits
            ]
        )

        assert torch.equal(whole, chunked)
        assert torch.equal(whole_generator.get_state(), chunk_generator.get_state())


def test_vectorised_sampler_reproduces_the_serial_draw_sequence() -> None:
    trainer = _multi_exposure_trainer()
    assert trainer.image_importance_sampling_enabled

    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(20260905)
    trainer.image_importance_generator.manual_seed(20260905)

    for _ in range(200):
        reference_indices, reference_weights = _serial_reference_batch(
            trainer, reference_generator
        )
        indices, weights = trainer._sample_importance_image_batch()
        assert torch.equal(indices, reference_indices)
        assert torch.equal(weights, reference_weights)

    assert torch.equal(
        reference_generator.get_state(),
        trainer.image_importance_generator.get_state(),
    )


def test_vectorised_sampler_matches_when_one_exposure_takes_the_whole_batch() -> None:
    # A batch that lands entirely in one exposure leaves every other row of the
    # padded search empty, which is the degenerate shape of the batched form.
    trainer = _multi_exposure_trainer(pixel_batch_size=1)
    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(11)
    trainer.image_importance_generator.manual_seed(11)

    for _ in range(200):
        reference_indices, reference_weights = _serial_reference_batch(
            trainer, reference_generator
        )
        indices, weights = trainer._sample_importance_image_batch()
        assert torch.equal(indices, reference_indices)
        assert torch.equal(weights, reference_weights)


def test_per_exposure_tables_are_views_holding_the_documented_values() -> None:
    trainer = _multi_exposure_trainer()

    for exp_id in trainer.image_importance_exposure_ids:
        probs = trainer.image_importance_local_probs_by_exposure[exp_id]
        n_local = int(probs.numel())
        assert torch.equal(
            trainer.image_importance_local_cdf_by_exposure[exp_id],
            DatumTrainer._probs_to_cdf(probs),
        )
        assert torch.equal(
            trainer.image_importance_correction_by_exposure[exp_id],
            1.0 / (float(n_local) * probs),
        )
        assert n_local == int(trainer.image_train_indices_by_exposure[exp_id].numel())

    offsets = trainer._image_importance_offsets.tolist()
    for slot, exp_id in enumerate(trainer.image_importance_exposure_ids):
        n_local = int(trainer.image_train_indices_by_exposure[exp_id].numel())
        start = offsets[slot]
        assert torch.equal(
            trainer._image_importance_index_flat[start : start + n_local],
            trainer.image_train_indices_by_exposure[exp_id],
        )
        # Padding sits past every draw, so it can never be selected.
        assert bool(
            (trainer._image_importance_cdf_matrix[slot, n_local:] > 1.0).all()
        )
