from __future__ import annotations

import numpy as np
import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory, OnesModel as ConstantModel, SyntheticStore as MockStore, image_exposure,
    make_batch, spec_exposure,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.distributed import DistributedContext


StandardForwardConstantModel = ConstantModel


def _make_trainer(
    image_values: list[float], *, sigma_values: list[float] | None = None,
    additive_values: list[float] | None = None, trainer_overrides: dict | None = None,
    distributed_context: DistributedContext | None = None,
) -> DatumTrainer:
    device = torch.device("cpu")
    image_np = np.asarray(image_values, dtype=np.float32).reshape(2, 2)
    sigma_np = (
        np.ones_like(image_np, dtype=np.float32)
        if sigma_values is None
        else np.asarray(sigma_values, dtype=np.float32).reshape(2, 2)
    )
    additive_np = (
        None
        if additive_values is None
        else np.asarray(additive_values, dtype=np.float32).reshape(2, 2)
    )
    batch = make_batch(
        y=torch.tensor(image_np.reshape(-1), dtype=torch.float32, device=device),
        sigma=torch.tensor(sigma_np.reshape(-1), dtype=torch.float32, device=device),
        exposure_id=torch.zeros(4, dtype=torch.long, device=device),
        datum_kind=torch.zeros(4, dtype=torch.long, device=device),
        datum_index=torch.arange(4, dtype=torch.long, device=device),
        additive=(
            None
            if additive_np is None
            else torch.tensor(additive_np.reshape(-1), dtype=torch.float32, device=device)
        ),
    )
    store = MockStore(
        exposures={0: image_exposure()}, batch=batch, image_band_by_exposure={0: "r"},
        kernel_factory=ConstantKernelFactory(batch), n_image=4, n_spec=0,
    )
    trainer_cfg = {
        "kernel_samples_k": 4, "replica_samples_k": 1, "epochs": 1, "learning_rate": 1e-3,
        "pixel_batch_size": 2, "spec_batch_size": 0, "apply_spatial_psf": False,
        "image_importance_sampling": {"enabled": True, "max_importance_weight": 5.0},
    }
    if trainer_overrides:
        trainer_cfg.update(trainer_overrides)
    return DatumTrainer(
        model=StandardForwardConstantModel()
        if distributed_context and distributed_context.enabled
        else ConstantModel(),
        grid=build_spatial_grid(2, 2, device),
        data=store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg=trainer_cfg,
        device=device,
        distributed_context=distributed_context,
    )


def _make_mixed_trainer() -> DatumTrainer:
    device = torch.device("cpu")
    batch = make_batch(
        y=torch.tensor([10.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 0, 0, 0, 1], dtype=torch.long, device=device),
        datum_kind=torch.tensor([0, 0, 0, 0, 1], dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device=device),
    )
    store = MockStore(
        exposures={0: image_exposure(), 1: spec_exposure(psf_fwhm_arcsec=1.0)}, batch=batch,
        image_band_by_exposure={0: "r"}, kernel_factory=ConstantKernelFactory(batch), n_image=4, n_spec=1,
    )
    return DatumTrainer(
        model=ConstantModel(),
        grid=build_spatial_grid(2, 2, device),
        data=store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 4, "replica_samples_k": 1, "epochs": 1, "learning_rate": 1e-3,
            "pixel_batch_size": 2, "spec_batch_size": 1, "apply_spatial_psf": False,
            "image_importance_sampling": {"enabled": True, "max_importance_weight": 5.0},
        },
        device=device,
    )


def test_image_importance_sampling_parser_rejects_removed_fields() -> None:
    cfg = DatumTrainer._parse_image_importance_sampling_cfg({"enabled": True, "max_importance_weight": 5.0})

    assert cfg.enabled is True
    assert cfg.max_importance_weight == pytest.approx(5.0)

    with pytest.raises(ValueError, match="removed/unsupported keys"):
        DatumTrainer._parse_image_importance_sampling_cfg({"obsolete_sampler_key": True})
    with pytest.raises(ValueError, match="removed/unsupported keys"):
        DatumTrainer._parse_image_importance_sampling_cfg({"score_power": 1.0})
    with pytest.raises(ValueError, match="removed/unsupported keys"):
        DatumTrainer._parse_image_importance_sampling_cfg({"smoothing": "none"})


def test_image_importance_sampling_builds_nonuniform_probabilities() -> None:
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])

    assert trainer.image_importance_sampling_enabled
    assert trainer.image_loader is None
    probs = trainer.image_importance_local_probs_by_exposure[0].detach().cpu().numpy()
    corrections = trainer.image_importance_correction_by_exposure[0].detach().cpu().numpy()

    assert trainer._steps_per_epoch() == 2
    assert probs[0] > np.max(probs[1:])
    assert float(np.max(corrections)) <= 5.0 + 1.0e-6


def test_image_importance_sampling_scores_source_proxy_when_additive_present() -> None:
    trainer = _make_trainer([20.0, 10.0, 0.0, 0.0], additive_values=[20.0, 0.0, 0.0, 0.0])

    probs = trainer.image_importance_local_probs_by_exposure[0].detach().cpu().numpy()

    assert probs[1] > probs[0]
    assert probs[1] > np.max(probs[[0, 2, 3]])


def test_image_importance_score_does_not_floor_positive_uncertainties() -> None:
    score = DatumTrainer._build_source_proxy_importance_score(
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 1.0e-20]),
    )

    assert score[1] > 1.0e18


def test_importance_sampled_image_estimator_preserves_uniform_image_mean() -> None:
    torch.manual_seed(1234)
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    estimates: list[float] = []

    for _ in range(20000):
        image_indices, image_weights = trainer._sample_importance_image_batch()
        estimate = float((losses[image_indices] * image_weights.cpu()).mean().item())
        estimates.append(estimate)

    assert np.mean(estimates) == pytest.approx(float(losses.mean().item()), abs=0.08)


def test_training_image_datum_count_reports_materialized_image_rows() -> None:
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])

    assert trainer.training_image_datum_count() == 4


def test_image_logging_summary_reports_materialized_rows() -> None:
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])

    assert trainer.image_training_summary() == {
        "n_image_datums": 4, "n_materialized_image_datums": 4, "n_total_image_datums": 4,
    }
    assert trainer.image_sampling_summary() == {
        "importance_sampling_enabled": True, "mode": "materialized", "max_importance_weight": 5.0,
    }


def test_scientific_modality_summary_uses_training_population_counts() -> None:
    trainer = _make_mixed_trainer()

    assert trainer.scientific_modality_summary() == {
        "mode": "full_training_population_mean_nll",
        "image_population_count": 4,
        "spectrum_population_count": 1,
        "spectral_bin_weight": 5.0,
        "image_weight": pytest.approx(0.8),
        "spectrum_weight": pytest.approx(5.0 * 0.2),
    }


def test_train_step_applies_image_importance_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])
    image_indices = torch.tensor([0, 1], dtype=torch.long)
    image_weights = torch.tensor([2.0, 3.0], dtype=torch.float32, device=trainer.device)

    def _fake_terms(global_indices: torch.Tensor) -> torch.Tensor:
        del global_indices
        return torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer.device, requires_grad=True)

    monkeypatch.setattr(trainer, "_compute_mc_unbiased_gaussian_terms", _fake_terms)

    stats = trainer._train_step(
        image_indices, torch.empty((0,), dtype=torch.long), image_weights=image_weights,
    )

    assert stats["local_image_nll"] == pytest.approx(8.0)


def test_importance_sampling_steps_per_epoch_uses_global_ddp_image_batch() -> None:
    trainer = _make_trainer(
        [10.0, 0.0, 0.0, 0.0],
        trainer_overrides={"pixel_batch_size": 1},
        distributed_context=DistributedContext(
            enabled=True, backend="gloo", world_size=2, rank=0, local_rank=0, master_addr="127.0.0.1",
            master_port=29500,
        ),
    )

    assert trainer.image_importance_sampling_enabled
    assert trainer.image_batch_size == 1
    assert trainer._resolve_effective_global_image_batch_size() == 2
    assert trainer._steps_per_epoch() == 2


def test_iter_epoch_batches_cycles_spec_loader_with_importance_sampled_images() -> None:
    trainer = _make_mixed_trainer()

    batches = list(trainer._iter_epoch_batches())

    assert trainer._steps_per_epoch() == 2
    assert len(batches) == 2
    assert all(img is not None for img, _ in batches)
    assert all(spec is not None for _, spec in batches)
    assert [int(img.numel()) for img, _ in batches] == [2, 2]
    assert [int(spec.numel()) for _, spec in batches] == [1, 1]
    assert torch.equal(batches[1][1], batches[0][1])


def test_iter_epoch_batches_does_not_cycle_empty_spec_loader_with_importance_images() -> None:
    trainer = _make_trainer([10.0, 0.0, 0.0, 0.0])
    trainer.spec_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.empty((0,), dtype=torch.long)), batch_size=1, drop_last=True,
    )

    batches = list(trainer._iter_epoch_batches())

    assert trainer._steps_per_epoch() == 2
    assert len(batches) == 2
    assert all(img is not None for img, _ in batches)
    assert all(spec is None for _, spec in batches)
