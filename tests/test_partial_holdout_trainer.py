from __future__ import annotations

import numpy as np
import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory, OnesModel as ExposureValueModel, SyntheticStore as MockStore, image_exposure,
    make_batch,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer


def _make_store(device: torch.device) -> MockStore:
    batch = make_batch(
        y=torch.zeros(8, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=device),
        datum_kind=torch.zeros(8, dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long, device=device),
    )
    return MockStore(
        exposures={0: image_exposure(band="g"), 1: image_exposure(band="i")},
        batch=batch,
        image_band_by_exposure={0: "g", 1: "i"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=8,
        n_spec=0,
        image_excluded_local_indices_by_exposure={1: np.array([0, 1], dtype=np.int64)},
        partial_holdout_enabled=True,
    )


def test_partial_holdout_excluded_pixels_removed_from_training_indices() -> None:
    device = torch.device("cpu")
    trainer = DatumTrainer(
        model=ExposureValueModel(),
        grid=build_spatial_grid(2, 2, device),
        data=_make_store(device),
        training_bands=["g", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 1,
            "epochs": 0,
            "learning_rate": 1e-3,
            "apply_spatial_psf": False,
        },
        device=device,
    )
    assert int(trainer.image_train_indices.numel()) == 6
    i_selected = trainer.image_train_indices[trainer.data.batch.exposure_id[trainer.image_train_indices] == 1]
    i_local = torch.index_select(trainer.data.batch.datum_index, 0, i_selected).detach().cpu().tolist()
    assert sorted(i_local) == [2, 3]


def test_partial_holdout_requires_held_out_band_in_training_bands() -> None:
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="held_out_band"):
        DatumTrainer(
            model=ExposureValueModel(),
            grid=build_spatial_grid(2, 2, device),
            data=_make_store(device),
            training_bands=["g"],
            held_out_band="i",
            lambda_step=20.0,
            trainer_cfg={
                "kernel_samples_k": 8,
                "replica_samples_k": 1,
                "epochs": 0,
                "learning_rate": 1e-3,
                "apply_spatial_psf": False,
            },
            device=device,
        )


def test_sample_point_spectra_uses_requested_image_exposure() -> None:
    device = torch.device("cpu")
    trainer = DatumTrainer(
        model=ExposureValueModel(),
        grid=build_spatial_grid(2, 2, device),
        data=_make_store(device),
        training_bands=["g", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 1,
            "epochs": 0,
            "learning_rate": 1e-3,
            "apply_spatial_psf": False,
        },
        device=device,
    )
    trainer.runtime_by_exposure[0].omega_xyz = torch.tensor(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    trainer.runtime_by_exposure[0].fiber_indices = torch.empty((0,), dtype=torch.long, device=device)
    trainer.runtime_by_exposure[1].omega_xyz = torch.tensor(
        [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0], [23.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    trainer.runtime_by_exposure[1].fiber_indices = torch.empty((0,), dtype=torch.long, device=device)

    recorded: list[tuple[int, np.ndarray]] = []

    def _capture_eval(exposure_id: int, omega: torch.Tensor, *, renderer=None) -> torch.Tensor:
        del renderer
        recorded.append((int(exposure_id), omega.detach().cpu().numpy().copy()))
        return torch.ones((omega.shape[0], 2), dtype=torch.float32, device=device)

    trainer._evaluate_spectra_for_omega = _capture_eval  # type: ignore[method-assign]
    trainer._project_spectra_to_band_fluxes = (  # type: ignore[method-assign]
        lambda spectra: {"g": np.ones((spectra.shape[0],), dtype=np.float32)}
    )

    samples = trainer.sample_point_spectra([("p", 0, 1)], exposure_id=1)

    assert len(samples) == 1
    assert recorded[0][0] == 1
    np.testing.assert_allclose(recorded[0][1], np.array([[21.0, 0.0, 0.0]], dtype=np.float32))
