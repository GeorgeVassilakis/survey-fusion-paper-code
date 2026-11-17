"""Lifecycle thresholds from fitted image data only (trainer.refine.lifecycle_statistics)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.synthetic_trainer import (  # noqa: E402
    ConstantKernelFactory, SyntheticStore, image_exposure, make_batch, spec_exposure,
)
from survey_fusion.data.grids import build_spatial_grid  # noqa: E402
from survey_fusion.models.splat_field import SplatField  # noqa: E402
from survey_fusion.models.splat_strategy import psf_based_grow_scale  # noqa: E402
from survey_fusion.pgm.contracts import DatumKind  # noqa: E402
from survey_fusion.training.datum_trainer import DatumTrainer  # noqa: E402

IMAGE, SPECTRUM = int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)


def _trainer(statistics: str | None, exclude=None) -> DatumTrainer:
    torch.manual_seed(3)
    field = SplatField(n_splats=4, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4))
    batch = make_batch(
        y=torch.zeros(12),
        sigma=torch.tensor([0.01, 0.01, 1000.0, 0.02, 0.02, 0.02] + [50.0] * 6),
        exposure_id=torch.tensor([0, 0, 0, 1, 1, 1] + [2] * 6),
        datum_kind=torch.tensor([IMAGE] * 6 + [SPECTRUM] * 6),
        datum_index=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5]),
    )
    store = SyntheticStore(
        exposures={
            0: image_exposure(band="r", psf_fwhm_arcsec=1.0),
            1: image_exposure(band="r", psf_fwhm_arcsec=2.0),
            2: spec_exposure(psf_fwhm_arcsec=5.0, lam_spec=np.linspace(5000.0, 5500.0, 6)),
        },
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=6,
        n_spec=6,
        image_random_holdout_indices_by_exposure={0: np.array([2])},
        image_excluded_local_indices_by_exposure={0: np.array([2])},
    )
    refine = {"enabled": True, "start": 0, "stop": 100, "every": 10}
    if statistics is not None:
        refine["lifecycle_statistics"] = statistics
    return DatumTrainer(
        model=field, grid=build_spatial_grid(2, 2, torch.device("cpu")), data=store,
        training_bands=["r"], held_out_band=None, lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": 0, "learning_rate": 1e-3,
            "apply_spatial_psf": False, "refine": refine,
        },
        device=torch.device("cpu"), exclude_image_exposure_ids=exclude,
    )


def _grow_scale(trainer: DatumTrainer, psf: float) -> float:
    return psf_based_grow_scale(psf, trainer._field_width, 0.396)


def test_legacy_statistics_pool_spectra_and_held_pixels() -> None:
    trainer = _trainer("legacy")
    assert trainer.refine_strategy.reference_sigma == pytest.approx(50.0)
    assert trainer.refine_strategy.config.grow_scale == pytest.approx(_grow_scale(trainer, 2.0))


def test_fitted_image_statistics_are_the_default() -> None:
    assert _trainer(None).refine_strategy.reference_sigma == pytest.approx(
        _trainer("fitted_images").refine_strategy.reference_sigma
    )


def test_fitted_image_statistics_use_only_fitted_image_pixels() -> None:
    trainer = _trainer("fitted_images")
    assert trainer.refine_strategy.reference_sigma == pytest.approx(0.02)
    assert trainer.refine_strategy.config.grow_scale == pytest.approx(_grow_scale(trainer, 1.5))


def test_fitted_image_statistics_drop_withheld_exposures() -> None:
    trainer = _trainer("fitted_images", exclude=[1])
    assert trainer.refine_strategy.reference_sigma == pytest.approx(0.01)
    assert trainer.refine_strategy.config.grow_scale == pytest.approx(_grow_scale(trainer, 1.0))


def test_unknown_lifecycle_statistics_are_refused() -> None:
    with pytest.raises(ValueError, match="lifecycle_statistics"):
        _trainer("all")
