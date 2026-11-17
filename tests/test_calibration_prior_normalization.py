"""The spectral calibration prior enters the objective (sum NLL + prior) / N."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.synthetic_trainer import make_image_spec_trainer  # noqa: E402
from survey_fusion.data.bandpasses import Bandpass  # noqa: E402
from survey_fusion.training.spectral_calibration import (  # noqa: E402
    SpectralCalibrationConfig, SpectralNuisanceCalibrator,
)


def _calibrator(normalization: str) -> SpectralNuisanceCalibrator:
    lam = np.array([4000.0, 5000.0, 6000.0])
    bandpasses = {
        band: Bandpass(band=band, lam=lam, throughput=np.ones(3), canonical_id=band) for band in ("g", "r")
    }
    cfg = SpectralCalibrationConfig(
        enabled=True, prior_sigma_log=0.15, prior_weight=2.0, prior_normalization=normalization,
    )
    calibrator = SpectralNuisanceCalibrator(
        spectral_lam_by_exposure={1: np.array([5000.0])}, bandpasses=bandpasses, cfg=cfg,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        calibrator.log_scales.copy_(torch.tensor([0.03, -0.06]))
    return calibrator


def test_legacy_prior_keeps_the_band_mean() -> None:
    expected = 2.0 * 0.5 * ((0.03 / 0.15) ** 2 + (0.06 / 0.15) ** 2) / 2
    assert float(_calibrator("legacy").prior_penalty()) == pytest.approx(expected, rel=1e-6)


def test_objective_prior_sums_the_bands() -> None:
    expected = 2.0 * 0.5 * ((0.03 / 0.15) ** 2 + (0.06 / 0.15) ** 2)
    assert float(_calibrator("objective").prior_penalty()) == pytest.approx(expected, rel=1e-6)


def test_unknown_prior_normalization_is_refused() -> None:
    with pytest.raises(ValueError, match="prior_normalization"):
        _calibrator("per_band")


def _trainer(normalization: str, denominator: int | None):
    calibration = {
        "spectral_nuisance": {
            "enabled": True, "anchor": "image", "mapping": "throughput_mix", "prior_sigma_log": 0.15,
            "prior_weight": 1.0, "init_scale": 1.2, "lr_multiplier": 1.0, "apply_to_predictions": True,
            "prior_normalization": normalization,
        }
    }
    cfg = {"calibration": calibration}
    if denominator is not None:
        cfg["objective_denominator"] = denominator
    return make_image_spec_trainer(trainer_cfg=cfg)


def _step_prior(trainer) -> float:
    stats = trainer._train_step(torch.tensor([0]), torch.tensor([1]))
    return float(stats["calibration_prior"])


def test_objective_prior_is_divided_by_the_configured_denominator() -> None:
    trainer = _trainer("objective", 1000)
    assert trainer._objective_denominator() == 1000
    n_bands = len(trainer.spectral_calibrator.band_names)
    expected = n_bands * 0.5 * (math.log(1.2) / 0.15) ** 2 / 1000
    assert _step_prior(trainer) == pytest.approx(expected, rel=1e-5)


def test_objective_denominator_defaults_to_the_training_population() -> None:
    trainer = _trainer("objective", None)
    assert trainer._objective_denominator() == 2
    n_bands = len(trainer.spectral_calibrator.band_names)
    expected = n_bands * 0.5 * (math.log(1.2) / 0.15) ** 2 / 2
    assert _step_prior(trainer) == pytest.approx(expected, rel=1e-5)


def test_legacy_prior_ignores_the_denominator() -> None:
    trainer = _trainer("legacy", 1000)
    expected = 0.5 * (math.log(1.2) / 0.15) ** 2
    assert _step_prior(trainer) == pytest.approx(expected, rel=1e-5)


def test_objective_denominator_must_be_positive() -> None:
    with pytest.raises(ValueError, match="objective_denominator"):
        _trainer("objective", 0)


def test_calibration_is_held_until_the_freeze_step_then_released() -> None:
    calibration = {
        "spectral_nuisance": {
            "enabled": True, "anchor": "image", "mapping": "throughput_mix", "prior_sigma_log": 0.15,
            "prior_weight": 1.0, "init_scale": 1.2, "lr_multiplier": 1.0, "apply_to_predictions": True,
            "prior_normalization": "objective", "freeze_until_step": 2,
        }
    }
    trainer = make_image_spec_trainer(trainer_cfg={"calibration": calibration}, y=torch.tensor([5.0, 40.0]))
    scales = lambda: trainer.spectral_calibrator.log_scales.detach().clone()
    start = scales()
    trainer._set_calibration_lr_for_update(1)
    for update in (1, 2):
        trainer._train_step(torch.tensor([0]), torch.tensor([1]))
        trainer._post_optimizer_step(update)
        assert torch.equal(scales(), start), update
    trainer._train_step(torch.tensor([0]), torch.tensor([1]))
    assert not torch.equal(scales(), start)


def test_no_freeze_leaves_the_calibration_rate_alone() -> None:
    trainer = _trainer("objective", 1000)
    group = next(g for g in trainer.optimizer.param_groups
                 if any(p is trainer.spectral_calibrator.log_scales for p in g["params"]))
    rate = group["lr"]
    trainer._set_calibration_lr_for_update(1)
    trainer._post_optimizer_step(1)
    assert group["lr"] == rate
