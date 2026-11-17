from __future__ import annotations

import numpy as np
import torch

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.training.spectral_calibration import(SpectralCalibrationConfig, SpectralNuisanceCalibrator)


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.array([4000.0, 5000.0, 6000.0], dtype=np.float64)
    throughput = np.array([0.2, 1.0, 0.2], dtype=np.float64)
    return {"r": Bandpass(band="r", lam=lam, throughput=throughput, canonical_id="test_r")}


def test_throughput_mix_maps_to_unity_outside_support() -> None:
    cfg = SpectralCalibrationConfig(
        enabled=True, anchor="image", mapping="throughput_mix", init_scale=2.0, prior_sigma_log=0.15,
        prior_weight=1.0,
    )
    calibrator = SpectralNuisanceCalibrator(
        spectral_lam_by_exposure={9: np.array([5000.0, 8000.0], dtype=np.float64)}, bandpasses=_bandpasses(),
        cfg=cfg, device=torch.device("cpu"),
    )
    scales = calibrator.scale_for_full_exposure(9).detach().cpu().numpy()
    assert np.isclose(scales[0], 2.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(scales[1], 1.0, rtol=1e-6, atol=1e-6)


def test_prior_penalty_zero_at_unity() -> None:
    cfg = SpectralCalibrationConfig(
        enabled=True, anchor="image", mapping="throughput_mix", init_scale=1.0, prior_sigma_log=0.15,
        prior_weight=1.0,
    )
    calibrator = SpectralNuisanceCalibrator(
        spectral_lam_by_exposure={4: np.array([5000.0], dtype=np.float64)}, bandpasses=_bandpasses(), cfg=cfg,
        device=torch.device("cpu"),
    )
    penalty = float(calibrator.prior_penalty().item())
    assert np.isclose(penalty, 0.0, atol=1e-8)


def test_positive_parameterization_exp_eta() -> None:
    cfg = SpectralCalibrationConfig(
        enabled=True, anchor="image", mapping="throughput_mix", init_scale=0.8, prior_sigma_log=0.15,
        prior_weight=1.0,
    )
    calibrator = SpectralNuisanceCalibrator(
        spectral_lam_by_exposure={2: np.array([5000.0], dtype=np.float64)}, bandpasses=_bandpasses(), cfg=cfg,
        device=torch.device("cpu"),
    )
    scales = calibrator.scales_by_band().detach().cpu().numpy()
    assert np.all(scales > 0.0)
    assert np.isclose(scales[0], 0.8, rtol=1e-6, atol=1e-6)
