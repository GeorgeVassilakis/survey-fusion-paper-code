from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_sdss_scene_sky_background import _cfg, _write_cutout, _write_frame, _write_spectrum
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.flux_systems import (
    IDLSPEC2D_AB_CORRECTION_MAG, pipeline_band_ratios, ratio_knots, spectrum_to_image_ratio,
)
from survey_fusion.data.sdss_scene import load_sdss_scene

FILTERS = Path(__file__).resolve().parents[1] / "data" / "filters"


def _sdss_bandpasses() -> dict[str, Bandpass]:
    bands = {}
    for band in "ugriz":
        curve = pd.read_csv(FILTERS / f"SLOAN_SDSS_{band}.csv")
        bands[band] = Bandpass(band, curve["lambda_A"].to_numpy(float), curve["T"].to_numpy(float), band)
    return bands


def test_pipeline_ratios_invert_readplugmap_factors() -> None:
    ratios = pipeline_band_ratios()
    assert IDLSPEC2D_AB_CORRECTION_MAG == {"u": -0.042, "g": 0.036, "r": 0.015, "i": 0.013, "z": -0.002}
    assert ratios["g"] == pytest.approx(0.967386, abs=1e-6)
    assert ratios["u"] == pytest.approx(1.039441, abs=1e-6)
    assert ratios["z"] == pytest.approx(1.001844, abs=1e-6)


def test_ratio_curve_band_averages_equal_pipeline_ratios() -> None:
    bandpasses = _sdss_bandpasses()
    knots = ratio_knots(bandpasses)
    lam = np.linspace(2980.0, 11230.0, 80001)
    responses = {
        band: np.clip(np.interp(lam, bp.lam, bp.throughput, left=0, right=0), 0, None) * lam
        for band, bp in bandpasses.items()
    }
    covered = sum(responses.values()) > 0
    curve = np.ones_like(lam)
    curve[covered] = spectrum_to_image_ratio(lam[covered], bandpasses, knots)
    for band, expected in pipeline_band_ratios().items():
        response = responses[band]
        mean = np.trapezoid(response * curve, lam) / np.trapezoid(response, lam)
        assert mean == pytest.approx(expected, abs=1e-5), band
    assert 0.95 < curve[covered].min() and curve[covered].max() < 1.06


def test_ratio_curve_refuses_uncovered_wavelengths() -> None:
    bandpasses = _sdss_bandpasses()
    with pytest.raises(ValueError, match="outside the bandpass coverage"):
        spectrum_to_image_ratio(np.array([2000.0, 5000.0]), bandpasses, ratio_knots(bandpasses))


def test_ratio_knots_require_the_five_pipeline_bands() -> None:
    bandpasses = _sdss_bandpasses()
    del bandpasses["u"]
    with pytest.raises(ValueError, match="needs bands"):
        ratio_knots(bandpasses)


def _scene(tmp_path: Path, **extra):
    cutout_path, frame_path, spectrum_path = tmp_path / "c.fits", tmp_path / "f.fits", tmp_path / "s.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spectrum_path, include_sky_column=True)
    cfg = dict(_cfg(cutout_path=cutout_path, frame_path=frame_path, spectrum_path=spectrum_path), **extra)
    return load_sdss_scene(cfg, _sdss_bandpasses())


def test_loader_converts_every_spectral_array_to_the_imaging_system(tmp_path: Path) -> None:
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "imaging").mkdir()
    pipeline = _scene(tmp_path / "pipeline")
    imaging = _scene(tmp_path / "imaging", spectral_flux_system="sdss_imaging")
    bandpasses = _sdss_bandpasses()
    ratio = spectrum_to_image_ratio(pipeline.lam_spec, bandpasses, ratio_knots(bandpasses))
    for name in ("flux_spec_flambda", "spec_source_flambda", "spec_additive_flambda",
                 "spec_total_flambda", "sigma_spec_flambda"):
        np.testing.assert_allclose(getattr(imaging, name), getattr(pipeline, name) / ratio, rtol=1e-6)
    np.testing.assert_allclose(imaging.spec_ivar, pipeline.spec_ivar * ratio ** 2, rtol=1e-12)
    np.testing.assert_array_equal(imaging.lam_spec, pipeline.lam_spec)
    assert "spectral_flux_system" not in pipeline.metadata
    assert imaging.metadata["spectral_flux_system"]["system"] == "sdss_imaging"


def test_loader_rejects_unknown_flux_system(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="spectral_flux_system"):
        _scene(tmp_path, spectral_flux_system="ab")
