from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_sdss_scene_sky_background import _bandpasses, _cfg, _write_cutout, _write_frame, _write_spectrum
from survey_fusion.data.sdss_scene import load_sdss_scene


def _paths(tmp_path: Path):
    paths = tmp_path / "c.fits", tmp_path / "f.fits", tmp_path / "s.fits"
    _write_cutout(paths[0])
    _write_frame(paths[1], include_sky_hdu=True)
    _write_spectrum(paths[2], include_sky_column=True)
    return paths


def _manifest(tmp_path: Path, model: np.ndarray, *, sha: str | None = None) -> Path:
    np.save(tmp_path / "r-exp1.npy", model)
    digest = sha or hashlib.sha256((tmp_path / "r-exp1.npy").read_bytes()).hexdigest()
    manifest = {"units": "training_source_flambda",
                "exposures": {"r-exp1": {"path": "r-exp1.npy", "sha256": digest}}}
    path = tmp_path / "sigma_model.json"
    path.write_text(json.dumps(manifest))
    return path


def _scene(tmp_path: Path, **extra):
    cutout, frame, spectrum = _paths(tmp_path)
    return load_sdss_scene(dict(_cfg(cutout_path=cutout, frame_path=frame, spectrum_path=spectrum), **extra),
                           _bandpasses())


def test_model_variance_replaces_observed_counts(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    data = _scene(tmp_path / "data")
    exposure = data.image_exposures[0]
    # A model equal to the observed source reproduces the data-based sigma exactly.
    same = _scene(tmp_path / "model", image_sigma_model_outlier_pull=10.0, image_sigma_model_path=str(
        _manifest(tmp_path / "model", np.asarray(exposure.source_flambda, dtype=np.float32))))
    np.testing.assert_allclose(same.image_exposures[0].sigma_flambda, exposure.sigma_flambda, rtol=1e-6)
    assert "image_sigma_source" in same.metadata and "image_sigma_source" not in data.metadata
    np.testing.assert_array_equal(same.image_exposures[0].source_flambda, exposure.source_flambda)


def test_brighter_model_raises_the_variance(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    exposure = _scene(tmp_path / "data").image_exposures[0]
    brighter = np.asarray(exposure.source_flambda, dtype=np.float32) * 3.0
    model = _scene(tmp_path / "model", image_sigma_model_outlier_pull=10.0, image_sigma_model_path=str(_manifest(tmp_path / "model", brighter)))
    assert np.all(model.image_exposures[0].sigma_flambda > exposure.sigma_flambda)


def test_model_variance_refuses_a_changed_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        _scene(tmp_path, image_sigma_model_outlier_pull=10.0, image_sigma_model_path=str(_manifest(tmp_path, np.ones((6, 6), np.float32), sha="0" * 64)))


def test_model_variance_refuses_a_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        _scene(tmp_path, image_sigma_model_outlier_pull=10.0, image_sigma_model_path=str(_manifest(tmp_path, np.ones((5, 6), np.float32))))


def test_pixels_the_model_misses_keep_the_data_variance(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "model").mkdir()
    exposure = _scene(tmp_path / "data").image_exposures[0]
    source = np.asarray(exposure.source_flambda, dtype=np.float32)
    model = source.copy()
    model[2, 3] = source[2, 3] + 1000.0 * float(exposure.sigma_flambda[2, 3])
    model[4, 1] = source[4, 1] * 3.0
    scene = _scene(tmp_path / "model", image_sigma_model_outlier_pull=10.0, image_sigma_model_path=str(_manifest(tmp_path / "model", model)))
    sigma = scene.image_exposures[0].sigma_flambda
    assert sigma[2, 3] == pytest.approx(exposure.sigma_flambda[2, 3], rel=1e-6)
    assert sigma[4, 1] > exposure.sigma_flambda[4, 1]


def test_model_variance_requires_an_outlier_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outlier_pull"):
        _scene(tmp_path, image_sigma_model_path=str(_manifest(tmp_path, np.ones((6, 6), np.float32))))
