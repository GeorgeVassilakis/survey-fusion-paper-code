from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from survey_fusion.data.adapters.sdss import _resolve_spectral_spatial_response, build_sdss_datums
from survey_fusion.data.prepared import prepared_digest
from test_adapter_datum_index_locality import _synthetic_sdss_scene, _bandpasses


def _response():
    return {
        "schema_version": 1,
        "model": "gaussian",
        "fwhm_arcsec": 2.0,
        "provenance": {
            "source": "matching spPlate",
            "field": "SEEING50",
            "method": "approximate scalar coadd response",
        },
        "flux_convention": "published_point_source_calibrated",
        "calibration": {
            "mode": "gaussian_point_source",
            "approximation": "centered_gaussian_standard_same_seeing",
            "provenance": {
                "source": "survey point-source convention",
                "field": "calibrated flux",
                "method": "conditional centered Gaussian approximation",
            },
        },
    }


def test_metadata_mode_requires_explicit_geometry():
    with pytest.raises(ValueError, match="requires explicit spatial_response"):
        _resolve_spectral_spatial_response(SimpleNamespace(metadata={}), mode="metadata", legacy_fwhm=1.4)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, "2.0"])
def test_invalid_seeing_does_not_fall_back(value):
    response = _response()
    response["fwhm_arcsec"] = value
    with pytest.raises(ValueError, match="positive and finite"):
        _resolve_spectral_spatial_response(
            SimpleNamespace(metadata={"spatial_response": response}), mode="metadata", legacy_fwhm=1.4
        )


@pytest.mark.parametrize("key", ["source", "field", "method"])
def test_metadata_requires_provenance(key):
    response = _response()
    del response["provenance"][key]
    with pytest.raises(ValueError, match="provenance"):
        _resolve_spectral_spatial_response(
            SimpleNamespace(metadata={"spatial_response": response}), mode="metadata", legacy_fwhm=1.4
        )


def test_adapter_uses_declared_seeing_without_rescaling_arrays():
    scene = _synthetic_sdss_scene()
    scene.spectra[0].metadata["spatial_response"] = _response()
    kwargs = dict(psf_fwhm_arcsec=1.3, device=torch.device("cpu"))
    legacy = build_sdss_datums(scene, _bandpasses(), spectral_response_mode="legacy_imaging_median", **kwargs)
    explicit = build_sdss_datums(scene, _bandpasses(), spectral_response_mode="metadata", **kwargs)
    spec_id = next(iter(explicit.spec_label_by_exposure))
    assert explicit.exposures[spec_id].psf_fwhm_arcsec == 2.0
    assert legacy.exposures[spec_id].psf_fwhm_arcsec == 1.3
    for name in ("y", "sigma", "additive"):
        np.testing.assert_array_equal(getattr(explicit.batch, name), getattr(legacy.batch, name))
    summary = explicit.holdout_region_summary["spectra"][0]["spatial_response"]
    response = _response()
    assert summary["resolved_source_scale"] > 1
    del summary["resolved_source_scale"]
    assert summary == response


def test_response_metadata_and_mode_change_cache_identity(tmp_path):
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"spectra": [{"spatial_response": _response()}]}))
    kwargs = dict(seed=42, bands=["r"], manifest_path=manifest)
    before = prepared_digest({"spectral_response_mode": "metadata"}, **kwargs)
    assert before != prepared_digest({"spectral_response_mode": "legacy_imaging_median"}, **kwargs)
    response = _response()
    response["provenance"]["source"] = "another reduction"
    manifest.write_text(json.dumps({"spectra": [{"spatial_response": response}]}))
    assert before != prepared_digest({"spectral_response_mode": "metadata"}, **kwargs)


def test_missing_response_exclusion_is_explicit_and_auditable():
    scene = _synthetic_sdss_scene()
    store = build_sdss_datums(
        scene,
        _bandpasses(),
        psf_fwhm_arcsec=1.3,
        device=torch.device("cpu"),
        spectral_response_missing_policy="exclude",
    )
    assert store.n_spec == 0
    summary = store.holdout_region_summary["spectra"][0]
    assert summary["excluded"]
    assert summary["excluded_reason"] == "missing_spectral_response"


def test_response_sidecar_content_changes_cache_identity(tmp_path):
    sidecar = tmp_path / "response.json"
    sidecar.write_text("[]")
    cfg = {"spectral_response_metadata_path": str(sidecar)}
    kwargs = dict(seed=0, bands=["r"], manifest_path=None)
    before = prepared_digest(cfg, **kwargs)
    sidecar.write_text('[{"label":"S1"}]')
    assert before != prepared_digest(cfg, **kwargs)


def test_loader_response_sidecar_preserves_geometry_and_raw_provenance(tmp_path):
    import json
    from test_sdss_scene_sky_background import _write_cutout, _write_frame, _write_spectrum, _cfg
    from survey_fusion.data.sdss_scene import load_sdss_scene

    cutout, frame, spectrum = (tmp_path / name for name in ("cutout.fits", "frame.fits", "spec.fits"))
    _write_cutout(cutout)
    _write_frame(frame)
    _write_spectrum(spectrum)
    cfg = _cfg(cutout_path=cutout, frame_path=frame, spectrum_path=spectrum)
    manifest = json.loads(Path(cfg["scene_manifest_path"]).read_text())
    label = manifest["spectra"][0]["label"]
    row = {"label": label, "spatial_response": _response(), "exposure_components": [{"expid": "native"}]}
    sidecar = tmp_path / "responses.json"
    sidecar.write_text(json.dumps([row]))
    cfg["spectral_response_metadata_path"] = str(sidecar)
    scene = load_sdss_scene(cfg, _bandpasses())
    assert scene.spectra[0].metadata["spatial_response"] == _response()
    assert scene.spectra[0].metadata["spectral_response_metadata"] == row
    assert scene.spectra[0].metadata["native_dlog10_lambda"] > 0
    sidecar.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_sdss_scene(cfg, _bandpasses())


@pytest.mark.parametrize("radius", [1.0, 1.5])
def test_gaussian_calibration_uses_actual_fiber_radius(radius):
    scene = _synthetic_sdss_scene()
    scene.spectra[0].metadata["spatial_response"] = _response()
    scene.spectra[0].fiber_radius_arcsec = radius
    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=torch.device("cpu"))
    exposure = store.exposures[next(iter(store.spec_label_by_exposure))]
    expected = 1.0 / -np.expm1(-0.5 * (radius / (2.0 / 2.355)) ** 2)
    assert exposure.spectral_source_scale == pytest.approx(expected)
    assert exposure.spectral_response_metadata["resolved_source_scale"] == pytest.approx(expected)


def test_published_flux_requires_explicit_calibration_approximation():
    response = _response()
    del response["calibration"]
    with pytest.raises(ValueError, match="explicit calibration"):
        _resolve_spectral_spatial_response(
            SimpleNamespace(metadata={"spatial_response": response}), mode="metadata", legacy_fwhm=1.4
        )
    response["calibration"] = {"mode": "finite_aperture"}
    with pytest.raises(ValueError, match="requires finite_aperture"):
        _resolve_spectral_spatial_response(
            SimpleNamespace(metadata={"spatial_response": response}), mode="metadata", legacy_fwhm=1.4
        )


def test_finite_aperture_metadata_retains_identity_source_scale():
    scene = _synthetic_sdss_scene()
    response = _response()
    response["flux_convention"] = "finite_aperture"
    response["calibration"] = {"mode": "finite_aperture"}
    scene.spectra[0].metadata["spatial_response"] = response
    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=torch.device("cpu"))
    exposure = store.exposures[next(iter(store.spec_label_by_exposure))]
    assert exposure.spectral_source_scale is None


@pytest.mark.parametrize("width", [0.0, -1.0, np.nan, np.inf])
def test_native_lsf_rejects_invalid_widths_but_legacy_preserves_clipping(width):
    from survey_fusion.data.adapters.sdss import _compute_lsf_fwhm_angstrom_by_bin

    loglam = np.array([3.7, 3.7001])
    lam = 10**loglam
    widths = np.array([width, 1.0])
    with pytest.raises(ValueError, match="Native spec_wdisp"):
        _compute_lsf_fwhm_angstrom_by_bin(loglam, widths, lam, dlog10_lambda=0.0001)
    legacy = _compute_lsf_fwhm_angstrom_by_bin(loglam, widths, lam)
    assert legacy[0] == 0.0
    assert legacy[1] > 0.0
