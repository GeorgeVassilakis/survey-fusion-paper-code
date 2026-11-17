from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.psf_mixture import psf_mixture_digest
from survey_fusion.data.sdss_scene import load_sdss_scene


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.array([5000.0, 5500.0, 6000.0], dtype=np.float64)
    throughput = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return {"r": Bandpass(band="r", lam=lam, throughput=throughput, canonical_id="test_r")}


def _simple_wcs_header(*, bunit: str | None = None) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    if bunit is not None:
        header["BUNIT"] = bunit
    return header


def _write_cutout(path: Path, *, bunit: str | None = None) -> None:
    cutout = np.full((6, 6), 1.0, dtype=np.float32)
    fits.PrimaryHDU(data=cutout, header=_simple_wcs_header(bunit=bunit)).writeto(path, overwrite=True)


def _write_frame(path: Path, *, include_sky_hdu: bool = True) -> None:
    frame_data = np.zeros((6, 6), dtype=np.float32)
    primary = fits.PrimaryHDU(data=frame_data, header=_simple_wcs_header(bunit="nanomaggy"))

    calib = np.full((6,), 2.0, dtype=np.float32)
    calib_hdu = fits.ImageHDU(data=calib)

    if include_sky_hdu:
        allsky = np.full((6, 6), 5.0, dtype=np.float32)
        xinterp = np.arange(6, dtype=np.float32)
        yinterp = np.arange(6, dtype=np.float32)
        sky_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="ALLSKY",
                    format=f"{allsky.size}E",
                    dim=f"({allsky.shape[1]},{allsky.shape[0]})",
                    array=[allsky],
                ),
                fits.Column(name="XINTERP", format=f"{xinterp.size}E", array=[xinterp]),
                fits.Column(name="YINTERP", format=f"{yinterp.size}E", array=[yinterp]),
            ]
        )
        fits.HDUList([primary, calib_hdu, sky_hdu]).writeto(path, overwrite=True)
        return

    fits.HDUList([primary, calib_hdu]).writeto(path, overwrite=True)


def _write_spectrum(
    path: Path, *, include_sky_column: bool = True, plug_ra_deg: float = 180.0, plug_dec_deg: float = 0.0,
    and_mask_value: int = 0, or_mask_value: int = 0,
) -> None:
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sky = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loglam = np.log10(np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)).astype(np.float32)
    ivar = np.ones_like(flux, dtype=np.float32)
    wdisp = np.ones_like(flux, dtype=np.float32)
    and_mask = np.full_like(flux, int(and_mask_value), dtype=np.int32)
    or_mask = np.full_like(flux, int(or_mask_value), dtype=np.int32)

    columns = [
        fits.Column(name="flux", format="E", array=flux),
        fits.Column(name="loglam", format="E", array=loglam),
        fits.Column(name="ivar", format="E", array=ivar),
        fits.Column(name="and_mask", format="J", array=and_mask),
        fits.Column(name="or_mask", format="J", array=or_mask),
        fits.Column(name="wdisp", format="E", array=wdisp),
    ]
    if include_sky_column:
        columns.append(fits.Column(name="sky", format="E", array=sky))
    coadd = fits.BinTableHDU.from_columns(columns, name="COADD")

    specobj = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PLUG_RA", format="D", array=np.array([plug_ra_deg], dtype=np.float64)),
            fits.Column(name="PLUG_DEC", format="D", array=np.array([plug_dec_deg], dtype=np.float64)),
        ],
        name="SPECOBJ",
    )
    primary = fits.PrimaryHDU(header=fits.Header({"BUNIT": "1E-17 erg/cm^2/s/Ang"}))
    fits.HDUList([primary, coadd, specobj]).writeto(path, overwrite=True)


def _manifest_image_exposure(
    *, cutout_path: Path | str, frame_path: Path | str, label: str="r-exp1", band: str = "r", gain: float=4.0,
    dark_variance: float = 1.0, psf_fwhm_arcsec: float = 1.2,
) -> dict:
    return {
        "label": label, "band": band, "cutout_path": str(cutout_path), "frame_path": str(frame_path),
        "gain": gain, "dark_variance": dark_variance, "psf_fwhm_arcsec": psf_fwhm_arcsec,
    }


def _cfg(*, cutout_path: Path, frame_path: Path, spectrum_path: Path) -> dict:
    manifest_path = cutout_path.with_name("manifest.json")
    manifest = {
        "image_exposures": [_manifest_image_exposure(cutout_path=cutout_path, frame_path=frame_path)],
        "spectra": [
            {"label": "S1", "role": "science", "path": str(spectrum_path), "ra_deg": 180.0, "dec_deg": 0.0}
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
    }


def _psf_digest_payload() -> dict:
    return {
        "dgpsf": {
            "operator_kind": "dgpsf", "qa_pass": True, "fallback": False,
            "pixel_response": "analytic_unfolded_native",
        },
        "kl": {
            "operator_kind": "kl", "qa_pass": True, "fallback": False,
            "pixel_response": "deconvolved_1_12_native",
        },
    }


def test_sdss_loader_rejects_mutated_psf_mixture_with_stale_digest(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path)
    _write_spectrum(spectrum_path)
    cfg = _cfg(
        cutout_path=cutout_path, frame_path=frame_path,
        spectrum_path=spectrum_path,
    )
    manifest_path = Path(cfg["scene_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixture = [
        [0.6, 1.0, 0.0, 1.0],
        [0.3, 4.0, 0.0, 4.0],
        [0.1, 9.0, 0.0, 9.0],
    ]
    exposure = manifest["image_exposures"][0]
    exposure["psf_mixture_dgpsf"] = mixture
    exposure["psf_mixture_kl"] = mixture
    exposure["psf_mixture_digest_payload"] = _psf_digest_payload()
    manifest["psf_mixture_digest_version"] = 2
    manifest["psf_mixture_digest"] = psf_mixture_digest(manifest["image_exposures"])
    exposure["psf_mixture_kl"][0][0] = 0.59
    exposure["psf_mixture_kl"][1][0] = 0.31
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the loaded mixture arrays"):
        load_sdss_scene(cfg, _bandpasses())


def test_sdss_loader_accepts_legacy_psf_digest_with_warning(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path)
    _write_spectrum(spectrum_path)
    cfg = _cfg(
        cutout_path=cutout_path, frame_path=frame_path,
        spectrum_path=spectrum_path,
    )
    manifest_path = Path(cfg["scene_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixture = [
        [0.6, 1.0, 0.0, 1.0],
        [0.3, 4.0, 0.0, 4.0],
        [0.1, 9.0, 0.0, 9.0],
    ]
    exposure = manifest["image_exposures"][0]
    exposure["psf_mixture_dgpsf"] = mixture
    exposure["psf_mixture_kl"] = mixture
    manifest["psf_mixture_digest"] = psf_mixture_digest(
        manifest["image_exposures"], version=1,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="predates QA.*convention binding"):
        scene = load_sdss_scene(cfg, _bandpasses())

    assert scene.metadata["psf_mixture_digest"] == manifest["psf_mixture_digest"]
    assert scene.metadata["psf_mixture_digest_version"] == 1


def test_sdss_loader_implicit_additive_background_keeps_source_additive_and_total(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spectrum_path, include_sky_column=True)

    cfg = _cfg(cutout_path=cutout_path, frame_path=frame_path, spectrum_path=spectrum_path)

    scene = load_sdss_scene(cfg, _bandpasses())
    spectrum = scene.spectra[0]

    np.testing.assert_allclose(
        scene.images_flambda["r"], scene.image_source_flambda["r"] + scene.image_additive_flambda["r"],
        rtol=1e-6, atol=1e-12,
    )
    np.testing.assert_allclose(
        scene.image_total_flambda["r"], scene.images_flambda["r"], rtol=1e-6, atol=1e-12,
    )
    assert float(np.max(scene.image_additive_flambda["r"])) > 0.0
    np.testing.assert_allclose(
        scene.flux_spec_flambda, scene.spec_source_flambda + scene.spec_additive_flambda, rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spectrum.flux_spec_flambda, spectrum.spec_source_flambda + spectrum.spec_additive_flambda, rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(scene.spec_total_flambda, scene.flux_spec_flambda, rtol=1e-6, atol=1e-12)
    assert float(np.max(scene.spec_additive_flambda)) > 0.0
    assert "background" not in scene.metadata


def test_sdss_spectrum_flux_and_uncertainty_remain_integrated_over_fiber(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spectrum_path, include_sky_column=True)

    scene = load_sdss_scene(
        _cfg(
            cutout_path=cutout_path,
            frame_path=frame_path,
            spectrum_path=spectrum_path,
        ),
        _bandpasses(),
    )

    np.testing.assert_allclose(
        scene.spec_source_flambda,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        scene.spec_additive_flambda,
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        scene.sigma_spec_flambda,
        np.ones(3, dtype=np.float32),
    )


def test_sdss_loader_missing_imaging_sky_inputs_fails(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path, include_sky_hdu=False)
    _write_spectrum(spectrum_path, include_sky_column=True)

    cfg = _cfg(cutout_path=cutout_path, frame_path=frame_path, spectrum_path=spectrum_path)
    with pytest.raises(ValueError, match="missing required HDUs"):
        load_sdss_scene(cfg, _bandpasses())


def test_sdss_loader_missing_coadd_sky_column_fails(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path)
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spectrum_path, include_sky_column=False)

    cfg = _cfg(cutout_path=cutout_path, frame_path=frame_path, spectrum_path=spectrum_path)
    with pytest.raises(ValueError, match="requires a COADD 'sky' column"):
        load_sdss_scene(cfg, _bandpasses())


def test_sdss_loader_centers_fiber_mask_on_plug_coordinates(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spectrum_path = tmp_path / "spec.fits"
    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)

    header = _simple_wcs_header(bunit="nanomaggy")
    wcs = WCS(header)
    target_x_pix = 4.0
    target_y_pix = 1.0
    plug_ra, plug_dec = wcs.all_pix2world(target_x_pix, target_y_pix, 0)
    _write_spectrum(
        spectrum_path, include_sky_column=True, plug_ra_deg=float(plug_ra), plug_dec_deg=float(plug_dec),
    )

    cfg = _cfg(cutout_path=cutout_path, frame_path=frame_path, spectrum_path=spectrum_path)
    scene = load_sdss_scene(cfg, _bandpasses())

    mask = scene.fiber_mask.astype(bool)
    yy, xx = np.nonzero(mask)
    assert yy.size > 0
    mask_cx = float(np.mean(xx))
    mask_cy = float(np.mean(yy))

    expected_x, expected_y = scene.wcs_per_band["r"].all_world2pix(float(plug_ra), float(plug_dec), 0)
    expected_x = float(expected_x)
    expected_y = float(expected_y)
    assert abs(mask_cx - expected_x) < 1.0
    assert abs(mask_cy - expected_y) < 1.0

    image_center_x = 0.5 * (scene.fiber_mask.shape[1] - 1)
    image_center_y = 0.5 * (scene.fiber_mask.shape[0] - 1)
    assert abs(mask_cx - image_center_x) > 1.0
    assert abs(mask_cy - image_center_y) > 1.0


def test_sdss_loader_reads_multispec_manifest(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spec1 = tmp_path / "spec-1.fits"
    spec2 = tmp_path / "spec-2.fits"
    spec3 = tmp_path / "spec-3.fits"
    manifest_path = tmp_path / "manifest.json"
    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spec1, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0)
    _write_spectrum(spec2, include_sky_column=True, plug_ra_deg=180.0002, plug_dec_deg=0.0001)
    _write_spectrum(spec3, include_sky_column=True, plug_ra_deg=180.0004, plug_dec_deg=0.0002)

    manifest = {
        "image_exposures": [_manifest_image_exposure(cutout_path=cutout_path, frame_path=frame_path)],
        "spectra": [
            {"label": "G1", "role": "science", "path": str(spec1), "ra_deg": 180.0, "dec_deg": 0.0},
            {"label": "G2", "role": "science", "path": str(spec2), "ra_deg": 180.0002, "dec_deg": 0.0001},
            {"label": "G3", "role": "science", "path": str(spec3), "ra_deg": 180.0004, "dec_deg": 0.0002},
            {"label": "SKY1", "role": "sky", "path": str(spec3), "ra_deg": 181.0, "dec_deg": 1.0},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
    }

    scene = load_sdss_scene(cfg, _bandpasses())

    assert len(scene.spectra) == 4
    assert [spec.label for spec in scene.spectra] == ["G1", "G2", "G3", "SKY1"]
    assert [spec.role for spec in scene.spectra] == ["science", "science", "science", "sky"]
    assert scene.metadata["spectra_count"] == 4
    assert scene.metadata["auxiliary_spectra"] == [{"label": "SKY1", "role": "sky", "path": str(spec3)}]
    assert scene.metadata["scene_manifest_path"] == str(manifest_path)


def test_sdss_loader_filters_spectra_by_role_and_class(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spec1 = tmp_path / "spec-galaxy.fits"
    spec2 = tmp_path / "spec-star.fits"
    spec3 = tmp_path / "spec-sky.fits"
    manifest_path = tmp_path / "manifest.json"
    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spec1, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0)
    _write_spectrum(spec2, include_sky_column=True, plug_ra_deg=180.0002, plug_dec_deg=0.0001)
    _write_spectrum(spec3, include_sky_column=True, plug_ra_deg=180.0004, plug_dec_deg=0.0002)

    manifest = {
        "image_exposures": [_manifest_image_exposure(cutout_path=cutout_path, frame_path=frame_path)],
        "spectra": [
            {
                "label": "G1", "role": "science", "class": "GALAXY", "path": str(spec1), "ra_deg": 180.0,
                "dec_deg": 0.0,
            },
            {
                "label": "S1", "role": "science", "class": "STAR", "path": str(spec2), "ra_deg": 180.0002,
                "dec_deg": 0.0001,
            },
            {
                "label": "SKY1", "role": "sky", "class": "SKY", "path": str(spec3), "ra_deg": 180.0004,
                "dec_deg": 0.0002,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
        "spectra": {"include_roles": ["science"], "include_classes": ["GALAXY"]},
    }

    scene = load_sdss_scene(cfg, _bandpasses())

    assert [spec.label for spec in scene.spectra] == ["G1"]
    assert scene.metadata["auxiliary_spectra"] == []




def test_sdss_loader_ignores_legacy_spectrum_selection_alias(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spec1 = tmp_path / "spec-galaxy.fits"
    spec2 = tmp_path / "spec-star.fits"
    spec3 = tmp_path / "spec-sky.fits"
    manifest_path = tmp_path / "manifest.json"
    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spec1, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0)
    _write_spectrum(spec2, include_sky_column=True, plug_ra_deg=180.0002, plug_dec_deg=0.0001)
    _write_spectrum(spec3, include_sky_column=True, plug_ra_deg=180.0004, plug_dec_deg=0.0002)

    manifest = {
        "image_exposures": [_manifest_image_exposure(cutout_path=cutout_path, frame_path=frame_path)],
        "spectra": [
            {
                "label": "G1", "role": "science", "class": "GALAXY", "path": str(spec1), "ra_deg": 180.0,
                "dec_deg": 0.0,
            },
            {
                "label": "S1", "role": "science", "class": "STAR", "path": str(spec2), "ra_deg": 180.0002,
                "dec_deg": 0.0001,
            },
            {
                "label": "SKY1", "role": "sky", "class": "SKY", "path": str(spec3), "ra_deg": 180.0004,
                "dec_deg": 0.0002,
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
        "spectrum_selection": {"include_roles": ["science"], "include_classes": ["GALAXY"]},
    }

    scene = load_sdss_scene(cfg, _bandpasses())

    assert [spec.label for spec in scene.spectra] == ["G1", "S1", "SKY1"]

def test_sdss_loader_and_adapter_support_repeated_image_exposures_with_global_model_wcs(
    tmp_path: Path,
) -> None:
    cutout1 = tmp_path / "cutout-r-1.fits"
    cutout2 = tmp_path / "cutout-r-2.fits"
    frame1 = tmp_path / "frame-r-000001-1-0001.fits"
    frame2 = tmp_path / "frame-r-000001-1-0002.fits"
    spec_path = tmp_path / "spec-1.fits"
    manifest_path = tmp_path / "manifest.json"
    _write_cutout(cutout1, bunit="nanomaggy")
    _write_cutout(cutout2, bunit="nanomaggy")
    _write_frame(frame1, include_sky_hdu=True)
    _write_frame(frame2, include_sky_hdu=True)
    _write_spectrum(spec_path, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0)

    manifest = {
        "center_ra_deg": 180.0,
        "center_dec_deg": 0.0,
        "half_size_x_arcsec": 3.0,
        "half_size_y_arcsec": 3.0,
        "model_pixel_scale_arcsec": 1.0,
        "image_exposures": [
            {
                "label": "r-exp1", "band": "r", "cutout_path": str(cutout1), "frame_path": str(frame1),
                "gain": 4.0, "dark_variance": 1.5, "psf_fwhm_arcsec": 1.2,
            },
            {
                "label": "r-exp2", "band": "r", "cutout_path": str(cutout2), "frame_path": str(frame2),
                "gain": 5.0, "dark_variance": 1.7, "psf_fwhm_arcsec": 1.4,
            },
        ],
        "spectra": [
            {"label": "G1", "role": "science", "path": str(spec_path), "ra_deg": 180.0, "dec_deg": 0.0}
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
    }

    scene = load_sdss_scene(cfg, _bandpasses())
    assert scene.model_wcs is not None
    assert scene.model_wcs.width == 7
    assert scene.model_wcs.height == 7
    assert scene.image_exposures is not None
    assert [exp.label for exp in scene.image_exposures] == ["r-exp1", "r-exp2"]
    assert scene.spectra[0].metadata["fiber_center_pixel_ref_band"] == "model_wcs"

    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=None, device=None, spectral_response_mode="legacy_imaging_median")

    assert store.n_image == 72
    assert store.n_spec == 3
    assert store.image_band_by_exposure == {0: "r", 1: "r"}
    assert store.exposures[0].wcs is scene.model_wcs
    assert store.exposures[1].wcs is scene.model_wcs
    assert store.exposures[0].obs_wcs is not None
    assert store.exposures[1].obs_wcs is not None
    assert store.exposures[0].psf_fwhm_arcsec == 1.2
    assert store.exposures[1].psf_fwhm_arcsec == 1.4
    assert set(store.spec_label_by_exposure) == {2}


def test_sdss_loader_skips_fully_masked_science_spectrum_when_clean_mask_enabled(tmp_path: Path) -> None:
    cutout_path = tmp_path / "cutout-r.fits"
    frame_path = tmp_path / "frame-r.fits"
    spec1 = tmp_path / "spec-1.fits"
    spec2 = tmp_path / "spec-2.fits"
    spec3 = tmp_path / "spec-3.fits"
    manifest_path = tmp_path / "manifest.json"
    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(
        spec1, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0, and_mask_value=4, or_mask_value=4
    )
    _write_spectrum(spec2, include_sky_column=True, plug_ra_deg=180.0002, plug_dec_deg=0.0001)
    _write_spectrum(spec3, include_sky_column=True, plug_ra_deg=180.0004, plug_dec_deg=0.0002)

    manifest = {
        "image_exposures": [_manifest_image_exposure(cutout_path=cutout_path, frame_path=frame_path)],
        "spectra": [
            {"label": "G1", "role": "science", "path": str(spec1), "ra_deg": 180.0, "dec_deg": 0.0},
            {"label": "G2", "role": "science", "path": str(spec2), "ra_deg": 180.0002, "dec_deg": 0.0001},
            {"label": "G3", "role": "science", "path": str(spec3), "ra_deg": 180.0004, "dec_deg": 0.0002},
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
        "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
        "spec_require_clean_mask": True,
    }

    scene = load_sdss_scene(cfg, _bandpasses())

    assert len(scene.spectra) == 2
    assert [spec.label for spec in scene.spectra] == ["G2", "G3"]
    assert scene.metadata["spectra_count"] == 2
    skipped = scene.metadata["skipped_spectra"]
    assert isinstance(skipped, list)
    assert len(skipped) == 1
    assert skipped[0]["label"] == "G1"


def test_sdss_loader_rebases_manifest_paths_to_observation_root(tmp_path: Path) -> None:
    observation_root = tmp_path / "data" / "multispec"
    observation_root.mkdir(parents=True)
    manifest_path = observation_root / "manifest.json"
    cutout_path = observation_root / "cutout-r-multispec.fits"
    frame_path = observation_root / "frame-r-000001-1-0001.fits"
    spec_path = observation_root / "spec-2612-54480-0035.fits"

    _write_cutout(cutout_path, bunit="nanomaggy")
    _write_frame(frame_path, include_sky_hdu=True)
    _write_spectrum(spec_path, include_sky_column=True, plug_ra_deg=180.0, plug_dec_deg=0.0)

    manifest = {
        "image_exposures": [
            _manifest_image_exposure(
                cutout_path="/nonexistent/export/machine/data/multispec/cutout-r-multispec.fits",
                frame_path="/nonexistent/export/machine/data/multispec/frame-r-000001-1-0001.fits",
            )
        ],
        "spectra": [
            {
                "label": "G1", "role": "science",
                "path": "/nonexistent/export/machine/data/multispec/spec-2612-54480-0035.fits",
                "ra_deg": 180.0, "dec_deg": 0.0,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {
        "observation_root": str(observation_root), "scene_manifest_path": str(manifest_path),
        "training_flux_scale": 1.0e17, "fiber_diameter_arcsec": 3.0, "reference_band": "r",
        "spec_min_ivar": 0.0, "spec_require_clean_mask": True,
    }

    scene = load_sdss_scene(cfg, _bandpasses())

    assert len(scene.spectra) == 1
    assert scene.spectra[0].path == str(spec_path)
    assert np.all(np.isfinite(scene.images_flambda["r"]))
