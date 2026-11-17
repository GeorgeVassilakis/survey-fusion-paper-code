"""Tests for Poisson + read-noise sigma computation and the 'poisson' adapter mode."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.sdss_scene import (
    SdssSceneData, SdssSpectrumData, _compute_poisson_sigma_nanomaggy, load_sdss_scene,
)


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


def _write_cutout(path: Path, value: float = 1.0) -> None:
    data = np.full((6, 6), value, dtype=np.float32)
    fits.PrimaryHDU(data=data, header=_simple_wcs_header(bunit="nanomaggy")).writeto(path, overwrite=True)


def _write_frame(path: Path, *, calib_value: float = 2.0, allsky_value: float = 5.0) -> None:
    frame_data = np.zeros((6, 6), dtype=np.float32)
    primary = fits.PrimaryHDU(data=frame_data, header=_simple_wcs_header(bunit="nanomaggy"))
    calib = np.full((6,), calib_value, dtype=np.float32)
    calib_hdu = fits.ImageHDU(data=calib)
    allsky = np.full((6, 6), allsky_value, dtype=np.float32)
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


def _write_spectrum(path: Path) -> None:
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sky = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loglam = np.log10(np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)).astype(np.float32)
    ivar = np.ones_like(flux, dtype=np.float32)
    wdisp = np.ones_like(flux, dtype=np.float32)
    and_mask = np.zeros_like(flux, dtype=np.int32)
    or_mask = np.zeros_like(flux, dtype=np.int32)
    coadd = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="flux", format="E", array=flux),
            fits.Column(name="loglam", format="E", array=loglam),
            fits.Column(name="ivar", format="E", array=ivar),
            fits.Column(name="and_mask", format="J", array=and_mask),
            fits.Column(name="or_mask", format="J", array=or_mask),
            fits.Column(name="wdisp", format="E", array=wdisp),
            fits.Column(name="sky", format="E", array=sky),
        ],
        name="COADD",
    )
    specobj = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PLUG_RA", format="D", array=np.array([180.0])),
            fits.Column(name="PLUG_DEC", format="D", array=np.array([0.0])),
        ],
        name="SPECOBJ",
    )
    primary = fits.PrimaryHDU(header=fits.Header({"BUNIT": "1E-17 erg/cm^2/s/Ang"}))
    fits.HDUList([primary, coadd, specobj]).writeto(path, overwrite=True)


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.array([5000.0, 5500.0, 6000.0], dtype=np.float64)
    throughput = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return {"r": Bandpass(band="r", lam=lam, throughput=throughput, canonical_id="test_r")}


def _manifest_image_exposure(
    *, cutout_path: Path, frame_path: Path, gain: float | None = 4.0, dark_variance: float | None = 1.0,
    psf_fwhm_arcsec: float | None = 1.2,
) -> dict:
    spec = {"label": "r-exp1", "band": "r", "cutout_path": str(cutout_path), "frame_path": str(frame_path)}
    if gain is not None:
        spec["gain"] = gain
    if dark_variance is not None:
        spec["dark_variance"] = dark_variance
    if psf_fwhm_arcsec is not None:
        spec["psf_fwhm_arcsec"] = psf_fwhm_arcsec
    return spec


class TestPoissonSigmaUnit:
    """Unit tests for _compute_poisson_sigma_nanomaggy."""

    def test_analytic_values(self, tmp_path: Path) -> None:
        """Sigma matches CCD noise model: var_DN = signal_DN/gain + dark_var."""
        frame_path = tmp_path / "frame.fits"
        _write_frame(frame_path, calib_value=2.0)

        total_nmgy = np.full((6, 6), 11.0, dtype=np.float64)
        gain = 4.0
        dark_variance = 1.0

        sigma = _compute_poisson_sigma_nanomaggy(
            total_nmgy, _simple_wcs_header(bunit="nanomaggy"), frame_path, gain, dark_variance,
        )

        # signal_DN = 11.0 / 2.0 = 5.5
        # var_DN = 5.5/4.0 + 1.0 = 2.375
        # sigma_nmgy = sqrt(2.375) * 2.0
        expected = np.sqrt(2.375) * 2.0
        np.testing.assert_allclose(sigma, expected, rtol=1e-5)

    def test_negative_signal_clamped(self, tmp_path: Path) -> None:
        """Negative signal DN is clamped to zero (only dark variance contributes)."""
        frame_path = tmp_path / "frame.fits"
        _write_frame(frame_path, calib_value=2.0)

        total_nmgy = np.full((6, 6), -1.0, dtype=np.float64)
        gain = 4.0
        dark_variance = 1.0

        sigma = _compute_poisson_sigma_nanomaggy(
            total_nmgy, _simple_wcs_header(bunit="nanomaggy"), frame_path, gain, dark_variance,
        )

        # signal_DN = -1.0/2.0 = -0.5, clamped to 0
        # var_DN = 0/4.0 + 1.0 = 1.0
        # sigma_nmgy = sqrt(1.0) * 2.0 = 2.0
        np.testing.assert_allclose(sigma, 2.0, rtol=1e-5)

    def test_bright_pixel_dominated_by_poisson(self, tmp_path: Path) -> None:
        """For bright pixels, Poisson term dominates over dark variance."""
        frame_path = tmp_path / "frame.fits"
        _write_frame(frame_path, calib_value=2.0)

        total_nmgy = np.full((6, 6), 1000.0, dtype=np.float64)
        gain = 4.0
        dark_variance = 1.0

        sigma = _compute_poisson_sigma_nanomaggy(
            total_nmgy, _simple_wcs_header(bunit="nanomaggy"), frame_path, gain, dark_variance,
        )

        signal_dn = 1000.0 / 2.0
        var_dn = signal_dn / gain + dark_variance
        poisson_fraction = (signal_dn / gain) / var_dn
        assert poisson_fraction > 0.99
        np.testing.assert_allclose(sigma, np.sqrt(var_dn) * 2.0, rtol=1e-5)

    def test_invalid_noise_parameters_raise_clear_error(self, tmp_path: Path) -> None:
        frame_path = tmp_path / "frame.fits"
        _write_frame(frame_path, calib_value=2.0)
        total_nmgy = np.full((6, 6), 11.0, dtype=np.float64)

        for gain in [0.0, -1.0, np.inf, np.nan]:
            with np.testing.assert_raises_regex(ValueError, "gain"):
                _compute_poisson_sigma_nanomaggy(
                    total_nmgy, _simple_wcs_header(bunit="nanomaggy"), frame_path, gain, 1.0,
                )

        for dark_variance in [-1.0, np.inf, np.nan]:
            with np.testing.assert_raises_regex(ValueError, "dark_variance"):
                _compute_poisson_sigma_nanomaggy(
                    total_nmgy, _simple_wcs_header(bunit="nanomaggy"), frame_path, 4.0, dark_variance,
                )


class TestPoissonSigmaIntegration:
    """Integration: manifest gain/dark_variance flows through to image_sigma_flambda."""

    def test_scene_with_noise_metadata_has_sigma_maps(self, tmp_path: Path) -> None:
        cutout_path = tmp_path / "cutout-r.fits"
        frame_path = tmp_path / "frame-r.fits"
        spectrum_path = tmp_path / "spec.fits"
        _write_cutout(cutout_path, value=1.0)
        _write_frame(frame_path, calib_value=2.0, allsky_value=5.0)
        _write_spectrum(spectrum_path)

        manifest = {
            "image_exposures": [
                _manifest_image_exposure(
                    cutout_path=cutout_path, frame_path=frame_path, gain=4.0, dark_variance=1.0,
                )
            ],
            "spectra": [
                {
                    "label": "S1", "role": "science", "ra_deg": 180.0, "dec_deg": 0.0,
                    "path": str(spectrum_path),
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        cfg = {
            "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
            "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
            "spec_require_clean_mask": True,
        }
        scene = load_sdss_scene(cfg, _bandpasses())

        assert scene.image_sigma_flambda is not None
        assert "r" in scene.image_sigma_flambda
        sigma_map = scene.image_sigma_flambda["r"]
        assert sigma_map.shape == (6, 6)
        assert np.all(np.isfinite(sigma_map))
        assert np.all(sigma_map > 0)

    def test_scene_implicit_sky_uses_total_count_poisson_sigma(self, tmp_path: Path) -> None:
        cutout_path = tmp_path / "cutout-r.fits"
        frame_path = tmp_path / "frame-r.fits"
        spectrum_path = tmp_path / "spec.fits"
        _write_cutout(cutout_path, value=1.0)
        _write_frame(frame_path, calib_value=2.0, allsky_value=5.0)
        _write_spectrum(spectrum_path)

        manifest = {
            "image_exposures": [
                _manifest_image_exposure(
                    cutout_path=cutout_path, frame_path=frame_path, gain=4.0, dark_variance=1.0,
                )
            ],
            "spectra": [
                {
                    "label": "S1", "role": "science", "ra_deg": 180.0, "dec_deg": 0.0,
                    "path": str(spectrum_path),
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        cfg = {
            "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
            "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
            "spec_require_clean_mask": True,
        }
        scene = load_sdss_scene(cfg, _bandpasses())

        image_with_sky = scene.images_flambda["r"]
        source = scene.image_source_flambda["r"]
        np.testing.assert_allclose(image_with_sky, source * 11.0, rtol=1e-6)

        sigma = scene.image_sigma_flambda["r"]
        nmgy_to_training = float(source[0, 0])
        expected_sigma_nmgy = np.sqrt(11.0 / 2.0 / 4.0 + 1.0) * 2.0
        np.testing.assert_allclose(sigma, expected_sigma_nmgy * nmgy_to_training, rtol=1e-6)
        np.testing.assert_allclose(scene.image_total_flambda["r"], image_with_sky, rtol=1e-6)

    def test_scene_image_exposure_requires_noise_metadata(self, tmp_path: Path) -> None:
        cutout_path = tmp_path / "cutout-r.fits"
        frame_path = tmp_path / "frame-r.fits"
        spectrum_path = tmp_path / "spec.fits"
        manifest_path = tmp_path / "manifest.json"
        _write_cutout(cutout_path)
        _write_frame(frame_path)
        _write_spectrum(spectrum_path)

        manifest = {
            "image_exposures": [
                _manifest_image_exposure(
                    cutout_path=cutout_path, frame_path=frame_path, gain=None, dark_variance=None,
                )
            ],
            "spectra": [
                {
                    "label": "S1", "role": "science", "ra_deg": 180.0, "dec_deg": 0.0,
                    "path": str(spectrum_path),
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cfg = {
            "scene_manifest_path": str(manifest_path), "training_flux_scale": 1.0e17,
            "fiber_diameter_arcsec": 3.0, "reference_band": "r", "spec_min_ivar": 0.0,
            "spec_require_clean_mask": True,
        }
        with np.testing.assert_raises_regex(ValueError, "gain and dark_variance"):
            load_sdss_scene(cfg, _bandpasses())


class _DummyWcs:
    """Minimal WCS stand-in for adapter tests (matches test_adapter_datum_index_locality)."""

    def all_pix2world(self, x, y, origin):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        return 180.0 + x_arr * 1e-4, 0.5 + y_arr * 1e-4


def _synthetic_scene_with_poisson_sigma() -> SdssSceneData:
    """Scene with known image_sigma_flambda for testing the adapter 'poisson' path."""
    shape = (4, 4)
    image = np.random.default_rng(0).uniform(0.5, 2.0, shape).astype(np.float32)
    sigma_map = np.random.default_rng(1).uniform(1e-4, 0.02, shape).astype(np.float32)
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    spec_sigma = np.array([1e-4, 0.02, 0.05], dtype=np.float32)
    wcs = _DummyWcs()
    spectrum = SdssSpectrumData(
        label="fiber", role="science", path="synthetic.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32), fiber_mask=np.ones(shape, dtype=np.float32),
        spec_ivar=np.ones_like(lam_spec), spec_wdisp=np.ones_like(lam_spec), spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=spec_sigma, center_ra_dec=(180.0, 0.5), fiber_radius_arcsec=1.5, metadata={},
    )
    return SdssSceneData(
        images={"r": image.copy()}, images_flambda={"r": image}, spectra=[spectrum], lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32), fiber_mask=np.ones(shape, dtype=np.float32),
        fiber_area_arcsec2=7.0, pixel_scale_arcsec=0.4, fiber_radius_arcsec=1.5, metadata={},
        wcs_per_band={"r": wcs}, pixel_scale_arcsec_per_band={"r": 0.4}, pixel_area_arcsec2={"r": 0.16},
        spec_ivar=np.ones_like(lam_spec), spec_wdisp=np.ones_like(lam_spec), spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=spec_sigma, fiber_center_ra_dec=(180.0, 0.5), image_sigma_flambda={"r": sigma_map},
    )


class TestAdapterPoissonMode:
    """build_sdss_datums preserves the data-provided uncertainties."""

    def test_uncertainties_pass_through_without_thresholding(self) -> None:
        scene = _synthetic_scene_with_poisson_sigma()
        store = build_sdss_datums(
            scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=torch.device("cpu"),
            spectral_response_mode="legacy_imaging_median",
        )
        n_pixels = scene.images_flambda["r"].size
        image_sigma = store.batch.sigma[:n_pixels].numpy()
        spectrum_sigma = store.batch.sigma[n_pixels:].numpy()
        np.testing.assert_array_equal(image_sigma, scene.image_sigma_flambda["r"].flatten())
        np.testing.assert_array_equal(spectrum_sigma, scene.sigma_spec_flambda)

    def test_poisson_mode_errors_without_sigma_maps(self) -> None:
        scene = _synthetic_scene_with_poisson_sigma()
        scene.image_sigma_flambda = None
        with np.testing.assert_raises(ValueError):
            build_sdss_datums(
                scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=torch.device("cpu"),
                spectral_response_mode="legacy_imaging_median",
            )
