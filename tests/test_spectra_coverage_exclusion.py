from __future__ import annotations

import math

import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.sdss_scene import SdssImageExposureData, SdssSceneData, SdssSpectrumData
from survey_fusion.utils.wcs import TangentPlaneWCS


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.linspace(5400.0, 7200.0, 5)
    return {
        "r": Bandpass(
            band="r", lam=lam, throughput=np.ones_like(lam), canonical_id="sdss-r"
        )
    }


def _frame_wcs() -> AstropyWCS:
    wcs = AstropyWCS(naxis=2)
    wcs.wcs.crpix = [16.0, 16.0]
    wcs.wcs.crval = [180.0, 0.5]
    wcs.wcs.cdelt = [-1.1e-4, 1.1e-4]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _spectrum(label: str, ra: float, dec: float) -> SdssSpectrumData:
    lam_spec = np.array([5500.0, 5600.0, 5700.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    return SdssSpectrumData(
        label=label, role="science", path=f"{label}.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((32, 32), dtype=np.float32),
        spec_ivar=np.ones_like(lam_spec), spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=np.full(3, 0.05, dtype=np.float32),
        center_ra_dec=(ra, dec), fiber_radius_arcsec=1.5, metadata={},
    )


def test_spectra_without_image_coverage_are_excluded() -> None:
    frame = np.ones((32, 32), dtype=np.float32)
    exposure = SdssImageExposureData(
        label="r-frame", band="r", cutout_path="cut.fits", frame_path="frame.fits",
        source_flambda=frame.copy(), additive_flambda=np.zeros_like(frame),
        total_flambda=frame.copy(), target_flambda=frame.copy(),
        sigma_flambda=np.full_like(frame, 0.05), wcs=_frame_wcs(),
        pixel_scale_arcsec=0.4, pixel_area_arcsec2=0.16, psf_fwhm_arcsec=1.3,
        metadata={},
    )
    covered = _spectrum("covered", 180.0, 0.5)
    uncovered = _spectrum("uncovered", 181.0, 0.5)
    lam_spec = covered.lam_spec.copy()
    scene = SdssSceneData(
        images={"r": frame.copy()},
        images_flambda={"r": frame.copy()},
        spectra=[covered, uncovered],
        lam_spec=lam_spec,
        flux_spec_flambda=covered.flux_spec_flambda.copy(),
        flux_spec_flam_lsf=covered.flux_spec_flam_lsf.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((32, 32), dtype=np.float32),
        fiber_area_arcsec2=math.pi * 1.5**2,
        pixel_scale_arcsec=0.4,
        fiber_radius_arcsec=1.5,
        metadata={},
        wcs_per_band={"r": _frame_wcs()},
        pixel_scale_arcsec_per_band={"r": 0.4},
        pixel_area_arcsec2={"r": 0.16},
        spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=np.full(3, 0.05, dtype=np.float32),
        fiber_center_ra_dec=(180.0, 0.5),
        image_sigma_flambda={"r": np.full_like(frame, 0.05)},
        image_exposures=[exposure],
        model_wcs=TangentPlaneWCS(
            ra0_rad=math.pi, dec0_rad=math.radians(0.5),
            pixel_scale_arcsec=0.4, width=64, height=64,
        ),
    )
    store = build_sdss_datums(
        scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=torch.device("cpu"),
        spectral_response_mode="legacy_imaging_median",
    )

    labels = sorted(store.spec_label_by_exposure.values())
    assert labels == ["covered"]
    summaries = {
        item["label"]: item for item in store.holdout_region_summary.get("spectra", [])
    }
    assert summaries["covered"]["excluded"] is False
    assert summaries["uncovered"]["excluded"] is True
    assert summaries["uncovered"]["excluded_reason"] == "no_image_coverage"
