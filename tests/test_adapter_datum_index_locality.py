from __future__ import annotations

import numpy as np
import pytest
import torch

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.regions import SkyCircleRegion, sky_circle_mask_from_omega_wcs
from survey_fusion.data.sdss_scene import SdssSceneData, SdssSpectrumData
from survey_fusion.pgm.contracts import DatumKind, SkyDiskApertureSpec
from survey_fusion.utils.wcs import AstropyWCSAdapter, radec_deg_to_omega_xyz


class DummySdssWcs:
    def all_pix2world(self, x, y, origin):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ra = 180.0 + x_arr * 1.0e-4
        dec = 0.5 + y_arr * 1.0e-4
        return ra, dec


class OffsetDummySdssWcs:
    def __init__(self, *, x_offset_pix: float = 0.0, y_offset_pix: float = 0.0) -> None:
        self.x_offset_pix = float(x_offset_pix)
        self.y_offset_pix = float(y_offset_pix)

    def all_pix2world(self, x, y, origin):
        del origin
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ra = 180.0 + (x_arr - self.x_offset_pix) * 1.0e-4
        dec = 0.5 + (y_arr - self.y_offset_pix) * 1.0e-4
        return ra, dec


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([0.2, 1.0, 0.2], dtype=np.float64)
    return {
        "g": Bandpass(band="g", lam=lam, throughput=throughput, canonical_id="test_g"),
        "r": Bandpass(band="r", lam=lam + 150.0, throughput=throughput, canonical_id="test_r"),
    }


def _bandpasses_with_i() -> dict[str, Bandpass]:
    base = _bandpasses()
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([0.2, 1.0, 0.2], dtype=np.float64)
    base["i"] = Bandpass(band="i", lam=lam + 300.0, throughput=throughput, canonical_id="test_i")
    return base


def _assert_locality(batch) -> None:
    exposure = batch.exposure_id.detach().cpu().numpy()
    kind = batch.datum_kind.detach().cpu().numpy()
    local = batch.datum_index.detach().cpu().numpy()
    for exp_id in np.unique(exposure):
        exp_mask = exposure == exp_id
        for kind_id in np.unique(kind[exp_mask]):
            group = np.sort(local[exp_mask & (kind == kind_id)])
            assert group.size > 0
            assert int(group[0]) == 0
            expected = np.arange(group.size, dtype=np.int64)
            assert np.array_equal(group.astype(np.int64), expected)


def _synthetic_sdss_scene() -> SdssSceneData:
    g = np.array([[0.8, 1.2, 1.4], [1.0, 1.3, 1.5]], dtype=np.float32)
    r = np.array([[1.0, 1.4, 1.6], [1.2, 1.5, 1.8]], dtype=np.float32)
    i = np.array([[1.1, 1.5, 1.7], [1.25, 1.6, 1.9]], dtype=np.float32)
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    sigma = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    wcs = DummySdssWcs()
    spectrum = SdssSpectrumData(
        label="fiber", role="science", path="synthetic.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((2, 3), dtype=np.float32), spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec), spec_loglam=np.log10(lam_spec), sigma_spec_flambda=sigma,
        center_ra_dec=(180.0, 0.5), fiber_radius_arcsec=1.5, metadata={},
    )
    return SdssSceneData(
        images={"g": g.copy(), "r": r.copy(), "i": i.copy()},
        images_flambda={"g": g, "r": r, "i": i},
        spectra=[spectrum],
        lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(),
        flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((2, 3), dtype=np.float32),
        fiber_area_arcsec2=7.0,
        pixel_scale_arcsec=0.4,
        fiber_radius_arcsec=1.5,
        metadata={},
        wcs_per_band={"g": wcs, "r": wcs, "i": wcs},
        pixel_scale_arcsec_per_band={"g": 0.4, "r": 0.4, "i": 0.4},
        pixel_area_arcsec2={"g": 0.16, "r": 0.16, "i": 0.16},
        spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=sigma,
        fiber_center_ra_dec=(180.0, 0.5),
        image_sigma_flambda={
            "g": np.full_like(g, 0.05, dtype=np.float32), "r": np.full_like(r, 0.05, dtype=np.float32),
            "i": np.full_like(i, 0.05, dtype=np.float32),
        },
    )


def _synthetic_multispec_scene() -> SdssSceneData:
    scene = _synthetic_sdss_scene()
    spectra = []
    for idx, (ra, dec, scale) in enumerate(
        [(180.0, 0.5, 1.0), (180.0002, 0.5001, 1.1), (180.0004, 0.5002, 1.2)], start=1,
    ):
        spectra.append(
            SdssSpectrumData(
                label=f"G{idx}", role="science", path=f"synthetic_{idx}.fits", lam_spec=scene.lam_spec.copy(),
                flux_spec_flambda=(scene.flux_spec_flambda * scale).astype(np.float32),
                flux_spec_flam_lsf=(scene.flux_spec_flam_lsf * scale).astype(np.float32),
                spec_throughput=np.ones_like(scene.lam_spec, dtype=np.float32),
                fiber_mask=np.ones((2, 3), dtype=np.float32), spec_ivar=np.ones_like(scene.lam_spec),
                spec_wdisp=np.ones_like(scene.lam_spec), spec_loglam=np.log10(scene.lam_spec),
                sigma_spec_flambda=scene.sigma_spec_flambda.copy(), center_ra_dec=(ra, dec),
                fiber_radius_arcsec=scene.fiber_radius_arcsec, metadata={},
            )
        )
    scene.spectra = spectra
    scene.lam_spec = spectra[0].lam_spec.copy()
    scene.flux_spec_flambda = spectra[0].flux_spec_flambda.copy()
    scene.flux_spec_flam_lsf = spectra[0].flux_spec_flam_lsf.copy()
    scene.spec_throughput = spectra[0].spec_throughput.copy()
    scene.spec_ivar = spectra[0].spec_ivar.copy()
    scene.spec_wdisp = spectra[0].spec_wdisp.copy()
    scene.spec_loglam = spectra[0].spec_loglam.copy()
    scene.sigma_spec_flambda = spectra[0].sigma_spec_flambda.copy()
    scene.fiber_center_ra_dec = spectra[0].center_ra_dec
    return scene


def test_sdss_adapter_emits_local_datum_indices() -> None:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _synthetic_sdss_scene(), _bandpasses(), psf_fwhm_arcsec=1.3, device=device,
        spectral_response_mode="legacy_imaging_median",
    )
    _assert_locality(store.batch)
    kind_values = set(torch.unique(store.batch.datum_kind).detach().cpu().tolist())
    assert kind_values == {int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)}
    image_ids = set(store.image_band_by_exposure.keys())
    for exp_id in sorted(image_ids):
        exposure = store.exposures[int(exp_id)]
        assert isinstance(exposure.obs_wcs, AstropyWCSAdapter)
        assert exposure.image_height is not None
        assert exposure.image_width is not None
    spec_ids = sorted(int(exp_id) for exp_id in store.exposures if int(exp_id) not in image_ids)
    assert spec_ids
    for exp_id in spec_ids:
        exposure = store.exposures[exp_id]
        assert exposure.lam_edges is not None
        assert exposure.lam_edges.shape == (int(exposure.lam_spec.size), 2)
        assert isinstance(exposure.aperture, SkyDiskApertureSpec)
    assert sorted(int(exp_id) for exp_id in store.spec_diagnostics_by_exposure.keys()) == spec_ids
    target = store.spec_diagnostics_by_exposure[spec_ids[0]]
    assert target.label == "fiber"
    assert target.role == "science"
    assert target.truth_counts is None


def test_sdss_adapter_gates_sky_fiber_source_supervision() -> None:
    device = torch.device("cpu")
    scene = _synthetic_multispec_scene()
    scene.spectra[1].role = "sky"

    store = build_sdss_datums(
        scene, _bandpasses_with_i(), psf_fwhm_arcsec=1.3, device=device,
        spectral_response_mode="legacy_imaging_median",
    )

    batch = store.batch
    kind = batch.datum_kind.detach().cpu()
    exposure = batch.exposure_id.detach().cpu()
    gate = batch.source_gate.detach().cpu()
    spectral = kind == int(DatumKind.SPECTRAL_BIN)
    for exp_id, role in store.spec_role_by_exposure.items():
        exp_mask = spectral & (exposure == int(exp_id))
        assert bool(exp_mask.any())
        expected_gate = 0.0 if role == "sky" else 1.0
        assert torch.all(gate[exp_mask] == expected_gate)


def test_sdss_adapter_partial_holdout_marks_excluded_pixels() -> None:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _synthetic_sdss_scene(),
        _bandpasses_with_i(),
        psf_fwhm_arcsec=1.3,
        device=device,
        holdout_band="i",
        holdout_region_cfg={
            "enabled": True, "center_radec_deg": [180.0002, 0.5000], "radius_arcsec": 0.5,
            "exclude_spectra_in_region": False,
        },
        region_metrics_cfg={
            "enabled": True,
            "primary_region": "R2",
            "regions": [{"name": "R2", "center_radec_deg": [180.0002, 0.5000], "radius_arcsec": 0.5}],
        },
        spectral_response_mode="legacy_imaging_median",
    )
    assert store.partial_holdout_enabled
    i_exp = max(store.image_band_by_exposure.keys())
    excluded = store.image_excluded_local_indices_by_exposure.get(i_exp)
    assert excluded is not None
    assert int(excluded.size) > 0
    assert int(excluded.size) < 6
    assert "R2" in store.region_masks_by_exposure[i_exp]
    assert store.n_spec > 0


def test_sdss_adapter_partial_holdout_can_exclude_spectrum() -> None:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _synthetic_sdss_scene(),
        _bandpasses_with_i(),
        psf_fwhm_arcsec=1.3,
        device=device,
        holdout_band="i",
        holdout_region_cfg={
            "enabled": True, "center_radec_deg": [180.0, 0.5], "radius_arcsec": 2.0,
            "exclude_spectra_in_region": True,
        },
        spectral_response_mode="legacy_imaging_median",
    )
    assert store.partial_holdout_enabled
    assert store.n_spec == 0
    assert bool(store.holdout_region_summary.get("spectrum_excluded"))
    assert int(torch.sum(store.batch.datum_kind == int(DatumKind.SPECTRAL_BIN)).item()) == 0


def test_sdss_adapter_partial_holdout_excludes_overlapping_spectrum() -> None:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _synthetic_sdss_scene(),
        _bandpasses_with_i(),
        psf_fwhm_arcsec=1.3,
        device=device,
        holdout_band="i",
        holdout_region_cfg={
            "enabled": True, "center_radec_deg": [180.0, 0.5 + (1.6 / 3600.0)], "radius_arcsec": 0.4,
            "exclude_spectra_in_region": True,
        },
        spectral_response_mode="legacy_imaging_median",
    )
    assert store.partial_holdout_enabled
    assert store.n_spec == 0
    assert bool(store.holdout_region_summary.get("spectrum_excluded"))
    assert int(torch.sum(store.batch.datum_kind == int(DatumKind.SPECTRAL_BIN)).item()) == 0


def test_sdss_multispec_adapter_builds_multiple_spectral_exposures() -> None:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _synthetic_multispec_scene(), _bandpasses(), psf_fwhm_arcsec=1.3, device=device,
        spectral_response_mode="legacy_imaging_median",
    )
    spec_ids = sorted(int(exp_id) for exp_id in store.spec_label_by_exposure.keys())
    assert len(spec_ids) == 3
    assert [store.spec_label_by_exposure[exp_id] for exp_id in spec_ids] == ["G1", "G2", "G3"]
    assert all(store.spec_role_by_exposure[exp_id] == "science" for exp_id in spec_ids)
    assert [store.spec_diagnostics_by_exposure[exp_id].label for exp_id in spec_ids] == ["G1", "G2", "G3"]
    assert all(store.spec_diagnostics_by_exposure[exp_id].truth_counts is None for exp_id in spec_ids)
    summary = store.holdout_region_summary.get("spectra", [])
    assert [item["label"] for item in summary] == ["G1", "G2", "G3"]
    assert all(item["exposure_id"] is not None for item in summary)


def test_sdss_multispec_image_exposures_use_union_science_fiber_mask() -> None:
    device = torch.device("cpu")
    scene = _synthetic_multispec_scene()
    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")

    for exp_id in sorted(store.image_band_by_exposure):
        exposure = store.exposures[int(exp_id)]
        expected = np.zeros_like(exposure.fiber_mask, dtype=bool)
        for spectrum in scene.spectra:
            expected |= sky_circle_mask_from_omega_wcs(
                exposure.obs_wcs,
                exposure.fiber_mask.shape,
                SkyCircleRegion(
                    name=spectrum.label, center_ra_deg=float(spectrum.center_ra_dec[0]),
                    center_dec_deg=float(spectrum.center_ra_dec[1]),
                    radius_arcsec=float(spectrum.fiber_radius_arcsec),
                ),
            )
        np.testing.assert_array_equal(exposure.fiber_mask, expected.astype(np.float32))

    spec_ids = sorted(int(exp_id) for exp_id in store.spec_label_by_exposure)
    for exp_id, spectrum in zip(spec_ids, scene.spectra, strict=True):
        np.testing.assert_array_equal(store.exposures[exp_id].fiber_mask, spectrum.fiber_mask)


def test_sdss_multispec_image_fiber_masks_respect_per_band_wcs() -> None:
    device = torch.device("cpu")
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    sigma = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    spectrum = SdssSpectrumData(
        label="G1", role="science", path="synthetic.fits", lam_spec=lam_spec, flux_spec_flambda=flux.copy(),
        flux_spec_flam_lsf=flux.copy(), spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.zeros((7, 7), dtype=np.float32), spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec), spec_loglam=np.log10(lam_spec), sigma_spec_flambda=sigma,
        center_ra_dec=(180.0, 0.5), fiber_radius_arcsec=0.45, metadata={},
    )
    scene = SdssSceneData(
        images={"g": np.zeros((7, 7), dtype=np.float32), "r": np.zeros((7, 7), dtype=np.float32)},
        images_flambda={"g": np.zeros((7, 7), dtype=np.float32), "r": np.zeros((7, 7), dtype=np.float32)},
        spectra=[spectrum],
        lam_spec=lam_spec.copy(),
        flux_spec_flambda=flux.copy(),
        flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.zeros((7, 7), dtype=np.float32),
        fiber_area_arcsec2=1.0,
        pixel_scale_arcsec=0.4,
        fiber_radius_arcsec=0.45,
        metadata={},
        wcs_per_band={
            "g": OffsetDummySdssWcs(x_offset_pix=0.0, y_offset_pix=0.0),
            "r": OffsetDummySdssWcs(x_offset_pix=1.5, y_offset_pix=0.5),
        },
        pixel_scale_arcsec_per_band={"g": 0.4, "r": 0.4},
        pixel_area_arcsec2={"g": 0.16, "r": 0.16},
        spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=sigma,
        fiber_center_ra_dec=(180.0, 0.5),
        image_sigma_flambda={
            "g": np.full((7, 7), 0.05, dtype=np.float32), "r": np.full((7, 7), 0.05, dtype=np.float32),
        },
    )

    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")

    g_exp = next(exp_id for exp_id, band in store.image_band_by_exposure.items() if band == "g")
    r_exp = next(exp_id for exp_id, band in store.image_band_by_exposure.items() if band == "r")
    assert not np.array_equal(store.exposures[int(g_exp)].fiber_mask, store.exposures[int(r_exp)].fiber_mask)


def test_sdss_multispec_spectral_exposures_share_image_coordinate_frame() -> None:
    device = torch.device("cpu")
    scene = _synthetic_multispec_scene()
    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")

    reference_image_exp_id = min(int(exp_id) for exp_id in store.image_band_by_exposure)
    reference_wcs = store.exposures[reference_image_exp_id].wcs

    for exp_id, label in sorted(store.spec_label_by_exposure.items()):
        spectrum = next(spec for spec in scene.spectra if spec.label == label)
        omega = radec_deg_to_omega_xyz(
            np.array([spectrum.center_ra_dec[0]], dtype=np.float64),
            np.array([spectrum.center_ra_dec[1]], dtype=np.float64),
        )
        ref_x, ref_y = reference_wcs.omega_xyz_to_norm(omega)
        spec_x, spec_y = store.exposures[int(exp_id)].wcs.omega_xyz_to_norm(omega)
        np.testing.assert_allclose(spec_x, ref_x, atol=1.0e-9, rtol=0.0)
        np.testing.assert_allclose(spec_y, ref_y, atol=1.0e-9, rtol=0.0)

    primary_spec_exp = min(int(exp_id) for exp_id in store.spec_label_by_exposure)
    secondary_spec_exp = sorted(int(exp_id) for exp_id in store.spec_label_by_exposure)[1]
    secondary_label = store.spec_label_by_exposure[secondary_spec_exp]
    secondary_spectrum = next(spec for spec in scene.spectra if spec.label == secondary_label)
    secondary_omega = radec_deg_to_omega_xyz(
        np.array([secondary_spectrum.center_ra_dec[0]], dtype=np.float64),
        np.array([secondary_spectrum.center_ra_dec[1]], dtype=np.float64),
    )
    primary_x, primary_y = store.exposures[primary_spec_exp].wcs.omega_xyz_to_norm(secondary_omega)
    secondary_x, secondary_y = store.exposures[secondary_spec_exp].wcs.omega_xyz_to_norm(secondary_omega)
    np.testing.assert_allclose(primary_x, secondary_x, atol=1.0e-9, rtol=0.0)
    np.testing.assert_allclose(primary_y, secondary_y, atol=1.0e-9, rtol=0.0)


def test_sdss_adapter_rejects_mismatched_image_shapes_for_scene_geometry() -> None:
    device = torch.device("cpu")
    scene = _synthetic_sdss_scene()
    scene.images_flambda["r"] = np.ones((3, 3), dtype=np.float32)
    scene.images["r"] = scene.images_flambda["r"].copy()

    with pytest.raises(ValueError, match="share one scene geometry"):
        build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")


def test_sdss_adapter_rejects_mismatched_pixel_scales_for_scene_geometry() -> None:
    device = torch.device("cpu")
    scene = _synthetic_sdss_scene()
    scene.pixel_scale_arcsec_per_band["r"] = 0.5

    with pytest.raises(ValueError, match="share one scene pixel scale"):
        build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")


def test_sdss_adapter_includes_sky_spectra_as_source_gated_datums() -> None:
    device = torch.device("cpu")
    scene = _synthetic_multispec_scene()
    scene.spectra.append(
        SdssSpectrumData(
            label="SKY1", role="sky", path="synthetic_sky.fits", lam_spec=scene.lam_spec.copy(),
            flux_spec_flambda=scene.flux_spec_flambda.copy(),
            flux_spec_flam_lsf=scene.flux_spec_flam_lsf.copy(),
            spec_throughput=np.ones_like(scene.lam_spec, dtype=np.float32),
            fiber_mask=np.ones((2, 3), dtype=np.float32), spec_ivar=np.ones_like(scene.lam_spec),
            spec_wdisp=np.ones_like(scene.lam_spec), spec_loglam=np.log10(scene.lam_spec),
            sigma_spec_flambda=scene.sigma_spec_flambda.copy(), center_ra_dec=(181.0, 1.0),
            fiber_radius_arcsec=scene.fiber_radius_arcsec, metadata={},
        )
    )

    store = build_sdss_datums(scene, _bandpasses(), psf_fwhm_arcsec=1.3, device=device, spectral_response_mode="legacy_imaging_median")

    spec_ids = sorted(int(exp_id) for exp_id in store.spec_label_by_exposure.keys())
    assert len(spec_ids) == 4
    assert [store.spec_label_by_exposure[exp_id] for exp_id in spec_ids] == ["G1", "G2", "G3", "SKY1"]
    assert [store.spec_role_by_exposure[exp_id] for exp_id in spec_ids] == [
        "science", "science", "science", "sky",
    ]

    batch = store.batch
    spectral = batch.datum_kind.detach().cpu() == int(DatumKind.SPECTRAL_BIN)
    exposure = batch.exposure_id.detach().cpu()
    gate = batch.source_gate.detach().cpu()
    sky_exp_id = spec_ids[-1]
    sky_mask = spectral & (exposure == sky_exp_id)
    assert bool(sky_mask.any())
    assert torch.all(gate[sky_mask] == 0.0)
