from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.axes import Axes

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.sdss_scene import SdssSpectrumData
from survey_fusion.evaluation.visualization import (
    composite_band_image, composite_band_image_on_model_wcs, configured_pixel_sed_points, plot_band_grid,
    plot_fiber_spectra_overlay, plot_history, plot_point_seds, plot_region_cutout_grid,
    plot_sed_sample_locations, project_sample_points_to_wcs, resolve_point_sed_region_masks,
    resolve_region_masks_by_exposure,
)
from survey_fusion.utils.wcs import TangentPlaneWCS


def _capture_semilogy_calls(monkeypatch) -> list[dict[str, np.ndarray | str | None]]:
    calls: list[dict[str, np.ndarray | str | None]] = []
    original = Axes.semilogy

    def _wrapped(self, *args, **kwargs):
        label = kwargs.get("label")
        if len(args) >= 2:
            x = np.asarray(args[0], dtype=np.float64)
            y = np.asarray(args[1], dtype=np.float64)
        elif len(args) == 1:
            y = np.asarray(args[0], dtype=np.float64)
            x = np.arange(y.size, dtype=np.float64)
        else:
            x = np.asarray([], dtype=np.float64)
            y = np.asarray([], dtype=np.float64)
        calls.append({"label": label, "x": x.copy(), "y": y.copy()})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", _wrapped)
    return calls


def _spectrum_call(calls: list[dict[str, np.ndarray | str | None]]) -> dict[str, np.ndarray | str | None]:
    spec_calls = [call for call in calls if call.get("label") == "Spectrum"]
    assert len(spec_calls) == 1
    return spec_calls[0]


def test_plot_history_uses_n_spec_mask_for_spectrum(monkeypatch, tmp_path: Path) -> None:
    calls = _capture_semilogy_calls(monkeypatch)
    history = {
        "loss": [1.0, 0.9, 0.8, 0.7], "image_loss": [0.8, 0.7, 0.6, 0.5],
        "spec_loss": [11.0, 12.0, 13.0, 14.0], "n_spec": [2, 0, 3, 0],
    }
    plot_history(history, tmp_path)
    spec = _spectrum_call(calls)
    np.testing.assert_array_equal(spec["x"], np.array([0.0, 2.0]))
    np.testing.assert_array_equal(spec["y"], np.array([11.0, 13.0]))
    assert (tmp_path / "training_history.png").exists()


def test_plot_history_falls_back_without_n_spec(monkeypatch, tmp_path: Path) -> None:
    calls = _capture_semilogy_calls(monkeypatch)
    history = {
        "loss": [1.0, 0.9, 0.8, 0.7], "image_loss": [0.8, 0.7, 0.6, 0.5],
        "spec_loss": [1.0, 0.0, float("nan"), 2.0],
    }
    plot_history(history, tmp_path)
    spec = _spectrum_call(calls)
    np.testing.assert_array_equal(spec["x"], np.array([0.0, 3.0]))
    np.testing.assert_array_equal(spec["y"], np.array([1.0, 2.0]))
    assert (tmp_path / "training_history.png").exists()


def _bandpasses() -> dict[str, Bandpass]:
    return {
        "g": Bandpass(
            band="g", lam=np.array([4000.0, 4700.0, 5400.0], dtype=np.float64),
            throughput=np.array([0.0, 1.0, 0.0], dtype=np.float64), canonical_id="test_g",
        ),
        "i": Bandpass(
            band="i", lam=np.array([6800.0, 7600.0, 8400.0], dtype=np.float64),
            throughput=np.array([0.0, 1.0, 0.0], dtype=np.float64), canonical_id="test_i",
        ),
    }


def test_plot_point_seds_writes_point_artifact_with_explicit_title(monkeypatch, tmp_path: Path) -> None:
    titles: list[str] = []

    def _capture_suptitle(title: str, *args, **kwargs):
        titles.append(str(title))
        return original_suptitle(title, *args, **kwargs)

    import matplotlib.pyplot as plt

    original_suptitle = plt.suptitle
    monkeypatch.setattr(plt, "suptitle", _capture_suptitle)

    lam = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0], dtype=np.float64)
    point_seds = [
        {
            "label": "core", "coords": (3, 4),
            "spectrum": np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64),
            "band_fluxes": {"g": 0.9, "i": 1.1},
        }
    ]

    plot_point_seds(lam, point_seds, _bandpasses(), tmp_path)

    assert titles == ["Continuous field point SED diagnostics"]
    assert (tmp_path / "point_seds.png").exists()


def test_plot_sed_sample_locations_writes_labeled_field_artifact(tmp_path: Path) -> None:
    image = np.arange(36, dtype=np.float64).reshape(6, 6)
    region_masks = {"G1": np.zeros((6, 6), dtype=bool)}
    region_masks["G1"][2:4, 2:4] = True
    region_overlays = {"G1": {"kind": "circle", "center_x": 2.5, "center_y": 2.5, "radius_pix": 1.2}}
    sample_points = [("G1", 2, 2), ("sky_1", 4, 4)]

    plot_sed_sample_locations(
        image, "i", tmp_path, sample_points=sample_points, region_masks=region_masks,
        region_overlays=region_overlays,
    )

    assert (tmp_path / "sed_sample_locations_i.png").exists()


def test_plot_sed_sample_locations_can_use_full_field_background(monkeypatch, tmp_path: Path) -> None:
    images: list[np.ndarray] = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, image, *args, **kwargs):
        images.append(np.asarray(image, dtype=np.float64).copy())
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    observation = np.full((20, 20), np.nan, dtype=np.float64)
    observation[8:11, 8:11] = 1.0
    learned_field = np.arange(400, dtype=np.float64).reshape(20, 20)

    plot_sed_sample_locations(
        observation, "i", tmp_path, sample_points=[("G1", 9, 9)], background_image=learned_field,
    )

    assert len(images) == 1
    np.testing.assert_array_equal(images[0], learned_field)
    assert (tmp_path / "sed_sample_locations_i.png").exists()


def test_plot_sed_sample_locations_downsamples_large_background_with_full_extent(
    monkeypatch, tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    original_imshow = Axes.imshow

    def _capture_imshow(self, image, *args, **kwargs):
        captured["shape"] = np.asarray(image).shape
        captured["extent"] = kwargs.get("extent")
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    observation = np.full((8, 8), np.nan, dtype=np.float64)
    observation[3:5, 3:5] = 1.0
    learned_field = np.arange(2050 * 8, dtype=np.float32).reshape(2050, 8)

    plot_sed_sample_locations(
        observation, "i", tmp_path, sample_points=[("G1", 1025, 4)], background_image=learned_field,
    )

    assert captured["shape"] == (1025, 4)
    assert captured["extent"] == (-0.5, 7.5, -0.5, 2049.5)
    assert (tmp_path / "sed_sample_locations_i.png").exists()


def test_composite_band_image_averages_finite_repeated_exposures() -> None:
    images = {
        1: {"i": np.array([[1.0, np.nan], [3.0, 4.0]])}, 2: {"i": np.array([[3.0, 5.0], [np.nan, 8.0]])},
        3: {"r": np.full((2, 2), 99.0)},
    }

    composite = composite_band_image(images, {1: "i", 2: "i", 3: "r"}, "i")

    assert composite is not None
    np.testing.assert_allclose(composite, np.array([[2.0, 5.0], [3.0, 6.0]], dtype=np.float32))


def test_composite_band_image_on_model_wcs_projects_trimmed_cutouts() -> None:
    model_wcs = TangentPlaneWCS(
        ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=1.0, width=8, height=8, ref_x=0.0, ref_y=0.0,
    )
    exposures = {1: SimpleNamespace(obs_wcs=model_wcs), 2: SimpleNamespace(obs_wcs=model_wcs)}
    images = {
        1: {"i": np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)},
        2: {"i": np.array([[5.0, 7.0], [np.nan, 9.0]], dtype=np.float32)},
    }

    composite = composite_band_image_on_model_wcs(images, {1: "i", 2: "i"}, exposures, "i", model_wcs)

    assert composite is not None
    assert composite.shape == (8, 8)
    np.testing.assert_allclose(composite[:2, :2], np.array([[3.0, 7.0], [3.0, 6.5]]))


def test_project_sample_points_to_wcs_maps_detector_to_model_pixels() -> None:
    source_wcs = TangentPlaneWCS(
        ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=1.0, width=4, height=4, ref_x=0.0, ref_y=0.0,
    )
    model_wcs = TangentPlaneWCS(
        ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=1.0, width=10, height=10, ref_x=5.0, ref_y=5.0,
    )

    points = project_sample_points_to_wcs([("G1", 2, 3)], from_wcs=source_wcs, to_wcs=model_wcs)

    assert points == [("G1", 7, 8)]


def test_plot_fiber_spectra_overlay_writes_artifact(tmp_path: Path) -> None:
    spectra = [
        {
            "label": "G1", "lam": np.array([5000.0, 5100.0, 5200.0]), "predicted": np.array([1.0, 1.1, 1.2]),
            "truth": np.array([0.9, 1.0, 1.1]),
        },
        {
            "label": "G2", "lam": np.array([5000.0, 5100.0, 5200.0]), "predicted": np.array([1.2, 1.25, 1.3]),
            "truth": np.array([1.1, 1.15, 1.2]),
        },
    ]
    plot_fiber_spectra_overlay(spectra, tmp_path)
    assert (tmp_path / "fiber_spectra_overlay.png").exists()


def test_plot_fiber_spectra_overlay_limits_number_of_spectra(monkeypatch, tmp_path: Path) -> None:
    plot_calls: list[str] = []
    original_plot = Axes.plot

    def _capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label is not None:
            plot_calls.append(str(label))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", _capture_plot)

    spectra = [
        {
            "label": f"S{idx}", "lam": np.array([5000.0, 5100.0, 5200.0]),
            "predicted": np.array([1.0, 1.1, 1.2]), "truth": np.array([0.9, 1.0, 1.1]),
        }
        for idx in range(6)
    ]

    plot_fiber_spectra_overlay(spectra, tmp_path, max_spectra=3)

    assert len(plot_calls) == 6
    assert all(label.startswith(("S0", "S1", "S2")) for label in plot_calls)
    assert (tmp_path / "fiber_spectra_overlay.png").exists()


def test_plot_band_grid_does_not_mark_bands_held_out_when_none(monkeypatch, tmp_path: Path) -> None:
    titles: list[str] = []
    original_set_title = Axes.set_title

    def _capture_title(self, label: str, *args, **kwargs):
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", _capture_title)

    image = np.ones((2, 2), dtype=np.float64)
    plot_band_grid({"g": image, "i": image}, {"g": image, "i": image}, tmp_path, held_out=None)

    assert all("HELD OUT" not in title for title in titles)
    assert (tmp_path / "band_grid.png").exists()


def test_plot_band_grid_uses_shared_robust_scales(monkeypatch, tmp_path: Path) -> None:
    imshow_calls: list[dict[str, object]] = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, image, *args, **kwargs):
        imshow_calls.append(
            {
                "shape": np.asarray(image).shape, "vmin": kwargs.get("vmin"), "vmax": kwargs.get("vmax"),
                "cmap": kwargs.get("cmap"),
            }
        )
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    observed = np.zeros((100, 100), dtype=np.float64)
    predicted = np.zeros((100, 100), dtype=np.float64)
    observed[40:45, 40:50] = 1.0
    predicted[40:45, 40:50] = 1.2
    observed[50, 50] = 100.0
    predicted[50, 50] = 80.0

    plot_band_grid({"i": observed}, {"i": predicted}, tmp_path, held_out=None)

    assert len(imshow_calls) == 3
    obs_call, pred_call, resid_call = imshow_calls
    assert obs_call["cmap"] == "viridis"
    assert pred_call["cmap"] == "viridis"
    assert resid_call["cmap"] == "RdBu_r"
    assert obs_call["vmin"] == pred_call["vmin"]
    assert obs_call["vmax"] == pred_call["vmax"]
    assert float(resid_call["vmin"]) == -float(resid_call["vmax"])
    assert float(resid_call["vmax"]) < float(np.abs(predicted - observed).max())
    assert (tmp_path / "band_grid.png").exists()


def test_plot_band_grid_masks_predictions_to_partial_frame_footprint(monkeypatch, tmp_path: Path) -> None:
    imshow_images: list[np.ndarray] = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, image, *args, **kwargs):
        imshow_images.append(np.asarray(image, dtype=np.float64).copy())
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    observed = np.full((10, 12), np.nan, dtype=np.float64)
    observed[2:5, 3:8] = 1.0
    predicted = np.full((10, 12), 99.0, dtype=np.float64)
    predicted[2:5, 3:8] = 1.5

    plot_band_grid(
        {"e3:g": observed},
        {"e3:g": predicted},
        tmp_path,
        held_out=None,
    )

    assert len(imshow_images) == 3
    obs_display, pred_display, resid_display = imshow_images
    assert obs_display.shape == (3, 5)
    assert pred_display.shape == (3, 5)
    assert resid_display.shape == (3, 5)
    assert np.nanmax(pred_display) == 1.5
    assert np.nanmax(resid_display) == 0.5
    assert (tmp_path / "band_grid.png").exists()


def test_plot_region_cutout_grid_handles_all_nan_partial_frame(tmp_path: Path) -> None:
    observed = np.full((12, 12), np.nan, dtype=np.float64)
    predicted = np.full((12, 12), np.nan, dtype=np.float64)
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    plot_region_cutout_grid(
        observed, predicted, {"fiber": mask}, "r", tmp_path, filename="region_grid_nan.png",
    )

    assert (tmp_path / "region_grid_nan.png").exists()


def test_plot_region_cutout_grid_ignores_predictions_without_observed_pixels(
    monkeypatch, tmp_path: Path,
) -> None:
    imshow_calls: list[np.ndarray] = []
    original_imshow = Axes.imshow

    def _capture_imshow(self, image, *args, **kwargs):
        imshow_calls.append(np.asarray(image, dtype=np.float64).copy())
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)

    observed = np.full((12, 12), np.nan, dtype=np.float64)
    predicted = np.ones((12, 12), dtype=np.float64)
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    plot_region_cutout_grid(
        observed, predicted, {"fiber": mask}, "r", tmp_path, filename="region_grid_pred_only.png",
    )

    assert imshow_calls == []
    assert (tmp_path / "region_grid_pred_only.png").exists()


def test_plot_region_cutout_grid_limits_to_regions_with_observed_pixels(monkeypatch, tmp_path: Path) -> None:
    titles: list[str] = []
    original_set_title = Axes.set_title

    def _capture_title(self, label: str, *args, **kwargs):
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", _capture_title)

    observed = np.full((20, 20), np.nan, dtype=np.float64)
    predicted = np.zeros((20, 20), dtype=np.float64)
    masks: dict[str, np.ndarray] = {}
    for idx in range(5):
        mask = np.zeros((20, 20), dtype=bool)
        y0 = 2 + idx * 3
        mask[y0:y0 + 2, 2:4] = True
        masks[f"R{idx}"] = mask
        if idx in {2, 4}:
            observed[y0:y0 + 2, 2:4] = 1.0

    plot_region_cutout_grid(
        observed, predicted, masks, "i", tmp_path, filename="region_grid_limited.png", max_regions=2,
    )

    joined = "\n".join(titles)
    assert "R2 observed" in joined
    assert "R4 observed" in joined
    assert "R0 observed" not in joined
    assert (tmp_path / "region_grid_limited.png").exists()


def test_resolve_region_masks_by_exposure_builds_fallback_masks_per_image_exposure() -> None:
    spectrum = SdssSpectrumData(
        label="G1", role="science", path="synthetic.fits", lam_spec=np.array([5000.0], dtype=np.float64),
        flux_spec_flambda=np.array([1.0], dtype=np.float32),
        flux_spec_flam_lsf=np.array([1.0], dtype=np.float32),
        spec_throughput=np.array([1.0], dtype=np.float32), fiber_mask=np.zeros((5, 5), dtype=np.float32),
        spec_ivar=np.array([1.0], dtype=np.float64), spec_wdisp=np.array([1.0], dtype=np.float64),
        spec_loglam=np.array([np.log10(5000.0)], dtype=np.float64),
        sigma_spec_flambda=np.array([0.1], dtype=np.float32), center_ra_dec=(180.0, 0.0),
        fiber_radius_arcsec=1.0, metadata={"fiber_diameter_arcsec": 2.0},
    )
    scene = SimpleNamespace(spectra=[spectrum])
    exposure_small = SimpleNamespace(
        obs_wcs=TangentPlaneWCS(
            ra0_rad=np.deg2rad(180.0), dec0_rad=0.0, pixel_scale_arcsec=1.0, width=5, height=5,
        ),
        image_height=5,
        image_width=5,
        fiber_mask=np.zeros((5, 5), dtype=np.float32),
    )
    exposure_large = SimpleNamespace(
        obs_wcs=TangentPlaneWCS(
            ra0_rad=np.deg2rad(180.0), dec0_rad=0.0, pixel_scale_arcsec=1.0, width=7, height=7,
        ),
        image_height=7,
        image_width=7,
        fiber_mask=np.zeros((7, 7), dtype=np.float32),
    )

    masks = resolve_region_masks_by_exposure(
        scene, configured_region_masks_by_exposure=None, exposures={1: exposure_small, 2: exposure_large},
        image_exposure_ids=[1, 2],
    )

    assert set(masks.keys()) == {1, 2}
    assert masks[1]["G1"].shape == (5, 5)
    assert masks[2]["G1"].shape == (7, 7)


def test_resolve_region_masks_by_exposure_skips_off_frame_fallback_regions() -> None:
    off_frame = SdssSpectrumData(
        label="off", role="science", path="synthetic.fits", lam_spec=np.array([5000.0], dtype=np.float64),
        flux_spec_flambda=np.array([1.0], dtype=np.float32),
        flux_spec_flam_lsf=np.array([1.0], dtype=np.float32),
        spec_throughput=np.array([1.0], dtype=np.float32), fiber_mask=np.zeros((5, 5), dtype=np.float32),
        spec_ivar=np.array([1.0], dtype=np.float64), spec_wdisp=np.array([1.0], dtype=np.float64),
        spec_loglam=np.array([np.log10(5000.0)], dtype=np.float64),
        sigma_spec_flambda=np.array([0.1], dtype=np.float32), center_ra_dec=(181.0, 0.0),
        fiber_radius_arcsec=1.0, metadata={"fiber_diameter_arcsec": 2.0},
    )
    on_frame = SdssSpectrumData(
        label="on", role="science", path="synthetic.fits", lam_spec=np.array([5000.0], dtype=np.float64),
        flux_spec_flambda=np.array([1.0], dtype=np.float32),
        flux_spec_flam_lsf=np.array([1.0], dtype=np.float32),
        spec_throughput=np.array([1.0], dtype=np.float32), fiber_mask=np.zeros((5, 5), dtype=np.float32),
        spec_ivar=np.array([1.0], dtype=np.float64), spec_wdisp=np.array([1.0], dtype=np.float64),
        spec_loglam=np.array([np.log10(5000.0)], dtype=np.float64),
        sigma_spec_flambda=np.array([0.1], dtype=np.float32), center_ra_dec=(180.0, 0.0),
        fiber_radius_arcsec=1.0, metadata={"fiber_diameter_arcsec": 2.0},
    )
    scene = SimpleNamespace(spectra=[off_frame, on_frame])
    exposure = SimpleNamespace(
        obs_wcs=TangentPlaneWCS(
            ra0_rad=np.deg2rad(180.0), dec0_rad=0.0, pixel_scale_arcsec=1.0, width=5, height=5,
        ),
        image_height=5,
        image_width=5,
        fiber_mask=np.zeros((5, 5), dtype=np.float32),
    )

    masks = resolve_region_masks_by_exposure(
        scene, configured_region_masks_by_exposure=None, exposures={1: exposure}, image_exposure_ids=[1],
    )

    assert set(masks[1].keys()) == {"on"}
    assert np.any(masks[1]["on"])


def test_resolve_region_masks_by_exposure_skips_fallback_when_disabled() -> None:
    spectrum = SdssSpectrumData(
        label="G1", role="science", path="synthetic.fits", lam_spec=np.array([5000.0], dtype=np.float64),
        flux_spec_flambda=np.array([1.0], dtype=np.float32),
        flux_spec_flam_lsf=np.array([1.0], dtype=np.float32),
        spec_throughput=np.array([1.0], dtype=np.float32), fiber_mask=np.zeros((5, 5), dtype=np.float32),
        spec_ivar=np.array([1.0], dtype=np.float64), spec_wdisp=np.array([1.0], dtype=np.float64),
        spec_loglam=np.array([np.log10(5000.0)], dtype=np.float64),
        sigma_spec_flambda=np.array([0.1], dtype=np.float32), center_ra_dec=(180.0, 0.0),
        fiber_radius_arcsec=1.0, metadata={"fiber_diameter_arcsec": 2.0},
    )
    scene = SimpleNamespace(spectra=[spectrum])
    exposure = SimpleNamespace(
        obs_wcs=TangentPlaneWCS(
            ra0_rad=np.deg2rad(180.0), dec0_rad=0.0, pixel_scale_arcsec=1.0, width=5, height=5,
        ),
        image_height=5,
        image_width=5,
        fiber_mask=np.ones((5, 5), dtype=np.float32),
    )

    masks = resolve_region_masks_by_exposure(
        scene, configured_region_masks_by_exposure=None, exposures={1: exposure}, image_exposure_ids=[1],
        allow_fallback=False,
    )

    assert masks == {}


def test_resolve_point_sed_region_masks_skips_fallback_when_disabled() -> None:
    spectrum = SdssSpectrumData(
        label="G1", role="science", path="synthetic.fits", lam_spec=np.array([5000.0], dtype=np.float64),
        flux_spec_flambda=np.array([1.0], dtype=np.float32),
        flux_spec_flam_lsf=np.array([1.0], dtype=np.float32),
        spec_throughput=np.array([1.0], dtype=np.float32), fiber_mask=np.zeros((5, 5), dtype=np.float32),
        spec_ivar=np.array([1.0], dtype=np.float64), spec_wdisp=np.array([1.0], dtype=np.float64),
        spec_loglam=np.array([np.log10(5000.0)], dtype=np.float64),
        sigma_spec_flambda=np.array([0.1], dtype=np.float32), center_ra_dec=(180.0, 0.0),
        fiber_radius_arcsec=1.0, metadata={"fiber_diameter_arcsec": 2.0},
    )
    scene = SimpleNamespace(spectra=[spectrum])

    masks = resolve_point_sed_region_masks(scene, None, allow_fallback=False)

    assert masks == {}


def test_configured_pixel_sed_points_uses_sky_points_without_regions() -> None:
    observation = np.zeros((30, 30), dtype=np.float64)
    points = configured_pixel_sed_points(observation, region_masks=None)

    assert len(points) == 4
    assert all(label.startswith("sky_") for label, _, _ in points)
    assert all(0 <= y < observation.shape[0] for _, y, _ in points)
    assert all(0 <= x < observation.shape[1] for _, _, x in points)


def test_configured_pixel_sed_points_uses_region_centroids_and_four_sky_points() -> None:
    observation = np.ones((30, 30), dtype=np.float64)
    observation[0:10, 0:10] = 0.0
    observation[0:10, 20:30] = 0.0
    observation[20:30, 0:10] = 0.0
    observation[20:30, 20:30] = 0.0
    region_masks = {"R2": np.zeros((30, 30), dtype=bool), "R1": np.zeros((30, 30), dtype=bool)}
    region_masks["R1"][10:13, 5:8] = True
    region_masks["R2"][18:21, 22:25] = True

    points = configured_pixel_sed_points(observation, region_masks=region_masks)

    labels = [label for label, _, _ in points]
    assert labels[:2] == ["R1", "R2"]
    assert len(points) == 6
    assert all(label.startswith("sky_") for label in labels[2:])
    assert ("R1", 11, 6) in points
    assert ("R2", 19, 23) in points
