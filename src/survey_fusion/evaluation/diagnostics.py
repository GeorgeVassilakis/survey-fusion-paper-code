from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class SpectrumDiagnosticTarget:
    label: str
    role: str
    lam: np.ndarray
    truth_flux_density: np.ndarray
    truth_counts: np.ndarray | None
    throughput: np.ndarray | None


def _sanitize_plot_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
    return cleaned or "fiber"


def _spectrum_error_summary(
    predicted: np.ndarray, truth: np.ndarray, lam: np.ndarray,
) -> dict[str, float]:
    """Summarize a predicted spectrum against truth.

    The existing ``flux_error_pct``, ``predicted_flux_sum``, and
    ``truth_flux_sum`` fields are unweighted sample sums.  They are retained
    for continuity, but are shape-dependent and are not integrated flux on a
    nonuniform wavelength grid.  The ``*_integral`` fields use trapezoidal
    wavelength weights over ``lam`` instead.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(truth, dtype=np.float64)
    wavelength = np.asarray(lam, dtype=np.float64)
    if wavelength.shape != pred.shape:
        raise ValueError(
            f"Wavelength grid shape {wavelength.shape} does not match spectrum shape {pred.shape}."
        )
    diff = pred - obs
    predicted_flux_integral = float(np.trapezoid(pred, x=wavelength))
    truth_flux_integral = float(np.trapezoid(obs, x=wavelength))
    return {
        "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff**2))),
        "flux_error_pct": float(100.0 * (pred.sum() - obs.sum()) / (obs.sum() + 1e-12)),
        "max_abs_residual": float(np.max(np.abs(diff))),
        # Raw sums let consumers rebuild ratios and reject near-zero or signed
        # denominators, where flux_error_pct is numerically meaningless.
        "predicted_flux_sum": float(pred.sum()),
        "truth_flux_sum": float(obs.sum()),
        "predicted_flux_integral": predicted_flux_integral,
        "truth_flux_integral": truth_flux_integral,
        "flux_error_integrated_pct": float(
            100.0 * (predicted_flux_integral - truth_flux_integral)
            / (truth_flux_integral + 1e-12)
        ),
    }


def _truth_for_target(
    target: SpectrumDiagnosticTarget, *, use_flux_density: bool, exposure_id: int,
) -> np.ndarray:
    if use_flux_density:
        return np.asarray(target.truth_flux_density, dtype=np.float64)
    if target.truth_counts is None:
        raise ValueError(
            "Spectrum diagnostics require counts truth when use_flux_density is false "
            f"(missing for exposure {int(exposure_id)} / label {target.label!r})."
        )
    return np.asarray(target.truth_counts, dtype=np.float64)


def _validated_lam_for_target(
    predicted_lam: np.ndarray, target: SpectrumDiagnosticTarget, *, exposure_id: int,
) -> np.ndarray:
    lam_pred = np.asarray(predicted_lam, dtype=np.float64)
    lam_target = np.asarray(target.lam, dtype=np.float64)
    if lam_pred.shape != lam_target.shape or not np.allclose(
        lam_pred, lam_target, rtol=1.0e-6, atol=1.0e-6, equal_nan=False,
    ):
        raise ValueError(
            "Predicted wavelength grid does not match diagnostics target "
            f"for exposure {int(exposure_id)} / label {target.label!r}."
        )
    return lam_target


def _record_diagnostic_timing(
    record_timing: Callable[[str, float], None] | None, key: str, started_at: float,
) -> None:
    if record_timing is not None:
        record_timing(key, started_at)


def write_point_sed_diagnostics(
    *, trainer: Any, scene: Any, observation: np.ndarray,
    background_observations: Mapping[int, Mapping[str, np.ndarray]],
    image_band_by_exposure: Mapping[int, str], exposures: Mapping[int, Any], diagnostic_band: str,
    primary_exp_id: int, configured_region_masks: Mapping[str, np.ndarray] | None,
    region_overlays: Mapping[str, Mapping[str, float | str]] | None, bandpasses: Mapping[str, Any],
    run_dir: Path, location_background_fallback: np.ndarray | None = None, plot_sample_locations: bool = True,
    allow_region_fallback: bool = True, record_timing: Callable[[str, float], None] | None = None,
) -> None:
    from . import visualization

    exposure = exposures[int(primary_exp_id)]
    started = time.perf_counter()
    point_region_masks = visualization.resolve_point_sed_region_masks(
        scene, dict(configured_region_masks or {}), exposure=exposure,
        allow_fallback=bool(allow_region_fallback),
    )
    sed_points = visualization.configured_pixel_sed_points(observation, region_masks=point_region_masks)
    _record_diagnostic_timing(record_timing, "configured_pixel_sed_points", started)

    started = time.perf_counter()
    point_seds = trainer.sample_point_spectra(
        [(label, y, x) for label, y, x in sed_points], exposure_id=int(primary_exp_id),
    )
    _record_diagnostic_timing(record_timing, "sample_point_spectra", started)

    if plot_sample_locations:
        started = time.perf_counter()
        sed_background = visualization.composite_band_image_on_model_wcs(
            {
                int(exp_id): {
                    str(band): np.asarray(image, dtype=np.float32) for band, image in band_map.items()
                }
                for exp_id, band_map in background_observations.items()
            },
            {int(exp_id): str(band) for exp_id, band in image_band_by_exposure.items()},
            dict(exposures),
            str(diagnostic_band),
            getattr(scene, "model_wcs", None),
        )
        if sed_background is None:
            sed_background = location_background_fallback
        sed_plot_points = list(sed_points)
        model_wcs = getattr(scene, "model_wcs", None)
        primary_obs_wcs = getattr(exposure, "obs_wcs", None) or getattr(exposure, "wcs", None)
        if (
            model_wcs is not None
            and sed_background is not None
            and tuple(np.asarray(sed_background).shape)
            == (int(getattr(model_wcs, "height")), int(getattr(model_wcs, "width")))
        ):
            sed_plot_points = visualization.project_sample_points_to_wcs(
                sed_points, from_wcs=primary_obs_wcs, to_wcs=model_wcs,
            )
        visualization.plot_sed_sample_locations(
            observation, str(diagnostic_band), run_dir, sample_points=sed_plot_points,
            region_masks=point_region_masks, region_overlays=dict(region_overlays or {}),
            background_image=sed_background,
            title=f"{diagnostic_band}-band SED sample locations on full composite",
        )
        _record_diagnostic_timing(record_timing, "plot_sed_sample_locations", started)

    lam_grid = trainer.diagnostic_renderer.grid.lam.detach().cpu().numpy()
    started = time.perf_counter()
    visualization.plot_point_seds(
        lam_grid, point_seds, dict(bandpasses), run_dir, truth_seds=getattr(scene, "pixel_truth_seds", None),
        use_flux_density=bool(trainer.use_flux_density),
    )
    _record_diagnostic_timing(record_timing, "plot_point_seds", started)

def write_fiber_spectrum_diagnostics(
    *, fiber_spectra: Mapping[int, tuple[np.ndarray, np.ndarray]],
    spec_diagnostics_by_exposure: Mapping[int, SpectrumDiagnosticTarget], use_flux_density: bool,
    run_dir: Path, max_overlay_spectra: int | None = 12, max_individual_spectra: int | None = None,
    include_additive: bool = True,
    spec_additive_by_exposure: Mapping[int, np.ndarray] | None = None,
    save_fiber_spectra: bool = False,
    spec_sigma_by_exposure: Mapping[int, np.ndarray] | None = None,
) -> None:
    if not fiber_spectra:
        return
    if not include_additive:
        # Additive-free predictions must be scored against additive-free truth;
        # dividing by source+sky dilutes recovery by the fiber's sky fraction.
        if not use_flux_density:
            raise ValueError(
                "include_additive=False requires flux-density truth; the stored "
                "additive component has no counts-space counterpart."
            )
        if spec_additive_by_exposure is None:
            raise ValueError(
                "include_additive=False requires spec_additive_by_exposure so the "
                "additive component can be removed from the truth spectra."
            )

    # A non-finite prediction is an evaluator failure, not a spectrum. Checking
    # every exposure first keeps the run directory untouched: no plot for an
    # earlier fiber, no fiber_metrics.json and no fiber_spectra.npz can present
    # the run as scored.
    for exp_id in sorted(int(exp_id) for exp_id in fiber_spectra.keys()):
        prediction = np.asarray(fiber_spectra[int(exp_id)][1], dtype=np.float64)
        n_nonfinite = int(np.count_nonzero(~np.isfinite(prediction)))
        if n_nonfinite:
            target = spec_diagnostics_by_exposure.get(int(exp_id))
            label = "<missing target>" if target is None else str(target.label)
            raise ValueError(
                f"Predicted spectrum for spectral exposure {int(exp_id)} / label "
                f"{label!r} has {n_nonfinite} non-finite of {prediction.size} bins."
            )

    from . import visualization

    overlay_items: list[dict[str, object]] = []
    metrics_payload: dict[str, dict[str, float | int | str]] = {}
    spectra_payload: dict[str, np.ndarray] = {}
    spectra_exposure_ids: list[int] = []
    primary_exp_id: int | None = None
    max_individual = None if max_individual_spectra is None else max(0, int(max_individual_spectra))

    for plot_index, exp_id in enumerate(sorted(int(exp_id) for exp_id in fiber_spectra.keys())):
        if int(exp_id) not in spec_diagnostics_by_exposure:
            raise ValueError(
                f"Missing spec_diagnostics_by_exposure entry for spectral exposure {int(exp_id)}."
            )
        target = spec_diagnostics_by_exposure[int(exp_id)]
        lam_pred, pred_spec = fiber_spectra[int(exp_id)]
        lam_plot = _validated_lam_for_target(lam_pred, target, exposure_id=int(exp_id))
        truth_spec = _truth_for_target(target, use_flux_density=use_flux_density, exposure_id=int(exp_id))
        pred_arr = np.asarray(pred_spec, dtype=np.float64)
        if pred_arr.shape != truth_spec.shape:
            raise ValueError(
                f"Predicted spectrum shape {pred_arr.shape} does not match truth "
                f"{truth_spec.shape} for spectral exposure {int(exp_id)}."
            )
        additive_sum: float | None = None
        if not include_additive:
            if int(exp_id) not in spec_additive_by_exposure:
                raise ValueError(
                    f"Missing spec_additive_by_exposure entry for spectral exposure {int(exp_id)}."
                )
            additive = np.asarray(spec_additive_by_exposure[int(exp_id)], dtype=np.float64)
            if additive.shape != truth_spec.shape:
                raise ValueError(
                    f"Additive spectrum shape {additive.shape} does not match truth "
                    f"{truth_spec.shape} for spectral exposure {int(exp_id)}."
                )
            truth_spec = truth_spec - additive
            additive_sum = float(additive.sum())
        label = str(target.label)
        plot_name = f"fiber_spectrum_e{int(exp_id)}_{_sanitize_plot_label(label)}.png"
        throughput = (None if target.throughput is None else np.asarray(target.throughput, dtype=np.float64))
        should_write_individual = max_individual is None or plot_index < max_individual
        if should_write_individual:
            visualization.plot_fiber_spectrum(
                lam_plot, pred_arr, truth_spec, throughput, run_dir,
                filename=plot_name, title=f"Fiber spectrum - {label} (e{int(exp_id)})",
            )
            if primary_exp_id is None:
                primary_exp_id = int(exp_id)
                visualization.plot_fiber_spectrum(
                    lam_plot, pred_arr, truth_spec, throughput, run_dir,
                    filename="fiber_spectrum.png", title=f"Fiber spectrum - {label} (e{int(exp_id)})",
                )
        overlay_items.append(
            {
                "label": f"{label} (e{int(exp_id)})", "lam": lam_plot,
                "predicted": pred_arr, "truth": truth_spec,
            }
        )
        entry: dict[str, float | int | str | bool] = {
            "exposure_id": int(exp_id), "label": label, "role": str(target.role),
            "truth_includes_additive": bool(include_additive),
            **_spectrum_error_summary(pred_arr, truth_spec, lam_plot),
        }
        if additive_sum is not None:
            entry["additive_flux_sum"] = additive_sum
        metrics_payload[str(int(exp_id))] = entry
        if save_fiber_spectra:
            if spec_sigma_by_exposure is None or int(exp_id) not in spec_sigma_by_exposure:
                raise ValueError(
                    "save_fiber_spectra=True requires NLL sigma for every scored "
                    f"spectral exposure (missing exposure {int(exp_id)})."
                )
            sigma = np.asarray(spec_sigma_by_exposure[int(exp_id)], dtype=np.float64)
            if sigma.shape != lam_plot.shape:
                raise ValueError(
                    f"NLL sigma shape {sigma.shape} does not match wavelength grid "
                    f"{lam_plot.shape} for spectral exposure {int(exp_id)}."
                )
            prefix = f"e{int(exp_id)}"
            spectra_payload.update(
                {
                    f"{prefix}_lam": np.asarray(lam_plot, dtype=np.float64),
                    f"{prefix}_predicted": pred_arr,
                    f"{prefix}_truth": np.asarray(truth_spec, dtype=np.float64),
                    # This is the exact per-datum sigma tensor consumed by the
                    # trainer NLL, sliced and ordered by spectral datum_index.
                    f"{prefix}_sigma": sigma,
                    f"{prefix}_label": np.asarray(label),
                    f"{prefix}_role": np.asarray(str(target.role)),
                    f"{prefix}_truth_includes_additive": np.asarray(bool(include_additive)),
                }
            )
            spectra_exposure_ids.append(int(exp_id))

    if len(overlay_items) >= 2:
        visualization.plot_fiber_spectra_overlay(overlay_items, run_dir, max_spectra=max_overlay_spectra)
    if metrics_payload:
        with (run_dir / "fiber_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
    if save_fiber_spectra and spectra_exposure_ids:
        np.savez(
            run_dir / "fiber_spectra.npz",
            exposure_ids=np.asarray(spectra_exposure_ids, dtype=np.int64),
            **spectra_payload,
        )


def _peak_center(image: np.ndarray) -> tuple[float, float]:
    image_arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image_arr)
    if not np.any(finite):
        return 0.5 * (image_arr.shape[0] - 1), 0.5 * (image_arr.shape[1] - 1)
    background = float(np.median(image_arr[finite]))
    signal = np.where(finite, image_arr - background, -np.inf)
    peak = float(np.max(signal))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0.5 * (image_arr.shape[0] - 1), 0.5 * (image_arr.shape[1] - 1)
    y_peak, x_peak = np.unravel_index(int(np.argmax(signal)), image_arr.shape)
    return float(y_peak), float(x_peak)


def log_radial_diagnostics(
    observations: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray], held_out_band: str | None = None,
    partial_holdout_enabled: bool = False, radial_bin_width: float = 0.5,
) -> None:
    if radial_bin_width <= 0:
        raise ValueError("radial_bin_width must be > 0.")
    if not observations:
        return
    def radial_profile(
        image_arr: np.ndarray, *, center: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image_arr, dtype=np.float64)
        H, W = image.shape
        yy, xx = np.indices((H, W), dtype=np.float64)
        cy, cx = center
        r_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        n_bins = int(np.ceil(r_map.max() / radial_bin_width))
        if n_bins <= 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        radial_centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * radial_bin_width
        bin_index = np.minimum((r_map.flatten() / radial_bin_width).astype(np.int64), n_bins - 1)
        counts = np.bincount(bin_index, minlength=n_bins)
        sums = np.bincount(bin_index, weights=image.flatten(), minlength=n_bins)
        profile = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        return radial_centers, profile

    def _interp_cross(
        radial_centers: np.ndarray, values: np.ndarray, half_level: float, search_from_start: bool,
    ) -> float:
        valid = np.isfinite(values)
        rvals = radial_centers[valid]
        vals = values[valid]
        if rvals.size == 0:
            return float("nan")
        if search_from_start:
            idx_candidates = np.where(vals >= half_level)[0]
            if idx_candidates.size == 0:
                return float("nan")
            idx = idx_candidates[0]
            if idx == 0:
                return float(rvals[0])
            r0, r1 = rvals[idx - 1], rvals[idx]
            v0, v1 = vals[idx - 1], vals[idx]
        else:
            idx_candidates = np.where(vals >= half_level)[0]
            if idx_candidates.size == 0:
                return float("nan")
            idx = idx_candidates[-1]
            if idx == len(rvals) - 1:
                return float(rvals[-1])
            r0, r1 = rvals[idx], rvals[idx + 1]
            v0, v1 = vals[idx], vals[idx + 1]
        if np.isclose(v1, v0):
            return float(r1)
        frac = (half_level - v0) / (v1 - v0)
        return float(r0 + np.clip(frac, 0.0, 1.0) * (r1 - r0))

    def compute_fwhm(radial_centers: np.ndarray, profile: np.ndarray) -> float:
        if not np.any(np.isfinite(profile)):
            return float("nan")
        peak = float(np.nanmax(profile))
        if not np.isfinite(peak) or peak <= 0:
            return float("nan")
        half = 0.5 * peak
        inner = _interp_cross(radial_centers, profile, half, True)
        outer = _interp_cross(radial_centers, profile, half, False)
        if not np.isfinite(inner) or not np.isfinite(outer):
            return float("nan")
        return max(outer - inner, 0.0)

    # Sort bands in ugriz order
    band_order = ["u", "g", "r", "i", "z"]
    sorted_bands = sorted(
        observations.keys(), key=lambda b: band_order.index(b) if b in band_order else len(band_order),
    )
    profile_data: list[
        tuple[str, tuple[float, float], tuple[float, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for band in sorted_bands:
        obs_arr = observations[band]
        if band not in predictions:
            continue
        pred_arr = predictions[band]
        obs_peak = _peak_center(obs_arr)
        pred_peak = _peak_center(pred_arr)
        obs_centers, obs_prof = radial_profile(obs_arr, center=obs_peak)
        pred_centers, pred_prof = radial_profile(pred_arr, center=obs_peak)
        if obs_centers.size == 0 or pred_centers.size == 0:
            continue
        profile_data.append((band, obs_peak, pred_peak, obs_centers, obs_prof, pred_centers, pred_prof))

    if not profile_data:
        return

    print("\n" + "=" * 70)
    print("RADIAL PROFILE TEST (ring diagnostics) [obs-peak-centered]")
    print("=" * 70)

    for band, obs_peak, pred_peak, obs_centers, obs_prof, pred_centers, pred_prof in profile_data:
        if obs_centers.shape != pred_centers.shape or not np.allclose(obs_centers, pred_centers):
            centers = obs_centers
            pred_prof = np.interp(centers, pred_centers, pred_prof, left=np.nan, right=np.nan)
        else:
            centers = obs_centers

        obs_fwhm = compute_fwhm(obs_centers, obs_prof)
        pred_fwhm = compute_fwhm(centers, pred_prof)

        if np.isfinite(obs_fwhm) and np.isfinite(pred_fwhm):
            if pred_fwhm > obs_fwhm:
                ring_hint = "pred broader → expect negative core / positive ring"
            elif pred_fwhm < obs_fwhm:
                ring_hint = "pred sharper → expect positive core / negative ring"
            else:
                ring_hint = "matched widths"
        else:
            ring_hint = "insufficient signal for FWHM"

        print(f"\nBand {band}:")
        peak_sep = float(np.hypot(pred_peak[0] - obs_peak[0], pred_peak[1] - obs_peak[1]))
        print(f"  Peak(obs)  = ({obs_peak[0]:.1f}, {obs_peak[1]:.1f}) pix")
        print(f"  Peak(pred) = ({pred_peak[0]:.1f}, {pred_peak[1]:.1f}) pix")
        print(f"  Peak Δ     = {peak_sep:.3f} pix")
        print(f"  FWHM(obs)  = {obs_fwhm:.3f} pix")
        print(f"  FWHM(pred) = {pred_fwhm:.3f} pix")
        print(f"  Diagnostic : {ring_hint}")
        if band == held_out_band:
            if partial_holdout_enabled:
                print("  *** PARTIAL-HOLDOUT TRAINING BAND ***")
            else:
                print("  *** HELD-OUT BAND ***")
