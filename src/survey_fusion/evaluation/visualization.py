from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from survey_fusion.data.regions import (
    SkyCircleRegion, local_sky_circle_mask_from_omega_wcs, sky_circle_mask_from_omega_wcs,
)
from survey_fusion.data.bandpasses import Bandpass


DEFAULT_MAX_FALLBACK_REGIONS = 12


def _robust_percentile_limits(*arrays: np.ndarray, lower_pct: float, upper_pct: float) -> tuple[float, float]:
    finite_values = [
        np.asarray(arr, dtype=np.float64)[np.isfinite(np.asarray(arr, dtype=np.float64))]
        for arr in arrays
    ]
    finite_values = [values for values in finite_values if values.size > 0]
    if not finite_values:
        return 0.0, 1.0
    combined = np.concatenate(finite_values, axis=0)
    vmin = float(np.nanpercentile(combined, lower_pct))
    vmax = float(np.nanpercentile(combined, upper_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def _robust_symmetric_limit(image: np.ndarray, *, upper_pct: float) -> float:
    values = np.abs(np.asarray(image, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(values, upper_pct))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return vmax


def _finite_bbox(mask: np.ndarray, pad_pixels: int = 0) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    y0 = max(int(yy.min()) - int(pad_pixels), 0)
    y1 = min(int(yy.max()) + int(pad_pixels) + 1, mask.shape[0])
    x0 = max(int(xx.min()) - int(pad_pixels), 0)
    x1 = min(int(xx.max()) + int(pad_pixels) + 1, mask.shape[1])
    return y0, y1, x0, x1


def _observed_footprint_views(
    observed: np.ndarray, predicted: np.ndarray, *, pad_pixels: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed/predicted/residual views restricted to observed finite pixels."""
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(f"predicted shape {pred.shape} does not match observed shape {obs.shape}.")
    finite_obs = np.isfinite(obs)
    bbox = _finite_bbox(finite_obs, pad_pixels=pad_pixels)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        obs = obs[y0:y1, x0:x1]
        pred = pred[y0:y1, x0:x1]
        finite_obs = finite_obs[y0:y1, x0:x1]
    pred_display = np.where(finite_obs, pred, np.nan)
    resid = pred_display - obs
    return obs, pred_display, resid


def plot_band_grid(
    observations: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray], run_dir: Path,
    held_out: str | None, *, filename: str = "band_grid.png",
) -> None:
    bands = [band for band in observations.keys() if band in predictions]
    if not bands:
        return
    n_bands = len(bands)
    fig, axes = plt.subplots(3, n_bands, figsize=(3.2 * n_bands, 9), squeeze=False)
    display_by_band: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for band in bands:
        display_by_band[band] = _observed_footprint_views(observations[band], predictions[band])
    for idx, band in enumerate(bands):
        obs_display, pred_display, _ = display_by_band[band]
        vmin, vmax = _robust_percentile_limits(obs_display, pred_display, lower_pct=5.0, upper_pct=99.7)
        ax = axes[0, idx]
        im = ax.imshow(obs_display, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"Observed {band}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for idx, band in enumerate(bands):
        obs_display, pred_display, _ = display_by_band[band]
        vmin, vmax = _robust_percentile_limits(obs_display, pred_display, lower_pct=5.0, upper_pct=99.7)
        ax = axes[1, idx]
        im = ax.imshow(pred_display, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        title = f"Predicted {band}"
        is_held_out = held_out is not None and (
            band == held_out or band.endswith(f":{held_out}")
        )
        if is_held_out:
            ax.set_title(f"{title} (HELD OUT)", color="red")
        else:
            ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for idx, band in enumerate(bands):
        ax = axes[2, idx]
        _, _, resid = display_by_band[band]
        vmax = _robust_symmetric_limit(resid, upper_pct=99.5)
        im = ax.imshow(resid, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Residual {band}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Galaxy Spectral Field Results")
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200)
    plt.close(fig)


def plot_true_band_with_fiber_mask(
    image: np.ndarray, fiber_mask: np.ndarray, band: str, run_dir: Path,
) -> None:
    """
    Draw the true image for a single band and overlay the SDSS science-fiber apertures.
    """
    mask = fiber_mask.astype(bool)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(image, cmap="viridis", origin="lower")
    ax.set_title(f"True {band} image with SDSS science fiber apertures")
    ax.axis("off")
    if mask.any():
        overlay = np.ma.masked_where(~mask, mask.astype(np.float32))
        ax.imshow(overlay, cmap="Reds", origin="lower", alpha=0.35, vmin=0, vmax=1)
        ax.contour(mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.25, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / f"true_{band}_fiber_overlay.png", dpi=200)
    plt.close(fig)


def plot_holdout_region_overlay(
    image: np.ndarray, holdout_mask: np.ndarray, band: str, run_dir: Path,
) -> None:
    mask = np.asarray(holdout_mask).astype(bool)
    if mask.shape != image.shape:
        raise ValueError(f"holdout mask shape {mask.shape} does not match image shape {image.shape}.")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    vmin = np.nanpercentile(image, 5)
    vmax = np.nanpercentile(image, 99.5)
    im = ax.imshow(image, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    if np.any(mask):
        overlay = np.ma.masked_where(~mask, mask.astype(np.float32))
        ax.imshow(overlay, cmap="Reds", origin="lower", alpha=0.3, vmin=0, vmax=1)
        ax.contour(mask.astype(np.float32), levels=[0.5], colors="white", linewidths=1.2)
    ax.set_title(f"{band}-band holdout region")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / f"holdout_region_{band}.png", dpi=200)
    plt.close(fig)


def default_region_masks_from_scene(
    scene, *, max_regions: int | None = DEFAULT_MAX_FALLBACK_REGIONS,
) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float | str]]]:
    spectra = list(getattr(scene, "spectra", []))
    region_masks: Dict[str, np.ndarray] = {}
    region_overlays: Dict[str, Dict[str, float | str]] = {}
    pixel_scale_arcsec_per_band = dict(getattr(scene, "pixel_scale_arcsec_per_band", {}) or {})
    for idx, spectrum in enumerate(spectra):
        role = str(getattr(spectrum, "role", "science")).strip().lower()
        if role != "science":
            continue
        if max_regions is not None and int(max_regions) > 0 and len(region_masks) >= int(max_regions):
            break
        mask = np.asarray(getattr(spectrum, "fiber_mask", None))
        if mask.ndim != 2 or not np.any(mask):
            continue
        label = str(getattr(spectrum, "label", f"S{idx + 1}")).strip() or f"S{idx + 1}"
        name = label
        suffix = 2
        while name in region_masks:
            name = f"{label}_{suffix}"
            suffix += 1
        region_masks[name] = mask.astype(bool)

        metadata: Dict[str, Any] = dict(getattr(spectrum, "metadata", {}) or {})
        center = metadata.get("fiber_center_pixel_ref")
        ref_band = str(metadata.get("fiber_center_pixel_ref_band", "") or "")
        diameter_arcsec = metadata.get("fiber_diameter_arcsec")
        if (
            isinstance(center, (list, tuple))
            and len(center) == 2
            and np.all(np.isfinite(center))
            and isinstance(diameter_arcsec, (int, float))
            and np.isfinite(float(diameter_arcsec))
            and float(diameter_arcsec) > 0.0
        ):
            pixel_scale_arcsec = pixel_scale_arcsec_per_band.get(ref_band)
            if pixel_scale_arcsec is None:
                pixel_scale_arcsec = getattr(scene, "pixel_scale_arcsec", None)
            if (
                pixel_scale_arcsec is not None
                and np.isfinite(float(pixel_scale_arcsec))
                and float(pixel_scale_arcsec) > 0.0
            ):
                region_overlays[name] = {
                    "kind": "circle", "center_x": float(center[0]), "center_y": float(center[1]),
                    "radius_pix": 0.5 * float(diameter_arcsec) / float(pixel_scale_arcsec),
                }
    return region_masks, region_overlays


def _science_regions_from_scene(
    scene, *, max_regions: int | None = DEFAULT_MAX_FALLBACK_REGIONS,
) -> list[tuple[str, SkyCircleRegion]]:
    spectra = list(getattr(scene, "spectra", []))
    regions: list[tuple[str, SkyCircleRegion]] = []
    seen_names: set[str] = set()
    for idx, spectrum in enumerate(spectra):
        role = str(getattr(spectrum, "role", "science")).strip().lower()
        if role != "science":
            continue
        if max_regions is not None and int(max_regions) > 0 and len(regions) >= int(max_regions):
            break
        label = str(getattr(spectrum, "label", f"S{idx + 1}")).strip() or f"S{idx + 1}"
        name = label
        suffix = 2
        while name in seen_names:
            name = f"{label}_{suffix}"
            suffix += 1
        seen_names.add(name)
        ra_deg, dec_deg = tuple(
            float(value) for value in getattr(spectrum, "center_ra_dec", (np.nan, np.nan))
        )
        if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
            continue
        metadata: Dict[str, Any] = dict(getattr(spectrum, "metadata", {}) or {})
        diameter_arcsec = metadata.get("fiber_diameter_arcsec")
        if isinstance(diameter_arcsec, (int, float)) and np.isfinite(float(diameter_arcsec)):
            radius_arcsec = 0.5 * float(diameter_arcsec)
        else:
            radius_arcsec = float(getattr(spectrum, "fiber_radius_arcsec", 0.0) or 0.0)
        if not np.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
            continue
        regions.append(
            (
                name,
                SkyCircleRegion(
                    name=name, center_ra_deg=ra_deg, center_dec_deg=dec_deg, radius_arcsec=radius_arcsec,
                ),
            )
        )
    return regions


def default_region_masks_for_exposure(
    scene, exposure, *, max_regions: int | None = DEFAULT_MAX_FALLBACK_REGIONS,
) -> Dict[str, np.ndarray]:
    if max_regions is not None and int(max_regions) <= 0:
        return {}
    image_h = getattr(exposure, "image_height", None)
    image_w = getattr(exposure, "image_width", None)
    if image_h is None or image_w is None:
        fiber_mask = np.asarray(getattr(exposure, "fiber_mask", None))
        if fiber_mask.ndim != 2:
            return {}
        image_shape = fiber_mask.shape
    else:
        image_shape = (int(image_h), int(image_w))
    obs_wcs = getattr(exposure, "obs_wcs", None) or getattr(exposure, "wcs", None)
    if obs_wcs is None:
        return {}
    region_masks: Dict[str, np.ndarray] = {}
    pixel_scale_arcsec = float(getattr(exposure, "pixel_scale_arcsec", 0.0) or 0.0)
    for name, region in _science_regions_from_scene(scene, max_regions=None):
        mask = local_sky_circle_mask_from_omega_wcs(
            obs_wcs, image_shape, region, pixel_scale_arcsec=pixel_scale_arcsec,
        )
        if mask is None:
            mask = sky_circle_mask_from_omega_wcs(obs_wcs, image_shape, region)
        mask_arr = np.asarray(mask, dtype=bool)
        if not np.any(mask_arr):
            continue
        region_masks[name] = mask_arr
        if max_regions is not None and len(region_masks) >= int(max_regions):
            break
    return region_masks


def resolve_region_masks_by_exposure(
    scene, configured_region_masks_by_exposure: Dict[int, Dict[str, np.ndarray]] | None,
    exposures: Dict[int, Any], image_exposure_ids: Sequence[int], *, allow_fallback: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    resolved: Dict[int, Dict[str, np.ndarray]] = {}
    for exp_id in image_exposure_ids:
        configured_masks = None
        if isinstance(configured_region_masks_by_exposure, dict):
            configured_masks = configured_region_masks_by_exposure.get(int(exp_id))
        if isinstance(configured_masks, dict) and configured_masks:
            resolved[int(exp_id)] = {
                str(name): np.asarray(mask, dtype=bool) for name, mask in configured_masks.items()
            }
            continue
        if not allow_fallback:
            continue
        exposure = exposures.get(int(exp_id))
        if exposure is None:
            continue
        fallback_masks = default_region_masks_for_exposure(scene, exposure)
        if fallback_masks:
            resolved[int(exp_id)] = fallback_masks
    return resolved


def resolve_region_masks(
    scene, configured_region_masks: Dict[str, np.ndarray] | None, *, primary_region: str | None,
    exposure=None, allow_fallback: bool = True,
) -> tuple[Dict[str, np.ndarray], str | None, int, Dict[str, Dict[str, float | str]]]:
    if isinstance(configured_region_masks, dict) and configured_region_masks:
        return configured_region_masks, primary_region, 6, {}
    if not allow_fallback:
        return {}, primary_region, 6, {}
    fallback_overlays: Dict[str, Dict[str, float | str]] = {}
    if exposure is not None:
        fallback_masks = default_region_masks_for_exposure(scene, exposure)
    else:
        fallback_masks, fallback_overlays = default_region_masks_from_scene(scene)
    if fallback_masks:
        resolved_primary = (
            primary_region if primary_region in fallback_masks else next(iter(fallback_masks))
        )
        return fallback_masks, resolved_primary, 10, fallback_overlays
    return {}, primary_region, 6, {}


def resolve_point_sed_region_masks(
    scene, configured_region_masks: Dict[str, np.ndarray] | None, *, exposure=None, allow_fallback: bool=True,
) -> Dict[str, np.ndarray]:
    if allow_fallback:
        if exposure is not None:
            fallback_masks = default_region_masks_for_exposure(scene, exposure)
        else:
            fallback_masks, _ = default_region_masks_from_scene(scene)
    else:
        fallback_masks = {}
    merged: Dict[str, np.ndarray] = dict(fallback_masks)
    if isinstance(configured_region_masks, dict):
        for name, mask in configured_region_masks.items():
            merged[str(name)] = np.asarray(mask, dtype=bool)
    return merged


def _mask_bbox(mask: np.ndarray, pad_pixels: int = 4) -> tuple[int, int, int, int] | None:
    return _finite_bbox(mask, pad_pixels=pad_pixels)


def plot_region_cutout_grid(
    observed: np.ndarray, predicted: np.ndarray, region_masks: Dict[str, np.ndarray], band: str,
    run_dir: Path, *, primary_region: str | None = None, pad_pixels: int = 6,
    region_overlays: Dict[str, Dict[str, float | str]] | None = None, filename: str | None = None,
    max_regions: int | None = 12,
) -> None:
    if not region_masks:
        return
    if observed.shape != predicted.shape:
        raise ValueError(f"predicted shape {predicted.shape} does not match observed shape {observed.shape}.")
    def _finite_observed_count(name: str) -> int:
        mask = np.asarray(region_masks[name]).astype(bool)
        if mask.shape != observed.shape:
            return -1
        bbox = _mask_bbox(mask, pad_pixels=pad_pixels)
        if bbox is None:
            return 0
        y0, y1, x0, x1 = bbox
        return int(np.count_nonzero(np.isfinite(observed[y0:y1, x0:x1]) & mask[y0:y1, x0:x1]))

    counts_by_name = {str(name): _finite_observed_count(str(name)) for name in region_masks.keys()}
    positive_names = [name for name, count in counts_by_name.items() if count > 0]
    names = positive_names if positive_names else list(region_masks.keys())
    names = sorted(names, key=lambda item: (-counts_by_name.get(str(item), 0), str(item)))
    if primary_region is not None and primary_region in region_masks:
        names = [primary_region] + [name for name in names if name != primary_region]
    if max_regions is not None and int(max_regions) > 0:
        names = names[: int(max_regions)]
    n_rows = len(names)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(10, max(3.0 * n_rows, 3.0)), squeeze=False, constrained_layout=True,
    )
    for row, name in enumerate(names):
        mask = np.asarray(region_masks[name]).astype(bool)
        if mask.shape != observed.shape:
            raise ValueError(
                f"region mask {name!r} shape {mask.shape} does not match image shape {observed.shape}."
            )
        bbox = _mask_bbox(mask, pad_pixels=pad_pixels)
        if bbox is None:
            for col in range(3):
                ax = axes[row, col]
                ax.axis("off")
                ax.text(0.5, 0.5, f"{name}: empty mask", ha="center", va="center")
            continue
        y0, y1, x0, x1 = bbox
        obs_cut = observed[y0:y1, x0:x1]
        pred_cut = predicted[y0:y1, x0:x1]
        obs_finite = np.isfinite(obs_cut)
        pred_cut = np.where(obs_finite, pred_cut, np.nan)
        resid_cut = pred_cut - obs_cut
        mask_cut = mask[y0:y1, x0:x1]
        title_prefix = f"{name}" + (" [PRIMARY]" if primary_region == name else "")
        ax_obs, ax_pred, ax_resid = axes[row]
        has_image_pixels = bool(obs_finite.any())
        if not has_image_pixels:
            for ax, suffix in ((ax_obs, "observed"), (ax_pred, "predicted"), (ax_resid, "residual")):
                ax.axis("off")
                ax.set_title(f"{title_prefix} {suffix}")
                ax.text(0.5, 0.5, "no finite observed pixels", ha="center", va="center")
            continue
        vmin, vmax = _robust_percentile_limits(obs_cut, pred_cut, lower_pct=0.0, upper_pct=100.0)
        rv = _robust_symmetric_limit(resid_cut, upper_pct=100.0)
        im_obs = ax_obs.imshow(obs_cut, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        im_pred = ax_pred.imshow(pred_cut, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        im_resid = ax_resid.imshow(resid_cut, origin="lower", cmap="RdBu_r", vmin=-rv, vmax=rv)
        overlay = {}
        if isinstance(region_overlays, dict):
            overlay = dict(region_overlays.get(name, {}) or {})
        for ax in (ax_obs, ax_pred, ax_resid):
            if (overlay.get("kind") == "circle" and {"center_x", "center_y", "radius_pix"} <= set(overlay)):
                circle = Circle(
                    (float(overlay["center_x"]) - float(x0), float(overlay["center_y"]) - float(y0)),
                    radius=float(overlay["radius_pix"]), facecolor="none", edgecolor="white", linewidth=1.0,
                )
                ax.add_patch(circle)
            elif np.any(mask_cut):
                ax.contour(mask_cut.astype(np.float32), levels=[0.5], colors="white", linewidths=1.0)
            ax.axis("off")
        ax_obs.set_title(f"{title_prefix} observed")
        ax_pred.set_title(f"{title_prefix} predicted")
        ax_resid.set_title(f"{title_prefix} residual")
        fig.colorbar(im_obs, ax=ax_obs, fraction=0.046)
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046)
        fig.colorbar(im_resid, ax=ax_resid, fraction=0.046)
    fig.savefig(run_dir / (filename or f"region_grid_{band}.png"), dpi=200)
    plt.close(fig)


def composite_band_image(
    images_by_exposure: Dict[int, Dict[str, np.ndarray]], image_band_by_exposure: Dict[int, str], band: str,
) -> np.ndarray | None:
    """Build a finite-pixel mean image for one band across repeated exposures."""
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    for exp_id in sorted(images_by_exposure.keys()):
        if str(image_band_by_exposure.get(int(exp_id), "")) != str(band):
            continue
        image = images_by_exposure.get(int(exp_id), {}).get(str(band))
        if image is None:
            continue
        arr = np.asarray(image, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        if sums is None:
            sums = np.zeros(arr.shape, dtype=np.float64)
            counts = np.zeros(arr.shape, dtype=np.int32)
        if sums.shape != arr.shape or counts is None or counts.shape != arr.shape:
            continue
        sums[finite] += arr[finite]
        counts[finite] += 1
    if sums is None or counts is None:
        return None
    composite = np.full(sums.shape, np.nan, dtype=np.float32)
    valid = counts > 0
    composite[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return composite


def composite_band_image_on_model_wcs(
    images_by_exposure: Dict[int, Dict[str, np.ndarray]], image_band_by_exposure: Dict[int, str],
    exposures: Dict[int, Any], band: str, model_wcs: Any, *, chunk_size: int = 1_000_000,
) -> np.ndarray | None:
    """Project repeated exposure cutouts onto the shared model WCS for diagnostics."""
    if model_wcs is None or not hasattr(model_wcs, "omega_xyz_to_pixel"):
        return composite_band_image(images_by_exposure, image_band_by_exposure, band)
    height = getattr(model_wcs, "height", None)
    width = getattr(model_wcs, "width", None)
    if height is None or width is None:
        return composite_band_image(images_by_exposure, image_band_by_exposure, band)
    out_shape = (int(height), int(width))
    sums = np.zeros(out_shape, dtype=np.float64)
    counts = np.zeros(out_shape, dtype=np.int32)
    any_pixels = False
    chunk = max(1, int(chunk_size))
    for exp_id in sorted(images_by_exposure.keys()):
        if str(image_band_by_exposure.get(int(exp_id), "")) != str(band):
            continue
        image = images_by_exposure.get(int(exp_id), {}).get(str(band))
        exposure = exposures.get(int(exp_id)) if isinstance(exposures, dict) else None
        obs_wcs = getattr(exposure, "obs_wcs", None) or getattr(exposure, "wcs", None)
        if image is None or obs_wcs is None or not hasattr(obs_wcs, "pixel_to_omega_xyz"):
            continue
        arr = np.asarray(image, dtype=np.float32)
        finite_flat = np.flatnonzero(np.isfinite(arr.reshape(-1)))
        if finite_flat.size == 0:
            continue
        h, w = arr.shape
        values_flat = arr.reshape(-1)
        for start in range(0, int(finite_flat.size), chunk):
            local = finite_flat[start:start + chunk]
            y_local, x_local = np.divmod(local, int(w))
            omega = np.asarray(
                obs_wcs.pixel_to_omega_xyz(
                    x_local.astype(np.float64, copy=False), y_local.astype(np.float64, copy=False),
                ),
                dtype=np.float64,
            )
            x_model, y_model = model_wcs.omega_xyz_to_pixel(omega)
            xi = np.rint(np.asarray(x_model, dtype=np.float64)).astype(np.int64)
            yi = np.rint(np.asarray(y_model, dtype=np.float64)).astype(np.int64)
            inside = (
                np.isfinite(x_model)
                & np.isfinite(y_model)
                & (xi >= 0)
                & (xi < out_shape[1])
                & (yi >= 0)
                & (yi < out_shape[0])
            )
            if not np.any(inside):
                continue
            np.add.at(sums, (yi[inside], xi[inside]), values_flat[local[inside]])
            np.add.at(counts, (yi[inside], xi[inside]), 1)
            any_pixels = True
    if not any_pixels:
        return None
    composite = np.full(out_shape, np.nan, dtype=np.float32)
    valid = counts > 0
    composite[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return composite


def project_sample_points_to_wcs(
    sample_points: Sequence[Tuple[str, int, int]], *, from_wcs: Any, to_wcs: Any,
) -> List[Tuple[str, int, int]]:
    if (
        from_wcs is None
        or to_wcs is None
        or not hasattr(from_wcs, "pixel_to_omega_xyz")
        or not hasattr(to_wcs, "omega_xyz_to_pixel")
    ):
        return [(str(label), int(y), int(x)) for label, y, x in sample_points]
    projected: List[Tuple[str, int, int]] = []
    for label, y, x in sample_points:
        omega = np.asarray(
            from_wcs.pixel_to_omega_xyz(
                np.asarray([float(x)], dtype=np.float64), np.asarray([float(y)], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        x_out, y_out = to_wcs.omega_xyz_to_pixel(omega)
        projected.append(
            (
                str(label), int(np.rint(float(np.asarray(y_out).reshape(-1)[0]))),
                int(np.rint(float(np.asarray(x_out).reshape(-1)[0]))),
            )
        )
    return projected


def plot_history(history: Dict[str, list[float]], run_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogy(history["loss"], label="Total", alpha=0.8)
    ax.semilogy(history["image_loss"], label="Images", alpha=0.6)
    spec_loss = np.asarray(history.get("spec_loss", []), dtype=np.float64)
    if spec_loss.size > 0:
        n_spec_raw = history.get("n_spec")
        if n_spec_raw is not None:
            n_spec = np.asarray(n_spec_raw, dtype=np.int64)
            n = min(spec_loss.size, n_spec.size)
            spec_loss = spec_loss[:n]
            x = np.arange(n, dtype=np.int64)
            mask = (n_spec[:n] > 0) & np.isfinite(spec_loss)
        else:
            x = np.arange(spec_loss.size, dtype=np.int64)
            mask = np.isfinite(spec_loss) & (spec_loss > 0.0)
        if np.any(mask):
            ax.semilogy(x[mask], spec_loss[mask], label="Spectrum", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_history.png", dpi=200)
    plt.close(fig)


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


def plot_radial_profiles(observed: np.ndarray, predicted: np.ndarray, band: str, run_dir: Path) -> None:
    H, W = observed.shape
    yy, xx = np.indices((H, W), dtype=np.float64)
    cy, cx = _peak_center(observed)
    r_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radial_bin_width = 0.5
    max_radius = r_map.max()
    n_bins = int(np.ceil(max_radius / radial_bin_width))
    if n_bins <= 0:
        return
    radial_centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * radial_bin_width
    bin_index = np.minimum((r_map.flatten() / radial_bin_width).astype(np.int64), n_bins - 1)

    def profile(image_flat: np.ndarray) -> np.ndarray:
        counts = np.bincount(bin_index, minlength=n_bins)
        sums = np.bincount(bin_index, weights=image_flat, minlength=n_bins)
        return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    obs_prof = profile(observed.flatten())
    pred_prof = profile(predicted.flatten())
    resid_prof = profile((predicted - observed).flatten())

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(radial_centers, obs_prof, label=f"Observed {band}", color="black")
    ax.plot(radial_centers, pred_prof, label=f"Predicted {band}", color="tab:orange")
    ax.plot(radial_centers, resid_prof, label="Residual", color="tab:red", linestyle="--")
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Azimuthal average")
    ax.set_title(f"Peak-centered radial profiles – {band}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / f"radial_profile_{band}.png", dpi=200)
    plt.close(fig)


def _mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if yy.size == 0:
        return None
    y = int(np.round(float(np.mean(yy))))
    x = int(np.round(float(np.mean(xx))))
    return y, x


def _expanded_bbox_mask(mask: np.ndarray, *, pad: int) -> np.ndarray:
    bbox = _mask_bbox(np.asarray(mask, dtype=bool), pad_pixels=int(pad))
    out = np.zeros_like(mask, dtype=bool)
    if bbox is None:
        return out
    y0, y1, x0, x1 = bbox
    out[y0:y1, x0:x1] = True
    return out


def _choose_sky_points(
    observation: np.ndarray, *, forbidden_mask: np.ndarray, n_points: int = 4,
) -> List[Tuple[str, int, int]]:
    image = np.asarray(observation, dtype=float)
    valid = np.isfinite(image) & ~np.asarray(forbidden_mask, dtype=bool)
    if not np.any(valid):
        return []
    flux_limit = float(np.nanpercentile(image[valid], 35.0))
    candidate_mask = valid & (image <= flux_limit)
    if np.count_nonzero(candidate_mask) < n_points:
        candidate_mask = valid

    yy, xx = np.nonzero(candidate_mask)
    if yy.size == 0:
        return []

    anchors = [
        (int(round(0.25 * (image.shape[0] - 1))), int(round(0.25 * (image.shape[1] - 1)))),
        (int(round(0.25 * (image.shape[0] - 1))), int(round(0.75 * (image.shape[1] - 1)))),
        (int(round(0.75 * (image.shape[0] - 1))), int(round(0.25 * (image.shape[1] - 1)))),
        (int(round(0.75 * (image.shape[0] - 1))), int(round(0.75 * (image.shape[1] - 1)))),
    ]
    chosen: List[Tuple[str, int, int]] = []
    used = np.zeros(yy.shape[0], dtype=bool)
    min_sep2 = float(max(10, min(image.shape) // 12) ** 2)

    for idx, (ay, ax) in enumerate(anchors[:n_points], start=1):
        dist2 = (yy - ay) ** 2 + (xx - ax) ** 2
        dist2 = np.where(used, np.inf, dist2.astype(np.float64))
        pick = int(np.argmin(dist2))
        if not np.isfinite(dist2[pick]):
            break
        y = int(yy[pick])
        x = int(xx[pick])
        chosen.append((f"sky_{idx}", y, x))
        used = used | (((yy - y) ** 2 + (xx - x) ** 2) <= min_sep2)
    return chosen


def configured_pixel_sed_points(
    observation: np.ndarray, *, region_masks: Dict[str, np.ndarray] | None = None,
    max_region_points: int | None = 12,
) -> List[Tuple[str, int, int]]:
    ordered_region_names = sorted((region_masks or {}).keys())
    if max_region_points is not None and int(max_region_points) > 0:
        ordered_region_names = ordered_region_names[: int(max_region_points)]
    galaxy_points: List[Tuple[str, int, int]] = []
    forbidden = np.zeros_like(observation, dtype=bool)
    pad = max(6, min(observation.shape) // 18)
    for name in ordered_region_names:
        mask = np.asarray(region_masks[name], dtype=bool)
        centroid = _mask_centroid(mask)
        if centroid is None:
            continue
        galaxy_points.append((str(name), int(centroid[0]), int(centroid[1])))
        forbidden |= _expanded_bbox_mask(mask, pad=pad)

    sky_points = _choose_sky_points(observation, forbidden_mask=forbidden, n_points=4)
    return galaxy_points + sky_points


def _band_centers(bandpasses: Dict[str, Bandpass]) -> Dict[str, float]:
    centers: Dict[str, float] = {}
    for band, bp in bandpasses.items():
        lam = bp.lam
        throughput = np.clip(bp.throughput, 0.0, None)
        norm = np.trapezoid(throughput, lam)
        if norm > 0:
            centers[band] = float(np.trapezoid(lam * throughput, lam) / norm)
        else:
            centers[band] = float(np.mean(lam))
    return centers


def _truth_sed_lookup(
    truth_seds: Sequence[Dict] | None,
) -> tuple[Dict[Tuple[int, int], Dict], Dict[str, Dict]]:
    truth_by_coord: Dict[Tuple[int, int], Dict] = {}
    truth_by_label: Dict[str, Dict] = {}
    if truth_seds:
        for sample in truth_seds:
            coords = sample.get("coords")
            label = sample.get("label")
            if coords and len(coords) == 2:
                truth_by_coord[(int(coords[0]), int(coords[1]))] = sample
            if label is not None:
                truth_by_label[str(label)] = sample
    return truth_by_coord, truth_by_label


def _plot_bandpass_background(
    ax: Axes, *, bands: Sequence[str], bandpasses: Dict[str, Bandpass], max_flux: float,
) -> None:
    for band in bands:
        bp = bandpasses[band]
        if np.max(bp.throughput) <= 0:
            continue
        scale = 0.2 * max_flux
        scaled = scale * (bp.throughput / (np.max(bp.throughput) + 1e-12))
        ax.fill_between(bp.lam, 0, scaled, color="gray", alpha=0.12, linewidth=0)


def plot_point_seds(
    lam_grid: np.ndarray, point_seds: Sequence[Dict], bandpasses: Dict[str, Bandpass], run_dir: Path,
    truth_seds: Sequence[Dict] | None = None, use_flux_density: bool = False,
) -> None:
    if not point_seds:
        return
    bands = list(bandpasses.keys())
    centers = _band_centers(bandpasses)
    n_samples = len(point_seds)
    cols = min(3, n_samples)
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    proj_label_used = False
    truth_label_used = False
    truth_by_coord, truth_by_label = _truth_sed_lookup(truth_seds)
    for idx, sample in enumerate(point_seds):
        ax = axes[idx // cols, idx % cols]
        spectrum = sample["spectrum"]
        coords = sample["coords"]
        y, x = coords
        max_flux = float(np.max(spectrum)) if np.size(spectrum) else 0.0
        max_flux = max(max_flux, 1e-6)
        _plot_bandpass_background(ax, bands=bands, bandpasses=bandpasses, max_flux=max_flux)
        ax.plot(lam_grid, spectrum, color="tab:orange", label="Point SED")
        for band in bands:
            band_fluxes = sample.get("band_fluxes", {})
            if band not in band_fluxes:
                continue
            center = centers[band]
            proj_label = "Projected band flux" if not proj_label_used else None
            ax.scatter(
                center,
                float(band_fluxes[band]),
                color="tab:blue",
                marker="s",
                s=35,
                label=proj_label,
            )
            proj_label_used = proj_label_used or proj_label is not None
        truth_sample = truth_by_coord.get((y, x)) or truth_by_label.get(sample.get("label"))
        if truth_sample is not None:
            lam_truth = np.asarray(truth_sample.get("lam", []), dtype=float)
            if lam_truth.size > 1:
                preferred_key = "flambda_lsf" if use_flux_density else "flambda"
                truth_spec = truth_sample.get(preferred_key)
                if truth_spec is None:
                    truth_spec = truth_sample.get("flambda_lsf")
                if truth_spec is None:
                    truth_spec = truth_sample.get("flambda")
                if truth_spec is not None:
                    truth_spec = np.asarray(truth_spec, dtype=float)
                    truth_interp = np.interp(lam_grid, lam_truth, truth_spec, left=np.nan, right=np.nan)
                    ax.plot(
                        lam_grid, truth_interp, color="gray", alpha=0.6, linewidth=1.0,
                        label="True point SED" if not truth_label_used else None,
                    )
                    truth_label_used = True
        ax.set_title(f"{sample['label']} (y={y}, x={x})")
        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Flux density")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(loc="best")
    for j in range(n_samples, rows * cols):
        ax = axes[j // cols, j % cols]
        ax.axis("off")
    plt.suptitle("Continuous field point SED diagnostics")
    fig.tight_layout()
    fig.savefig(run_dir / "point_seds.png", dpi=200)
    plt.close(fig)


def plot_sed_sample_locations(
    observation: np.ndarray, band: str, run_dir: Path, *, sample_points: Sequence[Tuple[str, int, int]],
    region_masks: Dict[str, np.ndarray] | None = None,
    region_overlays: Dict[str, Dict[str, float | str]] | None = None,
    background_image: np.ndarray | None = None, title: str | None = None,
) -> None:
    image = np.asarray(background_image if background_image is not None else observation, dtype=np.float32)
    display_image, display_extent = _downsample_image_for_display(image)
    footprint = np.isfinite(np.asarray(observation, dtype=float))
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    finite = display_image[np.isfinite(display_image)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanpercentile(finite, 5.0))
        vmax = float(np.nanpercentile(finite, 99.7))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        if finite.size:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
    im = ax.imshow(display_image, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, extent=display_extent)
    if (
        background_image is not None
        and footprint.shape == image.shape
        and min(footprint.shape) >= 2
        and np.any(footprint)
    ):
        ax.contour(footprint.astype(np.float32), levels=[0.5], colors="white", linewidths=0.8, alpha=0.55)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title(title or f"{band}-band SED diagnostic sample locations")

    overlay_map = region_overlays or {}
    region_map = region_masks or {}

    galaxy_names = [label for label, _, _ in sample_points if not str(label).startswith("sky_")]
    galaxy_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(galaxy_names), 1)))
    galaxy_color_map = {str(label): galaxy_colors[idx] for idx, label in enumerate(galaxy_names)}

    galaxy_offset_cycle = [(8.0, 8.0), (8.0, -12.0), (-12.0, 8.0), (-12.0, -12.0)]
    sky_offset_cycle = [(-12.0, 10.0), (10.0, 10.0), (-12.0, -10.0), (10.0, -10.0)]
    galaxy_seen = 0
    sky_seen = 0

    for name, mask in region_map.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != image.shape or not np.any(mask_arr):
            continue
        color = galaxy_color_map.get(str(name), (1.0, 1.0, 1.0, 1.0))
        overlay = dict(overlay_map.get(str(name), {}) or {})
        if (overlay.get("kind") == "circle" and {"center_x", "center_y", "radius_pix"} <= set(overlay)):
            circle = Circle(
                (float(overlay["center_x"]), float(overlay["center_y"])), radius=float(overlay["radius_pix"]),
                facecolor="none", edgecolor=color, linewidth=1.3, alpha=0.95,
            )
            ax.add_patch(circle)
        else:
            ax.contour(mask_arr.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.1)

    for label, y, x in sample_points:
        label_str = str(label)
        is_sky = label_str.startswith("sky_")
        color = (0.15, 0.9, 0.9, 1.0) if is_sky else galaxy_color_map.get(label_str, (1.0, 0.8, 0.2, 1.0))
        marker = "o"
        ax.scatter(
            [x], [y], s=140 if is_sky else 180, c=[color], marker=marker, edgecolors="black", linewidths=1.0,
            zorder=4,
        )
        if is_sky:
            dx, dy = sky_offset_cycle[sky_seen % len(sky_offset_cycle)]
            sky_seen += 1
        else:
            dx, dy = galaxy_offset_cycle[galaxy_seen % len(galaxy_offset_cycle)]
            galaxy_seen += 1
        if x + dx < 0:
            dx = abs(dx)
        if x + dx > image.shape[1]:
            dx = -abs(dx)
        if y + dy < 0:
            dy = abs(dy)
        if y + dy > image.shape[0]:
            dy = -abs(dy)
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(
            x + dx,
            y + dy,
            f"{label_str}\n(y={y}, x={x})",
            color=color,
            fontsize=11,
            fontweight="bold",
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": (0.05, 0.08, 0.12, 0.45), "edgecolor": "none"},
            zorder=5,
        )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(run_dir / f"sed_sample_locations_{band}.png", dpi=200)
    plt.close(fig)


def _downsample_image_for_display(
    image: np.ndarray, *, max_dim: int = 2048,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        return arr, None
    height, width = arr.shape
    if height <= 0 or width <= 0 or max(height, width) <= int(max_dim):
        return arr, None
    factor = int(np.ceil(max(height, width) / float(max_dim)))
    factor = max(1, factor)
    trimmed_h = (height // factor) * factor
    trimmed_w = (width // factor) * factor
    if trimmed_h <= 0 or trimmed_w <= 0:
        return arr, None
    block = arr[:trimmed_h, :trimmed_w].reshape(trimmed_h // factor, factor, trimmed_w // factor, factor)
    finite = np.isfinite(block)
    sums = np.nansum(block, axis=(1, 3), dtype=np.float64)
    counts = np.count_nonzero(finite, axis=(1, 3))
    out = np.full(sums.shape, np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    extent = (-0.5, float(width) - 0.5, -0.5, float(height) - 0.5)
    return out, extent




def plot_fiber_spectrum(
    lam_spec: np.ndarray, predicted: np.ndarray, truth: np.ndarray, throughput: np.ndarray | None,
    run_dir: Path, *, filename: str = "fiber_spectrum.png", title: str = "Fiber spectrum",
) -> None:
    lam_plot = lam_spec
    pred_plot = predicted
    truth_plot = truth
    # Mask regions with negligible throughput so plots reflect the instrument support.
    if throughput is not None and throughput.size == lam_spec.size and np.any(throughput > 0):
        thresh = 1.0e-3 * np.max(throughput)
        mask = throughput > thresh
        if np.any(mask):
            lam_plot = lam_spec[mask]
            pred_plot = predicted[mask]
            truth_plot = truth[mask]

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    ax1.plot(lam_plot, truth_plot, color="black", label="True fiber SED")
    ax1.plot(lam_plot, pred_plot, color="tab:orange", label="Model fiber SED")
    ax1.set_xlabel("Wavelength [Å]")
    ax1.set_ylabel("Flux (fiber-integrated)")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    if throughput is not None and np.any(throughput > 0):
        ax2 = ax1.twinx()
        norm = np.max(throughput)
        ax2.plot(
            lam_spec, throughput / (norm + 1e-12), color="gray", linestyle="--", alpha=0.5, label="Throughput"
        )
        ax2.set_ylabel("Normalized throughput")
        ax2.set_ylim(0, 1.05)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200)
    plt.close(fig)


def plot_fiber_spectra_overlay(
    spectra: Sequence[Dict[str, object]], run_dir: Path, *, filename: str = "fiber_spectra_overlay.png",
    title: str = "Science fiber spectra", max_spectra: int | None = 12,
) -> None:
    if not spectra:
        return
    if max_spectra is not None and int(max_spectra) > 0:
        spectra_to_plot = list(spectra[: int(max_spectra)])
    else:
        spectra_to_plot = list(spectra)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for item in spectra_to_plot:
        lam = np.asarray(item.get("lam"), dtype=float)
        pred = np.asarray(item.get("predicted"), dtype=float)
        truth = np.asarray(item.get("truth"), dtype=float)
        label = str(item.get("label", "fiber"))
        ax.plot(lam, truth, linewidth=1.2, alpha=0.55, label=f"{label} true")
        ax.plot(lam, pred, linewidth=1.2, linestyle="--", alpha=0.85, label=f"{label} pred")
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Flux (fiber-integrated)")
    if len(spectra_to_plot) < len(spectra):
        ax.set_title(f"{title} (first {len(spectra_to_plot)} of {len(spectra)})")
    else:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200)
    plt.close(fig)
