from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass(slots=True)
class BandMetrics:
    mse: float
    mae: float
    mean_relative_error: float
    mae_signal: float
    relative_mae_signal_pct: float
    rmse: float
    nrmse_pct: float
    peak_error_pct: float
    flux_error_pct: float
    fiber_flux_error_pct: float
    bgsub_flux_error_pct: float
    total_abs_flux_error_pct: float
    max_abs_residual: float
    max_abs_residual_pct: float
    signal_fraction_pct: float
    signal_pixels: int
    total_pixels: int
    observed_min: float
    observed_max: float
    predicted_min: float
    predicted_max: float
    status: str


def _flux_error(pred: np.ndarray, obs: np.ndarray) -> float:
    total_flux_pred = float(pred.sum())
    total_flux_obs = float(obs.sum())
    flux_error_pct = 100.0 * (total_flux_pred - total_flux_obs) / (total_flux_obs + 1e-12)
    return flux_error_pct


def _bgsub_positive_flux(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    bg = float(np.median(image[np.isfinite(image)])) if np.any(np.isfinite(image)) else 0.0
    data = image - bg
    data = np.where(np.isfinite(data), data, 0.0)
    data = np.maximum(data, 0.0)
    return float(data.sum())


def _flux_error_from_sums(sum_pred: float, sum_obs: float) -> float:
    return float(100.0 * (sum_pred - sum_obs) / (sum_obs + 1e-12))


def _metric_key(exposure_id: int, band: str) -> str:
    return f"e{int(exposure_id)}:{band}"


def _signal_mask(
    obs: np.ndarray, signal_threshold: float, *, background_subtracted_signal: bool,
) -> tuple[np.ndarray, float]:
    """Signal-pixel mask and the baseline the mask subtracted (0 when none).

    Callers must subtract the returned baseline from any denominator built
    from the masked observations; dividing a background-subtracted residual
    by background-bearing totals understates relative errors."""
    obs_arr = np.asarray(obs, dtype=np.float64)
    finite = np.isfinite(obs_arr)
    if not np.any(finite):
        return np.zeros_like(obs_arr, dtype=bool), 0.0
    if background_subtracted_signal:
        baseline = float(np.median(obs_arr[finite]))
        signal = obs_arr - baseline
        peak_val = float(np.max(signal[finite]))
        if not np.isfinite(peak_val) or peak_val <= 0.0:
            return np.zeros_like(obs_arr, dtype=bool), baseline
        return finite & (signal >= (float(signal_threshold) * peak_val)), baseline
    peak_val = float(np.max(obs_arr[finite]))
    if not np.isfinite(peak_val):
        return np.zeros_like(obs_arr, dtype=bool), 0.0
    return finite & (obs_arr >= (float(signal_threshold) * peak_val)), 0.0


def _status_for_band_metric(*, band: str, held_out_band: str | None, partial_holdout_enabled: bool) -> str:
    if held_out_band is None or band != held_out_band:
        return "training"
    if partial_holdout_enabled:
        return "PARTIAL-HOLDOUT-TRAINING-BAND"
    return "HELD-OUT"


def compute_metrics(
    predictions: Dict[int, Dict[str, np.ndarray]], observations: Dict[int, Dict[str, np.ndarray]], *,
    held_out_band: str | None = None, signal_threshold: float = 0.0,
    fiber_mask: np.ndarray | Dict[int, np.ndarray] | None = None, partial_holdout_enabled: bool = False,
    background_subtracted_signal: bool = False,
) -> Dict[str, BandMetrics]:
    results: Dict[str, BandMetrics] = {}
    for exp_id, pred_by_band in predictions.items():
        obs_by_band = observations.get(exp_id)
        if obs_by_band is None:
            continue
        if isinstance(fiber_mask, dict):
            fiber_mask_exp = fiber_mask.get(exp_id)
        else:
            fiber_mask_exp = fiber_mask
        fiber_mask_bool = None
        if fiber_mask_exp is not None:
            fiber_mask_arr = np.asarray(fiber_mask_exp)
            if fiber_mask_arr.ndim != 2:
                raise ValueError("fiber_mask must be a 2D array.")
            fiber_mask_bool = fiber_mask_arr.astype(bool)
        for band, pred in pred_by_band.items():
            obs = obs_by_band.get(band)
            if obs is None:
                continue
            obs = np.asarray(obs, dtype=np.float64)
            pred = np.asarray(pred, dtype=np.float64)
            if pred.shape != obs.shape:
                raise ValueError(
                    f"prediction shape {pred.shape} does not match observation shape {obs.shape}."
                )
            if fiber_mask_bool is not None and fiber_mask_bool.shape != obs.shape:
                raise ValueError(
                    f"fiber_mask shape {fiber_mask_bool.shape} does not match observation shape {obs.shape}."
                )
            valid = np.isfinite(obs) & np.isfinite(pred)
            if not np.any(valid):
                continue
            obs_eval = obs[valid]
            pred_eval = pred[valid]
            diff = pred_eval - obs_eval
            mse = float(np.mean(diff**2))
            mae = float(np.mean(np.abs(diff)))
            rel_error = float(np.mean(np.abs(diff) / (np.abs(obs_eval) + 1e-10)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            signal_range = float(obs_eval.max() - obs_eval.min())
            nrmse_pct = float(100.0 * rmse / (signal_range + 1e-12))

            signal_mask, signal_baseline = _signal_mask(
                obs_eval, signal_threshold, background_subtracted_signal=background_subtracted_signal,
            )
            n_signal = int(signal_mask.sum())
            if n_signal > 0:
                mae_signal = float(np.mean(np.abs(diff[signal_mask])))
                mean_signal = float(np.mean(obs_eval[signal_mask])) - signal_baseline
                relative_mae_signal_pct = float(100.0 * mae_signal / (mean_signal + 1e-12))
            else:
                mae_signal = float("nan")
                relative_mae_signal_pct = float("nan")
            signal_fraction_pct = float(100.0 * n_signal / obs_eval.size)

            peak_error = float(100.0 * np.abs(pred_eval.max() - obs_eval.max()) / (obs_eval.max() + 1e-12))
            max_abs_residual = float(np.abs(diff).max())
            max_abs_residual_pct = float(100.0 * max_abs_residual / (obs_eval.max() + 1e-12))
            flux_error_pct = _flux_error(pred_eval, obs_eval)
            if fiber_mask_bool is not None and fiber_mask_bool.any():
                fiber_valid = fiber_mask_bool & valid
                fiber_flux_error_pct = (
                    _flux_error(pred[fiber_valid], obs[fiber_valid])
                    if np.any(fiber_valid)
                    else float("nan")
                )
            else:
                fiber_flux_error_pct = float("nan")
            bgsub_flux_error_pct = _flux_error_from_sums(
                _bgsub_positive_flux(pred_eval), _bgsub_positive_flux(obs_eval),
            )
            total_abs_flux_error = float(np.sum(np.abs(diff)))
            total_flux_obs = float(np.sum(np.abs(obs_eval)))
            total_abs_flux_error_pct = float(100.0 * total_abs_flux_error / (total_flux_obs + 1e-12))

            key = _metric_key(exp_id, band)
            results[key] = BandMetrics(
                mse=mse,
                mae=mae,
                mean_relative_error=rel_error,
                mae_signal=mae_signal,
                relative_mae_signal_pct=relative_mae_signal_pct,
                rmse=rmse,
                nrmse_pct=nrmse_pct,
                peak_error_pct=peak_error,
                flux_error_pct=flux_error_pct,
                fiber_flux_error_pct=fiber_flux_error_pct,
                bgsub_flux_error_pct=bgsub_flux_error_pct,
                total_abs_flux_error_pct=total_abs_flux_error_pct,
                max_abs_residual=max_abs_residual,
                max_abs_residual_pct=max_abs_residual_pct,
                signal_fraction_pct=signal_fraction_pct,
                signal_pixels=n_signal,
                total_pixels=int(obs_eval.size),
                observed_min=float(obs_eval.min()),
                observed_max=float(obs_eval.max()),
                predicted_min=float(pred_eval.min()),
                predicted_max=float(pred_eval.max()),
                status=_status_for_band_metric(
                    band=band, held_out_band=held_out_band, partial_holdout_enabled=partial_holdout_enabled,
                ),
            )
    return results


def metrics_to_serializable(metrics: Dict[str, BandMetrics]) -> Dict[str, dict]:
    serializable: Dict[str, dict] = {}
    for key, value in metrics.items():
        item = asdict(value)
        for metric_name, metric_value in item.items():
            if isinstance(metric_value, float) and not np.isfinite(metric_value):
                item[metric_name] = None
        serializable[key] = item
    return serializable


def _sort_metric_keys(keys: list[str]) -> list[str]:
    band_order = ["u", "g", "r", "i", "z"]

    def key_fn(label: str) -> tuple[int, int]:
        exp_part, _, band = label.partition(":")
        exp_id = int(exp_part.lstrip("e")) if exp_part.startswith("e") else 0
        band_rank = band_order.index(band) if band in band_order else len(band_order)
        return exp_id, band_rank

    return sorted(keys, key=key_fn)


def log_metrics(metrics: Dict[str, BandMetrics], held_out_band: str | None = None) -> None:
    if not metrics:
        return
    print("\n" + "=" * 70)
    print("QUANTITATIVE ERROR ANALYSIS")
    print("=" * 70)
    sorted_keys = _sort_metric_keys(list(metrics.keys()))
    for key in sorted_keys:
        m = metrics[key]
        exp_part, _, band = key.partition(":")
        print(f"\nExposure {exp_part} Band {band} ({m.status}):")
        print(f"  Observed range: [{m.observed_min:.1f}, {m.observed_max:.1f}]")
        print(f"  Predicted range: [{m.predicted_min:.1f}, {m.predicted_max:.1f}]")
        print(
            f"  Max absolute residual: {m.max_abs_residual:.3f} ({m.max_abs_residual_pct:.1f}% of peak)"
        )
        print(f"  MAE (all pixels): {m.mae:.3f}")
        print(
            f"  MAE (signal only): {m.mae_signal:.3f} ({m.relative_mae_signal_pct:.1f}% of mean signal)"
        )
        print(f"  RMSE: {m.rmse:.3f}")
        print(f"  Normalized RMSE: {m.nrmse_pct:.1f}%")
        print(f"  Peak error: {m.peak_error_pct:.1f}%")
        print(
            f"  Signal pixels: {m.signal_pixels}/{m.total_pixels} ({m.signal_fraction_pct:.1f}%)"
        )
        extra = f", bgsub+: {m.bgsub_flux_error_pct:5.1f}%"
        if np.isfinite(m.fiber_flux_error_pct):
            extra = f", fiber: {m.fiber_flux_error_pct:5.1f}%" + extra
        print(
            f"  Flux error: {m.flux_error_pct:5.1f}%  |Δ|flux: {m.total_abs_flux_error_pct:5.1f}%{extra}"
        )
        if m.status != "training":
            print(f"  *** {m.status} ***")

    print("\n" + "-" * 40)
    print("FLUX-WEIGHTED ERRORS:")
    print("-" * 40)
    for key in sorted_keys:
        m = metrics[key]
        exp_part, _, band = key.partition(":")
        extra = f", bgsub+: {m.bgsub_flux_error_pct:5.1f}%"
        if np.isfinite(m.fiber_flux_error_pct):
            extra = f", fiber: {m.fiber_flux_error_pct:5.1f}%" + extra
        print(
            f"  {exp_part}:{band} ({m.status}): Flux error = {m.flux_error_pct:5.1f}%, "
            f"|Δ|flux = {m.total_abs_flux_error_pct:5.1f}%{extra}"
        )


def _region_metric_key(region_name: str, exposure_id: int, band: str) -> str:
    return f"region={region_name}|e{int(exposure_id)}:{band}"


def _parse_region_metric_key(label: str) -> tuple[str, int, str]:
    region_part, sep, remainder = label.partition("|")
    if sep == "":
        raise ValueError(f"Invalid region metric key: {label!r}")
    if not region_part.startswith("region="):
        raise ValueError(f"Invalid region metric key: {label!r}")
    region_name = region_part.split("=", 1)[1]
    exp_part, sep2, band = remainder.partition(":")
    if sep2 == "":
        raise ValueError(f"Invalid region metric key: {label!r}")
    exp_id = int(exp_part.lstrip("e")) if exp_part.startswith("e") else int(exp_part)
    return region_name, exp_id, band


def _status_for_region_metric(
    *, region_name: str, region_mask: np.ndarray, band: str, held_out_band: str | None,
    primary_region: str | None, partial_holdout_enabled: bool = False, holdout_mask: np.ndarray | None = None,
) -> str:
    is_primary = primary_region is not None and region_name == primary_region
    is_held_out = held_out_band is not None and band == held_out_band
    if not is_held_out:
        if is_primary:
            return "PRIMARY"
        return "training"
    if partial_holdout_enabled:
        overlap_pixels = 0
        region_pixels = int(np.count_nonzero(region_mask))
        if (holdout_mask is not None and holdout_mask.shape == region_mask.shape and region_pixels > 0):
            overlap_pixels = int(np.count_nonzero(region_mask & holdout_mask))
        if overlap_pixels <= 0:
            base = "PARTIAL-HOLDOUT-TRAINING-BAND"
        elif overlap_pixels >= region_pixels:
            base = "HOLDOUT-REGION"
        else:
            base = "MIXED-HOLDOUT-REGION"
        if is_primary:
            return f"PRIMARY_{base}"
        return base
    if is_primary and is_held_out:
        return "PRIMARY_HELD-OUT"
    if is_primary:
        return "PRIMARY"
    if is_held_out:
        return "HELD-OUT"
    return "training"


def compute_region_metrics(
    predictions: Dict[int, Dict[str, np.ndarray]], observations: Dict[int, Dict[str, np.ndarray]],
    region_masks_by_exposure: Dict[int, Dict[str, np.ndarray]], *, held_out_band: str | None = None,
    primary_region: str | None = None, signal_threshold: float = 0.0, partial_holdout_enabled: bool = False,
    holdout_mask_by_exposure: Dict[int, np.ndarray] | None = None, background_subtracted_signal: bool = False,
) -> Dict[str, BandMetrics]:
    results: Dict[str, BandMetrics] = {}
    for exp_id, pred_by_band in predictions.items():
        obs_by_band = observations.get(exp_id)
        if obs_by_band is None:
            continue
        region_masks = region_masks_by_exposure.get(int(exp_id), {})
        holdout_mask = None
        if isinstance(holdout_mask_by_exposure, dict):
            holdout_mask = holdout_mask_by_exposure.get(int(exp_id))
        for region_name, region_mask in region_masks.items():
            mask = np.asarray(region_mask).astype(bool)
            if mask.ndim != 2:
                raise ValueError("Region masks must be 2D arrays.")
            if not np.any(mask):
                continue
            for band, pred in pred_by_band.items():
                obs = obs_by_band.get(band)
                if obs is None:
                    continue
                obs = np.asarray(obs, dtype=np.float64)
                pred = np.asarray(pred, dtype=np.float64)
                if pred.shape != obs.shape:
                    raise ValueError(
                        f"prediction shape {pred.shape} does not match observation shape {obs.shape}."
                    )
                if mask.shape != obs.shape:
                    raise ValueError(
                        f"region mask shape {mask.shape} does not match observation shape {obs.shape}."
                    )
                valid = mask & np.isfinite(obs) & np.isfinite(pred)
                obs_sel = obs[valid]
                pred_sel = pred[valid]
                if obs_sel.size == 0:
                    continue
                diff = pred_sel - obs_sel
                mse = float(np.mean(diff**2))
                mae = float(np.mean(np.abs(diff)))
                rel_error = float(np.mean(np.abs(diff) / (np.abs(obs_sel) + 1e-10)))
                rmse = float(np.sqrt(np.mean(diff**2)))
                signal_range = float(obs_sel.max() - obs_sel.min())
                nrmse_pct = float(100.0 * rmse / (signal_range + 1e-12))
                signal_mask, signal_baseline = _signal_mask(
                    obs_sel, signal_threshold, background_subtracted_signal=background_subtracted_signal,
                )
                n_signal = int(signal_mask.sum())
                if n_signal > 0:
                    mae_signal = float(np.mean(np.abs(diff[signal_mask])))
                    mean_signal = float(np.mean(obs_sel[signal_mask])) - signal_baseline
                    relative_mae_signal_pct = float(100.0 * mae_signal / (mean_signal + 1e-12))
                else:
                    mae_signal = float("nan")
                    relative_mae_signal_pct = float("nan")
                signal_fraction_pct = float(100.0 * n_signal / obs_sel.size)
                peak_error = float(100.0 * np.abs(pred_sel.max() - obs_sel.max()) / (obs_sel.max() + 1e-12))
                max_abs_residual = float(np.abs(diff).max())
                max_abs_residual_pct = float(100.0 * max_abs_residual / (obs_sel.max() + 1e-12))
                flux_error_pct = _flux_error(pred_sel, obs_sel)
                bgsub_flux_error_pct = _flux_error_from_sums(
                    _bgsub_positive_flux(pred_sel), _bgsub_positive_flux(obs_sel),
                )
                total_abs_flux_error = float(np.sum(np.abs(diff)))
                total_flux_obs = float(np.sum(np.abs(obs_sel)))
                total_abs_flux_error_pct = float(100.0 * total_abs_flux_error / (total_flux_obs + 1e-12))
                status = _status_for_region_metric(
                    region_name=region_name, region_mask=mask, band=band, held_out_band=held_out_band,
                    primary_region=primary_region, partial_holdout_enabled=partial_holdout_enabled,
                    holdout_mask=holdout_mask,
                )
                key = _region_metric_key(region_name, int(exp_id), band)
                results[key] = BandMetrics(
                    mse=mse, mae=mae, mean_relative_error=rel_error, mae_signal=mae_signal,
                    relative_mae_signal_pct=relative_mae_signal_pct, rmse=rmse, nrmse_pct=nrmse_pct,
                    peak_error_pct=peak_error, flux_error_pct=flux_error_pct,
                    fiber_flux_error_pct=float("nan"), bgsub_flux_error_pct=bgsub_flux_error_pct,
                    total_abs_flux_error_pct=total_abs_flux_error_pct, max_abs_residual=max_abs_residual,
                    max_abs_residual_pct=max_abs_residual_pct, signal_fraction_pct=signal_fraction_pct,
                    signal_pixels=n_signal, total_pixels=int(obs_sel.size), observed_min=float(obs_sel.min()),
                    observed_max=float(obs_sel.max()), predicted_min=float(pred_sel.min()),
                    predicted_max=float(pred_sel.max()), status=status,
                )
    return results


def region_metrics_to_serializable(metrics: Dict[str, BandMetrics]) -> Dict[str, dict]:
    return metrics_to_serializable(metrics)


def _sort_region_metric_keys(
    keys: list[str], *, primary_region: str | None, held_out_band: str | None,
) -> list[str]:
    band_order = ["u", "g", "r", "i", "z"]

    def key_fn(label: str) -> tuple[int, int, str, int, int]:
        region_name, exp_id, band = _parse_region_metric_key(label)
        is_primary = int(primary_region is not None and region_name == primary_region)
        is_holdout = int(held_out_band is not None and band == held_out_band)
        band_rank = band_order.index(band) if band in band_order else len(band_order)
        return (-is_primary, -is_holdout, region_name, exp_id, band_rank)

    return sorted(keys, key=key_fn)


def log_region_metrics(
    metrics: Dict[str, BandMetrics], *, held_out_band: str | None = None, primary_region: str | None = None,
) -> None:
    if not metrics:
        return
    print("\n" + "=" * 70)
    print("REGION-LEVEL ERROR ANALYSIS")
    print("=" * 70)
    sorted_keys = _sort_region_metric_keys(
        list(metrics.keys()), primary_region=primary_region, held_out_band=held_out_band,
    )
    for key in sorted_keys:
        m = metrics[key]
        region_name, exp_id, band = _parse_region_metric_key(key)
        primary_suffix = " [PRIMARY]" if primary_region == region_name else ""
        print(f"\nRegion {region_name}{primary_suffix} Exposure e{exp_id} Band {band} ({m.status}):")
        print(f"  Observed range: [{m.observed_min:.3g}, {m.observed_max:.3g}]")
        print(f"  Predicted range: [{m.predicted_min:.3g}, {m.predicted_max:.3g}]")
        print(
            f"  Max absolute residual: {m.max_abs_residual:.3g} "
            f"({m.max_abs_residual_pct:.2f}% of peak)"
        )
        print(f"  MAE (all pixels): {m.mae:.3g}")
        print(
            f"  MAE (signal only): {m.mae_signal:.3g} "
            f"({m.relative_mae_signal_pct:.2f}% of mean signal)"
        )
        print(f"  RMSE: {m.rmse:.3g}")
        print(f"  Normalized RMSE: {m.nrmse_pct:.2f}%")
        print(f"  Peak error: {m.peak_error_pct:.2f}%")
        print(
            f"  Signal pixels: {m.signal_pixels}/{m.total_pixels} "
            f"({m.signal_fraction_pct:.2f}%)"
        )
        print(
            f"  Flux error: {m.flux_error_pct:6.2f}%  |Δ|flux: {m.total_abs_flux_error_pct:6.2f}%"
            f", bgsub+: {m.bgsub_flux_error_pct:6.2f}%"
        )
