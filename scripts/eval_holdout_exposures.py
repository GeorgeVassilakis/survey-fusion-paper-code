"""Novel-view evaluation of held-out image exposures.

Loads a checkpoint trained with trainer.exclude_exposure_labels, renders each
held-out exposure through the deterministic render path, and scores it against
the observed frame. Renders matched same-band trained exposures (nearest PSF
width) as controls, so held-out quality is read against the model's in-sample
rendering quality rather than an absolute scale.

Every frame is also scored stratified by training coverage: the pixels of a
held-out frame that no same-band training exposure ever saw ("hole") are
reported apart from the pixels other same-band frames cover, because a
whole-frame ratio averages the two and can hide (or cancel) a coverage-local
failure.

Usage:
  python scripts/eval_holdout_exposures.py --run-dir <dir with config.yaml+model.pt> \
      --outdir <outdir> [--prepared-dir ...] [--labels a,b,c] \
      [--controls-per-band 2] [--save-arrays] [--coverage-snr 3.0] \
      [--skip-coverage-strata]
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from collections.abc import Mapping
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
UNRECORDED_OPERATOR = "unrecorded_by_checkpoint_format"

# The prepared cache memmaps thousands of files; a default 1024-fd soft limit
# kills the load. Raise it here so the script does not
# depend on every caller remembering ulimit -n.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < 65536:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))


def load_trainer(
    run_dir: Path, output_dir: Path, prepared_dir: Path | None, device_name: str,
    *, allow_operator_reinterpretation: bool = False,
):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import train as train_entry
    from survey_fusion.utils.torch_utils import apply_precision_policy, set_seed

    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "model.pt"
    cfg = OmegaConf.load(config_path)
    cfg.paths.outdir = str(output_dir)
    cfg.paths.cache_dir = str(output_dir / "cache")
    cfg.checkpoint.path = str(checkpoint_path)
    cfg.resume.path = None
    cfg.platform.distributed.enabled = False
    cfg.platform.distributed.world_size = 1
    if prepared_dir is not None:
        cfg.data.prepared_dir = str(prepared_dir)
    device = torch.device(device_name)
    precision_policy = apply_precision_policy(
        str(cfg.platform.get("precision_policy", "tf32")), device
    )
    set_seed(int(cfg.run.get("seed", 42)))
    paths = train_entry._resolve_paths(cfg)
    checkpoint = train_entry._load_checkpoint(checkpoint_path, device)
    model_state = checkpoint.get("model_state", {})
    checkpoint_mu = model_state.get("mu") if isinstance(model_state, dict) else None
    if checkpoint_mu is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no model_state.mu tensor.")
    cfg.model.n_splats = int(checkpoint_mu.shape[0])
    _, _, _, data_store, trainer, _, _ = train_entry._build_trainer_components(
        cfg, paths=paths, device=device, precision_policy=precision_policy,
    )
    train_entry.validate_checkpoint_model_config(
        trainer.model, checkpoint, checkpoint_path=checkpoint_path,
        context="Holdout exposure evaluation",
    )
    current_operator_config = train_entry.operator_config_for_trainer(trainer)
    reinterpreted = train_entry.validate_checkpoint_operator_config(
        checkpoint,
        current_operator_config=current_operator_config,
        context="Holdout exposure evaluation",
        checkpoint_path=checkpoint_path,
        allow_operator_reinterpretation=allow_operator_reinterpretation,
    )
    trainer.model.load_state_dict(checkpoint["model_state"])
    trainer.load_calibration_state_dict(checkpoint.get("calibration_state"))
    return (
        cfg,
        data_store,
        trainer,
        train_entry,
        checkpoint,
        current_operator_config,
        reinterpreted,
    )


def evaluator_source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty source commit.")
    return commit


def operator_identity_for_evaluation(cfg, trainer, store, exposure_ids: list[int]) -> dict[str, Any]:
    from survey_fusion.utils.perf_env import exact_pair_pruning_enabled

    physics_cfg = OmegaConf.to_container(cfg.physics, resolve=True)
    if not isinstance(physics_cfg, dict):
        raise ValueError("Resolved physics configuration must be a mapping.")
    bandpasses = physics_cfg.get("bandpasses")
    if not isinstance(bandpasses, dict):
        raise ValueError("Resolved physics.bandpasses must be a mapping.")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(data_cfg, dict):
        raise ValueError("Resolved data configuration must be a mapping.")
    # Labels are not part of the prepared store: exposure id equals the
    # entry's position in the manifest's image_exposures list (same contract
    # as train._resolve_excluded_image_exposure_ids), so resolve them by
    # re-reading the manifest and cross-check the band map.
    manifest_value = data_cfg.get("scene_manifest_path")
    if not manifest_value:
        raise ValueError(
            "data.scene_manifest_path is required to resolve exposure labels."
        )
    with open(manifest_value, "r", encoding="utf-8") as fh:
        manifest_exposures = json.load(fh)["image_exposures"]
    psf_parameters = []
    for exp_id in sorted(set(exposure_ids)):
        exposure = store.exposures[exp_id]
        if not 0 <= exp_id < len(manifest_exposures):
            raise ValueError(
                f"Exposure id {exp_id} outside manifest range "
                f"0..{len(manifest_exposures) - 1}."
            )
        entry = manifest_exposures[exp_id]
        store_band = str(store.image_band_by_exposure[exp_id])
        if str(entry["band"]) != store_band:
            raise ValueError(
                f"Manifest/store band mismatch at exposure {exp_id}: "
                f"manifest {entry['band']!r} vs store {store_band!r}."
            )
        psf_parameters.append({
            "exposure_id": int(exp_id),
            "label": str(entry["label"]),
            "psf_fwhm_arcsec": float(exposure.psf_fwhm_arcsec),
            "pixel_scale_arcsec": float(exposure.pixel_scale_arcsec),
        })
    return {
        "psf_model": "gaussian",
        "psf_parameters_by_exposure": psf_parameters,
        "apply_spatial_psf": bool(trainer.apply_spatial_psf),
        "analytic_images": bool(trainer.analytic_images_enabled),
        "analytic_image_quadrature_order": int(
            trainer.analytic_image_quadrature_order
        ),
        "pair_n_sigma": float(trainer.pair_n_sigma),
        "exact_pair_prune": bool(exact_pair_pruning_enabled()),
        "bandpasses": dict(bandpasses),
        "scene_manifest_path": data_cfg.get("scene_manifest_path"),
        "pointing_offsets_arcsec": data_cfg.get("pointing_offsets_arcsec"),
        "training_flux_scale": float(data_cfg.get("training_flux_scale", 1.0)),
    }


def labels_by_exposure_id(
    data_cfg: Mapping[str, Any], exposure_ids: list[int], store,
) -> dict[int, str]:
    """Manifest label for each exposure id, cross-checked against the store band.

    Exposure id is the index into the manifest's image_exposures list, the same
    contract train._resolve_excluded_image_exposure_ids relies on.
    """
    manifest_value = data_cfg.get("scene_manifest_path")
    if not manifest_value:
        raise ValueError(
            "data.scene_manifest_path is required to resolve exposure labels."
        )
    with open(manifest_value, "r", encoding="utf-8") as fh:
        entries = json.load(fh)["image_exposures"]
    resolved: dict[int, str] = {}
    for exp_id in sorted({int(e) for e in exposure_ids}):
        if not 0 <= exp_id < len(entries):
            raise ValueError(
                f"Exposure id {exp_id} outside manifest range "
                f"0..{len(entries) - 1}."
            )
        entry = entries[exp_id]
        store_band = str(store.image_band_by_exposure[exp_id])
        if str(entry["band"]) != store_band:
            raise ValueError(
                f"Manifest/store band mismatch at exposure {exp_id}: "
                f"manifest {entry['band']!r} vs store {store_band!r}."
            )
        resolved[exp_id] = str(entry["label"])
    return resolved


def assemble_verdict(
    *, labels: list[str], evaluating: Mapping[str, Any], checkpoint: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Report skeleton.

    "labels" is the held-out set in CONFIG order, while "holdout" rows are
    emitted in sorted-exposure-id order, so the two are not positionally
    paired: zipping them silently misattributes results whenever a band holds
    more than one frame. Every row therefore carries its own "label" and
    "exposure_id"; consumers must key on those.
    """
    operator_config = checkpoint.get("operator_config")
    trained: dict[str, Any] | str = UNRECORDED_OPERATOR
    if isinstance(operator_config, Mapping):
        recorded = {
            str(key): value for key, value in operator_config.items()
            if key != "exclude_exposure_labels"
        }
        if recorded:
            trained = recorded
    return {
        "holdout": [],
        "controls": [],
        "labels": labels,
        "operator_provenance": {
            "evaluating": dict(evaluating),
            "trained": trained,
            "evaluator_source_commit": source_commit,
        },
    }


def sigma_map_for_exposure(trainer, store, exp_id: int) -> np.ndarray:
    """Reconstruct the full-frame per-pixel sigma map (NaN where invalid)."""
    from survey_fusion.data.adapters.sdss import DatumKind

    exposure = store.exposures[exp_id]
    height, width = int(exposure.image_height), int(exposure.image_width)
    idx = trainer._global_indices_for(exp_id, DatumKind.IMAGE_PIXEL)
    local = torch.index_select(store.batch.datum_index.cpu(), 0, idx.cpu()).numpy()
    sigma = torch.index_select(store.batch.sigma.cpu(), 0, idx.cpu()).numpy()
    out = np.full(height * width, np.nan, dtype=np.float64)
    out[local] = sigma
    return out.reshape(height, width)


def score_exposure(trainer, store, exp_id: int, chunk_size: int) -> dict[str, Any]:
    exposure = store.exposures[exp_id]
    height, width = int(exposure.image_height), int(exposure.image_width)
    with torch.no_grad():
        model_source = np.asarray(
            trainer.render_datum_image_window_deterministic(
                exp_id, y0=0, y1=height, x0=0, x1=width,
                supersample=1, chunk_size=chunk_size, include_additive=False,
            ),
            dtype=np.float64,
        )
    obs_source = np.asarray(store.image_source_by_exposure[exp_id], dtype=np.float64)
    sigma = sigma_map_for_exposure(trainer, store, exp_id)
    valid = np.isfinite(sigma) & np.isfinite(obs_source) & np.isfinite(model_source)
    if not np.any(valid):
        raise RuntimeError(f"Exposure {exp_id} has no valid pixels to score.")
    if np.any(sigma[valid] <= 0.0):
        raise ValueError(f"Exposure {exp_id} has non-positive sigma pixels to score.")
    pull = (model_source[valid] - obs_source[valid]) / sigma[valid]
    model_sum = float(model_source[valid].sum())
    obs_sum = float(obs_source[valid].sum())
    row = {
        "exposure_id": int(exp_id),
        "band": str(store.image_band_by_exposure[exp_id]),
        "psf_fwhm_arcsec": (
            float(exposure.psf_fwhm_arcsec)
            if exposure.psf_fwhm_arcsec is not None else None
        ),
        "n_valid_pixels": int(valid.sum()),
        "mean_square_pull": float(np.mean(pull**2)),
        "median_abs_pull": float(np.median(np.abs(pull))),
        "p95_abs_pull": float(np.percentile(np.abs(pull), 95.0)),
        "model_total_flux": model_sum,
        "obs_total_flux": obs_sum,
        "_arrays": {
            "model_source": model_source, "obs_source": obs_source, "sigma": sigma,
        },
    }
    if obs_sum == 0.0:
        row.update(_undefined_metric("flux_ratio_model_over_obs", "zero_denominator"))
    else:
        row["flux_ratio_model_over_obs"] = float(model_sum / obs_sum)
    return row


def _frame_shape(exposure, exposure_id: int) -> tuple[int, int]:
    height, width = exposure.image_height, exposure.image_width
    if height is None or width is None:
        raise ValueError(
            f"Exposure {exposure_id} has no image frame shape; coverage strata are "
            "defined for image exposures only."
        )
    return int(height), int(width)


def _observation_wcs(exposure, exposure_id: int):
    obs_wcs = getattr(exposure, "obs_wcs", None)
    if obs_wcs is None:
        raise ValueError(
            f"Exposure {exposure_id} has no obs_wcs; coverage strata need the "
            "per-exposure observation WCS."
        )
    return obs_wcs


def _omega_to_radec_deg(omega_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(omega_xyz, dtype=np.float64)
    if omega.shape[-1] != 3:
        raise ValueError("omega_xyz last dimension must be 3.")
    ra = np.rad2deg(np.arctan2(omega[..., 1], omega[..., 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(omega[..., 2], -1.0, 1.0)))
    return ra, dec


def _wrap_deg(delta_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(delta_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _undefined_metric(name: str, reason: str) -> dict[str, None | str]:
    return {name: None, f"{name}_undefined_reason": reason}


def exposure_sky_grid(exposure, exposure_id: int) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec of every pixel center of an exposure, through the same forward
    transform the trainer renders with (pointing offsets included)."""
    height, width = _frame_shape(exposure, exposure_id)
    obs_wcs = _observation_wcs(exposure, exposure_id)
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    omega = np.asarray(obs_wcs.pixel_to_omega_xyz(xx, yy), dtype=np.float64)
    if omega.shape != (height, width, 3):
        raise ValueError(
            f"Exposure {exposure_id} obs_wcs.pixel_to_omega_xyz must return "
            f"[height, width, 3]; got {omega.shape}."
        )
    return _omega_to_radec_deg(omega)


def sky_to_pixel(
    exposure, exposure_id: int, ra_deg: np.ndarray, dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the observation WCS, undoing the adapter's pointing offset
    exactly as AstropyWCSAdapter.pixel_to_omega_xyz applies it.

    Uses Astropy's iterative all_world2pix inverse when the forward WCS carries
    SIP distortion, and the core wcs_world2pix inverse otherwise.
    """
    obs_wcs = _observation_wcs(exposure, exposure_id)
    raw_wcs = getattr(obs_wcs, "wcs", None)
    if raw_wcs is None:
        raise RuntimeError(
            f"Exposure {exposure_id} observation WCS exposes no underlying "
            "Astropy WCS, so training coverage cannot be resolved. Pass "
            "--skip-coverage-strata to score without coverage stratification."
        )
    ctype = getattr(getattr(raw_wcs, "wcs", None), "ctype", ())
    has_distortion = getattr(raw_wcs, "sip", None) is not None or any(
        str(axis).upper().endswith("-SIP") for axis in ctype
    )
    if has_distortion:
        world2pix = getattr(raw_wcs, "all_world2pix", None)
        if not callable(world2pix):
            raise RuntimeError(
                f"Exposure {exposure_id} has distortion terms but no "
                "all_world2pix inverse; refusing a core-inverse fallback."
            )
    else:
        world2pix = getattr(raw_wcs, "wcs_world2pix", None)
        if not callable(world2pix):
            raise RuntimeError(
                f"Exposure {exposure_id} observation WCS exposes no "
                "wcs_world2pix inverse, so training coverage cannot be resolved. "
                "Pass --skip-coverage-strata to score without coverage "
                "stratification."
            )
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    dx_arcsec = float(getattr(obs_wcs, "dx_arcsec", 0.0) or 0.0)
    dy_arcsec = float(getattr(obs_wcs, "dy_arcsec", 0.0) or 0.0)
    if dx_arcsec != 0.0 or dy_arcsec != 0.0:
        dec = dec - dy_arcsec / 3600.0
        cos_dec = np.clip(np.cos(np.deg2rad(dec)), 1e-12, None)
        ra = ra - dx_arcsec / (3600.0 * cos_dec)
    try:
        if has_distortion:
            x_pixel, y_pixel = world2pix(
                ra, dec, 0, tolerance=1.0e-10, maxiter=100,
                detect_divergence=True, quiet=False,
            )
        else:
            x_pixel, y_pixel = world2pix(ra, dec, 0)
    except Exception as exc:
        if has_distortion:
            raise RuntimeError(
                f"Exposure {exposure_id} distorted WCS all_world2pix "
                f"failed or diverged: {exc}"
            ) from exc
        raise
    if has_distortion and (
        not np.all(np.isfinite(x_pixel)) or not np.all(np.isfinite(y_pixel))
    ):
        raise RuntimeError(
            f"Exposure {exposure_id} distorted WCS all_world2pix returned "
            "non-finite pixel coordinates."
        )
    return (
        np.asarray(x_pixel, dtype=np.float64),
        np.asarray(y_pixel, dtype=np.float64),
    )


def _footprint_sky_bounds(
    exposure, exposure_id: int, ra_reference_deg: float,
) -> tuple[float, float, float, float]:
    """Padded RA-relative/Dec bounding box of an exposure's frame footprint,
    sampled along the frame border at pixel resolution so that any WCS
    curvature between the corners is inside the box."""
    height, width = _frame_shape(exposure, exposure_id)
    obs_wcs = _observation_wcs(exposure, exposure_id)
    xs = np.linspace(-0.5, width - 0.5, width + 1, dtype=np.float64)
    ys = np.linspace(-0.5, height - 0.5, height + 1, dtype=np.float64)
    border_x = np.concatenate(
        [xs, xs, np.full(ys.size, -0.5), np.full(ys.size, width - 0.5)]
    )
    border_y = np.concatenate(
        [np.full(xs.size, -0.5), np.full(xs.size, height - 0.5), ys, ys]
    )
    omega = np.asarray(obs_wcs.pixel_to_omega_xyz(border_x, border_y), dtype=np.float64)
    ra, dec = _omega_to_radec_deg(omega)
    ra_relative = _wrap_deg(ra - float(ra_reference_deg))
    span = float(ra_relative.max() - ra_relative.min())
    if span > 90.0:
        raise ValueError(
            f"Exposure {exposure_id} spans {span:.3f} deg of relative RA; the "
            "coverage bounding-box filter assumes survey-frame footprints."
        )
    pad_deg = float(exposure.pixel_scale_arcsec or 0.0) / 3600.0
    cos_dec = max(float(np.cos(np.deg2rad(np.median(dec)))), 1e-6)
    pad_ra_deg = pad_deg / cos_dec
    return (
        float(ra_relative.min()) - pad_ra_deg, float(ra_relative.max()) + pad_ra_deg,
        float(dec.min()) - pad_deg, float(dec.max()) + pad_deg,
    )


def coverage_source_exposure_ids(
    store, exposure_id: int, trained_exposure_ids,
) -> list[int]:
    """Same-band exposures this run actually trained on, excluding the frame
    itself, so a trained control is scored under the same leave-one-out
    coverage definition as a held-out frame."""
    band = str(store.image_band_by_exposure[int(exposure_id)])
    trained = {int(e) for e in trained_exposure_ids}
    return sorted(
        int(other)
        for other, other_band in store.image_band_by_exposure.items()
        if str(other_band) == band
        and int(other) in trained
        and int(other) != int(exposure_id)
    )


def build_coverage_depth_map(
    store, exposure_id: int, source_exposure_ids,
) -> np.ndarray:
    """Per-pixel count of source exposures whose frame footprint contains the
    target pixel center.

    Serial on purpose, like the trainer's per-exposure WCS transforms: the work
    per source is one vectorized array transform, and a pool would only add
    queue and GIL handoffs around it.
    """
    target = store.exposures[int(exposure_id)]
    height, width = _frame_shape(target, exposure_id)
    ra, dec = exposure_sky_grid(target, exposure_id)
    ra_reference = float(ra[height // 2, width // 2])
    ra_flat = ra.reshape(-1)
    dec_flat = dec.reshape(-1)
    ra_relative_flat = _wrap_deg(ra_flat - ra_reference)
    depth = np.zeros((height, width), dtype=np.int32)
    depth_flat = depth.reshape(-1)
    for source_id in sorted({int(e) for e in source_exposure_ids}):
        if source_id == int(exposure_id):
            raise ValueError(
                f"Exposure {exposure_id} appears in its own coverage source set."
            )
        source = store.exposures[source_id]
        ra_low, ra_high, dec_low, dec_high = _footprint_sky_bounds(
            source, source_id, ra_reference
        )
        candidate = (
            (ra_relative_flat >= ra_low)
            & (ra_relative_flat <= ra_high)
            & (dec_flat >= dec_low)
            & (dec_flat <= dec_high)
        )
        candidate_indices = np.flatnonzero(candidate)
        if candidate_indices.size == 0:
            continue
        x_pixel, y_pixel = sky_to_pixel(
            source, source_id, ra_flat[candidate_indices], dec_flat[candidate_indices]
        )
        source_height, source_width = _frame_shape(source, source_id)
        inside = (
            (x_pixel >= -0.5)
            & (x_pixel < source_width - 0.5)
            & (y_pixel >= -0.5)
            & (y_pixel < source_height - 0.5)
        )
        if not np.any(inside):
            continue
        depth_flat[candidate_indices[inside]] += 1
    return depth


def _stratum_metrics(
    model: np.ndarray, obs: np.ndarray, sigma: np.ndarray, *, snr_threshold: float,
) -> dict[str, Any]:
    model_sum = float(model.sum())
    obs_sum = float(obs.sum())
    metrics: dict[str, Any] = {
        "n_pixels": int(model.size),
        "model_flux": model_sum,
        "obs_flux": obs_sum,
    }
    if model.size == 0:
        metrics.update(_undefined_metric("flux_ratio_model_over_obs", "empty_stratum"))
        metrics["n_significant_pixels"] = 0
        metrics.update(_undefined_metric("median_pixel_flux_ratio", "empty_stratum"))
        metrics.update(_undefined_metric("mean_square_pull", "empty_stratum"))
        metrics.update(_undefined_metric("median_abs_pull", "empty_stratum"))
        return metrics

    if obs_sum == 0.0:
        metrics.update(_undefined_metric("flux_ratio_model_over_obs", "zero_denominator"))
    else:
        metrics["flux_ratio_model_over_obs"] = float(model_sum / obs_sum)
    significant = obs > snr_threshold * sigma
    n_significant = int(significant.sum())
    metrics["n_significant_pixels"] = n_significant
    if n_significant == 0:
        metrics.update(_undefined_metric("median_pixel_flux_ratio", "no_significant_pixels"))
    else:
        metrics["median_pixel_flux_ratio"] = float(
            np.median(model[significant] / obs[significant])
        )
    pull = (model - obs) / sigma
    metrics["mean_square_pull"] = float(np.mean(pull**2))
    metrics["median_abs_pull"] = float(np.median(np.abs(pull)))
    return metrics


def stratified_flux_metrics(
    model_source: np.ndarray, obs_source: np.ndarray, sigma: np.ndarray,
    coverage_depth: np.ndarray, *, snr_threshold: float = 3.0,
) -> dict[str, Any]:
    """Split the frame's flux metrics by training coverage depth.

    The "all" stratum is computed over exactly the pixels score_exposure scores,
    so its flux ratio reproduces the whole-frame number rather than approximating
    it.
    """
    shapes = {
        np.asarray(array).shape
        for array in (model_source, obs_source, sigma, coverage_depth)
    }
    if len(shapes) != 1:
        raise ValueError(f"Coverage stratification needs matching shapes; got {shapes}.")
    if not np.isfinite(snr_threshold) or snr_threshold <= 0.0:
        raise ValueError("snr_threshold must be positive and finite.")
    model = np.asarray(model_source, dtype=np.float64)
    obs = np.asarray(obs_source, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    depth = np.asarray(coverage_depth)
    valid = np.isfinite(sig) & np.isfinite(obs) & np.isfinite(model)
    if not np.any(valid):
        raise RuntimeError("Coverage stratification found no valid pixels to score.")
    if np.any(sig[valid] <= 0.0):
        raise ValueError("Coverage stratification requires positive sigma pixels.")
    model, obs, sig, depth = model[valid], obs[valid], sig[valid], depth[valid]
    hole = depth == 0
    covered = ~hole
    obs_covered_sum = float(obs[covered].sum())
    counts = np.bincount(depth.astype(np.int64))
    metrics = {
        "n_valid_pixels": int(model.size),
        "n_hole_pixels": int(hole.sum()),
        "n_covered_pixels": int(covered.sum()),
        "hole_fraction": float(hole.sum() / model.size),
        "max_depth": int(depth.max()),
        "depth_histogram": {
            str(index): int(count) for index, count in enumerate(counts) if count
        },
        "robust_snr_threshold": float(snr_threshold),
        "strata": {
            "all": _stratum_metrics(model, obs, sig, snr_threshold=snr_threshold),
            "hole": _stratum_metrics(
                model[hole], obs[hole], sig[hole], snr_threshold=snr_threshold
            ),
            "covered": _stratum_metrics(
                model[covered], obs[covered], sig[covered], snr_threshold=snr_threshold
            ),
        },
    }
    if obs_covered_sum == 0.0:
        metrics.update(
            _undefined_metric("flux_ratio_masked_denominator", "zero_denominator")
        )
    else:
        metrics["flux_ratio_masked_denominator"] = float(
            model.sum() / obs_covered_sum
        )
    return metrics


COVERAGE_METHOD = "same_band_training_footprint_overlap_leave_one_out"

COVERAGE_DEFINITIONS = {
    "coverage_depth": (
        "number of same-band exposures this run trained on, excluding the scored "
        "exposure itself and every held-out exposure, whose frame footprint "
        "contains the scored pixel center; membership is -0.5 <= x < width - 0.5 "
        "and -0.5 <= y < height - 0.5 in the source frame, through the source "
        "observation WCS inverse. Frame footprints, not per-pixel training masks: "
        "a source frame's own masked or randomly held-out pixels still count as "
        "coverage."
    ),
    "hole": "coverage_depth == 0",
    "covered": "coverage_depth >= 1",
    "hole_fraction": (
        "n_hole_pixels / n_valid_pixels over the pixels the whole-frame metrics "
        "already score (finite sigma, observation and model)"
    ),
    "flux_ratio_model_over_obs": (
        "sum(model) / sum(obs) over the stratum's pixels; the 'all' stratum "
        "reproduces the top-level flux_ratio_model_over_obs exactly"
    ),
    "flux_ratio_masked_denominator": (
        "sum(model over all valid pixels) / sum(obs over covered pixels only): "
        "total rendered flux against the only denominator same-band training data "
        "constrains"
    ),
    "median_pixel_flux_ratio": (
        "median of the per-pixel ratio model/obs over the stratum's pixels with "
        "obs > robust_snr_threshold * sigma; a rank statistic on a denominator "
        "bounded away from zero, so unlike the signed sums it is not set by a "
        "handful of bright pixels"
    ),
    "mean_square_pull": "as the whole-frame metric, restricted to the stratum",
    "median_abs_pull": "as the whole-frame metric, restricted to the stratum",
}


def attach_coverage_metrics(
    row: dict[str, Any], store, *, source_exposure_ids, snr_threshold: float = 3.0,
) -> dict[str, Any]:
    """Add the coverage block to a score_exposure row, in place."""
    arrays = row["_arrays"]
    source_ids = sorted({int(e) for e in source_exposure_ids})
    depth = build_coverage_depth_map(store, int(row["exposure_id"]), source_ids)
    arrays["coverage_depth"] = depth
    coverage = stratified_flux_metrics(
        arrays["model_source"], arrays["obs_source"], arrays["sigma"], depth,
        snr_threshold=snr_threshold,
    )
    coverage["method"] = COVERAGE_METHOD
    coverage["n_source_exposures"] = len(source_ids)
    coverage["source_exposure_ids"] = source_ids
    row["coverage"] = coverage
    return row


def format_coverage_summary(role: str, row: Mapping[str, Any]) -> str:
    coverage = row["coverage"]
    hole = coverage["strata"]["hole"]
    covered = coverage["strata"]["covered"]
    hole_flux_ratio = _format_metric(
        hole["flux_ratio_model_over_obs"],
        hole.get("flux_ratio_model_over_obs_undefined_reason"),
    )
    covered_flux_ratio = _format_metric(
        covered["flux_ratio_model_over_obs"],
        covered.get("flux_ratio_model_over_obs_undefined_reason"),
    )
    masked_denominator = _format_metric(
        coverage["flux_ratio_masked_denominator"],
        coverage.get("flux_ratio_masked_denominator_undefined_reason"),
    )
    hole_median_ratio = _format_metric(
        hole["median_pixel_flux_ratio"],
        hole.get("median_pixel_flux_ratio_undefined_reason"),
    )
    covered_median_ratio = _format_metric(
        covered["median_pixel_flux_ratio"],
        covered.get("median_pixel_flux_ratio_undefined_reason"),
    )
    return (
        f"[holdout-eval] {role} exp {row['exposure_id']} coverage: hole "
        f"{coverage['hole_fraction'] * 100.0:.1f}% of {coverage['n_valid_pixels']} "
        f"valid px over {coverage['n_source_exposures']} same-band training "
        f"exposures | flux ratio hole {hole_flux_ratio} covered {covered_flux_ratio} "
        f"| masked denominator {masked_denominator} | median pixel ratio "
        f"(>{coverage['robust_snr_threshold']:g} sigma) hole "
        f"{hole_median_ratio} covered {covered_median_ratio}"
    )


def _format_metric(value: Any, reason: str | None = None) -> str:
    if value is None:
        if not reason:
            raise ValueError("Undefined report metric is missing its reason.")
        return f"n/a ({reason})"
    return f"{float(value):.4f}"


def evaluate_exposure_stratified(
    trainer, store, exposure_id: int, *, trained_exposure_ids, chunk_size: int = 16384,
    snr_threshold: float = 3.0, keep_arrays: bool = False,
) -> dict[str, Any]:
    """Score one exposure and stratify it by training coverage.

    The importable entry point: an onset-versus-step sweep loads a trainer once
    and calls this per checkpoint.
    """
    row = score_exposure(trainer, store, int(exposure_id), chunk_size)
    attach_coverage_metrics(
        row, store,
        source_exposure_ids=coverage_source_exposure_ids(
            store, int(exposure_id), trained_exposure_ids
        ),
        snr_threshold=snr_threshold,
    )
    if not keep_arrays:
        row.pop("_arrays")
    return row


def _psf_width_field(psf_model: str) -> str:
    model = str(psf_model).strip().lower()
    fields = {
        "gaussian": "psf_fwhm_arcsec",
        "dgpsf": "psf_effective_fwhm_dg_arcsec",
        "mog": "psf_effective_fwhm_kl_arcsec",
        "kl": "psf_effective_fwhm_kl_arcsec",
    }
    if model not in fields:
        raise ValueError(
            f"Unsupported psf_model {psf_model!r} for control matching; "
            f"expected one of {sorted(fields)}."
        )
    return fields[model]


def pick_controls(
    store, holdout_ids: list[int], per_band: int, trained_ids: set[int],
    psf_model: str,
) -> tuple[list[int], dict[str, Any]]:
    """Same-band actually-trained exposures, nearest in PSF width to each
    held-out one. trained_ids must come from the trainer's training indices so
    an excluded or untrained frame can never masquerade as a control."""
    preferred_field = _psf_width_field(psf_model)
    legacy_field = "psf_fwhm_arcsec"
    holdout_set = set(holdout_ids)
    controls: list[int] = []
    width_field_by_exposure: dict[str, str] = {}

    def width(exp_id: int) -> float:
        exposure = store.exposures[exp_id]
        value = getattr(exposure, preferred_field, None)
        actual_field = preferred_field
        if value is None and preferred_field != legacy_field:
            value = getattr(exposure, legacy_field, None)
            actual_field = legacy_field
            warnings.warn(
                f"Exposure {exp_id} is missing {preferred_field} for "
                f"psf_model={psf_model!r}; falling back to {legacy_field}.",
                RuntimeWarning,
                stacklevel=2,
            )
        value = float(value if value is not None else np.nan)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Exposure {exp_id} has no finite positive {actual_field} for "
                f"psf_model={psf_model!r}."
            )
        width_field_by_exposure[str(exp_id)] = actual_field
        return value

    for held_id in holdout_ids:
        band = str(store.image_band_by_exposure[held_id])
        held_psf = width(held_id)
        candidates = [
            int(e) for e, b in store.image_band_by_exposure.items()
            if str(b) == band
            and int(e) in trained_ids
            and int(e) not in holdout_set
            and int(e) not in controls
        ]
        if not candidates:
            raise RuntimeError(
                f"No trained same-band control available for held-out exposure "
                f"{held_id} (band {band})."
            )
        candidates.sort(
            key=lambda exposure_id: abs(width(exposure_id) - held_psf)
        )
        controls.extend(candidates[:per_band])
    matching = {
        "psf_model": str(psf_model).strip().lower(),
        "preferred_width_field": preferred_field,
        "width_field_by_exposure": width_field_by_exposure,
    }
    return controls, matching


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated held-out labels; default reads "
        "trainer.exclude_exposure_labels from the run config.",
    )
    parser.add_argument("--controls-per-band", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--allow-operator-reinterpretation", action="store_true")
    parser.add_argument(
        "--coverage-snr", type=float, default=3.0,
        help="Pixels with obs > this many sigma carry the median pixel flux ratio.",
    )
    parser.add_argument(
        "--skip-coverage-strata", action="store_true",
        help="Score whole frames only; needed for stores whose WCS objects have "
        "no inverse.",
    )
    args = parser.parse_args()

    source_commit = evaluator_source_commit()
    (
        cfg,
        store,
        trainer,
        train_entry,
        checkpoint,
        current_operator_config,
        reinterpreted,
    ) = load_trainer(
        args.run_dir, args.outdir, args.prepared_dir, args.device,
        allow_operator_reinterpretation=args.allow_operator_reinterpretation,
    )
    if reinterpreted:
        args.outdir = args.outdir.with_name(f"{args.outdir.name}_reinterpreted")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.labels:
        labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    else:
        labels = [
            str(item)
            for item in (
                OmegaConf.to_container(cfg.trainer, resolve=True).get(
                    "exclude_exposure_labels"
                )
                or []
            )
        ]
    if not labels:
        raise SystemExit(
            "No held-out labels: pass --labels or use a run trained with "
            "trainer.exclude_exposure_labels."
        )
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    holdout_ids = list(
        train_entry._resolve_excluded_image_exposure_ids(
            data_cfg, {"exclude_exposure_labels": labels}, store
        )
    )
    if not getattr(trainer, "analytic_images_enabled", False):
        raise SystemExit(
            "This run did not train with analytic_images: the deterministic "
            "render path would not match the training operator, so the "
            "novel-view comparison is not operator-clean. Refusing."
        )
    not_held_out = [
        exp_id for exp_id in holdout_ids
        if exp_id not in trainer.exclude_image_exposure_ids
    ]
    if not_held_out:
        raise SystemExit(
            f"Exposures {not_held_out} were NOT excluded during this run's "
            f"training (trained exclusion set: "
            f"{sorted(trainer.exclude_image_exposure_ids)}). Reporting them as "
            "holdout would be false. Refusing."
        )
    trained_ids = {int(e) for e in trainer.image_train_indices_by_exposure}
    control_ids, control_matching = pick_controls(
        store, holdout_ids, args.controls_per_band, trained_ids, trainer.psf_model,
    )
    print(f"[holdout-eval] held-out ids {holdout_ids}, controls {control_ids}", flush=True)

    evaluating = operator_identity_for_evaluation(
        cfg, trainer, store, holdout_ids + control_ids
    )
    evaluating.update(current_operator_config)
    report = assemble_verdict(
        labels=labels,
        evaluating=evaluating,
        checkpoint=checkpoint,
        source_commit=source_commit,
    )
    report["control_matching"] = control_matching
    if not args.skip_coverage_strata:
        report["coverage_stratification"] = {
            "method": COVERAGE_METHOD,
            "robust_snr_threshold": float(args.coverage_snr),
            "definitions": dict(COVERAGE_DEFINITIONS),
        }
    row_labels = labels_by_exposure_id(data_cfg, holdout_ids + control_ids, store)
    for role, ids in (("holdout", holdout_ids), ("controls", control_ids)):
        for exp_id in ids:
            row = score_exposure(trainer, store, exp_id, args.chunk_size)
            row["label"] = row_labels[int(exp_id)]
            if not args.skip_coverage_strata:
                attach_coverage_metrics(
                    row, store,
                    source_exposure_ids=coverage_source_exposure_ids(
                        store, exp_id, trained_ids
                    ),
                    snr_threshold=float(args.coverage_snr),
                )
            arrays = row.pop("_arrays")
            if args.save_arrays:
                np.savez_compressed(
                    args.outdir / f"exposure_{exp_id:04d}_{role}.npz", **arrays
                )
            report[role].append(row)
            mean_square_pull = _format_metric(
                row["mean_square_pull"],
                row.get("mean_square_pull_undefined_reason"),
            )
            median_abs_pull = _format_metric(
                row["median_abs_pull"], row.get("median_abs_pull_undefined_reason")
            )
            flux_ratio = _format_metric(
                row["flux_ratio_model_over_obs"],
                row.get("flux_ratio_model_over_obs_undefined_reason"),
            )
            print(
                f"[holdout-eval] {role} exp {exp_id} band {row['band']}: "
                f"mean sq pull {mean_square_pull}, median|pull| {median_abs_pull}, "
                f"flux ratio {flux_ratio}",
                flush=True,
            )
            if "coverage" in row:
                print(format_coverage_summary(role, row), flush=True)

    out_path = args.outdir / "holdout_exposure_report.json"
    write_report(report, out_path)
    print(f"[holdout-eval] report -> {out_path}", flush=True)


def write_report(report: Mapping[str, Any], out_path: Path) -> None:
    with Path(out_path).open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
