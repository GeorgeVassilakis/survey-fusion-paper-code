from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class ImageDiagnosticConfig:
    profile_timing: bool = False
    deterministic_supersample: int = 2
    deterministic_chunk_size: int = 2048
    max_exposures: int | None = None


def resolve_image_diagnostic_config(cfg: object | None) -> ImageDiagnosticConfig:
    profile_timing = False
    deterministic_supersample = 2
    deterministic_chunk_size = 2048
    max_exposures: int | None = None
    if isinstance(cfg, dict):
        profile_timing = bool(cfg.get("profile_timing", False))
        deterministic_supersample = int(cfg.get("deterministic_supersample", 2) or 2)
        deterministic_chunk_size = int(cfg.get("deterministic_chunk_size", 2048) or 2048)
        raw_max_exposures = cfg.get("max_exposures", None)
    elif cfg is not None and hasattr(cfg, "get"):
        profile_timing = bool(cfg.get("profile_timing", False))
        deterministic_supersample = int(cfg.get("deterministic_supersample", 2) or 2)
        deterministic_chunk_size = int(cfg.get("deterministic_chunk_size", 2048) or 2048)
        raw_max_exposures = cfg.get("max_exposures", None)
    else:
        raw_max_exposures = None
    if raw_max_exposures is not None:
        max_exposures = max(1, int(raw_max_exposures))
    return ImageDiagnosticConfig(
        profile_timing=bool(profile_timing), deterministic_supersample=max(1, int(deterministic_supersample)),
        deterministic_chunk_size=max(1, int(deterministic_chunk_size)), max_exposures=max_exposures,
    )


def compute_image_diagnostic_predictions(
    trainer, *, postprocess_cfg: object | None = None, include_additive: bool = True,
    profile: Dict[str, Any] | None = None, required_exposure_ids: list[int] | tuple[int, ...] | None = None,
) -> Dict[int, Dict[str, object]]:
    cfg = resolve_image_diagnostic_config(postprocess_cfg)
    active_profile = profile if cfg.profile_timing and profile is not None else None
    exposure_ids = None
    if cfg.max_exposures is not None:
        available_ids = sorted(int(exp_id) for exp_id in trainer.image_exposure_ids)
        exposure_ids = available_ids[: int(cfg.max_exposures)]
        required_ids = [
            int(exp_id)
            for exp_id in (required_exposure_ids or [])
            if int(exp_id) in available_ids
        ]
        for required_id in sorted(set(required_ids)):
            if required_id in exposure_ids:
                continue
            if len(exposure_ids) < int(cfg.max_exposures):
                exposure_ids.append(required_id)
            elif exposure_ids:
                exposure_ids[-1] = required_id
            exposure_ids = sorted(set(exposure_ids))
    if active_profile is not None:
        active_profile["requested_mode"] = "deterministic"
        active_profile["actual_mode"] = "deterministic"
        active_profile["deterministic_supersample"] = int(cfg.deterministic_supersample)
        active_profile["deterministic_chunk_size"] = int(cfg.deterministic_chunk_size)
        active_profile["max_exposures"] = cfg.max_exposures
        active_profile["selected_exposure_ids"] = exposure_ids
        deterministic_profile: Dict[str, Any] = {}
        start = time.perf_counter()
        predictions = trainer.compute_datum_image_predictions_deterministic(
            supersample=int(cfg.deterministic_supersample), chunk_size=int(cfg.deterministic_chunk_size),
            include_additive=bool(include_additive), exposure_ids=exposure_ids, profile=deterministic_profile,
        )
        active_profile["timings_sec"] = {"compute_image_diagnostic_predictions": time.perf_counter() - start}
        active_profile["deterministic"] = deterministic_profile
        return predictions
    return trainer.compute_datum_image_predictions_deterministic(
        supersample=int(cfg.deterministic_supersample), chunk_size=int(cfg.deterministic_chunk_size),
        include_additive=bool(include_additive), exposure_ids=exposure_ids,
    )
