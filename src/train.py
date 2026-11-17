from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import nullcontext
import datetime as dt
import hashlib
import json
import os
import pickle
import resource
import shutil
import stat as stat_module
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from survey_fusion.data.bandpasses import load_bandpasses
from survey_fusion.data.image_variance import image_variance_columns, poisson_current_requested
from survey_fusion.data.sdss_scene import load_sdss_scene
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.fpm_masks import check_sidecar_binding
from survey_fusion.data.prepared import (
    load_prepared, prepare_seed, prepared_digest, prepared_path, save_prepared,
)
from survey_fusion.evaluation import (diagnostics, image_diagnostics, metrics as metrics_mod, visualization)
from survey_fusion.models.spectral_field import GalaxySpectralField
from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_init import seed_from_catalog, seed_from_store
from survey_fusion.utils.perf_env import (
    exact_pair_pruning_enabled as _exact_pair_pruning_enabled,
)
from survey_fusion.training.checkpoints import (
    GAUGE_SIDECAR_FILENAME,
    RNG_STATE_KEY,
    SPLAT_LINEAGE_BUFFER_DEFAULTS,  # noqa: F401 - public compatibility export
    build_gauge_canonical_sidecar,
    capture_rng_state,
    checkpoint_absolute_step,
    checkpoint_is_gauge_canonical,
    checkpoint_operator_identity,
    migrate_checkpoint_model_state as _migrate_checkpoint_model_state,
    model_config_for_checkpoint,
    operator_config_for_trainer,
    restore_rng_state,
    validate_checkpoint_model_config,
    validate_checkpoint_operator_config,
)
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.paths import (build_run_dir, dump_config, ensure_dir, paths_from_config, resolve_path)
from survey_fusion.utils.distributed import (
    broadcast_object, destroy_distributed, initialize_distributed, is_distributed_active,
    resolve_distributed_batch_size, resolve_distributed_context, synchronize_distributed,
)
from survey_fusion.utils.run import RunContext
from survey_fusion.utils.speed import SpeedLogger
from survey_fusion.utils.torch_utils import (
    apply_precision_policy, describe_runtime, resolve_amp_settings, select_runtime_device, set_seed,
    validate_runtime_environment,
)
from survey_fusion.utils.wandb_utils import WandbRun

DEFAULT_SDSS_DIAGNOSTIC_BAND = "i"
DEFAULT_DIAGNOSTIC_SIGNAL_THRESHOLD = 0.05
CURRENT_CHECKPOINT_VERSION = 4
# v3 checkpoints predate the per-splat lineage buffers; loading migrates them
# by injecting default lineage tensors (see _migrate_checkpoint_model_state).
SUPPORTED_CHECKPOINT_VERSIONS = (3, CURRENT_CHECKPOINT_VERSION)
REQUIRED_CHECKPOINT_KEYS = (
    "checkpoint_version", "model_config", "model_state", "calibration_state", "optimizer_state",
    "lr_scheduler_state", "resume_meta",
)


def _logspace_steps(*, max_steps: int, count: int) -> tuple[int, ...]:
    if max_steps <= 0 or count <= 0:
        return ()
    values = np.geomspace(1, max_steps, num=count)
    steps = {min(max_steps, max(1, int(round(value)))) for value in values}
    steps.update((1, max_steps))
    return tuple(sorted(steps))


def _visualization_step_due(
    *, step: int, every_n_steps: int, scheduled_steps: Collection[int] = (),
) -> bool:
    if step <= 0:
        return False
    return (
        every_n_steps > 0 and step % every_n_steps == 0
    ) or step in scheduled_steps


def _fiber_sigma_by_exposure(data_store: Any) -> dict[int, np.ndarray]:
    """Slice the exact trainer-NLL sigma tensor into wavelength-ordered fibers."""
    batch = data_store.batch
    exposure_ids = batch.exposure_id.detach().cpu().numpy()
    datum_indices = batch.datum_index.detach().cpu().numpy()
    sigma = batch.sigma.detach().cpu().numpy()
    result: dict[int, np.ndarray] = {}
    for exp_id, target in data_store.spec_diagnostics_by_exposure.items():
        rows = np.flatnonzero(exposure_ids == int(exp_id))
        if rows.size != np.asarray(target.lam).size:
            raise ValueError(
                f"Spectral exposure {int(exp_id)} has {rows.size} sigma rows but "
                f"{np.asarray(target.lam).size} wavelength bins."
            )
        local = np.asarray(datum_indices[rows], dtype=np.int64)
        order = np.argsort(local, kind="stable")
        if not np.array_equal(local[order], np.arange(rows.size, dtype=np.int64)):
            raise ValueError(
                f"Spectral exposure {int(exp_id)} datum indices are not exactly 0..n_bins-1."
            )
        result[int(exp_id)] = np.asarray(sigma[rows[order]], dtype=np.float64)
    return result


def _checkpoint_retention_due(
    *, step: int, every_n_steps: int, final_step: int | None = None,
    extra_steps: Collection[int] = (),
) -> bool:
    return (
        final_step is not None and step == final_step
    ) or (
        step > 0 and every_n_steps > 0 and step % every_n_steps == 0
    ) or step in extra_steps


def _epoch_retention_steps(*, steps_per_epoch: int, epochs: int, every_n_epochs: int) -> set[int]:
    """Absolute steps at every every_n_epochs-th epoch boundary of the run."""
    if every_n_epochs <= 0 or steps_per_epoch <= 0:
        return set()
    return {
        epoch * steps_per_epoch for epoch in range(every_n_epochs, epochs + 1, every_n_epochs)
    }


def _checkpoint_plan(
    trainer_cfg: Mapping[str, object], *, epochs: int, steps_per_epoch: int,
) -> tuple[int, int, set[int]]:
    """(rolling checkpoint cadence, step-based retention cadence, retained absolute steps)."""
    checkpoint_every = int(trainer_cfg.get("checkpoint_every_n_steps", 50_000) or 0)
    retain_every = int(trainer_cfg.get("checkpoint_retain_every_n_steps", 0) or 0)
    retain_every_epochs = int(trainer_cfg.get("checkpoint_retain_every_n_epochs", 1) or 0)
    if retain_every_epochs < 0:
        raise ValueError("trainer.checkpoint_retain_every_n_epochs must be non-negative.")
    raw_steps = trainer_cfg.get("checkpoint_retain_steps", []) or []
    if isinstance(raw_steps, (str, bytes, Mapping)):
        raise ValueError("trainer.checkpoint_retain_steps must be a list of absolute steps.")
    retained = {int(step) for step in raw_steps} | _epoch_retention_steps(
        steps_per_epoch=steps_per_epoch, epochs=epochs, every_n_epochs=retain_every_epochs,
    )
    return checkpoint_every, retain_every, retained


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_checkpoint_atomically(source: Path, destination: Path) -> Path:
    tmp = destination.with_name(f"{destination.name}.partial")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)
    return destination


class _CheckpointArchiver:
    def __init__(self, *, run_dir: Path, archive_dir: Path, keep_local: int) -> None:
        if keep_local < 0:
            raise ValueError("trainer.checkpoint_keep_local must be non-negative.")
        self.run_dir = run_dir
        self.archive_dir = archive_dir
        self.keep_local = int(keep_local)
        self.marker_path = run_dir / "CHECKPOINT_ARCHIVE_FAILED"
        self.pointer_path = run_dir / "ARCHIVED_CKPTS.txt"
        self._condition = threading.Condition()
        self._pending: dict[Path, int] = {}
        self._archived_local: set[Path] = set()
        self._generation = 0
        self._active = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="checkpoint-archiver", daemon=True,
        )
        self._thread.start()

    def submit(self, *, step: int, path: Path) -> None:
        with self._condition:
            if self._stopping:
                print(
                    f"WARNING: checkpoint archiver is stopped; kept {path} locally.",
                    file=sys.stderr, flush=True,
                )
                return
            self._pending[path] = int(step)
            self._generation += 1
            self._condition.notify_all()

    def _run(self) -> None:
        observed_generation = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopping or self._generation != observed_generation
                )
                if self._stopping:
                    return
                observed_generation = self._generation
                pending = list(self._pending.items())
                self._active = bool(pending)
            for local_path, step in pending:
                try:
                    digest = self._archive_one(step=step, local_path=local_path)
                except Exception as exc:
                    self._record_failure(step=step, local_path=local_path, exc=exc)
                    continue
                with self._condition:
                    if self._pending.get(local_path) == step:
                        del self._pending[local_path]
                    self._archived_local.add(local_path)
                try:
                    self._prune_archived_local_checkpoints()
                except Exception as exc:
                    self._record_failure(step=step, local_path=local_path, exc=exc)
                print(
                    f"Archived retained checkpoint step {step} to "
                    f"{self.archive_dir / local_path.name} (sha256={digest}).",
                    flush=True,
                )
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def _archive_one(self, *, step: int, local_path: Path) -> str:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.archive_dir / local_path.name
        tmp_path = archive_path.with_name(
            f".{archive_path.name}.{os.getpid()}.partial"
        )
        try:
            shutil.copy2(local_path, tmp_path)
            local_size = local_path.stat().st_size
            archive_size = tmp_path.stat().st_size
            if archive_size != local_size:
                raise OSError(
                    f"checkpoint size mismatch: local={local_size}, archive={archive_size}"
                )
            local_digest = _sha256(local_path)
            archive_digest = _sha256(tmp_path)
            if archive_digest != local_digest:
                raise OSError(
                    "checkpoint sha256 mismatch: "
                    f"local={local_digest}, archive={archive_digest}"
                )
            os.replace(tmp_path, archive_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        with self.pointer_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{step} {archive_path} {local_digest}\n")
            fh.flush()
            os.fsync(fh.fileno())
        return local_digest

    def _prune_archived_local_checkpoints(self) -> None:
        retained = []
        for path in self.run_dir.glob("checkpoint_step_*.pt"):
            raw_step = path.stem.removeprefix("checkpoint_step_")
            try:
                step = int(raw_step)
            except ValueError:
                continue
            retained.append((step, path))
        retained.sort(reverse=True)
        protected = {path for _, path in retained[: self.keep_local]}
        for _, path in retained[self.keep_local :]:
            if path in self._archived_local and path not in protected:
                path.unlink()
                self._archived_local.discard(path)

    def _record_failure(self, *, step: int, local_path: Path, exc: Exception) -> None:
        message = (
            f"checkpoint archive failed for absolute step {step} ({local_path}): "
            f"{type(exc).__name__}: {exc}"
        )
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
        try:
            self.marker_path.write_text(message + "\n", encoding="utf-8")
        except Exception as marker_exc:
            print(
                "WARNING: failed to write checkpoint archive failure marker: "
                f"{type(marker_exc).__name__}: {marker_exc}",
                file=sys.stderr, flush=True,
            )

    def drain(self, *, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        last_retry = 0.0
        while True:
            with self._condition:
                remaining = len(self._pending) + int(self._active)
                if remaining == 0:
                    break
                now = time.monotonic()
                if now >= deadline:
                    print(
                        "WARNING: checkpoint archive drain timed out with "
                        f"{len(self._pending)} pending file(s); local copies were kept.",
                        file=sys.stderr, flush=True,
                    )
                    break
                print(
                    "Waiting for checkpoint archive drain: "
                    f"{len(self._pending)} pending file(s).",
                    flush=True,
                )
                if not self._active and now - last_retry >= 1.0:
                    self._generation += 1
                    last_retry = now
                    self._condition.notify_all()
                self._condition.wait(timeout=min(1.0, deadline - now))
        self.close(timeout=max(0.0, deadline - time.monotonic()))

    def close(self, *, timeout: float = 1.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout)))


def _build_checkpoint_archiver(
    trainer_cfg: Mapping[str, object], *, run_dir: Path, project_root: Path,
) -> _CheckpointArchiver | None:
    raw_archive_dir = trainer_cfg.get("checkpoint_archive_dir")
    if raw_archive_dir is None or not str(raw_archive_dir).strip():
        return None
    archive_dir = resolve_path(str(raw_archive_dir), project_root)
    keep_local = int(trainer_cfg.get("checkpoint_keep_local", 2) or 0)
    return _CheckpointArchiver(
        run_dir=run_dir, archive_dir=archive_dir, keep_local=keep_local,
    )


def _disable_hydra_job_logging() -> None:
    for arg in sys.argv[1:]:
        if arg.startswith("hydra/job_logging="):
            return
    sys.argv.append("hydra/job_logging=disabled")


def _resolve_paths(cfg: DictConfig) -> Dict[str, Path]:
    paths_cfg = OmegaConf.to_container(cfg.paths, resolve=True)
    resolved = paths_from_config({"paths": paths_cfg})
    resolved["outdir"] = ensure_dir(resolved["outdir"])
    resolved["cache_dir"] = ensure_dir(resolved.get("cache_dir", resolved["project_root"] / ".cache"))
    return resolved


def _init_model(cfg: DictConfig):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if str(model_cfg.get("name") or "mlp").strip().lower() == "splat":
        return SplatField(
            n_splats=int(model_cfg.get("n_splats", 20000)),
            n_templates=int(model_cfg.get("n_templates", 32)),
            loglam_grid=torch.linspace(
                -1.0, 1.0, int(model_cfg.get("n_loglam_grid", 96))
            ),
            init_extent=float(model_cfg.get("init_extent", 1.0)),
            init_scale=float(model_cfg.get("init_scale", 0.02)),
            sed_off_grid=str(model_cfg.get("sed_off_grid", "clamp")),
            sed_pin_gauge_knots=bool(model_cfg.get("sed_pin_gauge_knots", False)),
            sed_pad_knots_blue=int(model_cfg.get("sed_pad_knots_blue", 0) or 0),
            sed_pad_knots_red=int(model_cfg.get("sed_pad_knots_red", 0) or 0),
            sed_rng_reference_templates=model_cfg.get("sed_rng_reference_templates"),
        )
    return GalaxySpectralField(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)), n_layers=int(model_cfg.get("n_layers", 4)),
        softplus_beta=float(model_cfg.get("softplus_beta", 2.0)),
        positional_encoding=model_cfg.get("positional_encoding"),
        output_head=str(model_cfg.get("output_head") or "softplus"),
    )


def _save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


_POSTPROCESS_STAGE_NAMES = (
    "metadata",
    "predictions",
    "band_metrics",
    "metric_logging",
    "region_metrics",
    "band_grid",
    "true_band_fiber_mask",
    "holdout_region_overlay",
    "region_grid",
    "training_history_plot",
    "point_sed_diagnostics",
    "fiber_spectrum_diagnostics",
    "radial_profiles",
    "radial_diagnostics",
    "prediction_outputs",
    "profile",
    "wandb",
    "metrics_operator_provenance",
)


class _PostprocessStatus:
    def __init__(self, operator_identity: dict[str, object]) -> None:
        self.operator_identity = operator_identity
        self.entries = {
            name: {
                "name": name,
                "status": "skipped",
                "reason": "not reached",
                "operator_identity": operator_identity,
            }
            for name in _POSTPROCESS_STAGE_NAMES
        }
        self.active_stage: str | None = None

    def start(self, name: str) -> None:
        self.active_stage = name
        self.entries[name] = {
            "name": name,
            "status": "skipped",
            "reason": "in progress",
            "operator_identity": self.operator_identity,
        }

    def ok(self, name: str) -> None:
        self.entries[name] = {
            "name": name,
            "status": "ok",
            "reason": None,
            "operator_identity": self.operator_identity,
        }
        if self.active_stage == name:
            self.active_stage = None

    def skip(self, name: str, reason: str) -> None:
        self.entries[name] = {
            "name": name,
            "status": "skipped",
            "reason": str(reason),
            "operator_identity": self.operator_identity,
        }
        if self.active_stage == name:
            self.active_stage = None

    def fail(self, name: str, exc: BaseException) -> None:
        self.entries[name] = {
            "name": name,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "operator_identity": self.operator_identity,
        }
        if self.active_stage == name:
            self.active_stage = None

    def fail_active(self, exc: BaseException) -> None:
        self.fail(self.active_stage or "pipeline", exc)

    def run_optional(self, name: str, callback, *, skip_reason: str | None = None):
        if skip_reason is not None:
            self.skip(name, skip_reason)
            return None
        self.start(name)
        try:
            result = callback()
        except Exception as exc:
            self.fail(name, exc)
            print(
                f"Warning: optional postprocessing stage {name!r} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            return None
        self.ok(name)
        return result

    def run_optional_calls(self, name: str, calls: list[tuple[str, object]]) -> None:
        errors: list[str] = []
        self.start(name)
        for label, callback in calls:
            try:
                callback()
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                print(
                    f"Warning: optional postprocessing stage {name!r} item "
                    f"{label!r} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
        if errors:
            self.fail(name, RuntimeError("; ".join(errors)))
        else:
            self.ok(name)

    def payload(
        self, *, mandatory_outputs: dict[str, bool], pipeline_error: BaseException | None,
    ) -> dict[str, object]:
        missing = [name for name, exists in mandatory_outputs.items() if not exists]
        failed = [
            entry["name"] for entry in self.entries.values()
            if entry["status"] == "failed"
        ]
        if pipeline_error is not None or missing:
            overall = "failed"
        elif failed:
            overall = "partial"
        else:
            overall = "complete"
        return {
            "overall": overall,
            "mandatory_outputs": mandatory_outputs,
            "missing_mandatory_outputs": missing,
            "operator_identity": self.operator_identity,
            "stages": list(self.entries.values()),
        }


def _postprocess_operator_identity(kwargs: dict[str, object]) -> dict[str, object]:
    provenance = kwargs.get("operator_provenance")
    if isinstance(provenance, dict):
        return provenance
    trainer = kwargs.get("trainer")
    current = operator_config_for_trainer(trainer)
    return {"trained": current, "evaluating": current}


def _mandatory_postprocess_outputs(kwargs: dict[str, object]) -> dict[str, Path]:
    run_dir = Path(kwargs["run_dir"])
    outputs = {
        "metrics.json": run_dir / "metrics.json",
        "history.json": run_dir / "history.json",
    }
    data_store = kwargs.get("data_store")
    if bool(getattr(data_store, "spec_diagnostics_by_exposure", {})):
        outputs["fiber_metrics.json"] = run_dir / "fiber_metrics.json"
    return outputs


def _annotate_metrics_operator_provenance(
    run_dir: Path, operator_provenance: dict[str, object],
) -> None:
    for path in sorted(run_dir.rglob("*metrics*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Metrics payload at {path} must be a JSON object.")
        payload["operator_provenance"] = operator_provenance
        _save_json(payload, path)


def _maybe_build_postprocess_profile(postprocess_cfg: object | None, *, stage: str) -> Dict[str, Any] | None:
    cfg = image_diagnostics.resolve_image_diagnostic_config(postprocess_cfg)
    if not cfg.profile_timing:
        return None
    return {
        "stage": str(stage), "requested_mode": "deterministic",
        "deterministic_supersample": int(cfg.deterministic_supersample),
        "deterministic_chunk_size": int(cfg.deterministic_chunk_size), "timings_sec": {},
    }


def _record_profile_timing(profile: Dict[str, Any] | None, key: str, started_at: float) -> None:
    if profile is None:
        return
    timings = profile.setdefault("timings_sec", {})
    if isinstance(timings, dict):
        timings[str(key)] = float(time.perf_counter() - started_at)




def _cfg_get(postprocess_cfg: object | None, key: str, default: object = None) -> object:
    if postprocess_cfg is None:
        return default
    getter = getattr(postprocess_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(postprocess_cfg, key, default)


def _resolve_diagnostic_outputs(postprocess_cfg: object | None) -> tuple[str, ...]:
    raw = _cfg_get(postprocess_cfg, "diagnostic_outputs", None)
    if raw is None:
        return ("total", "source")
    values = [raw] if isinstance(raw, str) else list(raw)
    outputs: list[str] = []
    for value in values:
        output = str(value).strip().lower()
        if not output:
            continue
        if output not in {"total", "source"}:
            raise ValueError("postprocess.diagnostic_outputs entries must be 'total' or 'source'.")
        if output not in outputs:
            outputs.append(output)
    if not outputs:
        raise ValueError("postprocess.diagnostic_outputs must include at least one output.")
    return tuple(outputs)


def _diagnostic_suffix(output: str) -> str:
    return "" if output == "total" else f"_{output}"


def _write_postprocess_profile(profile: Dict[str, Any] | None, *, run_dir: Path) -> None:
    if profile is None:
        return
    _save_json(profile, run_dir / "postprocess_profile.json")
    timings = profile.get("timings_sec", {})
    if isinstance(timings, dict):
        total = timings.get("total_postprocess")
        image_preds = timings.get("image_predictions")
        fiber = timings.get("fiber_spectra_diagnostics")
        radial = timings.get("radial_profiles")
        print("Postprocess profiling:")
        if total is not None:
            print(f"  total_postprocess_sec = {float(total):.3f}")
        if image_preds is not None:
            print(f"  image_predictions_sec = {float(image_preds):.3f}")
        if fiber is not None:
            print(f"  fiber_spectra_diagnostics_sec = {float(fiber):.3f}")
        if radial is not None:
            print(f"  radial_profiles_sec = {float(radial):.3f}")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _should_wrap_model_for_distributed(model: torch.nn.Module) -> bool:
    return not isinstance(_unwrap_model(model), SplatField)


def _resolve_distributed_cuda_device(local_rank: int) -> torch.device:
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        raise RuntimeError("Distributed CUDA training requires at least one visible CUDA device.")
    if visible_count == 1:
        return torch.device("cuda:0")
    if int(local_rank) < 0 or int(local_rank) >= visible_count:
        raise RuntimeError(
            f"Distributed local_rank={local_rank} is out of range for {visible_count} "
            "visible CUDA devices."
        )
    return torch.device(f"cuda:{int(local_rank)}")


def _export_distributed_cuda_env(device: torch.device) -> None:
    if device.type != "cuda":
        return
    os.environ["LOCAL_RANK"] = str(int(device.index or 0))


def _wrap_model_for_distributed(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    # The spectral field only registers fixed positional-encoding buffers.
    # They do not carry mutable running state, so per-forward DDP buffer
    # broadcasts are unnecessary and can desynchronize ranks when different
    # ranks evaluate different exposure-block layouts within a step.
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None, broadcast_buffers=False,
    )


def _resolve_holdout_band(cfg: DictConfig) -> str | None:
    value = cfg.physics.get("held_out_band")
    if value is None:
        return None
    resolved = str(value).strip()
    if resolved == "" or resolved.lower() == "none":
        return None
    return resolved


def _primary_image_exposure_id(data_store, preferred_band: str | None) -> int:
    image_map = getattr(data_store, "image_band_by_exposure", {})
    if not image_map:
        raise ValueError("No image exposures available.")
    target_band = preferred_band or DEFAULT_SDSS_DIAGNOSTIC_BAND
    for exp_id, band in sorted(image_map.items()):
        if str(band) == target_band:
            return int(exp_id)
    return int(min(image_map.keys()))


def _tree_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(p.stat().st_size) for p in path.rglob("*") if p.is_file())


def _stage_prepared_cache(source: str | os.PathLike, target: str | os.PathLike) -> int:
    """Copy the prepared cache (a directory of meta + array sidecars) to
    node-local storage. A same-size existing target is treated as current."""
    source_path = Path(source)
    target_path = Path(target)
    # stat() raises on transient filesystem errors; is_dir() would silently
    # return False and route a directory into the single-file copy branch
    source_is_dir = stat_module.S_ISDIR(source_path.stat().st_mode)
    source_size = _tree_size(source_path)
    if target_path.exists() and _tree_size(target_path) == source_size:
        return source_size

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(f".stage.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if source_is_dir:
        shutil.copytree(source_path, temporary)
    else:
        shutil.copy2(source_path, temporary)
    if target_path.exists():
        shutil.rmtree(target_path) if target_path.is_dir() else target_path.unlink()
    os.replace(temporary, target_path)
    return source_size


def _stage_prepared_cache_from_environment(
    distributed_context, wandb_run: WandbRun | None,
) -> None:
    source = os.environ.get("SURVEY_FUSION_PREPARED_SOURCE")
    target = os.environ.get("SURVEY_FUSION_PREPARED_TARGET")
    if bool(source) != bool(target):
        raise RuntimeError(
            "SURVEY_FUSION_PREPARED_SOURCE and SURVEY_FUSION_PREPARED_TARGET must be set together."
        )
    if not source:
        return

    payload = {"ok": True, "size_bytes": 0, "elapsed_sec": 0.0, "error": None}
    if distributed_context.is_primary:
        started = time.perf_counter()
        try:
            payload["size_bytes"] = _stage_prepared_cache(source, target)
            payload["elapsed_sec"] = time.perf_counter() - started
        except OSError as exc:
            payload = {
                "ok": False,
                "size_bytes": 0,
                "elapsed_sec": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if distributed_context.enabled:
        payload = broadcast_object(payload, distributed_context)
    if not bool(payload["ok"]):
        raise RuntimeError(f"Prepared-cache staging failed: {payload['error']}")
    synchronize_distributed(distributed_context)
    if distributed_context.is_primary and wandb_run:
        wandb_run.log(
            {
                "system/prepared_cache_stage_sec": float(payload["elapsed_sec"]),
                "system/prepared_cache_size_bytes": int(payload["size_bytes"]),
            },
            step=0,
        )


def _prepared_store_key(
    data_cfg: dict, bandpasses: dict, *, seed: int, holdout_band: str | None,
) -> tuple[str | None, Path | None, str | None]:
    """(digest, cache file, resolved manifest path) of the prepared store, or Nones without a cache."""
    if not bool(data_cfg.get("prepared_cache", True)):
        return None, None, None
    observation_root = data_cfg.get("observation_root")
    prepared_dir = data_cfg.get("prepared_dir") or (
        os.path.join(str(observation_root), "_prepared") if observation_root else None
    )
    if prepared_dir is None:
        return None, None, None
    manifest_path = data_cfg.get("scene_manifest_path")
    if manifest_path and observation_root and not os.path.isabs(str(manifest_path)):
        manifest_path = os.path.join(str(observation_root), str(manifest_path))
    digest = prepared_digest(
        data_cfg, seed=int(seed), bands=list(bandpasses.keys()), manifest_path=manifest_path,
        holdout_band=holdout_band,
    )
    return digest, prepared_path(prepared_dir, digest), manifest_path


def prepare_scene_and_datums(
    data_cfg: dict, bandpasses: dict, *, trainer_cfg: dict, seed: int,
    device: torch.device, holdout_band: str | None = None,
    require_prepared_cache: bool = False,
):
    """Load or build the scene and datum store.

    Scene parsing and datum building are pure functions of the data config; at
    COSMOS scale they cost tens of minutes per launch. Reuse the prepared cache
    when one matches, build and save it when none does. scripts/prepare_data.py
    calls this from a CPU job so GPU runs start on a warm cache.
    """
    digest, cache_file, manifest_path = _prepared_store_key(
        data_cfg, bandpasses, seed=seed, holdout_band=holdout_band,
    )
    expected_digest = data_cfg.get("expected_prepared_digest")
    if expected_digest is not None:
        if cache_file is None:
            raise ValueError(
                "data.expected_prepared_digest needs the prepared cache: set data.prepared_dir "
                "or data.observation_root and keep data.prepared_cache on."
            )
        if digest != str(expected_digest):
            raise RuntimeError(
                f"Prepared store mismatch: this data config keys store {digest}, but "
                f"data.expected_prepared_digest is {expected_digest}. A data key changed."
            )

    prepared = (
        load_prepared(
            cache_file, data_cfg=data_cfg, seed=int(seed),
            bands=list(bandpasses.keys()), manifest_path=manifest_path,
            holdout_band=holdout_band,
        )
        if cache_file is not None else None
    )
    if prepared is not None:
        scene, data_store = prepared
        print(f"[prepared] loaded scene+datums from {cache_file}", flush=True)
        return scene, data_store
    if require_prepared_cache:
        raise RuntimeError(
            f"This run requires a readable prepared cache at {cache_file} (distributed "
            "training, or data.expected_prepared_digest is set). Run scripts/prepare_data.py first."
        )

    scene = load_sdss_scene(data_cfg, bandpasses)
    data_store = build_sdss_datums(
        scene,
        bandpasses,
        psf_fwhm_arcsec=data_cfg.get("psf_fwhm_arcsec"),
        spectral_response_mode=str(data_cfg.get("spectral_response_mode", "metadata")),
        spectral_lsf_sampling=str(data_cfg.get("spectral_lsf_sampling", "native")),
        spectral_response_missing_policy=str(data_cfg.get("spectral_response_missing_policy", "error")),
        pointing_offsets_arcsec=data_cfg.get("pointing_offsets_arcsec"),
        rng_seed=int(seed),
        device=device,
        holdout_band=holdout_band,
        holdout_region_cfg=data_cfg.get("holdout_region"),
        random_holdout_cfg=data_cfg.get("random_holdout"),
        region_metrics_cfg=data_cfg.get("region_metrics"),
        validate_kernel_samples=bool(trainer_cfg.get("validate_kernel_samples", True)),
        spill_dir=data_cfg.get("datum_spill_dir"),
    )
    if cache_file is not None:
        print(f"[prepared] saving scene+datums to {cache_file}", flush=True)
        try:
            save_prepared(
                cache_file, scene, data_store, data_cfg=data_cfg, seed=int(seed),
                bands=list(bandpasses.keys()), manifest_path=manifest_path,
                holdout_band=holdout_band, expected_digest=digest,
            )
        except (OSError, OverflowError, pickle.PicklingError, AttributeError) as exc:
            # a failed cache write must cost the NEXT run its warm start, not
            # kill THIS run after a completed 30-minute build
            print(f"[prepared] cache save failed ({exc!r}); continuing without cache.", flush=True)
    return scene, data_store


def _resolve_excluded_image_exposure_ids(
    data_cfg: Dict[str, Any], trainer_cfg: Dict[str, Any], data_store,
) -> tuple[int, ...]:
    """Map trainer.exclude_exposure_labels to store exposure ids.

    Exposure id equals the entry's position in the manifest's image_exposures
    list (the adapter enumerates in manifest order), so labels are resolved by
    re-reading the manifest rather than widening the prepared-store schema.
    Each resolved id is cross-checked against the store's band map to catch
    manifest/store order drift.
    """
    labels = [str(label) for label in (trainer_cfg.get("exclude_exposure_labels") or [])]
    if not labels:
        return ()
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"trainer.exclude_exposure_labels contains duplicates: {labels}."
        )
    manifest_value = data_cfg.get("scene_manifest_path")
    if not manifest_value:
        raise ValueError(
            "trainer.exclude_exposure_labels requires data.scene_manifest_path."
        )
    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = Path(str(data_cfg.get("observation_root", "."))) / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("image_exposures") or []
    hits_by_label: Dict[str, list[tuple[int, str]]] = {}
    for idx, item in enumerate(entries):
        band = str(item.get("band", "")).strip()
        label = str(item.get("label", f"{band}-exp{idx + 1:04d}"))
        hits_by_label.setdefault(label, []).append((idx, band))
    excluded_ids: list[int] = []
    for label in labels:
        hits = hits_by_label.get(label)
        if not hits:
            raise ValueError(
                f"trainer.exclude_exposure_labels: {label!r} is not an "
                f"image_exposures label in {manifest_path}."
            )
        if len(hits) > 1:
            raise ValueError(
                f"trainer.exclude_exposure_labels: {label!r} is ambiguous in "
                f"{manifest_path} (manifest indices {[h[0] for h in hits]})."
            )
        exp_id, band = hits[0]
        store_band = getattr(data_store, "image_band_by_exposure", {}).get(exp_id)
        if store_band is None or str(store_band) != band:
            raise RuntimeError(
                f"Manifest/store exposure order drift: manifest entry {exp_id} "
                f"({label!r}, band {band!r}) maps to store band {store_band!r}. "
                "Refusing to exclude by a misaligned id."
            )
        excluded_ids.append(exp_id)
        print(
            f"[holdout] resolved excluded exposure {label!r} -> id {exp_id} "
            f"(band {band})",
            flush=True,
        )
    return tuple(sorted(excluded_ids))


def _resume_discards_seeded_model(cfg: DictConfig, *, project_root: Path) -> bool:
    """True when a resume overwrites everything seeding writes and restores the
    RNG streams seeding advances.

    Seeding costs a minute and a half on full84 and the checkpoint's
    model_state overwrites every parameter it produced, so the work is wasted.
    Dropping it is only safe when the resume also restores the RNG state: the
    peak search draws from the global torch stream, and a resume that instead
    replays the original draw prefix would, with a shorter prefix, go on to
    sample different pixels.
    """
    resume_path = _resolve_resume_path(cfg, project_root)
    if resume_path is None or not bool(cfg.trainer.get("checkpoint_rng_state", False)):
        return False
    checkpoint = torch.load(
        resume_path, map_location="cpu", mmap=True, weights_only=False
    )
    return RNG_STATE_KEY in checkpoint and "model_state" in checkpoint


def _build_trainer_components(
    cfg: DictConfig, *, paths: Dict[str, Path], device: torch.device, precision_policy: str,
    distributed_context=None, platform_cfg: Dict[str, Any] | None = None, wandb_run: WandbRun | None = None,
    speed_logger: SpeedLogger | None = None, wandb_log_every_n_steps: int | None = None,
    epoch_log_every: int | None = None, wrap_distributed_model: bool = False,
    require_prepared_cache: bool = False,
) -> tuple:
    bandpasses = load_bandpasses(paths["registry_path"], cfg.physics["bandpasses"])
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True, throw_on_missing=True)
    data_cfg = data_cfg if isinstance(data_cfg, dict) else {}
    data_name = str(data_cfg.get("name", "sdss")).lower()
    if data_name != "sdss":
        raise ValueError(f"Unknown data.name {data_name!r}; expected 'sdss'.")

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True, throw_on_missing=True)
    trainer_cfg = trainer_cfg if isinstance(trainer_cfg, dict) else {}
    random_holdout_cfg = data_cfg.get("random_holdout", {})
    if isinstance(random_holdout_cfg, dict):
        trainer_cfg["random_holdout"] = dict(random_holdout_cfg)
    calibration_cfg = OmegaConf.to_container(cfg.get("calibration", {}), resolve=True)
    if isinstance(calibration_cfg, dict):
        trainer_cfg["calibration"] = calibration_cfg
    if distributed_context is not None:
        batch_mode = str(
            ((platform_cfg or {}).get("distributed", {}) or {}).get("batch_mode", "per_gpu")
            or "per_gpu"
        )
        for key in ("pixel_batch_size", "spec_batch_size"):
            trainer_cfg[key] = resolve_distributed_batch_size(
                int(trainer_cfg.get(key, 0) or 0), distributed_context, mode=batch_mode,
                label=f"trainer.{key}",
            )
    if int(trainer_cfg.get("kernel_samples_k", 0) or 0) <= 0:
        trainer_kernel_samples = int(data_cfg.get("kernel_samples_k", 0) or 0)
        if trainer_kernel_samples > 0:
            trainer_cfg["kernel_samples_k"] = trainer_kernel_samples

    holdout_band = _resolve_holdout_band(cfg)
    poisson_current = poisson_current_requested(trainer_cfg, data_cfg)
    store_seed = prepare_seed(data_cfg)
    scene, data_store = prepare_scene_and_datums(
        data_cfg, bandpasses, trainer_cfg=trainer_cfg, seed=store_seed, device=device,
        holdout_band=holdout_band,
        # A pinned store is loaded, never built: five runs building a missing
        # 107 GB store at once is how SIGBUS deaths happen. scripts/prepare_data.py
        # builds it.
        require_prepared_cache=bool(
            require_prepared_cache
            or (distributed_context is not None and distributed_context.enabled)
            or data_cfg.get("expected_prepared_digest") is not None
        ),
    )
    store_digest, _, _ = _prepared_store_key(
        data_cfg, bandpasses, seed=store_seed, holdout_band=holdout_band,
    )
    if trainer_cfg.get("image_pixel_mask") is not None:
        check_sidecar_binding(
            trainer_cfg["image_pixel_mask"], store_digest=store_digest,
            image_labels={
                exp_id: exposure.label
                for exp_id, exposure in enumerate(getattr(scene, "image_exposures", None) or [])
            },
        )
    primary_exp_id = _primary_image_exposure_id(data_store, holdout_band)
    primary_exposure = data_store.exposures[primary_exp_id]
    height, width = primary_exposure.fiber_mask.shape
    excluded_exposure_ids = _resolve_excluded_image_exposure_ids(
        data_cfg, trainer_cfg, data_store
    )
    model = _init_model(cfg)
    # Seed on CPU before the device move: seeding reads the store's numpy
    # source images and writes straight into the parameter tensors.
    model_cfg_container = OmegaConf.to_container(cfg.model, resolve=True)
    seed_catalog = model_cfg_container.get("seed_catalog")
    if isinstance(model, SplatField) and _resume_discards_seeded_model(
        cfg, project_root=paths["project_root"]
    ):
        print(
            "[survey_fusion] resume restores model_state and the RNG streams; "
            "skipping splat seeding",
            flush=True,
        )
    elif isinstance(model, SplatField) and seed_catalog:
        catalog_path = resolve_path(str(seed_catalog), paths["project_root"])
        n_catalog_sources = seed_from_catalog(
            model, data_store, catalog_path,
            exclude_exposure_ids=excluded_exposure_ids,
        )
        print(
            f"[survey_fusion] seeded splats from external catalog {catalog_path} "
            f"({n_catalog_sources} sources)",
            flush=True,
        )
    elif isinstance(model, SplatField) and bool(
        model_cfg_container.get("seed_from_peaks", True)
    ):
        seed_n_scales = int(model_cfg_container.get("seed_n_scales", 4) or 4)
        seed_prefer_band = model_cfg_container.get("seed_prefer_band", None)
        seed_prefer_band = (
            None if seed_prefer_band is None else str(seed_prefer_band)
        )
        n_seed_exposures = seed_from_store(
            model, data_store, prefer_band=seed_prefer_band,
            n_scales=seed_n_scales,
            exclude_exposure_ids=excluded_exposure_ids,
        )
        seed_mode = (
            "multiband peaks" if seed_prefer_band is None
            else f"{seed_prefer_band}-band peaks"
        )
        print(
            f"[survey_fusion] seeded splats from {seed_mode} across "
            f"{n_seed_exposures} exposures "
            f"(n_scales={seed_n_scales})",
            flush=True,
        )
    model = model.to(device)
    if (
        wrap_distributed_model
        and distributed_context is not None
        and distributed_context.enabled
        and _should_wrap_model_for_distributed(model)
    ):
        model = _wrap_model_for_distributed(model, device)
    variance_columns = image_variance_columns(scene, data_store) if poisson_current else None
    trainer = DatumTrainer(
        model=model,
        grid=build_spatial_grid(height, width, device),
        data=data_store,
        training_bands=list(cfg.physics.get("training_bands", [])),
        held_out_band=holdout_band,
        lambda_step=float(cfg.physics.get("lambda_step_angstrom", 20.0)),
        trainer_cfg=trainer_cfg,
        device=device,
        precision_policy=precision_policy,
        distributed_context=distributed_context,
        wandb_logger=wandb_run,
        speed_logger=speed_logger,
        wandb_log_every_n_steps=(1 if wandb_log_every_n_steps is None else wandb_log_every_n_steps),
        epoch_log_every=epoch_log_every,
        exclude_image_exposure_ids=excluded_exposure_ids,
        image_variance_columns=variance_columns,
        prepared_store=(
            None if store_digest is None else {"digest": store_digest, "prepare_seed": store_seed}
        ),
    )
    return bandpasses, data_name, scene, data_store, trainer, int(primary_exp_id), primary_exposure


def _requested_exposure_set(exposure_ids: list[int] | tuple[int, ...] | set[int] | None,) -> set[int] | None:
    return None if exposure_ids is None else {int(exp_id) for exp_id in exposure_ids}


def _nested_images_from_exposure_map(
    data_store, images_by_exposure, requested: set[int] | None,
) -> Dict[int, Dict[str, np.ndarray]]:
    nested: Dict[int, Dict[str, np.ndarray]] = {}
    for exp_id, band in getattr(data_store, "image_band_by_exposure", {}).items():
        exp_id_int = int(exp_id)
        if requested is not None and exp_id_int not in requested:
            continue
        if exp_id_int in images_by_exposure:
            nested[exp_id_int] = {str(band): np.asarray(images_by_exposure[exp_id_int], dtype=np.float32)}
    return nested


def _nested_materialized_images(
    data_store, materialize, requested: set[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    return {
        int(exp_id): {str(band): np.asarray(materialize(int(exp_id)), dtype=np.float32)}
        for exp_id, band in getattr(data_store, "image_band_by_exposure", {}).items()
        if int(exp_id) in requested
    }


def _nested_images_from_scene_bands(
    data_store, obs_src, requested: set[int] | None,
) -> Dict[int, Dict[str, np.ndarray]]:
    nested: Dict[int, Dict[str, np.ndarray]] = {}
    for exp_id, band in getattr(data_store, "image_band_by_exposure", {}).items():
        exp_id_int = int(exp_id)
        if requested is not None and exp_id_int not in requested:
            continue
        if band in obs_src:
            nested[exp_id_int] = {str(band): obs_src[str(band)]}
    return nested


def _nested_observations(
    scene, data_store, *, use_flux_density: bool,
    exposure_ids: list[int] | tuple[int, ...] | set[int] | None = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    requested = _requested_exposure_set(exposure_ids)
    total_by_exposure = getattr(data_store, "image_total_by_exposure", {})
    if use_flux_density and total_by_exposure:
        return _nested_images_from_exposure_map(data_store, total_by_exposure, requested)
    materialize = getattr(data_store, "materialize_image_total", None)
    if use_flux_density and callable(materialize) and requested is not None:
        return _nested_materialized_images(data_store, materialize, requested)
    obs_src = (
        scene.images_flambda if use_flux_density and hasattr(scene, "images_flambda") else scene.images
    )
    return _nested_images_from_scene_bands(data_store, obs_src, requested)


def _nested_source_observations(
    scene, data_store, *, use_flux_density: bool,
    exposure_ids: list[int] | tuple[int, ...] | set[int] | None = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    requested = _requested_exposure_set(exposure_ids)
    source_by_exposure = getattr(data_store, "image_source_by_exposure", {})
    image_band_map = getattr(data_store, "image_band_by_exposure", {})
    bands = [str(band) for band in image_band_map.values()]
    source_images = getattr(scene, "image_source_flambda", None)
    prefer_store_source = bool(source_by_exposure) and (len(set(bands)) < len(bands) or not source_images)
    if use_flux_density and prefer_store_source:
        return _nested_images_from_exposure_map(data_store, source_by_exposure, requested)
    materialize = getattr(data_store, "materialize_image_source", None)
    if use_flux_density and callable(materialize) and requested is not None:
        return _nested_materialized_images(data_store, materialize, requested)
    if use_flux_density and source_images is not None:
        return _nested_images_from_scene_bands(data_store, source_images, requested)
    return _nested_images_from_exposure_map(data_store, source_by_exposure, requested)


def _image_exposure_ids_for_band(data_store, band: str) -> list[int]:
    band_str = str(band)
    return sorted(
        int(exp_id)
        for exp_id, exp_band in getattr(data_store, "image_band_by_exposure", {}).items()
        if str(exp_band) == band_str
    )


def _flatten_nested_images(nested: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    flat: Dict[str, np.ndarray] = {}
    for exp_id, band_map in nested.items():
        for band, image in band_map.items():
            flat[f"e{int(exp_id)}__{band}"] = image
    return flat


def _load_nested_images_npz(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction NPZ not found: {path}")
    nested: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            if not key.startswith("e") or "__" not in key:
                raise ValueError(
                    f"Prediction NPZ key {key!r} does not match expected 'e<exposure_id>__<band>'."
                )
            raw_exp_id, band = key.split("__", 1)
            try:
                exp_id = int(raw_exp_id[1:])
            except ValueError as exc:
                raise ValueError(f"Prediction NPZ key {key!r} has invalid exposure id.") from exc
            if not band:
                raise ValueError(f"Prediction NPZ key {key!r} has empty band label.")
            nested.setdefault(exp_id, {})[str(band)] = np.asarray(data[key], dtype=np.float32)
    if not nested:
        raise ValueError(f"Prediction NPZ {path} did not contain any images.")
    return nested


def _prediction_npz_path(
    *, run_dir: Path, output: str, eval_cfg: object | None, project_root: Path | None = None,
) -> Path:
    if output == "total":
        raw_path = _cfg_get(eval_cfg, "predictions_path", None)
        return resolve_path(str(raw_path), project_root) if raw_path else run_dir / "predictions.npz"
    raw_path = _cfg_get(eval_cfg, f"{output}_predictions_path", None)
    if raw_path:
        return resolve_path(str(raw_path), project_root)
    total_raw_path = _cfg_get(eval_cfg, "predictions_path", None)
    if total_raw_path:
        return resolve_path(str(total_raw_path), project_root).with_name(f"predictions_{output}.npz")
    return run_dir / f"predictions_{output}.npz"


def _load_eval_prediction_overrides(
    *, run_dir: Path, postprocess_cfg: object | None, eval_cfg: object | None,
    project_root: Path | None = None,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]] | None:
    if bool(_cfg_get(eval_cfg, "recompute_predictions", True)):
        return None
    overrides: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for output in _resolve_diagnostic_outputs(postprocess_cfg):
        path = _prediction_npz_path(
            run_dir=run_dir, output=output, eval_cfg=eval_cfg, project_root=project_root,
        )
        overrides[output] = _load_nested_images_npz(path)
    return overrides


def _merge_nested_images_for_plot(
    observations: Dict[int, Dict[str, np.ndarray]], predictions: Dict[int, Dict[str, np.ndarray]],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    obs_plot: Dict[str, np.ndarray] = {}
    pred_plot: Dict[str, np.ndarray] = {}
    used_labels: set[str] = set()
    for exp_id in sorted(observations.keys()):
        obs_by_band = observations.get(exp_id, {})
        pred_by_band = predictions.get(exp_id, {})
        for band in sorted(obs_by_band.keys()):
            if band not in pred_by_band:
                continue
            label = band
            if label in used_labels:
                label = f"e{int(exp_id)}:{band}"
            used_labels.add(label)
            obs_plot[label] = obs_by_band[band]
            pred_plot[label] = pred_by_band[band]
    return obs_plot, pred_plot


def _region_grid_exposure_id(
    *, observations: Dict[int, Dict[str, np.ndarray]], predictions: Dict[int, Dict[str, np.ndarray]],
    image_band_by_exposure: Dict[int, str], region_masks_by_exposure: Dict[int, Dict[str, np.ndarray]],
    fallback_exp_id: int,
) -> int:
    best_exp_id = int(fallback_exp_id)
    best_count = -1
    for exp_id, pred_by_band in predictions.items():
        exp_id_int = int(exp_id)
        band = str(image_band_by_exposure.get(exp_id_int, ""))
        obs = observations.get(exp_id_int, {}).get(band)
        if obs is None or band not in pred_by_band:
            continue
        masks = region_masks_by_exposure.get(exp_id_int, {})
        count = 0
        finite_obs = np.isfinite(np.asarray(obs))
        for mask in masks.values():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape == finite_obs.shape:
                count += int(np.count_nonzero(finite_obs & mask_arr))
        if count > best_count:
            best_count = count
            best_exp_id = exp_id_int
    return best_exp_id


def _compute_image_diagnostic_predictions(
    trainer: DatumTrainer, *, postprocess_cfg: object | None = None, include_additive: bool = True,
    profile: Dict[str, Any] | None = None, required_exposure_ids: list[int] | tuple[int, ...] | None = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    return image_diagnostics.compute_image_diagnostic_predictions(
        trainer, postprocess_cfg=postprocess_cfg, include_additive=bool(include_additive), profile=profile,
        required_exposure_ids=required_exposure_ids,
    )



def _build_diagnostic_output_bundles(
    *, trainer, scene, data_store, observations, diagnostic_outputs, primary_exp_id, postprocess_cfg, profile,
    prediction_overrides,
):
    predictions_by_output = {}
    for output in diagnostic_outputs:
        image_profile = {} if profile is not None else None
        started = time.perf_counter()
        if output in prediction_overrides:
            predictions_by_output[output] = {
                int(exp_id): {
                    str(band): np.asarray(image, dtype=np.float32) for band, image in band_map.items()
                }
                for exp_id, band_map in prediction_overrides[output].items()
            }
            if profile is not None and image_profile is not None:
                image_profile["source"] = "npz"
        else:
            predictions_by_output[output] = _compute_image_diagnostic_predictions(
                trainer, postprocess_cfg=postprocess_cfg, include_additive=(output == "total"),
                profile=image_profile, required_exposure_ids=[int(primary_exp_id)],
            )
        timing_key = "image_predictions" if output == "total" else f"image_{output}_predictions"
        _record_profile_timing(profile, timing_key, started)
        if profile is not None and image_profile is not None:
            profile[timing_key] = image_profile

    diagnostic_exposure_ids = sorted(
        int(exp_id)
        for output_predictions in predictions_by_output.values()
        for exp_id in output_predictions.keys()
    )
    missing_observations = [
        int(exp_id) for exp_id in diagnostic_exposure_ids if int(exp_id) not in observations
    ]
    if missing_observations:
        observations = dict(observations)
        observations.update(
            _nested_observations(
                scene, data_store, use_flux_density=trainer.use_flux_density,
                exposure_ids=missing_observations,
            )
        )
    source_observations = (
        _nested_source_observations(
            scene, data_store, use_flux_density=trainer.use_flux_density,
            exposure_ids=diagnostic_exposure_ids,
        )
        if "source" in diagnostic_outputs
        else {}
    )
    observations_by_output = {"total": observations, "source": source_observations}
    return (
        {
            output: SimpleNamespace(
                output=output, suffix=_diagnostic_suffix(output), predictions=predictions_by_output[output],
                observations=observations_by_output[output], band_metrics=None, region_metrics=None,
            )
            for output in diagnostic_outputs
        },
        observations,
        diagnostic_exposure_ids,
    )


def _save_diagnostic_prediction_outputs(*, run_dir, bundles, profile) -> None:
    started = time.perf_counter()
    for bundle in bundles.values():
        output_path = run_dir / (
            "predictions.npz"
            if bundle.output == "total"
            else f"predictions_{bundle.output}.npz"
        )
        np.savez(output_path, **_flatten_nested_images(bundle.predictions))
    _record_profile_timing(profile, "save_predictions_npz", started)
    if "total" in bundles:
        print(f"Saved total predictions to {run_dir / 'predictions.npz'}")
    if "source" in bundles:
        print(f"Saved source predictions to {run_dir / 'predictions_source.npz'}")


def _holdout_masks_by_exposure(
    observations: Dict[int, Dict[str, np.ndarray]], excluded_local_indices_by_exposure,
) -> Dict[int, np.ndarray]:
    holdout_masks: Dict[int, np.ndarray] = {}
    if not isinstance(excluded_local_indices_by_exposure, dict):
        return holdout_masks
    for exp_id, obs_by_band in observations.items():
        if int(exp_id) not in excluded_local_indices_by_exposure or not obs_by_band:
            continue
        image = next(iter(obs_by_band.values()))
        mask = np.zeros_like(image, dtype=bool).reshape(-1)
        holdout_local = np.asarray(excluded_local_indices_by_exposure[int(exp_id)], dtype=np.int64)
        valid = (holdout_local >= 0) & (holdout_local < mask.size)
        mask[holdout_local[valid]] = True
        holdout_masks[int(exp_id)] = mask.reshape(image.shape)
    return holdout_masks


def _maybe_log_periodic_region_grid(
    *, step: int, every_n_steps: int, trainer: DatumTrainer, wandb_run: WandbRun | None, primary_exp_id: int,
    held_out_band: str | None, observations_primary: Dict[str, np.ndarray],
    region_masks_primary: Dict[str, np.ndarray], region_overlays_primary: Dict[str, Dict[str, float | str]],
    primary_region: str | None, pad_pixels: int, run_dir: Path,
    scheduled_steps: Collection[int] = (),
) -> None:
    if not _visualization_step_due(
        step=step,
        every_n_steps=every_n_steps,
        scheduled_steps=scheduled_steps,
    ):
        return
    if not wandb_run:
        return
    if not held_out_band:
        return
    if not isinstance(region_masks_primary, dict) or not region_masks_primary:
        return
    observed = observations_primary.get(held_out_band)
    if observed is None:
        return
    predictions = _compute_image_diagnostic_predictions(trainer)
    predictions_primary = predictions.get(int(primary_exp_id), {})
    predicted = predictions_primary.get(held_out_band)
    if predicted is None:
        return
    visualization.plot_region_cutout_grid(
        observed, predicted, region_masks_primary, held_out_band, run_dir, primary_region=primary_region,
        pad_pixels=pad_pixels, region_overlays=region_overlays_primary,
        filename=f"region_grid_e{int(primary_exp_id)}_{held_out_band}.png",
    )
    image_path = run_dir / f"region_grid_e{int(primary_exp_id)}_{held_out_band}.png"
    wandb_run.log_images({f"region_grid_e{int(primary_exp_id)}_{held_out_band}": image_path}, step=step)


def _maybe_log_model_cutouts(
    *, step: int, every_n_steps: int, trainer: DatumTrainer,
    wandb_run: WandbRun | None, run_dir: Path, size: int = 512,
    scheduled_steps: Collection[int] = (),
) -> None:
    """Periodic data, model, and residual cutouts, one exposure per band.

    Unlike the region grid this needs no held-out band and no region metrics,
    so it works on every config; on full COSMOS the region grid's guards all
    fail and the run is otherwise invisible on wandb.
    """
    if (
        not _visualization_step_due(
            step=step,
            every_n_steps=every_n_steps,
            scheduled_steps=scheduled_steps,
        )
        or not wandb_run
    ):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp_by_band: Dict[str, int] = {}
    for e, b in trainer.image_band_by_exposure.items():
        if int(e) in trainer.exclude_image_exposure_ids:
            continue
        exp_by_band.setdefault(str(b), int(e))
    bands = sorted(exp_by_band)
    fig, axes = plt.subplots(3, len(bands), figsize=(3.2 * len(bands), 9.8), squeeze=False)
    for j, band in enumerate(bands):
        exp_id = exp_by_band[band]
        exposure = trainer.runtime_by_exposure[exp_id].exposure
        h, w = int(exposure.image_height), int(exposure.image_width)
        s = min(size, h, w)
        r0, c0 = (h - s) // 2, (w - s) // 2
        model_img = trainer.render_datum_image_window_deterministic(
            exp_id,
            y0=r0,
            y1=r0 + s,
            x0=c0,
            x1=c0 + s,
            include_additive=False,
        )
        data_img = np.asarray(
            trainer.data.materialize_image_by_exposure(exp_id, "source"), dtype=np.float32
        )[r0 : r0 + s, c0 : c0 + s]

        med = float(np.nanmedian(data_img))
        mad = float(np.nanmedian(np.abs(data_img - med)) * 1.4826) or 1e-6
        stretch = lambda a: np.arcsinh((a - med) / (2 * mad))  # noqa: E731
        vmax = float(np.nanpercentile(stretch(data_img), 99.5))
        for row, img, label in (
            (0, data_img, "data"),
            (1, model_img, "model"),
            (2, data_img - model_img, "residual"),
        ):
            ax = axes[row][j]
            if label == "residual":
                ax.imshow(
                    stretch(img),
                    vmin=-0.6 * vmax,
                    vmax=0.6 * vmax,
                    cmap="RdBu_r",
                    origin="lower",
                )
            else:
                ax.imshow(stretch(img), vmin=-1, vmax=vmax, cmap="magma", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(band, fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=11)
    fig.suptitle(f"step {step}", fontsize=12)
    fig.tight_layout()
    image_path = run_dir / "model_cutouts.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    wandb_run.log_images({"model_cutouts": image_path}, step=step)


def _resolve_resume_path(cfg: DictConfig, project_root: Path) -> Path | None:
    resume_cfg = cfg.get("resume", {})
    resume_path: Any = None
    if isinstance(resume_cfg, DictConfig):
        resume_path = resume_cfg.get("path")
    elif isinstance(resume_cfg, dict):
        resume_path = resume_cfg.get("path")
    if resume_path is None:
        return None
    value = str(resume_path).strip()
    if value == "" or value.lower() == "none":
        return None
    return resolve_path(value, project_root)


def _validate_current_checkpoint_schema(
    checkpoint: Dict[str, object], *, checkpoint_path: Path, context: str,
) -> None:
    version = checkpoint.get("checkpoint_version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"{context} at {checkpoint_path} must have checkpoint_version in "
            f"{SUPPORTED_CHECKPOINT_VERSIONS}, got {version!r}."
        )
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{context} at {checkpoint_path} is missing required checkpoint key(s): {joined}."
        )
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"{context} at {checkpoint_path} is missing required model_state mapping.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError(f"{context} at {checkpoint_path} is missing required model_config mapping.")
    resume_meta = checkpoint.get("resume_meta")
    if not isinstance(resume_meta, Mapping):
        raise ValueError(f"{context} at {checkpoint_path} is missing required resume_meta mapping.")


def _load_checkpoint(path: Path, device: torch.device, *, context: str = "Checkpoint") -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{context} not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{context} at {path} must be a dictionary, got {type(checkpoint).__name__}.")
    _validate_current_checkpoint_schema(checkpoint, checkpoint_path=path, context=context)
    return checkpoint


def _load_resume_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    return _load_checkpoint(path, device, context="Resume checkpoint")


def _resume_model_only(cfg: DictConfig) -> bool:
    resume_cfg = cfg.get("resume", {})
    if isinstance(resume_cfg, (DictConfig, dict)):
        return bool(resume_cfg.get("model_only", False))
    return False


def _load_resume_state_into_trainer(
    trainer: DatumTrainer, checkpoint: Dict[str, object], *, checkpoint_path: Path,
    model_only: bool = False, allow_operator_change: bool = False,
) -> None:
    _validate_current_checkpoint_schema(
        checkpoint, checkpoint_path=checkpoint_path, context="Resume checkpoint",
    )
    if checkpoint_is_gauge_canonical(checkpoint):
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} is a gauge-canonical sidecar. "
            "Its raw_flux/SED split was rewritten for offline analysis and it "
            "carries no optimizer moments; resume from the run's model.pt."
        )
    version = checkpoint["checkpoint_version"]
    print(f"Resuming from checkpoint: {checkpoint_path}")
    print(f"Resume checkpoint version: {version}")

    resume_meta = checkpoint["resume_meta"]
    global_step = checkpoint_absolute_step(
        checkpoint, checkpoint_path=checkpoint_path,
    )
    if "global_step_completed" not in resume_meta:
        legacy_step_field = (
            "absolute_step"
            if "absolute_step" in resume_meta
            else "steps_ran_this_invocation"
        )
        print(
            "Warning: migrating legacy resume step field "
            f"{legacy_step_field}={global_step} to canonical "
            "global_step_completed semantics."
        )
    print(
        "Resume metadata: "
        f"epochs_ran={resume_meta.get('epochs_ran_this_invocation')}, "
        f"global_step={global_step}"
    )

    validate_checkpoint_model_config(
        trainer.model, checkpoint, checkpoint_path=checkpoint_path, context="Resume",
    )
    validate_checkpoint_operator_config(
        checkpoint,
        current_operator_config=operator_config_for_trainer(trainer),
        context="Resume",
        checkpoint_path=checkpoint_path,
        resume=True,
        allow_psf_model_change=bool(getattr(trainer, "allow_psf_model_change", False)),
        allow_operator_change=allow_operator_change,
    )
    operator_config = checkpoint.get("operator_config")
    checkpoint_excluded = (
        operator_config.get("exclude_exposure_labels")
        if isinstance(operator_config, Mapping)
        else None
    )
    current_excluded = sorted(getattr(trainer, "exclude_exposure_labels", []) or [])
    if checkpoint_excluded is None:
        if current_excluded:
            print(
                "Warning: resume checkpoint has no exposure-holdout provenance "
                "(pre-holdout checkpoint); it may already contain the exposures "
                f"now excluded ({current_excluded}). The resumed model does NOT "
                "satisfy a novel-view holdout claim for them."
            )
    elif sorted(str(label) for label in checkpoint_excluded) != current_excluded:
        if allow_operator_change:
            print(
                "resume.allow_operator_change=true: continuing a checkpoint with "
                f"exposure holdout {sorted(checkpoint_excluded)} under "
                f"{current_excluded}."
            )
        else:
            raise ValueError(
                f"Resume checkpoint {checkpoint_path} was trained with "
                f"exclude_exposure_labels={sorted(checkpoint_excluded)} but the "
                f"current trainer has {current_excluded}. A model that already "
                "saw an exposure can never un-see it, so silently changing the "
                "holdout set breaks the novel-view claim. Match "
                "trainer.exclude_exposure_labels to the checkpoint, or pass "
                "+resume.allow_operator_change=true to accept the change "
                "explicitly."
            )
    _migrate_checkpoint_model_state(checkpoint)
    _unwrap_model(trainer.model).load_state_dict(checkpoint["model_state"])
    print("Loaded model_state from resume checkpoint.")

    start_step = int(global_step)
    trainer.resume_start_step = start_step
    print(
        f"Resuming at global step {start_step}: lr decay and refine schedules "
        "continue rather than replay."
    )

    calibration_state = checkpoint.get("calibration_state")
    calibration_loaded = trainer.load_calibration_state_dict(calibration_state)
    if calibration_state is not None and not calibration_loaded:
        print("Warning: resume checkpoint calibration_state could not be applied; using defaults.")
    elif calibration_state is None and trainer.get_calibration_report().get("enabled", False):
        print("Warning: resume checkpoint has no calibration_state; using unity nuisance scale.")
    elif calibration_state is not None and calibration_loaded:
        print("Loaded calibration_state from resume checkpoint.")

    if model_only:
        print(
            "resume.model_only=true: skipping optimizer_state and "
            "lr_scheduler_state; optimizer starts fresh."
        )
        return

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is None:
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} is missing required optimizer_state. "
            "Use a checkpoint written by the current training entrypoint."
        )
    try:
        trainer.load_optimizer_state_dict(optimizer_state)
    except Exception as exc:
        raise RuntimeError(
            "Resume checkpoint optimizer_state could not be loaded. "
            "Ensure model/trainer optimizer configuration matches the checkpoint."
        ) from exc
    print("Loaded optimizer_state from resume checkpoint.")

    refine_state = checkpoint.get("refine_state")
    if trainer.refine_strategy is not None:
        if refine_state is not None:
            trainer.load_refine_state_dict(refine_state)
            print("Loaded refine_state from resume checkpoint.")
        else:
            print(
                "Warning: resume checkpoint has no refine_state; the grow "
                "threshold will recalibrate from mid-training gradients and "
                "densification may diverge from the uninterrupted run."
            )
    elif refine_state is not None:
        print(
            "Warning: resume checkpoint has refine_state but refinement is "
            "disabled in this trainer; ignoring it."
        )

    scheduler_state = checkpoint.get("lr_scheduler_state")
    if trainer.lr_scheduler is not None:
        if scheduler_state is None:
            raise ValueError(
                f"Resume checkpoint {checkpoint_path} is missing required "
                "lr_scheduler_state for the current scheduler-enabled trainer."
            )
        try:
            trainer.lr_scheduler.load_state_dict(scheduler_state)
        except Exception as exc:
            raise RuntimeError(
                "Resume checkpoint lr_scheduler_state could not be loaded. "
                "Ensure trainer.lr_scheduler config matches the checkpoint."
            ) from exc
        print("Loaded lr_scheduler_state from resume checkpoint.")
    elif scheduler_state is not None:
        print(
            "Warning: resume checkpoint has lr_scheduler_state but current trainer has no "
            "scheduler enabled; ignoring scheduler state."
        )


RECIPE_KEY = "recipe"
# Trainer keys a full resume may change: the budget, checkpoint and logging
# cadence, data-loader plumbing, and machine paths whose content the recipe
# already pins by hash (mask, overlap table) or the operator config validates
# (eigen-MoG sidecar).
RESUMABLE_TRAINER_KEYS = frozenset({
    "max_steps", "epochs", "eval_interval", "band_loss_log_every_n_steps",
    "checkpoint_every_n_steps", "checkpoint_retain_every_n_steps", "checkpoint_retain_every_n_epochs",
    "checkpoint_retain_steps", "checkpoint_archive_dir", "checkpoint_keep_local",
    "checkpoint_archive_drain_timeout_s", "checkpoint_gauge_sidecar",
    "fixed_holdout_eval_every_n_steps", "fixed_holdout_max_image_per_band", "fixed_holdout_max_spec",
    "dataloader_num_workers", "dataloader_pin_memory", "validate_kernel_samples",
    "eigen_mog_sidecar", "image_pixel_mask", "field_overlap_table",
})


# What resume.allow_operator_change releases: the withheld sets and the
# populations and normalization that follow from them.
OPERATOR_CHANGE_KEYS = frozenset({
    "exclude_exposure_labels", "exclude_spectrum_labels", "objective_population", "objective_denominator",
})


def _resume_contract(record: Mapping[str, object], *, allow_operator_change: bool = False) -> Dict[str, object]:
    released = OPERATOR_CHANGE_KEYS if allow_operator_change else frozenset()
    contract = {
        key: value for key, value in record.items()
        if key not in ("trainer_config", "execution_flags") and key not in released
    }
    contract["trainer_config"] = {
        key: value for key, value in (record.get("trainer_config") or {}).items()
        if key not in RESUMABLE_TRAINER_KEYS and key not in released
    }
    contract["execution_flags"] = (record.get("execution_flags") or {}).get("resolved")
    return contract


def _validate_resume_recipe(
    trainer: DatumTrainer, checkpoint: Mapping[str, object], *, checkpoint_path: Path, model_only: bool,
    allow_operator_change: bool = False,
) -> None:
    """A full resume must continue the objective and execution it was trained under.

    A model-only resume starts a new fit from old parameters and may change
    both. Every resolved execution switch is compared: several (chunk sizes,
    band contraction, approximate exponentials) change the arithmetic.
    """
    if model_only:
        return
    saved = checkpoint.get(RECIPE_KEY)
    if saved is None:
        warnings.warn(
            f"Resume checkpoint {checkpoint_path} predates recipe provenance; its objective and "
            "data settings cannot be checked against this run.",
            RuntimeWarning, stacklevel=2,
        )
        return
    saved_contract = _resume_contract(saved, allow_operator_change=allow_operator_change)
    current_contract = _resume_contract(trainer.recipe_record(), allow_operator_change=allow_operator_change)
    differing = []
    for key in sorted(set(saved_contract) | set(current_contract)):
        before, after = saved_contract.get(key), current_contract.get(key)
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            differing += [
                f"{key}.{sub}: {before.get(sub)!r} -> {after.get(sub)!r}"
                for sub in sorted(set(before) | set(after)) if before.get(sub) != after.get(sub)
            ]
        elif before != after:
            differing.append(f"{key}: {before!r} -> {after!r}")
    if differing:
        raise RuntimeError(
            f"Resume checkpoint {checkpoint_path} was trained under a different recipe: " + "; ".join(differing)
        )


def _build_training_checkpoint(trainer: DatumTrainer, fit_state) -> Dict[str, object]:
    saved_at_utc = (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    lr_scheduler_state = None
    if trainer.lr_scheduler is not None:
        lr_scheduler_state = trainer.lr_scheduler.state_dict()
    absolute_step = int(getattr(fit_state, "steps_ran", 0))
    invocation_steps = int(
        getattr(
            fit_state,
            "steps_ran_this_invocation",
            max(0, absolute_step - int(getattr(trainer, "resume_start_step", 0))),
        )
    )
    checkpoint: Dict[str, object] = {
        "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
        "operator_config": {
            **operator_config_for_trainer(trainer),
            "exclude_exposure_labels": sorted(
                getattr(trainer, "exclude_exposure_labels", []) or []
            ),
        },
        "model_config": model_config_for_checkpoint(trainer.model),
        "model_state": _unwrap_model(trainer.model).state_dict(),
        "calibration_state": trainer.get_calibration_state_dict(),
        "optimizer_state": trainer.optimizer_state_dict(),
        "lr_scheduler_state": lr_scheduler_state,
        "refine_state": trainer.refine_state_dict(),
        RECIPE_KEY: trainer.recipe_record(),
        "resume_meta": {
            "epochs_ran_this_invocation": int(getattr(fit_state, "epochs_ran", 0)),
            "steps_ran_this_invocation": invocation_steps,
            "global_step_completed": absolute_step,
            "saved_at_utc": saved_at_utc,
        },
    }
    if getattr(trainer, "checkpoint_rng_state_enabled", False):
        checkpoint[RNG_STATE_KEY] = capture_rng_state(trainer)
    return checkpoint


def _maybe_save_gauge_canonical_sidecar(
    trainer: DatumTrainer, checkpoint: Dict[str, object], run_dir: Path, *, step: int,
) -> Path | None:
    """Write the gauge-canonical companion to model.pt, when configured.

    Local-only by design: the archiver's retention and sha bookkeeping keys on
    checkpoint_step_*.pt and must keep seeing exactly the raw checkpoints, and
    the sidecar is derived from the raw state, so archiving it would double the
    transfer for no recoverable information. One rolling file per run keeps the
    storage contract intact."""
    if not getattr(trainer, "checkpoint_gauge_sidecar_enabled", False):
        return None
    base_model = _unwrap_model(trainer.model)
    sidecar = build_gauge_canonical_sidecar(
        checkpoint,
        gauge_factors=trainer._splat_gauge_factors(base_model),
        source_step=int(step),
        source_name="model.pt",
    )
    sidecar_path = run_dir / GAUGE_SIDECAR_FILENAME
    tmp = run_dir / f"{GAUGE_SIDECAR_FILENAME}.partial"
    torch.save(sidecar, tmp)
    os.replace(tmp, sidecar_path)
    print(
        f"Saved gauge-canonical sidecar to {sidecar_path} "
        f"({sidecar['gauge_projected_splats']} splats canonicalized at step {step})."
    )
    return sidecar_path


def _save_training_checkpoint(trainer: DatumTrainer, fit_state, run_dir: Path) -> Path:
    checkpoint = _build_training_checkpoint(trainer, fit_state)
    ckpt_path = run_dir / "model.pt"
    tmp = run_dir / "model.pt.partial"
    torch.save(checkpoint, tmp)
    os.replace(tmp, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    _maybe_save_gauge_canonical_sidecar(
        trainer, checkpoint, run_dir, step=int(getattr(fit_state, "steps_ran", 0)),
    )
    return ckpt_path


def _run_train_postprocessing(
    *, trainer: DatumTrainer, fit_state, bandpasses, scene, data_store,
    observations: Dict[int, Dict[str, np.ndarray]], region_masks_by_exposure, primary_region: str | None,
    holdout_summary, run_dir: Path, wandb_run: WandbRun | None, wandb_cfg: Dict[str, Any], run_name: str,
    ckpt_path: Path | None, postprocess_cfg: object | None = None, stage: str = "train",
    prediction_overrides: Dict[str, Dict[int, Dict[str, np.ndarray]]] | None = None,
    operator_provenance: dict[str, object] | None = None,
    annotate_operator_provenance: bool = False,
    _postprocess_status: _PostprocessStatus | None = None,
) -> None:
    status = _postprocess_status or _PostprocessStatus(
        _postprocess_operator_identity({
            "trainer": trainer, "operator_provenance": operator_provenance,
        })
    )
    profile = _maybe_build_postprocess_profile(postprocess_cfg, stage=stage)
    total_postprocess_start = time.perf_counter()

    def _write_metadata() -> None:
        _save_json(trainer.get_calibration_report(), run_dir / "calibration.json")
        if holdout_summary is not None:
            _save_json(holdout_summary, run_dir / "holdout_region.json")

    status.run_optional("metadata", _write_metadata)
    primary_exp_id = _primary_image_exposure_id(data_store, trainer.held_out_band)
    diagnostic_outputs = _resolve_diagnostic_outputs(postprocess_cfg)
    primary_output = "total" if "total" in diagnostic_outputs else diagnostic_outputs[0]
    prediction_overrides = prediction_overrides or {}
    status.start("predictions")
    bundles, observations, diagnostic_exposure_ids = _build_diagnostic_output_bundles(
        trainer=trainer, scene=scene, data_store=data_store, observations=observations,
        diagnostic_outputs=diagnostic_outputs, primary_exp_id=int(primary_exp_id),
        postprocess_cfg=postprocess_cfg, profile=profile, prediction_overrides=prediction_overrides,
    )
    status.ok("predictions")
    primary_bundle = bundles[primary_output]
    predictions = primary_bundle.predictions
    predictions_source = bundles["source"].predictions if "source" in bundles else {}
    partial_holdout_enabled = bool(getattr(trainer, "partial_holdout_enabled", False))
    region_diagnostics_enabled = bool(getattr(data_store, "region_metrics_enabled", True))
    fiber_mask_by_exposure = {
        int(exp_id): (exposure.fiber_mask.to_dense()
                      if hasattr(exposure.fiber_mask, "to_dense") else exposure.fiber_mask)
        for exp_id, exposure in data_store.exposures.items()
    }
    status.start("band_metrics")
    for bundle in bundles.values():
        started = time.perf_counter()
        bundle.band_metrics = metrics_mod.compute_metrics(
            bundle.predictions, bundle.observations, held_out_band=trainer.held_out_band,
            signal_threshold=DEFAULT_DIAGNOSTIC_SIGNAL_THRESHOLD, fiber_mask=fiber_mask_by_exposure,
            partial_holdout_enabled=partial_holdout_enabled, background_subtracted_signal=True,
        )
        timing_key = "band_metrics" if bundle.output == "total" else f"band_metrics_{bundle.output}"
        _record_profile_timing(profile, timing_key, started)
    band_metrics = primary_bundle.band_metrics or {}
    excluded_map = getattr(data_store, "image_excluded_local_indices_by_exposure", {})
    holdout_mask_by_exposure = _holdout_masks_by_exposure(observations, excluded_map)
    resolved_region_masks_by_exposure = {}
    configured_region_masks_primary = {}
    region_masks_primary = {}
    region_grid_pad_pixels = 0
    region_overlays_primary = {}
    region_metrics = {}
    for bundle in bundles.values():
        _save_json(
            metrics_mod.metrics_to_serializable(bundle.band_metrics or {}),
            run_dir / f"metrics{bundle.suffix}.json",
        )
    if primary_bundle.suffix:
        _save_json(
            metrics_mod.metrics_to_serializable(primary_bundle.band_metrics or {}),
            run_dir / "metrics.json",
        )
    _save_json(getattr(trainer, "history", {}), run_dir / "history.json")
    status.ok("band_metrics")

    def _compute_region_diagnostics():
        resolved = visualization.resolve_region_masks_by_exposure(
            scene, region_masks_by_exposure if isinstance(region_masks_by_exposure, dict) else None,
            data_store.exposures, diagnostic_exposure_ids, allow_fallback=True,
        )
        configured_primary = resolved.get(int(primary_exp_id), {})
        masks_primary, resolved_primary_region, pad_pixels, overlays_primary = (
            visualization.resolve_region_masks(
                scene, configured_primary, primary_region=primary_region,
                exposure=data_store.exposures[int(primary_exp_id)], allow_fallback=True,
            )
        )
        for bundle in bundles.values():
            started = time.perf_counter()
            bundle.region_metrics = metrics_mod.compute_region_metrics(
                bundle.predictions, bundle.observations, resolved,
                held_out_band=trainer.held_out_band,
                primary_region=resolved_primary_region,
                signal_threshold=DEFAULT_DIAGNOSTIC_SIGNAL_THRESHOLD,
                partial_holdout_enabled=partial_holdout_enabled,
                holdout_mask_by_exposure=holdout_mask_by_exposure,
                background_subtracted_signal=True,
            )
            timing_key = (
                "region_metrics" if bundle.output == "total"
                else f"region_metrics_{bundle.output}"
            )
            _record_profile_timing(profile, timing_key, started)
        for bundle in bundles.values():
            _save_json(
                metrics_mod.region_metrics_to_serializable(bundle.region_metrics or {}),
                run_dir / f"region_metrics{bundle.suffix}.json",
            )
        return (
            resolved, configured_primary, masks_primary, resolved_primary_region,
            pad_pixels, overlays_primary, primary_bundle.region_metrics or {},
        )

    region_result = status.run_optional(
        "region_metrics", _compute_region_diagnostics,
        skip_reason=None if region_diagnostics_enabled else "region metrics disabled",
    )
    if region_result is not None:
        (
            resolved_region_masks_by_exposure,
            configured_region_masks_primary,
            region_masks_primary,
            primary_region,
            region_grid_pad_pixels,
            region_overlays_primary,
            region_metrics,
        ) = region_result

    def _log_metrics() -> None:
        metrics_mod.log_metrics(band_metrics, trainer.held_out_band)
        if region_diagnostics_enabled:
            metrics_mod.log_region_metrics(
                region_metrics, held_out_band=trainer.held_out_band,
                primary_region=primary_region,
            )

    status.run_optional("metric_logging", _log_metrics)
    observations_primary = primary_bundle.observations[primary_exp_id]
    predictions_primary = predictions[primary_exp_id]
    diagnostic_band = str(data_store.image_band_by_exposure[int(primary_exp_id)])
    region_grid_exp_id = _region_grid_exposure_id(
        observations=primary_bundle.observations,
        predictions=predictions,
        image_band_by_exposure={
            int(exp_id): str(band) for exp_id, band in data_store.image_band_by_exposure.items()
        },
        region_masks_by_exposure=resolved_region_masks_by_exposure,
        fallback_exp_id=int(primary_exp_id),
    )
    region_grid_band = str(data_store.image_band_by_exposure[int(region_grid_exp_id)])
    region_grid_observations = primary_bundle.observations.get(int(region_grid_exp_id), {})
    region_grid_predictions = predictions.get(int(region_grid_exp_id), {})
    region_grid_masks = resolved_region_masks_by_exposure.get(int(region_grid_exp_id), {})
    max_region_grid_regions = getattr(postprocess_cfg, "max_region_grid_regions", 6)
    band_grid_calls = []
    for bundle in bundles.values():
        def _plot_band_grid(bundle=bundle) -> None:
            observations_for_band_grid, predictions_for_band_grid = (
                _merge_nested_images_for_plot(
                    bundle.observations, bundle.predictions,
                )
            )
            started = time.perf_counter()
            visualization.plot_band_grid(
                observations_for_band_grid, predictions_for_band_grid, run_dir,
                trainer.held_out_band, filename=f"band_grid{bundle.suffix}.png",
            )
            timing_key = (
                "band_grid" if bundle.output == "total"
                else f"band_grid_{bundle.output}"
            )
            _record_profile_timing(profile, timing_key, started)
        band_grid_calls.append((bundle.output, _plot_band_grid))
    status.run_optional_calls("band_grid", band_grid_calls)

    def _plot_true_band_with_fiber_mask() -> None:
        started = time.perf_counter()
        visualization.plot_true_band_with_fiber_mask(
            observations_primary[diagnostic_band],
            (lambda m: m.to_dense() if hasattr(m, "to_dense") else m)(
                data_store.exposures[primary_exp_id].fiber_mask),
            diagnostic_band, run_dir,
        )
        _record_profile_timing(profile, "true_band_with_fiber_mask", started)

    status.run_optional("true_band_fiber_mask", _plot_true_band_with_fiber_mask)
    if (int(primary_exp_id) in holdout_mask_by_exposure and diagnostic_band in observations_primary):
        def _plot_holdout_region_overlay() -> None:
            started = time.perf_counter()
            visualization.plot_holdout_region_overlay(
                observations_primary[diagnostic_band],
                holdout_mask_by_exposure[int(primary_exp_id)], diagnostic_band, run_dir,
            )
            _record_profile_timing(profile, "holdout_region_overlay", started)

        status.run_optional("holdout_region_overlay", _plot_holdout_region_overlay)
    else:
        status.skip("holdout_region_overlay", "no partial holdout mask")
    region_grid_calls = []
    if (
        isinstance(region_grid_masks, dict)
        and region_grid_masks
        and region_grid_band in region_grid_observations
        and region_grid_band in region_grid_predictions
    ):
        def _plot_primary_region_grid() -> None:
            started = time.perf_counter()
            visualization.plot_region_cutout_grid(
                region_grid_observations[region_grid_band],
                region_grid_predictions[region_grid_band], region_grid_masks,
                region_grid_band, run_dir, primary_region=primary_region,
                pad_pixels=region_grid_pad_pixels, region_overlays={},
                filename=f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}.png",
                max_regions=max_region_grid_regions,
            )
            _record_profile_timing(profile, "region_grid", started)
        region_grid_calls.append((primary_output, _plot_primary_region_grid))
    for bundle in bundles.values():
        if bundle.output == primary_output:
            continue
        output_region_grid_observations = bundle.observations.get(int(region_grid_exp_id), {})
        output_region_grid_predictions = bundle.predictions.get(int(region_grid_exp_id), {})
        if (
            isinstance(region_grid_masks, dict)
            and region_grid_masks
            and region_grid_band in output_region_grid_observations
            and region_grid_band in output_region_grid_predictions
        ):
            def _plot_output_region_grid(
                bundle=bundle,
                output_region_grid_observations=output_region_grid_observations,
                output_region_grid_predictions=output_region_grid_predictions,
            ) -> None:
                started = time.perf_counter()
                visualization.plot_region_cutout_grid(
                    output_region_grid_observations[region_grid_band],
                    output_region_grid_predictions[region_grid_band], region_grid_masks,
                    region_grid_band, run_dir, primary_region=primary_region,
                    pad_pixels=region_grid_pad_pixels, region_overlays={},
                    filename=(
                        f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}"
                        f"{bundle.suffix}.png"
                    ),
                    max_regions=max_region_grid_regions,
                )
                _record_profile_timing(profile, f"region_grid_{bundle.output}", started)
            region_grid_calls.append((bundle.output, _plot_output_region_grid))
    if region_grid_calls:
        status.run_optional_calls("region_grid", region_grid_calls)
    else:
        status.skip("region_grid", "no resolved region masks")
    source_predictions_primary = predictions_source.get(primary_exp_id, {})

    def _plot_training_history() -> None:
        started = time.perf_counter()
        visualization.plot_history(getattr(trainer, "history", {}), run_dir)
        _record_profile_timing(profile, "training_history_plot", started)

    status.run_optional("training_history_plot", _plot_training_history)

    plot_sed_sample_locations = bool(getattr(postprocess_cfg, "plot_sed_sample_locations", True))

    def _record_point_sed_timing(key: str, started_at: float) -> None:
        _record_profile_timing(profile, key, started_at)

    if trainer.psf_model != "gaussian":
        print(
            "Skipping point-SED diagnostics: mask-aperture image diagnostics "
            f"do not support psf_model={trainer.psf_model!r}."
        )
        status.skip(
            "point_sed_diagnostics",
            f"mixture point-SED skip for psf_model={trainer.psf_model!r}",
        )
    else:
        def _write_point_sed_diagnostics() -> None:
            sed_background_observations = {}
            if plot_sed_sample_locations:
                sed_background_exposure_ids = _image_exposure_ids_for_band(
                    data_store, diagnostic_band,
                )
                sed_background_observations = (
                    _nested_source_observations(
                        scene, data_store, use_flux_density=trainer.use_flux_density,
                        exposure_ids=sed_background_exposure_ids,
                    )
                    if "source" in diagnostic_outputs
                    else _nested_observations(
                        scene, data_store, use_flux_density=trainer.use_flux_density,
                        exposure_ids=sed_background_exposure_ids,
                    )
                )
            diagnostics.write_point_sed_diagnostics(
                trainer=trainer,
                scene=scene,
                observation=observations_primary[diagnostic_band],
                background_observations=sed_background_observations,
                image_band_by_exposure={
                    int(exp_id): str(band)
                    for exp_id, band in data_store.image_band_by_exposure.items()
                },
                exposures=data_store.exposures,
                diagnostic_band=diagnostic_band,
                primary_exp_id=int(primary_exp_id),
                configured_region_masks=(
                    region_masks_primary if isinstance(region_masks_primary, dict) else None
                ),
                region_overlays=region_overlays_primary,
                bandpasses=bandpasses,
                run_dir=run_dir,
                location_background_fallback=source_predictions_primary.get(
                    diagnostic_band, predictions_primary.get(diagnostic_band),
                ),
                plot_sample_locations=plot_sed_sample_locations,
                allow_region_fallback=region_diagnostics_enabled,
                record_timing=_record_point_sed_timing,
            )

        status.run_optional("point_sed_diagnostics", _write_point_sed_diagnostics)

    started = time.perf_counter()
    # Fiber diagnostics default to source-only: with-sky spectra are
    # sky-dominated for faint fibers and hide source recovery. Pass
    # postprocess.fiber_include_additive=true for the with-sky convention.
    fiber_include_additive = bool(getattr(postprocess_cfg, "fiber_include_additive", False))
    fiber_both_variants = bool(
        getattr(postprocess_cfg, "fiber_both_additive_variants", False)
    )
    save_fiber_spectra = bool(getattr(postprocess_cfg, "save_fiber_spectra", False))
    spec_sigma_by_exposure = (
        _fiber_sigma_by_exposure(data_store) if save_fiber_spectra else None
    )
    has_fiber_diagnostics = bool(
        getattr(data_store, "spec_diagnostics_by_exposure", {})
    )
    if has_fiber_diagnostics:
        status.start("fiber_spectrum_diagnostics")
    if has_fiber_diagnostics and fiber_both_variants:
        # One prediction pass scores both additive conventions: the
        # with-additive spectra go to the canonical run_dir outputs and the
        # source-only spectra to the srconly/ subdirectory, replacing two
        # full postprocess runs that recomputed identical source spectra.
        source_spectra, withadd_spectra = trainer.predict_fiber_spectra_variants(
            max_spectra=getattr(postprocess_cfg, "max_fiber_spectra", None),
        )
        srconly_dir = run_dir / "srconly"
        srconly_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.write_fiber_spectrum_diagnostics(
            fiber_spectra=withadd_spectra,
            spec_diagnostics_by_exposure=data_store.spec_diagnostics_by_exposure,
            use_flux_density=trainer.use_flux_density,
            run_dir=run_dir,
            max_overlay_spectra=getattr(postprocess_cfg, "max_fiber_spectra_overlay", 12),
            max_individual_spectra=getattr(postprocess_cfg, "max_fiber_spectrum_plots", None),
            include_additive=True,
            spec_additive_by_exposure=getattr(data_store, "spec_additive_by_exposure", None),
            save_fiber_spectra=save_fiber_spectra,
            spec_sigma_by_exposure=spec_sigma_by_exposure,
        )
        diagnostics.write_fiber_spectrum_diagnostics(
            fiber_spectra=source_spectra,
            spec_diagnostics_by_exposure=data_store.spec_diagnostics_by_exposure,
            use_flux_density=trainer.use_flux_density,
            run_dir=srconly_dir,
            max_overlay_spectra=getattr(postprocess_cfg, "max_fiber_spectra_overlay", 12),
            max_individual_spectra=getattr(postprocess_cfg, "max_fiber_spectrum_plots", None),
            include_additive=False,
            spec_additive_by_exposure=getattr(data_store, "spec_additive_by_exposure", None),
            save_fiber_spectra=save_fiber_spectra,
            spec_sigma_by_exposure=spec_sigma_by_exposure,
        )
    elif has_fiber_diagnostics:
        diagnostics.write_fiber_spectrum_diagnostics(
            fiber_spectra=trainer.predict_fiber_spectra(
                include_additive=fiber_include_additive,
                max_spectra=getattr(postprocess_cfg, "max_fiber_spectra", None),
            ),
            spec_diagnostics_by_exposure=data_store.spec_diagnostics_by_exposure,
            use_flux_density=trainer.use_flux_density,
            run_dir=run_dir,
            max_overlay_spectra=getattr(postprocess_cfg, "max_fiber_spectra_overlay", 12),
            max_individual_spectra=getattr(postprocess_cfg, "max_fiber_spectrum_plots", None),
            include_additive=fiber_include_additive,
            spec_additive_by_exposure=getattr(data_store, "spec_additive_by_exposure", None),
            save_fiber_spectra=save_fiber_spectra,
            spec_sigma_by_exposure=spec_sigma_by_exposure,
        )
    if has_fiber_diagnostics:
        _record_profile_timing(profile, "fiber_spectra_diagnostics", started)
        status.ok("fiber_spectrum_diagnostics")
    else:
        status.skip("fiber_spectrum_diagnostics", "no spectral diagnostics targets")

    def _plot_radial_profiles() -> None:
        started = time.perf_counter()
        visualization.plot_radial_profiles(
            observations_primary[diagnostic_band], predictions_primary[diagnostic_band],
            diagnostic_band, run_dir,
        )
        _record_profile_timing(profile, "radial_profiles", started)

    status.run_optional("radial_profiles", _plot_radial_profiles)

    def _log_radial_diagnostics() -> None:
        started = time.perf_counter()
        diagnostics.log_radial_diagnostics(
            observations_primary, predictions_primary, trainer.held_out_band,
            partial_holdout_enabled=partial_holdout_enabled,
        )
        _record_profile_timing(profile, "log_radial_diagnostics", started)

    status.run_optional("radial_diagnostics", _log_radial_diagnostics)
    status.run_optional(
        "prediction_outputs",
        lambda: _save_diagnostic_prediction_outputs(
            run_dir=run_dir, bundles=bundles, profile=profile,
        ),
    )
    _record_profile_timing(profile, "total_postprocess", total_postprocess_start)
    if profile is not None:
        profile["n_image_exposures"] = int(len(getattr(data_store, "image_band_by_exposure", {})))
        profile["n_spec_exposures"] = int(len(getattr(trainer, "spec_exposure_ids", [])))
        profile["region_diagnostics_enabled"] = bool(region_diagnostics_enabled)
        status.run_optional(
            "profile", lambda: _write_postprocess_profile(profile, run_dir=run_dir),
        )
    else:
        status.skip("profile", "postprocess profiling disabled")

    if annotate_operator_provenance:
        if operator_provenance is None:
            raise ValueError(
                "annotate_operator_provenance requires trained and evaluating identities."
            )
        status.start("metrics_operator_provenance")
        _annotate_metrics_operator_provenance(run_dir, operator_provenance)
        status.ok("metrics_operator_provenance")
    else:
        status.skip(
            "metrics_operator_provenance", "operator was not reinterpreted",
        )

    def _log_wandb() -> None:
        wandb_step = max(0, int(getattr(fit_state, "steps_ran", 0) or 0))
        metrics_payload = {}
        for key, m in band_metrics.items():
            metric_key = key.replace(":", "/")
            prefix = f"metrics/{metric_key}"
            metrics_payload[f"{prefix}/mse"] = m.mse
            metrics_payload[f"{prefix}/mae"] = m.mae
            metrics_payload[f"{prefix}/rmse"] = m.rmse
            metrics_payload[f"{prefix}/flux_error_pct"] = m.flux_error_pct
            metrics_payload[f"{prefix}/nrmse_pct"] = m.nrmse_pct
        if region_diagnostics_enabled:
            for key, m in region_metrics.items():
                metric_key = key.replace(":", "/")
                prefix = f"region_metrics/{metric_key}"
                metrics_payload[f"{prefix}/mse"] = m.mse
                metrics_payload[f"{prefix}/mae"] = m.mae
                metrics_payload[f"{prefix}/rmse"] = m.rmse
                metrics_payload[f"{prefix}/flux_error_pct"] = m.flux_error_pct
                metrics_payload[f"{prefix}/nrmse_pct"] = m.nrmse_pct
        if metrics_payload:
            wandb_run.log(metrics_payload, step=wandb_step)
        image_candidates = {
            "band_grid": run_dir / "band_grid.png",
            "band_grid_source": run_dir / "band_grid_source.png",
            "training_history": run_dir / "training_history.png",
            f"sed_sample_locations_{diagnostic_band}": run_dir
            / f"sed_sample_locations_{diagnostic_band}.png",
            "point_seds": run_dir / "point_seds.png",
            "fiber_spectrum": run_dir / "fiber_spectrum.png",
            "fiber_spectra_overlay": run_dir / "fiber_spectra_overlay.png",
            f"true_{diagnostic_band}_fiber_overlay": run_dir
            / f"true_{diagnostic_band}_fiber_overlay.png",
            f"radial_profile_{diagnostic_band}": run_dir
            / f"radial_profile_{diagnostic_band}.png",
            f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}": run_dir
            / f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}.png",
            f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}_source": run_dir
            / f"region_grid_e{int(region_grid_exp_id)}_{region_grid_band}_source.png",
            f"holdout_region_{diagnostic_band}": run_dir
            / f"holdout_region_{diagnostic_band}.png",
        }
        image_payload = {key: path for key, path in image_candidates.items() if path.exists()}
        wandb_run.log_images(image_payload, step=wandb_step)
        if wandb_cfg.get("log_artifacts", True):
            artifact_name = f"{run_name}_artifacts"
            artifact_candidates = {
                "metrics.json": run_dir / "metrics.json",
                "metrics_source.json": run_dir / "metrics_source.json",
                "history.json": run_dir / "history.json",
                "region_metrics.json": run_dir / "region_metrics.json",
                "region_metrics_source.json": run_dir / "region_metrics_source.json",
                "fiber_metrics.json": run_dir / "fiber_metrics.json",
                "calibration.json": run_dir / "calibration.json",
                "holdout_region.json": run_dir / "holdout_region.json",
                "predictions.npz": run_dir / "predictions.npz",
                "predictions_source.npz": run_dir / "predictions_source.npz", "model.pt": ckpt_path,
                "run.log": run_dir / "run.log", "speed.log": run_dir / "speed.log",
                "config.yaml": run_dir / "config.yaml",
            }
            artifact_files = {key: path for key, path in artifact_candidates.items() if path.exists()}
            wandb_run.log_artifacts(artifact_name, artifact_files)

    if wandb_run:
        status.run_optional("wandb", _log_wandb)
    else:
        status.skip("wandb", "W&B disabled")


def _run_postprocessing_safely(callback, warning: str, **kwargs) -> None:
    status = _PostprocessStatus(_postprocess_operator_identity(kwargs))
    callback_error: BaseException | None = None
    try:
        callback(**kwargs, _postprocess_status=status)
    except Exception as exc:
        callback_error = exc
        status.fail_active(exc)
        print(f"{warning} {type(exc).__name__}: {exc}")
        traceback.print_exc()
    mandatory_paths = _mandatory_postprocess_outputs(kwargs)
    mandatory_outputs = {
        name: path.is_file() for name, path in mandatory_paths.items()
    }
    run_dir = Path(kwargs["run_dir"])
    _save_json(
        status.payload(
            mandatory_outputs=mandatory_outputs, pipeline_error=callback_error,
        ),
        run_dir / "postprocess_status.json",
    )
    missing = [name for name, exists in mandatory_outputs.items() if not exists]
    if callback_error is not None or missing:
        detail = (
            f"; missing mandatory outputs: {', '.join(missing)}" if missing else ""
        )
        raise RuntimeError(
            f"Postprocessing failed closed{detail}. See "
            f"{run_dir / 'postprocess_status.json'}."
        ) from callback_error


def _run_train_postprocessing_safely(**kwargs) -> None:
    _run_postprocessing_safely(
        _run_train_postprocessing,
        "Warning: post-training diagnostics failed after checkpoint save; "
        f"kept checkpoint at {kwargs.get('ckpt_path')}.",
        **kwargs,
    )


def _run_eval_postprocessing(**kwargs) -> None:
    holdout_summary = kwargs.get("holdout_summary")
    kwargs.setdefault(
        "primary_region",
        holdout_summary.get("primary_region") if isinstance(holdout_summary, dict) else None,
    )
    kwargs.setdefault("fit_state", None)
    kwargs.setdefault("wandb_run", None)
    kwargs.setdefault("wandb_cfg", {})
    kwargs.setdefault("run_name", "eval")
    kwargs.setdefault("ckpt_path", None)
    kwargs.setdefault("stage", "eval")
    _run_train_postprocessing(**kwargs)


def _run_eval_postprocessing_safely(**kwargs) -> None:
    _run_postprocessing_safely(
        _run_eval_postprocessing, "Warning: eval diagnostics failed after writing initial metadata.",
        **kwargs,
    )


def _run_train_mode(cfg: DictConfig) -> None:
    platform_cfg = (OmegaConf.to_container(cfg.platform, resolve=True) if "platform" in cfg else {})
    distributed_context = resolve_distributed_context(platform_cfg)
    device = select_runtime_device(cfg.run.get("device", "auto"), platform_cfg)
    if distributed_context.enabled:
        if device.type != "cuda":
            raise RuntimeError("Distributed training requires CUDA.")
        device = _resolve_distributed_cuda_device(int(distributed_context.local_rank))
        _export_distributed_cuda_env(device)
        torch.cuda.set_device(device)
    validate_runtime_environment(device, platform_cfg)
    wandb_run = None
    speed_logger = None
    fit_completed = False
    ckpt_path: Path | None = None
    checkpoint_archiver: _CheckpointArchiver | None = None
    try:
        initialize_distributed(distributed_context)
        synchronize_distributed(distributed_context, device)
        precision_policy = apply_precision_policy(
            platform_cfg.get("precision_policy", "portable_fp32"), device,
        )
        paths = _resolve_paths(cfg)
        base_seed = int(cfg.run.get("seed", 42))
        set_seed(base_seed)
        run_dir_payload = {"ok": True, "run_dir": None, "error": None}
        if distributed_context.is_primary:
            try:
                raw_run_dir = build_run_dir(
                    paths["outdir"], cfg.run.get("run_name", "run"),
                    append_timestamp=cfg.run.get("append_timestamp", True),
                    use_outdir_directly=cfg.run.get("use_outdir_directly", False),
                )
                run_dir_payload["run_dir"] = str(raw_run_dir)
            except Exception as exc:
                run_dir_payload = {
                    "ok": False,
                    "run_dir": None,
                    "error": (
                        "Primary-rank run directory setup failed before training start: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        if distributed_context.enabled:
            run_dir_payload = broadcast_object(run_dir_payload, distributed_context)
        if not bool(run_dir_payload.get("ok", False)):
            raise RuntimeError(str(run_dir_payload.get("error", "Primary-rank setup failed.")))
        run_dir = Path(str(run_dir_payload["run_dir"]))

        wandb_cfg = OmegaConf.to_container(cfg.logging.get("wandb", {}), resolve=True)
        raw_speed_cfg = cfg.logging.get("speed", {})
        if isinstance(raw_speed_cfg, DictConfig):
            speed_cfg = OmegaConf.to_container(raw_speed_cfg, resolve=True)
        elif isinstance(raw_speed_cfg, dict):
            speed_cfg = dict(raw_speed_cfg)
        else:
            speed_cfg = {}
        speed_profile_enabled = bool(speed_cfg.get("enabled", True))
        wandb_cfg.setdefault("run_name", cfg.run.get("run_name", "run"))
        try:
            wandb_log_every_n_steps = int(wandb_cfg.get("log_every_n_steps", 1))
        except (TypeError, ValueError):
            wandb_log_every_n_steps = 1
        wandb_log_every_n_steps = max(1, wandb_log_every_n_steps)
        try:
            wandb_region_grid_every_n_steps = int(wandb_cfg.get("region_grid_every_n_steps", 0))
        except (TypeError, ValueError):
            wandb_region_grid_every_n_steps = 0
        wandb_region_grid_every_n_steps = max(0, wandb_region_grid_every_n_steps)
        try:
            wandb_region_grid_logspace_count = int(
                wandb_cfg.get("region_grid_logspace_count", 0)
            )
        except (TypeError, ValueError):
            wandb_region_grid_logspace_count = 0
        wandb_region_grid_steps = _logspace_steps(
            max_steps=int(cfg.trainer.get("max_steps", 0) or 0),
            count=max(0, wandb_region_grid_logspace_count),
        )
        try:
            fixed_holdout_eval_every_n_steps = int(
                cfg.trainer.get(
                    "fixed_holdout_eval_every_n_steps",
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            fixed_holdout_eval_every_n_steps = 0
        fixed_holdout_eval_every_n_steps = max(
            0, fixed_holdout_eval_every_n_steps
        )
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        epoch_log_every = cfg.logging.get("epoch_log_every", 100)
        if epoch_log_every is not None:
            epoch_log_every = int(epoch_log_every)
        dist_cfg = dict(platform_cfg.get("distributed", {}) or {})
        batch_mode = str(dist_cfg.get("batch_mode", "per_gpu") or "per_gpu")
        logging_setup_payload = {"ok": True, "error": None}
        if distributed_context.is_primary:
            try:
                dump_config(cfg, run_dir / "config.yaml")
                wandb_run = WandbRun(wandb_cfg, run_dir, config_dict)
            except Exception as exc:
                logging_setup_payload = {
                    "ok": False,
                    "error": (
                        "Primary-rank logging setup failed before data loading: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        if distributed_context.enabled:
            logging_setup_payload = broadcast_object(
                logging_setup_payload, distributed_context
            )
        if not bool(logging_setup_payload.get("ok", False)):
            raise RuntimeError(
                str(logging_setup_payload.get("error", "Primary-rank logging setup failed."))
            )
        _stage_prepared_cache_from_environment(distributed_context, wandb_run)
        # A refined checkpoint carries a grown splat count; the model must be
        # built at the checkpoint's size, not the config seed size.
        early_resume_path = _resolve_resume_path(cfg, paths["project_root"])
        if early_resume_path is not None:
            resume_state = torch.load(
                early_resume_path, map_location="cpu", mmap=True, weights_only=False
            ).get("model_state", {})
            resume_mu = resume_state.get("mu")
            if resume_mu is not None and int(resume_mu.shape[0]) != int(cfg.model.get("n_splats", 0)):
                print(
                    f"[resume] overriding model.n_splats {cfg.model.get('n_splats')} -> "
                    f"{int(resume_mu.shape[0])} from checkpoint",
                    flush=True,
                )
                cfg.model.n_splats = int(resume_mu.shape[0])
            del resume_state, resume_mu
        (
            bandpasses, data_name, scene, data_store, trainer, primary_exp_id, primary_exposure
        ) = _build_trainer_components(
            cfg, paths=paths, device=device, precision_policy=precision_policy,
            distributed_context=distributed_context,
            platform_cfg=platform_cfg, wandb_run=wandb_run, speed_logger=speed_logger,
            wandb_log_every_n_steps=wandb_log_every_n_steps, epoch_log_every=epoch_log_every,
            wrap_distributed_model=True,
        )
        trainer._speed_profile_enabled = speed_profile_enabled
        resume_checkpoint_path = _resolve_resume_path(cfg, paths["project_root"])
        resume_checkpoint = None
        if resume_checkpoint_path is not None:
            resume_checkpoint = _load_resume_checkpoint(resume_checkpoint_path, device)
        holdout_summary = getattr(data_store, "holdout_region_summary", None)
        region_masks_by_exposure = getattr(data_store, "region_masks_by_exposure", {})
        primary_region = holdout_summary.get("primary_region") if isinstance(holdout_summary, dict) else None
        observations: Dict[int, Dict[str, np.ndarray]] = {}
        observations_primary_for_periodic: Dict[str, np.ndarray] = {}
        region_masks_primary = {}
        region_grid_pad_pixels = 0
        region_overlays_primary = {}
        setup_payload = {"ok": True, "error": None}
        if distributed_context.is_primary:
            try:
                if wandb_cfg.get("watch_model") and wandb_run:
                    wandb_run.watch(
                        _unwrap_model(trainer.model),
                        log=str(wandb_cfg.get("watch", "gradients")),
                    )
                speed_logger = SpeedLogger(
                    run_dir, enabled=bool(speed_cfg.get("enabled", True)),
                    log_every_n_steps=int(speed_cfg.get("log_every_n_steps", 10) or 10),
                    flush_every_records=int(speed_cfg.get("flush_every_records", 5) or 5),
                )
                observations = _nested_observations(
                    scene, data_store, use_flux_density=trainer.use_flux_density,
                )
                observations_primary_for_periodic = observations.get(int(primary_exp_id), {})
                resolved_region_masks_by_exposure = visualization.resolve_region_masks_by_exposure(
                    scene, region_masks_by_exposure if isinstance(region_masks_by_exposure, dict) else None,
                    data_store.exposures, [int(primary_exp_id)],
                )
                region_masks_primary = resolved_region_masks_by_exposure.get(int(primary_exp_id), {})
                (
                    region_masks_primary,
                    primary_region,
                    region_grid_pad_pixels,
                    region_overlays_primary,
                ) = visualization.resolve_region_masks(
                    scene, region_masks_primary, primary_region=primary_region, exposure=primary_exposure,
                )
                trainer.wandb_logger = wandb_run
                trainer.speed_logger = speed_logger
                trainer._speed_profile_enabled = speed_profile_enabled
            except Exception as exc:
                setup_payload = {
                    "ok": False,
                    "error": (
                        "Primary-rank prefit setup failed before distributed training start: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        if distributed_context.enabled:
            setup_payload = broadcast_object(setup_payload, distributed_context)
        if not bool(setup_payload.get("ok", False)):
            raise RuntimeError(str(setup_payload.get("error", "Primary-rank prefit setup failed.")))

        held_out_band = trainer.held_out_band
        on_step_end = None
        # Checkpoint periodically, not only at the end. A run killed by the
        # scheduler wall clock otherwise leaves nothing at all, and long fits on
        # a time-limited reservation are exactly the ones worth keeping.
        checkpoint_every, checkpoint_retain_every, checkpoint_retain_steps = _checkpoint_plan(
            cfg.trainer, epochs=int(trainer.epochs), steps_per_epoch=int(trainer._steps_per_epoch()),
        )
        if distributed_context.is_primary:
            checkpoint_archiver = _build_checkpoint_archiver(
                cfg.trainer,
                run_dir=run_dir,
                project_root=paths["project_root"],
            )
        max_epoch_steps = int(trainer.epochs) * int(trainer._steps_per_epoch())
        target_absolute_step = (
            min(int(trainer.max_steps), max_epoch_steps)
            if int(trainer.max_steps) > 0 and max_epoch_steps > 0
            else max_epoch_steps
        )

        def _periodic_checkpoint(step: int) -> None:
            retain = _checkpoint_retention_due(
                step=step,
                every_n_steps=checkpoint_retain_every,
                extra_steps=checkpoint_retain_steps,
            )
            write = checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0
            if not write and not retain:
                return
            if not distributed_context.is_primary:
                return
            tmp = run_dir / "model.pt.partial"
            checkpoint = _build_training_checkpoint(trainer, SimpleNamespace(
                epochs_ran=0, steps_ran=int(step)))
            torch.save(checkpoint, tmp)
            os.replace(tmp, run_dir / "model.pt")
            _maybe_save_gauge_canonical_sidecar(
                trainer, checkpoint, run_dir, step=int(step),
            )
            if retain:
                retained_path = _copy_checkpoint_atomically(
                    run_dir / "model.pt",
                    run_dir / f"checkpoint_step_{step}.pt",
                )
                if checkpoint_archiver is not None and step != target_absolute_step:
                    checkpoint_archiver.submit(step=step, path=retained_path)

        if (
            wandb_region_grid_every_n_steps > 0
            or wandb_region_grid_steps
            or fixed_holdout_eval_every_n_steps > 0
        ):
            fixed_holdout_history: list[Dict[str, Any]] = []

            def _on_step_end(step: int, stats: Dict[str, float]) -> None:
                del stats
                _periodic_checkpoint(step)
                region_due = _visualization_step_due(
                    step=step,
                    every_n_steps=wandb_region_grid_every_n_steps,
                    scheduled_steps=wandb_region_grid_steps,
                )
                holdout_due = (
                    fixed_holdout_eval_every_n_steps > 0
                    and step > 0
                    and step % fixed_holdout_eval_every_n_steps == 0
                )
                if not region_due and not holdout_due:
                    return
                if distributed_context.enabled:
                    synchronize_distributed(distributed_context, device)
                render_payload = {"ok": True, "error": None}
                if distributed_context.is_primary:
                    try:
                        if region_due:
                            _maybe_log_model_cutouts(
                                step=step,
                                every_n_steps=wandb_region_grid_every_n_steps,
                                trainer=trainer,
                                wandb_run=wandb_run,
                                run_dir=run_dir,
                                scheduled_steps=wandb_region_grid_steps,
                            )
                            if held_out_band:
                                _maybe_log_periodic_region_grid(
                                    step=step,
                                    every_n_steps=wandb_region_grid_every_n_steps,
                                    trainer=trainer,
                                    wandb_run=wandb_run,
                                    primary_exp_id=int(primary_exp_id),
                                    held_out_band=held_out_band,
                                    observations_primary=observations_primary_for_periodic,
                                    region_masks_primary=region_masks_primary,
                                    region_overlays_primary=region_overlays_primary,
                                    primary_region=primary_region,
                                    pad_pixels=region_grid_pad_pixels,
                                    run_dir=run_dir,
                                    scheduled_steps=wandb_region_grid_steps,
                                )
                        if holdout_due:
                            holdout_metrics = trainer.evaluate_fixed_holdout()
                            fixed_holdout_history.append(
                                {
                                    "step": int(step),
                                    "metrics": holdout_metrics,
                                }
                            )
                            _save_json(
                                fixed_holdout_history,
                                run_dir / "fixed_holdout_history.json",
                            )
                            if wandb_run:
                                payload: Dict[str, float | int] = {}
                                for kind, groups in holdout_metrics.items():
                                    for group, metrics in groups.items():
                                        for metric, value in metrics.items():
                                            payload[
                                                f"fixed_holdout/{kind}/{group}/{metric}"
                                            ] = value
                                if payload:
                                    wandb_run.log(payload, step=int(step))
                    except Exception as exc:
                        render_payload = {
                            "ok": False,
                            "error": (
                                f"Periodic diagnostics failed at step {step}: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                if distributed_context.enabled:
                    render_payload = broadcast_object(render_payload, distributed_context)
                    synchronize_distributed(distributed_context, device)
                if not bool(render_payload.get("ok", False)):
                    raise RuntimeError(str(render_payload["error"]))

            on_step_end = _on_step_end
        elif checkpoint_every > 0 or checkpoint_retain_every > 0 or checkpoint_retain_steps:
            on_step_end = lambda step, stats: _periodic_checkpoint(step)  # noqa: E731
        run_context = (
            RunContext(run_dir, epoch_log_every=epoch_log_every)
            if distributed_context.is_primary
            else nullcontext()
        )
        with run_context:
            if distributed_context.is_primary:
                print(f"Run directory: {run_dir}")
                print(describe_runtime(device, platform_cfg, precision_policy=precision_policy))
            amp_settings = resolve_amp_settings(precision_policy, device)
            if speed_logger:
                speed_logger.log_run_start(
                    {
                        "run_dir": run_dir,
                        "run_name": cfg.run.get("run_name", "run"),
                        "data_name": data_name,
                        "device": device.type,
                        "precision_policy": precision_policy,
                        "amp_enabled": bool(amp_settings.enabled),
                        "amp_dtype": (
                            str(amp_settings.dtype).replace("torch.", "")
                            if amp_settings.dtype is not None
                            else None
                        ),
                        "seed": base_seed,
                        "distributed": {
                            "enabled": bool(distributed_context.enabled),
                            "backend": distributed_context.backend, "batch_mode": batch_mode,
                            "world_size": int(distributed_context.world_size),
                            "rank": int(distributed_context.rank),
                            "local_rank": int(distributed_context.local_rank),
                            "is_primary": bool(distributed_context.is_primary),
                        },
                        "trainer": {
                            "epochs": int(trainer.epochs), "max_steps": int(trainer.max_steps),
                            "pixel_batch_size": int(trainer.image_batch_size),
                            "spec_batch_size": int(trainer.spec_batch_size),
                            "replica_samples_k": int(trainer.replica_samples_k),
                            "kernel_samples_k": int(trainer.kernel_samples_k),
                        },
                        "dataset": {
                            **trainer.image_training_summary(),
                            "n_spec_datums": int(trainer.spec_train_indices.numel()),
                            "steps_per_epoch": int(trainer._steps_per_epoch()),
                            "n_image_exposures": int(len(trainer.image_exposure_ids)),
                            "n_spec_exposures": int(len(trainer.spec_exposure_ids)),
                        },
                        "image_sampling": trainer.image_sampling_summary(),
                        "scientific_objective": trainer.scientific_modality_summary(),
                        "model": {
                            "parameter_count": int(
                                sum(param.numel() for param in _unwrap_model(trainer.model).parameters())
                            ),
                        },
                        "platform": platform_cfg,
                    }
                )
            if resume_checkpoint is not None and resume_checkpoint_path is not None:
                resume_cfg = cfg.get("resume", {})
                _validate_resume_recipe(
                    trainer, resume_checkpoint, checkpoint_path=resume_checkpoint_path,
                    model_only=_resume_model_only(cfg),
                    allow_operator_change=bool(
                        resume_cfg.get("allow_operator_change", False)
                        if isinstance(resume_cfg, (DictConfig, dict))
                        else False
                    ),
                )
                _load_resume_state_into_trainer(
                    trainer, resume_checkpoint, checkpoint_path=resume_checkpoint_path,
                    model_only=_resume_model_only(cfg),
                    allow_operator_change=bool(
                        resume_cfg.get("allow_operator_change", False)
                        if isinstance(resume_cfg, (DictConfig, dict))
                        else False
                    ),
                )
                # last thing before the loop: model, optimizer and schedules are
                # loaded, and nothing else may draw from the restored streams
                restore_rng_state(
                    trainer, resume_checkpoint, checkpoint_path=resume_checkpoint_path,
                )
            fit_state = trainer.fit(on_step_end=on_step_end)
            fit_completed = True
            if distributed_context.is_primary:
                ckpt_path = _save_training_checkpoint(trainer, fit_state, run_dir)
                final_step = int(getattr(fit_state, "steps_ran", 0))
                if _checkpoint_retention_due(
                    step=final_step,
                    every_n_steps=checkpoint_retain_every,
                    final_step=final_step,
                    extra_steps=checkpoint_retain_steps,
                ):
                    retained_path = _copy_checkpoint_atomically(
                        ckpt_path,
                        run_dir / f"checkpoint_step_{final_step}.pt",
                    )
                    if checkpoint_archiver is not None:
                        checkpoint_archiver.submit(step=final_step, path=retained_path)
            if is_distributed_active(distributed_context):
                torch.distributed.barrier()
                destroy_distributed(distributed_context)
            if (
                distributed_context.is_primary and ckpt_path is not None
                and bool(_cfg_get(cfg.get("postprocess"), "enabled", True))
            ):
                trainer.model = _unwrap_model(trainer.model)
                _run_train_postprocessing_safely(
                    trainer=trainer, fit_state=fit_state, bandpasses=bandpasses, scene=scene,
                    data_store=data_store, observations=observations,
                    region_masks_by_exposure=region_masks_by_exposure, primary_region=primary_region,
                    holdout_summary=holdout_summary, run_dir=run_dir, wandb_run=wandb_run,
                    wandb_cfg=wandb_cfg,
                    run_name=str(wandb_cfg.get("run_name", cfg.run.get("run_name", "run"))),
                    ckpt_path=ckpt_path, postprocess_cfg=cfg.get("postprocess"),
                )
    finally:
        if checkpoint_archiver is not None:
            if fit_completed:
                checkpoint_archiver.drain(
                    timeout=float(cfg.trainer.get("checkpoint_archive_drain_timeout_s", 600.0))
                )
            else:
                checkpoint_archiver.close()
        if is_distributed_active(distributed_context):
            destroy_distributed(distributed_context)
        if speed_logger:
            speed_logger.log_run_end({"completed": fit_completed})
            speed_logger.close()
        if wandb_run:
            wandb_run.finish()



def _checkpoint_path_from_cfg(cfg: DictConfig, project_root: Path) -> Path:
    checkpoint_cfg = cfg.get("checkpoint", {})
    raw_path = checkpoint_cfg.get("path") if hasattr(checkpoint_cfg, "get") else None
    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError("mode=eval requires checkpoint.path to point at a saved model.pt.")
    checkpoint_path = resolve_path(str(raw_path), project_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _run_eval_mode(cfg: DictConfig) -> None:
    platform_cfg = (OmegaConf.to_container(cfg.platform, resolve=True) if "platform" in cfg else {})
    device = select_runtime_device(cfg.run.get("device", "auto"), platform_cfg)
    validate_runtime_environment(device, platform_cfg)
    precision_policy = apply_precision_policy(platform_cfg.get("precision_policy", "portable_fp32"), device)
    paths = _resolve_paths(cfg)
    seed = int(cfg.run.get("seed", 42))
    set_seed(seed)
    checkpoint_path = _checkpoint_path_from_cfg(cfg, paths["project_root"])
    checkpoint = _load_checkpoint(checkpoint_path, device)

    (
        bandpasses, _data_name, scene, data_store, trainer, _primary_exp_id, _primary_exposure
    ) = _build_trainer_components(
        cfg, paths=paths, device=device, precision_policy=precision_policy,
    )
    validate_checkpoint_model_config(
        trainer.model, checkpoint, checkpoint_path=checkpoint_path, context="Eval",
    )
    current_operator_config = operator_config_for_trainer(trainer)
    eval_cfg = cfg.get("eval", {})
    allow_reinterpretation = bool(
        eval_cfg.get("allow_operator_reinterpretation", False)
        if isinstance(eval_cfg, (DictConfig, dict)) else False
    )
    reinterpreted = validate_checkpoint_operator_config(
        checkpoint,
        current_operator_config=current_operator_config,
        context="Eval",
        checkpoint_path=checkpoint_path,
        allow_operator_reinterpretation=allow_reinterpretation,
    )
    run_dir = build_run_dir(
        paths["outdir"], cfg.run.get("run_name", "eval"),
        append_timestamp=cfg.run.get("append_timestamp", True),
        use_outdir_directly=cfg.run.get("use_outdir_directly", False),
        suffix="_reinterpreted" if reinterpreted else "",
    )
    dump_config(cfg, run_dir / "config.yaml")
    operator_provenance = {
        "trained": checkpoint_operator_identity(checkpoint),
        "evaluating": current_operator_config,
    }
    _migrate_checkpoint_model_state(checkpoint)
    trainer.model.load_state_dict(checkpoint["model_state"])
    calibration_state = checkpoint.get("calibration_state")
    calibration_loaded = trainer.load_calibration_state_dict(calibration_state)
    if calibration_state is not None and not calibration_loaded:
        print("Warning: checkpoint calibration_state could not be applied; using defaults.")
    if calibration_state is None and trainer.get_calibration_report().get("enabled", False):
        print("Warning: checkpoint has no calibration_state; using unity nuisance scale.")

    with RunContext(run_dir):
        print(f"Run directory: {run_dir}")
        print(describe_runtime(device, platform_cfg, precision_policy=precision_policy))
        _save_json(trainer.get_calibration_report(), run_dir / "calibration.json")
        holdout_summary = getattr(data_store, "holdout_region_summary", None)
        if holdout_summary is not None:
            _save_json(holdout_summary, run_dir / "holdout_region.json")
        observations = _nested_observations(scene, data_store, use_flux_density=trainer.use_flux_density)
        prediction_overrides = _load_eval_prediction_overrides(
            run_dir=run_dir, postprocess_cfg=cfg.get("postprocess"), eval_cfg=cfg.get("eval"),
            project_root=paths["project_root"],
        )
        _run_eval_postprocessing_safely(
            trainer=trainer, bandpasses=bandpasses, scene=scene, data_store=data_store,
            observations=observations,
            region_masks_by_exposure=getattr(data_store, "region_masks_by_exposure", {}),
            holdout_summary=holdout_summary, run_dir=run_dir, postprocess_cfg=cfg.get("postprocess"),
            prediction_overrides=prediction_overrides,
            operator_provenance=operator_provenance,
            annotate_operator_provenance=reinterpreted,
        )


# The prepared store memory-maps one file per array (4,141 for full84), above
# the common soft limit of 1,024 open files; the paper runs raised it to this.
OPEN_FILE_LIMIT = 65536


def _raise_open_file_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = OPEN_FILE_LIMIT if hard == resource.RLIM_INFINITY else min(hard, OPEN_FILE_LIMIT)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


@hydra.main(version_base=None, config_path="../configs", config_name="train_sdss_cosmos_full84_paper")
def main(cfg: DictConfig) -> None:
    _raise_open_file_limit()
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"Config values marked ??? must be supplied: {', '.join(missing)}.")
    mode = str(cfg.get("mode", "train")).strip().lower()
    if mode == "train":
        _run_train_mode(cfg)
        return
    if mode == "eval":
        _run_eval_mode(cfg)
        return
    raise ValueError("mode must be 'train' or 'eval'.")

if __name__ == "__main__":
    _disable_hydra_job_logging()
    main()
