"""Benchmark-oriented speed logging utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return str(value)


class SpeedLogger:
    """Buffered JSONL logger for benchmark-relevant training telemetry."""

    def __init__(
        self, run_dir: Path, *, enabled: bool = True, log_every_n_steps: int = 10, flush_every_records: int=5,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(run_dir) / "speed.log"
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.flush_every_records = max(1, int(flush_every_records))
        self._fh = None
        self._window_steps: list[dict[str, Any]] = []
        self._record_buffer: list[str] = []
        self._records_since_flush = 0
        self._overall_steps = 0
        self._overall_wall_sec = 0.0
        self._overall_datums = 0
        self._overall_mc_samples = 0
        self._overall_peak_mem_allocated_mb = 0.0
        self._overall_peak_mem_reserved_mb = 0.0

        if not self.enabled:
            return
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", buffering=1024 * 1024)

    def log_run_start(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._write_record({"type": "run_start", **_sanitize(dict(payload))})

    def record_step(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        record = _sanitize(dict(payload))
        self._window_steps.append(record)
        if len(self._window_steps) >= self.log_every_n_steps:
            self._flush_window()

    def log_run_end(self, payload: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._flush_window()
        summary = {
            "type": "run_end",
            "total_steps": self._overall_steps,
            "total_wall_sec": self._overall_wall_sec,
            "mean_steps_per_sec": (
                self._overall_steps / self._overall_wall_sec if self._overall_wall_sec > 0.0 else float("nan")
            ),
            "total_datums": self._overall_datums,
            "mean_datums_per_sec": (
                self._overall_datums / self._overall_wall_sec
                if self._overall_wall_sec > 0.0
                else float("nan")
            ),
            "total_mc_samples": self._overall_mc_samples,
            "mean_mc_samples_per_sec": (
                self._overall_mc_samples / self._overall_wall_sec
                if self._overall_wall_sec > 0.0
                else float("nan")
            ),
            "peak_gpu_max_mem_allocated_mb": self._overall_peak_mem_allocated_mb,
            "peak_gpu_max_mem_reserved_mb": self._overall_peak_mem_reserved_mb,
        }
        if payload:
            summary.update(_sanitize(dict(payload)))
        self._write_record(summary)
        self._flush_records(force=True)

    def close(self) -> None:
        if self._fh is None:
            return
        self._flush_window()
        self._flush_records(force=True)
        self._fh.close()
        self._fh = None

    def _flush_window(self) -> None:
        if not self._window_steps:
            return
        steps = self._window_steps
        self._window_steps = []
        step_start = int(steps[0]["step"])
        step_end = int(steps[-1]["step"])
        wall_sec = sum(float(step.get("total_step_sec", 0.0) or 0.0) for step in steps)
        total_datums = sum(int(step.get("total_datums", 0) or 0) for step in steps)
        total_mc_samples = sum(int(step.get("mc_samples", 0) or 0) for step in steps)
        peak_alloc = max(float(step.get("gpu_peak_mem_allocated_mb", 0.0) or 0.0) for step in steps)
        peak_reserved = max(float(step.get("gpu_peak_mem_reserved_mb", 0.0) or 0.0) for step in steps)
        self._overall_steps += len(steps)
        self._overall_wall_sec += wall_sec
        self._overall_datums += total_datums
        self._overall_mc_samples += total_mc_samples
        self._overall_peak_mem_allocated_mb = max(self._overall_peak_mem_allocated_mb, peak_alloc)
        self._overall_peak_mem_reserved_mb = max(self._overall_peak_mem_reserved_mb, peak_reserved)

        last = steps[-1]
        summary = {
            "type": "step_window", "step_start": step_start, "step_end": step_end, "n_steps": len(steps),
            "epoch_last": int(last.get("epoch", 0) or 0), "wall_sec": wall_sec,
            "steps_per_sec": len(steps) / wall_sec if wall_sec > 0.0 else float("nan"),
            "datums_total": total_datums,
            "datums_per_sec": total_datums / wall_sec if wall_sec > 0.0 else float("nan"),
            "mc_samples_total": total_mc_samples,
            "mc_samples_per_sec": (total_mc_samples / wall_sec if wall_sec > 0.0 else float("nan")),
            "last_total_loss": last.get("total_loss"), "last_image_loss": last.get("image_loss"),
            "last_spec_loss": last.get("spec_loss"), "last_grad_norm_preclip": last.get("grad_norm_preclip"),
            "last_grad_norm_postclip": last.get("grad_norm_postclip"),
            "max_gpu_peak_mem_allocated_mb": peak_alloc, "max_gpu_peak_mem_reserved_mb": peak_reserved,
            "steps": steps,
        }
        self._write_record(summary)

    def _write_record(self, payload: Mapping[str, Any]) -> None:
        if self._fh is None:
            return
        record = _sanitize(dict(payload))
        self._record_buffer.append(json.dumps(record, sort_keys=True, allow_nan=False))
        self._records_since_flush += 1
        self._flush_records(force=False)

    def _flush_records(self, *, force: bool) -> None:
        if self._fh is None or not self._record_buffer:
            return
        if not force and self._records_since_flush < self.flush_every_records:
            return
        self._fh.write("\n".join(self._record_buffer))
        self._fh.write("\n")
        self._fh.flush()
        self._record_buffer.clear()
        self._records_since_flush = 0

    def __bool__(self) -> bool:
        return bool(self.enabled and self._fh is not None)
