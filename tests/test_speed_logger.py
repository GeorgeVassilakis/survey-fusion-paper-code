from __future__ import annotations

import json
from pathlib import Path

from survey_fusion.utils.speed import SpeedLogger


def test_speed_logger_writes_windowed_jsonl(tmp_path: Path) -> None:
    logger = SpeedLogger(tmp_path, enabled=True, log_every_n_steps=2, flush_every_records=1)
    logger.log_run_start({"run_name": "bench_test", "device": "cpu"})
    logger.record_step(
        {
            "step": 1, "epoch": 1, "total_step_sec": 0.5, "total_datums": 100, "mc_samples": 3200,
            "total_loss": 1.0, "gpu_peak_mem_allocated_mb": 0.0, "gpu_peak_mem_reserved_mb": 0.0,
        }
    )
    logger.record_step(
        {
            "step": 2, "epoch": 1, "total_step_sec": 0.25, "total_datums": 120, "mc_samples": 3840,
            "total_loss": 0.8, "gpu_peak_mem_allocated_mb": 0.0, "gpu_peak_mem_reserved_mb": 0.0,
        }
    )
    logger.log_run_end({"completed": True})
    logger.close()

    log_path = tmp_path / "speed.log"
    assert log_path.exists()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["type"] for record in records] == ["run_start", "step_window", "run_end"]

    window = records[1]
    assert window["step_start"] == 1
    assert window["step_end"] == 2
    assert window["n_steps"] == 2
    assert len(window["steps"]) == 2
    assert window["datums_total"] == 220
    assert window["mc_samples_total"] == 7040
    assert window["last_total_loss"] == 0.8

    summary = records[2]
    assert summary["completed"] is True
    assert summary["total_steps"] == 2
    assert summary["total_datums"] == 220
    assert summary["total_mc_samples"] == 7040


def test_speed_logger_serializes_non_finite_values_as_null(tmp_path: Path) -> None:
    logger = SpeedLogger(tmp_path, enabled=True, log_every_n_steps=2, flush_every_records=1)
    logger.log_run_end({"completed": False})
    logger.close()

    log_path = tmp_path / "speed.log"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    summary = records[0]
    assert summary["type"] == "run_end"
    assert summary["completed"] is False
    assert summary["mean_steps_per_sec"] is None
    assert summary["mean_datums_per_sec"] is None
    assert summary["mean_mc_samples_per_sec"] is None
