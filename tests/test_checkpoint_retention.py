from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import ModuleType
import time


def _load_entrypoint_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("survey_fusion_train_retention_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_entrypoint_module()


def test_checkpoint_retention_uses_absolute_cadence_final_and_extra_steps() -> None:
    final_step = 35_000
    extra_steps = {7_500, 17_500}
    retained = {
        step
        for step in range(2_500, final_step + 1, 2_500)
        if train_entry._checkpoint_retention_due(
            step=step,
            every_n_steps=10_000,
            final_step=final_step,
            extra_steps=extra_steps,
        )
    }

    assert retained == {7_500, 10_000, 17_500, 20_000, 30_000, 35_000}


def test_epoch_retention_lands_on_epoch_boundaries() -> None:
    assert train_entry._epoch_retention_steps(steps_per_epoch=100, epochs=5, every_n_epochs=1) == {
        100, 200, 300, 400, 500,
    }
    assert train_entry._epoch_retention_steps(steps_per_epoch=100, epochs=5, every_n_epochs=2) == {
        200, 400,
    }
    assert train_entry._epoch_retention_steps(steps_per_epoch=100, epochs=5, every_n_epochs=0) == set()
    assert train_entry._epoch_retention_steps(steps_per_epoch=0, epochs=5, every_n_epochs=1) == set()


def _write_retained_checkpoint(run_dir: Path, step: int, payload: bytes) -> Path:
    path = run_dir / f"checkpoint_step_{step}.pt"
    path.write_bytes(payload)
    return path


def _wait_for(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def test_checkpoint_archiver_verifies_records_and_then_deletes_local(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    archive_dir = tmp_path / "archive"
    run_dir.mkdir()
    payload = b"retained-checkpoint"
    local_path = _write_retained_checkpoint(run_dir, 10_000, payload)
    archiver = train_entry._CheckpointArchiver(
        run_dir=run_dir, archive_dir=archive_dir, keep_local=0
    )

    archiver.submit(step=10_000, path=local_path)
    archiver.drain(timeout=5.0)

    archive_path = archive_dir / local_path.name
    digest = hashlib.sha256(payload).hexdigest()
    assert archive_path.read_bytes() == payload
    assert not local_path.exists()
    assert (run_dir / "ARCHIVED_CKPTS.txt").read_text(encoding="utf-8") == (
        f"10000 {archive_path} {digest}\n"
    )


def test_checkpoint_archiver_keeps_newest_local_floor(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    archive_dir = tmp_path / "archive"
    run_dir.mkdir()
    archiver = train_entry._CheckpointArchiver(
        run_dir=run_dir, archive_dir=archive_dir, keep_local=2
    )
    steps = (10_000, 20_000, 30_000)
    paths = [
        _write_retained_checkpoint(run_dir, step, f"step-{step}".encode())
        for step in steps
    ]

    for step, path in zip(steps, paths, strict=True):
        archiver.submit(step=step, path=path)
    archiver.drain(timeout=5.0)

    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()
    assert {path.name for path in archive_dir.glob("checkpoint_step_*.pt")} == {
        path.name for path in paths
    }


def test_checkpoint_archiver_failure_is_nonfatal_and_retries_on_next_submit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    archive_dir = tmp_path / "archive"
    run_dir.mkdir()
    archive_dir.write_text("not a directory", encoding="utf-8")
    first = _write_retained_checkpoint(run_dir, 10_000, b"first")
    archiver = train_entry._CheckpointArchiver(
        run_dir=run_dir, archive_dir=archive_dir, keep_local=0
    )

    archiver.submit(step=10_000, path=first)
    marker = run_dir / "CHECKPOINT_ARCHIVE_FAILED"
    _wait_for(marker)
    assert first.exists()

    archive_dir.unlink()
    archive_dir.mkdir()
    second = _write_retained_checkpoint(run_dir, 20_000, b"second")
    archiver.submit(step=20_000, path=second)
    archiver.drain(timeout=5.0)

    assert (archive_dir / first.name).read_bytes() == b"first"
    assert (archive_dir / second.name).read_bytes() == b"second"
    assert not first.exists()
    assert not second.exists()
    assert marker.exists()


def test_checkpoint_archiver_is_inert_when_archive_dir_is_unset(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    archiver = train_entry._build_checkpoint_archiver(
        {}, run_dir=run_dir, project_root=tmp_path
    )

    assert archiver is None
    assert list(run_dir.iterdir()) == []
