from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_train_entrypoint() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("survey_fusion_train_region_grid_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_train_entrypoint()


def test_logspace_steps_span_training_without_fixed_step_values() -> None:
    steps = train_entry._logspace_steps(max_steps=10_000, count=12)

    assert steps[0] == 1
    assert steps[-1] == 10_000
    assert len(steps) <= 12
    assert steps == tuple(sorted(set(steps)))
    assert all(step > 0 for step in steps)


def test_logspace_steps_handle_short_runs_and_disabled_schedule() -> None:
    assert train_entry._logspace_steps(max_steps=3, count=12) == (1, 2, 3)
    assert train_entry._logspace_steps(max_steps=100, count=0) == ()


class RecordingWandb:
    def __init__(self) -> None:
        self.calls: list[tuple[int | None, dict[str, Path]]] = []

    def log_images(self, payload, step: int | None = None) -> None:
        self.calls.append((step, dict(payload)))


class FakeTrainer:
    def __init__(self, *, predictions: dict[int, dict[str, np.ndarray]]) -> None:
        self._predictions = predictions
        self.det_calls = 0

    def compute_datum_image_predictions_deterministic(
        self, *, supersample=2, chunk_size=2048, include_additive=True, exposure_ids=None, profile=None,
    ) -> dict[int, dict[str, np.ndarray]]:
        del supersample, chunk_size, include_additive, exposure_ids, profile
        self.det_calls += 1
        return self._predictions


def test_periodic_region_grid_logging_logs_only_on_step_multiples(tmp_path, monkeypatch) -> None:
    held_out_band = "i"
    primary_exp_id = 1
    observed = np.ones((2, 2), dtype=np.float32)
    predicted = np.full((2, 2), 2.0, dtype=np.float32)
    trainer = FakeTrainer(predictions={primary_exp_id: {held_out_band: predicted}})
    wandb = RecordingWandb()
    plot_calls: list[tuple[str, str | None]] = []

    def _fake_plot(
        observed_arr, predicted_arr, region_masks, band, run_dir, *, primary_region=None, pad_pixels=6,
        region_overlays=None, filename=None,
    ) -> None:
        del pad_pixels
        assert region_overlays == {}
        np.testing.assert_allclose(observed_arr, observed)
        np.testing.assert_allclose(predicted_arr, predicted)
        assert isinstance(region_masks, dict)
        assert filename == "region_grid_e1_i.png"
        plot_calls.append((str(band), primary_region))
        (Path(run_dir) / filename).write_bytes(b"png")

    monkeypatch.setattr(train_entry.visualization, "plot_region_cutout_grid", _fake_plot)

    for step in range(1, 6):
        train_entry._maybe_log_periodic_region_grid(
            step=step, every_n_steps=2, trainer=trainer, wandb_run=wandb, primary_exp_id=primary_exp_id,
            held_out_band=held_out_band, observations_primary={held_out_band: observed},
            region_masks_primary={"R2": np.ones((2, 2), dtype=bool)}, region_overlays_primary={},
            primary_region="R2", pad_pixels=6, run_dir=tmp_path,
        )

    assert [step for step, _ in wandb.calls] == [2, 4]
    assert all(set(payload.keys()) == {"region_grid_e1_i"} for _, payload in wandb.calls)
    assert len(plot_calls) == 2
    assert trainer.det_calls == 2


def test_periodic_region_grid_logging_disabled_when_every_n_steps_is_zero(tmp_path) -> None:
    trainer = FakeTrainer(predictions={1: {"i": np.ones((2, 2), dtype=np.float32)}})
    wandb = RecordingWandb()
    train_entry._maybe_log_periodic_region_grid(
        step=2, every_n_steps=0, trainer=trainer, wandb_run=wandb, primary_exp_id=1, held_out_band="i",
        observations_primary={"i": np.ones((2, 2), dtype=np.float32)},
        region_masks_primary={"R2": np.ones((2, 2), dtype=bool)}, region_overlays_primary={},
        primary_region="R2", pad_pixels=6, run_dir=tmp_path,
    )
    assert wandb.calls == []
    assert trainer.det_calls == 0
