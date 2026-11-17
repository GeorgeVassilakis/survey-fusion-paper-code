from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_module(
    Path(__file__).resolve().parents[1] / "src" / "train.py", "survey_fusion_train_image_diag_dispatch",
)
eval_entry = train_entry


class FakeTrainer:
    def __init__(self) -> None:
        self.det_calls = 0
        self._predictions = {3: {"i": np.ones((2, 2), dtype=np.float32)}}
        self.image_exposure_ids = [3, 4, 5]
        self.exposure_ids = None

    def compute_datum_image_predictions_deterministic(
        self, *, supersample=2, chunk_size=2048, include_additive=True, exposure_ids=None, profile=None,
    ):
        del supersample, chunk_size, include_additive, profile
        self.det_calls += 1
        self.exposure_ids = exposure_ids
        return self._predictions


def test_train_image_diagnostic_predictions_use_deterministic_datum_path() -> None:
    trainer = FakeTrainer()
    preds = train_entry._compute_image_diagnostic_predictions(trainer)
    assert preds == trainer._predictions
    assert trainer.det_calls == 1


def test_eval_image_diagnostic_predictions_use_deterministic_datum_path() -> None:
    trainer = FakeTrainer()
    preds = eval_entry._compute_image_diagnostic_predictions(trainer)
    assert preds == trainer._predictions
    assert trainer.det_calls == 1


def test_image_diagnostic_predictions_can_limit_exposures() -> None:
    trainer = FakeTrainer()
    preds = train_entry._compute_image_diagnostic_predictions(trainer, postprocess_cfg={"max_exposures": 2})
    assert preds == trainer._predictions
    assert trainer.exposure_ids == [3, 4]


def test_image_diagnostic_predictions_keep_required_exposure_when_limited() -> None:
    trainer = FakeTrainer()
    trainer.image_exposure_ids = [0, 1, 2, 3, 4, 11]
    preds = train_entry._compute_image_diagnostic_predictions(
        trainer, postprocess_cfg={"max_exposures": 5}, required_exposure_ids=[11],
    )
    assert preds == trainer._predictions
    assert trainer.exposure_ids == [0, 1, 2, 3, 11]
