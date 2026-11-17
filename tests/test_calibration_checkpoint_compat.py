from __future__ import annotations

import numpy as np

from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.training.datum_trainer import DatumTrainer


def _make_trainer(*, init_scale: float) -> DatumTrainer:
    return make_image_spec_trainer(calibration_init_scale=init_scale, calibration_enabled=True)


def test_checkpoint_without_calibration_state_defaults_to_unity_behavior() -> None:
    trainer = _make_trainer(init_scale=1.0)
    loaded = trainer.load_calibration_state_dict(None)
    assert loaded is False
    report = trainer.get_calibration_report()
    assert report.get("enabled", False)
    assert np.isclose(report["band_scales"]["r"], 1.0, rtol=1e-6, atol=1e-6)


def test_checkpoint_calibration_state_roundtrip_restores_scales() -> None:
    trainer_a = _make_trainer(init_scale=1.0)
    state = trainer_a.get_calibration_state_dict()
    assert state is not None
    state["log_scales"] = [float(np.log(1.2))]

    trainer_b = _make_trainer(init_scale=1.0)
    loaded = trainer_b.load_calibration_state_dict(state)
    assert loaded is True
    report = trainer_b.get_calibration_report()
    assert np.isclose(report["band_scales"]["r"], 1.2, rtol=1e-6, atol=1e-6)
