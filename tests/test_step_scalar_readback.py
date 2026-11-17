"""The step's scalars come off the device in one read, unchanged.

The losses, the calibration prior, both gradient norms and the per-band
calibration scales used to be eight separate blocking copies. They are now one
concatenated float64 read. float32 widens to float64 exactly and a Python
float is a float64, so the numbers in the stats dict, history.json and
speed.log must be bit for bit what the separate reads produced.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.synthetic_trainer import make_image_spec_trainer  # noqa: E402


def _indices():
    return torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)


def test_step_scalars_match_the_tensors_they_are_read_from():
    trainer = make_image_spec_trainer(calibration_init_scale=1.3)
    assert trainer.spectral_calibrator is not None
    image_indices, spec_indices = _indices()

    stats = trainer._train_step(image_indices, spec_indices)

    scales = trainer.spectral_calibrator.scales_by_band()
    expected = {
        band: float(value)
        for band, value in zip(
            trainer.spectral_calibrator.band_names, scales.detach().cpu().tolist()
        )
    }
    assert stats["calibration_scales"] == expected
    assert stats["calibration_scales"].keys() == set(
        trainer.spectral_calibrator.band_names
    )
    for key in (
        "local_total_nll", "local_image_nll", "local_spec_nll",
        "calibration_prior", "grad_norm_preclip", "grad_norm_postclip",
    ):
        assert isinstance(stats[key], float)
    assert stats["grad_norm_postclip"] <= stats["grad_norm_preclip"]


def test_widening_to_float64_leaves_a_float32_read_unchanged():
    """The property the single read rests on, over the exponent range."""
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 1e-30, -3.4e38, 0.1, 1.0 / 3.0, float("inf")],
        dtype=torch.float32,
    )
    assert values.tolist() == values.to(torch.float64).tolist()


def test_scalars_are_read_without_a_calibrator():
    trainer = make_image_spec_trainer()
    assert trainer.spectral_calibrator is None
    image_indices, spec_indices = _indices()

    stats = trainer._train_step(image_indices, spec_indices)

    assert stats["calibration_scales"] == {}
    assert stats["calibration_prior"] == 0.0


def test_report_still_serves_its_other_caller():
    trainer = make_image_spec_trainer(calibration_init_scale=1.3)
    report = trainer.get_calibration_report()
    assert report["enabled"] is True
    assert set(report["band_scales"]) == set(trainer.spectral_calibrator.band_names)
    assert set(report["band_log_scales"]) == set(
        trainer.spectral_calibrator.band_names
    )
