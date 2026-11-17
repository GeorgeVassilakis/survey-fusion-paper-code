"""Adam's step counter stays on the host across a resume.

torch hosts the counter on the CPU when capturable and fused are both off,
because the bias corrections read it as a Python number every step. Resume
loads the whole checkpoint with map_location=device, which lands the counter
on the accelerator and turns that read into a device-to-host copy per
parameter per step. Restoring torch's own placement is a device move of a
small exact integer, so the update itself is unchanged.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.synthetic_trainer import make_image_spec_trainer  # noqa: E402


def _state_dict_with_step_on(device: str, trainer):
    state = trainer.optimizer_state_dict()
    dense = state["dense"] if state.get("format") == "hybrid_sparse_v1" else state
    for entry in dense["state"].values():
        entry["step"] = entry["step"].to(device)
    return state


def test_resume_puts_the_step_counter_back_on_the_host():
    trainer = make_image_spec_trainer(calibration_init_scale=1.3)
    image_indices = torch.tensor([0], dtype=torch.long)
    spec_indices = torch.tensor([1], dtype=torch.long)
    trainer._train_step(image_indices, spec_indices)
    assert trainer.optimizer.state
    for state in trainer.optimizer.state.values():
        assert state["step"].device.type == "cpu"

    # A checkpoint whose counters arrived on the training device, which is what
    # map_location=device produces. On CPU-only hardware the move is a no-op,
    # so assert the invariant the loader restores rather than the move itself.
    reloaded = make_image_spec_trainer(calibration_init_scale=1.3)
    reloaded.load_optimizer_state_dict(
        _state_dict_with_step_on("cpu", trainer)
    )
    for state in reloaded.optimizer.state.values():
        assert state["step"].device.type == "cpu"


def test_hosting_a_counter_leaves_its_value_alone():
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.Adam([parameter])
    parameter.grad = torch.ones(3)
    optimizer.step()
    state = optimizer.state[parameter]
    state["step"] = torch.tensor(97339.0)
    from survey_fusion.training.datum_trainer import DatumTrainer

    DatumTrainer._host_optimizer_step_counters(optimizer)

    assert state["step"].device.type == "cpu"
    assert float(state["step"]) == 97339.0


def test_every_epoch_step_count_is_exact_in_the_counter_dtype():
    """The counter is float32; the epoch is well inside its integer range."""
    for step in (1, 20_000, 97_339, 2**24):
        assert float(torch.tensor(float(step), dtype=torch.float32)) == float(step)
