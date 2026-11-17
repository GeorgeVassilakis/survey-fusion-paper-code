from __future__ import annotations

import pytest
import torch

from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.models.splat_field import SplatField


def _make_trainer(*, max_steps: int):
    model = SplatField(
        n_splats=4,
        n_templates=2,
        loglam_grid=torch.linspace(-1.0, 1.0, 4),
    )
    return make_image_spec_trainer(
        model=model,
        trainer_cfg={"max_steps": int(max_steps)},
    )


def test_position_lr_horizon_is_independent_of_max_steps() -> None:
    with pytest.warns(RuntimeWarning, match="position-LR schedule will not complete"):
        short_trainer = _make_trainer(max_steps=15_000)
    full_trainer = _make_trainer(max_steps=30_000)

    short_trainer._post_optimizer_step(15_000)
    full_trainer._post_optimizer_step(15_000)

    assert short_trainer._position_lr_horizon == 30_000
    assert full_trainer._position_lr_horizon == 30_000
    assert short_trainer._position_group is not None
    assert full_trainer._position_group is not None
    assert short_trainer._position_group["lr"] == pytest.approx(
        full_trainer._position_group["lr"]
    )
