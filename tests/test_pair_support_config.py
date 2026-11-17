from __future__ import annotations

import math

import pytest
import torch

import survey_fusion.training.datum_trainer as datum_trainer_module
from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.models.splat_field import SplatField


def _field(n_splats: int = 4) -> SplatField:
    torch.manual_seed(5)
    return SplatField(
        n_splats=n_splats, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4)
    )


def _trainer(**refine_overrides):
    refine = {"enabled": True, "start": 0, "stop": 100, "every": 10}
    refine.update(refine_overrides)
    return make_image_spec_trainer(model=_field(), trainer_cfg={"refine": refine})


def test_pair_n_sigma_defaults_to_eight() -> None:
    assert _trainer().pair_n_sigma == 8.0


def test_pair_n_sigma_resolves_from_config() -> None:
    assert _trainer(pair_n_sigma=5.5).pair_n_sigma == 5.5


@pytest.mark.parametrize("bad", [0.0, -3.5, math.inf, math.nan])
def test_pair_n_sigma_rejects_non_positive_or_non_finite(bad) -> None:
    with pytest.raises(ValueError, match="pair_n_sigma"):
        _trainer(pair_n_sigma=bad)


def test_resolved_pair_n_sigma_reaches_the_packed_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(pair_n_sigma=5.5)
    captured = {}

    def capture(field, x, y, psf_sigma, n_sigma=None):
        captured["n_sigma"] = n_sigma
        return "index"

    monkeypatch.setattr(
        datum_trainer_module, "build_packed_point_index", capture
    )
    xs = torch.zeros(3)
    result = trainer._build_point_index(
        trainer._base_model(), xs, xs, torch.full((3,), 0.1)
    )
    assert result == "index"
    assert captured["n_sigma"] == 5.5
