from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from train import _init_model as init_train_model


def _splat_cfg(**overrides):
    model = {
        "name": "splat", "n_splats": 4, "n_templates": 3, "n_loglam_grid": 12,
        "init_extent": 1.0, "init_scale": 0.02,
    }
    model.update(overrides)
    return OmegaConf.create({"model": model})


def _cfg(output_head):
    return OmegaConf.create(
        {
            "model": {
                "hidden_dim": 8, "n_layers": 2, "softplus_beta": 2.0, "positional_encoding": None,
                "output_head": output_head,
            }
        }
    )


def test_train_init_model_treats_null_output_head_as_default() -> None:
    model = init_train_model(_cfg(None))

    assert model.output_head == "softplus"


def test_splat_sed_basis_flags_default_to_the_pre_fix_behaviour() -> None:
    model = init_train_model(_splat_cfg())

    assert model.sed.off_grid == "clamp"
    assert model.sed.pin_gauge_knots is False


def test_splat_sed_basis_flags_flow_from_the_model_config() -> None:
    model = init_train_model(
        _splat_cfg(sed_off_grid="zero", sed_pin_gauge_knots=True)
    )

    assert model.sed.off_grid == "zero"
    assert model.sed.pin_gauge_knots is True


def test_splat_sed_off_grid_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="off_grid must be one of"):
        init_train_model(_splat_cfg(sed_off_grid="extrapolate"))
