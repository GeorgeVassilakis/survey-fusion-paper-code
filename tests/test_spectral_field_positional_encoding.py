from __future__ import annotations

import torch
import pytest

from survey_fusion.models.spectral_field import GalaxySpectralField


def test_positional_encoding_includes_lambda_features_by_default() -> None:
    model = GalaxySpectralField(
        hidden_dim=16, n_layers=2,
        positional_encoding={"enabled": True, "n_frequencies_xy": 3, "include_input_xy": False},
    )
    assert model.xy_encoder is not None
    assert model.lambda_encoder is not None
    assert model.lambda_encoder.output_dim == model.xy_encoder.output_dim

    first_linear = model.network[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == (model.xy_encoder.output_dim * 2 + model.lambda_encoder.output_dim)


def test_positional_encoding_uses_lambda_specific_settings() -> None:
    model = GalaxySpectralField(
        hidden_dim=16,
        n_layers=2,
        positional_encoding={
            "enabled": True, "n_frequencies_xy": 2, "include_input_xy": True, "n_frequencies_lambda": 5,
            "include_input_lambda": False,
        },
    )
    assert model.xy_encoder is not None
    assert model.lambda_encoder is not None
    assert model.lambda_encoder.output_dim != model.xy_encoder.output_dim

    x = torch.zeros(8)
    y = torch.zeros(8)
    log_lam = torch.linspace(-1.0, 1.0, 8)
    out = model(x, y, log_lam)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_default_model_keeps_plain_sequential_state_dict_layout() -> None:
    model = GalaxySpectralField(hidden_dim=16, n_layers=4, positional_encoding=None)

    assert isinstance(model.network[0], torch.nn.Linear)
    assert isinstance(model.network[1], torch.nn.ReLU)
    assert isinstance(model.network[2], torch.nn.Linear)
    assert list(model.state_dict()) == [
        "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias", "network.4.weight",
        "network.4.bias", "network.6.weight", "network.6.bias",
    ]


def test_zero_centered_softplus_head_is_zero_at_zero_raw_output() -> None:
    model = GalaxySpectralField(
        hidden_dim=8, n_layers=2, positional_encoding=None, output_head="zero_centered_softplus",
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    out = model(torch.zeros(4), torch.zeros(4), torch.zeros(4))

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7, rtol=0.0)


def test_invalid_output_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported output_head"):
        GalaxySpectralField(output_head="definitely_not_a_head")


def test_linear_output_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported output_head"):
        GalaxySpectralField(output_head="linear")
