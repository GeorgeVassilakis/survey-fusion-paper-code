from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatureEncoding(nn.Module):
    """Applies NeRF-style sinusoidal features to a 1D coordinate."""

    def __init__(self, n_frequencies: int, include_input: bool = True):
        super().__init__()
        if n_frequencies <= 0:
            raise ValueError("n_frequencies must be positive for positional encoding.")
        self.include_input = include_input
        freq_exponents = torch.arange(n_frequencies, dtype=torch.float32)
        freq_bands = (2.0**freq_exponents) * math.pi
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self) -> int:
        base = 1 if self.include_input else 0
        return base + 2 * int(self.freq_bands.numel())

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        coord = coord.unsqueeze(-1)
        scaled = coord * self.freq_bands
        features = [torch.sin(scaled), torch.cos(scaled)]
        if self.include_input:
            features.insert(0, coord)
        return torch.cat(features, dim=-1)


class GalaxySpectralField(nn.Module):
    """Neural field mapping (x, y, log_lambda) -> flux."""

    def __init__(
        self, hidden_dim: int = 256, n_layers: int = 4, softplus_beta: float = 2.0,
        positional_encoding: Mapping[str, Any] | None = None, output_head: str = "softplus",
    ):
        super().__init__()
        layers = []
        in_dim = 3
        self.softplus_beta = float(softplus_beta)
        self.output_head = str(output_head or "softplus").strip().lower()
        valid_heads = {"softplus", "zero_centered_softplus"}
        if self.output_head not in valid_heads:
            raise ValueError(
                f"Unsupported output_head={output_head!r}; expected one of {sorted(valid_heads)}."
            )
        self.xy_encoder: FourierFeatureEncoding | None = None
        self.lambda_encoder: FourierFeatureEncoding | None = None
        if positional_encoding and positional_encoding.get("enabled", False):
            n_freq_xy = int(positional_encoding.get("n_frequencies_xy", 4))
            include_input_xy = bool(positional_encoding.get("include_input_xy", True))
            n_freq_lambda = int(positional_encoding.get("n_frequencies_lambda", n_freq_xy))
            include_input_lambda = bool(positional_encoding.get("include_input_lambda", include_input_xy))
            self.xy_encoder = FourierFeatureEncoding(n_freq_xy, include_input_xy)
            self.lambda_encoder = FourierFeatureEncoding(n_freq_lambda, include_input_lambda)
            in_dim = self.xy_encoder.output_dim * 2 + self.lambda_encoder.output_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=self.softplus_beta)

    def forward(self, x, y, log_lam):
        if self.xy_encoder is not None and self.lambda_encoder is not None:
            x_feat = self.xy_encoder(x)
            y_feat = self.xy_encoder(y)
            lam_feat = self.lambda_encoder(log_lam)
            coords = torch.cat([x_feat, y_feat, lam_feat], dim=-1)
        else:
            coords = torch.stack([x, y, log_lam], dim=-1)
        h = self.network(coords)
        raw = h.squeeze(-1)
        if self.output_head == "zero_centered_softplus":
            zero = math.log(2.0) / self.softplus_beta
            return F.softplus(raw, beta=self.softplus_beta) - zero
        return self.softplus(raw)
