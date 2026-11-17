"""Learned non-negative SED template dictionary shared across splats."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


OFF_GRID_MODES = ("clamp", "zero")


def _rng_reference_count(n_templates: int, reference: int | None) -> int:
    """Rows to draw so the global RNG advances as for a reference-rank model.

    A smaller dictionary then leaves every later draw (spatial seeding jitter,
    refinement) on the same stream as the reference rank, and its random
    initial values are the reference draw's leading templates.
    """
    if reference is None:
        return int(n_templates)
    if isinstance(reference, bool) or int(reference) != reference or int(reference) < int(n_templates):
        raise ValueError(
            f"rng_reference_templates must be an integer >= n_templates; got {reference!r}."
        )
    return int(reference)


class SEDDictionary(nn.Module):
    """K non-negative spectral templates on a fixed log-lambda grid.

    Splats mix templates with softplus-positive coefficients, so every SED in
    the model lives in the K-dimensional cone spanned by the learned templates.

    off_grid selects what a query outside the grid reads. "clamp" holds the
    terminal knot value, which routes off-grid amplitude gradient onto knots no
    measurement functional constrains. "zero" returns exactly zero there, so an
    off-grid query neither contributes flux nor receives gradient.

    pin_gauge_knots ties each terminal knot to its interior neighbour. The
    parameter tensor keeps its shape, so the pin costs no state-dict key and no
    grid node; the two pinned columns simply stop being free directions.

    pad_knots_blue / pad_knots_red extend the stored grid past the given
    (data-span) grid by that many knots per side at the same spacing. Padding
    knots carry learned values reachable only through per-splat shifts, so a
    shifted band reads real template values instead of hitting the off_grid
    boundary rule at the edge of throughput support; the boundary rule still
    applies, just past the padding. Template init is anchored to the data span
    so the interior initialization does not depend on the padding.
    """

    def __init__(
        self,
        n_templates: int,
        loglam_grid: torch.Tensor,
        *,
        off_grid: str = "clamp",
        pin_gauge_knots: bool = False,
        pad_knots_blue: int = 0,
        pad_knots_red: int = 0,
        rng_reference_templates: int | None = None,
    ):
        super().__init__()
        rng_templates = _rng_reference_count(n_templates, rng_reference_templates)
        if loglam_grid.ndim != 1 or loglam_grid.numel() < 2:
            raise ValueError("loglam_grid must be a 1D tensor with >= 2 nodes.")
        if bool(torch.any(loglam_grid[1:] <= loglam_grid[:-1])):
            raise ValueError("loglam_grid must be strictly increasing.")
        if off_grid not in OFF_GRID_MODES:
            raise ValueError(
                f"off_grid must be one of {OFF_GRID_MODES}; got {off_grid!r}."
            )
        for label, value in (
            ("pad_knots_blue", pad_knots_blue), ("pad_knots_red", pad_knots_red),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{label} must be a non-negative integer; got {value!r}."
                )
        if pad_knots_blue or pad_knots_red:
            steps = loglam_grid[1:] - loglam_grid[:-1]
            spacing = (loglam_grid[-1] - loglam_grid[0]) / (loglam_grid.numel() - 1)
            # float32 linspace nodes carry ~4e-6 relative spacing jitter at 96
            # knots; the check only needs to reject genuinely ragged grids
            if bool(torch.any((steps - spacing).abs() > 1e-4 * spacing.abs())):
                raise ValueError(
                    "Grid padding requires a uniformly spaced loglam_grid."
                )
            blue = loglam_grid[0] - spacing * torch.arange(
                pad_knots_blue, 0, -1, dtype=loglam_grid.dtype
            )
            red = loglam_grid[-1] + spacing * torch.arange(
                1, pad_knots_red + 1, dtype=loglam_grid.dtype
            )
            loglam_grid = torch.cat((blue, loglam_grid, red))
        if pin_gauge_knots and loglam_grid.numel() < 3:
            raise ValueError(
                "pin_gauge_knots needs >= 3 grid nodes: each terminal knot "
                "reads an interior neighbour."
            )
        self.off_grid = str(off_grid)
        self.pin_gauge_knots = bool(pin_gauge_knots)
        self.pad_knots_blue = int(pad_knots_blue)
        self.pad_knots_red = int(pad_knots_red)
        self.register_buffer("loglam_grid", loglam_grid.clone())
        self.register_buffer("loglam_scale", torch.tensor(1.0))
        # Symmetry breaking matters here. If every template starts at the same
        # value, then with coefficients also equal each template receives an
        # identical gradient (dL/dT_k = sum_j softplus(c_jk) dL/dS_j) and they
        # stay identical forever: the dictionary collapses to rank 1 and every
        # splat is forced to share one spectral shape, so a field containing
        # both red and blue sources cannot be fitted at all.
        # Start instead from distinct spectral slopes spanning blue to red.
        # The slope coordinate is anchored to the data span, not the padded
        # span, so padding extends the same exponentials smoothly outward
        # instead of rescaling the interior initialization.
        lam = loglam_grid.clone()
        data_lo = lam[self.pad_knots_blue]
        data_hi = lam[lam.numel() - 1 - self.pad_knots_red]
        span = (data_hi - data_lo).clamp_min(1e-6)
        lam_n = 2 * (lam - data_lo) / span - 1.0
        slopes = torch.linspace(-2.0, 2.0, n_templates)
        shape = torch.exp(slopes[:, None] * lam_n[None, :])
        shape = shape / shape.mean(dim=1, keepdim=True)
        self.raw_templates = nn.Parameter(
            torch.log(torch.expm1(shape.clamp_min(1e-3)))
            + 0.01 * torch.randn(rng_templates, lam.numel())[:n_templates].contiguous()
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "loglam_scale"
        missing_scale = key not in state_dict
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if missing_scale:
            warnings.warn(
                "Loaded an old SEDDictionary checkpoint without "
                "'loglam_scale'; preserving the module's initialized value.",
                UserWarning,
                stacklevel=2,
            )
            if key in missing_keys:
                missing_keys.remove(key)

    def templates(self) -> torch.Tensor:
        T = F.softplus(self.raw_templates)
        if not self.pin_gauge_knots:
            return T
        interior = T[:, 1:-1]
        return torch.cat((interior[:, :1], interior, interior[:, -1:]), dim=1)

    def templates_at(self, loglam: torch.Tensor) -> torch.Tensor:
        """Interpolate linearly in wavelength on log-spaced knots."""
        T = self.templates()
        grid = self.loglam_grid
        if self.off_grid == "zero":
            on_grid = (loglam >= grid[0]) & (loglam <= grid[-1])
        loglam = loglam.clamp(grid[0], grid[-1])
        idx = torch.clamp(
            torch.searchsorted(grid, loglam.contiguous()), 1, grid.numel() - 1
        )
        lam = torch.exp(self.loglam_scale * loglam)
        lam_grid = torch.exp(self.loglam_scale * grid)
        lo, hi = lam_grid[idx - 1], lam_grid[idx]
        w = ((lam - lo) / (hi - lo)).clamp(0.0, 1.0)
        values = T[:, idx - 1] * (1 - w) + T[:, idx] * w
        if self.off_grid == "zero":
            # torch.where masks the gradient as well as the value, so an
            # off-grid query reaches no knot in the backward pass either.
            return torch.where(on_grid, values, values.new_zeros(()))
        return values

    def set_loglam_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("loglam scale must be positive and finite.")
        self.loglam_scale.fill_(float(scale))

    def eval(
        self,
        coeffs: torch.Tensor,
        loglam: torch.Tensor,
        shift: torch.Tensor | None = None,
        max_elements: int = 50_000_000,
    ) -> torch.Tensor:
        """coeffs: (n_splats, K) raw; loglam: (n_query,); shift: optional
        (n_splats,) per-splat translation in loglam (redshift: log(1+z)).
        Templates are stored in the rest frame and evaluated at
        loglam - shift_j for splat j. Returns (n_splats, n_query).

        The shifted branch is the only place in the model that allocates
        (K, n_splats, n_query); the result it returns is a factor of K smaller,
        so callers cannot see how wide the intermediate was and every one of
        them used to size its own chunks as if this cost (n_splats, n_query).
        Bound it here instead, by chunking over splats, so callers can budget on
        the shape they actually receive.
        """
        if shift is None:
            return F.softplus(coeffs) @ self.templates_at(loglam)
        J, n = shift.numel(), loglam.numel()
        k = int(self.raw_templates.shape[0])
        step = max(1, int(max_elements // max(k * n, 1)))
        if J <= step:
            return self._eval_shifted(coeffs, loglam, shift)
        # Checkpoint each chunk. Chunking alone bounds the transient allocation
        # but not the graph: under autograd every chunk's (K, chunk, n_query)
        # activations are retained, so at 200k splats the ~128 chunks per band
        # still add up to the same tens of GiB and the step dies holding 94 GiB
        # while asking for 192 MiB.
        parts = []
        for s in range(0, J, step):
            args = (coeffs[s : s + step], loglam, shift[s : s + step])
            if torch.is_grad_enabled():
                parts.append(
                    checkpoint(self._eval_shifted, *args, use_reentrant=False)
                )
            else:
                parts.append(self._eval_shifted(*args))
        return torch.cat(parts, dim=0)

    def _eval_shifted(
        self, coeffs: torch.Tensor, loglam: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        J, n = shift.numel(), loglam.numel()
        lam = loglam[None, :] - shift[:, None]               # (J, n)
        T_at = self.templates_at(lam.reshape(-1))            # (K, J*n)
        T_at = T_at.reshape(-1, J, n)                        # (K, J, n)
        return torch.einsum("jk,kjn->jn", F.softplus(coeffs), T_at)
