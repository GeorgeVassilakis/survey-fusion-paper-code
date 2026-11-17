"""Residual-guided birth statistics.

Importance-weighted per-cell accumulator over the normalized model tangent
plane. During training, each sampled image datum contributes its source-only
standardized residual to a coarse (PSF-scale) grid, weighted by the same
sampling correction the NLL uses; with-replacement duplicates enter through
the squared-weight sum instead of inflating an unweighted count. Per cell and
band the accumulator keeps

    S = sum w * r     W = sum w      Q = sum w^2
    M = sum w * m     R = sum w * f

with r = (y - mu)/sigma (mu includes the known additive sky, so r is the
source deficit plus calibration error), m = (mu - additive)/sigma the model's
own rendered contribution, and f = y - mu the raw residual flux. Cell
z-score = S / sqrt(Q) (self-normalized Horvitz-Thompson); effective sample
size W^2 / Q gates thin cells; M against S vetoes cells the census already
explains (a wrong-SED or near-zero splat whose geometric footprint covers the
cell contributes nothing to M and cannot suppress a birth there).

Both z and the ESS gate are draw-level (Kish): repeated draws of ONE fixed
pixel count as repeated evidence, so neither statistic bounds locally
concentrated duplication. These are screening statistics; the offline
protocol (null calibration on real sampling, held-out exposures) is the
arbiter before any birth trains. The grid is also run-scoped and not
checkpointed: a mid-cadence resume restarts the current interval's
evidence.

The z-score is exact when each datum enters at most once per refine
interval. Under heavy with-replacement duplication (bright cells sampled
every step), a duplicated residual re-enters without fresh noise and z
overstates significance by up to sqrt(mean duplication); the production
z_min threshold must therefore be set from the offline null-field
calibration, which measures exactly this inflation, never assumed to be
in unit-normal units.

Exposure-level coherence is NOT accumulated online; the offline validation
protocol (held-out exposures) carries that gate. Statistics reset at each
refine event.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

_STATS = ("S", "W", "Q", "M", "R")


class ResidualBirthAccumulator:
    def __init__(
        self,
        *,
        cell_size: float,
        bands: Sequence[str],
        device: torch.device | str,
    ) -> None:
        if not math.isfinite(cell_size) or cell_size <= 0 or cell_size >= 2.0:
            raise ValueError(
                f"cell_size must be a normalized length in (0, 2), got {cell_size!r}."
            )
        if not bands:
            raise ValueError("At least one band is required.")
        self.cell_size = float(cell_size)
        self.side = int(math.ceil(2.0 / self.cell_size))
        self.bands = tuple(str(b) for b in bands)
        self.band_index = {b: i for i, b in enumerate(self.bands)}
        self.device = torch.device(device)
        n = len(self.bands) * self.side * self.side
        for name in _STATS:
            setattr(
                self, name, torch.zeros(n, device=self.device, dtype=torch.float32)
            )

    def reset(self) -> None:
        for name in _STATS:
            getattr(self, name).zero_()

    def _flat_cells(
        self, band_ids: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor
    ) -> torch.Tensor:
        cx = ((xs + 1.0) / self.cell_size).long().clamp_(0, self.side - 1)
        cy = ((ys + 1.0) / self.cell_size).long().clamp_(0, self.side - 1)
        return (band_ids * self.side + cy) * self.side + cx

    @torch.no_grad()
    def accumulate(
        self,
        band_ids: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        residual_over_sigma: torch.Tensor,
        model_over_sigma: torch.Tensor,
        residual_flux: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        n = int(xs.numel())
        for tensor in (band_ids, ys, residual_over_sigma, model_over_sigma,
                       residual_flux, weights):
            if int(tensor.numel()) != n:
                raise ValueError("Birth accumulator inputs must share length.")
        if n == 0:
            return
        flat = self._flat_cells(
            band_ids.to(self.device), xs.to(self.device), ys.to(self.device)
        )
        w = weights.to(self.device, dtype=torch.float32)
        r = residual_over_sigma.to(self.device, dtype=torch.float32)
        m = model_over_sigma.to(self.device, dtype=torch.float32)
        f = residual_flux.to(self.device, dtype=torch.float32)
        self.S.index_add_(0, flat, w * r)
        self.W.index_add_(0, flat, w)
        self.Q.index_add_(0, flat, w * w)
        self.M.index_add_(0, flat, w * m)
        self.R.index_add_(0, flat, w * f)

    @torch.no_grad()
    def candidates(
        self,
        *,
        z_min: float,
        n_eff_min: float,
        kappa: float,
        min_bands: int = 2,
        top_k: int,
    ) -> dict[str, torch.Tensor] | None:
        """Cells with coherent unexplained excess, ranked by significance.

        Returns cell centers, per-band z, per-band HT-mean residual flux, and
        the triggering-band count; None when nothing qualifies."""
        n_bands = len(self.bands)
        cells = self.side * self.side
        shape = (n_bands, cells)
        S = self.S.view(shape)
        W = self.W.view(shape)
        Q = self.Q.view(shape)
        M = self.M.view(shape)
        R = self.R.view(shape)
        z = S / Q.clamp_min(1e-30).sqrt()
        n_eff = W.square() / Q.clamp_min(1e-30)
        explained = M > float(kappa) * S
        eligible = (
            (z > float(z_min))
            & (n_eff >= float(n_eff_min))
            & ~explained
            & (Q > 0)
        )
        band_count = eligible.sum(dim=0)
        hit = band_count >= int(min_bands)
        if not bool(hit.any()):
            return None
        # rank by the weakest triggering band: every triggering band must be
        # individually convincing
        ranked_z = torch.where(eligible, z, torch.inf).amin(dim=0)
        cell_ids = torch.where(hit)[0]
        order = torch.argsort(ranked_z[cell_ids], descending=True)
        cell_ids = cell_ids[order][: int(top_k)]
        cy = (cell_ids // self.side).float()
        cx = (cell_ids % self.side).float()
        centers = torch.stack(
            [
                (cx + 0.5) * self.cell_size - 1.0,
                (cy + 0.5) * self.cell_size - 1.0,
            ],
            dim=1,
        )
        mean_residual_flux = R[:, cell_ids] / W[:, cell_ids].clamp_min(1e-30)
        return {
            "cell_ids": cell_ids,
            "centers": centers,
            "z": z[:, cell_ids],
            "eligible": eligible[:, cell_ids],
            "mean_residual_flux": mean_residual_flux,
            "band_count": band_count[cell_ids],
        }
