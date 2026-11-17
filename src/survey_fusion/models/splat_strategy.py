# Portions of this module are adapted from gsplat (https://github.com/nerfstudio-project/gsplat),
# files gsplat/strategy/default.py and gsplat/strategy/ops.py, which carry these notices:
#   SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#   SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
# The changes made to the upstream code are described in the module docstring below.
"""gsplat DefaultStrategy adapted to additive photometric splats.

Port of gsplat's densification strategy (strategy/default.py + strategy/ops.py
at gsplat main, 2026-07) with the deviations required by this model, each
noted inline:

- No opacity reset: the model has no opacity. Amplitudes are physical flux,
  so an invisible splat is one whose peak surface brightness is far below the
  pixel noise; that is the prune criterion (gsplat: sigmoid(opacity) < 0.005).
- Flux is halved when a splat is duplicated or split. gsplat repeats the
  opacity, which roughly preserves appearance under alpha compositing, but in
  an additive model repeating the amplitude doubles the photometry.
- Positional gradients are accumulated from mu.grad in model-frame pixel
  units (grad * field_width / 2), the same convention as gsplat's screen-space
  normalization. The grow threshold is auto-calibrated per run (see
  SplatRefineConfig): gsplat's absolute constant assumes mean-reduced image
  losses and does not transfer to an NLL objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import torch
import torch.nn.functional as F

from ..utils.perf_env import prune_only_fast_enabled

PER_SPLAT_PARAMS = (
    "mu",
    "log_scales",
    "theta",
    "raw_flux",
    "sed_coeffs",
    "sed_shift",
)

# Lineage buffers registered on the field; refine surgery keeps their rows
# aligned with the parameters through every duplicate/split/remove.
PER_SPLAT_LINEAGE_BUFFERS = (
    "halving_depth",
    "last_halving_event",
    "birth_event",
)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


@torch.no_grad()
def canonicalize_gauge_rows(
    sed_coeffs: torch.Tensor,
    raw_flux: torch.Tensor,
    gauge_factors: torch.Tensor,
    *,
    drift_log: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows outside the drift band, with their canonicalized coefficients.

    The gauge algebra alone, on plain tensors: SplatRefineStrategy.project_gauge
    applies it to a live model plus optimizer surgery, the checkpoint sidecar
    applies it to a detached copy with no optimizer at all."""
    positive = gauge_factors > 0
    drifted = positive & (gauge_factors.clamp_min(1e-300).log().abs() > drift_log)
    rows = torch.where(drifted)[0]
    if rows.numel() == 0:
        return (
            rows,
            sed_coeffs[rows],
            raw_flux[rows],
        )
    # fp64 transform: softplus of very negative coefficients underflows
    # fp32 long before it loses meaning in the gauge algebra
    factor = gauge_factors[rows].to(torch.float64)
    coeffs = sed_coeffs[rows].to(torch.float64)
    flux = raw_flux[rows].to(torch.float64)
    new_coeffs = _inv_softplus(F.softplus(coeffs) / factor[:, None]).to(
        sed_coeffs.dtype
    )
    new_flux = _inv_softplus(factor * F.softplus(flux)).to(raw_flux.dtype)
    # Atomicity: an extreme but admissible factor (a dying SED near the
    # fp32 subnormal floor) can push the transform outside the parameter
    # dtype. Such rows are skipped wholesale BEFORE any mutation; they
    # are exactly the rows the significance prune exists to remove.
    representable = (
        torch.isfinite(new_coeffs).all(dim=1) & torch.isfinite(new_flux)
    )
    if not bool(representable.all()):
        rows = rows[representable]
        new_coeffs = new_coeffs[representable]
        new_flux = new_flux[representable]
    return rows, new_coeffs, new_flux


def psf_based_grow_scale(
    psf_fwhm_arcsec: float, field_width_px: int, pixel_scale_arcsec: float,
) -> float:
    """Duplicate/split size boundary in normalized units: 3x the PSF FWHM."""
    # The lifecycle uses a legacy scalar FWHM even when the forward operator is
    # a broad mixture.
    norm_per_arcsec = 2.0 / (float(field_width_px) * float(pixel_scale_arcsec))
    return 3.0 * float(psf_fwhm_arcsec) * norm_per_arcsec


@dataclass
class SplatRefineConfig:
    # Scale thresholds are relative to the normalized scene half-extent
    # (scene_scale = 1 for the [-1, 1] field).
    # grow_grad2d = 0 auto-calibrates at the first refine to the
    # (1 - grow_target_frac) quantile of visited mean gradients, then holds.
    grow_grad2d: float = 0.0
    grow_target_frac: float = 0.05
    # grow_scale = 0 resolves to psf_based_grow_scale at trainer construction.
    grow_scale: float = 0.0
    prune_scale: float = 0.1
    # Delete a splat whose major/minor scale ratio exceeds this, whatever
    # flux it carries; None disables the shape test (it deleted fitted
    # sources in earlier runs).
    prune_axis_ratio: float | None = None
    prune_peak_sb_frac: float = 0.005
    # When true, the peak-SB prune tests the RENDERED brightness: peak
    # surface brightness in the splat's best band over that band's image
    # noise, instead of the bandless raw_flux amplitude over one pooled
    # sigma. The raw_flux/SED normalization is gauge-degenerate, so the
    # bandless test can prune splats that render far above threshold.
    # Requires a band_snr_fn on the strategy.
    prune_band_snr: bool = False
    # Kill threshold for the band-SNR criterion, in peak-SNR sigma units:
    # a splat dies when its PSF-convolved rendered peak in its best band is
    # below this many sigma of that band's image noise. prune_peak_sb_frac is
    # a fraction of one pooled reference sigma, a different unit, so the two
    # thresholds are not interchangeable; this one has no default and must be
    # set per run whenever prune_band_snr is on.
    prune_band_snr_sigma: float | None = None
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    max_splats: int = 0  # 0 = unlimited
    # Residual-guided birth. birth_cap_per_event = 0 disables
    # the stage; births are budgeted before growth (they are the only
    # mechanism serving unmodeled regions and would otherwise be starved by
    # top-grad growth). Born splats are exempt from the flux prune while
    # step - birth_event <= birth_grace_steps: with grace equal to the
    # refine cadence, a newborn is exempt at the first event after its
    # birth and first judged at the second.
    birth_cap_per_event: int = 0
    birth_grace_steps: int = 100


# Every per-row field a birth must define; a partial row is not a splat.
BIRTH_ROW_FIELDS = (
    "mu",
    "log_scales",
    "theta",
    "raw_flux",
    "sed_coeffs",
    "sed_shift",
)


@dataclass
class SplatRefinePlan:
    step: int
    duplicate: torch.BoolTensor
    split: torch.BoolTensor
    prune_after_growth: torch.BoolTensor
    split_noise: torch.FloatTensor
    threshold: float | None
    # Residual-guided births: None, or a mapping with one tensor per
    # BIRTH_ROW_FIELDS entry, all sharing leading dimension k. Born rows are
    # appended after growth and removal with fresh optimizer state and
    # birth_event = step (a condemned donor is simply removed and a fresh row
    # appended: identical census arithmetic, no index remapping through the
    # growth ordering).
    birth: Mapping[str, torch.Tensor] | None = None
    # Reason-separated prune counts (flux / axis_ratio / scale), populated by
    # make_plan for logging. Rank-local diagnostics only: not part of the
    # broadcast payload, so plans rebuilt via from_payload carry None.
    prune_reasons: dict | None = None

    def to_payload(self) -> dict[str, object]:
        self._validate_structure()
        return {
            "step": self.step,
            "duplicate": self.duplicate.detach().cpu().clone(),
            "split": self.split.detach().cpu().clone(),
            "prune_after_growth": self.prune_after_growth.detach().cpu().clone(),
            "split_noise": self.split_noise.detach().cpu().clone(),
            "threshold": self.threshold,
            "birth": (
                None
                if self.birth is None
                else {
                    name: self.birth[name].detach().cpu().clone()
                    for name in BIRTH_ROW_FIELDS
                }
            ),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], device: torch.device | str
    ) -> SplatRefinePlan:
        if not isinstance(payload, Mapping):
            raise TypeError("Splat refinement payload must be a mapping.")
        expected = {
            "step",
            "duplicate",
            "split",
            "prune_after_growth",
            "split_noise",
            "threshold",
            "birth",
        }
        if set(payload) != expected:
            raise ValueError(
                f"Splat refinement payload keys must be exactly {sorted(expected)}."
            )
        step = payload["step"]
        threshold = payload["threshold"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("Splat refinement plan step must be an integer.")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise TypeError(
                    "Splat refinement threshold must be a real number or None."
                )
            threshold = float(threshold)
        tensors = {}
        for name in ("duplicate", "split", "prune_after_growth", "split_noise"):
            value = payload[name]
            if not torch.is_tensor(value):
                raise TypeError(f"Splat refinement plan {name} must be a tensor.")
            tensors[name] = value.detach().to(device=device).clone()
        birth_payload = payload["birth"]
        birth = None
        if birth_payload is not None:
            if not isinstance(birth_payload, Mapping) or set(birth_payload) != set(
                BIRTH_ROW_FIELDS
            ):
                raise ValueError(
                    f"Splat birth payload keys must be exactly {sorted(BIRTH_ROW_FIELDS)}."
                )
            birth = {}
            for name in BIRTH_ROW_FIELDS:
                value = birth_payload[name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Splat birth {name} must be a tensor.")
                birth[name] = value.detach().to(device=device).clone()
        plan = cls(
            step=step,
            duplicate=tensors["duplicate"],
            split=tensors["split"],
            prune_after_growth=tensors["prune_after_growth"],
            split_noise=tensors["split_noise"],
            threshold=threshold,
            birth=birth,
        )
        plan._validate_structure()
        return plan

    def _validate_structure(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise TypeError("Splat refinement plan step must be an integer.")
        if self.threshold is not None:
            if isinstance(self.threshold, bool) or not isinstance(
                self.threshold, Real
            ):
                raise TypeError(
                    "Splat refinement threshold must be a real number or None."
                )
            if not math.isfinite(float(self.threshold)):
                raise ValueError("Splat refinement threshold must be finite.")
        for name in ("duplicate", "split", "prune_after_growth"):
            value = getattr(self, name)
            if not torch.is_tensor(value):
                raise TypeError(f"Splat refinement plan {name} must be a tensor.")
            if value.dtype != torch.bool or value.ndim != 1:
                raise ValueError(
                    f"Splat refinement plan {name} must be a one-dimensional "
                    "bool tensor."
                )
        if self.duplicate.shape != self.split.shape:
            raise ValueError("Duplicate and split masks must have the same length.")
        if bool((self.duplicate & self.split).any()):
            raise ValueError("A splat cannot be both duplicated and split.")
        expected_prune = (
            self.duplicate.numel()
            + int(self.duplicate.sum())
            + int(self.split.sum())
        )
        if self.prune_after_growth.numel() != expected_prune:
            raise ValueError(
                "Post-growth prune mask length does not match the planned growth."
            )
        expected_noise = (2, int(self.split.sum()), 2)
        if self.split_noise.shape != expected_noise:
            raise ValueError(
                f"Split noise must have shape {expected_noise}, "
                f"got {tuple(self.split_noise.shape)}."
            )
        if not torch.is_floating_point(self.split_noise):
            raise ValueError("Split noise must be floating point.")
        if not bool(torch.isfinite(self.split_noise).all()):
            raise ValueError("Split noise must be finite.")
        if self.birth is not None:
            if not isinstance(self.birth, Mapping) or set(self.birth) != set(
                BIRTH_ROW_FIELDS
            ):
                raise ValueError(
                    f"Splat birth fields must be exactly {sorted(BIRTH_ROW_FIELDS)}."
                )
            k = None
            for name in BIRTH_ROW_FIELDS:
                value = self.birth[name]
                if not torch.is_tensor(value) or not torch.is_floating_point(value):
                    raise ValueError(f"Splat birth {name} must be a float tensor.")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Splat birth {name} must be finite.")
                rows = int(value.shape[0]) if value.ndim else -1
                if k is None:
                    k = rows
                elif rows != k:
                    raise ValueError(
                        "Splat birth fields must share their leading dimension."
                    )
            if k == 0:
                raise ValueError("Splat birth must carry at least one row or be None.")
            if self.birth["mu"].ndim != 2 or self.birth["mu"].shape[1] != 2:
                raise ValueError("Splat birth mu must have shape (k, 2).")
            if (
                self.birth["log_scales"].ndim != 2
                or self.birth["log_scales"].shape[1] != 2
            ):
                raise ValueError("Splat birth log_scales must have shape (k, 2).")
            for name in ("theta", "raw_flux", "sed_shift"):
                if self.birth[name].ndim != 1:
                    raise ValueError(f"Splat birth {name} must be one-dimensional.")
            if self.birth["sed_coeffs"].ndim != 2:
                raise ValueError("Splat birth sed_coeffs must have shape (k, K).")


class SplatRefineStrategy:
    """Densify and prune a SplatField in place, with optimizer-state surgery."""

    def __init__(
        self,
        config: SplatRefineConfig,
        *,
        field_width: int,
        reference_sigma: float,
        band_snr_fn=None,
    ) -> None:
        if reference_sigma <= 0 or not math.isfinite(reference_sigma):
            raise ValueError("reference_sigma must be a finite positive noise level.")
        if config.prune_band_snr and band_snr_fn is None:
            raise ValueError(
                "prune_band_snr requires a band_snr_fn mapping the model to "
                "the per-splat PSF-convolved rendered peak significance "
                "(max over bands of rendered peak over band image sigma)."
            )
        if config.prune_band_snr:
            if config.prune_band_snr_sigma is None:
                raise ValueError(
                    "prune_band_snr requires prune_band_snr_sigma, the kill "
                    "threshold in peak-SNR sigma units. prune_peak_sb_frac is "
                    "a fraction of one pooled reference sigma and does not "
                    "carry over; set the band-SNR threshold explicitly."
                )
            sigma_threshold = float(config.prune_band_snr_sigma)
            if not math.isfinite(sigma_threshold) or sigma_threshold < 0.0:
                raise ValueError(
                    "prune_band_snr_sigma must be finite and non-negative "
                    f"(got {config.prune_band_snr_sigma!r})."
                )
        elif config.prune_band_snr_sigma is not None:
            raise ValueError(
                "prune_band_snr_sigma is set but prune_band_snr is false; the "
                "threshold would be silently ignored. Enable prune_band_snr "
                "or drop the threshold."
            )
        self.config = config
        self.field_width = int(field_width)
        self.reference_sigma = float(reference_sigma)
        self.band_snr_fn = band_snr_fn
        self.grad_accum: torch.Tensor | None = None
        self.count: torch.Tensor | None = None
        self.threshold: float | None = None

    # --- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialize the run-scoped refinement state for checkpointing.

        The auto-calibrated grow threshold is set once, at the first refine,
        from the gradient scale of a barely-trained model. A resumed run that
        recalibrates from mid-training gradients gets a much lower threshold
        and grows differently than the uninterrupted run would have."""
        return {
            "threshold": None if self.threshold is None else float(self.threshold),
            "grad_accum": (
                None if self.grad_accum is None else self.grad_accum.detach().cpu()
            ),
            "count": None if self.count is None else self.count.detach().cpu(),
        }

    def load_state_dict(self, state: dict, *, model) -> None:
        threshold = state.get("threshold")
        self.threshold = None if threshold is None else float(threshold)
        grad_accum = state.get("grad_accum")
        count = state.get("count")
        if (grad_accum is None) != (count is None):
            raise ValueError(
                "Refinement state must carry grad_accum and count together."
            )
        if grad_accum is None:
            self.grad_accum = None
            self.count = None
            return
        n = int(model.mu.shape[0])
        if grad_accum.shape != (n,) or count.shape != (n,):
            raise ValueError(
                f"Refinement state accumulators have shape {tuple(grad_accum.shape)} "
                f"but the model holds {n} splats."
            )
        device = model.mu.device
        self.grad_accum = grad_accum.to(device=device)
        self.count = count.to(device=device)

    # --- gradient accumulation -------------------------------------------
    @torch.no_grad()
    def accumulate(self, model) -> None:
        """Record this step's positional gradients. Call after backward and
        before any gradient clipping, as gsplat does."""
        grad = model.mu.grad
        if grad is None:
            return
        n = int(model.mu.shape[0])
        device = model.mu.device
        if self.grad_accum is None or self.grad_accum.shape[0] != n:
            self.grad_accum = torch.zeros(n, device=device)
            self.count = torch.zeros(n, device=device)
        pixel_scale = 0.5 * self.field_width
        if grad.is_sparse:
            grad = grad.coalesce()
            ids = grad.indices()[0]
            norms = (grad.values() * pixel_scale).norm(dim=-1)
        else:
            norms = (grad * pixel_scale).norm(dim=-1)
            ids = torch.nonzero(norms > 0, as_tuple=True)[0]
            norms = norms[ids]
        self.grad_accum.index_add_(0, ids, norms)
        self.count.index_add_(0, ids, torch.ones_like(ids, dtype=self.count.dtype))

    # --- refine ----------------------------------------------------------
    @staticmethod
    def _cadence_matches(config: SplatRefineConfig, step: int) -> bool:
        return step > config.refine_start and step % config.refine_every == 0

    @staticmethod
    def _candidate_passthrough(
        value: torch.Tensor,
        duplicate: torch.Tensor,
        split: torch.Tensor,
    ) -> torch.Tensor:
        """Map a per-splat vector onto the post-growth candidate ordering.

        Children inherit the parent's value unchanged; must mirror the row
        ordering of _candidate_after_growth exactly."""
        duplicate_ids = torch.where(duplicate)[0]
        after_duplicate = torch.cat([value, value[duplicate_ids]])
        split_after_duplicate = torch.cat(
            [
                split,
                torch.zeros(
                    duplicate_ids.numel(), dtype=torch.bool, device=split.device
                ),
            ]
        )
        split_ids = torch.where(split_after_duplicate)[0]
        rest = torch.where(~split_after_duplicate)[0]
        repeats = [2] + [1] * (after_duplicate.dim() - 1)
        return torch.cat(
            [after_duplicate[rest], after_duplicate[split_ids].repeat(repeats)]
        )

    @staticmethod
    def _candidate_after_growth(
        model,
        duplicate: torch.Tensor,
        split: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        duplicate_ids = torch.where(duplicate)[0]
        split_after_duplicate = torch.cat(
            [
                split,
                torch.zeros(
                    duplicate_ids.numel(),
                    dtype=torch.bool,
                    device=split.device,
                ),
            ]
        )
        split_ids = torch.where(split_after_duplicate)[0]
        rest = torch.where(~split_after_duplicate)[0]
        candidate = {}
        for name in ("raw_flux", "log_scales", "theta"):
            value = getattr(model, name).detach()
            duplicate_extra = value[duplicate_ids]
            if name == "raw_flux":
                duplicate_half = _inv_softplus(
                    (0.5 * F.softplus(value[duplicate_ids])).clamp_min(1e-30)
                )
                value = value.clone()
                value[duplicate_ids] = duplicate_half
                duplicate_extra = duplicate_half
            after_duplicate = torch.cat([value, duplicate_extra])
            repeats = [2] + [1] * (after_duplicate.dim() - 1)
            if name == "log_scales":
                children = (after_duplicate[split_ids] - math.log(1.6)).repeat(
                    repeats
                )
            elif name == "raw_flux":
                children = _inv_softplus(
                    (0.5 * F.softplus(after_duplicate[split_ids])).clamp_min(1e-30)
                ).repeat(repeats)
            else:
                children = after_duplicate[split_ids].repeat(repeats)
            candidate[name] = torch.cat([after_duplicate[rest], children])
        return candidate

    def _flux_prune_mask(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        band_snr_factor: torch.Tensor | None = None,
    ) -> torch.BoolTensor:
        cfg = self.config
        if band_snr_factor is not None:
            # band_snr_fn returns the PSF-convolved rendered peak significance
            # (max over bands); it already folds in flux, SED, covariance,
            # per-band PSF variance, and per-band noise, so the verdict is a
            # direct threshold in sigma units. Gauge-invariant and PSF-limited:
            # a shrinking compact splat cannot inflate its own significance.
            # The threshold is non-None for every strategy whose config has
            # prune_band_snr set; the constructor refuses any other pairing.
            return band_snr_factor < cfg.prune_band_snr_sigma
        scales = torch.exp(params["log_scales"])
        theta = params["theta"]
        c, snt = torch.cos(theta), torch.sin(theta)
        rotation = torch.stack(
            [torch.stack([c, -snt], -1), torch.stack([snt, c], -1)], -2
        )
        covariance = (
            rotation
            @ torch.diag_embed(scales**2)
            @ rotation.transpose(-1, -2)
        )
        peak_sb = F.softplus(params["raw_flux"]) / (
            2.0 * math.pi * torch.sqrt(torch.linalg.det(covariance))
        )
        return peak_sb < cfg.prune_peak_sb_frac * self.reference_sigma

    def _axis_ratio_prune_mask(self, log_scales: torch.Tensor) -> torch.BoolTensor:
        if self.config.prune_axis_ratio is None:
            return torch.zeros(log_scales.shape[0], device=log_scales.device, dtype=torch.bool)
        scales = torch.exp(log_scales)
        return scales.amax(dim=1) > self.config.prune_axis_ratio * scales.amin(dim=1)

    def _scale_prune_mask(self, log_scales: torch.Tensor) -> torch.BoolTensor:
        return torch.exp(log_scales).amax(dim=1) > self.config.prune_scale

    @torch.no_grad()
    def make_plan(
        self,
        model,
        step: int,
        *,
        birth_rows: Mapping[str, torch.Tensor] | None = None,
    ) -> SplatRefinePlan | None:
        cfg = self.config
        if not self._cadence_matches(cfg, step):
            return None
        n_splats = int(model.mu.shape[0])

        # Life-or-death verdicts come FIRST, on the pre-growth census at
        # trained, un-halved fluxes: growth must never change a flux-based
        # prune outcome. Reproduction halves flux, so a post-growth verdict
        # would kill a parent in [1x, 2x) of threshold together with its
        # child in the same event. Axis ratio is judged pre-growth too:
        # splitting shifts both axes by the same log factor, so it cannot
        # rescue an axis-ratio failure and would only spend budget on two
        # equally condemned children. prune_scale stays post-growth because
        # splitting genuinely shrinks the max scale (split-rescues-oversized).
        band_snr_factor = None
        if cfg.prune_band_snr:
            factor = self.band_snr_fn(model)
            if not torch.is_tensor(factor) or factor.shape != (n_splats,):
                raise ValueError(
                    "band_snr_fn must return one factor per splat "
                    f"(got shape {getattr(factor, 'shape', None)} for {n_splats} splats)."
                )
            if not bool(torch.isfinite(factor).all()) or bool((factor < 0).any()):
                raise ValueError("band_snr_fn factors must be finite and non-negative.")
            band_snr_factor = factor.detach()
        pre_growth = {
            "raw_flux": model.raw_flux.detach(),
            "log_scales": model.log_scales.detach(),
            "theta": model.theta.detach(),
        }
        flux_condemned = self._flux_prune_mask(
            pre_growth, band_snr_factor=band_snr_factor
        )
        birth_event = getattr(model, "birth_event", None)
        if birth_event is not None:
            # birth grace: a newborn has no trained SED/shift yet; it faces
            # the flux criterion only after one full training interval
            in_grace = (birth_event >= 0) & (
                step - birth_event.to(torch.int64) <= int(cfg.birth_grace_steps)
            )
            flux_condemned = flux_condemned & ~in_grace
        axis_condemned = self._axis_ratio_prune_mask(pre_growth["log_scales"])
        condemned = flux_condemned | axis_condemned

        # Births are budgeted before growth: they are the only mechanism that
        # can serve unmodeled regions and would otherwise be starved by
        # top-grad growth, which is abundant by construction. Requested rows
        # arrive ranked by significance, so trimming keeps the best.
        n_birth = 0 if birth_rows is None else int(birth_rows["mu"].shape[0])
        n_birth = min(n_birth, max(0, int(cfg.birth_cap_per_event)))
        if cfg.max_splats:
            n_free_pre = int(cfg.max_splats) - (n_splats - int(condemned.sum()))
            n_birth = max(0, min(n_birth, n_free_pre))
        birth = None
        if n_birth:
            birth = {
                name: birth_rows[name][:n_birth].detach()
                for name in BIRTH_ROW_FIELDS
            }

        duplicate = torch.zeros(
            n_splats, dtype=torch.bool, device=model.mu.device
        )
        split = torch.zeros_like(duplicate)
        threshold = self.threshold
        grow_active = step < cfg.refine_stop and self.grad_accum is not None
        if grow_active:
            if self.count is None:
                raise ValueError("Splat refinement count is missing.")
            if self.grad_accum.shape != (n_splats,) or self.count.shape != (
                n_splats,
            ):
                raise ValueError(
                    "Splat refinement statistics must match the model size."
                )
            grad_accum = self.grad_accum.to(device=model.mu.device)
            count = self.count.to(device=model.mu.device)
            grads = grad_accum / count.clamp_min(1)
            if threshold is None:
                if cfg.grow_grad2d > 0:
                    threshold = float(cfg.grow_grad2d)
                else:
                    visited = grads[count > 0]
                    if visited.numel() > 0:
                        threshold = float(
                            torch.quantile(
                                visited.float(),
                                1.0 - cfg.grow_target_frac,
                            )
                        )
                        print(
                            f"[refine] calibrated grow threshold {threshold:.6g} "
                            f"(1-{cfg.grow_target_frac} quantile of visited mean grads)",
                            flush=True,
                        )
            if threshold is not None:
                is_grad_high = grads > threshold
                max_scale = torch.exp(model.log_scales).amax(dim=1)
                is_small = max_scale <= cfg.grow_scale
                # Do not reproduce the doomed: condemned splats leave this
                # event dead whatever growth does, so growing them wastes
                # budget and, for duplicates, would newly halve a flux the
                # verdict was computed on.
                duplicate = is_grad_high & is_small & ~condemned
                split = is_grad_high & ~is_small & ~condemned
                if cfg.max_splats:
                    # One census equation: slots freed by this event's
                    # pre-growth condemnations are spendable this event,
                    # births first, growth takes the remainder.
                    budget = (
                        int(cfg.max_splats)
                        - (n_splats - int(condemned.sum()))
                        - n_birth
                    )
                    grow = duplicate | split
                    n_grow = int(grow.sum())
                    if budget <= 0:
                        duplicate = torch.zeros_like(duplicate)
                        split = torch.zeros_like(split)
                    elif n_grow > budget:
                        candidates = torch.where(grow)[0]
                        keep = candidates[
                            torch.topk(grads[candidates], budget).indices
                        ]
                        keep_mask = torch.zeros_like(grow)
                        keep_mask[keep] = True
                        duplicate = duplicate & keep_mask
                        split = split & keep_mask

        prune_only_fast = not grow_active and prune_only_fast_enabled()
        split_ids = torch.empty(0,device=split.device,dtype=torch.long) if prune_only_fast else torch.where(split)[0]
        if split_ids.numel():
            cov = model.covariances(split_ids)
            a = cov[:, 0, 0].clamp_min(1e-12)
            b = cov[:, 1, 0]
            l11 = torch.sqrt(a)
            l21 = b / l11
            l22 = torch.sqrt(
                (cov[:, 1, 1] - l21.square()).clamp_min(1e-12)
            )
            zeros = torch.zeros_like(l11)
            chol = torch.stack(
                [torch.stack([l11, zeros], -1), torch.stack([l21, l22], -1)], -2
            )
            split_noise = torch.einsum(
                "nij,bnj->bni",
                chol,
                torch.randn(2, split_ids.numel(), 2, device=cov.device),
            )
        else:
            split_noise = torch.empty(
                (2, 0, 2), dtype=model.mu.dtype, device=model.mu.device
            )
        candidate = pre_growth if prune_only_fast else self._candidate_after_growth(model, duplicate, split)
        # Condemned splats are excluded from growth, so mapping their verdicts
        # onto the candidate ordering marks exactly the original rows: every
        # newborn inherits False and is first judged at the next event.
        condemned_after_growth = condemned if prune_only_fast else self._candidate_passthrough(
            condemned, duplicate, split
        )
        scale_prune = self._scale_prune_mask(candidate["log_scales"])
        prune_after_growth = condemned_after_growth | scale_prune
        plan = SplatRefinePlan(
            step=step,
            duplicate=duplicate,
            split=split,
            prune_after_growth=prune_after_growth,
            split_noise=split_noise,
            threshold=threshold,
            birth=birth,
            prune_reasons={
                "flux": int(flux_condemned.sum()),
                "axis_ratio": int((axis_condemned & ~flux_condemned).sum()),
                "scale": int((scale_prune & ~condemned_after_growth).sum()),
            },
        )
        plan._validate_structure()
        return plan

    @torch.no_grad()
    def apply_plan(
        self,
        model,
        optimizers: list[torch.optim.Optimizer],
        plan: SplatRefinePlan,
    ) -> dict:
        if not isinstance(plan, SplatRefinePlan):
            raise TypeError("plan must be a SplatRefinePlan.")
        plan._validate_structure()
        n_splats = int(model.mu.shape[0])
        if plan.duplicate.numel() != n_splats:
            raise ValueError("Splat refinement masks do not match the model size.")
        devices = {
            plan.duplicate.device,
            plan.split.device,
            plan.prune_after_growth.device,
            plan.split_noise.device,
        }
        if devices != {model.mu.device}:
            raise ValueError(
                "Splat refinement plan tensors must be on the model device."
            )
        if plan.split_noise.dtype != model.mu.dtype:
            raise ValueError(
                "Splat refinement split noise dtype must match model positions."
            )
        n_dupli = int(plan.duplicate.sum())
        n_split = int(plan.split.sum())
        n_prune = int(plan.prune_after_growth.sum())
        planned_birth = 0 if plan.birth is None else int(plan.birth["mu"].shape[0])
        # Atomic preflight: every census-limit and shape violation must fail
        # BEFORE any surgery mutates the model. make_plan produces trusted
        # plans; replayed or hand-built payloads must not bypass the limits.
        if planned_birth:
            if plan.birth["sed_coeffs"].shape[1] != model.sed_coeffs.shape[1]:
                raise ValueError(
                    "Splat birth sed_coeffs width does not match the model "
                    f"({plan.birth['sed_coeffs'].shape[1]} vs "
                    f"{model.sed_coeffs.shape[1]})."
                )
            if planned_birth > max(0, int(self.config.birth_cap_per_event)):
                raise ValueError(
                    f"Plan carries {planned_birth} births but "
                    f"birth_cap_per_event={self.config.birth_cap_per_event}."
                )
        if self.config.max_splats:
            n_after = n_splats + n_dupli + n_split - n_prune + planned_birth
            if n_after > int(self.config.max_splats):
                raise ValueError(
                    f"Plan would leave {n_after} splats, above "
                    f"max_splats={self.config.max_splats}."
                )
        if n_dupli:
            self._duplicate(model, optimizers, plan.duplicate, step=plan.step)
        if n_split:
            split_after_duplicate = torch.cat(
                [
                    plan.split,
                    torch.zeros(
                        n_dupli,
                        dtype=torch.bool,
                        device=plan.split.device,
                    ),
                ]
            )
            self._split(
                model,
                optimizers,
                split_after_duplicate,
                split_noise=plan.split_noise,
                step=plan.step,
            )
        if n_prune:
            self._remove(model, optimizers, plan.prune_after_growth)
        n_birth = 0 if plan.birth is None else int(plan.birth["mu"].shape[0])
        if n_birth:
            self._birth(model, optimizers, plan.birth, step=plan.step)
        n = int(model.mu.shape[0])
        model.n_splats = n
        if plan.threshold is not None:
            self.threshold = float(plan.threshold)
        self.grad_accum = None
        self.count = None
        if not (n_dupli or n_split or n_prune or n_birth):
            return {}
        report = {
            "dupli": n_dupli,
            "split": n_split,
            "prune": n_prune,
            "n_splats": n,
        }
        if n_birth:
            report["birth"] = n_birth
        if plan.prune_reasons is not None:
            report["prune_reasons"] = dict(plan.prune_reasons)
        return report

    @torch.no_grad()
    def refine(self, model, optimizers: list[torch.optim.Optimizer], step: int) -> dict:
        """Duplicate, split, and prune on the gsplat cadence. Returns counts."""
        plan = self.make_plan(model, step)
        if plan is None:
            return {}
        return self.apply_plan(model, optimizers, plan)

    @torch.no_grad()
    def project_gauge(
        self,
        model,
        optimizers: list[torch.optim.Optimizer],
        gauge_factors: torch.Tensor,
        *,
        drift_log: float = math.log(2.0),
    ) -> int:
        """Canonicalize drifted splats so band-max unit amplitude is 1.

        gauge_factors: per-splat band-max band-average SED amplitude per unit
        flux (a). The raw_flux/SED split is a gauge; this projection folds a
        into raw_flux so raw_flux is physical rendered flux for every
        consumer. Exactly prediction-preserving in the model's softplus
        coordinates:

            sed_coeffs' = inv_softplus(softplus(sed_coeffs) / a)
            raw_flux'   = inv_softplus(a * softplus(raw_flux))

        Only splats with |log a| > drift_log are touched, bounding how many
        rows lose optimizer state per event: no exact Adam-moment transport
        exists through softplus (the Jacobian is coefficient-dependent), so
        projected rows' moments are RESET, the same idiom surgery applies to
        new rows. Splats whose SED renders to zero in every band (a == 0)
        cannot be normalized and are left for the significance prune."""
        n_splats = int(model.mu.shape[0])
        if not torch.is_tensor(gauge_factors) or gauge_factors.shape != (n_splats,):
            raise ValueError(
                "gauge_factors must be a per-splat vector "
                f"(got shape {getattr(gauge_factors, 'shape', None)} for {n_splats} splats)."
            )
        if not bool(torch.isfinite(gauge_factors).all()) or bool(
            (gauge_factors < 0).any()
        ):
            raise ValueError("gauge_factors must be finite and non-negative.")
        rows, new_coeffs, new_flux = canonicalize_gauge_rows(
            model.sed_coeffs.data,
            model.raw_flux.data,
            gauge_factors,
            drift_log=drift_log,
        )
        if rows.numel() == 0:
            return 0
        model.sed_coeffs.data[rows] = new_coeffs
        model.raw_flux.data[rows] = new_flux
        for optimizer in optimizers:
            for param in (model.sed_coeffs, model.raw_flux):
                state = optimizer.state.get(param)
                if not state:
                    continue
                for value in state.values():
                    if torch.is_tensor(value) and value.shape[:1] == param.shape[:1]:
                        value[rows] = 0
        return int(rows.numel())

    # --- surgery ---------------------------------------------------------
    @staticmethod
    def _update_params(
        model, optimizers, param_fn, optimizer_fn, buffer_fn=None
    ) -> None:
        """gsplat's _update_param_with_optimizer over the per-splat attrs.

        Handles param groups that mix per-splat and shared tensors: only the
        per-splat entry is replaced, by identity. buffer_fn applies the same
        row surgery to the lineage buffers; models without them (test stubs)
        are skipped."""
        for name in PER_SPLAT_PARAMS:
            param = getattr(model, name)
            new_param = torch.nn.Parameter(
                param_fn(name, param.data), requires_grad=param.requires_grad
            )
            for optimizer in optimizers:
                if param in optimizer.state:
                    state = optimizer.state.pop(param)
                    for key, value in state.items():
                        if torch.is_tensor(value) and value.shape[:1] == param.shape[:1]:
                            state[key] = optimizer_fn(key, value)
                    optimizer.state[new_param] = state
                for group in optimizer.param_groups:
                    group["params"] = [
                        new_param if p is param else p for p in group["params"]
                    ]
            setattr(model, name, new_param)
        if buffer_fn is None:
            return
        for name in PER_SPLAT_LINEAGE_BUFFERS:
            buffer = getattr(model, name, None)
            if buffer is None:
                continue
            setattr(model, name, buffer_fn(name, buffer))

    def _duplicate(
        self, model, optimizers, mask: torch.Tensor, *, step: int
    ) -> None:
        sel = torch.where(mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            extra = p[sel]
            if name == "raw_flux":
                # additive model: conserve total flux across the pair
                half = _inv_softplus((0.5 * F.softplus(p[sel])).clamp_min(1e-30))
                p = p.clone()
                p[sel] = half
                extra = half
            return torch.cat([p, extra])

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [v, torch.zeros(sel.numel(), *v.shape[1:], device=v.device, dtype=v.dtype)]
            )

        def buffer_fn(name: str, b: torch.Tensor) -> torch.Tensor:
            # parent and child both carry the halving; birth lineage is shared
            if name == "halving_depth":
                b = b.clone()
                b[sel] = b[sel].clamp_max(254) + 1
            elif name == "last_halving_event":
                b = b.clone()
                b[sel] = int(step)
            return torch.cat([b, b[sel]])

        self._update_params(model, optimizers, param_fn, optimizer_fn, buffer_fn)

    def _split(
        self,
        model,
        optimizers,
        mask: torch.Tensor,
        *,
        split_noise: torch.Tensor | None = None,
        step: int,
    ) -> None:
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        if split_noise is None:
            cov = model.covariances(sel)
            # closed-form 2x2 Cholesky with a variance floor: LAPACK rejects the
            # near-degenerate covariances that extreme axis ratios produce in
            # float32, and one such splat must not kill a week-long run
            a = cov[:, 0, 0].clamp_min(1e-12)
            b = cov[:, 1, 0]
            l11 = torch.sqrt(a)
            l21 = b / l11
            l22 = torch.sqrt((cov[:, 1, 1] - l21.square()).clamp_min(1e-12))
            zeros = torch.zeros_like(l11)
            chol = torch.stack(
                [torch.stack([l11, zeros], -1), torch.stack([l21, l22], -1)], -2
            )
            samples = torch.einsum(
                "nij,bnj->bni",
                chol,
                torch.randn(2, sel.numel(), 2, device=cov.device),
            )
        else:
            samples = split_noise

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "mu":
                p_split = (p[sel][None, :, :] + samples).reshape(-1, 2)
            elif name == "log_scales":
                p_split = (p[sel] - math.log(1.6)).repeat(repeats)
            elif name == "raw_flux":
                # additive model: each child carries half the parent's flux
                p_split = _inv_softplus(
                    (0.5 * F.softplus(p[sel])).clamp_min(1e-30)
                ).repeat(repeats)
            else:
                p_split = p[sel].repeat(repeats)
            return torch.cat([p[rest], p_split])

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            zeros = torch.zeros(
                2 * sel.numel(), *v.shape[1:], device=v.device, dtype=v.dtype
            )
            return torch.cat([v[rest], zeros])

        def buffer_fn(name: str, b: torch.Tensor) -> torch.Tensor:
            # children replace the parent, one halving deeper; row order must
            # mirror param_fn exactly: survivors first, then both child sets
            if name == "halving_depth":
                children = (b[sel].clamp_max(254) + 1).repeat(2)
            elif name == "last_halving_event":
                children = torch.full(
                    (2 * sel.numel(),), int(step), dtype=b.dtype, device=b.device
                )
            else:
                children = b[sel].repeat(2)
            return torch.cat([b[rest], children])

        self._update_params(model, optimizers, param_fn, optimizer_fn, buffer_fn)

    def _remove(self, model, optimizers, mask: torch.Tensor) -> None:
        keep = torch.where(~mask)[0]

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return p[keep]

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[keep]

        def buffer_fn(name: str, b: torch.Tensor) -> torch.Tensor:
            return b[keep]

        self._update_params(model, optimizers, param_fn, optimizer_fn, buffer_fn)

    def _birth(
        self, model, optimizers, rows: Mapping[str, torch.Tensor], *, step: int
    ) -> None:
        """Append fully-initialized rows with fresh optimizer state."""
        k = int(rows["mu"].shape[0])
        if rows["sed_coeffs"].shape[1] != model.sed_coeffs.shape[1]:
            raise ValueError(
                "Splat birth sed_coeffs width does not match the model "
                f"({rows['sed_coeffs'].shape[1]} vs {model.sed_coeffs.shape[1]})."
            )

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [p, rows[name].to(device=p.device, dtype=p.dtype)]
            )

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [v, torch.zeros(k, *v.shape[1:], device=v.device, dtype=v.dtype)]
            )

        def buffer_fn(name: str, b: torch.Tensor) -> torch.Tensor:
            if name == "birth_event":
                extra = torch.full((k,), int(step), dtype=b.dtype, device=b.device)
            elif name == "last_halving_event":
                extra = torch.full((k,), -1, dtype=b.dtype, device=b.device)
            else:
                extra = torch.zeros(k, dtype=b.dtype, device=b.device)
            return torch.cat([b, extra])

        self._update_params(model, optimizers, param_fn, optimizer_fn, buffer_fn)
