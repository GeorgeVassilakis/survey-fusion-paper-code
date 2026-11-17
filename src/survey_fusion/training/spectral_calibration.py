from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

from ..data.bandpasses import Bandpass, build_throughput_mix_weights
from ..utils.perf_env import calibration_singleton_mode, packed_calibration_enabled
from ..utils.torch_utils import as_tensor_on_device


class _BatchedSpectralScales(torch.autograd.Function):
    """Share launch work while retaining each exposure's gradient contribution.

    Repeating the same log-scale input creates one autograd edge per exposure.
    Returning their gradients in reverse exposure order reproduces the eager
    engine's accumulation order, including a separately evaluated prior.
    Summing them here and returning one gradient would reassociate that sum.
    """

    @staticmethod
    def forward(ctx, mix, covered, offsets, *log_scales):
        scales = torch.exp(log_scales[0])
        ctx.singleton_mode = calibration_singleton_mode()
        ctx.packed = packed_calibration_enabled()
        singles=[a for a,b in zip(offsets[:-1],offsets[1:]) if b-a==1]
        supported=False
        if ctx.singleton_mode=='both' and singles:
            from ..pgm.calibration_singleton_kernels import applies,singleton_scales
            supported=applies(mix,scales)
        if supported and ctx.packed:
            # Compute singleton rows together, then replace the few multi-bin
            # blocks with their original GEMVs. Avoid one view per exposure.
            index = torch.arange(mix.shape[0], device=mix.device)
            mixed = singleton_scales(mix, scales, index)
            for a, b in zip(offsets[:-1], offsets[1:]):
                if b - a != 1:
                    mixed[a:b] = torch.mv(mix[a:b], scales)
        elif supported:
            index=torch.tensor(singles,device=mix.device,dtype=torch.long)
            single_values=singleton_scales(mix,scales,index)
            parts=[]
            cursor=0
            for a,b in zip(offsets[:-1],offsets[1:]):
                if b-a==1:
                    parts.append(single_values[cursor:cursor+1])
                    cursor+=1
                else:
                    parts.append(torch.mv(mix[a:b],scales))
            mixed=torch.cat(parts)
        else:
            mixed = torch.cat([
                torch.mv(mix[a:b], scales) for a, b in zip(offsets[:-1], offsets[1:])
            ])
        ctx.save_for_backward(mix, covered, scales)
        ctx.offsets = offsets
        return covered * mixed + (1.0 - covered)

    @staticmethod
    def backward(ctx, grad_output):
        mix, covered, scales = ctx.saved_tensors
        grad_mixed = grad_output * covered
        if ctx.singleton_mode!='0':
            # The one-bin GEMV is a product, including its signed zero.
            single_partials=mix*grad_mixed[:,None]
            if ctx.packed:
                starts = torch.tensor(ctx.offsets[:-1], device=mix.device)
                partials = single_partials.index_select(0, starts)
                for i, (a, b) in enumerate(zip(ctx.offsets[:-1], ctx.offsets[1:])):
                    if b - a != 1:
                        partials[i] = torch.mv(mix[a:b].T, grad_mixed[a:b])
                partials = partials * scales[None, :]
            else:
                partials=torch.stack([single_partials[a] if b-a==1 else torch.mv(mix[a:b].T,grad_mixed[a:b])
                                      for a,b in zip(ctx.offsets[:-1],ctx.offsets[1:])])*scales[None,:]
        else:
            partials = torch.stack([
                torch.mv(mix[a:b].T, grad_mixed[a:b])
                for a, b in zip(ctx.offsets[:-1], ctx.offsets[1:])
            ]) * scales[None, :]
        return None, None, None, *reversed(partials.unbind(0))


def _as_sorted_unique_bands(bands: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for band in bands:
        band_name = str(band)
        if band_name in seen:
            continue
        seen.add(band_name)
        ordered.append(band_name)
    if not ordered:
        raise ValueError("At least one band is required for spectral calibration.")
    return ordered


PRIOR_NORMALIZATIONS = ("legacy", "objective")


@dataclass(slots=True)
class SpectralCalibrationConfig:
    enabled: bool = False
    anchor: str = "image"
    mapping: str = "throughput_mix"
    prior_sigma_log: float = 0.15
    prior_weight: float = 1.0
    init_scale: float = 1.0
    lr_multiplier: float = 1.0
    apply_to_predictions: bool = True
    prior_normalization: str = "legacy"
    freeze_until_step: int = 0

    @classmethod
    def from_dict(cls, cfg: Dict[str, object] | None) -> "SpectralCalibrationConfig":
        cfg_map: Dict[str, object] = dict(cfg or {})
        return cls(
            enabled=bool(cfg_map.get("enabled", False)), anchor=str(cfg_map.get("anchor", "image")),
            mapping=str(cfg_map.get("mapping", "throughput_mix")),
            prior_sigma_log=float(cfg_map.get("prior_sigma_log", 0.15)),
            prior_weight=float(cfg_map.get("prior_weight", 1.0)),
            init_scale=float(cfg_map.get("init_scale", 1.0)),
            lr_multiplier=float(cfg_map.get("lr_multiplier", 1.0)),
            apply_to_predictions=bool(cfg_map.get("apply_to_predictions", True)),
            prior_normalization=str(cfg_map.get("prior_normalization", "legacy")),
            freeze_until_step=int(cfg_map.get("freeze_until_step", 0)),
        )


class SpectralNuisanceCalibrator(nn.Module):
    def __init__(
        self, *, spectral_lam_by_exposure: Dict[int, np.ndarray], bandpasses: Dict[str, Bandpass],
        cfg: SpectralCalibrationConfig, device: torch.device,
    ):
        super().__init__()
        if cfg.anchor.lower() != "image":
            raise ValueError(f"spectral nuisance anchor must be 'image', got {cfg.anchor!r}.")
        if cfg.mapping.lower() != "throughput_mix":
            raise ValueError(f"spectral nuisance mapping must be 'throughput_mix', got {cfg.mapping!r}.")
        if cfg.prior_sigma_log <= 0:
            raise ValueError("spectral nuisance prior_sigma_log must be positive.")
        if cfg.init_scale <= 0:
            raise ValueError("spectral nuisance init_scale must be positive.")
        if cfg.freeze_until_step < 0:
            raise ValueError("spectral nuisance freeze_until_step must be non-negative.")
        if cfg.prior_normalization not in PRIOR_NORMALIZATIONS:
            raise ValueError(
                f"spectral nuisance prior_normalization must be one of {PRIOR_NORMALIZATIONS}, "
                f"got {cfg.prior_normalization!r}."
            )
        self.cfg = cfg
        self.band_names = _as_sorted_unique_bands(bandpasses.keys())
        init_log_scale = float(np.log(cfg.init_scale))
        self.log_scales = nn.Parameter(
            torch.full((len(self.band_names), ), init_log_scale, dtype=torch.float32, device=device)
        )
        self._mix_weights: Dict[int, torch.Tensor] = {}
        self._covered: Dict[int, torch.Tensor] = {}
        self._batch_mix_table = None
        self._batch_covered_table = None
        self._batch_exposure_bases = None
        for exp_id, lam_spec in sorted(spectral_lam_by_exposure.items()):
            mix_np, covered_np = build_throughput_mix_weights(
                np.asarray(lam_spec, dtype=np.float64), bandpasses, self.band_names,
            )
            self._mix_weights[int(exp_id)] = torch.tensor(mix_np, dtype=torch.float32, device=device)
            self._covered[int(exp_id)] = torch.tensor(covered_np, dtype=torch.float32, device=device)

    def scales_by_band(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def scale_for_indices(self, exposure_id: int, local_spec_indices: torch.Tensor) -> torch.Tensor:
        idx = local_spec_indices.to(dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError("local_spec_indices must be 1D.")
        if idx.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.log_scales.device)
        exp_key = int(exposure_id)
        if exp_key not in self._mix_weights:
            raise KeyError(f"No spectral calibration weights for exposure {exposure_id}.")
        mix = self._mix_weights[exp_key]
        covered = self._covered[exp_key]
        mix_sel = torch.index_select(mix, 0, idx)
        covered_sel = torch.index_select(covered, 0, idx)
        mixed_scale = torch.matmul(mix_sel, self.scales_by_band())
        return covered_sel * mixed_scale + (1.0 - covered_sel)

    def scale_for_full_exposure(self, exposure_id: int) -> torch.Tensor:
        exp_key = int(exposure_id)
        if exp_key not in self._mix_weights:
            raise KeyError(f"No spectral calibration weights for exposure {exposure_id}.")
        mix = self._mix_weights[exp_key]
        covered = self._covered[exp_key]
        mixed_scale = torch.matmul(mix, self.scales_by_band())
        return covered * mixed_scale + (1.0 - covered)

    def scale_for_batch(self, exposure_ids, offsets, local_spec_indices):
        """Batch scales after the trainer's CPU check of exposure-local indices."""
        if not exposure_ids:
            return self.log_scales.new_empty((0,))
        if (
            not self.log_scales.is_cuda
            or torch.__version__.split("+")[0] != "2.11.0"
            or any(b <= a for a, b in zip(offsets[:-1], offsets[1:]))
        ):
            return torch.cat([
                self.scale_for_indices(int(exp_id), local_spec_indices[offsets[i]:offsets[i + 1]])
                for i, exp_id in enumerate(exposure_ids)
            ])
        if self._batch_mix_table is None:
            bases = {}
            base = 0
            for exp_id, table in sorted(self._mix_weights.items()):
                bases[exp_id] = base
                base += table.shape[0]
            self._batch_exposure_bases = bases
            self._batch_mix_table = torch.cat([self._mix_weights[e] for e in sorted(bases)])
            self._batch_covered_table = torch.cat([self._covered[e] for e in sorted(bases)])
        counts = np.diff(offsets)
        base_rows = np.repeat([self._batch_exposure_bases[int(e)] for e in exposure_ids], counts)
        bases_device = as_tensor_on_device(base_rows, dtype=torch.long, device=self.log_scales.device)
        indices = local_spec_indices.to(dtype=torch.long) + bases_device
        mix = self._batch_mix_table.index_select(0, indices)
        covered = self._batch_covered_table.index_select(0, indices)
        return _BatchedSpectralScales.apply(
            mix, covered, tuple(offsets), *([self.log_scales] * len(exposure_ids))
        )

    def prior_penalty(self) -> torch.Tensor:
        if self.cfg.prior_weight <= 0:
            return torch.zeros((), dtype=torch.float32, device=self.log_scales.device)
        sigma = float(self.cfg.prior_sigma_log)
        if self.cfg.prior_normalization == "objective":
            # Negative log prior summed over bands; the trainer divides it by
            # the objective's datum count, as it does the likelihood.
            penalty = 0.5 * torch.square(self.log_scales / sigma).sum()
        else:
            penalty = 0.5 * torch.square(self.log_scales / sigma).mean()
        return penalty * float(self.cfg.prior_weight)

    def get_state(self) -> Dict[str, object]:
        return {
            "band_names": list(self.band_names),
            "log_scales": [float(x) for x in self.log_scales.detach().cpu().tolist()],
            "config": {
                "anchor": self.cfg.anchor, "mapping": self.cfg.mapping,
                "prior_sigma_log": float(self.cfg.prior_sigma_log),
                "prior_weight": float(self.cfg.prior_weight), "init_scale": float(self.cfg.init_scale),
                "lr_multiplier": float(self.cfg.lr_multiplier),
                "apply_to_predictions": bool(self.cfg.apply_to_predictions),
                "prior_normalization": self.cfg.prior_normalization,
                "freeze_until_step": int(self.cfg.freeze_until_step),
            },
        }

    def load_state(self, state: Dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Spectral calibration state must be a mapping.")
        bands = state.get("band_names")
        values = state.get("log_scales")
        if not isinstance(bands, list) or not isinstance(values, list):
            raise ValueError("Spectral calibration state must contain band_names and log_scales.")
        if [str(b) for b in bands] != list(self.band_names):
            raise ValueError(
                f"Spectral calibration bands {bands!r} do not match trainer bands {self.band_names!r}."
            )
        if len(values) != self.log_scales.numel():
            raise ValueError(f"Expected {self.log_scales.numel()} log scales, got {len(values)}.")
        value_t = torch.tensor(
            [float(v) for v in values], dtype=self.log_scales.dtype, device=self.log_scales.device,
        )
        with torch.no_grad():
            self.log_scales.copy_(value_t)

    def report(self) -> Dict[str, object]:
        scales = self.scales_by_band().detach().cpu().tolist()
        logs = self.log_scales.detach().cpu().tolist()
        return {
            "enabled": True, "anchor": self.cfg.anchor, "mapping": self.cfg.mapping,
            "prior_sigma_log": float(self.cfg.prior_sigma_log), "prior_weight": float(self.cfg.prior_weight),
            "apply_to_predictions": bool(self.cfg.apply_to_predictions),
            "prior_normalization": self.cfg.prior_normalization,
            "band_scales": {band: float(scale) for band, scale in zip(self.band_names, scales)},
            "band_log_scales": {band: float(logv) for band, logv in zip(self.band_names, logs)},
        }
