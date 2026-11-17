from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from support.kernel_factory import PlanPairKernelFactoryMixin
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import DatumBatch, DatumKind, ExposureState, GroupedKernelSamples, KernelSamples
from survey_fusion.training.datum_trainer import DatumTrainer


class DummyWCS:
    def __init__(self, value: float = 0.0, *, use_omega_x: bool = False) -> None:
        self.value = float(value)
        self.use_omega_x = bool(use_omega_x)

    def pixel_to_omega_xyz(self, x, y):
        shape = np.broadcast(x, y).shape
        omega = np.zeros(shape + (3,), dtype=np.float64)
        omega[..., 0] = 1.0
        return omega

    def omega_xyz_to_norm_torch(self, omega_xyz: torch.Tensor):
        if self.use_omega_x:
            value = omega_xyz[..., 0].to(dtype=omega_xyz.dtype)
        else:
            value = torch.full(
                omega_xyz.shape[: -1], self.value, dtype=omega_xyz.dtype, device=omega_xyz.device,
            )
        return value, torch.zeros_like(value)


class NoNormWCS:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def pixel_to_omega_xyz(self, x, y):
        shape = np.broadcast(x, y).shape
        omega = np.zeros(shape + (3,), dtype=np.float64)
        omega[..., 0] = 1.0
        return omega


class ExposureValueModel(torch.nn.Module):
    def __init__(self, *, trainable_bias: bool = True, zero_bias_grad: bool = False) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(())) if trainable_bias else None
        self.zero_bias_grad = bool(zero_bias_grad)

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        out = x_norm.to(dtype=torch.float32)
        if self.bias is None:
            return out
        if self.zero_bias_grad:
            return out + 0.0 * self.bias
        return out + self.bias


class OnesModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        return torch.ones_like(x_norm, dtype=torch.float32, device=self.bias.device) + 0.0 * self.bias


class OffsetModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        return x_norm.to(dtype=torch.float32) + self.offset + 0.0 * self.bias


class ConstantKernelFactory(PlanPairKernelFactoryMixin):
    def __init__(self, batch: DatumBatch, *, record_calls: bool = False) -> None:
        self._batch = batch
        self.record_calls = bool(record_calls)
        self.calls = 0
        self.last_indices: list[int] = []
        self.call_history: list[list[int]] = []

    def build_kernel_samples_for_indices(
        self, global_indices: torch.Tensor, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples:
        if self.record_calls:
            self.calls += 1
            self.last_indices = global_indices.detach().cpu().tolist()
            self.call_history.append(list(self.last_indices))
        n = int(global_indices.numel())
        omega = torch.zeros((n, k, 3), dtype=torch.float32, device=device)
        omega[..., 0] = 1.0
        loglam = torch.full((n, k), float(np.log(5500.0)), dtype=torch.float32, device=device)
        w = torch.full((n, k), 1.0 / float(k), dtype=torch.float32, device=device)
        z = torch.ones((n,), dtype=torch.float32, device=device)
        idx = global_indices.to(dtype=torch.long, device=self._batch.exposure_id.device)
        return GroupedKernelSamples(
            samples=KernelSamples(omega_xyz=omega, loglam=loglam, w=w, Z=z),
            row_original_index=torch.arange(n, dtype=torch.long, device=device),
            row_exposure_id=torch.index_select(self._batch.exposure_id, 0, idx).to(
                dtype=torch.long, device=device
            ),
            row_datum_index=torch.index_select(self._batch.datum_index, 0, idx).to(
                dtype=torch.long, device=device
            ),
        )

    def build_kernel_sample_plan_for_indices(self, global_indices: torch.Tensor) -> torch.Tensor:
        return global_indices.clone()

    def build_kernel_samples_from_plan(
        self, plan: torch.Tensor, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples:
        return self.build_kernel_samples_for_indices(plan, k=k, device=device)


@dataclass(slots=True)
class SyntheticStore:
    exposures: dict[int, ExposureState]
    batch: DatumBatch
    image_band_by_exposure: dict[int, str]
    kernel_factory: object | None
    n_image: int
    n_spec: int
    image_excluded_local_indices_by_exposure: dict[int, Any] = field(default_factory=dict)
    image_random_holdout_indices_by_exposure: dict[int, Any] = field(default_factory=dict)
    region_masks_by_exposure: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    holdout_region_summary: dict[str, object] = field(default_factory=dict)
    partial_holdout_enabled: bool = False
    image_additive_by_exposure: dict[int, np.ndarray] = field(default_factory=dict)
    spec_source_by_exposure: dict[int, np.ndarray] = field(default_factory=dict)
    spec_additive_by_exposure: dict[int, np.ndarray] = field(default_factory=dict)
    spec_label_by_exposure: dict[int, str] = field(default_factory=dict)


def bandpass(band: str = "r") -> Bandpass:
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return Bandpass(band=band, lam=lam, throughput=throughput, canonical_id=f"test_{band}")


def image_exposure(
    value: float = 0.0, *, band: str = "r", height: int = 2, width: int = 2, wcs: object | None = None,
    obs_wcs: object | None = None, psf_fwhm_arcsec: float | None = None,
) -> ExposureState:
    active_wcs = DummyWCS(value) if wcs is None else wcs
    return ExposureState(
        wcs=active_wcs, obs_wcs=active_wcs if obs_wcs is None else obs_wcs, psf_fwhm_arcsec=psf_fwhm_arcsec,
        lsf_fwhm_angstrom=None, bandpasses={band: bandpass(band)},
        lam_spec=np.array([5000.0, 5100.0], dtype=np.float64),
        fiber_mask=np.ones((height, width), dtype=np.float32), pixel_scale_arcsec=1.0, image_height=height,
        image_width=width,
    )


def spec_exposure(
    value: float = 0.0, *, wcs: object | None = None, obs_wcs: object | None = None,
    lsf_fwhm_angstrom: float | None = 2.0, psf_fwhm_arcsec: float | None = None,
    lam_spec: np.ndarray | None = None,
) -> ExposureState:
    active_wcs = DummyWCS(value) if wcs is None else wcs
    return ExposureState(
        wcs=active_wcs, obs_wcs=obs_wcs, psf_fwhm_arcsec=psf_fwhm_arcsec, lsf_fwhm_angstrom=lsf_fwhm_angstrom,
        bandpasses={}, lam_spec=np.array([5000.0], dtype=np.float64) if lam_spec is None else lam_spec,
        fiber_mask=np.ones((2, 2), dtype=np.float32), pixel_scale_arcsec=1.0,
    )


def make_batch(
    *, y: torch.Tensor, sigma: torch.Tensor | None = None, exposure_id: torch.Tensor,
    datum_kind: torch.Tensor, datum_index: torch.Tensor, additive: torch.Tensor | None = None,
    source_gate: torch.Tensor | None = None,
) -> DatumBatch:
    device = y.device
    return DatumBatch(
        y=y.to(dtype=torch.float32, device=device),
        sigma=(
            torch.ones_like(y, dtype=torch.float32, device=device)
            if sigma is None
            else sigma.to(dtype=torch.float32, device=device)
        ),
        exposure_id=exposure_id.to(dtype=torch.long, device=device),
        datum_kind=datum_kind.to(dtype=torch.long, device=device),
        datum_index=datum_index.to(dtype=torch.long, device=device),
        kernel_samples=None,
        additive=None if additive is None else additive.to(dtype=torch.float32, device=device),
        source_gate=None if source_gate is None else source_gate.to(dtype=torch.float32, device=device),
    )


def make_image_spec_trainer(
    *, image_value: float = 3.0, spec_value: float = 11.0, y: torch.Tensor | None = None,
    additive: torch.Tensor | None = None, source_gate: torch.Tensor | None = None,
    model: torch.nn.Module | None = None, trainer_cfg: dict[str, Any] | None = None,
    calibration_init_scale: float | None = None, calibration_enabled: bool | None = None,
) -> DatumTrainer:
    device = torch.device("cpu")
    batch = make_batch(
        y=(
            y.to(dtype=torch.float32, device=device)
            if y is not None
            else torch.zeros(2, dtype=torch.float32, device=device)
        ),
        exposure_id=torch.tensor([0, 1], dtype=torch.long, device=device),
        datum_kind=torch.tensor(
            [int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)], dtype=torch.long, device=device,
        ),
        datum_index=torch.tensor([0, 0], dtype=torch.long, device=device),
        additive=additive,
        source_gate=source_gate,
    )
    store = SyntheticStore(
        exposures={0: image_exposure(image_value), 1: spec_exposure(spec_value)}, batch=batch,
        image_band_by_exposure={0: "r"}, kernel_factory=ConstantKernelFactory(batch), n_image=1, n_spec=1,
    )
    cfg: dict[str, Any] = {
        "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": 0, "learning_rate": 1e-3,
        "apply_spatial_psf": False,
    }
    if calibration_init_scale is not None:
        enabled = calibration_init_scale != 1.0 if calibration_enabled is None else calibration_enabled
        cfg["calibration"] = {
            "spectral_nuisance": {
                "enabled": bool(enabled), "anchor": "image", "mapping": "throughput_mix",
                "prior_sigma_log": 0.15, "prior_weight": 0.0, "init_scale": float(calibration_init_scale),
                "lr_multiplier": 1.0, "apply_to_predictions": True,
            }
        }
    if trainer_cfg:
        cfg.update(trainer_cfg)
    return DatumTrainer(
        model=model or ExposureValueModel(), grid=build_spatial_grid(2, 2, device), data=store,
        training_bands=["r"], held_out_band="i", lambda_step=20.0, trainer_cfg=cfg, device=device,
    )


def make_image_only_trainer(
    *, y_values: list[float] | None = None, sigma_values: list[float] | None = None,
    additive_values: list[float] | None = None, exposure_ids: list[int] | None = None,
    datum_indices: list[int] | None = None, exposure_values: dict[int, float] | None = None,
    model: torch.nn.Module | None = None, include_kernel_factory: bool = True,
    record_kernel_calls: bool = False, trainer_cfg: dict[str, Any] | None = None,
    distributed_context=None,
    exposure_overrides: dict[int, ExposureState] | None = None,
) -> tuple[DatumTrainer, ConstantKernelFactory | None]:
    device = torch.device("cpu")
    y_array = np.zeros(4, dtype=np.float32) if y_values is None else np.asarray(y_values, dtype=np.float32)
    sigma_array = np.ones_like(y_array) if sigma_values is None else np.asarray(
        sigma_values, dtype=np.float32
    )
    ids = exposure_ids if exposure_ids is not None else [0, 1, 0, 1]
    local = datum_indices if datum_indices is not None else [0, 0, 1, 1]
    batch = make_batch(
        y=torch.tensor(y_array, dtype=torch.float32, device=device),
        sigma=torch.tensor(sigma_array, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(ids, dtype=torch.long, device=device),
        datum_kind=torch.zeros(len(y_array), dtype=torch.long, device=device),
        datum_index=torch.tensor(local, dtype=torch.long, device=device),
        additive=(
            None
            if additive_values is None
            else torch.tensor(additive_values, dtype=torch.float32, device=device)
        ),
    )
    values = exposure_values or {int(exp_id): float(exp_id) for exp_id in sorted(set(ids))}
    exposures = {
        exp_id: image_exposure(value, wcs=DummyWCS(value, use_omega_x=True))
        for exp_id, value in values.items()
    }
    if exposure_overrides:
        exposures.update(exposure_overrides)
    factory = (
        ConstantKernelFactory(batch, record_calls=record_kernel_calls)
        if include_kernel_factory
        else None
    )
    store = SyntheticStore(
        exposures=exposures, batch=batch, image_band_by_exposure={exp_id: "r" for exp_id in exposures},
        kernel_factory=factory, n_image=len(y_array), n_spec=0,
    )
    cfg: dict[str, Any] = {
        "kernel_samples_k": 4, "replica_samples_k": 1, "epochs": 1, "learning_rate": 1e-3,
        "pixel_batch_size": 1, "spec_batch_size": 0, "apply_spatial_psf": False,
    }
    if trainer_cfg:
        cfg.update(trainer_cfg)
    trainer = DatumTrainer(
        model=model or ExposureValueModel(zero_bias_grad=True), grid=build_spatial_grid(2, 2, device),
        data=store, training_bands=["r"], held_out_band="i", lambda_step=20.0, trainer_cfg=cfg, device=device,
        distributed_context=distributed_context,
    )
    return trainer, factory
