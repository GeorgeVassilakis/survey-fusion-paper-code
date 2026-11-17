from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Dict, Protocol, Sequence, TypeAlias

import numpy as np
import torch

from ..data.bandpasses import Bandpass


@dataclass(slots=True)
class KernelSamples:
    omega_xyz: torch.Tensor # [M, K, 3]
    loglam: torch.Tensor # [M, K]
    w: torch.Tensor # [M, K]
    Z: torch.Tensor # [M]


@dataclass(slots=True)
class GroupedKernelSamples:
    samples: KernelSamples
    row_original_index: torch.Tensor # [M]
    row_exposure_id: torch.Tensor # [M]
    row_datum_index: torch.Tensor # [M]
    exposure_blocks: tuple[tuple[int, int, int], ...] = ()


@dataclass(slots=True)
class DatumBatch:
    y: torch.Tensor
    sigma: torch.Tensor
    exposure_id: torch.Tensor
    datum_kind: torch.Tensor
    datum_index: torch.Tensor
    kernel_samples: GroupedKernelSamples | None
    additive: torch.Tensor | None = None
    source_gate: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MaskApertureSpec:
    mask: np.ndarray # [H, W] non-negative weights


@dataclass(frozen=True, slots=True)
class SkyDiskApertureSpec:
    center_radec_deg: tuple[float, float] # (ra_deg, dec_deg)
    radius_arcsec: float


ApertureSpec: TypeAlias = MaskApertureSpec | SkyDiskApertureSpec


@dataclass(frozen=True, slots=True)
class PSFMixtureComponents:
    """Canonical co-centered Gaussian components in native detector pixels."""

    components: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("PSF mixture must contain at least one component.")
        weights = []
        previous_scale = -math.inf
        for index, component in enumerate(self.components):
            if len(component) != 4:
                raise ValueError(
                    f"PSF mixture component {index} must be [weight, vxx, vxy, vyy]."
                )
            weight, vxx, vxy, vyy = (float(value) for value in component)
            if not all(math.isfinite(value) for value in (weight, vxx, vxy, vyy)):
                raise ValueError(f"PSF mixture component {index} contains non-finite values.")
            determinant = vxx * vyy - vxy * vxy
            if weight <= 0.0:
                raise ValueError(f"PSF mixture component {index} weight must be positive.")
            if vxx <= 0.0 or vyy <= 0.0 or determinant <= 0.0:
                raise ValueError(f"PSF mixture component {index} covariance must be SPD.")
            scale = determinant ** 0.25
            if scale < previous_scale:
                raise ValueError("PSF mixture components must be ordered by geometric-mean sigma.")
            previous_scale = scale
            weights.append(weight)
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"PSF mixture weights must sum to one, got {sum(weights)!r}.")

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[float]]) -> "PSFMixtureComponents":
        return cls(tuple(tuple(float(value) for value in row) for row in rows))


def aperture_area_arcsec2(
    aperture: ApertureSpec, *, pixel_scale_arcsec: float
) -> float:
    pixel_scale = float(pixel_scale_arcsec)
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError("pixel_scale_arcsec must be positive and finite.")
    if isinstance(aperture, SkyDiskApertureSpec):
        radius = float(aperture.radius_arcsec)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("SkyDiskApertureSpec.radius_arcsec must be positive and finite.")
        return math.pi * radius**2
    mask = np.asarray(aperture.mask, dtype=np.float64)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("MaskApertureSpec.mask must be a non-empty 2D array.")
    if np.any(~np.isfinite(mask)) or np.any(mask < 0.0):
        raise ValueError("MaskApertureSpec.mask must contain finite non-negative weights.")
    area = float(mask.sum()) * pixel_scale**2
    if area <= 0.0:
        raise ValueError("MaskApertureSpec.mask must have positive total weight.")
    return area


@dataclass(slots=True)
class ExposureState:
    wcs: object
    psf_fwhm_arcsec: float | None
    lsf_fwhm_angstrom: float | np.ndarray | None
    bandpasses: Dict[str, Bandpass]
    lam_spec: np.ndarray
    fiber_mask: np.ndarray
    pixel_scale_arcsec: float
    obs_wcs: object | None = None
    image_height: int | None = None
    image_width: int | None = None
    # (n_bins, 2) per-bin [lower, upper] wavelength edges; bins bordering
    # masked pipeline pixels are narrower than the gap between surviving
    # centers, so edges cannot be reconstructed from lam_spec
    lam_edges: np.ndarray | None = None
    aperture: ApertureSpec | None = None
    psf_mixture_dgpsf: PSFMixtureComponents | None = None
    psf_mixture_kl: PSFMixtureComponents | None = None
    psf_effective_fwhm_arcsec: float | None = None
    psf_effective_fwhm_dg_arcsec: float | None = None
    psf_effective_fwhm_kl_arcsec: float | None = None
    label: str | None = None
    spectral_response_metadata: dict | None = None
    # Known observation response acting on source light, never additive sky.
    spectral_source_scale: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        validated_spectral_source_scale(self)



def spectral_source_geometry_key(exposure: ExposureState) -> tuple | None:
    """Validate a declared Gaussian calibration and fingerprint its geometry."""
    metadata = exposure.spectral_response_metadata or {}
    if metadata.get("calibration", {}).get("mode") != "gaussian_point_source":
        return None
    aperture = exposure.aperture
    if not isinstance(aperture, SkyDiskApertureSpec):
        raise ValueError("gaussian_point_source response requires a circular sky aperture.")
    radius = float(aperture.radius_arcsec)
    fwhm = float(exposure.psf_fwhm_arcsec or 0.)
    declared_fwhm = float(metadata.get("fwhm_arcsec", 0.))
    if not all(math.isfinite(x) and x > 0 for x in (radius, fwhm, declared_fwhm)):
        raise ValueError("gaussian_point_source response requires positive finite radius and seeing.")
    if fwhm != declared_fwhm:
        raise ValueError("gaussian_point_source response seeing disagrees with metadata.")
    expected = metadata.get("resolved_source_scale")
    divisor = -math.expm1(-0.5 * (radius / (fwhm / 2.355)) ** 2)
    if divisor <= 0 or expected is None or not math.isclose(float(expected), 1. / divisor, rel_tol=1e-12, abs_tol=0.):
        raise ValueError("resolved spectral_source_scale disagrees with calibration aperture/seeing.")
    return (aperture.center_radec_deg, radius, fwhm, repr(metadata))


def validate_spectral_source_psf(exposure: ExposureState, *, apply_spatial_psf: bool) -> None:
    if spectral_source_geometry_key(exposure) is not None and not apply_spatial_psf:
        raise ValueError("gaussian_point_source response requires apply_spatial_psf=True.")


def validated_spectral_source_scale(exposure: ExposureState) -> np.ndarray | None:
    """Return a validated immutable copy of the response in native-bin order."""
    value = exposure.spectral_source_scale
    metadata = exposure.spectral_response_metadata or {}
    mode = metadata.get("calibration", {}).get("mode")
    spectral_source_geometry_key(exposure)
    expected = metadata.get("resolved_source_scale")
    if mode == "gaussian_point_source" and (value is None or expected is None):
        raise ValueError("gaussian_point_source metadata requires resolved spectral_source_scale.")
    if value is None:
        return None
    n_bins = int(np.asarray(exposure.lam_spec).size)
    if n_bins == 0:
        raise ValueError("spectral_source_scale requires non-empty spectral bins.")
    scale = np.asarray(value, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.full(n_bins, float(scale), dtype=np.float64)
    elif scale.shape != (n_bins,):
        raise ValueError("spectral_source_scale must be scalar or one value per native spectral bin.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("spectral_source_scale must be positive and finite.")
    if mode == "gaussian_point_source":
        expected_array = np.asarray(expected, dtype=np.float64)
        if expected_array.ndim > 1 or expected_array.shape not in ((), scale.shape):
            raise ValueError("resolved spectral_source_scale metadata has invalid shape.")
        if not np.isfinite(expected_array).all() or np.any(expected_array <= 0):
            raise ValueError("resolved spectral_source_scale metadata must be positive and finite.")
        if not np.array_equal(scale, np.broadcast_to(expected_array, scale.shape)):
            raise ValueError("spectral_source_scale disagrees with resolved response metadata.")
    if mode == "finite_aperture" and np.any(scale != 1.):
        raise ValueError("finite_aperture metadata requires identity spectral_source_scale.")
    result = scale.copy()
    result.flags.writeable = False
    return result


ExposureTable: TypeAlias = dict[int, ExposureState]


class DatumKind(IntEnum):
    IMAGE_PIXEL = 0
    SPECTRAL_BIN = 1


class DatumKernelFactory(Protocol):
    def build_kernel_sample_plan_for_indices(self, global_indices: torch.Tensor) -> object: ...

    def build_kernel_samples_from_plan(
        self, plan: object, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples: ...

    def build_kernel_sample_pair_from_plan(
        self, plan: object, *, k: int, device: torch.device,
    ) -> tuple[GroupedKernelSamples, GroupedKernelSamples]: ...

    def build_kernel_samples_for_indices(
        self, global_indices: torch.Tensor, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples: ...


def validate_batch_exposure_links(batch: DatumBatch, exposures: ExposureTable) -> None:
    if not exposures:
        raise ValueError("Exposure table must be non-empty.")
    if batch.exposure_id.ndim != 1:
        raise ValueError("DatumBatch.exposure_id must be a 1D tensor.")
    unique_ids = torch.unique(batch.exposure_id).detach().cpu().tolist()
    missing = sorted(int(exp_id) for exp_id in unique_ids if int(exp_id) not in exposures)
    if missing:
        raise ValueError(f"DatumBatch references exposure_id values not present in exposure table: {missing}")


def validate_batch_datum_kinds(
    batch: DatumBatch, *, supported: tuple[DatumKind, ...]=(DatumKind.IMAGE_PIXEL, DatumKind.SPECTRAL_BIN),
) -> None:
    if batch.sigma.ndim != 1:
        raise ValueError("DatumBatch.sigma must be a 1D tensor.")
    if batch.datum_kind.ndim != 1:
        raise ValueError("DatumBatch.datum_kind must be a 1D tensor.")
    if batch.datum_index.ndim != 1:
        raise ValueError("DatumBatch.datum_index must be a 1D tensor.")
    n = int(batch.y.shape[0])
    if (
        int(batch.sigma.shape[0]) != n
        or int(batch.datum_kind.shape[0]) != n
        or int(batch.datum_index.shape[0]) != n
    ):
        raise ValueError("DatumBatch sigma/kind/index tensors must align with y length.")
    if batch.additive is not None:
        if batch.additive.ndim != 1:
            raise ValueError("DatumBatch.additive must be a 1D tensor when provided.")
        if int(batch.additive.shape[0]) != n:
            raise ValueError("DatumBatch.additive must align with y length.")
    if batch.source_gate is not None:
        if batch.source_gate.ndim != 1:
            raise ValueError("DatumBatch.source_gate must be a 1D tensor when provided.")
        if int(batch.source_gate.shape[0]) != n:
            raise ValueError("DatumBatch.source_gate must align with y length.")
    supported_values = {int(kind) for kind in supported}
    kind_values = torch.unique(batch.datum_kind).detach().cpu().tolist()
    unknown = sorted(int(value) for value in kind_values if int(value) not in supported_values)
    if unknown:
        raise ValueError(f"DatumBatch contains unsupported datum_kind values: {unknown}")
    if torch.any(batch.datum_index < 0):
        raise ValueError("DatumBatch.datum_index must be non-negative.")


def validate_kernel_samples(
    samples: KernelSamples, *, atol: float = 1e-5, omega_norm_atol: float = 1e-4,
    require_z_close_to_one: bool = False, z_atol: float = 1e-5,
) -> None:
    if samples.omega_xyz.ndim != 3 or samples.omega_xyz.shape[-1] != 3:
        raise ValueError(f"omega_xyz must be [M,K,3], got {tuple(samples.omega_xyz.shape)}.")
    if samples.loglam.ndim != 2:
        raise ValueError(f"loglam must be [M,K], got {tuple(samples.loglam.shape)}.")
    if samples.w.ndim != 2:
        raise ValueError(f"w must be [M,K], got {tuple(samples.w.shape)}.")
    if samples.Z.ndim != 1:
        raise ValueError(f"Z must be [M], got {tuple(samples.Z.shape)}.")
    if samples.loglam.shape != samples.w.shape:
        raise ValueError("loglam and w must have the same [M,K] shape.")
    if samples.omega_xyz.shape[:2] != samples.w.shape:
        raise ValueError("omega_xyz and w must share the same [M,K] prefix.")
    if samples.Z.shape[0] != samples.w.shape[0]:
        raise ValueError("Z must have length M matching kernel samples.")
    if not torch.isfinite(samples.omega_xyz).all():
        raise ValueError("omega_xyz must be finite.")
    if not torch.isfinite(samples.loglam).all():
        raise ValueError("loglam must be finite.")
    if not torch.isfinite(samples.w).all():
        raise ValueError("Kernel weights must be finite.")
    if not torch.isfinite(samples.Z).all():
        raise ValueError("Z must be finite.")
    if torch.any(samples.Z <= 0):
        raise ValueError("Z must be positive.")
    weights = samples.w.sum(dim=1)
    if not torch.allclose(weights, torch.ones_like(weights), atol=atol, rtol=0.0):
        raise ValueError("Kernel weights must sum to 1 for each datum.")
    if torch.any(samples.w < 0):
        raise ValueError("Kernel weights must be non-negative.")
    omega_norm = torch.linalg.norm(samples.omega_xyz, dim=-1)
    if not torch.allclose(omega_norm, torch.ones_like(omega_norm), atol=omega_norm_atol, rtol=0.0):
        raise ValueError("omega_xyz vectors must be unit-normalized.")
    lam = torch.exp(samples.loglam)
    if not torch.isfinite(lam).all():
        raise ValueError("loglam implies non-finite wavelengths.")
    if torch.any(lam <= 0):
        raise ValueError("Sampled wavelengths must be strictly positive.")
    if require_z_close_to_one and not torch.allclose(
        samples.Z, torch.ones_like(samples.Z), atol=z_atol, rtol=0.0,
    ):
        raise ValueError("In flux-density mode, Z must be 1 for every datum.")
