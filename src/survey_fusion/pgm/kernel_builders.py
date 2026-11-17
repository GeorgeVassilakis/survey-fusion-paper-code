from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..data.bandpasses import Bandpass
from ..utils.wcs import PixelToOmegaWCS, TangentPlaneWCS, radec_deg_to_omega_xyz_torch
from .contracts import (
    ApertureSpec, DatumBatch, DatumKernelFactory, DatumKind, ExposureState, ExposureTable,
    GroupedKernelSamples, KernelSamples, MaskApertureSpec, SkyDiskApertureSpec,
    aperture_area_arcsec2, validate_kernel_samples, validated_spectral_source_scale,
    spectral_source_geometry_key,
)


@dataclass(slots=True)
class _KernelBuildPlan:
    n_rows: int
    order: np.ndarray
    sorted_exp: np.ndarray
    sorted_kind: np.ndarray
    sorted_idx: np.ndarray
    exposure_blocks: tuple[tuple[int, int, int], ...]
    sample_groups: tuple[tuple[int, int, int, int], ...]
    _tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

    def tensors_for_device(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = _device_key(device)
        if self._tensor_cache is None:
            self._tensor_cache = {}
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = (
                torch.tensor(self.order, dtype=torch.long, device=device),
                torch.tensor(self.sorted_exp, dtype=torch.long, device=device),
                torch.tensor(self.sorted_idx, dtype=torch.long, device=device),
            )
            self._tensor_cache[key] = cached
        return cached


def _device_key(device: torch.device) -> str:
    return str(torch.device(device))


def _normalize_omega(omega: torch.Tensor) -> torch.Tensor:
    norm_sq = omega.square().sum(dim=-1, keepdim=True)
    return omega * torch.rsqrt(norm_sq.clamp_min(1.0e-24))


def _to_kernel_samples_torch(
    *, omega: torch.Tensor, lam: torch.Tensor, k: int, normalization: float = 1.0
) -> KernelSamples:
    if omega.ndim != 3 or omega.shape[-1] != 3:
        raise ValueError("omega must have shape [M, K, 3].")
    if lam.shape != omega.shape[:2]:
        raise ValueError("lam must have shape [M, K] matching omega.")
    if k <= 0:
        raise ValueError("k must be positive.")
    device = omega.device
    n_rows = int(omega.shape[0])
    omega_t = omega.to(dtype=torch.float32)
    loglam_t = torch.log(torch.clamp(lam, min=1.0e-6)).to(dtype=torch.float32)
    w_t = torch.full((n_rows, k), 1.0 / float(k), dtype=torch.float32, device=device)
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Kernel normalization must be positive and finite.")
    z_t = torch.full(
        (n_rows,), float(normalization), dtype=torch.float32, device=device
    )
    return KernelSamples(omega_xyz=omega_t, loglam=loglam_t, w=w_t, Z=z_t)


def _pad_pixels_for_sigma(sigma_pix: float) -> int:
    sigma = max(float(sigma_pix), 0.0)
    return max(1, int(math.ceil(4.0 * sigma)) + 2)


class _PixelToOmegaBackend:
    def pixel_to_omega_xyz(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _TangentPlaneTorchBackend(_PixelToOmegaBackend):
    def __init__(self, wcs: TangentPlaneWCS):
        self._wcs = wcs

    def pixel_to_omega_xyz(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._wcs.pixel_to_omega_xyz_torch(x, y)


class _LookupPixelToOmegaBackend(_PixelToOmegaBackend):
    def __init__(self, obs_wcs: PixelToOmegaWCS, *, width: int, height: int, pad_pixels: int):
        if width <= 0 or height <= 0:
            raise ValueError("Lookup pixel-to-omega backend requires positive width and height.")
        pad = max(int(pad_pixels), 1)
        x_coords = np.arange(-pad, width + pad + 1, dtype=np.float64) - 0.5
        y_coords = np.arange(-pad, height + pad + 1, dtype=np.float64) - 0.5
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
        omega = np.asarray(obs_wcs.pixel_to_omega_xyz(xx, yy), dtype=np.float64)
        expected_shape = (int(y_coords.size), int(x_coords.size), 3)
        if omega.shape != expected_shape:
            raise ValueError(
                f"obs_wcs.pixel_to_omega_xyz must return shape {expected_shape}, got {omega.shape}."
            )
        norms = np.linalg.norm(omega, axis=-1)
        self._normalize_output = bool(np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0))
        self._lookup_cpu = torch.tensor(omega, dtype=torch.float32)
        self._lookup_by_device: dict[str, torch.Tensor] = {}
        self._lookup_grid_by_device: dict[str, torch.Tensor] = {}
        self._max_cached_devices = 2
        self._x0 = float(x_coords[0])
        self._y0 = float(y_coords[0])
        self._grid_width = int(x_coords.size)
        self._grid_height = int(y_coords.size)

    def _lookup(self, device: torch.device) -> torch.Tensor:
        key = _device_key(device)
        lookup = self._lookup_by_device.get(key)
        if lookup is None:
            if len(self._lookup_by_device) >= self._max_cached_devices:
                oldest = next(iter(self._lookup_by_device))
                self._lookup_by_device.pop(oldest, None)
                self._lookup_grid_by_device.pop(oldest, None)
            lookup = self._lookup_cpu.to(device=device)
            self._lookup_by_device[key] = lookup
        else:
            self._lookup_by_device.pop(key)
            self._lookup_by_device[key] = lookup
        return lookup

    def _lookup_grid(self, device: torch.device) -> torch.Tensor:
        key = _device_key(device)
        grid = self._lookup_grid_by_device.get(key)
        if grid is None:
            lookup = self._lookup(device)
            grid = lookup.permute(2, 0, 1).unsqueeze(0).contiguous()
            self._lookup_grid_by_device[key] = grid
        else:
            self._lookup_grid_by_device.pop(key)
            self._lookup_grid_by_device[key] = grid
        return grid

    def pixel_to_omega_xyz(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("x and y must have matching shapes.")
        device = x.device
        lookup = self._lookup(device)
        dtype = lookup.dtype
        x_in = x.to(dtype=dtype)
        y_in = y.to(dtype=dtype)
        x_last = torch.full((), self._x0 + float(self._grid_width - 1), device=device, dtype=dtype)
        y_last = torch.full((), self._y0 + float(self._grid_height - 1), device=device, dtype=dtype)
        neg_inf = torch.full((), float("-inf"), device=device, dtype=dtype)
        x_max = torch.nextafter(x_last, neg_inf)
        y_max = torch.nextafter(y_last, neg_inf)
        x_clamped = x_in.clamp(min=self._x0, max=x_max)
        y_clamped = y_in.clamp(min=self._y0, max=y_max)
        if device.type == "cuda":
            grid_x = 2.0 * (x_clamped.reshape(1, -1, 1) - self._x0) / float(self._grid_width - 1) - 1.0
            grid_y = 2.0 * (y_clamped.reshape(1, -1, 1) - self._y0) / float(self._grid_height - 1) - 1.0
            sample_grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(
                self._lookup_grid(device), sample_grid, mode="bilinear", padding_mode="border",
                align_corners=True,
            )
            omega = sampled.squeeze(0).squeeze(-1).transpose(0, 1).view(x.shape + (3,))
            if self._normalize_output:
                omega = _normalize_omega(omega)
            return omega
        ix0 = torch.floor(x_clamped - self._x0).to(dtype=torch.long)
        iy0 = torch.floor(y_clamped - self._y0).to(dtype=torch.long)
        tx = (x_clamped - (ix0.to(dtype=dtype) + self._x0)).unsqueeze(-1)
        ty = (y_clamped - (iy0.to(dtype=dtype) + self._y0)).unsqueeze(-1)

        flat = lookup.view(-1, 3)
        stride = self._grid_width
        idx00 = iy0 * stride + ix0
        idx10 = idx00 + 1
        idx01 = idx00 + stride
        idx11 = idx01 + 1

        shape = x.shape + (3,)
        v00 = flat[idx00.reshape(-1)].view(shape)
        v10 = flat[idx10.reshape(-1)].view(shape)
        v01 = flat[idx01.reshape(-1)].view(shape)
        v11 = flat[idx11.reshape(-1)].view(shape)

        top = v00 + tx * (v10 - v00)
        bottom = v01 + tx * (v11 - v01)
        omega = top + ty * (bottom - top)
        if self._normalize_output:
            omega = _normalize_omega(omega)
        return omega


def _make_wcs_backend(
    obs_wcs: PixelToOmegaWCS, *, width: int, height: int, pad_pixels: int
) -> _PixelToOmegaBackend:
    if isinstance(obs_wcs, TangentPlaneWCS):
        return _TangentPlaneTorchBackend(obs_wcs)
    return _LookupPixelToOmegaBackend(
        obs_wcs, width=int(width), height=int(height), pad_pixels=int(pad_pixels)
    )


class BandpassSampler:
    """Sample wavelengths from a photon-weighted bandpass CDF."""

    def __init__(self, bandpass: Bandpass):
        lam = np.asarray(bandpass.lam, dtype=np.float64)
        throughput = np.asarray(bandpass.throughput, dtype=np.float64)
        if lam.ndim != 1 or throughput.ndim != 1:
            raise ValueError("Bandpass wavelength and throughput arrays must be 1D.")
        if lam.size < 2 or throughput.size != lam.size:
            raise ValueError("Bandpass wavelength and throughput arrays must be same length >= 2.")
        order = np.argsort(lam)
        lam = lam[order]
        throughput = throughput[order]
        if np.any(np.diff(lam) <= 0):
            raise ValueError(f"Bandpass {bandpass.band} must have strictly increasing wavelengths.")
        weights = np.maximum(throughput, 0.0) * lam
        segment = 0.5 * (weights[:-1] + weights[1:]) * np.diff(lam)
        cdf = np.concatenate(([0.0], np.cumsum(segment)))
        if cdf[-1] <= 0:
            raise ValueError(f"Bandpass {bandpass.band} has non-positive photon-weighted integral.")
        self._lam_cpu = torch.tensor(lam, dtype=torch.float32)
        self._cdf_cpu = torch.tensor(cdf / cdf[-1], dtype=torch.float32)
        self._tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _device_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = _device_key(device)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = (self._lam_cpu.to(device=device), self._cdf_cpu.to(device=device))
            self._tensor_cache[key] = cached
        return cached

    def sample(
        self, rng: torch.Generator, n_rows: int, k: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        if n_rows < 0 or k <= 0:
            raise ValueError("n_rows must be non-negative and k must be positive.")
        device = torch.device("cpu") if device is None else torch.device(device)
        lam, cdf = self._device_tensors(device)
        if n_rows == 0:
            return torch.empty((0, k), dtype=lam.dtype, device=device)
        strata = torch.arange(k, device=device, dtype=lam.dtype).unsqueeze(0)
        u = (
            strata
            + torch.rand(
                (n_rows, k),
                generator=rng,
                device=device,
                dtype=lam.dtype,
            )
        ) / float(k)
        flat_u = u.reshape(-1)
        hi = torch.searchsorted(cdf, flat_u, right=True)
        hi = hi.clamp(1, cdf.numel() - 1)
        lo = hi - 1
        cdf_lo = cdf[lo]
        cdf_hi = cdf[hi]
        lam_lo = lam[lo]
        lam_hi = lam[hi]
        frac = (flat_u - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1.0e-12)
        return (lam_lo + frac * (lam_hi - lam_lo)).view(n_rows, k)


class PixelFootprintSampler:
    """Uniform sub-pixel footprint jitter in detector pixel units."""

    def sample(
        self, rng: torch.Generator, n_rows: int, k: int, *, device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_rows < 0 or k <= 0:
            raise ValueError("n_rows must be non-negative and k must be positive.")
        device = torch.device("cpu") if device is None else torch.device(device)
        dx = torch.rand((n_rows, k), generator=rng, device=device, dtype=dtype) - 0.5
        dy = torch.rand((n_rows, k), generator=rng, device=device, dtype=dtype) - 0.5
        return dx, dy


@dataclass(slots=True)
class GaussianPSF:
    sigma_pix: float

    def apply(
        self, rng: torch.Generator, dx: torch.Tensor, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = float(self.sigma_pix)
        if sigma <= 0:
            return dx, dy
        noise_x = torch.randn(dx.shape, generator=rng, device=dx.device, dtype=dx.dtype) * sigma
        noise_y = torch.randn(dy.shape, generator=rng, device=dy.device, dtype=dy.dtype) * sigma
        return dx + noise_x, dy + noise_y


class MaskApertureSampler:
    """Sample sky directions from a weighted detector-space aperture mask."""

    def __init__(
        self, aperture: MaskApertureSpec, obs_wcs: PixelToOmegaWCS,
        footprint_sampler: PixelFootprintSampler | None = None, *, psf_sigma_pix: float = 0.0,
    ):
        mask = np.asarray(aperture.mask, dtype=np.float64)
        if mask.ndim != 2:
            raise ValueError("MaskApertureSpec.mask must be a 2D array.")
        if not np.isfinite(mask).all():
            raise ValueError("MaskApertureSpec.mask must be finite.")
        if np.any(mask < 0):
            raise ValueError("MaskApertureSpec.mask must be non-negative.")
        flat = mask.reshape(-1)
        total = float(np.sum(flat))
        if total <= 0:
            raise ValueError("MaskApertureSpec.mask must have positive total weight.")
        self._height = int(mask.shape[0])
        self._width = int(mask.shape[1])
        self._cdf_cpu = torch.tensor(np.cumsum(flat) / total, dtype=torch.float32)
        self._cdf_by_device: dict[str, torch.Tensor] = {}
        self._backend = _make_wcs_backend(
            obs_wcs, width=self._width, height=self._height, pad_pixels=_pad_pixels_for_sigma(psf_sigma_pix),
        )
        self._footprint_sampler = footprint_sampler or PixelFootprintSampler()
        self._psf = GaussianPSF(float(psf_sigma_pix))

    def _cdf(self, device: torch.device) -> torch.Tensor:
        key = _device_key(device)
        cached = self._cdf_by_device.get(key)
        if cached is None:
            cached = self._cdf_cpu.to(device=device)
            self._cdf_by_device[key] = cached
        return cached

    def sample(
        self, rng: torch.Generator, n_rows: int, k: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else torch.device(device)
        cdf = self._cdf(device)
        dtype = cdf.dtype
        u = torch.rand((n_rows, k), generator=rng, device=device, dtype=dtype)
        flat_idx = torch.searchsorted(cdf, u.reshape(-1), right=True)
        flat_idx = flat_idx.clamp(max=cdf.numel() - 1).view(n_rows, k)
        y = torch.div(flat_idx, self._width, rounding_mode="floor").to(dtype=dtype)
        x = torch.remainder(flat_idx, self._width).to(dtype=dtype)
        dx, dy = self._footprint_sampler.sample(rng, n_rows, k, device=device, dtype=dtype)
        dx, dy = self._psf.apply(rng, dx, dy)
        omega = self._backend.pixel_to_omega_xyz(x + dx, y + dy)
        if omega.shape != (n_rows, k, 3):
            raise ValueError(
                "obs_wcs.pixel_to_omega_xyz must return shape [n_rows, k, 3] for mask apertures."
            )
        return omega


@dataclass(slots=True)
class SkyDiskApertureSampler:
    aperture: SkyDiskApertureSpec
    jitter_sigma_arcsec: float = 0.0

    def sample(
        self, rng: torch.Generator, n_rows: int, k: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        device = torch.device("cpu") if device is None else torch.device(device)
        center_ra_deg, center_dec_deg = self.aperture.center_radec_deg
        radius_arcsec = float(self.aperture.radius_arcsec)
        if radius_arcsec <= 0:
            raise ValueError("SkyDiskApertureSpec.radius_arcsec must be positive.")
        dtype = torch.float32
        r = radius_arcsec * torch.sqrt(torch.rand((n_rows, k), generator=rng, device=device, dtype=dtype))
        theta = 2.0 * math.pi * torch.rand((n_rows, k), generator=rng, device=device, dtype=dtype)
        dx_arcsec = r * torch.cos(theta)
        dy_arcsec = r * torch.sin(theta)
        if self.jitter_sigma_arcsec > 0:
            sigma = float(self.jitter_sigma_arcsec)
            dx_arcsec = (
                dx_arcsec + torch.randn((n_rows, k), generator=rng, device=device, dtype=dtype) * sigma
            )
            dy_arcsec = (
                dy_arcsec + torch.randn((n_rows, k), generator=rng, device=device, dtype=dtype) * sigma
            )
        cos_dec = math.cos(math.radians(center_dec_deg))
        if abs(cos_dec) < 1e-12:
            raise ValueError("SkyDiskApertureSpec center is too close to a pole for small-angle RA offsets.")
        ra = torch.full((n_rows, k), float(center_ra_deg), device=device, dtype=dtype)
        dec = torch.full((n_rows, k), float(center_dec_deg), device=device, dtype=dtype)
        ra = ra + dx_arcsec / (3600.0 * cos_dec)
        dec = dec + dy_arcsec / 3600.0
        return radec_deg_to_omega_xyz_torch(ra, dec)


class TophatBinSampler:
    def __init__(self, lam_edges: np.ndarray):
        edges = np.asarray(lam_edges, dtype=np.float64)
        if edges.ndim != 2 or edges.shape[-1] != 2 or edges.shape[0] == 0:
            raise ValueError("lam_edges must be a (n_bins, 2) array of [lower, upper] edges.")
        if np.any(edges[:, 1] <= edges[:, 0]):
            raise ValueError("lam_edges bins must have strictly positive width.")
        self._lam_edges_cpu = torch.tensor(edges, dtype=torch.float32)
        self._edge_cache: dict[str, torch.Tensor] = {}

    @property
    def n_bins(self) -> int:
        return int(self._lam_edges_cpu.shape[0])

    def _edges(self, device: torch.device) -> torch.Tensor:
        key = _device_key(device)
        cached = self._edge_cache.get(key)
        if cached is None:
            cached = self._lam_edges_cpu.to(device=device)
            self._edge_cache[key] = cached
        return cached

    def sample(
        self, rng: torch.Generator, datum_index: torch.Tensor, k: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        idx = datum_index.to(dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError("datum_index must be a 1D tensor.")
        device = idx.device if device is None else torch.device(device)
        if device != idx.device:
            idx = idx.to(device=device)
        if torch.any(idx < 0) or torch.any(idx >= self.n_bins):
            raise ValueError("datum_index contains out-of-bounds spectral bin ids.")
        edges = self._edges(device)
        lo = edges[idx, 0].unsqueeze(1)
        hi = edges[idx, 1].unsqueeze(1)
        return lo + torch.rand((idx.numel(), k), generator=rng, device=device, dtype=edges.dtype) * (hi - lo)


class GaussianLSF:
    def __init__(self, fwhm_angstrom: float | np.ndarray | None):
        if fwhm_angstrom is None:
            self._fwhm_scalar: float | None = None
            self._fwhm_cpu: torch.Tensor | None = None
            self._fwhm_cache: dict[str, torch.Tensor] = {}
            return
        if np.isscalar(fwhm_angstrom):
            self._fwhm_scalar = float(fwhm_angstrom)
            self._fwhm_cpu = None
            self._fwhm_cache = {}
            return
        fwhm_array = np.asarray(fwhm_angstrom, dtype=np.float64)
        if fwhm_array.ndim != 1:
            raise ValueError("Per-bin LSF FWHM must be a 1D array.")
        if not np.isfinite(fwhm_array).all():
            raise ValueError("Per-bin LSF FWHM must be finite.")
        if np.any(fwhm_array < 0):
            raise ValueError("Per-bin LSF FWHM must be non-negative.")
        self._fwhm_scalar = None
        self._fwhm_cpu = torch.tensor(fwhm_array, dtype=torch.float32)
        self._fwhm_cache: dict[str, torch.Tensor] = {}

    def _fwhm(self, device: torch.device) -> torch.Tensor | None:
        if self._fwhm_cpu is None:
            return None
        key = _device_key(device)
        cached = self._fwhm_cache.get(key)
        if cached is None:
            cached = self._fwhm_cpu.to(device=device)
            self._fwhm_cache[key] = cached
        return cached

    def sample_offsets(self, rng: torch.Generator, datum_index: torch.Tensor, k: int) -> torch.Tensor:
        idx = datum_index.to(dtype=torch.long)
        if idx.ndim != 1:
            raise ValueError("datum_index must be a 1D tensor.")
        device = idx.device
        if self._fwhm_scalar is not None:
            sigma = float(self._fwhm_scalar) / 2.355
            if sigma <= 0:
                return torch.zeros((idx.numel(), k), dtype=torch.float32, device=device)
            return torch.randn((idx.numel(), k), generator=rng, device=device, dtype=torch.float32) * sigma
        fwhm = self._fwhm(device)
        if fwhm is None:
            return torch.zeros((idx.numel(), k), dtype=torch.float32, device=device)
        if torch.any(idx < 0) or torch.any(idx >= fwhm.numel()):
            raise ValueError("datum_index contains out-of-bounds spectral bin ids for LSF array.")
        sigma = (fwhm[idx] / 2.355).unsqueeze(1)
        if torch.all(sigma <= 0):
            return torch.zeros((idx.numel(), k), dtype=torch.float32, device=device)
        return torch.randn((idx.numel(), k), generator=rng, device=device, dtype=torch.float32) * sigma


class ImagePixelKernelBuilder:
    def __init__(self, *, pixel_footprint_sampler: PixelFootprintSampler | None = None):
        self._pixel_footprint_sampler = pixel_footprint_sampler or PixelFootprintSampler()

    def build(
        self, exposure: ExposureState, datum_index: torch.Tensor, *, k: int, rng: torch.Generator,
        bandpass_sampler: BandpassSampler, device: torch.device, wcs_backend: _PixelToOmegaBackend,
        psf: GaussianPSF, validate_bounds: bool = True,
    ) -> KernelSamples:
        if exposure.image_height is None or exposure.image_width is None:
            raise ValueError("Image exposures must provide image_height and image_width.")
        if exposure.obs_wcs is None or not isinstance(exposure.obs_wcs, PixelToOmegaWCS):
            raise ValueError("Image exposures must provide obs_wcs implementing PixelToOmegaWCS.")
        if k <= 0:
            raise ValueError("k must be positive.")
        idx = datum_index.to(dtype=torch.long, device=device)
        if idx.ndim != 1:
            raise ValueError("datum_index must be a 1D tensor.")
        n_rows = int(idx.numel())
        height = int(exposure.image_height)
        width = int(exposure.image_width)
        n_pix = height * width
        if validate_bounds and (torch.any(idx < 0) or torch.any(idx >= n_pix)):
            raise ValueError("Image datum_index is out of bounds for the configured image shape.")

        x = torch.remainder(idx, width).to(dtype=torch.float32).unsqueeze(1)
        y = torch.div(idx, width, rounding_mode="floor").to(dtype=torch.float32).unsqueeze(1)
        dx, dy = self._pixel_footprint_sampler.sample(rng, n_rows, k, device=device, dtype=x.dtype)
        dx, dy = psf.apply(rng, dx, dy)
        omega = wcs_backend.pixel_to_omega_xyz(x + dx, y + dy)
        if omega.shape != (n_rows, k, 3):
            raise ValueError("obs_wcs.pixel_to_omega_xyz must return shape [n_rows, k, 3] for image pixels.")
        lam = bandpass_sampler.sample(rng, n_rows, k, device=device)
        return _to_kernel_samples_torch(omega=omega, lam=lam, k=k)


class SpectralBinKernelBuilder:
    def __init__(self, *, pixel_footprint_sampler: PixelFootprintSampler | None = None):
        self._pixel_footprint_sampler = pixel_footprint_sampler or PixelFootprintSampler()

    def _aperture_sampler(self, exposure: ExposureState) -> MaskApertureSampler | SkyDiskApertureSampler:
        aperture: ApertureSpec | None = exposure.aperture
        if aperture is None:
            raise ValueError("Spectral exposures must provide aperture metadata.")
        if isinstance(aperture, MaskApertureSpec):
            if exposure.obs_wcs is None or not isinstance(exposure.obs_wcs, PixelToOmegaWCS):
                raise ValueError("MaskApertureSpec requires obs_wcs implementing PixelToOmegaWCS.")
            psf_sigma_pix = 0.0
            psf_fwhm = float(exposure.psf_fwhm_arcsec or 0.0)
            if psf_fwhm > 0.0:
                pixel_scale = float(exposure.pixel_scale_arcsec)
                if pixel_scale <= 0.0:
                    raise ValueError("pixel_scale_arcsec must be positive for mask-aperture PSF jitter.")
                psf_sigma_pix = psf_fwhm / 2.355 / pixel_scale
            return MaskApertureSampler(
                aperture=aperture, obs_wcs=exposure.obs_wcs, footprint_sampler=self._pixel_footprint_sampler,
                psf_sigma_pix=psf_sigma_pix,
            )
        if isinstance(aperture, SkyDiskApertureSpec):
            psf_fwhm = float(exposure.psf_fwhm_arcsec or 0.0)
            jitter_sigma_arcsec = psf_fwhm / 2.355 if psf_fwhm > 0.0 else 0.0
            return SkyDiskApertureSampler(aperture=aperture, jitter_sigma_arcsec=jitter_sigma_arcsec)
        raise ValueError(f"Unsupported aperture type: {type(aperture)!r}")

    def build(
        self, exposure: ExposureState, datum_index: torch.Tensor, *, k: int, rng: torch.Generator,
        device: torch.device, lam_sampler: TophatBinSampler, lsf: GaussianLSF,
        aperture_sampler: MaskApertureSampler | SkyDiskApertureSampler, validate_bounds: bool = True,
    ) -> KernelSamples:
        if k <= 0:
            raise ValueError("k must be positive.")
        idx = datum_index.to(dtype=torch.long, device=device)
        if idx.ndim != 1:
            raise ValueError("datum_index must be a 1D tensor.")
        if validate_bounds and (torch.any(idx < 0) or torch.any(idx >= lam_sampler.n_bins)):
            raise ValueError("Spectral datum_index is out of bounds for lam_edges.")
        omega = aperture_sampler.sample(rng, idx.numel(), k, device=device)
        if omega.shape != (idx.numel(), k, 3):
            raise ValueError("Aperture samplers must return shape [M, K, 3].")
        lam = lam_sampler.sample(rng, idx, k, device=device)
        lam = lam + lsf.sample_offsets(rng, idx, k)
        lam = torch.clamp(lam, min=1.0e-6)
        if exposure.aperture is None:
            raise ValueError("Spectral exposures must provide aperture metadata.")
        samples = _to_kernel_samples_torch(
            omega=omega,
            lam=lam,
            k=k,
            normalization=aperture_area_arcsec2(
                exposure.aperture,
                pixel_scale_arcsec=exposure.pixel_scale_arcsec,
            ),
        )
        response = validated_spectral_source_scale(exposure)
        if response is not None:
            if response.shape != (lam_sampler.n_bins,):
                raise ValueError("spectral_source_scale must match wavelength-bin edges.")
            scale = torch.tensor(response, device=device, dtype=samples.Z.dtype)[idx]
            samples.Z = samples.Z * scale
            if not bool(torch.isfinite(samples.Z).all()) or bool(torch.any(samples.Z <= 0)):
                raise ValueError("spectral_source_scale normalization is not representable in MC dtype.")
        return samples


class KernelFactoryFromBuilders(DatumKernelFactory):
    def __init__(
        self, *, batch: DatumBatch, exposures: ExposureTable, image_band_by_exposure: Dict[int, str],
        rng_seed: int | None, image_builder: ImagePixelKernelBuilder | None = None,
        spectral_builder: SpectralBinKernelBuilder | None = None, validate: bool = True,
    ):
        self._batch = batch
        self._exposures = exposures
        self._image_band_by_exposure = {int(k): str(v) for k, v in image_band_by_exposure.items()}
        self._rng_seed = int(rng_seed) if rng_seed is not None else int(torch.seed())
        self._generators: dict[str, torch.Generator] = {}
        self._batch_exposure_id_cpu = batch.exposure_id.detach().to("cpu").numpy()
        self._batch_datum_kind_cpu = batch.datum_kind.detach().to("cpu").numpy()
        self._batch_datum_index_cpu = batch.datum_index.detach().to("cpu").numpy()
        self._image_builder = image_builder or ImagePixelKernelBuilder()
        self._spectral_builder = spectral_builder or SpectralBinKernelBuilder()
        self._bandpass_sampler_by_exposure: dict[int, BandpassSampler] = {}
        self._spectral_lam_sampler_by_exposure: dict[int, TophatBinSampler] = {}
        self._spectral_lsf_by_exposure: dict[int, GaussianLSF] = {}
        self._spectral_response_geometry: dict[int, tuple | None] = {}
        self._spectral_aperture_sampler_by_exposure: dict[
            int, MaskApertureSampler | SkyDiskApertureSampler
        ] = {}
        self._image_wcs_backend_by_exposure: dict[int, _PixelToOmegaBackend] = {}
        self._image_psf_by_exposure: dict[int, GaussianPSF] = {}
        self._validate = bool(validate)
        for exp_id, band in self._image_band_by_exposure.items():
            exposure = self._exposures.get(int(exp_id))
            if exposure is None:
                raise ValueError(f"image_band_by_exposure references unknown exposure_id={exp_id}.")
            bandpass = exposure.bandpasses.get(str(band))
            if bandpass is None:
                raise ValueError(f"Exposure {exp_id} does not provide bandpass metadata for band {band!r}.")
            self._bandpass_sampler_by_exposure[int(exp_id)] = BandpassSampler(bandpass)
            self._image_wcs_backend_by_exposure[int(exp_id)] = self._build_image_wcs_backend(exposure)
            self._image_psf_by_exposure[int(exp_id)] = self._build_image_psf(exposure)

    def _generator_for_device(self, device: torch.device) -> torch.Generator:
        normalized = torch.device(device)
        if normalized.type == "cuda" and normalized.index is None:
            normalized = torch.device(f"cuda:{torch.cuda.current_device()}")
        key = str(normalized)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=normalized)
            generator.manual_seed(self._rng_seed)
            self._generators[key] = generator
        return generator

    def rng_state_dict(self) -> dict[str, object]:
        """Per-device jitter-generator positions, for checkpointing.

        The seed alone only identifies the stream; without the positions a
        resumed run re-draws the original run's jitter prefix."""
        return {
            "rng_seed": int(self._rng_seed),
            "generators": {
                key: generator.get_state()
                for key, generator in self._generators.items()
            },
        }

    def load_rng_state_dict(self, state: Dict[str, object]) -> list[str]:
        """Restore generator positions; returns keys this host cannot honour."""
        generators = state.get("generators") or {}
        if not isinstance(generators, dict):
            raise ValueError("kernel factory rng state 'generators' must be a mapping.")
        skipped: list[str] = []
        for key, value in generators.items():
            device = torch.device(str(key))
            if device.type == "cuda" and (
                not torch.cuda.is_available()
                or (device.index or 0) >= torch.cuda.device_count()
            ):
                skipped.append(str(key))
                continue
            # generator states are CPU byte tensors even for CUDA generators;
            # a checkpoint loaded with map_location may have moved them
            self._generator_for_device(device).set_state(value.cpu())
        return skipped

    @staticmethod
    def _empty_kernel_samples(k: int, device: torch.device) -> KernelSamples:
        return KernelSamples(
            omega_xyz=torch.empty((0, k, 3), dtype=torch.float32, device=device),
            loglam=torch.empty((0, k), dtype=torch.float32, device=device),
            w=torch.empty((0, k), dtype=torch.float32, device=device),
            Z=torch.empty((0,), dtype=torch.float32, device=device),
        )

    @staticmethod
    def _concat_samples(parts: list[KernelSamples]) -> KernelSamples:
        if not parts:
            raise ValueError("Cannot concatenate an empty sample list.")
        if len(parts) == 1:
            return parts[0]
        return KernelSamples(
            omega_xyz=torch.cat([part.omega_xyz for part in parts], dim=0),
            loglam=torch.cat([part.loglam for part in parts], dim=0),
            w=torch.cat([part.w for part in parts], dim=0), Z=torch.cat([part.Z for part in parts], dim=0),
        )

    @staticmethod
    def _split_kernel_sample_replicas(samples: KernelSamples, k: int) -> tuple[KernelSamples, KernelSamples]:
        if k <= 0:
            raise ValueError("k must be positive.")
        if samples.w.shape[1] != 2 * int(k):
            raise ValueError("Fused replica samples must have exactly 2*k columns.")
        first = slice(0, int(k))
        second = slice(int(k), 2 * int(k))
        return (
            KernelSamples(
                omega_xyz=samples.omega_xyz[: , first,: ], loglam=samples.loglam[: , first],
                w=samples.w[: , first] * 2.0, Z=samples.Z,
            ),
            KernelSamples(
                omega_xyz=samples.omega_xyz[: , second,: ], loglam=samples.loglam[: , second],
                w=samples.w[: , second] * 2.0, Z=samples.Z,
            ),
        )

    @staticmethod
    def _build_image_psf(exposure: ExposureState) -> GaussianPSF:
        psf_fwhm = float(exposure.psf_fwhm_arcsec or 0.0)
        sigma_pix = 0.0
        if psf_fwhm > 0:
            pixel_scale = float(exposure.pixel_scale_arcsec)
            if pixel_scale <= 0:
                raise ValueError("pixel_scale_arcsec must be positive when PSF is enabled.")
            sigma_pix = (psf_fwhm / 2.355) / pixel_scale
        return GaussianPSF(sigma_pix=sigma_pix)

    def _build_image_wcs_backend(self, exposure: ExposureState) -> _PixelToOmegaBackend:
        if exposure.obs_wcs is None or not isinstance(exposure.obs_wcs, PixelToOmegaWCS):
            raise ValueError("Image exposures must provide obs_wcs implementing PixelToOmegaWCS.")
        if exposure.image_height is None or exposure.image_width is None:
            raise ValueError("Image exposures must provide image_height and image_width.")
        sigma_pix = self._build_image_psf(exposure).sigma_pix
        return _make_wcs_backend(
            exposure.obs_wcs, width=int(exposure.image_width), height=int(exposure.image_height),
            pad_pixels=_pad_pixels_for_sigma(sigma_pix),
        )

    def build_kernel_sample_plan_for_indices(self, global_indices: torch.Tensor) -> _KernelBuildPlan:
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        n_rows = int(global_indices.numel())
        if n_rows == 0:
            empty = np.empty((0,), dtype=np.int64)
            return _KernelBuildPlan(
                n_rows=0, order=empty, sorted_exp=empty, sorted_kind=empty, sorted_idx=empty,
                exposure_blocks=(), sample_groups=(),
            )

        idx = global_indices.detach().to(dtype=torch.long, device="cpu").numpy()
        exposure_id = self._batch_exposure_id_cpu[idx]
        datum_kind = self._batch_datum_kind_cpu[idx]
        datum_index = self._batch_datum_index_cpu[idx]
        original_position = np.arange(n_rows, dtype=np.int64)
        exp_i = exposure_id.astype(np.int64, copy=False)
        kind_i = datum_kind.astype(np.int64, copy=False)
        datum_i = datum_index.astype(np.int64, copy=False)
        if n_rows == 1:
            order = original_position
        elif bool(
            np.all(
                (exp_i[:-1] < exp_i[1:])
                | ((exp_i[:-1] == exp_i[1:]) & (kind_i[:-1] < kind_i[1:]))
                | ((exp_i[:-1] == exp_i[1:]) & (kind_i[:-1] == kind_i[1:]) & (datum_i[:-1] <= datum_i[1:]))
            )
        ):
            order = original_position
        else:
            order = np.lexsort((original_position, datum_i, kind_i, exp_i))

        sorted_exp = exposure_id[order]
        sorted_kind = datum_kind[order]
        sorted_idx = datum_index[order]

        exposure_blocks: list[tuple[int, int, int]] = []
        block_start = 0
        while block_start < n_rows:
            block_exp_id = int(sorted_exp[block_start])
            block_end = block_start + 1
            while block_end < n_rows and int(sorted_exp[block_end]) == block_exp_id:
                block_end += 1
            exposure_blocks.append((block_exp_id, block_start, block_end))
            block_start = block_end

        sample_groups: list[tuple[int, int, int, int]] = []
        start = 0
        while start < n_rows:
            exp_id = int(sorted_exp[start])
            kind = int(sorted_kind[start])
            end = start + 1
            while end < n_rows and int(sorted_exp[end]) == exp_id and int(sorted_kind[end]) == kind:
                end += 1
            sample_groups.append((exp_id, kind, start, end))
            start = end

        return _KernelBuildPlan(
            n_rows=n_rows, order=order, sorted_exp=sorted_exp, sorted_kind=sorted_kind, sorted_idx=sorted_idx,
            exposure_blocks=tuple(exposure_blocks), sample_groups=tuple(sample_groups),
        )

    def build_kernel_samples_from_plan(
        self, plan: _KernelBuildPlan, *, k: int, device: torch.device, validate: bool | None = None
    ) -> GroupedKernelSamples:
        device = torch.device(device)
        if k <= 0:
            raise ValueError("MC mode requires a positive kernel sample count k.")
        if plan.n_rows == 0:
            empty = self._empty_kernel_samples(k=k, device=device)
            empty_idx = torch.empty((0,), dtype=torch.long, device=device)
            return GroupedKernelSamples(
                samples=empty, row_original_index=empty_idx, row_exposure_id=empty_idx,
                row_datum_index=empty_idx, exposure_blocks=(),
            )

        do_validate = self._validate if validate is None else bool(validate)
        generator = self._generator_for_device(device)
        row_original_index_t, row_exposure_id_t, sorted_idx_t = plan.tensors_for_device(device)
        grouped: list[KernelSamples] = []
        for exp_id, kind, start, end in plan.sample_groups:
            exposure = self._exposures.get(exp_id)
            if exposure is None:
                raise ValueError(f"Batch references missing exposure_id={exp_id}.")
            local_idx = sorted_idx_t[start:end]
            if kind == int(DatumKind.IMAGE_PIXEL):
                bandpass_sampler = self._bandpass_sampler_by_exposure.get(exp_id)
                if bandpass_sampler is None:
                    raise ValueError(f"No image bandpass sampler is configured for exposure {exp_id}.")
                samples = self._image_builder.build(
                    exposure, local_idx, k=k, rng=generator, bandpass_sampler=bandpass_sampler, device=device,
                    wcs_backend=self._image_wcs_backend_by_exposure[exp_id],
                    psf=self._image_psf_by_exposure[exp_id], validate_bounds=do_validate,
                )
            elif kind == int(DatumKind.SPECTRAL_BIN):
                geometry = spectral_source_geometry_key(exposure)
                if exp_id in self._spectral_response_geometry:
                    if geometry != self._spectral_response_geometry[exp_id]:
                        raise ValueError("MC spectral response geometry is stale; rebuild the kernel factory.")
                else:
                    self._spectral_response_geometry[exp_id] = geometry
                lam_sampler = self._spectral_lam_sampler_by_exposure.get(exp_id)
                if lam_sampler is None:
                    if exposure.lam_edges is None:
                        raise ValueError("Spectral exposures must provide lam_edges.")
                    lam_sampler = TophatBinSampler(np.asarray(exposure.lam_edges, dtype=np.float64))
                    self._spectral_lam_sampler_by_exposure[exp_id] = lam_sampler

                lsf = self._spectral_lsf_by_exposure.get(exp_id)
                if lsf is None:
                    if isinstance(exposure.lsf_fwhm_angstrom, np.ndarray):
                        if exposure.lsf_fwhm_angstrom.ndim != 1:
                            raise ValueError("ExposureState.lsf_fwhm_angstrom array must be 1D.")
                        if exposure.lsf_fwhm_angstrom.shape[0] != lam_sampler.n_bins:
                            raise ValueError(
                                "ExposureState.lsf_fwhm_angstrom array length must match spectral bin count."
                            )
                    lsf = GaussianLSF(exposure.lsf_fwhm_angstrom)
                    self._spectral_lsf_by_exposure[exp_id] = lsf

                aperture_sampler = self._spectral_aperture_sampler_by_exposure.get(exp_id)
                if aperture_sampler is None:
                    aperture_sampler = self._spectral_builder._aperture_sampler(exposure)
                    self._spectral_aperture_sampler_by_exposure[exp_id] = aperture_sampler

                samples = self._spectral_builder.build(
                    exposure, local_idx, k=k, rng=generator, device=device, lam_sampler=lam_sampler, lsf=lsf,
                    aperture_sampler=aperture_sampler, validate_bounds=do_validate,
                )
            else:
                raise ValueError(f"Unsupported datum_kind={kind}.")
            grouped.append(samples)

        sorted_samples = self._concat_samples(grouped)
        if do_validate:
            validate_kernel_samples(sorted_samples)
        return GroupedKernelSamples(
            samples=sorted_samples, row_original_index=row_original_index_t,
            row_exposure_id=row_exposure_id_t, row_datum_index=sorted_idx_t,
            exposure_blocks=plan.exposure_blocks,
        )

    def build_kernel_sample_pair_from_plan(
        self, plan: _KernelBuildPlan, *, k: int, device: torch.device, validate: bool | None = None
    ) -> tuple[GroupedKernelSamples, GroupedKernelSamples]:
        fused = self.build_kernel_samples_from_plan(plan, k=2 * int(k), device=device, validate=validate)
        samples_a, samples_b = self._split_kernel_sample_replicas(fused.samples, int(k))
        grouped_a = GroupedKernelSamples(
            samples=samples_a, row_original_index=fused.row_original_index,
            row_exposure_id=fused.row_exposure_id, row_datum_index=fused.row_datum_index,
            exposure_blocks=fused.exposure_blocks,
        )
        grouped_b = GroupedKernelSamples(
            samples=samples_b, row_original_index=fused.row_original_index,
            row_exposure_id=fused.row_exposure_id, row_datum_index=fused.row_datum_index,
            exposure_blocks=fused.exposure_blocks,
        )
        return grouped_a, grouped_b

    def build_kernel_samples_for_indices(
        self, global_indices: torch.Tensor, *, k: int, device: torch.device, validate: bool | None = None
    ) -> GroupedKernelSamples:
        plan = self.build_kernel_sample_plan_for_indices(global_indices)
        return self.build_kernel_samples_from_plan(plan, k=k, device=device, validate=validate)
