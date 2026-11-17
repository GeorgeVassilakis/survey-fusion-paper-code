from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..data.bandpasses import Bandpass
from ..pgm.contracts import PSFMixtureComponents


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_psf_kernel(
    fwhm_arcsec: float, pixel_scale_arcsec: float, device: torch.device, *, dtype: torch.dtype=torch.float32,
    truncate: float = 4.0,
) -> torch.Tensor:
    """Build a 2D Gaussian PSF kernel for discrete image convolution.

    Returns a tensor of shape [1, 1, k, k] normalized to sum to 1.
    """

    fwhm = float(fwhm_arcsec)
    pixel_scale = float(pixel_scale_arcsec)
    trunc = float(truncate)
    if fwhm <= 0 or not math.isfinite(fwhm):
        raise ValueError(f"PSF FWHM must be positive and finite, got {fwhm_arcsec!r}.")
    if pixel_scale <= 0 or not math.isfinite(pixel_scale):
        raise ValueError(f"pixel_scale_arcsec must be positive and finite, got {pixel_scale_arcsec!r}.")
    if trunc <= 0 or not math.isfinite(trunc):
        raise ValueError(f"truncate must be positive and finite, got {truncate!r}.")

    # FWHM = 2 * sqrt(2 * ln(2)) * sigma for a Gaussian.
    sigma_pix = (fwhm * FWHM_TO_SIGMA) / pixel_scale
    radius = int(math.ceil(trunc * sigma_pix))
    radius = max(radius, 0)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kern = torch.exp(-0.5 * ((xx / sigma_pix) ** 2 + (yy / sigma_pix) ** 2))
    kern = kern / torch.clamp(kern.sum(), min=1e-12)
    return kern.view(1, 1, kern.shape[0], kern.shape[1])


def gaussian_mixture_psf_kernel(
    mixture: PSFMixtureComponents,
    *,
    supersample: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    truncate: float = 4.0,
) -> torch.Tensor:
    """Build one normalized discrete kernel for a native-pixel PSF mixture.

    The grid reaches ``truncate`` times the widest component sigma and the
    sampled sum is renormalized.  The analytic image operator is the oracle;
    this deterministic-convolution path necessarily has discrete sampling and
    truncation error.
    """
    factor = int(supersample)
    trunc = float(truncate)
    if factor < 1:
        raise ValueError("supersample must be a positive integer.")
    if not math.isfinite(trunc) or trunc <= 0.0:
        raise ValueError("truncate must be positive and finite.")
    widest = 0.0
    for _, vxx, vxy, vyy in mixture.components:
        eigen_max = 0.5 * (vxx + vyy) + math.sqrt(
            max(0.25 * (vxx - vyy) ** 2 + vxy**2, 0.0)
        )
        widest = max(widest, math.sqrt(eigen_max))
    radius = int(math.ceil(trunc * widest * factor))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.zeros_like(xx)
    covariance_scale = float(factor * factor)
    for weight, vxx, vxy, vyy in mixture.components:
        a = float(vxx) * covariance_scale
        b = float(vxy) * covariance_scale
        d = float(vyy) * covariance_scale
        determinant = a * d - b * b
        q = (d * xx.square() - 2.0 * b * xx * yy + a * yy.square()) / determinant
        kernel = kernel + float(weight) * torch.exp(-0.5 * q) / (
            2.0 * math.pi * math.sqrt(determinant)
        )
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def convolve_flat_image(
    image_flat: torch.Tensor, height: int, width: int, kernel: torch.Tensor
) -> torch.Tensor:
    """Convolve a flattened image with a [1,1,k,k] kernel using same-size output."""

    if image_flat.numel() != height * width:
        raise ValueError(f"image_flat has {image_flat.numel()} elements but height*width={height*width}.")
    if kernel.ndim != 4 or kernel.shape[0] != 1 or kernel.shape[1] != 1:
        raise ValueError(f"kernel must have shape [1,1,k,k], got {tuple(kernel.shape)}.")
    padding = kernel.shape[-1] // 2
    image = image_flat.view(1, 1, height, width)
    convolved = F.conv2d(image, kernel, padding=padding)
    return convolved.view(-1)


def integrate_aperture(
    field_pixels: torch.Tensor, weights_flat: torch.Tensor, *, normalize: bool,
) -> torch.Tensor:
    """Integrate a spectral field over pixels with optional normalization.

    Args:
        field_pixels: [n_pix, n_lambda] tensor.
        weights_flat: [n_pix] tensor of non-negative weights.
        normalize: If True, divide by sum(weights_flat) to form an area-average.

    Returns:
        [n_lambda] tensor.
    """

    if field_pixels.ndim != 2:
        raise ValueError(f"field_pixels must be 2D [n_pix,n_lambda], got {field_pixels.ndim}D.")
    if weights_flat.ndim != 1:
        raise ValueError(f"weights_flat must be 1D [n_pix], got {weights_flat.ndim}D.")
    if field_pixels.shape[0] != weights_flat.shape[0]:
        raise ValueError(
            f"weights_flat has length {weights_flat.shape[0]} but field_pixels has "
            f"{field_pixels.shape[0]} pixels."
        )
    weighted_sum = (field_pixels * weights_flat.unsqueeze(1)).sum(dim=0)
    if not normalize:
        return weighted_sum
    denom = torch.clamp(weights_flat.sum(), min=1e-12)
    return weighted_sum / denom


@dataclass(slots=True)
class SpectralGrid:
    lam: torch.Tensor
    log_lam: torch.Tensor
    log_min: float
    log_max: float
    dlam: torch.Tensor


@dataclass(slots=True)
class SpectrumProjector:
    idx_left: torch.Tensor
    idx_right: torch.Tensor
    weight: torch.Tensor
    throughput: torch.Tensor | None
    lsf_kernel: torch.Tensor | None
    valid_mask: torch.Tensor | None
    lam_spec: torch.Tensor
    photon_weight: torch.Tensor
    edge_idx_left: torch.Tensor | None = None
    edge_idx_right: torch.Tensor | None = None
    edge_weight: torch.Tensor | None = None
    bin_width: torch.Tensor | None = None


class SpectralRenderer:
    def __init__(self, bandpasses: Dict[str, Bandpass], lambda_step: float, device: torch.device):
        self.device = device
        self.default_normalize = False
        lam_all = np.concatenate([bp.lam for bp in bandpasses.values()])
        lam_min, lam_max = lam_all.min(), lam_all.max()
        lam_grid_np = np.arange(lam_min, lam_max + lambda_step, lambda_step, dtype=np.float64)
        lam_grid = torch.tensor(lam_grid_np, dtype=torch.float32, device=device)
        log_lam = torch.log(lam_grid)
        dlam = (lam_grid[1] - lam_grid[0]) if lam_grid.numel() > 1 else torch.tensor(1.0, device=device)
        self.grid = SpectralGrid(lam_grid, log_lam, float(log_lam.min()), float(log_lam.max()), dlam)
        self.band_weights, self.band_norms = self._build_band_weights(bandpasses)

    def _build_band_weights(
        self, bandpasses: Dict[str, Bandpass]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights = {}
        norms = {}
        lam_target = self.grid.lam.detach().cpu().numpy()
        for band, bp in bandpasses.items():
            T = np.interp(lam_target, bp.lam, bp.throughput, left=0.0, right=0.0)
            wt = torch.tensor(T * lam_target, dtype=torch.float32, device=self.device)
            weights[band] = wt
            norm_val = float(np.trapezoid(bp.throughput * bp.lam, bp.lam))
            norms[band] = torch.tensor(norm_val, dtype=torch.float32, device=self.device)
        return weights, norms

    def render_band(self, field_vals: torch.Tensor, band: str, normalize: bool = False) -> torch.Tensor:
        weights = self.band_weights[band]
        dlam = self.grid.dlam
        band_vals = (field_vals * weights).sum(dim=-1) * dlam
        if normalize:
            norm = torch.clamp(self.band_norms[band], min=1e-12)
            band_vals = band_vals / norm
        return band_vals

    def build_spectrum_projector(
        self, lam_spec: np.ndarray | None, spec_throughput: np.ndarray | None, lsf_fwhm: float,
        lam_edges: np.ndarray | None = None,
    ) -> SpectrumProjector | None:
        if lam_spec is None:
            return None
        lam_spec_np = lam_spec.astype(np.float64)
        lam_grid_np = self.grid.lam.detach().cpu().numpy()
        def _interp_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            idx_right_local = np.searchsorted(lam_grid_np, values, side="right")
            idx_right_local = np.clip(idx_right_local, 1, len(lam_grid_np) - 1)
            idx_left_local = np.clip(idx_right_local - 1, 0, len(lam_grid_np) - 1)
            lam_left_local = lam_grid_np[idx_left_local]
            lam_right_local = lam_grid_np[idx_right_local]
            denom_local = lam_right_local - lam_left_local
            denom_local[denom_local == 0] = 1.0
            weights_local = (values - lam_left_local) / denom_local
            return idx_left_local, idx_right_local, weights_local

        idx_left, idx_right, weights = _interp_indices(lam_spec_np)
        idx_left_t = torch.tensor(idx_left, dtype=torch.long, device=self.device)
        idx_right_t = torch.tensor(idx_right, dtype=torch.long, device=self.device)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        throughput_t = (
            torch.tensor(spec_throughput, dtype=torch.float32, device=self.device)
            if spec_throughput is not None
            else None
        )
        edge_idx_left_t = None
        edge_idx_right_t = None
        edge_weight_t = None
        bin_width_t = None
        if lam_edges is not None:
            lam_edges_np = np.asarray(lam_edges, dtype=np.float64)
            if lam_edges_np.shape == (lam_spec_np.size, 2):
                # per-bin [lower, upper] pairs, interpolated as one flat
                # sequence of 2n edges and re-paired at projection time
                edge_idx_left, edge_idx_right, edge_weight = _interp_indices(
                    lam_edges_np.reshape(-1)
                )
                edge_idx_left_t = torch.tensor(edge_idx_left, dtype=torch.long, device=self.device)
                edge_idx_right_t = torch.tensor(edge_idx_right, dtype=torch.long, device=self.device)
                edge_weight_t = torch.tensor(edge_weight, dtype=torch.float32, device=self.device)
                bin_width_t = torch.tensor(
                    lam_edges_np[:, 1] - lam_edges_np[:, 0], dtype=torch.float32, device=self.device,
                )
        if len(lam_spec_np) > 1:
            dlam = float(np.median(np.diff(lam_spec_np)))
        else:
            dlam = 0.5
        sigma_pix = (lsf_fwhm / 2.355) / dlam if dlam > 0 else 0.0
        if sigma_pix <= 0 or not np.isfinite(sigma_pix):
            lsf_kernel = None
        else:
            half_width = int(np.ceil(6 * sigma_pix))
            x = torch.arange(-half_width, half_width + 1, dtype=torch.float32, device=self.device)
            lsf_kernel = torch.exp(-0.5 * (x / sigma_pix) ** 2)
            lsf_kernel /= lsf_kernel.sum()
        lam_spec_t = torch.tensor(lam_spec_np, dtype=torch.float32, device=self.device)
        if bin_width_t is not None:
            photon_weight_t = bin_width_t / 10.0
        else:
            photon_weight_t = torch.full_like(lam_spec_t, float(dlam / 10.0))
        return SpectrumProjector(
            idx_left_t, idx_right_t, weight_t, throughput_t, lsf_kernel, None, lam_spec_t, photon_weight_t,
            edge_idx_left_t, edge_idx_right_t, edge_weight_t, bin_width_t,
        )

    def project_field_to_spectrum(
        self, field_sum: torch.Tensor, projector: SpectrumProjector, apply_photon_weight: bool = True,
        apply_throughput: bool = True, apply_lsf: bool = True,
    ) -> torch.Tensor:
        if (
            projector.edge_idx_left is not None
            and projector.edge_idx_right is not None
            and projector.edge_weight is not None
            and projector.bin_width is not None
        ):
            lam_grid = self.grid.lam.to(device=field_sum.device, dtype=field_sum.dtype)
            if lam_grid.numel() <= 1:
                field_spec = field_sum.new_zeros(projector.bin_width.shape)
            else:
                trapezoids = 0.5 * (field_sum[1:] + field_sum[:-1]) * (lam_grid[1:] - lam_grid[:-1])
                cumulative = torch.cat([field_sum.new_zeros((1,)), torch.cumsum(trapezoids, dim=0)], dim=0)
                edge_left = torch.index_select(cumulative, 0, projector.edge_idx_left)
                edge_right = torch.index_select(cumulative, 0, projector.edge_idx_right)
                cumulative_edges = (
                    (1.0 - projector.edge_weight) * edge_left
                    + projector.edge_weight * edge_right
                ).view(-1, 2)
                field_spec = (cumulative_edges[:, 1] - cumulative_edges[:, 0]) / torch.clamp(
                    projector.bin_width.to(device=field_sum.device, dtype=field_sum.dtype), min=1.0e-12,
                )
        else:
            field_left = torch.index_select(field_sum, 0, projector.idx_left)
            field_right = torch.index_select(field_sum, 0, projector.idx_right)
            field_spec = (1.0 - projector.weight) * field_left + projector.weight * field_right
        if apply_photon_weight:
            field_spec = field_spec * projector.photon_weight
        if apply_throughput and projector.throughput is not None:
            field_spec = field_spec * projector.throughput
        if apply_lsf and projector.lsf_kernel is not None:
            padding = projector.lsf_kernel.shape[0] // 2
            field_spec = F.conv1d(
                field_spec.view(1, 1, -1), projector.lsf_kernel.view(1, 1, -1), padding=padding,
            ).view(-1)
        return field_spec
