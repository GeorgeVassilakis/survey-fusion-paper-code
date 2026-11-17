"""Closed-form band-image prediction for SplatField.

A Gaussian splat convolved with a Gaussian PSF is another Gaussian:
G(cov) * PSF(sigma^2 I) = G(cov + sigma^2 I), so per-pixel band predictions
need no Monte-Carlo kernel sampling. The bandpass integral is handled by
averaging each splat's SED over the band's discrete loglam samples, matching
the pipeline's discrete bandpass grids. The smooth sky term is evaluated
pointwise (it varies slowly, so PSF convolution is a no-op at working
tolerance) and band-averaged the same way.
"""

from __future__ import annotations

import math

import torch

from survey_fusion.pgm.splat_raster import PointPSFMixture, band_amplitudes


def analytic_band_prediction(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    psf_sigma: float,
    bandpass_ll: torch.Tensor,
    bandpass_weights: torch.Tensor | None = None,
    *,
    psf_mixture: PointPSFMixture | None = None,
) -> torch.Tensor:
    """x, y: (n,) pixel centers in normalized tangent-plane coords;
    psf_sigma: Gaussian PSF stddev in the same coords; bandpass_ll: (m,)
    loglam samples of the band; bandpass_weights: optional (m,) photon
    weights (normalized internally; uniform if omitted). Returns (n,)
    band-averaged flux."""
    # Per-splat band amplitude comes from the shared, splat-chunked helper.
    # Computing it inline here materialised (n_templates, n_splats, n_bandpass)
    # in one allocation, which at 200k splats is 3 GiB and OOMed the run before
    # the first step.
    amps = band_amplitudes(field, bandpass_ll, bandpass_weights)
    if psf_mixture is not None:
        if psf_mixture.weights.shape[0] not in (1, int(x.numel())):
            raise ValueError("Dense PSF mixture must have one row or one row per query point.")
        result = torch.zeros_like(x)
        base_covariance = field.covariances()
        for component in range(psf_mixture.n_components):
            weights = psf_mixture.weights[:, component]
            component_covariance = psf_mixture.covariances[:, component]
            if weights.numel() == 1:
                weights = weights.expand(x.numel())
                component_covariance = component_covariance.expand(
                    (x.numel(),) + component_covariance.shape[1:]
                )
            if psf_mixture.isotropic:
                covariance = base_covariance[None, :, :, :] + component_covariance[:, None, None, None] * torch.eye(
                    2, dtype=x.dtype, device=x.device
                )
            else:
                matrix = torch.stack((
                    component_covariance[:, 0], component_covariance[:, 1],
                    component_covariance[:, 1], component_covariance[:, 2],
                ), dim=-1).reshape(-1, 2, 2)
                covariance = base_covariance[None, :, :, :] + matrix[:, None, :, :]
            a, b = covariance[..., 0, 0], covariance[..., 0, 1]
            c, dd = covariance[..., 1, 0], covariance[..., 1, 1]
            det = (a * dd - b * c).clamp_min(1e-30)
            dx = x[:, None] - field.positions()[None, :, 0]
            dy = y[:, None] - field.positions()[None, :, 1]
            q = (dd * dx.square() - (b + c) * dx * dy + a * dy.square()) / det
            gaussian = torch.exp(-0.5 * q) / (2.0 * math.pi * torch.sqrt(det))
            result = result + weights * (gaussian * amps[None, :]).sum(dim=-1)
        return result
    cov = field.covariances() + (float(psf_sigma) ** 2) * torch.eye(
        2, dtype=x.dtype, device=x.device
    )
    # closed-form 2x2 inverse/determinant (exact; keeps MPS/Metal happy)
    a, b = cov[:, 0, 0], cov[:, 0, 1]
    c, dd = cov[:, 1, 0], cov[:, 1, 1]
    det = (a * dd - b * c).clamp_min(1e-30)
    dx = x[:, None] - field.positions()[None, :, 0]
    dy = y[:, None] - field.positions()[None, :, 1]
    q = (dd[None, :] * dx * dx - (b + c)[None, :] * dx * dy + a[None, :] * dy * dy) / det[None, :]
    norm = 1.0 / (2 * math.pi * torch.sqrt(det))
    G = norm[None, :] * torch.exp(-0.5 * q)                          # (n, J)
    return (G * amps[None, :]).sum(-1)
