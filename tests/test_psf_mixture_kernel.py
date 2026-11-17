from __future__ import annotations

import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.analytic_image import analytic_band_prediction
from survey_fusion.pgm.contracts import PSFMixtureComponents
from survey_fusion.pgm.splat_raster import (
    PointPSFMixture, band_amplitudes, build_mixture_packed_point_index,
    packed_mixture_multiband_point_prediction, tiled_mixture_point_prediction,
)
from survey_fusion.training.renderer import gaussian_mixture_psf_kernel


def test_gaussian_mixture_kernel_is_one_normalized_widest_component_grid() -> None:
    mixture = PSFMixtureComponents.from_rows([
        [0.7, 1.0, 0.0, 1.0],
        [0.2, 4.0, 0.4, 3.0],
        [0.1, 16.0, 0.0, 16.0],
    ])
    kernel = gaussian_mixture_psf_kernel(
        mixture, supersample=2, device=torch.device("cpu"), truncate=4.0,
    )
    assert kernel.shape == (1, 1, 65, 65)
    torch.testing.assert_close(kernel.sum(), torch.tensor(1.0), rtol=0.0, atol=1e-7)
    assert kernel[0, 0, 32, 32] == kernel.max()


def test_gaussian_mixture_kernel_flux_is_supersample_invariant() -> None:
    mixture = PSFMixtureComponents.from_rows([
        [0.6, 1.0, 0.1, 1.2], [0.3, 4.0, -0.3, 5.0], [0.1, 9.0, 0.0, 9.0],
    ])
    for supersample in (1, 2, 3):
        kernel = gaussian_mixture_psf_kernel(
            mixture, supersample=supersample, device=torch.device("cpu"),
        )
        torch.testing.assert_close(kernel.sum(), torch.tensor(1.0), rtol=0.0, atol=2e-7)


def test_conv_fallback_matches_analytic_tiled_and_packed_oracle_with_discrete_tolerance() -> None:
    components = PSFMixtureComponents.from_rows([
        [0.6, 1.0, 0.1, 1.2], [0.3, 4.0, -0.3, 5.0], [0.1, 9.0, 0.0, 9.0],
    ])
    field = SplatField(
        n_splats=1, n_templates=2, loglam_grid=torch.linspace(-1, 1, 16), init_scale=1e-4,
    ).double()
    with torch.no_grad():
        field.mu.zero_()
        field.log_scales.fill_(torch.log(torch.tensor(1e-4, dtype=torch.float64)))
        field.raw_flux.fill_(0.2)
    kernel = gaussian_mixture_psf_kernel(
        components, supersample=1, device=torch.device("cpu"), dtype=torch.float64,
    )[0, 0]
    radius = kernel.shape[0] // 2
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    x, y = xx.reshape(-1), yy.reshape(-1)
    rows = torch.tensor(components.components, dtype=torch.float64)
    covariance = rows[:, 1:].expand(x.numel(), -1, -1)
    mixture = PointPSFMixture(rows[:, 0].expand(x.numel(), -1), covariance)
    bandpass = torch.linspace(-0.2, 0.1, 7, dtype=torch.float64)
    amplitudes = band_amplitudes(field, bandpass)
    analytic = analytic_band_prediction(
        field, x, y, 0.0, bandpass, psf_mixture=mixture,
    )
    tiled = tiled_mixture_point_prediction(
        field, x, y, mixture, amplitudes, checkpoint_chunks=False,
    )
    index = build_mixture_packed_point_index(field, x, y, mixture)
    packed = packed_mixture_multiband_point_prediction(
        field, x, y, mixture, torch.zeros_like(x, dtype=torch.long),
        amplitudes[index.active_splat, None], index, checkpoint_chunks=False,
    )
    normalized = [value.reshape_as(kernel) / value.sum() for value in (analytic, tiled, packed)]
    for value in normalized:
        torch.testing.assert_close(value, kernel, rtol=2e-6, atol=2e-9)
