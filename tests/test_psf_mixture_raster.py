from __future__ import annotations

import copy

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.analytic_image import analytic_band_prediction
from survey_fusion.pgm.splat_raster import (
    PointPSFMixture,
    PackedPointIndex,
    band_amplitudes,
    build_pair_index,
    build_mixture_packed_point_index,
    packed_mixture_multiband_point_prediction,
    tiled_mixture_point_prediction,
)


def _global_mixture_index(
    field: SplatField,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    n_sigma: float,
) -> PackedPointIndex:
    if mixture.isotropic:
        psf_variance_max = mixture.covariances.max()
    else:
        matrices = torch.zeros(
            (*mixture.covariances.shape[:2], 2, 2), dtype=x.dtype
        )
        matrices[..., 0, 0] = mixture.covariances[..., 0]
        matrices[..., 0, 1] = mixture.covariances[..., 1]
        matrices[..., 1, 0] = mixture.covariances[..., 1]
        matrices[..., 1, 1] = mixture.covariances[..., 2]
        psf_variance_max = torch.linalg.eigvalsh(matrices).amax()
    splat_variance_max = torch.exp(2.0 * field.log_scales.detach().amax(dim=1))
    radius = float(n_sigma) * torch.sqrt(splat_variance_max + psf_variance_max)
    pair_point, pair_splat = build_pair_index(field.mu, radius, x, y)
    active_splat, pair_active_splat = torch.unique(
        pair_splat, sorted=True, return_inverse=True
    )
    return PackedPointIndex(pair_point, active_splat, pair_active_splat)


def _retained_component_pairs(
    field: SplatField,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    index: PackedPointIndex,
    n_sigma: float,
) -> set[tuple[int, int, int]]:
    pair_splat = index.active_splat[index.pair_active_splat]
    splat_covariance = field.covariances().detach()
    retained = set()
    for pair_index in range(index.pair_point.numel()):
        point = int(index.pair_point[pair_index])
        splat = int(pair_splat[pair_index])
        delta = torch.stack((x[point], y[point])) - field.mu.detach()[splat]
        for component in range(mixture.n_components):
            if mixture.isotropic:
                psf_covariance = mixture.covariances[point, component] * torch.eye(
                    2, dtype=x.dtype
                )
            else:
                vxx, vxy, vyy = mixture.covariances[point, component]
                psf_covariance = torch.stack((vxx, vxy, vxy, vyy)).reshape(2, 2)
            covariance = splat_covariance[splat] + psf_covariance
            q = delta @ torch.linalg.solve(covariance, delta)
            if float(q) <= n_sigma**2:
                retained.add((point, splat, component))
    return retained


def _brute_force_retained_component_pairs(
    field: SplatField,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    n_sigma: float,
) -> set[tuple[int, int, int]]:
    n_points = x.numel()
    n_splats = field.mu.shape[0]
    pair_point = torch.arange(n_points).repeat_interleave(n_splats)
    pair_splat = torch.arange(n_splats).repeat(n_points)
    return _retained_component_pairs(
        field,
        x,
        y,
        mixture,
        PackedPointIndex(pair_point, torch.arange(n_splats), pair_splat),
        n_sigma,
    )


def _field(dtype=torch.float64) -> SplatField:
    torch.manual_seed(17)
    field = SplatField(
        n_splats=4, n_templates=2, loglam_grid=torch.linspace(-1, 1, 24),
        init_scale=0.045,
    ).to(dtype=dtype)
    with torch.no_grad():
        field.mu.copy_(torch.tensor([
            [-0.19, -0.12], [0.08, 0.11], [0.27, -0.03], [-0.05, 0.31],
        ], dtype=dtype))
        field.log_scales.copy_(torch.log(torch.tensor([
            [0.035, 0.052], [0.06, 0.025], [0.04, 0.07], [0.05, 0.03],
        ], dtype=dtype)))
        field.theta.copy_(torch.tensor([0.2, -0.6, 0.8, -0.1], dtype=dtype))
        field.raw_flux.copy_(torch.tensor([0.1, 0.5, -0.2, 0.35], dtype=dtype))
    return field


def _mixture(x: torch.Tensor, k: int, anisotropic: bool) -> PointPSFMixture:
    base_weights = torch.tensor([0.55, 0.30, 0.15], dtype=x.dtype, device=x.device)[:k]
    base_weights = base_weights / base_weights.sum()
    weights = base_weights.expand(x.numel(), -1).clone()
    variances = torch.tensor([0.0002, 0.0011, 0.0035], dtype=x.dtype, device=x.device)[:k]
    if not anisotropic:
        return PointPSFMixture(weights, variances.expand(x.numel(), -1).clone())
    covariance = torch.stack((
        variances * 0.7,
        torch.tensor([0.00003, -0.00018, 0.00045], dtype=x.dtype, device=x.device)[:k],
        variances * 1.4,
    ), dim=-1)
    return PointPSFMixture(weights, covariance.expand(x.numel(), -1, -1).clone())


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_dense_packed_and_tiled_mixture_parity(k: int, anisotropic: bool) -> None:
    field = _field()
    x = torch.tensor([-0.25, -0.10, 0.02, 0.16, 0.31], dtype=torch.float64)
    y = torch.tensor([-0.10, 0.02, 0.15, -0.07, 0.24], dtype=torch.float64)
    mixture = _mixture(x, k, anisotropic)
    bandpass = torch.linspace(-0.2, 0.15, 11, dtype=torch.float64)
    amplitudes = band_amplitudes(field, bandpass)
    dense = analytic_band_prediction(
        field, x, y, 0.0, bandpass, psf_mixture=mixture,
    )
    tiled = tiled_mixture_point_prediction(
        field, x, y, mixture, amplitudes, n_sigma=8.0, checkpoint_chunks=False,
    )
    index = build_mixture_packed_point_index(field, x, y, mixture, n_sigma=8.0)
    packed = packed_mixture_multiband_point_prediction(
        field, x, y, mixture, torch.zeros_like(x, dtype=torch.long),
        amplitudes[index.active_splat, None], index, n_sigma=8.0,
        checkpoint_chunks=False,
    )
    torch.testing.assert_close(tiled, dense, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(packed, dense, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("anisotropic", [False, True])
def test_mixture_tiled_fp64_gradcheck(anisotropic: bool) -> None:
    field = _field()
    amplitudes = band_amplitudes(field, torch.linspace(-0.2, 0.15, 7, dtype=torch.float64)).detach()
    x = torch.tensor([-0.12, 0.04, 0.22], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.01, 0.14, -0.04], dtype=torch.float64, requires_grad=True)

    def render(x_value, y_value):
        mixture = _mixture(x_value, 3, anisotropic)
        return tiled_mixture_point_prediction(
            field, x_value, y_value, mixture, amplitudes,
            n_sigma=8.0, checkpoint_chunks=False,
        )

    assert torch.autograd.gradcheck(render, (x, y), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_low_weight_wide_component_controls_conservative_index() -> None:
    field = _field()
    x = torch.tensor([0.55], dtype=torch.float64)
    y = torch.tensor([-0.03], dtype=torch.float64)
    weights = torch.tensor([[0.999, 0.001]], dtype=torch.float64)
    mixture = PointPSFMixture(weights, torch.tensor([[1e-5, 0.04]], dtype=torch.float64))
    index = build_mixture_packed_point_index(field, x, y, mixture, n_sigma=4.0)
    assert index.pair_point.numel() > 0
    amplitudes = band_amplitudes(field, torch.linspace(-0.2, 0.1, 5, dtype=torch.float64))
    full = packed_mixture_multiband_point_prediction(
        field, x, y, mixture, torch.zeros(1, dtype=torch.long),
        amplitudes[index.active_splat, None], index, n_sigma=4.0,
        checkpoint_chunks=False,
    )
    narrow_only = PointPSFMixture(torch.ones((1, 1), dtype=torch.float64), mixture.covariances[:, :1])
    narrow_index = build_mixture_packed_point_index(field, x, y, narrow_only, n_sigma=4.0)
    narrow = packed_mixture_multiband_point_prediction(
        field, x, y, narrow_only, torch.zeros(1, dtype=torch.long),
        amplitudes[narrow_index.active_splat, None], narrow_index, n_sigma=4.0,
        checkpoint_chunks=False,
    )
    assert not torch.equal(full, narrow)


def test_grouped_index_keeps_narrow_exposure_independent_of_broad_exposure() -> None:
    field = SplatField(
        n_splats=2, n_templates=2, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.01,
    ).double()
    with torch.no_grad():
        field.mu.copy_(torch.tensor([[0.01, 0.0], [0.20, 0.0]], dtype=torch.float64))
        field.log_scales.fill_(torch.log(torch.tensor(0.01, dtype=torch.float64)))
    narrow = PointPSFMixture(
        torch.ones((1, 1), dtype=torch.float64),
        torch.full((1, 1), 1e-4, dtype=torch.float64),
    )
    alone = build_mixture_packed_point_index(
        field, torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64), narrow, n_sigma=4.0,
        point_group=torch.tensor([0]),
    )
    mixed = PointPSFMixture(
        torch.ones((2, 1), dtype=torch.float64),
        torch.tensor([[1e-4], [14.5**2]], dtype=torch.float64),
    )
    batched = build_mixture_packed_point_index(
        field, torch.tensor([0.0, 100.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64), mixed, n_sigma=4.0,
        point_group=torch.tensor([0, 1]),
    )

    alone_splats = alone.active_splat[alone.pair_active_splat]
    batched_splats = batched.active_splat[batched.pair_active_splat]
    torch.testing.assert_close(
        batched_splats[batched.pair_point == 0], alone_splats, rtol=0.0, atol=0.0
    )


def test_grouped_index_post_prune_matches_global_and_brute_force() -> None:
    torch.manual_seed(314)
    field = SplatField(
        n_splats=17, n_templates=2, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.03,
    ).double()
    with torch.no_grad():
        field.mu.uniform_(-0.8, 0.8)
        field.log_scales.uniform_(-4.2, -2.5)
        field.theta.uniform_(-1.0, 1.0)
    x = torch.empty(12, dtype=torch.float64).uniform_(-0.9, 0.9)
    y = torch.empty(12, dtype=torch.float64).uniform_(-0.9, 0.9)
    weights = torch.tensor([0.75, 0.20, 0.05], dtype=torch.float64).expand(12, -1)
    covariance = torch.empty((12, 3, 3), dtype=torch.float64)
    covariance[:, 0] = torch.tensor([4e-4, 5e-5, 8e-4])
    covariance[:, 1] = torch.tensor([4e-3, -3e-4, 7e-3])
    covariance[:, 2] = torch.tensor([3e-2, 2e-3, 5e-2])
    covariance[6:, 2] *= 8.0
    mixture = PointPSFMixture(weights, covariance)
    point_group = torch.tensor([0] * 6 + [1] * 6)
    n_sigma = 4.0

    grouped = build_mixture_packed_point_index(
        field, x, y, mixture, n_sigma=n_sigma, point_group=point_group
    )
    global_index = _global_mixture_index(field, x, y, mixture, n_sigma)
    grouped_retained = _retained_component_pairs(
        field, x, y, mixture, grouped, n_sigma
    )
    global_retained = _retained_component_pairs(
        field, x, y, mixture, global_index, n_sigma
    )
    brute_force = _brute_force_retained_component_pairs(
        field, x, y, mixture, n_sigma
    )

    assert grouped_retained == global_retained == brute_force


def test_grouped_index_matches_global_forward_and_gradients_deterministically() -> None:
    torch.manual_seed(818)
    grouped_field = _field()
    global_field = copy.deepcopy(grouped_field)
    x = torch.tensor([-0.24, -0.08, 0.04, 0.18, 0.32, 0.47], dtype=torch.float64)
    y = torch.tensor([-0.12, 0.03, 0.16, -0.06, 0.22, -0.18], dtype=torch.float64)
    weights = torch.tensor([0.65, 0.25, 0.10], dtype=torch.float64).expand(6, -1)
    variances = torch.tensor([
        [2e-4, 8e-4, 2e-3], [2e-4, 8e-4, 2e-3],
        [3e-4, 1e-3, 3e-3], [3e-4, 1e-3, 3e-3],
        [1e-2, 4e-2, 1.5e-1], [1e-2, 4e-2, 1.5e-1],
    ], dtype=torch.float64)
    mixture = PointPSFMixture(weights, variances)
    point_group = torch.tensor([0, 0, 1, 1, 2, 2])
    point_band = torch.zeros(6, dtype=torch.long)
    bandpass = torch.linspace(-0.2, 0.15, 11, dtype=torch.float64)

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        grouped_index = build_mixture_packed_point_index(
            grouped_field, x, y, mixture, n_sigma=5.0, point_group=point_group
        )
        global_index = _global_mixture_index(global_field, x, y, mixture, 5.0)
        grouped_amplitudes = band_amplitudes(grouped_field, bandpass)
        global_amplitudes = band_amplitudes(global_field, bandpass)
        grouped_values = packed_mixture_multiband_point_prediction(
            grouped_field, x, y, mixture, point_band,
            grouped_amplitudes[grouped_index.active_splat, None], grouped_index,
            n_sigma=5.0, checkpoint_chunks=False,
        )
        global_values = packed_mixture_multiband_point_prediction(
            global_field, x, y, mixture, point_band,
            global_amplitudes[global_index.active_splat, None], global_index,
            n_sigma=5.0, checkpoint_chunks=False,
        )
        loss_weights = torch.arange(1, 7, dtype=torch.float64)
        grouped_values.mul(loss_weights).sum().backward()
        global_values.mul(loss_weights).sum().backward()
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    torch.testing.assert_close(grouped_values, global_values, rtol=2e-13, atol=2e-15)
    for grouped_parameter, global_parameter in zip(
        grouped_field.parameters(), global_field.parameters()
    ):
        assert grouped_parameter.grad is not None
        assert global_parameter.grad is not None
        grouped_gradient = grouped_parameter.grad
        global_gradient = global_parameter.grad
        if grouped_gradient.layout != torch.strided:
            grouped_gradient = grouped_gradient.to_dense()
        if global_gradient.layout != torch.strided:
            global_gradient = global_gradient.to_dense()
        torch.testing.assert_close(
            grouped_gradient, global_gradient, rtol=2e-12, atol=2e-14
        )


def test_grouped_index_narrow_pair_count_does_not_grow_when_batched() -> None:
    field = _field()
    x_narrow = torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float64)
    y_narrow = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    narrow = PointPSFMixture(
        torch.ones((3, 1), dtype=torch.float64),
        torch.full((3, 1), 2e-4, dtype=torch.float64),
    )
    alone = build_mixture_packed_point_index(
        field, x_narrow, y_narrow, narrow, n_sigma=4.0,
        point_group=torch.zeros(3, dtype=torch.long),
    )
    mixed = PointPSFMixture(
        torch.ones((4, 1), dtype=torch.float64),
        torch.tensor([[2e-4], [2e-4], [2e-4], [2.0]], dtype=torch.float64),
    )
    batched = build_mixture_packed_point_index(
        field, torch.cat((x_narrow, torch.tensor([100.0], dtype=torch.float64))),
        field.mu.new_zeros(4), mixed, n_sigma=4.0,
        point_group=torch.tensor([0, 0, 0, 1]),
    )

    assert int((batched.pair_point < 3).sum()) <= alone.pair_point.numel()


def test_isolated_splat_flux_is_conserved_by_valid_mixture() -> None:
    field = SplatField(
        n_splats=1, n_templates=2, loglam_grid=torch.linspace(-1, 1, 24),
        init_scale=0.012,
    ).double()
    with torch.no_grad():
        field.mu.zero_()
        field.log_scales.copy_(torch.log(torch.tensor([[0.012, 0.018]], dtype=torch.float64)))
        field.theta.fill_(0.3)
        field.raw_flux.fill_(0.2)
    coordinate = torch.linspace(-0.35, 0.35, 281, dtype=torch.float64)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    x, y = xx.reshape(-1), yy.reshape(-1)
    weights = torch.tensor([[0.65, 0.25, 0.10]], dtype=torch.float64).expand(x.numel(), -1)
    covariance = torch.tensor([
        [0.0001, 0.0, 0.0001],
        [0.0004, 0.00005, 0.0007],
        [0.0016, -0.0002, 0.0020],
    ], dtype=torch.float64).expand(x.numel(), -1, -1)
    mixture = PointPSFMixture(weights, covariance)
    amplitudes = band_amplitudes(field, torch.linspace(-0.2, 0.1, 9, dtype=torch.float64))
    rendered = tiled_mixture_point_prediction(
        field, x, y, mixture, amplitudes, n_sigma=8.0,
        pair_chunk=200_000, checkpoint_chunks=False,
    )
    pixel_area = float(coordinate[1] - coordinate[0]) ** 2
    assert float(rendered.detach().sum() * pixel_area) == pytest.approx(
        float(amplitudes[0].detach()), rel=3e-4
    )
