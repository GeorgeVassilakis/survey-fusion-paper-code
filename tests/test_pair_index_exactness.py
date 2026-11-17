"""The tiled point index must retain exactly the pairs the operator defines.

The tile stage only proposes candidates; the exact Mahalanobis prune decides
the retained set, and the stable per-tile ordering makes the retained pairs
come out ascending in (point, splat) whatever tile width or key width produced
them. That makes a dense sweep over every (point, splat) an implementation-
independent ground truth, both for the pairs and for their order, so this file
gates the index against arithmetic rather than against a previous version of
itself.
"""

from __future__ import annotations

import math

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.splat_raster import (
    PackedPointIndex,
    build_packed_point_index,
    build_pair_index,
    packed_multiband_point_prediction,
)

N_SIGMA = 8.0


def _field(
    *,
    n_splats: int,
    seed: int,
    scale: float,
    anisotropy: float,
    clumps: int,
) -> SplatField:
    torch.manual_seed(seed)
    field = SplatField(
        n_splats=n_splats,
        n_templates=3,
        loglam_grid=torch.linspace(-1.0, 1.0, 16),
        init_scale=scale,
    )
    with torch.no_grad():
        field.mu.uniform_(-1.0, 1.0)
        if clumps:
            centres = torch.rand(clumps, 2) * 1.6 - 0.8
            owner = torch.randint(0, clumps, (n_splats // 2,))
            field.mu[: n_splats // 2] = centres[owner] + 0.01 * torch.randn(
                n_splats // 2, 2
            )
        field.log_scales[:, 0].uniform_(
            math.log(scale), math.log(scale * anisotropy)
        )
        field.log_scales[:, 1].uniform_(
            math.log(scale / anisotropy), math.log(scale)
        )
        field.theta.uniform_(0.0, math.pi)
        field.raw_flux.normal_(-3.0, 1.0)
    return field


def _dense_retained_pairs(field, x, y, psf_sigma, n_sigma):
    """Every (point, splat) pair inside the operator's ellipse, row-major.

    Written with the same float32 operations the prune uses, so the boolean it
    produces per pair is the same bit; `nonzero` on the (n_points, n_splats)
    mask emits them ascending in (point, splat), which is the order the tiled
    index claims to produce.
    """
    a, b, d = field.covariance_components()
    if torch.is_tensor(psf_sigma) and psf_sigma.numel() > 1:
        variance = psf_sigma.reshape(-1).square()[:, None]
    else:
        variance = float(psf_sigma) ** 2
    a_pair = a[None, :] + variance
    d_pair = d[None, :] + variance
    determinant = (a_pair * d_pair - b[None, :].square()).clamp_min(1e-30)
    dx = x[:, None] - field.mu[None, :, 0]
    dy = y[:, None] - field.mu[None, :, 1]
    q = (
        d_pair * dx.square() - (2.0 * b[None, :]) * dx * dy + a_pair * dy.square()
    ) / determinant
    keep = torch.nonzero(q <= float(n_sigma) ** 2)
    return keep[:, 0].contiguous(), keep[:, 1].contiguous()


# n_splats, scale, anisotropy, clumps, per-point sigma
CONFIGS = [
    pytest.param(3000, 0.010, 1.0, 0, False, id="isotropic"),
    pytest.param(3000, 0.010, 6.0, 0, False, id="anisotropic"),
    pytest.param(3000, 0.004, 20.0, 0, True, id="very-anisotropic-per-point-sigma"),
    pytest.param(5000, 0.006, 3.0, 5, True, id="clustered"),
    pytest.param(5000, 0.020, 1.5, 3, False, id="wide-and-clustered"),
    pytest.param(800, 0.002, 8.0, 0, True, id="sparse-tiny"),
]


@pytest.mark.parametrize(
    "n_splats,scale,anisotropy,clumps,per_point_sigma", CONFIGS
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_retained_pairs_match_a_dense_sweep(
    n_splats, scale, anisotropy, clumps, per_point_sigma, seed
):
    field = _field(
        n_splats=n_splats,
        seed=seed,
        scale=scale,
        anisotropy=anisotropy,
        clumps=clumps,
    )
    generator = torch.Generator().manual_seed(seed + 100)
    n_points = 500
    x = torch.rand(n_points, generator=generator) * 2.0 - 1.0
    y = torch.rand(n_points, generator=generator) * 2.0 - 1.0
    if per_point_sigma:
        psf_sigma = 5.0e-4 + 5.0e-4 * torch.rand(n_points, generator=generator)
    else:
        psf_sigma = 8.0e-4

    index = build_packed_point_index(field, x, y, psf_sigma, n_sigma=N_SIGMA)
    expected_point, expected_splat = _dense_retained_pairs(
        field, x, y, psf_sigma, N_SIGMA
    )
    pair_splat = index.active_splat[index.pair_active_splat]

    assert torch.equal(index.pair_point, expected_point)
    assert torch.equal(pair_splat, expected_splat)
    assert torch.equal(index.active_splat, torch.unique(expected_splat))
    assert index.retained_pairs == int(expected_point.numel())
    assert index.candidate_pairs >= index.retained_pairs


def test_prediction_is_identical_to_one_built_from_the_dense_pairs():
    """What the raster consumes, not just the pair arrays, has to match."""
    field = _field(n_splats=4000, seed=5, scale=0.008, anisotropy=4.0, clumps=3)
    generator = torch.Generator().manual_seed(9)
    n_points = 400
    x = torch.rand(n_points, generator=generator) * 2.0 - 1.0
    y = torch.rand(n_points, generator=generator) * 2.0 - 1.0
    psf_sigma = 5.0e-4 + 5.0e-4 * torch.rand(n_points, generator=generator)
    point_band = torch.randint(0, 5, (n_points,), generator=generator)

    index = build_packed_point_index(field, x, y, psf_sigma, n_sigma=N_SIGMA)

    expected_point, expected_splat = _dense_retained_pairs(
        field, x, y, psf_sigma, N_SIGMA
    )
    active_splat, pair_active_splat = torch.unique(
        expected_splat, sorted=True, return_inverse=True
    )
    reference_index = PackedPointIndex(
        expected_point, active_splat, pair_active_splat,
        index.candidate_pairs, int(expected_point.numel()),
    )

    torch.manual_seed(13)
    amplitudes = torch.rand(active_splat.numel(), 5)
    actual = packed_multiband_point_prediction(
        field, x, y, psf_sigma, point_band, amplitudes, index
    )
    expected = packed_multiband_point_prediction(
        field, x, y, psf_sigma, point_band, amplitudes, reference_index
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "counts", [[0, 0, 0], [3, 0, 1, 0, 2], [1], [0, 4], [5, 5, 5]]
)
def test_slot_to_point_by_binary_search_is_repeat_interleave(counts):
    """The identity that let the candidate assembly drop repeat_interleave.

    Zero-length runs are the case that matters: a point whose tile is empty
    must be skipped, not given a slot.
    """
    counts_t = torch.tensor(counts, dtype=torch.long)
    ends = counts_t.cumsum(0)
    expected = torch.repeat_interleave(torch.arange(counts_t.numel()), counts_t)
    actual = torch.searchsorted(ends, torch.arange(int(ends[-1])), right=True)
    assert torch.equal(actual, expected)


def test_candidate_pairs_are_grouped_by_point_in_ascending_order():
    """Every candidate slot must sit inside its own point's contiguous run."""
    field = _field(n_splats=4000, seed=11, scale=0.010, anisotropy=3.0, clumps=2)
    generator = torch.Generator().manual_seed(31)
    x = torch.rand(400, generator=generator) * 2.0 - 1.0
    y = torch.rand(400, generator=generator) * 2.0 - 1.0
    radius = torch.full((4000,), 0.02)

    pair_point, pair_splat = build_pair_index(field.mu.detach(), radius, x, y)

    assert pair_point.numel() == pair_splat.numel()
    assert pair_point.numel() > 0
    assert bool(torch.all(pair_point[1:] >= pair_point[:-1]))
    assert int(pair_point.min()) >= 0
    assert int(pair_point.max()) < x.numel()


def test_pairs_are_ascending_in_point_then_splat():
    """The ordering the bitwise-neutrality of tile choices rests on."""
    field = _field(n_splats=6000, seed=8, scale=0.012, anisotropy=5.0, clumps=4)
    generator = torch.Generator().manual_seed(21)
    x = torch.rand(300, generator=generator) * 2.0 - 1.0
    y = torch.rand(300, generator=generator) * 2.0 - 1.0
    index = build_packed_point_index(field, x, y, 8.0e-4, n_sigma=N_SIGMA)
    pair_splat = index.active_splat[index.pair_active_splat]
    key = index.pair_point * (int(field.mu.shape[0]) + 1) + pair_splat
    assert bool(torch.all(key[1:] > key[:-1]))
