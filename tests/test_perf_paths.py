"""Regressions for the hot-path performance work: overflow-safe gradient
norms, sorted-unique sparse gathers, sync-free LSF modes, and the
direct-address active-splat map."""

import math
from types import SimpleNamespace

import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.splat_raster import build_packed_point_index
from survey_fusion.pgm.splat_spectrum import piecewise_linear_bin_average
from survey_fusion.training.datum_trainer import DatumTrainer


def test_gradient_norm_survives_large_finite_per_param_norms() -> None:
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    p1.grad = torch.full((1,), 1.5e19)
    p2.grad = torch.full((1,), 1.5e19)
    holder = SimpleNamespace(params=[p1, p2], _clip_param_ids={id(p1), id(p2)})
    total = float(DatumTrainer._gradient_norm(holder))
    assert math.isfinite(total)
    expected = (2.0 * (1.5e19) ** 2) ** 0.5
    assert abs(total - expected) / expected < 1e-6


def _small_field() -> SplatField:
    torch.manual_seed(0)
    field = SplatField(
        n_splats=12, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8)
    )
    field.set_sparse_gradients(True)
    return field


def test_sorted_unique_gather_matches_generic_gather_gradients() -> None:
    indices = torch.tensor([1, 4, 7, 9])
    upstream = torch.randn(indices.numel(), 3)

    field_a = _small_field()
    out_a = field_a.sed_coefficients(indices)
    (grad_a,) = torch.autograd.grad(
        (out_a * upstream).sum(), field_a.sed_coeffs
    )
    grad_a = grad_a.coalesce()

    field_b = _small_field()
    out_b = field_b.sed_coefficients(indices, sorted_unique=True)
    (grad_b,) = torch.autograd.grad(
        (out_b * upstream).sum(), field_b.sed_coeffs
    )

    assert torch.equal(out_a, out_b)
    # The flagged path must emit an already-coalesced gradient (that is the
    # skipped sort) with values identical to the generic path.
    assert grad_b.is_coalesced()
    assert torch.equal(grad_a.indices(), grad_b.indices())
    assert torch.equal(grad_a.values(), grad_b.values())


def test_lsf_mode_matches_masked_branches() -> None:
    torch.manual_seed(1)
    knot_lam = torch.sort(torch.rand(5, 9) * 100 + 4000, dim=1).values
    knot_values = torch.rand(5, 9)
    lower = torch.tensor([4010.0, 4030.0, 4050.0])
    upper = lower + 7.0

    positive = torch.tensor([0.9, 1.7, 0.4])
    reference = piecewise_linear_bin_average(
        knot_lam, knot_values, lower, upper, positive
    )
    fast = piecewise_linear_bin_average(
        knot_lam, knot_values, lower, upper, positive,
        validate=False, lsf_mode="positive",
    )
    assert torch.equal(reference, fast)

    zero = torch.zeros(3)
    reference = piecewise_linear_bin_average(
        knot_lam, knot_values, lower, upper, zero
    )
    fast = piecewise_linear_bin_average(
        knot_lam, knot_values, lower, upper, zero,
        validate=False, lsf_mode="zero",
    )
    assert torch.equal(reference, fast)


def test_packed_point_index_matches_torch_unique() -> None:
    torch.manual_seed(2)
    field = _small_field()
    x = torch.rand(40) * 2 - 1
    y = torch.rand(40) * 2 - 1
    index = build_packed_point_index(field, x, y, psf_sigma=0.05)
    assert index.pair_point.numel() > 0

    pair_splat = index.active_splat[index.pair_active_splat]
    expected_active, expected_inverse = torch.unique(
        pair_splat, sorted=True, return_inverse=True
    )
    assert torch.equal(index.active_splat, expected_active)
    assert torch.equal(index.pair_active_splat, expected_inverse)
