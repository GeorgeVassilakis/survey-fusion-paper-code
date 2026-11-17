"""Raw-bit checks of the generated VJP and its dispatch boundaries."""

import pytest
import torch

from survey_fusion.pgm.splat_spectrum import paired_positive_hat_bin_average

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _inputs(rows, knots=96):
    device = "cuda"
    r = torch.arange(rows, dtype=torch.float32, device=device)
    grid = torch.logspace(3.47712125472, 4.04139268516, knots, device=device)
    lam = (grid[None, :] * torch.exp(0.015 * torch.sin(r * 0.017))[:, None]).contiguous()
    values = 0.25 + torch.sin(r[:, None] * 0.013 + torch.arange(knots, device=device) * 0.11)
    values[0, 0] = -0.0
    center = 3000.0 + torch.remainder(r * 37.0, 8000.0)
    width = torch.where(r.remainder(3) == 0, 4.0, 180.0)
    lower, upper = (center - width / 2).contiguous(), (center + width / 2).contiguous()
    sigma = (0.7 + torch.remainder(r, 23)).contiguous()
    lower[:4] = torch.tensor([1000.0, 12000.0, 2800.0, 10999.0], device=device)
    upper[:4] = torch.tensor([2000.0, 13000.0, 2990.0, 11200.0], device=device)
    # Equality in the min/max boundaries must retain ATen's half-gradient rule.
    lower[4:8] = lam[4:8, 0]
    upper[8:12] = lam[8:12, -1]
    gen = torch.Generator(device=device).manual_seed(846)
    grad = torch.randn(rows, generator=gen, device=device)
    grad[::97] = 0.0
    grad[::101] = -0.0
    return (lam, values.contiguous(), lower, upper, sigma), grad


def _run(inputs, grad, blocks, mode, monkeypatch, all_grads=False):
    monkeypatch.setenv("SPLAT_FIBER_BINS", mode)
    monkeypatch.setenv("SPLAT_FIBER_BINS_BACKWARD", "fused")
    leaves = tuple(t.detach().requires_grad_(all_grads or i < 2) for i, t in enumerate(inputs))
    y = paired_positive_hat_bin_average(*leaves, lsf_mode="positive", off_grid="zero", blocks=blocks)
    gradients = torch.autograd.grad(y, leaves if all_grads else leaves[:2], grad)
    return y, *gradients


def _raw_equal(actual, expected):
    assert bool(torch.isfinite(actual).all())
    assert actual.dtype == expected.dtype
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int64
    count = int((actual.view(bits) != expected.view(bits)).sum())
    assert count == 0, f"{count}/{actual.numel()} elements differ in raw bits"


@pytest.mark.parametrize("rows", [1_000, 200_001])
@pytest.mark.parametrize("permuted", [False, True])
def test_full_fiber_vjp_raw_bits(monkeypatch, rows, permuted):
    inputs, grad = _inputs(rows)
    if permuted:
        order = torch.arange(rows - 1, -1, -1, device="cuda")
        inputs = tuple(t[order].contiguous() for t in inputs)
        grad = grad[order].contiguous()
    expected = _run(inputs, grad, [(0, rows)], "eager", monkeypatch)
    actual = _run(inputs, grad, [(0, rows)], "fused", monkeypatch)
    for a, b in zip(actual, expected):
        _raw_equal(a, b)


@pytest.mark.parametrize("case", ["multiple_blocks", "all_gradients", "other_knot_count"])
def test_fiber_vjp_ineligible_cases_use_eager_backward(monkeypatch, case):
    rows = 31
    inputs, grad = _inputs(rows, knots=7 if case == "other_knot_count" else 96)
    blocks = [(0, 17), (17, rows)] if case == "multiple_blocks" else [(0, rows)]
    expected = _run(inputs, grad, blocks, "eager", monkeypatch, all_grads=case == "all_gradients")
    actual = _run(inputs, grad, blocks, "fused", monkeypatch, all_grads=case == "all_gradients")
    for a, b in zip(actual, expected):
        _raw_equal(a, b)
