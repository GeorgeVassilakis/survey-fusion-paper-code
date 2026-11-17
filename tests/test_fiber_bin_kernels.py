import pytest
import torch

from survey_fusion.pgm.splat_spectrum import paired_positive_hat_bin_average


def _run(inputs, blocks, mode, monkeypatch, *, lsf_mode="positive"):
    monkeypatch.setenv("SPLAT_FIBER_BINS", mode)
    leaves = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
    output = paired_positive_hat_bin_average(
        *leaves,
        lsf_mode=lsf_mode,
        off_grid="zero",
        blocks=blocks,
    )
    grad_output = torch.linspace(
        -0.75, 1.25, output.numel(), dtype=output.dtype, device=output.device
    )
    gradients = torch.autograd.grad(output, leaves, grad_output)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _assert_raw_equal(actual, expected):
    assert actual.dtype == expected.dtype
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int64
    mismatch = actual.view(bits) != expected.view(bits)
    if bool(mismatch.any()):
        flat = torch.nonzero(mismatch.flatten())[:10].flatten()
        detail = [
            (
                int(index),
                actual.flatten()[index].item(),
                expected.flatten()[index].item(),
                int(actual.view(bits).flatten()[index]),
                int(expected.view(bits).flatten()[index]),
            )
            for index in flat
        ]
        pytest.fail(f"{int(mismatch.sum())} raw-bit mismatches: {detail}")
    assert torch.signbit(actual).equal(torch.signbit(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("unsorted", [False, True])
def test_fused_fiber_bins_match_eager_raw_bits(monkeypatch, unsorted):
    rows = 1_000_000
    knots = 96
    device = torch.device("cuda")
    base = torch.logspace(
        torch.log10(torch.tensor(3000.0)),
        torch.log10(torch.tensor(11000.0)),
        knots,
        dtype=torch.float32,
        device=device,
    )
    row = torch.arange(rows, dtype=torch.float32, device=device)
    shifts = torch.exp(0.015 * torch.sin(row * 0.017))
    knot_lam = (base[None, :] * shifts[:, None]).contiguous()
    knot_values = (
        0.25
        + torch.sin(row[:, None] * 0.013 + torch.arange(knots, device=device) * 0.11)
    ).contiguous()
    knot_values[0, 0] = -0.0
    centers = 3000.0 + torch.remainder(row * 37.0, 8000.0)
    widths = torch.where(row.remainder(3) == 0, 4.0, 180.0)
    lower = (centers - 0.5 * widths).contiguous()
    upper = (centers + 0.5 * widths).contiguous()
    lower[:4] = torch.tensor([1000.0, 12000.0, 2800.0, 10999.0], device=device)
    upper[:4] = torch.tensor([2000.0, 13000.0, 2990.0, 11200.0], device=device)
    sigma = (0.7 + torch.remainder(row, 23.0)).contiguous()

    if unsorted:
        order = torch.remainder(torch.arange(rows, device=device) * 607, rows).long()
        inputs = tuple(
            value[order].contiguous()
            for value in (knot_lam, knot_values, lower, upper, sigma)
        )
    else:
        inputs = (knot_lam, knot_values, lower, upper, sigma)
    blocks = [(0, 17), (17, rows)] if unsorted else [(0, rows)]
    eager = _run(inputs, blocks, "eager", monkeypatch)
    fused = _run(inputs, blocks, "fused", monkeypatch)
    _assert_raw_equal(fused[0], eager[0])
    for actual, expected in zip(fused[1], eager[1]):
        _assert_raw_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_fiber_bins_zero_lsf_uses_exact_eager_fallback(monkeypatch):
    device = torch.device("cuda")
    knot_lam = torch.tensor(
        [[3000.0, 5000.0, 11000.0], [3100.0, 5100.0, 10900.0]],
        dtype=torch.float32,
        device=device,
    )
    knot_values = torch.tensor(
        [[-0.0, 2.0, 0.5], [1.0, 3.0, 1.5]],
        dtype=torch.float32,
        device=device,
    )
    lower = torch.tensor([3200.0, 4900.0], device=device)
    upper = torch.tensor([4800.0, 5300.0], device=device)
    sigma = torch.tensor([0.0, 12.0], device=device)
    inputs = (knot_lam, knot_values, lower, upper, sigma)
    eager = _run(inputs, [(0, 2)], "eager", monkeypatch, lsf_mode=None)
    fused = _run(inputs, [(0, 2)], "fused", monkeypatch, lsf_mode=None)
    _assert_raw_equal(fused[0], eager[0])
    for actual, expected in zip(fused[1], eager[1]):
        _assert_raw_equal(actual, expected)
