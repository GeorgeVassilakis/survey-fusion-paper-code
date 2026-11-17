import pytest
import torch

from survey_fusion.pgm.splat_spectrum import paired_positive_hat_bin_average


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int64
    assert torch.equal(actual.view(bits), expected.view(bits))
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def _inputs(rows):
    device = torch.device("cuda")
    knots = 96
    base = torch.logspace(
        torch.log10(torch.tensor(3000.0)),
        torch.log10(torch.tensor(11000.0)),
        knots,
        dtype=torch.float32,
        device=device,
    )
    row = torch.arange(rows, dtype=torch.float32, device=device)
    shift = torch.exp(0.01 * torch.sin(row * 0.019))
    knot_lam = (base[None] * shift[:, None]).contiguous()
    knot_values = (
        0.25
        + torch.sin(
            row[:, None] * 0.013 + torch.arange(knots, device=device) * 0.11
        )
    ).contiguous()
    knot_values[0, 0] = -0.0
    centers = 3000.0 + torch.remainder(row * 37.0, 8000.0)
    widths = torch.where(row.remainder(3) == 0, 4.0, 180.0)
    lower = (centers - widths * 0.5).contiguous()
    upper = (centers + widths * 0.5).contiguous()
    lower[:4] = torch.tensor([1000.0, 12000.0, 2800.0, 10999.0], device=device)
    upper[:4] = torch.tensor([2000.0, 13000.0, 2990.0, 11200.0], device=device)
    sigma = (0.7 + torch.remainder(row, 23.0)).contiguous()
    return knot_lam, knot_values, lower, upper, sigma


def _run(inputs, mode, monkeypatch):
    monkeypatch.setenv("SPLAT_FIBER_BINS", mode)
    knot_lam = inputs[0].detach().clone().requires_grad_()
    knot_values = inputs[1].detach().clone().requires_grad_()
    fixed = tuple(value.detach().clone() for value in inputs[2:])
    result = paired_positive_hat_bin_average(
        knot_lam,
        knot_values,
        *fixed,
        lsf_mode="positive",
        off_grid="zero",
        blocks=((0, knot_lam.shape[0]),),
    )
    grad_result = torch.linspace(
        -0.75, 1.25, result.numel(), dtype=result.dtype, device=result.device
    )
    gradients = torch.autograd.grad(result, (knot_lam, knot_values), grad_result)
    return result.detach(), tuple(gradient.detach() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graphed_fiber_bins_match_eager_raw_bits_across_replays(monkeypatch):
    for rows in (70_000, 131_072, 200_001, 200_001):
        inputs = _inputs(rows)
        eager = _run(inputs, "eager", monkeypatch)
        graphed = _run(inputs, "graphed", monkeypatch)
        _assert_bits(graphed[0], eager[0])
        _assert_bits(graphed[1][0], eager[1][0])
        _assert_bits(graphed[1][1], eager[1][1])
