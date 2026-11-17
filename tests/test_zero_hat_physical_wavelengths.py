"""Independent analytic controls at physical SDSS wavelengths.

A constant/linear spectrum remains constant/linear after Gaussian convolution
and averaging over a centered bin. The tested bins are hundreds of LSF sigmas
from the domain edges, so zero-extension tails are negligible at this precision.
"""

import pytest
import torch

from survey_fusion.pgm.splat_spectrum import (
    paired_positive_hat_bin_average,
    positive_hat_bin_average,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device(request.param)


def _render(knots, values, centers, *, paired, off_grid):
    lower, upper = centers - 0.5, centers + 0.5
    sigma = torch.ones_like(centers)
    if paired:
        count = len(centers)
        return paired_positive_hat_bin_average(
            knots.expand(count, -1).clone(),
            values.expand(count, -1),
            lower,
            upper,
            sigma,
            lsf_mode="positive",
            off_grid=off_grid,
            blocks=[(0, 1), (1, count)],
        )
    return positive_hat_bin_average(
        knots, values, lower, upper, sigma, lsf_mode="positive", off_grid=off_grid
    ).flatten()


def _physical_knots(device):
    # Use the same float32-representable grid on both devices.
    return (
        torch.logspace(
            torch.log10(torch.tensor(3000.0)),
            torch.log10(torch.tensor(10000.0)),
            96,
        )
        .reshape(1, -1)
        .to(device)
    )


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("off_grid", ["zero", "clamp"])
def test_constant_spectrum_at_physical_wavelengths(device, paired, off_grid):
    knots = _physical_knots(device)
    centers = torch.tensor([4200.0, 6000.0, 8500.0, 10500.0], device=device)
    amplitude = torch.tensor(2.5, device=device, requires_grad=True)
    values = torch.ones_like(knots) * amplitude
    predicted = _render(knots, values, centers, paired=paired, off_grid=off_grid)
    expected = torch.tensor([2.5, 2.5, 2.5, 0.0 if off_grid == "zero" else 2.5], device=device)
    assert predicted.dtype == values.dtype
    torch.testing.assert_close(predicted, expected, atol=3e-6, rtol=3e-6)

    predicted.sum().backward()
    expected_gradient = torch.tensor(3.0 if off_grid == "zero" else 4.0, device=device)
    torch.testing.assert_close(amplitude.grad, expected_gradient, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("paired", [False, True])
def test_linear_spectrum_and_gradient_at_physical_wavelengths(device, paired):
    knots = _physical_knots(device)
    centers = torch.tensor([4200.0, 6000.0, 8500.0], device=device)
    intercept = torch.tensor(0.25, device=device, requires_grad=True)
    slope = torch.tensor(1e-4, device=device, requires_grad=True)
    values = intercept + slope * knots
    predicted = _render(knots, values, centers, paired=paired, off_grid="zero")
    expected = 0.25 + 1e-4 * centers
    assert predicted.dtype == values.dtype
    torch.testing.assert_close(predicted, expected, atol=3e-6, rtol=3e-6)

    predicted.sum().backward()
    torch.testing.assert_close(intercept.grad, torch.tensor(3.0, device=device), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(slope.grad, centers.sum(), atol=0.01, rtol=1e-6)
