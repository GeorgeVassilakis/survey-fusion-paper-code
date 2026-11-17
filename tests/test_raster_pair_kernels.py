"""Raw-bit gates for the fused packed-raster pair evaluation."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("triton", reason="the fused raster pairs need Triton")

from survey_fusion.pgm.raster_pair_kernels import pair_values
from survey_fusion.utils.perf_env import (
    fused_raster_pairs_backward_enabled,
    fused_raster_pairs_enabled,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused raster pairs need CUDA"
)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def _reference(
    pp, active, rows, amplitudes, x, y, point_band, point_variance
):
    r = rows[active]
    variance = point_variance[pp] if torch.is_tensor(point_variance) else point_variance
    a_pair = r[:, 0] + variance
    d_pair = r[:, 1] + variance
    determinant = (a_pair * d_pair - r[:, 3]).clamp_min(1e-30)
    dx = x[pp] - r[:, 4]
    dy = y[pp] - r[:, 5]
    exponent = (
        d_pair * dx.square()
        - r[:, 2] * dx * dy
        + a_pair * dy.square()
    ) / determinant
    pair_amplitude = amplitudes[active, point_band[pp]]
    return (
        pair_amplitude
        / (2.0 * math.pi * determinant.sqrt())
        * torch.exp(-0.5 * exponent)
    )


@requires_cuda
@pytest.mark.parametrize("backward_path", ["autograd", "fused"])
@pytest.mark.parametrize("index_order", ["sorted_unique", "unsorted"])
def test_fused_raster_pairs_match_every_output_and_gradient_bit(
    monkeypatch, index_order, backward_path
) -> None:
    monkeypatch.setenv("SPLAT_RASTER_PAIRS_BACKWARD", backward_path)
    device = torch.device("cuda")
    n_pairs = 1_048_576
    n_points = 10_000
    n_active = n_pairs if index_order == "sorted_unique" else 300_007
    offsets = torch.arange(n_pairs, device=device)
    if index_order == "sorted_unique":
        active = offsets
        pp = offsets.remainder(n_points)
    else:
        active = (offsets * 104_729 + 17).remainder(n_active)
        pp = (offsets * 7_919 + 23).remainder(n_points)

    generator = torch.Generator(device=device).manual_seed(20260907)
    rows = torch.empty((n_active, 6), device=device).uniform_(
        -0.02, 0.02, generator=generator
    )
    rows[:, 0:2].uniform_(0.0005, 0.002, generator=generator)
    rows[:, 2].uniform_(-0.0004, 0.0004, generator=generator)
    rows[:, 3].uniform_(0.0, 1e-7, generator=generator)
    amplitudes = torch.empty((n_active, 5), device=device).uniform_(
        -2.0, 2.0, generator=generator
    )
    x = torch.empty(n_points, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    y = torch.empty(n_points, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    point_band = torch.arange(n_points, device=device).remainder(5)
    x[0] = -0.0
    y[1] = -0.0
    amplitudes[0, 0] = -0.0
    rows[0, 4] = -0.0
    rows[1, 5] = -0.0
    if index_order == "unsorted":
        point_variance = torch.empty(n_points, device=device).uniform_(
            1e-6, 1e-4, generator=generator
        )
    else:
        point_variance = 4e-5
    grad_output = torch.empty(n_pairs, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    grad_output[0] = -0.0

    eager_inputs = [
        rows.detach().clone().requires_grad_(),
        amplitudes.detach().clone().requires_grad_(),
        x.detach().clone().requires_grad_(),
        y.detach().clone().requires_grad_(),
    ]
    fused_inputs = [value.detach().clone().requires_grad_() for value in eager_inputs]
    if torch.is_tensor(point_variance):
        eager_variance = point_variance.detach().clone().requires_grad_()
        fused_variance = point_variance.detach().clone().requires_grad_()
        eager_grad_inputs = (*eager_inputs, eager_variance)
        fused_grad_inputs = (*fused_inputs, fused_variance)
    else:
        eager_variance = fused_variance = point_variance
        eager_grad_inputs = tuple(eager_inputs)
        fused_grad_inputs = tuple(fused_inputs)

    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        want = _reference(
            pp, active, *eager_inputs[:2], *eager_inputs[2:], point_band,
            eager_variance,
        )
        want_grads = torch.autograd.grad(
            want, eager_grad_inputs, grad_outputs=grad_output
        )
        got = pair_values(
            pp, active, *fused_inputs[:2], *fused_inputs[2:], point_band,
            fused_variance,
        )
        got_grads = torch.autograd.grad(
            got, fused_grad_inputs, grad_outputs=grad_output
        )
    finally:
        torch.use_deterministic_algorithms(old_determinism)

    if not _same_bits(got, want):
        got_bits = got.contiguous().view(torch.int32)
        want_bits = want.contiguous().view(torch.int32)
        mismatch = torch.nonzero(got_bits != want_bits).flatten()
        sample = mismatch[:8]
        pytest.fail(
            f"output differs for {index_order}/{backward_path}: "
            f"{mismatch.numel()} "
            f"first={sample.tolist()} got={got[sample].tolist()} "
            f"want={want[sample].tolist()} "
            f"got_bits={got_bits[sample].tolist()} "
            f"want_bits={want_bits[sample].tolist()}"
        )
    for name, actual, expected in zip(
        ("rows", "amplitudes", "x", "y", "point_variance"),
        got_grads,
        want_grads,
    ):
        if not _same_bits(actual, expected):
            actual_bits = actual.contiguous().view(torch.int32).flatten()
            expected_bits = expected.contiguous().view(torch.int32).flatten()
            mismatch = torch.nonzero(actual_bits != expected_bits).flatten()
            sample = mismatch[:8]
            pytest.fail(
                f"{name} gradient: {index_order}/{backward_path}: "
                f"{mismatch.numel()} first={sample.tolist()} "
                f"got={actual.flatten()[sample].tolist()} "
                f"want={expected.flatten()[sample].tolist()} "
                f"got_bits={actual_bits[sample].tolist()} "
                f"want_bits={expected_bits[sample].tolist()}"
            )

    if index_order == "unsorted" and backward_path == "fused":
        for parameter, gradient in zip(eager_grad_inputs, want_grads):
            parameter.grad = gradient.detach().clone()
        for parameter, gradient in zip(fused_grad_inputs, got_grads):
            parameter.grad = gradient.detach().clone()
        eager_optimizer = torch.optim.Adam(eager_grad_inputs, lr=3e-4)
        fused_optimizer = torch.optim.Adam(fused_grad_inputs, lr=3e-4)
        eager_optimizer.step()
        fused_optimizer.step()
        for parameter_name, eager_parameter, fused_parameter in zip(
            ("rows", "amplitudes", "x", "y", "point_variance"),
            eager_grad_inputs,
            fused_grad_inputs,
        ):
            assert _same_bits(fused_parameter, eager_parameter), (
                f"{parameter_name} optimizer update"
            )
            eager_state = eager_optimizer.state[eager_parameter]
            fused_state = fused_optimizer.state[fused_parameter]
            assert eager_state.keys() == fused_state.keys()
            for state_name in eager_state:
                assert _same_bits(
                    fused_state[state_name], eager_state[state_name]
                ), f"{parameter_name} optimizer state {state_name}"


def test_raster_pair_switches_default_and_validate(monkeypatch) -> None:
    monkeypatch.delenv("SPLAT_RASTER_PAIRS", raising=False)
    monkeypatch.delenv("SPLAT_RASTER_PAIRS_BACKWARD", raising=False)
    assert fused_raster_pairs_enabled()
    assert fused_raster_pairs_backward_enabled()
    monkeypatch.setenv("SPLAT_RASTER_PAIRS", "eager")
    monkeypatch.setenv("SPLAT_RASTER_PAIRS_BACKWARD", "autograd")
    assert not fused_raster_pairs_enabled()
    assert not fused_raster_pairs_backward_enabled()
    monkeypatch.setenv("SPLAT_RASTER_PAIRS", "invalid")
    with pytest.raises(ValueError, match="SPLAT_RASTER_PAIRS"):
        fused_raster_pairs_enabled()
    monkeypatch.setenv("SPLAT_RASTER_PAIRS_BACKWARD", "invalid")
    with pytest.raises(ValueError, match="SPLAT_RASTER_PAIRS_BACKWARD"):
        fused_raster_pairs_backward_enabled()
