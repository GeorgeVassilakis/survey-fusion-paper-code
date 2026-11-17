"""Bitwise-exact fused evaluation of packed raster pairs.

ATen evaluates the eager expression as a chain of separate kernels, rounding
after every float32 operation. Plain Triton expressions may contract a
multiply and add even with fusion disabled, so every arithmetic operation here
uses an explicit round-to-nearest libdevice intrinsic.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from survey_fusion.utils.perf_env import fused_raster_pairs_backward_enabled

_exp = libdevice.exp

_BLOCK = 256
_TWO_PI_F32 = float(np.float32(2.0 * math.pi))
_DET_MIN_F32 = float(np.float32(1e-30))


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _sqrt_ieee(a):
    """Round-to-nearest square root matching ATen's CUDA instruction."""
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;", "=f,f", [a], dtype=tl.float32,
        is_pure=True, pack=1,
    )


@triton.jit
def _forward_kernel(
    pp_ptr, active_ptr, rows_ptr, amp_ptr, x_ptr, y_ptr, band_ptr,
    variance_ptr, values_ptr, n_pairs, n_bands, scalar_variance, det_min,
    two_pi, PER_POINT_SIGMA: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offs < n_pairs
    pp = tl.load(pp_ptr + offs, mask=keep, other=0)
    active = tl.load(active_ptr + offs, mask=keep, other=0)
    base = active * 6

    r0 = tl.load(rows_ptr + base, mask=keep, other=0.0)
    r1 = tl.load(rows_ptr + base + 1, mask=keep, other=0.0)
    r2 = tl.load(rows_ptr + base + 2, mask=keep, other=0.0)
    r3 = tl.load(rows_ptr + base + 3, mask=keep, other=0.0)
    r4 = tl.load(rows_ptr + base + 4, mask=keep, other=0.0)
    r5 = tl.load(rows_ptr + base + 5, mask=keep, other=0.0)
    if PER_POINT_SIGMA:
        variance = tl.load(variance_ptr + pp, mask=keep, other=0.0)
    else:
        variance = scalar_variance.to(tl.float32)

    a_pair = _add(r0, variance)
    d_pair = _add(r1, variance)
    determinant = tl.maximum(
        _sub(_mul(a_pair, d_pair), r3), det_min.to(tl.float32)
    )
    dx = _sub(tl.load(x_ptr + pp, mask=keep, other=0.0), r4)
    dy = _sub(tl.load(y_ptr + pp, mask=keep, other=0.0), r5)
    dx2 = _mul(dx, dx)
    first = _mul(d_pair, dx2)
    cross = _mul(_mul(r2, dx), dy)
    dy2 = _mul(dy, dy)
    numerator = _add(_sub(first, cross), _mul(a_pair, dy2))
    exponent = _div(numerator, determinant)

    band = tl.load(band_ptr + pp, mask=keep, other=0)
    amplitude = tl.load(
        amp_ptr + active * n_bands + band, mask=keep, other=0.0
    )
    root = _sqrt_ieee(determinant)
    denominator = _mul(two_pi.to(tl.float32), root)
    scaled = _div(amplitude, denominator)
    exponential = _exp(_mul(-0.5, exponent))
    value = _mul(scaled, exponential)
    tl.store(values_ptr + offs, value, mask=keep)


@triton.jit
def _backward_kernel(
    pp_ptr, active_ptr, rows_ptr, amp_ptr, x_ptr, y_ptr, band_ptr,
    variance_ptr, grad_values_ptr, grad_rows_ptr, grad_amp_ptr, grad_x_ptr,
    grad_y_ptr, grad_variance_ptr, n_pairs, n_bands, scalar_variance,
    det_min, two_pi, PER_POINT_SIGMA: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offs < n_pairs
    pp = tl.load(pp_ptr + offs, mask=keep, other=0)
    active = tl.load(active_ptr + offs, mask=keep, other=0)
    base = active * 6
    r0 = tl.load(rows_ptr + base, mask=keep, other=0.0)
    r1 = tl.load(rows_ptr + base + 1, mask=keep, other=0.0)
    r2 = tl.load(rows_ptr + base + 2, mask=keep, other=0.0)
    r3 = tl.load(rows_ptr + base + 3, mask=keep, other=0.0)
    r4 = tl.load(rows_ptr + base + 4, mask=keep, other=0.0)
    r5 = tl.load(rows_ptr + base + 5, mask=keep, other=0.0)
    if PER_POINT_SIGMA:
        variance = tl.load(variance_ptr + pp, mask=keep, other=0.0)
    else:
        variance = scalar_variance.to(tl.float32)

    a_pair = _add(r0, variance)
    d_pair = _add(r1, variance)
    determinant_raw = _sub(_mul(a_pair, d_pair), r3)
    minimum = det_min.to(tl.float32)
    determinant = tl.maximum(determinant_raw, minimum)
    dx = _sub(tl.load(x_ptr + pp, mask=keep, other=0.0), r4)
    dy = _sub(tl.load(y_ptr + pp, mask=keep, other=0.0), r5)
    dx2 = _mul(dx, dx)
    first = _mul(d_pair, dx2)
    cross1 = _mul(r2, dx)
    cross = _mul(cross1, dy)
    dy2 = _mul(dy, dy)
    numerator = _add(_sub(first, cross), _mul(a_pair, dy2))
    exponent = _div(numerator, determinant)
    band = tl.load(band_ptr + pp, mask=keep, other=0)
    amplitude = tl.load(
        amp_ptr + active * n_bands + band, mask=keep, other=0.0
    )
    root = _sqrt_ieee(determinant)
    denominator = _mul(two_pi.to(tl.float32), root)
    scaled = _div(amplitude, denominator)
    exponential = _exp(_mul(-0.5, exponent))

    grad_value = tl.load(grad_values_ptr + offs, mask=keep, other=0.0)
    grad_scaled = _mul(grad_value, exponential)
    grad_exponential = _mul(grad_value, scaled)
    grad_negative_exponent = _mul(grad_exponential, exponential)
    grad_exponent = _mul(grad_negative_exponent, -0.5)

    grad_amplitude = _div(grad_scaled, denominator)
    grad_denominator = _mul(
        _mul(grad_scaled, -1.0),
        _div(_div(amplitude, denominator), denominator),
    )
    grad_root = _mul(grad_denominator, two_pi.to(tl.float32))
    grad_det_sqrt = _div(grad_root, _mul(2.0, root))

    grad_numerator = _div(grad_exponent, determinant)
    grad_det_div = _mul(
        _mul(grad_exponent, -1.0),
        _div(_div(numerator, determinant), determinant),
    )
    grad_determinant = _add(grad_det_sqrt, grad_det_div)
    grad_determinant_raw = tl.where(
        determinant_raw >= minimum, grad_determinant, 0.0
    )

    grad_d_from_num = _mul(grad_numerator, dx2)
    grad_dx2 = _mul(grad_numerator, d_pair)
    grad_dx_square = _mul(grad_dx2, _mul(2.0, dx))
    grad_cross = _mul(grad_numerator, -1.0)
    grad_cross1 = _mul(grad_cross, dy)
    grad_dy_cross = _mul(grad_cross, cross1)
    grad_r2 = _mul(grad_cross1, dx)
    grad_dx_cross = _mul(grad_cross1, r2)
    grad_a_from_num = _mul(grad_numerator, dy2)
    grad_dy2 = _mul(grad_numerator, a_pair)
    grad_dy_square = _mul(grad_dy2, _mul(2.0, dy))
    grad_dx = _add(grad_dx_square, grad_dx_cross)
    grad_dy = _add(grad_dy_cross, grad_dy_square)

    grad_a_from_det = _mul(grad_determinant_raw, d_pair)
    grad_d_from_det = _mul(grad_determinant_raw, a_pair)
    grad_r0 = _add(grad_a_from_num, grad_a_from_det)
    grad_r1 = _add(grad_d_from_num, grad_d_from_det)
    grad_r3 = _mul(grad_determinant_raw, -1.0)
    grad_r4 = _mul(grad_dx, -1.0)
    grad_r5 = _mul(grad_dy, -1.0)

    row_base = offs * 6
    tl.store(grad_rows_ptr + row_base, grad_r0, mask=keep)
    tl.store(grad_rows_ptr + row_base + 1, grad_r1, mask=keep)
    tl.store(grad_rows_ptr + row_base + 2, grad_r2, mask=keep)
    tl.store(grad_rows_ptr + row_base + 3, grad_r3, mask=keep)
    tl.store(grad_rows_ptr + row_base + 4, grad_r4, mask=keep)
    tl.store(grad_rows_ptr + row_base + 5, grad_r5, mask=keep)
    tl.store(grad_amp_ptr + offs, grad_amplitude, mask=keep)
    tl.store(grad_x_ptr + offs, grad_dx, mask=keep)
    tl.store(grad_y_ptr + offs, grad_dy, mask=keep)
    if PER_POINT_SIGMA:
        tl.store(
            grad_variance_ptr + offs, _add(grad_r0, grad_r1), mask=keep
        )


def applies(
    pp: torch.Tensor,
    active: torch.Tensor,
    rows: torch.Tensor,
    amplitudes: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    point_band: torch.Tensor,
    point_variance,
) -> bool:
    tensors = (pp, active, rows, amplitudes, x, y, point_band)
    if torch.is_tensor(point_variance):
        tensors += (point_variance,)
    return (
        rows.is_cuda
        and rows.dtype is torch.float32
        and all(t.is_cuda and t.is_contiguous() for t in tensors)
        and pp.dtype is torch.long
        and active.dtype is torch.long
        and point_band.dtype is torch.long
        and amplitudes.dtype is torch.float32
        and x.dtype is torch.float32
        and y.dtype is torch.float32
    )


def _eager_values(
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


class _RasterPairValues(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, pp, active, rows, amplitudes, x, y, point_band, point_variance
    ):
        per_point_sigma = torch.is_tensor(point_variance)
        variance_tensor = point_variance if per_point_sigma else rows
        ctx.save_for_backward(
            pp, active, rows, amplitudes, x, y, point_band, variance_tensor
        )
        ctx.per_point_sigma = per_point_sigma
        ctx.scalar_variance = None if per_point_sigma else float(point_variance)

        values = torch.empty(pp.shape, dtype=rows.dtype, device=rows.device)
        n_pairs = int(pp.numel())
        if n_pairs:
            _forward_kernel[(triton.cdiv(n_pairs, _BLOCK),)](
                pp, active, rows, amplitudes, x, y, point_band,
                variance_tensor, values, n_pairs, int(amplitudes.shape[1]),
                0.0 if per_point_sigma else ctx.scalar_variance,
                _DET_MIN_F32, _TWO_PI_F32,
                PER_POINT_SIGMA=per_point_sigma, BLOCK=_BLOCK, num_warps=4,
                enable_fp_fusion=False,
            )
        return values

    @staticmethod
    def backward(ctx, grad_values):
        pp, active, rows, amplitudes, x, y, point_band, variance_tensor = (
            ctx.saved_tensors
        )
        if fused_raster_pairs_backward_enabled():
            n_pairs = int(pp.numel())
            grad_rows_pair = torch.empty(
                (n_pairs, 6), dtype=rows.dtype, device=rows.device
            )
            pair_buffers = torch.empty(
                (4 if ctx.per_point_sigma else 3, n_pairs),
                dtype=rows.dtype, device=rows.device,
            )
            grad_amp_pair, grad_x_pair, grad_y_pair = pair_buffers[:3]
            grad_variance_pair = (
                pair_buffers[3] if ctx.per_point_sigma else grad_x_pair
            )
            _backward_kernel[(triton.cdiv(n_pairs, _BLOCK),)](
                pp, active, rows, amplitudes, x, y, point_band,
                variance_tensor, grad_values.contiguous(), grad_rows_pair,
                grad_amp_pair, grad_x_pair, grad_y_pair, grad_variance_pair,
                n_pairs, int(amplitudes.shape[1]),
                0.0 if ctx.per_point_sigma else ctx.scalar_variance,
                _DET_MIN_F32, _TWO_PI_F32,
                PER_POINT_SIGMA=ctx.per_point_sigma, BLOCK=_BLOCK,
                num_warps=4, enable_fp_fusion=False,
            )
            grad_rows = torch.zeros_like(rows)
            grad_rows.index_put_((active,), grad_rows_pair, accumulate=True)
            pair_band = point_band[pp]
            grad_amp = torch.zeros_like(amplitudes)
            grad_amp.index_put_(
                (active, pair_band), grad_amp_pair, accumulate=True
            )
            grad_x = None
            if ctx.needs_input_grad[4]:
                grad_x = torch.zeros_like(x)
                grad_x.index_put_((pp,), grad_x_pair, accumulate=True)
            grad_y = None
            if ctx.needs_input_grad[5]:
                grad_y = torch.zeros_like(y)
                grad_y.index_put_((pp,), grad_y_pair, accumulate=True)
            grad_variance = None
            if ctx.per_point_sigma and ctx.needs_input_grad[7]:
                grad_variance = torch.zeros_like(variance_tensor)
                grad_variance.index_put_(
                    (pp,), grad_variance_pair, accumulate=True
                )
            return (
                None, None, grad_rows, grad_amp, grad_x, grad_y, None,
                grad_variance,
            )

        with torch.enable_grad():
            rows_ref = rows.detach().requires_grad_(True)
            amp_ref = amplitudes.detach().requires_grad_(True)
            x_ref = x.detach().requires_grad_(True)
            y_ref = y.detach().requires_grad_(True)
            if ctx.per_point_sigma:
                variance_ref = variance_tensor.detach().requires_grad_(True)
                inputs = (rows_ref, amp_ref, x_ref, y_ref, variance_ref)
            else:
                variance_ref = ctx.scalar_variance
                inputs = (rows_ref, amp_ref, x_ref, y_ref)
            values = _eager_values(
                pp, active, rows_ref, amp_ref, x_ref, y_ref, point_band,
                variance_ref,
            )
            grads = torch.autograd.grad(values, inputs, grad_values)

        grad_rows, grad_amp, grad_x, grad_y = grads[:4]
        grad_variance = grads[4] if ctx.per_point_sigma else None
        return (
            None, None, grad_rows, grad_amp, grad_x, grad_y, None,
            grad_variance,
        )


def pair_values(
    pp, active, rows, amplitudes, x, y, point_band, point_variance
):
    """Evaluate one contiguous pair chunk with the fused CUDA kernel."""
    return _RasterPairValues.apply(
        pp, active, rows, amplitudes, x, y, point_band, point_variance
    )
