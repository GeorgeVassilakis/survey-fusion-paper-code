"""Bitwise-exact fused forward for paired zero-extension fiber bins.

The eager implementation evaluates each float64 arithmetic operation in a
separate ATen kernel. Every arithmetic operation here therefore uses an
explicit round-to-nearest instruction. Plain Triton operators are not enough:
the compiler can contract a multiply feeding an add even when fp fusion is
disabled.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 256
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INDEX_LIMIT = 2 ** 31


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
        dtype=tl.float64, is_pure=True, pack=1,
    )


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
        dtype=tl.float64, is_pure=True, pack=1,
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
        dtype=tl.float64, is_pure=True, pack=1,
    )


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f64 $0, $1, $2;", "=d,d,d", [a, b],
        dtype=tl.float64, is_pure=True, pack=1,
    )


@triton.jit
def _normal_cdf(z, inv_sqrt_2):
    half = tl.full(z.shape, 0.5, tl.float64)
    one = tl.full(z.shape, 1.0, tl.float64)
    return _mul(half, _add(one, libdevice.erf(_mul(z, inv_sqrt_2))))


@triton.jit
def _normal_pdf(z, inv_sqrt_2pi):
    negative_half = tl.full(z.shape, -0.5, tl.float64)
    return _mul(libdevice.exp(_mul(negative_half, _mul(z, z))), inv_sqrt_2pi)


@triton.jit
def _cdf_integral(q, y, sigma, inv_sqrt_2, inv_sqrt_2pi):
    z = _div(_sub(q, y), sigma)
    cdf = _normal_cdf(z, inv_sqrt_2)
    pdf = _normal_pdf(z, inv_sqrt_2pi)
    return _sub(_mul(_sub(y, q), cdf), _mul(sigma, pdf))


@triton.jit
def _first_moment_integral(q, y, sigma, inv_sqrt_2, inv_sqrt_2pi):
    delta = _sub(q, y)
    z = _div(delta, sigma)
    cdf = _normal_cdf(z, inv_sqrt_2)
    pdf = _normal_pdf(z, inv_sqrt_2pi)
    a0 = _sub(_mul(_sub(y, q), cdf), _mul(sigma, pdf))
    half = tl.full(delta.shape, 0.5, tl.float64)
    b0 = _mul(
        half,
        _add(
            _mul(_add(_mul(delta, delta), _mul(sigma, sigma)), cdf),
            _mul(_mul(sigma, delta), pdf),
        ),
    )
    return _sub(_mul(y, a0), b0)


@triton.jit
def _sigma_pdf_integral(q, y, sigma, inv_sqrt_2):
    z = _div(_sub(q, y), sigma)
    negative_one = tl.full(z.shape, -1.0, tl.float64)
    return _mul(
        _mul(negative_one, _mul(sigma, sigma)), _normal_cdf(z, inv_sqrt_2)
    )


@triton.jit
def _four_corner(fn_a, fn_b, fn_c, fn_d):
    return _add(_sub(_sub(fn_a, fn_b), fn_c), fn_d)


@triton.jit
def _linear_integral(
    bin_lower, bin_upper, bin_width, sigma, source_lower, source_upper,
    slope, intercept, inv_sqrt_2, inv_sqrt_2pi,
):
    p0 = _four_corner(
        _cdf_integral(source_upper, bin_upper, sigma, inv_sqrt_2, inv_sqrt_2pi),
        _cdf_integral(source_lower, bin_upper, sigma, inv_sqrt_2, inv_sqrt_2pi),
        _cdf_integral(source_upper, bin_lower, sigma, inv_sqrt_2, inv_sqrt_2pi),
        _cdf_integral(source_lower, bin_lower, sigma, inv_sqrt_2, inv_sqrt_2pi),
    )
    p1 = _four_corner(
        _first_moment_integral(
            source_upper, bin_upper, sigma, inv_sqrt_2, inv_sqrt_2pi
        ),
        _first_moment_integral(
            source_lower, bin_upper, sigma, inv_sqrt_2, inv_sqrt_2pi
        ),
        _first_moment_integral(
            source_upper, bin_lower, sigma, inv_sqrt_2, inv_sqrt_2pi
        ),
        _first_moment_integral(
            source_lower, bin_lower, sigma, inv_sqrt_2, inv_sqrt_2pi
        ),
    )
    q = _four_corner(
        _sigma_pdf_integral(source_lower, bin_upper, sigma, inv_sqrt_2),
        _sigma_pdf_integral(source_upper, bin_upper, sigma, inv_sqrt_2),
        _sigma_pdf_integral(source_lower, bin_lower, sigma, inv_sqrt_2),
        _sigma_pdf_integral(source_upper, bin_lower, sigma, inv_sqrt_2),
    )
    result = _div(
        _add(_mul(slope, _add(p1, q)), _mul(intercept, p0)), bin_width
    )
    active = (bin_upper > bin_lower) & (source_upper > source_lower)
    return tl.where(active, result, 0.0)


@triton.jit
def _moment_kernel(
    knot_lam_ptr, lower_ptr, upper_ptr, sigma_ptr, moments_ptr,
    n_rows, n_knots: tl.constexpr, inv_sqrt_2: tl.constexpr,
    inv_sqrt_2pi: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n_elem = n_rows * (n_knots + 1)
    keep = offsets < n_elem
    row = offsets // (n_knots + 1)
    col = offsets - row * (n_knots + 1)
    knot_col = tl.minimum(col, n_knots - 1)

    row_base = row * n_knots
    knot = tl.load(knot_lam_ptr + row_base + knot_col, mask=keep, other=0.0).to(
        tl.float64
    )
    knot_first = tl.load(knot_lam_ptr + row_base, mask=keep, other=0.0).to(
        tl.float64
    )
    knot_last = tl.load(
        knot_lam_ptr + row_base + n_knots - 1, mask=keep, other=0.0
    ).to(tl.float64)
    lower = tl.load(lower_ptr + row, mask=keep, other=0.0).to(tl.float64)
    upper = tl.load(upper_ptr + row, mask=keep, other=0.0).to(tl.float64)
    sigma = tl.load(sigma_ptr + row, mask=keep, other=1.0).to(tl.float64)
    inv_sqrt_2 = tl.full(knot.shape, inv_sqrt_2, tl.float64)
    inv_sqrt_2pi = tl.full(knot.shape, inv_sqrt_2pi, tl.float64)
    width = _sub(upper, lower)
    bin_lower = tl.maximum(lower, knot_first)
    bin_upper = tl.minimum(upper, knot_last)
    is_base = col == n_knots
    source_lower = tl.where(is_base, knot_first, knot)
    zero = tl.full(knot.shape, 0.0, tl.float64)
    one = tl.full(knot.shape, 1.0, tl.float64)
    negative_one = tl.full(knot.shape, -1.0, tl.float64)
    slope = tl.where(is_base, zero, one)
    intercept = tl.where(is_base, one, _mul(knot, negative_one))
    value = _linear_integral(
        bin_lower, bin_upper, width, sigma, source_lower, knot_last,
        slope, intercept, inv_sqrt_2, inv_sqrt_2pi,
    )
    tl.store(moments_ptr + offsets, value, mask=keep)


@triton.jit
def _contribution_kernel(
    knot_lam_ptr, knot_values_ptr, moments_ptr, contribution_ptr,
    n_elem, n_knots: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offsets < n_elem
    row = offsets // n_knots
    col = offsets - row * n_knots
    knot_base = row * n_knots
    moment_base = row * (n_knots + 1)

    left_col = tl.maximum(col - 1, 0)
    right_col = tl.minimum(col + 1, n_knots - 1)
    h_left = tl.load(moments_ptr + moment_base + left_col, mask=keep)
    h_mid = tl.load(moments_ptr + moment_base + col, mask=keep)
    h_right = tl.load(moments_ptr + moment_base + right_col, mask=keep)
    k_left = tl.load(knot_lam_ptr + knot_base + left_col, mask=keep).to(tl.float64)
    k_mid = tl.load(knot_lam_ptr + knot_base + col, mask=keep).to(tl.float64)
    k_right = tl.load(knot_lam_ptr + knot_base + right_col, mask=keep).to(tl.float64)
    scaled_left = _div(_sub(h_left, h_mid), _sub(k_mid, k_left))
    scaled_right = _div(_sub(h_mid, h_right), _sub(k_right, k_mid))
    base = tl.load(moments_ptr + moment_base + n_knots, mask=keep)
    weight = tl.where(
        col == 0,
        _sub(base, scaled_right),
        tl.where(col == n_knots - 1, scaled_left, _sub(scaled_left, scaled_right)),
    )
    weight = tl.maximum(weight, 0.0)
    knot_value = tl.load(knot_values_ptr + offsets, mask=keep).to(tl.float64)
    tl.store(contribution_ptr + offsets, _mul(weight, knot_value), mask=keep)


def applies(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    lsf_mode: str | None,
    off_grid: str,
) -> bool:
    tensors = (knot_lam, knot_values, lower, upper, lsf_sigma)
    return (
        off_grid == "zero"
        and lsf_mode == "positive"
        and knot_lam.is_cuda
        and knot_lam.dtype == torch.float32
        and all(t.is_cuda and t.is_contiguous() for t in tensors)
        and all(t.dtype == torch.float32 for t in tensors)
        and knot_lam.numel() < _INDEX_LIMIT
    )


def contributions(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
) -> torch.Tensor:
    """Return the eager expression's float64 weighted knot values."""
    n_rows, n_knots = knot_lam.shape
    moments = torch.empty(
        (n_rows, n_knots + 1), dtype=torch.float64, device=knot_lam.device
    )
    n_moments = moments.numel()
    if n_moments:
        _moment_kernel[(triton.cdiv(n_moments, _BLOCK),)](
            knot_lam, lower, upper, lsf_sigma, moments,
            n_rows, n_knots, _INV_SQRT_2, _INV_SQRT_2PI,
            BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False,
        )
    result = torch.empty(
        (n_rows, n_knots), dtype=torch.float64, device=knot_lam.device
    )
    n_elem = result.numel()
    if n_elem:
        _contribution_kernel[(triton.cdiv(n_elem, _BLOCK),)](
            knot_lam, knot_values, moments, result, n_elem, n_knots,
            BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False,
        )
    return result
