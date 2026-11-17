"""Fused Triton kernels for the hat-basis band weights.

The eager chain in `_hat_weights_and_grad` spends about forty separate ATen
kernels walking [rows, window] arrays that never leave the stage, and at
production shape that traffic is the whole cost. These kernels evaluate the
same expressions on the same operands in registers, and the results are bit
for bit what the eager chain produces. Four things make that hold:

* every arithmetic op is an explicit round-to-nearest libdevice intrinsic.
  ATen evaluates each op as its own kernel and so rounds each one; a Triton
  expression written with plain operators does not, because the compiler
  contracts a multiply feeding an add into an FMA. `enable_fp_fusion=False`
  is set as well but is not sufficient on its own: it was measured still
  contracting `a * b - c * d` inside this kernel.
* division by a Python scalar multiplies by the float32 reciprocal instead,
  which is the substitution ATen makes for a CPU-scalar divisor. The
  reciprocal is taken in float64 and rounded once, which is what ATen does
  and is not the same as dividing in float32 (they differ for total0 of the
  SDSS z band, among others).
* negation is a multiply by -1.0, an exact sign flip on zeros as well,
  rather than a subtraction from zero.
* the segment lookup repeats `searchsorted(..., right=True)` as a branchless
  upper bound. Comparing against the smallest float32 at or above each
  float64 boundary decides the same predicate for a float32 query, so the
  search needs no float64 arithmetic.

Two kernels rather than one because the weight at a knot sums the two
intervals flanking it, and no Triton block can shift a value across lanes:
the interval kernel writes the four interval arrays and the knot kernel reads
the shifted pair back. Accumulating the pair with atomics would be cheaper
still, but only up to the sign of a zero, so the sum is written out.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary
from triton.language.extra import libdevice

_add = libdevice.add_rn
_sub = libdevice.sub_rn
_mul = libdevice.mul_rn
_div = libdevice.div_rn

_BLOCK_INTERVAL = 512
_BLOCK_DENSE = 512
# Element indices are int32 inside the kernels.
_INDEX_LIMIT = 2 ** 31

_F32_BOUND_CACHE = WeakTensorKeyDictionary()


def scalar_reciprocal(value: float) -> float:
    """The float32 reciprocal ATen multiplies by in place of dividing."""
    return float(np.float32(1.0 / float(value)))


_INV6 = scalar_reciprocal(6.0)


def float32_segment_bounds(interior: torch.Tensor) -> torch.Tensor:
    """The smallest float32 at or above each float64 response boundary.

    For a float32 query x, `boundary <= x` and `ceil_f32(boundary) <= x` are
    the same predicate, so the segment search runs entirely in float32 and
    still returns the index `torch.searchsorted` returns in float64.
    """
    cached = _F32_BOUND_CACHE.get(interior)
    if cached is not None and cached[0] == interior._version:
        return cached[1]
    exact = interior.detach().cpu().to(torch.float64).numpy()
    near = exact.astype(np.float32)
    above = np.nextafter(near, np.float32(np.inf))
    bounds = torch.from_numpy(
        np.where(near.astype(np.float64) >= exact, near, above)
    ).to(interior.device)
    _F32_BOUND_CACHE[interior] = (interior._version, bounds)
    return bounds


@triton.jit
def _segment_index(bounds, x, seed, S: tl.constexpr, NBITS: tl.constexpr):
    """searchsorted(bounds[:S], x, right=True), clamped to S - 1."""
    index = seed
    for bit in tl.static_range(NBITS):
        probe = index + (1 << (NBITS - 1 - bit))
        inside = probe <= S
        edge = tl.load(bounds + tl.where(inside, probe - 1, 0), mask=inside,
                       other=0.0)
        index = tl.where(inside & (edge <= x), probe, index)
    return tl.minimum(index, S - 1)


@triton.jit
def _photon_response(x, edge, value, slope):
    """The piecewise-linear photon response x * T(x) inside one segment."""
    return _mul(x, _add(value, _mul(slope, _sub(x, edge))))


@triton.jit
def _ramp_integral(r_p, p, r_m, m, r_q, q, ramp):
    """Simpson's rule for the response against the hat ramp from `ramp`."""
    f_p = _mul(r_p, _sub(p, ramp))
    f_m = _mul(r_m, _sub(m, ramp))
    f_q = _mul(r_q, _sub(q, ramp))
    return _add(_add(f_p, _mul(4.0, f_m)), f_q)


@triton.jit
def _prefix_delta(hi_ptr, lo_ptr, upper, lower, keep):
    """The double-single prefix difference over complete response segments."""
    return _add(
        _sub(tl.load(hi_ptr + upper, mask=keep, other=0.0),
             tl.load(hi_ptr + lower, mask=keep, other=0.0)),
        _sub(tl.load(lo_ptr + upper, mask=keep, other=0.0),
             tl.load(lo_ptr + lower, mask=keep, other=0.0)),
    )


@triton.jit
def _interval_kernel(
    t_ptr, c_ptr, start_ptr, bound_ptr, edge_ptr, value_ptr, slope_ptr,
    p0_hi_ptr, p0_lo_ptr, p1_hi_ptr, p1_lo_ptr,
    left_ptr, right_ptr, d_left_ptr, d_right_ptr,
    n_elem, lo, hi, inv6,
    NI: tl.constexpr, S: tl.constexpr, NBITS: tl.constexpr,
    WINDOWED: tl.constexpr, NEED_GRAD: tl.constexpr, BLOCK: tl.constexpr,
):
    """Hat-ramp integrals over one knot interval, and their derivative in t."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offs < n_elem
    row = offs // NI
    col = offs - row * NI
    lo = lo.to(tl.float32)
    hi = hi.to(tl.float32)
    inv6 = inv6.to(tl.float32)

    knot = col
    if WINDOWED:
        knot = tl.load(start_ptr + row, mask=keep, other=0).to(tl.int32) + col
    scale = tl.load(t_ptr + row, mask=keep, other=1.0)
    c_lo = tl.load(c_ptr + knot, mask=keep, other=0.0)
    c_hi = tl.load(c_ptr + knot + 1, mask=keep, other=0.0)
    lam_lo = _mul(scale, c_lo)
    lam_hi = _mul(scale, c_hi)
    a = tl.minimum(tl.maximum(lam_lo, lo), hi)
    b = tl.minimum(tl.maximum(lam_hi, lo), hi)

    seed = tl.zeros_like(col)
    seg_a = _segment_index(bound_ptr, a, seed, S, NBITS)
    seg_b = _segment_index(bound_ptr, b, seed, S, NBITS)
    edge_a = tl.load(edge_ptr + seg_a, mask=keep, other=0.0)
    value_a = tl.load(value_ptr + seg_a, mask=keep, other=0.0)
    slope_a = tl.load(slope_ptr + seg_a, mask=keep, other=0.0)
    edge_b = tl.load(edge_ptr + seg_b, mask=keep, other=0.0)
    value_b = tl.load(value_ptr + seg_b, mask=keep, other=0.0)
    slope_b = tl.load(slope_ptr + seg_b, mask=keep, other=0.0)
    resp_a = _photon_response(a, edge_a, value_a, slope_a)
    resp_b = _photon_response(b, edge_b, value_b, slope_b)

    active = b > a
    span = _mul(scale, _sub(c_hi, c_lo))
    seg_next = seg_a + 1
    a_up = tl.minimum(tl.load(edge_ptr + seg_next, mask=keep, other=0.0), b)
    b_lo = tl.maximum(edge_b, a_up)

    # One Simpson panel inside each of the two response segments the interval
    # can touch; the shared knot samples are the responses computed above.
    mid_a = _mul(0.5, _add(a, a_up))
    h6_a = _mul(_sub(a_up, a), inv6)
    resp_mid_a = _photon_response(mid_a, edge_a, value_a, slope_a)
    resp_up_a = _photon_response(a_up, edge_a, value_a, slope_a)
    mid_b = _mul(0.5, _add(b_lo, b))
    h6_b = _mul(_sub(b, b_lo), inv6)
    resp_lo_b = _photon_response(b_lo, edge_b, value_b, slope_b)
    resp_mid_b = _photon_response(mid_b, edge_b, value_b, slope_b)

    i_right = _add(
        _mul(h6_a, _ramp_integral(
            resp_a, a, resp_mid_a, mid_a, resp_up_a, a_up, lam_lo)),
        _mul(h6_b, _ramp_integral(
            resp_lo_b, b_lo, resp_mid_b, mid_b, resp_b, b, lam_lo)),
    )
    # The reversed ramp negates all three samples; the sign rides the panel
    # scale instead, which is the same value bit for bit.
    i_left = _add(
        _mul(_mul(-1.0, h6_a), _ramp_integral(
            resp_a, a, resp_mid_a, mid_a, resp_up_a, a_up, lam_hi)),
        _mul(_mul(-1.0, h6_b), _ramp_integral(
            resp_lo_b, b_lo, resp_mid_b, mid_b, resp_b, b, lam_hi)),
    )

    upper = tl.maximum(seg_b, seg_next)
    delta0 = _prefix_delta(p0_hi_ptr, p0_lo_ptr, upper, seg_next, keep)
    delta1 = _prefix_delta(p1_hi_ptr, p1_lo_ptr, upper, seg_next, keep)
    i_right = _add(i_right, _sub(delta1, _mul(lam_lo, delta0)))
    i_left = _add(i_left, _sub(_mul(lam_hi, delta0), delta1))

    right = tl.where(active, _div(i_right, span), 0.0)
    left = tl.where(active, _div(i_left, span), 0.0)
    tl.store(right_ptr + offs, right, mask=keep)
    tl.store(left_ptr + offs, left, mask=keep)

    if NEED_GRAD:
        mass = _add(right, left)
        alpha_lo = tl.where((lam_lo > lo) & (lam_lo < hi), 1.0, 0.0)
        alpha_hi = tl.where((lam_hi > lo) & (lam_hi < hi), 1.0, 0.0)
        d_right = _sub(
            _sub(_mul(_mul(alpha_hi, c_hi), resp_b),
                 _mul(_div(c_lo, span), mass)),
            _div(right, scale),
        )
        d_left = _sub(
            _add(_mul(_mul(_mul(-1.0, alpha_lo), c_lo), resp_a),
                 _mul(_div(c_hi, span), mass)),
            _div(left, scale),
        )
        tl.store(d_right_ptr + offs, tl.where(active, d_right, 0.0), mask=keep)
        tl.store(d_left_ptr + offs, tl.where(active, d_left, 0.0), mask=keep)


@triton.jit
def _knot_stagger_sum(source_left, source_right, base, knot, keep,
                      from_left, from_right):
    """left[knot] + right[knot - 1], with the window ends as plain copies."""
    lv = tl.load(source_left + base + knot, mask=keep & from_left, other=0.0)
    rv = tl.load(
        source_right + base + knot - 1, mask=keep & from_right, other=0.0
    )
    total = tl.where(
        from_left & from_right, _add(lv, rv), tl.where(from_left, lv, rv)
    )
    return tl.where(from_left | from_right, total, 0.0)


@triton.jit
def _knot_kernel(
    left_ptr, right_ptr, d_left_ptr, d_right_ptr, start_ptr,
    w_ptr, dw_ptr, n_elem, inv_total0,
    G: tl.constexpr, NI: tl.constexpr, WINDOWED: tl.constexpr,
    NEED_GRAD: tl.constexpr, BLOCK: tl.constexpr,
):
    """Stagger-sum the intervals flanking each knot, then clamp to positive."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offs < n_elem
    row = offs // G
    knot = offs - row * G
    if WINDOWED:
        knot = knot - tl.load(start_ptr + row, mask=keep, other=0).to(tl.int32)
    base = row * NI
    from_left = (knot >= 0) & (knot < NI)
    from_right = (knot >= 1) & (knot <= NI)
    inv_total0 = inv_total0.to(tl.float32)

    weight = _mul(
        _knot_stagger_sum(
            left_ptr, right_ptr, base, knot, keep, from_left, from_right
        ),
        inv_total0,
    )
    positive = weight > 0.0
    tl.store(w_ptr + offs, tl.where(positive, weight, 0.0), mask=keep)

    if NEED_GRAD:
        derivative = _mul(
            _knot_stagger_sum(
                d_left_ptr, d_right_ptr, base, knot, keep, from_left, from_right
            ),
            inv_total0,
        )
        tl.store(
            dw_ptr + offs, tl.where(positive, derivative, 0.0), mask=keep
        )


def applies(t: torch.Tensor, window: int, G: int) -> bool:
    """Whether the fused path can reproduce the eager chain for these inputs."""
    rows = int(t.shape[0])
    return (
        t.is_cuda
        and t.dtype is torch.float32
        and window >= 2
        and rows * window < _INDEX_LIMIT
        and rows * G < _INDEX_LIMIT
    )


def interval_integrals(
    t, C, start, window, bounds, Ld, Td, sloped,
    p0_hi, p0_lo, p1_hi, p1_lo, R0, R1, *, need_grad,
):
    """The hat-ramp integrals and derivatives for every knot interval."""
    rows = int(t.shape[0])
    intervals = window - 1
    segments = int(Ld.shape[0]) - 1
    # The gradient planes are unread when NEED_GRAD is false, and the kernel
    # stores nothing to them, so they alias the forward planes there.
    buffer = torch.empty(
        (4 if need_grad else 2, rows, intervals), dtype=t.dtype, device=t.device
    )
    left, right = buffer[0], buffer[1]
    d_left = buffer[2] if need_grad else left
    d_right = buffer[3] if need_grad else right
    n_elem = rows * intervals
    if n_elem:
        _interval_kernel[(triton.cdiv(n_elem, _BLOCK_INTERVAL),)](
            t.contiguous(), C, t if start is None else start,
            float32_segment_bounds(bounds), Ld, Td, sloped,
            p0_hi, p0_lo, p1_hi, p1_lo, left, right, d_left, d_right,
            n_elem, R0, R1, _INV6,
            NI=intervals, S=segments, NBITS=segments.bit_length(),
            WINDOWED=start is not None, NEED_GRAD=need_grad,
            BLOCK=_BLOCK_INTERVAL, num_warps=4, enable_fp_fusion=False,
        )
    if need_grad:
        return left, right, d_left, d_right
    return left, right, None, None


def dropped_knot_weights(
    left, right, d_left, d_right, start, G, total0, *, need_grad,
):
    """Dense [rows, G] weights and derivatives for the drop-off-grid operator."""
    rows, intervals = left.shape
    w = torch.empty((rows, G), dtype=left.dtype, device=left.device)
    dw = torch.empty_like(w) if need_grad else w
    n_elem = rows * G
    if n_elem:
        _knot_kernel[(triton.cdiv(n_elem, _BLOCK_DENSE),)](
            left, right,
            left if d_left is None else d_left,
            right if d_right is None else d_right,
            left if start is None else start,
            w, dw, n_elem, scalar_reciprocal(total0),
            G=G, NI=intervals, WINDOWED=start is not None,
            NEED_GRAD=need_grad, BLOCK=_BLOCK_DENSE, num_warps=4,
            enable_fp_fusion=False,
        )
    return w, (dw if need_grad else None)
