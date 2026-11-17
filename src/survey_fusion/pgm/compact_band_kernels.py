"""Compact band-weight storage with the production reduction orders retained."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import hat_band_kernels
from .band_contraction_kernels import _init_kernel
from .raster_pair_kernels import _add, _mul


@triton.jit
def _forward_band(
    values,
    rows,
    weights,
    starts,
    averages,
    positions,
    count,
    WIDTH: tl.constexpr,
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 4)
    keep = local < count
    row = tl.load(rows + local, mask=keep, other=0)
    start = tl.load(starts + local, mask=keep, other=0).to(tl.int32)
    accumulator = tl.full((BLOCK, 4), 0.0, tl.float32)
    for i in tl.static_range(24):
        knot = i * 4 + lane
        column = knot[None, :] - start[:, None]
        valid = keep[:, None] & (column >= 0) & (column < WIDTH)
        value = tl.load(values + row[:, None] * 96 + knot[None, :], mask=keep[:, None], other=0.0)
        weight = tl.load(weights + local[:, None] * WIDTH + column, mask=valid, other=0.0)
        accumulator = tl.fma(value, weight, accumulator)
    total = tl.sum(accumulator, 1)
    tl.store(averages + row * 5 + BAND, total, mask=keep)
    tl.store(positions + row * 5 + BAND, local, mask=keep)


@triton.jit
def _backward_values(
    grad,
    positions,
    w0,
    w1,
    w2,
    w3,
    w4,
    s0,
    s1,
    s2,
    s3,
    s4,
    result,
    count,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    W3: tl.constexpr,
    W4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    row, knot = offset // 96, offset % 96
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for band in tl.static_range(5):
        if band == 0:
            weights, starts, width = w0, s0, W0
        elif band == 1:
            weights, starts, width = w1, s1, W1
        elif band == 2:
            weights, starts, width = w2, s2, W2
        elif band == 3:
            weights, starts, width = w3, s3, W3
        else:
            weights, starts, width = w4, s4, W4
        position = tl.load(positions + row * 5 + band, mask=mask, other=-1)
        start = tl.load(starts + position, mask=mask & (position >= 0), other=0).to(tl.int32)
        column = knot - start
        valid = mask & (position >= 0) & (column >= 0) & (column < width)
        weight = tl.load(weights + position * width + column, mask=valid, other=0.0)
        go = tl.load(grad + row * 5 + band, mask=mask, other=0.0)
        total = tl.fma(go, weight, total)
    tl.store(result + offset, total, mask=mask)


@triton.jit
def _backward_weights(
    values, grad, rows, starts, result, count, WIDTH: tl.constexpr, BAND: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offset < count * WIDTH
    local, column = offset // WIDTH, offset % WIDTH
    row = tl.load(rows + local, mask=keep, other=0)
    start = tl.load(starts + local, mask=keep, other=0).to(tl.int32)
    value = tl.load(values + row * 96 + start + column, mask=keep, other=0.0)
    go = tl.load(grad + row * 5 + BAND, mask=keep, other=0.0)
    tl.store(result + offset, _mul(value, go), mask=keep)


@triton.jit
def _gradient_t(grad, derivative, starts, result, count, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 32)
    keep = row < count
    start = tl.load(starts + row, mask=keep, other=0).to(tl.int32)
    accumulator = tl.full((BLOCK, 32), 0.0, tl.float32)
    for i in tl.static_range(3):
        knot = lane + i * 32
        column = knot[None, :] - start[:, None]
        valid = keep[:, None] & (column >= 0) & (column < WIDTH)
        go = tl.load(grad + row[:, None] * WIDTH + column, mask=valid, other=0.0)
        dw = tl.load(derivative + row[:, None] * WIDTH + column, mask=valid, other=0.0)
        accumulator = _add(accumulator, _mul(go, dw))
    total = tl.sum(accumulator, 1)
    tl.store(result + row, total, mask=keep)


class CompactHatWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, C, tables, window):
        from .splat_spectrum import _hat_interior_bounds

        assert C.numel() == 96 and t.dtype == torch.float32
        L64, Ld, Td, sloped, p0_hi, p0_lo, p1_hi, p1_lo, total0, R0, R1 = tables
        G = C.numel()
        window = min(window, G)
        if window >= G:
            start = None
        else:
            start = torch.searchsorted(C.detach(), (R0 / t).to(C.dtype).contiguous(), right=True)
            start = (start - 2).clamp(0, G - window)
        left, right, dl, dr = hat_band_kernels.interval_integrals(
            t,
            C,
            start,
            window,
            _hat_interior_bounds(L64),
            Ld,
            Td,
            sloped,
            p0_hi,
            p0_lo,
            p1_hi,
            p1_lo,
            R0,
            R1,
            need_grad=True,
        )
        weights, derivative = hat_band_kernels.dropped_knot_weights(
            left,
            right,
            dl,
            dr,
            None,
            window,
            total0,
            need_grad=True,
        )
        if start is None:
            start = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
        ctx.save_for_backward(derivative, start)
        ctx.mark_non_differentiable(start)
        return weights, start

    @staticmethod
    def backward(ctx, grad_weights, grad_start):
        derivative, starts = ctx.saved_tensors
        rows, width = derivative.shape
        grad_weights = grad_weights.contiguous()
        if rows >= 4096:
            result = torch.empty(rows, dtype=derivative.dtype, device=derivative.device)
            _gradient_t[(triton.cdiv(rows, 4),)](
                grad_weights,
                derivative,
                starts,
                result,
                rows,
                WIDTH=width,
                BLOCK=4,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            # Small row counts use another ATen reduction configuration.
            product = grad_weights * derivative
            indices = starts[:, None] + torch.arange(width, device=starts.device)[None, :]
            dense = torch.zeros((rows, 96), dtype=product.dtype, device=product.device)
            dense.scatter_(1, indices, product)
            result = dense.sum(1)
        return result, None, None, None


class CompactBandContraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, placements, starts, *weights):
        rows = values.shape[0]
        averages = torch.empty((rows, 5), dtype=values.dtype, device=values.device)
        positions = torch.empty((rows, 5), dtype=torch.int32, device=values.device)
        _init_kernel[(triton.cdiv(rows * 5, 256),)](
            averages,
            positions,
            rows * 5,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for (band, selected), start, weight in zip(placements, starts, weights):
            _forward_band[(triton.cdiv(selected.numel(), 32),)](
                values,
                selected,
                weight,
                start,
                averages,
                positions,
                selected.numel(),
                WIDTH=weight.shape[1],
                BAND=band,
                BLOCK=32,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(values, positions, *starts, *weights)
        ctx.placements = placements
        return averages

    @staticmethod
    def backward(ctx, grad_averages):
        count = len(ctx.placements)
        values, positions, *saved = ctx.saved_tensors
        starts, weights = saved[:count], saved[count:]
        grad_averages = grad_averages.contiguous()
        wb, sb, widths = [weights[0]] * 5, [starts[0]] * 5, [weights[0].shape[1]] * 5
        for (band, _), start, weight in zip(ctx.placements, starts, weights):
            wb[band], sb[band], widths[band] = weight, start, weight.shape[1]
        grad_values = torch.empty_like(values)
        _backward_values[(triton.cdiv(values.numel(), 256),)](
            grad_averages,
            positions,
            *wb,
            *sb,
            grad_values,
            values.numel(),
            *widths,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        gradients = []
        for (band, selected), start, weight in zip(ctx.placements, starts, weights):
            gradient = torch.empty_like(weight)
            _backward_weights[(triton.cdiv(weight.numel(), 256),)](
                values,
                grad_averages,
                selected,
                start,
                gradient,
                selected.numel(),
                WIDTH=weight.shape[1],
                BAND=band,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
            gradients.append(gradient)
        return grad_values, None, None, *gradients
