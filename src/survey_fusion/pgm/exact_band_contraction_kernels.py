"""Sparse contraction with the measured cuBLAS fp32 reduction order.

For the production 96-knot, five-band B200 operation, forward cuBLAS uses
four interleaved FMA accumulators followed by a binary sum. Its spectrum
gradient uses five ascending-band FMAs. These kernels reproduce those orders
without constructing the dense row/band/knot weight block.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from survey_fusion.utils.perf_env import exact_band_reduction_rows_supported, validated_cublas_runtime

from .band_contraction_kernels import _by_band, _init_kernel


def applies(values, n_bands, placements, band_weights):
    return (
        values.is_cuda
        and values.dtype == torch.float32
        and values.is_contiguous()
        and exact_band_reduction_rows_supported(values.shape[0])
        and values.shape[1] == 96
        and n_bands == 5
        and values.numel() < 2**31
        and all(r.is_cuda and r.dtype == torch.int64 and r.is_contiguous() for _, r in placements)
        and all(w.is_cuda and w.dtype == torch.float32 and w.is_contiguous() for w in band_weights)
        and validated_cublas_runtime()
        and torch.cuda.get_device_capability(values.device) == (10, 0)
    )


@triton.jit
def _forward_band(
    values,
    rows,
    weights,
    averages,
    positions,
    count,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    RS: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 4)
    keep = local < count
    row = tl.load(rows + local * RS, mask=keep, other=0)
    accumulator = tl.full((BLOCK, 4), 0.0, tl.float32)
    for i in tl.static_range(24):
        knot = i * 4 + lane
        value = tl.load(values + row[:, None] * VS0 + knot[None, :] * VS1, mask=keep[:, None], other=0.0)
        weight = tl.load(weights + local[:, None] * WS0 + knot[None, :] * WS1, mask=keep[:, None], other=0.0)
        accumulator = tl.fma(value, weight, accumulator)
    total = tl.sum(accumulator, axis=1)
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
    result,
    count,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    W00: tl.constexpr,
    W01: tl.constexpr,
    W10: tl.constexpr,
    W11: tl.constexpr,
    W20: tl.constexpr,
    W21: tl.constexpr,
    W30: tl.constexpr,
    W31: tl.constexpr,
    W40: tl.constexpr,
    W41: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    row = offset // 96
    knot = offset % 96
    total = tl.full((BLOCK,), 0.0, tl.float32)
    p0 = tl.load(positions + row * 5, mask=mask, other=-1)
    weight = tl.load(w0 + p0 * W00 + knot * W01, mask=mask & (p0 >= 0), other=0.0)
    total = tl.fma(tl.load(grad + row * GS0, mask=mask, other=0.0), weight, total)
    p1 = tl.load(positions + row * 5 + 1, mask=mask, other=-1)
    weight = tl.load(w1 + p1 * W10 + knot * W11, mask=mask & (p1 >= 0), other=0.0)
    total = tl.fma(tl.load(grad + row * GS0 + GS1, mask=mask, other=0.0), weight, total)
    p2 = tl.load(positions + row * 5 + 2, mask=mask, other=-1)
    weight = tl.load(w2 + p2 * W20 + knot * W21, mask=mask & (p2 >= 0), other=0.0)
    total = tl.fma(tl.load(grad + row * GS0 + 2 * GS1, mask=mask, other=0.0), weight, total)
    p3 = tl.load(positions + row * 5 + 3, mask=mask, other=-1)
    weight = tl.load(w3 + p3 * W30 + knot * W31, mask=mask & (p3 >= 0), other=0.0)
    total = tl.fma(tl.load(grad + row * GS0 + 3 * GS1, mask=mask, other=0.0), weight, total)
    p4 = tl.load(positions + row * 5 + 4, mask=mask, other=-1)
    weight = tl.load(w4 + p4 * W40 + knot * W41, mask=mask & (p4 >= 0), other=0.0)
    total = tl.fma(tl.load(grad + row * GS0 + 4 * GS1, mask=mask, other=0.0), weight, total)
    tl.store(result + offset, total, mask=mask)


def forward(values, n_bands, placements, band_weights):
    assert n_bands == 5 and values.shape[1] == 96
    rows = values.shape[0]
    averages = torch.empty((rows, 5), dtype=values.dtype, device=values.device)
    positions = torch.empty((rows, 5), dtype=torch.int32, device=values.device)
    if rows:
        _init_kernel[(triton.cdiv(rows * 5, 256),)](
            averages,
            positions,
            rows * 5,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    for (band, selected), block in zip(placements, band_weights):
        if selected.numel():
            _forward_band[(triton.cdiv(selected.numel(), 32),)](
                values,
                selected,
                block,
                averages,
                positions,
                selected.numel(),
                *values.stride(),
                selected.stride(0),
                *block.stride(),
                band,
                BLOCK=32,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return averages, positions


def backward_values(grad_averages, positions, n_bands, placements, band_weights, knots):
    assert n_bands == 5 and knots == 96
    _, weights, _ = _by_band(n_bands, placements, band_weights)
    result = torch.empty((grad_averages.shape[0], 96), dtype=grad_averages.dtype, device=grad_averages.device)
    if result.numel():
        _backward_values[(triton.cdiv(result.numel(), 256),)](
            grad_averages,
            positions,
            *weights,
            result,
            result.numel(),
            *grad_averages.stride(),
            *(stride for t in weights for stride in t.stride()),
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result
