"""Sparse Triton contraction of per-band spectral-weight blocks.

The forward dot product visits knots in ascending order with compensated fp32
summation and rounds every operation separately. The spectrum gradient visits
bands in ascending order with the same scheme. This makes the sparse result
stable across runs and closer to a float64 control than the cuBLAS reduction,
but not bitwise equal to the dense path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_add = libdevice.add_rn
_mul = libdevice.mul_rn
_sub = libdevice.sub_rn

_BLOCK_PAIRS = 128
_BLOCK_GRAD = 256
_MAX_BANDS = 5
_INDEX_LIMIT = 2 ** 31


@triton.jit(
    do_not_specialize=["n_elem"],
    do_not_specialize_on_alignment=["n_elem"],
)
def _init_kernel(averages_ptr, positions_ptr, n_elem, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offsets < n_elem
    tl.store(averages_ptr + offsets, 0.0, mask=keep)
    tl.store(positions_ptr + offsets, -1, mask=keep)


@triton.jit(
    do_not_specialize=["n0", "n1", "n2", "n3", "n4", "n_pairs"],
    do_not_specialize_on_alignment=["n0", "n1", "n2", "n3", "n4", "n_pairs"],
)
def _forward_kernel(
    values_ptr,
    rows0_ptr, rows1_ptr, rows2_ptr, rows3_ptr, rows4_ptr,
    weights0_ptr, weights1_ptr, weights2_ptr, weights3_ptr, weights4_ptr,
    averages_ptr, positions_ptr,
    n0, n1, n2, n3, n4, n_pairs,
    values_row_stride, values_knot_stride,
    rows0_stride, rows1_stride, rows2_stride, rows3_stride, rows4_stride,
    weights0_row_stride, weights0_knot_stride,
    weights1_row_stride, weights1_knot_stride,
    weights2_row_stride, weights2_knot_stride,
    weights3_row_stride, weights3_knot_stride,
    weights4_row_stride, weights4_knot_stride,
    N_BANDS: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = pair < n_pairs
    end0 = n0
    end1 = end0 + n1
    end2 = end1 + n2
    end3 = end2 + n3
    band = tl.where(
        pair < end0, 0,
        tl.where(pair < end1, 1, tl.where(pair < end2, 2, tl.where(pair < end3, 3, 4))),
    )
    start = tl.where(
        band == 0, 0,
        tl.where(band == 1, end0, tl.where(band == 2, end1, tl.where(band == 3, end2, end3))),
    )
    local = pair - start

    row = tl.load(rows0_ptr + local * rows0_stride, mask=keep & (band == 0), other=0)
    row = tl.where(
        band == 1,
        tl.load(rows1_ptr + local * rows1_stride, mask=keep & (band == 1), other=0),
        row,
    )
    row = tl.where(
        band == 2,
        tl.load(rows2_ptr + local * rows2_stride, mask=keep & (band == 2), other=0),
        row,
    )
    row = tl.where(
        band == 3,
        tl.load(rows3_ptr + local * rows3_stride, mask=keep & (band == 3), other=0),
        row,
    )
    row = tl.where(
        band == 4,
        tl.load(rows4_ptr + local * rows4_stride, mask=keep & (band == 4), other=0),
        row,
    )

    total = tl.full(pair.shape, 0.0, tl.float32)
    compensation = tl.full(pair.shape, 0.0, tl.float32)
    for knot in tl.static_range(0, G):
        weight = tl.load(
            weights0_ptr + local * weights0_row_stride + knot * weights0_knot_stride,
            mask=keep & (band == 0), other=0.0,
        )
        weight = tl.where(
            band == 1,
            tl.load(
                weights1_ptr + local * weights1_row_stride + knot * weights1_knot_stride,
                mask=keep & (band == 1), other=0.0,
            ),
            weight,
        )
        weight = tl.where(
            band == 2,
            tl.load(
                weights2_ptr + local * weights2_row_stride + knot * weights2_knot_stride,
                mask=keep & (band == 2), other=0.0,
            ),
            weight,
        )
        weight = tl.where(
            band == 3,
            tl.load(
                weights3_ptr + local * weights3_row_stride + knot * weights3_knot_stride,
                mask=keep & (band == 3), other=0.0,
            ),
            weight,
        )
        weight = tl.where(
            band == 4,
            tl.load(
                weights4_ptr + local * weights4_row_stride + knot * weights4_knot_stride,
                mask=keep & (band == 4), other=0.0,
            ),
            weight,
        )
        value = tl.load(
            values_ptr + row * values_row_stride + knot * values_knot_stride,
            mask=keep,
            other=0.0,
        )
        adjusted = _sub(_mul(value, weight), compensation)
        next_total = _add(total, adjusted)
        compensation = _sub(_sub(next_total, total), adjusted)
        total = next_total

    output_offset = row * N_BANDS + band
    tl.store(averages_ptr + output_offset, total, mask=keep)
    tl.store(positions_ptr + output_offset, local, mask=keep)


@triton.jit(
    do_not_specialize=["n_elem"],
    do_not_specialize_on_alignment=["n_elem"],
)
def _backward_values_kernel(
    grad_averages_ptr, positions_ptr,
    weights0_ptr, weights1_ptr, weights2_ptr, weights3_ptr, weights4_ptr,
    grad_values_ptr, n_elem,
    grad_row_stride, grad_band_stride,
    weights0_row_stride, weights0_knot_stride,
    weights1_row_stride, weights1_knot_stride,
    weights2_row_stride, weights2_knot_stride,
    weights3_row_stride, weights3_knot_stride,
    weights4_row_stride, weights4_knot_stride,
    N_BANDS: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offset < n_elem
    row = offset // G
    knot = offset - row * G
    total = tl.full(offset.shape, 0.0, tl.float32)
    compensation = tl.full(offset.shape, 0.0, tl.float32)

    position = tl.load(positions_ptr + row * N_BANDS, mask=keep, other=-1)
    present = position >= 0
    term = _mul(
        tl.load(grad_averages_ptr + row * grad_row_stride, mask=keep & present, other=0.0),
        tl.load(
            weights0_ptr + position * weights0_row_stride + knot * weights0_knot_stride,
            mask=keep & present, other=0.0,
        ),
    )
    adjusted = _sub(term, compensation)
    next_total = _add(total, adjusted)
    compensation = tl.where(present, _sub(_sub(next_total, total), adjusted), compensation)
    total = tl.where(present, next_total, total)

    if N_BANDS > 1:
        position = tl.load(positions_ptr + row * N_BANDS + 1, mask=keep, other=-1)
        present = position >= 0
        term = _mul(
            tl.load(
                grad_averages_ptr + row * grad_row_stride + grad_band_stride,
                mask=keep & present, other=0.0,
            ),
            tl.load(
                weights1_ptr + position * weights1_row_stride + knot * weights1_knot_stride,
                mask=keep & present, other=0.0,
            ),
        )
        adjusted = _sub(term, compensation)
        next_total = _add(total, adjusted)
        compensation = tl.where(present, _sub(_sub(next_total, total), adjusted), compensation)
        total = tl.where(present, next_total, total)
    if N_BANDS > 2:
        position = tl.load(positions_ptr + row * N_BANDS + 2, mask=keep, other=-1)
        present = position >= 0
        term = _mul(
            tl.load(
                grad_averages_ptr + row * grad_row_stride + 2 * grad_band_stride,
                mask=keep & present, other=0.0,
            ),
            tl.load(
                weights2_ptr + position * weights2_row_stride + knot * weights2_knot_stride,
                mask=keep & present, other=0.0,
            ),
        )
        adjusted = _sub(term, compensation)
        next_total = _add(total, adjusted)
        compensation = tl.where(present, _sub(_sub(next_total, total), adjusted), compensation)
        total = tl.where(present, next_total, total)
    if N_BANDS > 3:
        position = tl.load(positions_ptr + row * N_BANDS + 3, mask=keep, other=-1)
        present = position >= 0
        term = _mul(
            tl.load(
                grad_averages_ptr + row * grad_row_stride + 3 * grad_band_stride,
                mask=keep & present, other=0.0,
            ),
            tl.load(
                weights3_ptr + position * weights3_row_stride + knot * weights3_knot_stride,
                mask=keep & present, other=0.0,
            ),
        )
        adjusted = _sub(term, compensation)
        next_total = _add(total, adjusted)
        compensation = tl.where(present, _sub(_sub(next_total, total), adjusted), compensation)
        total = tl.where(present, next_total, total)
    if N_BANDS > 4:
        position = tl.load(positions_ptr + row * N_BANDS + 4, mask=keep, other=-1)
        present = position >= 0
        term = _mul(
            tl.load(
                grad_averages_ptr + row * grad_row_stride + 4 * grad_band_stride,
                mask=keep & present, other=0.0,
            ),
            tl.load(
                weights4_ptr + position * weights4_row_stride + knot * weights4_knot_stride,
                mask=keep & present, other=0.0,
            ),
        )
        adjusted = _sub(term, compensation)
        next_total = _add(total, adjusted)
        total = tl.where(present, next_total, total)

    tl.store(grad_values_ptr + offset, total, mask=keep)


def applies(values: torch.Tensor, n_bands: int, placements, band_weights) -> bool:
    """Whether the sparse CUDA kernel supports this contraction."""
    rows, knots = values.shape
    pair_count = sum(int(block.shape[0]) for block in band_weights)
    return (
        values.is_cuda
        and values.dtype is torch.float32
        and 0 < n_bands <= _MAX_BANDS
        and rows * n_bands < _INDEX_LIMIT
        and rows * knots < _INDEX_LIMIT
        and pair_count < _INDEX_LIMIT
        and all(rows_tensor.is_cuda and rows_tensor.dtype is torch.long for _, rows_tensor in placements)
        and all(block.is_cuda and block.dtype is torch.float32 for block in band_weights)
    )


def _by_band(n_bands, placements, band_weights):
    first_rows = placements[0][1]
    first_weights = band_weights[0]
    rows = [first_rows] * _MAX_BANDS
    weights = [first_weights] * _MAX_BANDS
    sizes = [0] * _MAX_BANDS
    for (band, band_rows), block in zip(placements, band_weights):
        rows[band] = band_rows
        weights[band] = block
        sizes[band] = int(block.shape[0])
    return rows, weights, sizes


def forward(values, n_bands, placements, band_weights):
    """Return sparse averages and the compact row-to-block-position lookup."""
    row_count, knots = values.shape
    averages = torch.empty((row_count, n_bands), dtype=values.dtype, device=values.device)
    positions = torch.empty((row_count, n_bands), dtype=torch.int32, device=values.device)
    n_output = row_count * n_bands
    if n_output:
        _init_kernel[(triton.cdiv(n_output, _BLOCK_GRAD),)](
            averages, positions, n_output, BLOCK=_BLOCK_GRAD, num_warps=4,
            enable_fp_fusion=False,
        )

    rows, weights, sizes = _by_band(n_bands, placements, band_weights)
    n_pairs = sum(sizes)
    if n_pairs:
        _forward_kernel[(triton.cdiv(n_pairs, _BLOCK_PAIRS),)](
            values, *rows, *weights, averages, positions,
            *sizes, n_pairs,
            values.stride(0), values.stride(1),
            *(tensor.stride(0) for tensor in rows),
            *(stride for tensor in weights for stride in tensor.stride()),
            N_BANDS=n_bands, G=knots, BLOCK=_BLOCK_PAIRS, num_warps=4,
            enable_fp_fusion=False,
        )
    return averages, positions


def backward_values(grad_averages, positions, n_bands, placements, band_weights, knots):
    """Accumulate spectrum gradients in ascending band order."""
    rows, weights, _ = _by_band(n_bands, placements, band_weights)
    del rows
    row_count = int(grad_averages.shape[0])
    grad_values = torch.empty((row_count, knots), dtype=grad_averages.dtype, device=grad_averages.device)
    n_elem = row_count * knots
    if n_elem:
        _backward_values_kernel[(triton.cdiv(n_elem, _BLOCK_GRAD),)](
            grad_averages, positions, *weights, grad_values, n_elem,
            grad_averages.stride(0), grad_averages.stride(1),
            *(stride for tensor in weights for stride in tensor.stride()),
            N_BANDS=n_bands, G=knots, BLOCK=_BLOCK_GRAD, num_warps=4,
            enable_fp_fusion=False,
        )
    return grad_values
