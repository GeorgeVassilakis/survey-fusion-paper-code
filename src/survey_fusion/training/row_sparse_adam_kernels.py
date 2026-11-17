"""Bitwise-exact fused CUDA update for :class:`RowSparseAdam`.

Internal arithmetic preserves subnormals. Storage additions match CUDA atomic
index_add (flush-to-zero) in ordinary mode and its ordered non-atomic path in
deterministic mode. This distinction matters for tiny optimizer moments.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _sqrt(a):
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _storage_add(a, b, DETERMINISTIC: tl.constexpr):
    if DETERMINISTIC:
        return _add(a, b)
    # CUDA global atomicAdd flushes subnormal inputs/results; deterministic
    # index_add instead uses the ordered, non-atomic accumulation path.
    return tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


_BLOCK = 256
_INDEX_LIMIT = 2**31


@triton.jit
def _row_sparse_adam_kernel(
    param_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    values_ptr,
    rows_ptr,
    n_elem,
    one_minus_beta1,
    one_minus_beta2,
    eps,
    neg_step_size,
    WIDTH: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = offsets < n_elem
    touched_row = offsets // WIDTH
    column = offsets - touched_row * WIDTH
    dense_row = tl.load(rows_ptr + touched_row, mask=keep, other=0)
    dense_offset = dense_row * WIDTH + column

    one_minus_beta1 = one_minus_beta1.to(tl.float32)
    one_minus_beta2 = one_minus_beta2.to(tl.float32)
    eps = eps.to(tl.float32)
    neg_step_size = neg_step_size.to(tl.float32)

    gradient = tl.load(values_ptr + offsets, mask=keep, other=0.0)
    old_exp_avg = tl.load(exp_avg_ptr + dense_offset, mask=keep, other=0.0)
    exp_avg_update = _mul(_sub(gradient, old_exp_avg), one_minus_beta1)
    exp_avg_new = _storage_add(old_exp_avg, exp_avg_update, DETERMINISTIC)

    old_exp_avg_sq = tl.load(exp_avg_sq_ptr + dense_offset, mask=keep, other=0.0)
    gradient_sq = _mul(gradient, gradient)
    exp_avg_sq_update = _mul(_sub(gradient_sq, old_exp_avg_sq), one_minus_beta2)
    exp_avg_sq_new = _storage_add(old_exp_avg_sq, exp_avg_sq_update, DETERMINISTIC)

    numer = _add(exp_avg_update, old_exp_avg)
    denom = _add(_sqrt(_add(exp_avg_sq_update, old_exp_avg_sq)), eps)
    parameter_update = _mul(neg_step_size, _div(numer, denom))
    parameter = tl.load(param_ptr + dense_offset, mask=keep, other=0.0)

    tl.store(exp_avg_ptr + dense_offset, exp_avg_new, mask=keep)
    tl.store(exp_avg_sq_ptr + dense_offset, exp_avg_sq_new, mask=keep)
    tl.store(param_ptr + dense_offset, _storage_add(parameter, parameter_update, DETERMINISTIC), mask=keep)


def applies(
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    values: torch.Tensor,
    rows: torch.Tensor,
) -> bool:
    """Whether the fused kernel can reproduce the eager update."""
    return (
        param.is_cuda
        and param.dtype is torch.float32
        and exp_avg.dtype is torch.float32
        and exp_avg_sq.dtype is torch.float32
        and values.dtype is torch.float32
        and rows.dtype is torch.int64
        and exp_avg.device == param.device
        and exp_avg_sq.device == param.device
        and values.device == param.device
        and rows.device == param.device
        and param.is_contiguous()
        and exp_avg.is_contiguous()
        and exp_avg_sq.is_contiguous()
        and values.is_contiguous()
        and rows.is_contiguous()
        and param.shape == exp_avg.shape == exp_avg_sq.shape
        and param.ndim in (1, 2)
        and values.ndim == param.ndim
        and values.shape[1:] == param.shape[1:]
        and values.numel() < _INDEX_LIMIT
        and param.numel() < _INDEX_LIMIT
    )


def _float32(value: float) -> float:
    """Round a Python scalar exactly once before passing it to Triton."""
    return float(np.float32(value))


def update_rows_(
    param: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    values: torch.Tensor,
    rows: torch.Tensor,
    *,
    one_minus_beta1: float,
    one_minus_beta2: float,
    eps: float,
    neg_step_size: float,
) -> None:
    """Update unique rows in one launch with eager-equivalent fp32 rounding."""
    n_elem = values.numel()
    width = param.numel() // param.shape[0]
    _row_sparse_adam_kernel[(triton.cdiv(n_elem, _BLOCK),)](
        param,
        exp_avg,
        exp_avg_sq,
        values,
        rows,
        n_elem,
        _float32(one_minus_beta1),
        _float32(one_minus_beta2),
        _float32(eps),
        _float32(neg_step_size),
        WIDTH=width,
        DETERMINISTIC=torch.are_deterministic_algorithms_enabled(),
        BLOCK=_BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
