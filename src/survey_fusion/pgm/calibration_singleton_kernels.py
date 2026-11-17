"""Batched singleton GEMVs with the verified B200/cuBLAS five-band order."""

import os

import torch
import triton
import triton.language as tl

from .raster_pair_kernels import _add, _mul
from ..utils.perf_env import validated_cublas_runtime


def applies(mix, scales):
    return (
        mix.is_cuda
        and mix.dtype == scales.dtype == torch.float32
        and mix.is_contiguous()
        and mix.shape[1] == 5
        and validated_cublas_runtime()
        and torch.cuda.get_device_capability(mix.device) == (10, 0)
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") in ("", ":4096:8")
    )


@triton.jit
def _singleton(mix, scales, rows, out, n, TREE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    row = tl.load(rows + i, valid, 0).to(tl.int64)
    a = tl.load(mix + row * 5, valid, 0)
    b = tl.load(mix + row * 5 + 1, valid, 0)
    c = tl.load(mix + row * 5 + 2, valid, 0)
    d = tl.load(mix + row * 5 + 3, valid, 0)
    e = tl.load(mix + row * 5 + 4, valid, 0)
    s0 = tl.load(scales)
    s1 = tl.load(scales + 1)
    s2 = tl.load(scales + 2)
    s3 = tl.load(scales + 3)
    s4 = tl.load(scales + 4)
    if TREE:
        value = _add(_add(_add(_mul(a, s0), _mul(e, s4)), _mul(c, s2)), _add(_mul(b, s1), _mul(d, s3)))
    else:
        head = tl.fma(d, s3, tl.fma(c, s2, tl.fma(b, s1, _mul(a, s0))))
        value = _add(head, _mul(e, s4))
    tl.store(out + i, value, valid)


def singleton_scales(mix, scales, rows):
    result = scales.new_empty(rows.numel())
    if rows.numel():
        _singleton[(triton.cdiv(rows.numel(), 128),)](
            mix,
            scales,
            rows,
            result,
            rows.numel(),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8",
            128,
        )
    return result
