"""Exact coalescence of two sorted-unique sparse gather blocks without float atomics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mark(first, second, positions, mask, anchor, N0, N1, CENSUS, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(first + i, i < N0, 0).to(tl.int64)
    b = tl.load(second + i, i < N1, 0).to(tl.int64)
    tl.store(positions + a, i, i < N0)
    tl.store(positions + CENSUS + b, i, i < N1)
    tl.store(mask + a, 1, i < N0)
    tl.store(mask + b, 1, i < N1)
    n = tl.minimum(N0, N1)
    different = (i < n) & (a != b)
    earliest = tl.min(tl.where(different, i, n), 0).to(tl.int64)
    tl.atomic_min(anchor, earliest, sem="relaxed")


@triton.jit
def _assemble(
    values,
    actual_indices,
    first,
    second,
    rows,
    positions,
    anchor,
    out,
    N0,
    N1,
    CENSUS,
    NROWS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // WIDTH, offset % WIDTH
    valid = row < NROWS
    source = tl.load(rows + row, valid, 0).to(tl.int64)
    p0 = tl.load(positions + source, valid, -1).to(tl.int64)
    p1 = tl.load(positions + CENSUS + source, valid, -1).to(tl.int64)
    at = tl.load(anchor)
    n = tl.minimum(N0, N1)
    if at < n:
        expected = tl.load(first + at)
        first_then_second = tl.load(actual_indices + at) == expected
    elif N0 == N1:
        first_then_second = True
    elif N0 > N1:
        first_then_second = tl.load(actual_indices + N1) == tl.load(first + N1)
    else:
        first_then_second = tl.load(actual_indices + N0) == tl.load(second)
    a = tl.where(first_then_second, p0, p0 + N1)
    b = tl.where(first_then_second, p1 + N0, p1)
    va = tl.load(values + a * WIDTH + col, valid & (p0 >= 0), 0.0)
    vb = tl.load(values + b * WIDTH + col, valid & (p1 >= 0), 0.0)
    tl.store(out + offset, va + vb, valid)


def coalesce_pair(gathered, parameters):
    """Return False for unsupported layouts so the established path can handle them."""
    if len(gathered) != 2 or not all(
        i.is_cuda and i.dtype == torch.long and i.is_contiguous() for i in gathered
    ):
        return False
    n0, n1 = map(lambda i: i.numel(), gathered)
    if not min(n0, n1) or max(n0, n1) >= 2**31:
        return False
    if not all(
        p.grad.dtype == torch.float32
        and p.grad._values().is_contiguous()
        and p.grad._indices().is_contiguous()
        for p in parameters
    ):
        return False
    census = parameters[0].shape[0]
    device = parameters[0].device
    positions = torch.full((2, census), -1, device=device, dtype=torch.int32)
    mask = torch.zeros(census, device=device, dtype=torch.bool)
    anchor = torch.full((), min(n0, n1), device=device, dtype=torch.long)
    _mark[(triton.cdiv(max(n0, n1), 256),)](*gathered, positions, mask, anchor, n0, n1, census, 256)
    rows = mask.nonzero(as_tuple=True)[0]
    for parameter in parameters:
        grad = parameter.grad
        values = grad._values()
        width = values.numel() // values.shape[0]
        dense = values.new_empty((rows.numel(),) + values.shape[1:])
        _assemble[(triton.cdiv(dense.numel(), 256),)](
            values,
            grad._indices()[0],
            *gathered,
            rows,
            positions,
            anchor,
            dense,
            n0,
            n1,
            census,
            rows.numel(),
            width,
            256,
            enable_fp_fusion=False,
        )
        parameter.grad = torch.sparse_coo_tensor(
            rows[None],
            dense,
            grad.shape,
            device=device,
            dtype=values.dtype,
            is_coalesced=True,
            check_invariants=False,
        )
    return True
