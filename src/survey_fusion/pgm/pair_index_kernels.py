"""Fuse exact pair-support arithmetic while preserving ordered compaction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .raster_pair_kernels import _add, _div, _mul, _sub


@triton.jit
def _prune_kernel(
    pp_ptr,
    ps_ptr,
    x_ptr,
    y_ptr,
    mu_ptr,
    a_ptr,
    b_ptr,
    d_ptr,
    c_ptr,
    variance_ptr,
    keep_ptr,
    q_ptr,
    count,
    scalar_variance,
    threshold,
    SX: tl.constexpr,
    SY: tl.constexpr,
    SM0: tl.constexpr,
    SM1: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    SD: tl.constexpr,
    SC: tl.constexpr,
    SV: tl.constexpr,
    TENSOR_VARIANCE: tl.constexpr,
    HAS_C: tl.constexpr,
    SAVE_Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < count
    pp = tl.load(pp_ptr + i, mask=mask, other=0)
    ps = tl.load(ps_ptr + i, mask=mask, other=0)
    if TENSOR_VARIANCE:
        variance = tl.load(variance_ptr + pp * SV, mask=mask, other=0.0)
    else:
        variance = scalar_variance.to(tl.float32)
    aa = _add(tl.load(a_ptr + ps * SA, mask=mask, other=0.0), variance)
    bb = tl.load(b_ptr + ps * SB, mask=mask, other=0.0)
    dd = _add(tl.load(d_ptr + ps * SD, mask=mask, other=0.0), variance)
    if HAS_C:
        cc = tl.load(c_ptr + ps * SC, mask=mask, other=0.0)
        covariance_product = _mul(bb, cc)
        cross = _add(bb, cc)
    else:
        covariance_product = _mul(bb, bb)
        cross = _mul(2.0, bb)
    determinant = tl.maximum(_sub(_mul(aa, dd), covariance_product), 1.0e-30)
    dx = _sub(
        tl.load(x_ptr + pp * SX, mask=mask, other=0.0), tl.load(mu_ptr + ps * SM0, mask=mask, other=0.0)
    )
    dy = _sub(
        tl.load(y_ptr + pp * SY, mask=mask, other=0.0), tl.load(mu_ptr + ps * SM0 + SM1, mask=mask, other=0.0)
    )
    numerator = _add(_sub(_mul(dd, _mul(dx, dx)), _mul(_mul(cross, dx), dy)), _mul(aa, _mul(dy, dy)))
    q = _div(numerator, determinant)
    tl.store(keep_ptr + i, q <= threshold.to(tl.float32), mask=mask)
    if SAVE_Q:
        tl.store(q_ptr + i, q, mask=mask)


def applies(pair_point, pair_splat, x, y, mu, a, b, d, point_variance, c):
    tensors = (x, y, mu, a, b, d) + (() if c is None else (c,))
    return (
        pair_point.is_cuda
        and pair_point.is_contiguous()
        and pair_splat.is_contiguous()
        and pair_point.dtype == pair_splat.dtype == torch.int64
        and all(t.is_cuda and t.dtype == torch.float32 for t in tensors)
        and (
            not torch.is_tensor(point_variance)
            or (point_variance.is_cuda and point_variance.dtype == torch.float32 and point_variance.ndim == 1)
        )
    )


def support_mask(
    pair_point, pair_splat, x, y, mu, a, b, d, point_variance, n_sigma, c=None, *, return_q=False
):
    tensor_variance = torch.is_tensor(point_variance)
    variance = point_variance if tensor_variance else a
    cc = c if c is not None else b
    mask = torch.empty(pair_point.numel(), dtype=torch.bool, device=mu.device)
    q = torch.empty_like(pair_point, dtype=mu.dtype) if return_q else mask
    if pair_point.numel():
        _prune_kernel[(triton.cdiv(pair_point.numel(), 256),)](
            pair_point,
            pair_splat,
            x,
            y,
            mu,
            a,
            b,
            d,
            cc,
            variance,
            mask,
            q,
            pair_point.numel(),
            0.0 if tensor_variance else float(point_variance),
            float(n_sigma) ** 2,
            x.stride(0),
            y.stride(0),
            mu.stride(0),
            mu.stride(1),
            a.stride(0),
            b.stride(0),
            d.stride(0),
            cc.stride(0),
            variance.stride(0),
            tensor_variance,
            c is not None,
            return_q,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (mask, q) if return_q else mask


def prune_pairs(*args, **kwargs):
    pair_point, pair_splat = args[:2]
    keep = torch.nonzero(support_mask(*args, **kwargs), as_tuple=True)[0]
    return pair_point.index_select(0, keep), pair_splat.index_select(0, keep)


@triton.jit
def _spatial_keep(
    mu,
    radius,
    radius_y,
    result,
    count,
    xmin,
    xmax,
    ymin,
    ymax,
    SM0: tl.constexpr,
    SM1: tl.constexpr,
    SR: tl.constexpr,
    SRY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    mx = tl.load(mu + i * SM0, mask=valid, other=0.0)
    my = tl.load(mu + i * SM0 + SM1, mask=valid, other=0.0)
    rx = tl.load(radius + i * SR, mask=valid, other=0.0)
    ry = tl.load(radius_y + i * SRY, mask=valid, other=0.0)
    keep = (
        (_add(mx, rx) >= xmin.to(tl.float32))
        & (_sub(mx, rx) <= xmax.to(tl.float32))
        & (_add(my, ry) >= ymin.to(tl.float32))
        & (_sub(my, ry) <= ymax.to(tl.float32))
    )
    tl.store(result + i, keep, mask=valid)


@triton.jit
def _tile_bounds(
    mu,
    radius,
    radius_y,
    selected,
    origins,
    widths,
    counts,
    count,
    xmin,
    ymin,
    inv_tile,
    nx,
    ny,
    SM0: tl.constexpr,
    SM1: tl.constexpr,
    SR: tl.constexpr,
    SRY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    s = tl.load(selected + i, mask=valid, other=0)
    mx = tl.load(mu + s * SM0, mask=valid, other=0.0)
    my = tl.load(mu + s * SM0 + SM1, mask=valid, other=0.0)
    rx = tl.load(radius + s * SR, mask=valid, other=0.0)
    ry = tl.load(radius_y + s * SRY, mask=valid, other=0.0)
    x0 = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(_sub(mx, rx), xmin.to(tl.float32)), inv_tile.to(tl.float32))), 0),
        nx - 1,
    ).to(tl.int32)
    x1 = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(_add(mx, rx), xmin.to(tl.float32)), inv_tile.to(tl.float32))), 0),
        nx - 1,
    ).to(tl.int32)
    y0 = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(_sub(my, ry), ymin.to(tl.float32)), inv_tile.to(tl.float32))), 0),
        ny - 1,
    ).to(tl.int32)
    y1 = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(_add(my, ry), ymin.to(tl.float32)), inv_tile.to(tl.float32))), 0),
        ny - 1,
    ).to(tl.int32)
    cx = x1 - x0 + 1
    cy = y1 - y0 + 1
    tl.store(origins + i, y0 * nx + x0, mask=valid)
    tl.store(widths + i, cx, mask=valid)
    tl.store(counts + i, cx.to(tl.int64) * cy, mask=valid)


@triton.jit
def _upper_bound(ends, value, count, NBITS: tl.constexpr):
    lower = tl.full(value.shape, 0, tl.int32)
    upper = tl.full(value.shape, count, tl.int32)
    for _ in tl.static_range(NBITS):
        mid = (lower + upper) // 2
        end = tl.load(ends + mid, mask=(lower < upper) & (mid < count), other=9223372036854775807)
        right = end <= value
        lower = tl.where((lower < upper) & right, mid + 1, lower)
        upper = tl.where((lower < upper) & ~right, mid, upper)
    return lower


@triton.jit
def _duplicate_tiles(
    ends,
    counts,
    origins,
    widths,
    selected,
    tile_ids,
    splat_ids,
    duplicates,
    selected_count,
    nx,
    NBITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = slot < duplicates
    rep = _upper_bound(ends, slot, selected_count, NBITS)
    end = tl.load(ends + rep, mask=valid, other=0)
    count = tl.load(counts + rep, mask=valid, other=0)
    within = (slot - (end - count)).to(tl.int32)
    cx = tl.load(widths + rep, mask=valid, other=1)
    origin = tl.load(origins + rep, mask=valid, other=0)
    tid = origin + (within // cx) * nx + within % cx
    splat = tl.load(selected + rep, mask=valid, other=0)
    tl.store(tile_ids + slot, tid, mask=valid)
    tl.store(splat_ids + slot, splat, mask=valid)


@triton.jit
def _point_counts(
    x,
    y,
    tile_counts,
    ptid,
    counts,
    size,
    xmin,
    ymin,
    inv_tile,
    nx,
    ny,
    SX: tl.constexpr,
    SY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = i < size
    px = tl.load(x + i * SX, mask=keep, other=0.0)
    py = tl.load(y + i * SY, mask=keep, other=0.0)
    tx = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(px, xmin.to(tl.float32)), inv_tile.to(tl.float32))), 0), nx - 1
    ).to(tl.int32)
    ty = tl.minimum(
        tl.maximum(tl.floor(_mul(_sub(py, ymin.to(tl.float32)), inv_tile.to(tl.float32))), 0), ny - 1
    ).to(tl.int32)
    tid = ty * nx + tx
    count = tl.load(tile_counts + tid, mask=keep, other=0)
    tl.store(ptid + i, tid, mask=keep)
    tl.store(counts + i, count, mask=keep)


@triton.jit
def _emit_pairs(
    ends, bases, splats, pair_point, pair_splat, total, n_points, NBITS: tl.constexpr, BLOCK: tl.constexpr
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = slot < total
    point = _upper_bound(ends, slot, n_points, NBITS)
    base = tl.load(bases + point, mask=valid, other=0)
    splat = tl.load(splats + base + slot, mask=valid, other=0)
    tl.store(pair_point + slot, point, mask=valid)
    tl.store(pair_splat + slot, splat, mask=valid)


def build_pair_index(mu, radius, x, y, duplication_budget=32_000_000, max_tiles_per_axis=2048, radius_y=None):
    """Keep the reference tile grid, stable ordering and every integer census."""
    import numpy as np

    device = mu.device
    n = x.numel()
    if n == 0 or mu.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    mu, radius = mu.detach(), radius.detach()
    radius_y = radius if radius_y is None else radius_y.detach()
    xmin, xmax, ymin, ymax, median = torch.stack(
        (x.min(), x.max(), y.min(), y.max(), radius.median().to(dtype=x.dtype))
    ).tolist()
    extent = max(xmax - xmin, ymax - ymin, 1e-12)
    keep = torch.empty(mu.shape[0], dtype=torch.bool, device=device)
    _spatial_keep[(triton.cdiv(mu.shape[0], 256),)](
        mu,
        radius,
        radius_y,
        keep,
        mu.shape[0],
        xmin,
        xmax,
        ymin,
        ymax,
        *mu.stride(),
        radius.stride(0),
        radius_y.stride(0),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    selected = torch.nonzero(keep, as_tuple=True)[0]
    count = selected.numel()
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    origins = torch.empty(count, dtype=torch.int32, device=device)
    widths = torch.empty_like(origins)
    counts = torch.empty(count, dtype=torch.long, device=device)
    tile = max(2.0 * median, extent / max_tiles_per_axis, 1e-9)
    while True:
        nx, ny = int((xmax - xmin) / tile) + 1, int((ymax - ymin) / tile) + 1
        inv_tile = float(np.float32(1.0 / tile))
        _tile_bounds[(triton.cdiv(count, 256),)](
            mu,
            radius,
            radius_y,
            selected,
            origins,
            widths,
            counts,
            count,
            xmin,
            ymin,
            inv_tile,
            nx,
            ny,
            *mu.stride(),
            radius.stride(0),
            radius_y.stride(0),
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        duplicates = int(counts.sum())
        if duplicates <= duplication_budget or nx * ny <= 4:
            break
        tile *= 2.0
    ends = counts.cumsum(0)
    key_dtype = torch.int16 if nx * ny <= 32767 else torch.int32
    tile_ids = torch.empty(duplicates, dtype=key_dtype, device=device)
    splat_ids = torch.empty(duplicates, dtype=torch.int32, device=device)
    _duplicate_tiles[(triton.cdiv(duplicates, 256),)](
        ends,
        counts,
        origins,
        widths,
        selected,
        tile_ids,
        splat_ids,
        duplicates,
        count,
        nx,
        NBITS=count.bit_length(),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    sorted_ids, order = torch.sort(tile_ids, stable=True)
    splat_sorted = splat_ids.index_select(0, order)
    tile_offsets = torch.searchsorted(sorted_ids, torch.arange(nx * ny + 1, device=device, dtype=key_dtype))
    tile_counts = tile_offsets[1:] - tile_offsets[:-1]
    tile_offsets = tile_offsets[:-1]
    ptid = torch.empty(n, dtype=torch.long, device=device)
    cnt_p = torch.empty_like(ptid)
    _point_counts[(triton.cdiv(n, 256),)](
        x,
        y,
        tile_counts,
        ptid,
        cnt_p,
        n,
        xmin,
        ymin,
        inv_tile,
        nx,
        ny,
        x.stride(0),
        y.stride(0),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    point_ends = cnt_p.cumsum(0)
    total = int(point_ends[-1])
    if total >= 2**31:
        from .splat_raster import _build_pair_index_eager

        return _build_pair_index_eager(mu, radius, x, y, duplication_budget, max_tiles_per_axis, radius_y)
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    base = tile_offsets[ptid] - (point_ends - cnt_p)
    pair_point = torch.empty(total, dtype=torch.long, device=device)
    pair_splat = torch.empty_like(pair_point)
    _emit_pairs[(triton.cdiv(total, 256),)](
        point_ends,
        base,
        splat_sorted,
        pair_point,
        pair_splat,
        total,
        n,
        NBITS=n.bit_length(),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return pair_point, pair_splat
