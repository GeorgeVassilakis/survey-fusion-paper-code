"""Fused eigen_mog pair evaluation with analytic geometry gradients."""
from __future__ import annotations

import torch

from .eigen_mog import eigen_mog_pair_values

try:  # pragma: no cover - CPU-only import guard
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

if tl is not None:
    INV_TWO_PI = tl.constexpr(0.15915494309189535)


if triton is not None:
    @triton.jit
    def _forward_kernel(
        dnx_ptr, dny_ptr, cxx_ptr, cxy_ptr, cyy_ptr, point_ptr,
        w_ptr, mx_ptr, my_ptr, sxx_ptr, sxy_ptr, syy_ptr, out_ptr,
        n_pairs, n_components, table_stride,
        BLOCK: tl.constexpr, CBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_pairs
        dnx = tl.load(dnx_ptr + offsets, mask=mask, other=0.0)
        dny = tl.load(dny_ptr + offsets, mask=mask, other=0.0)
        cxx = tl.load(cxx_ptr + offsets, mask=mask, other=1.0)
        cxy = tl.load(cxy_ptr + offsets, mask=mask, other=0.0)
        cyy = tl.load(cyy_ptr + offsets, mask=mask, other=1.0)
        point = tl.load(point_ptr + offsets, mask=mask, other=0)
        rows = point.to(tl.int64) * table_stride
        acc = tl.zeros([BLOCK], dtype=dnx.dtype)
        for start in range(0, n_components, CBLOCK):
            cidx = start + tl.arange(0, CBLOCK)
            table_mask = mask[:, None] & (cidx < n_components)[None, :]
            address = rows[:, None] + cidx[None, :]
            w = tl.load(w_ptr + address, mask=table_mask, other=0.0)
            mx = tl.load(mx_ptr + address, mask=table_mask, other=0.0)
            my = tl.load(my_ptr + address, mask=table_mask, other=0.0)
            sxx = tl.load(sxx_ptr + address, mask=table_mask, other=1.0)
            sxy = tl.load(sxy_ptr + address, mask=table_mask, other=0.0)
            syy = tl.load(syy_ptr + address, mask=table_mask, other=1.0)
            ex = dnx[:, None] - mx
            ey = dny[:, None] - my
            a = cxx[:, None] + sxx
            b = cxy[:, None] + sxy
            d = cyy[:, None] + syy
            determinant = a * d - b * b
            quadratic = (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / determinant
            density = tl.exp(-0.5 * quadratic) * (INV_TWO_PI / tl.sqrt(determinant))
            acc += tl.sum(w * density, axis=1)
        tl.store(out_ptr + offsets, acc, mask=mask)

    @triton.jit
    def _backward_kernel(
        dnx_ptr, dny_ptr, cxx_ptr, cxy_ptr, cyy_ptr, point_ptr,
        w_ptr, mx_ptr, my_ptr, sxx_ptr, sxy_ptr, syy_ptr, gout_ptr,
        gdnx_ptr, gdny_ptr, gcxx_ptr, gcxy_ptr, gcyy_ptr,
        n_pairs, n_components, table_stride,
        BLOCK: tl.constexpr, CBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_pairs
        dnx = tl.load(dnx_ptr + offsets, mask=mask, other=0.0)
        dny = tl.load(dny_ptr + offsets, mask=mask, other=0.0)
        cxx = tl.load(cxx_ptr + offsets, mask=mask, other=1.0)
        cxy = tl.load(cxy_ptr + offsets, mask=mask, other=0.0)
        cyy = tl.load(cyy_ptr + offsets, mask=mask, other=1.0)
        gout = tl.load(gout_ptr + offsets, mask=mask, other=0.0)
        point = tl.load(point_ptr + offsets, mask=mask, other=0)
        rows = point.to(tl.int64) * table_stride
        gdnx = tl.zeros([BLOCK], dtype=dnx.dtype)
        gdny = tl.zeros([BLOCK], dtype=dnx.dtype)
        gcxx = tl.zeros([BLOCK], dtype=dnx.dtype)
        gcxy = tl.zeros([BLOCK], dtype=dnx.dtype)
        gcyy = tl.zeros([BLOCK], dtype=dnx.dtype)
        for start in range(0, n_components, CBLOCK):
            cidx = start + tl.arange(0, CBLOCK)
            table_mask = mask[:, None] & (cidx < n_components)[None, :]
            address = rows[:, None] + cidx[None, :]
            w = tl.load(w_ptr + address, mask=table_mask, other=0.0)
            mx = tl.load(mx_ptr + address, mask=table_mask, other=0.0)
            my = tl.load(my_ptr + address, mask=table_mask, other=0.0)
            sxx = tl.load(sxx_ptr + address, mask=table_mask, other=1.0)
            sxy = tl.load(sxy_ptr + address, mask=table_mask, other=0.0)
            syy = tl.load(syy_ptr + address, mask=table_mask, other=1.0)
            ex = dnx[:, None] - mx
            ey = dny[:, None] - my
            a = cxx[:, None] + sxx
            b = cxy[:, None] + sxy
            d = cyy[:, None] + syy
            determinant = a * d - b * b
            inverse_determinant = 1.0 / determinant
            quadratic = (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) * inverse_determinant
            weighted = w * tl.exp(-0.5 * quadratic) * (INV_TWO_PI / tl.sqrt(determinant)) * gout[:, None]
            gdnx += tl.sum(weighted * (-(d * ex - b * ey) * inverse_determinant), axis=1)
            gdny += tl.sum(weighted * (-(a * ey - b * ex) * inverse_determinant), axis=1)
            gcxx += tl.sum(weighted * (-0.5 * ((ey * ey - quadratic * d) * inverse_determinant + d * inverse_determinant)), axis=1)
            gcyy += tl.sum(weighted * (-0.5 * ((ex * ex - quadratic * a) * inverse_determinant + a * inverse_determinant)), axis=1)
            gcxy += tl.sum(weighted * ((ex * ey - quadratic * b) * inverse_determinant + b * inverse_determinant), axis=1)
        tl.store(gdnx_ptr + offsets, gdnx, mask=mask)
        tl.store(gdny_ptr + offsets, gdny, mask=mask)
        tl.store(gcxx_ptr + offsets, gcxx, mask=mask)
        tl.store(gcxy_ptr + offsets, gcxy, mask=mask)
        tl.store(gcyy_ptr + offsets, gcyy, mask=mask)


class _EigenMogPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dnx, dny, cxx, cxy, cyy, point, *tables):
        n_pairs = dnx.numel()
        output = torch.empty_like(dnx)
        if n_pairs:
            grid = (triton.cdiv(n_pairs, 128),)
            _forward_kernel[grid](
                dnx, dny, cxx, cxy, cyy, point, *tables, output,
                n_pairs, int(tables[0].shape[1]), int(tables[0].stride(0)),
                BLOCK=128, CBLOCK=64,
            )
        ctx.save_for_backward(dnx, dny, cxx, cxy, cyy, point, *tables)
        return output

    @staticmethod
    def backward(ctx, gout):
        dnx, dny, cxx, cxy, cyy, point, *tables = ctx.saved_tensors
        n_pairs = dnx.numel()
        gradients = [torch.empty_like(dnx) for _ in range(5)]
        if n_pairs:
            grid = (triton.cdiv(n_pairs, 128),)
            _backward_kernel[grid](
                dnx, dny, cxx, cxy, cyy, point, *tables, gout.contiguous(), *gradients,
                n_pairs, int(tables[0].shape[1]), int(tables[0].stride(0)),
                BLOCK=128, CBLOCK=64,
            )
        return (*gradients, None, None, None, None, None, None, None)


def eigen_mog_pairs(
    dnx, dny, cxx, cxy, cyy, point,
    weights, mx, my, sxx, sxy, syy,
    *, implementation="auto",
):
    """Evaluate all 40 per-pixel components for each source-pixel pair."""
    tables = (weights, mx, my, sxx, sxy, syy)
    dtype = dnx.dtype
    if any(tensor.dtype != dtype for tensor in (dnx, dny, cxx, cxy, cyy, *tables)):
        raise ValueError("eigen_mog inputs must share one floating dtype")
    if dnx.ndim != 1 or any(tensor.shape != dnx.shape for tensor in (dny, cxx, cxy, cyy, point)):
        raise ValueError("Pair geometry tensors must be one-dimensional and aligned")
    if any(table.ndim != 2 or table.shape != weights.shape for table in tables):
        raise ValueError("Per-pixel tables must have one common (P, C) shape")
    if point.dtype not in (torch.int32, torch.int64):
        raise ValueError("point indices must be integer")
    use_triton = implementation == "triton" or (implementation == "auto" and dnx.is_cuda)
    if use_triton:
        if triton is None or not dnx.is_cuda:
            raise ValueError("The Triton path requires CUDA tensors and a Triton install")
        geometry = [tensor.contiguous() for tensor in (dnx, dny, cxx, cxy, cyy)]
        tables = [table.contiguous() for table in tables]
        return _EigenMogPairs.apply(*geometry, point.contiguous(), *tables)
    if implementation not in ("auto", "torch"):
        raise ValueError(f"Unknown implementation {implementation!r}")
    gathered = [table[point] for table in tables]
    return eigen_mog_pair_values(dnx, dny, cxx, cxy, cyy, *gathered)
