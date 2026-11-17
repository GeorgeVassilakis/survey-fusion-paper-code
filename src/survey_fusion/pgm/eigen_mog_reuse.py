"""Exact mixture evaluation with persistent geometry and reusable derivatives.

The optional grouped path prepares convolution constants once for each distinct
source/exposure/Jacobian combination. Spatial weights remain per landing pixel.
Neither path changes components, precision, support, or the PSF coefficients.
"""
from __future__ import annotations

import torch

from ..utils.perf_env import eigen_block, eigen_cblock

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU reference remains usable.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _prepare(cx, cy, cz, exposure, sx, sy, sz, prepared,
                 N, C: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        g, k = i // C, i % C
        mask = i < N * C
        e = tl.load(exposure + g, mask, 0)
        a = tl.load(cx + g, mask, 0) + tl.load(sx + e * C + k, mask, 1)
        b = tl.load(cy + g, mask, 0) + tl.load(sy + e * C + k, mask, 0)
        d = tl.load(cz + g, mask, 0) + tl.load(sz + e * C + k, mask, 1)
        det = a * d - b * b
        norm = 0.15915494309189535 / tl.sqrt(det)
        tl.store(prepared + i, a, mask)
        tl.store(prepared + N * C + i, b, mask)
        tl.store(prepared + 2 * N * C + i, d, mask)
        tl.store(prepared + 3 * N * C + i, det, mask)
        tl.store(prepared + 4 * N * C + i, norm, mask)

    @triton.jit
    def _evaluate(dx, dy, cx, cy, cz, point, exposure, group,
                  weights, mx, my, sx, sy, sz, prepared, output, derivatives,
                  N, G, C: tl.constexpr,
                  GROUPED: tl.constexpr, BLOCK: tl.constexpr, CBLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        p = tl.load(point + i, mask, 0)
        g = tl.load(group + i, mask, 0) if GROUPED else i
        e = tl.load(exposure + g, mask, 0)
        x = tl.load(dx + i, mask, 0)
        y = tl.load(dy + i, mask, 0)
        if not GROUPED:
            xx = tl.load(cx + i, mask, 1)
            xy = tl.load(cy + i, mask, 0)
            yy = tl.load(cz + i, mask, 1)
        value = tl.full((BLOCK,), 0, x.dtype)
        gx = tl.full((BLOCK,), 0, x.dtype)
        gy = tl.full((BLOCK,), 0, x.dtype)
        gxx = tl.full((BLOCK,), 0, x.dtype)
        gxy = tl.full((BLOCK,), 0, x.dtype)
        gyy = tl.full((BLOCK,), 0, x.dtype)
        for start in range(tl.cdiv(C, CBLOCK)):
            k = start * CBLOCK + tl.arange(0, CBLOCK)
            valid = mask[:, None] & (k[None, :] < C)
            address = e[:, None] * C + k[None, :]
            w = tl.load(weights + p[:, None] * C + k[None, :], valid, 0)
            ex = x[:, None] - tl.load(mx + address, valid, 0)
            ey = y[:, None] - tl.load(my + address, valid, 0)
            if GROUPED:
                at = g[:, None] * C + k[None, :]
                a = tl.load(prepared + at, valid, 1)
                b = tl.load(prepared + G * C + at, valid, 0)
                d = tl.load(prepared + 2 * G * C + at, valid, 1)
                det = tl.load(prepared + 3 * G * C + at, valid, 1)
                norm = tl.load(prepared + 4 * G * C + at, valid, 1)
            else:
                a = xx[:, None] + tl.load(sx + address, valid, 1)
                b = xy[:, None] + tl.load(sy + address, valid, 0)
                d = yy[:, None] + tl.load(sz + address, valid, 1)
                det = a * d - b * b
                norm = 0.15915494309189535 / tl.sqrt(det)
            inv = 1.0 / det
            q = (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / det
            weighted = w * tl.exp(-0.5 * q) * norm
            value += tl.sum(weighted, 1)
            gx += tl.sum(weighted * (-(d * ex - b * ey) * inv), 1)
            gy += tl.sum(weighted * (-(a * ey - b * ex) * inv), 1)
            gxx += tl.sum(weighted * (-0.5 * ((ey * ey - q * d) * inv + d * inv)), 1)
            gyy += tl.sum(weighted * (-0.5 * ((ex * ex - q * a) * inv + a * inv)), 1)
            gxy += tl.sum(weighted * ((ex * ey - q * b) * inv + b * inv), 1)
        tl.store(output + i, value, mask)
        tl.store(derivatives + i, gx, mask)
        tl.store(derivatives + N + i, gy, mask)
        tl.store(derivatives + 2 * N + i, gxx, mask)
        tl.store(derivatives + 3 * N + i, gxy, mask)
        tl.store(derivatives + 4 * N + i, gyy, mask)


class _ReusePairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dx, dy, cx, cy, cz, point, exposure, group, *tables):
        n, c = dx.numel(), tables[0].shape[1]
        grouped = group is not None
        geometry = [v.contiguous() for v in (dx, dy, cx, cy, cz)]
        weights, mx, my, sx, sy, sz = [v.contiguous() for v in tables]
        groups = cx.numel()
        prepared = dx.new_empty((5, groups, c)) if grouped else dx.new_empty(0)
        if n and grouped:
            _prepare[(triton.cdiv(groups * c, 256),)](
                *geometry[2:], exposure, sx, sy, sz, prepared, groups, c, 256)
        derivatives = dx.new_empty((5, n))
        output = torch.empty_like(dx)
        if n:
            block = eigen_block()
            cblock = eigen_cblock()
            _evaluate[(triton.cdiv(n, block),)](
                *geometry, point, exposure, group if grouped else point,
                weights, mx, my, sx, sy, sz, prepared, output, derivatives,
                n, groups, c, grouped, block, cblock)
        ctx.save_for_backward(derivatives, group)
        ctx.groups = groups
        return output

    @staticmethod
    def backward(ctx, gout):
        derivatives, group = ctx.saved_tensors
        grads = derivatives * gout[None, :]
        gx, gy, gxx, gxy, gyy = grads.unbind(0)
        if group is not None:
            combined = grads.new_zeros((3, ctx.groups)).index_add_(1, group, grads[2:])
            gxx, gxy, gyy = combined.unbind(0)
        return gx, gy, gxx, gxy, gyy, None, None, None, None, None, None, None, None, None


def reuse_pairs(dx, dy, cx, cy, cz, point, exposure, tables, *, group=None, implementation="auto"):
    """Evaluate pairs using exposure geometry; optional covariance-group reuse."""
    if dx.is_cuda and implementation != "torch":
        if triton is None:
            raise RuntimeError("Triton is required for CUDA eigen_mog reuse")
        return _ReusePairs.apply(dx, dy, cx, cy, cz, point, exposure, group, *tables)
    from .eigen_mog import eigen_mog_pair_values
    e = exposure if group is None else exposure[group]
    covariance = (cx, cy, cz) if group is None else (cx[group], cy[group], cz[group])
    return eigen_mog_pair_values(dx, dy, *covariance, tables[0][point], *[t[e] for t in tables[1:]])
