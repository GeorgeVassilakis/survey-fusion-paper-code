"""Exact elementwise Gaussian aperture quadrature with original ATen sums."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .raster_pair_kernels import _add, _mul, _sub


@triton.jit
def _neg(x):
    return tl.inline_asm_elementwise("neg.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _quadrature(
    px,
    py,
    qw,
    fibers,
    mu,
    p00p,
    p01p,
    p10p,
    p11p,
    normp,
    go,
    result,
    count,
    Q: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < count * Q
    row, node = offset // Q, offset % Q
    fiber = tl.load(fibers + row, mask=valid, other=0)
    dx = _sub(
        tl.load(px + fiber * Q + node, mask=valid, other=0.0), tl.load(mu + row * 2, mask=valid, other=0.0)
    )
    dy = _sub(
        tl.load(py + fiber * Q + node, mask=valid, other=0.0),
        tl.load(mu + row * 2 + 1, mask=valid, other=0.0),
    )
    p00 = tl.load(p00p + row, mask=valid, other=0.0)
    p01 = tl.load(p01p + row, mask=valid, other=0.0)
    p10 = tl.load(p10p + row, mask=valid, other=0.0)
    p11 = tl.load(p11p + row, mask=valid, other=0.0)
    norm = tl.load(normp + row, mask=valid, other=0.0)
    weight = tl.load(qw + fiber * Q + node, mask=valid, other=0.0)
    first = tl.fma(dy, p10, _mul(dx, p00))
    second = tl.fma(dy, p11, _mul(dx, p01))
    mahalanobis = tl.fma(second, dy, _mul(first, dx))
    exponential = libdevice.exp(_mul(mahalanobis, -0.5))
    if not BACKWARD:
        value = _mul(_mul(norm, exponential), weight)
        tl.store(result + offset, value, mask=valid)
    else:
        upstream = tl.load(go + row, mask=valid, other=0.0)
        mul6 = _mul(upstream, weight)
        mul7 = _mul(mul6, norm)
        gnorm = _mul(mul6, exponential)
        mul10 = _mul(_mul(mul7, exponential), -0.5)
        mul12 = _mul(mul10, _mul(dy, 1.0))
        mul14 = _mul(mul10, _mul(second, 1.0))
        mul15 = _mul(mul10, first)
        mul16 = _mul(mul10, dx)
        mul18 = _mul(mul12, _mul(p11, 1.0))
        gp11 = _mul(mul12, _mul(dy, 1.0))
        add0 = _add(mul14, mul18)
        gp01 = _mul(mul12, dx)
        mul22 = _mul(mul12, p01)
        add1 = _add(mul15, mul22)
        mul24 = _mul(mul16, _mul(p10, 1.0))
        gp10 = _mul(mul16, _mul(dy, 1.0))
        add2 = _add(add0, mul24)
        gp00 = _mul(mul16, dx)
        mul28 = _mul(mul16, p00)
        add3 = _add(add1, mul28)
        gmy, gmx = _neg(add2), _neg(add3)
        stride = count * Q
        tl.store(result + offset, gmx, mask=valid)
        tl.store(result + stride + offset, gmy, mask=valid)
        tl.store(result + 2 * stride + offset, gp00, mask=valid)
        tl.store(result + 3 * stride + offset, gp01, mask=valid)
        tl.store(result + 4 * stride + offset, gp10, mask=valid)
        tl.store(result + 5 * stride + offset, gp11, mask=valid)
        tl.store(result + 6 * stride + offset, gnorm, mask=valid)


def applies(px, py, weights, fibers, mu, precision, norm):
    tensors = (px, py, weights, mu, *precision, norm)
    return (
        mu.is_cuda
        and all(t.device == mu.device and t.dtype == torch.float32 and t.is_contiguous() for t in tensors)
        and fibers.device == mu.device
        and fibers.dtype == torch.long
        and fibers.is_contiguous()
        and not any(t.requires_grad for t in (px, py, weights))
        # With one node, autograd omits broadcast reductions and retains -0.
        and px.shape[1] > 1
        and mu.shape[0] * px.shape[1] < 2**31
        and torch.__version__.split("+")[0] == "2.11.0"
    )


class GaussianQuadrature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, px, py, weights, fibers, mu, p00, p01, p10, p11, norm):
        ctx.save_for_backward(px, py, weights, fibers, mu, p00, p01, p10, p11, norm)
        rows, q = mu.shape[0], px.shape[1]
        contribution = torch.empty((rows, q), dtype=mu.dtype, device=mu.device)
        if rows * q:
            _quadrature[(triton.cdiv(rows * q, 256),)](
                px,
                py,
                weights,
                fibers,
                mu,
                p00,
                p01,
                p10,
                p11,
                norm,
                norm,
                contribution,
                rows,
                Q=q,
                BACKWARD=False,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return contribution.sum(1)

    @staticmethod
    def backward(ctx, grad_output):
        px, py, weights, fibers, mu, p00, p01, p10, p11, norm = ctx.saved_tensors
        if torch.is_grad_enabled():
            from .splat_spectrum import _paired_gaussian_quadrature_eager

            with torch.enable_grad():
                result = _paired_gaussian_quadrature_eager(
                    px, py, weights, fibers, mu, (p00, p01, p10, p11), norm
                )
                inputs = [t for t, needed in zip(ctx.saved_tensors, ctx.needs_input_grad) if needed]
                grads = torch.autograd.grad(result, inputs, grad_output, create_graph=True)
            iterator = iter(grads)
            return tuple(next(iterator) if needed else None for needed in ctx.needs_input_grad)
        rows, q = mu.shape[0], px.shape[1]
        contribution = torch.empty((7, rows, q), dtype=mu.dtype, device=mu.device)
        if rows * q:
            _quadrature[(triton.cdiv(rows * q, 256),)](
                px,
                py,
                weights,
                fibers,
                mu,
                p00,
                p01,
                p10,
                p11,
                norm,
                grad_output.contiguous(),
                contribution,
                rows,
                Q=q,
                BACKWARD=True,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        grads = [contribution[i].sum(1) for i in range(7)]
        # Mirror the two zero-filled SelectBackward tensors and their final add.
        grad_mu = torch.stack((0.0 + grads[0], grads[1] + 0.0), dim=1)
        return None, None, None, None, grad_mu, *grads[2:]
