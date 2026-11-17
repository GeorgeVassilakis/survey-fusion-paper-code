"""Fused exact ellipse/PSF-box test for the existing eigen_mog pair candidates."""

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _basis(logs, theta, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < n
        angle = tl.load(theta + i, valid, 0).to(tl.float64)
        co, si = libdevice.cos(angle), libdevice.sin(angle)
        sx = libdevice.exp(tl.load(logs + 2 * i, valid, 0).to(tl.float64))
        sy = libdevice.exp(tl.load(logs + 2 * i + 1, valid, 0).to(tl.float64))
        tl.store(out + 4 * i, co, valid)
        tl.store(out + 4 * i + 1, si, valid)
        tl.store(out + 4 * i + 2, sx, valid)
        tl.store(out + 4 * i + 3, sy, valid)

    @triton.jit
    def _prune(
        point,
        source,
        x,
        y,
        mu,
        logs,
        theta,
        inv,
        forward,
        basis,
        keep,
        n,
        radius,
        threshold,
        CACHED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < n
        p = tl.load(point + i, valid, 0).to(tl.int64)
        s = tl.load(source + i, valid, 0).to(tl.int64)
        if CACHED:
            co = tl.load(basis + 4 * s, valid, 1)
            si = tl.load(basis + 4 * s + 1, valid, 0)
            sx = tl.load(basis + 4 * s + 2, valid, 1)
            sy = tl.load(basis + 4 * s + 3, valid, 1)
        else:
            angle = tl.load(theta + s, valid, 0).to(tl.float64)
            co, si = libdevice.cos(angle), libdevice.sin(angle)
            sx = libdevice.exp(tl.load(logs + 2 * s, valid, 0).to(tl.float64))
            sy = libdevice.exp(tl.load(logs + 2 * s + 1, valid, 0).to(tl.float64))
        i00 = tl.load(inv + 4 * p, valid, 1)
        i01 = tl.load(inv + 4 * p + 1, valid, 0)
        i10 = tl.load(inv + 4 * p + 2, valid, 0)
        i11 = tl.load(inv + 4 * p + 3, valid, 1)
        f00 = tl.load(forward + 4 * p, valid, 1)
        f01 = tl.load(forward + 4 * p + 1, valid, 0)
        f10 = tl.load(forward + 4 * p + 2, valid, 0)
        f11 = tl.load(forward + 4 * p + 3, valid, 1)
        fx0, fy0 = (i00 * co + i01 * si) * sx, (i10 * co + i11 * si) * sx
        fx1, fy1 = (-i00 * si + i01 * co) * sy, (-i10 * si + i11 * co) * sy
        cxx, cxy, cyy = fx0 * fx0 + fx1 * fx1, fx0 * fy0 + fx1 * fy1, fy0 * fy0 + fy1 * fy1
        w00, w01 = (co * f00 + si * f10) / sx, (co * f01 + si * f11) / sx
        w10, w11 = (-si * f00 + co * f10) / sy, (-si * f01 + co * f11) / sy
        ux = tl.load(x + p, valid, 0).to(tl.float64) - tl.load(mu + 2 * s, valid, 0).to(tl.float64)
        uy = tl.load(y + p, valid, 0).to(tl.float64) - tl.load(mu + 2 * s + 1, valid, 0).to(tl.float64)
        dx, dy = i00 * ux + i01 * uy, i10 * ux + i11 * uy
        distance = tl.full((BLOCK,), float("inf"), tl.float64)
        for branch in tl.static_range(2):
            sign = 2.0 * branch - 1.0
            a = dx - sign * radius
            b = dy - tl.minimum(tl.maximum(dy - cxy / cxx * a, -radius), radius)
            u, v = w00 * a + w01 * b, w10 * a + w11 * b
            distance = tl.minimum(distance, u * u + v * v)
            b = dy - sign * radius
            a = dx - tl.minimum(tl.maximum(dx - cxy / cyy * b, -radius), radius)
            u, v = w00 * a + w01 * b, w10 * a + w11 * b
            distance = tl.minimum(distance, u * u + v * v)
        inside = tl.maximum(tl.abs(dx), tl.abs(dy)) <= radius
        nonfinite = (distance != distance) | (tl.abs(distance) == float("inf"))
        tl.store(keep + i, inside | (distance <= threshold) | nonfinite, valid)


def shape_basis(model):
    n = model.mu.shape[0]
    result = torch.empty((n, 4), dtype=torch.float64, device=model.mu.device)
    if n:
        _basis[(triton.cdiv(n, 256),)](model.log_scales, model.theta, result, n, 256, enable_fp_fusion=False)
    return result


def ellipse_box_keep(model, x, y, point, source, context, radius, threshold, *, basis=None):
    """Return the existing conservative membership mask, with no support change."""
    if triton is None or not x.is_cuda:
        raise RuntimeError("Fused ellipse-box pruning requires CUDA/Triton")
    keep = torch.empty(point.numel(), dtype=torch.bool, device=x.device)
    if point.numel():
        _prune[(triton.cdiv(point.numel(), 256),)](
            point.contiguous(),
            source.contiguous(),
            x.contiguous(),
            y.contiguous(),
            model.mu.contiguous(),
            model.log_scales.contiguous(),
            model.theta.contiguous(),
            context["inverse"].contiguous(),
            context["forward"].contiguous(),
            basis if basis is not None else model.theta,
            keep,
            point.numel(),
            radius,
            threshold,
            basis is not None,
            256,
            enable_fp_fusion=False,
        )
    return keep
