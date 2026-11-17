"""Fused observation geometry, mixture and first derivatives for image pairs."""

import torch

from ..utils.perf_env import (
    eigen_exp_poly_enabled, eigen_full_backward_mode, eigen_full_block, eigen_full_cblock, eigen_full_fma_enabled,
    eigen_full_warps, eigen_shared_recip_enabled, eigen_strict_vjp_enabled,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _f64(value: tl.constexpr):
        return tl.full((), value, tl.float64)

    @triton.jit
    def _exp_polynomial(x):
        """FP64 range reduction and degree-11 exp polynomial (Gaussian arguments).

        On the reduced interval |r|<=log(2)/2 the Taylor remainder is <1.3e-14
        relative. Hi/lo log(2) keeps range-reduction error below that bound.
        This is experimental and must pass the real renderer/gradient gates.
        """
        safe = tl.minimum(tl.maximum(x, -745.1332191019411), 709.782712893384)
        n = tl.floor(safe * 1.4426950408889634074 + 0.5)
        r = (safe - n * 0.69314718036912381649) - n * 1.90821492927058770002e-10
        value = tl.full(x.shape, 1.0 / 39916800.0, tl.float64)
        value = tl.fma(value, r, _f64(1.0 / 3628800.0))
        value = tl.fma(value, r, _f64(1.0 / 362880.0))
        value = tl.fma(value, r, _f64(1.0 / 40320.0))
        value = tl.fma(value, r, _f64(1.0 / 5040.0))
        value = tl.fma(value, r, _f64(1.0 / 720.0))
        value = tl.fma(value, r, _f64(1.0 / 120.0))
        value = tl.fma(value, r, _f64(1.0 / 24.0))
        value = tl.fma(value, r, _f64(1.0 / 6.0))
        value = tl.fma(value, r, 0.5)
        value = tl.fma(value, r, 1.0)
        value = tl.fma(value, r, 1.0)
        center = tl.minimum(tl.maximum(n, -1022.0), 1023.0).to(tl.int64)
        extra = n.to(tl.int64) - center
        power = ((center + 1023) << 52).to(tl.float64, bitcast=True)
        small = ((extra + 1023) << 52).to(tl.float64, bitcast=True)
        value = (value * small) * power
        value = tl.where(x < -745.1332191019411, 0.0, value)
        value = tl.where(x > 709.782712893384, float("inf"), value)
        return tl.where(x != x, x, value)

    @triton.jit
    def _forward(
        pos,
        cx,
        cy,
        cz,
        amp,
        x,
        y,
        band,
        point,
        active,
        exposure,
        inverse,
        detj,
        weights,
        mx,
        my,
        sx,
        sy,
        sz,
        out,
        deriv,
        N,
        BANDS: tl.constexpr,
        C: tl.constexpr,
        EXP_POLY: tl.constexpr,
        SHARED_RECIP: tl.constexpr,
        STRICT_VJP: tl.constexpr,
        BLOCK: tl.constexpr,
        CBLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        p = tl.load(point + i, mask, 0).to(tl.int64)
        s = tl.load(active + i, mask, 0).to(tl.int64)
        e = tl.load(exposure + p, mask, 0).to(tl.int64)
        bnd = tl.load(band + p, mask, 0).to(tl.int64)
        i00 = tl.load(inverse + 4 * p, mask, 1)
        i01 = tl.load(inverse + 4 * p + 1, mask, 0)
        i10 = tl.load(inverse + 4 * p + 2, mask, 0)
        i11 = tl.load(inverse + 4 * p + 3, mask, 1)
        ddx = tl.load(x + p, mask, 0).to(tl.float64) - tl.load(pos + 2 * s, mask, 0)
        ddy = tl.load(y + p, mask, 0).to(tl.float64) - tl.load(pos + 2 * s + 1, mask, 0)
        dx = i00 * ddx + i01 * ddy
        dy = i10 * ddx + i11 * ddy
        a0 = tl.load(cx + s, mask, 1)
        b0 = tl.load(cy + s, mask, 0)
        d0 = tl.load(cz + s, mask, 1)
        xx = i00 * i00 * a0 + 2 * i00 * i01 * b0 + i01 * i01 * d0
        xy = i00 * i10 * a0 + (i00 * i11 + i01 * i10) * b0 + i01 * i11 * d0
        yy = i10 * i10 * a0 + 2 * i10 * i11 * b0 + i11 * i11 * d0
        if STRICT_VJP:
            dx = libdevice.add_rn(libdevice.mul_rn(i00, ddx), libdevice.mul_rn(i01, ddy))
            dy = libdevice.add_rn(libdevice.mul_rn(i10, ddx), libdevice.mul_rn(i11, ddy))
            xx = libdevice.add_rn(
                libdevice.add_rn(
                    libdevice.mul_rn(libdevice.mul_rn(i00, i00), a0),
                    libdevice.mul_rn(libdevice.mul_rn(libdevice.mul_rn(_f64(2.0), i00), i01), b0),
                ),
                libdevice.mul_rn(libdevice.mul_rn(i01, i01), d0),
            )
            xy = libdevice.add_rn(
                libdevice.add_rn(
                    libdevice.mul_rn(libdevice.mul_rn(i00, i10), a0),
                    libdevice.mul_rn(
                        libdevice.add_rn(libdevice.mul_rn(i00, i11), libdevice.mul_rn(i01, i10)), b0
                    ),
                ),
                libdevice.mul_rn(libdevice.mul_rn(i01, i11), d0),
            )
            yy = libdevice.add_rn(
                libdevice.add_rn(
                    libdevice.mul_rn(libdevice.mul_rn(i10, i10), a0),
                    libdevice.mul_rn(libdevice.mul_rn(libdevice.mul_rn(_f64(2.0), i10), i11), b0),
                ),
                libdevice.mul_rn(libdevice.mul_rn(i11, i11), d0),
            )
        amplitude = tl.load(amp + s * BANDS + bnd, mask, 0).to(tl.float64)
        area = tl.load(detj + p, mask, 1)
        value = tl.full((BLOCK,), 0, tl.float64)
        gx = tl.full((BLOCK,), 0, tl.float64)
        gy = tl.full((BLOCK,), 0, tl.float64)
        gxx = tl.full((BLOCK,), 0, tl.float64)
        gxy = tl.full((BLOCK,), 0, tl.float64)
        gyy = tl.full((BLOCK,), 0, tl.float64)
        for start in range(tl.cdiv(C, CBLOCK)):
            k = start * CBLOCK + tl.arange(0, CBLOCK)
            valid = mask[:, None] & (k[None, :] < C)
            at = e[:, None] * C + k[None, :]
            w = tl.load(weights + p[:, None] * C + k[None, :], valid, 0)
            ex = dx[:, None] - tl.load(mx + at, valid, 0)
            ey = dy[:, None] - tl.load(my + at, valid, 0)
            a = xx[:, None] + tl.load(sx + at, valid, 1)
            b = xy[:, None] + tl.load(sy + at, valid, 0)
            d = yy[:, None] + tl.load(sz + at, valid, 1)
            determinant = a * d - b * b
            invdet = 1.0 / determinant
            numerator = d * ex * ex - 2.0 * b * ex * ey + a * ey * ey
            q = numerator * invdet if SHARED_RECIP else numerator / determinant
            exponential = _exp_polynomial(-0.5 * q) if EXP_POLY else tl.exp(-0.5 * q)
            norm = (
                0.15915494309189535 * tl.sqrt(invdet)
                if SHARED_RECIP
                else 0.15915494309189535 / tl.sqrt(determinant)
            )
            weighted = w * exponential * norm
            value += tl.sum(weighted, 1)
            gx += tl.sum(weighted * (-(d * ex - b * ey) * invdet), 1)
            gy += tl.sum(weighted * (-(a * ey - b * ex) * invdet), 1)
            gxx += tl.sum(weighted * (-0.5 * ((ey * ey - q * d) * invdet + d * invdet)), 1)
            gyy += tl.sum(weighted * (-0.5 * ((ex * ex - q * a) * invdet + a * invdet)), 1)
            gxy += tl.sum(weighted * ((ex * ey - q * b) * invdet + b * invdet), 1)
        factor = amplitude / area
        tl.store(out + i, value * amplitude / area, mask)
        if STRICT_VJP:
            tl.store(deriv + i, gx, mask)
            tl.store(deriv + N + i, gy, mask)
            tl.store(deriv + 2 * N + i, gxx, mask)
            tl.store(deriv + 3 * N + i, gxy, mask)
            tl.store(deriv + 4 * N + i, gyy, mask)
            tl.store(deriv + 5 * N + i, value, mask)
            return
        tl.store(deriv + i, -(i00 * gx + i10 * gy) * factor, mask)
        tl.store(deriv + N + i, -(i01 * gx + i11 * gy) * factor, mask)
        tl.store(deriv + 2 * N + i, (i00 * i00 * gxx + i00 * i10 * gxy + i10 * i10 * gyy) * factor, mask)
        tl.store(
            deriv + 3 * N + i,
            (2 * i00 * i01 * gxx + (i00 * i11 + i01 * i10) * gxy + 2 * i10 * i11 * gyy) * factor,
            mask,
        )
        tl.store(deriv + 4 * N + i, (i01 * i01 * gxx + i01 * i11 * gxy + i11 * i11 * gyy) * factor, mask)
        tl.store(deriv + 5 * N + i, value / area, mask)

    @triton.jit
    def _backward(
        deriv,
        gout,
        active,
        point,
        band,
        geometry,
        amplitude,
        N,
        ROWS,
        BANDS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        source = tl.load(active + i, mask, 0).to(tl.int64)
        pixel = tl.load(point + i, mask, 0).to(tl.int64)
        b = tl.load(band + pixel, mask, 0).to(tl.int64)
        g = tl.load(gout + i, mask, 0)
        for k in tl.static_range(5):
            value = tl.load(deriv + k * N + i, mask, 0) * g
            tl.atomic_add(geometry + k * ROWS + source, value, mask, sem="relaxed")
        value = tl.load(deriv + 5 * N + i, mask, 0) * g
        value = value.to(amplitude.dtype.element_ty)
        tl.atomic_add(amplitude + source * BANDS + b, value, mask, sem="relaxed")

    @triton.jit
    def _strict_vjp(
        deriv,
        gout,
        active,
        point,
        band,
        amp,
        inverse,
        area,
        geometry,
        amp_pair,
        N,
        BANDS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        p = tl.load(point + i, valid, 0).to(tl.int64)
        s = tl.load(active + i, valid, 0).to(tl.int64)
        b = tl.load(band + p, valid, 0).to(tl.int64)
        g = tl.load(gout + i, valid, 0)
        am = tl.load(amp + s * BANDS + b, valid, 0).to(tl.float64)
        ar = tl.load(area + p, valid, 1)
        divided = libdevice.div_rn(g, ar)
        scaled = libdevice.mul_rn(divided, am)
        dx = libdevice.mul_rn(tl.load(deriv + i, valid, 0), scaled)
        dy = libdevice.mul_rn(tl.load(deriv + N + i, valid, 0), scaled)
        gx = libdevice.mul_rn(tl.load(deriv + 2 * N + i, valid, 0), scaled)
        gy = libdevice.mul_rn(tl.load(deriv + 3 * N + i, valid, 0), scaled)
        gz = libdevice.mul_rn(tl.load(deriv + 4 * N + i, valid, 0), scaled)
        u = tl.load(inverse + 4 * p, valid, 1)
        v = tl.load(inverse + 4 * p + 1, valid, 0)
        w = tl.load(inverse + 4 * p + 2, valid, 0)
        z = tl.load(inverse + 4 * p + 3, valid, 1)
        px = -libdevice.add_rn(libdevice.mul_rn(dy, w), libdevice.mul_rn(dx, u))
        py = -libdevice.add_rn(libdevice.mul_rn(dy, z), libdevice.mul_rn(dx, v))
        ca = libdevice.add_rn(
            libdevice.add_rn(
                libdevice.mul_rn(gz, libdevice.mul_rn(w, w)), libdevice.mul_rn(gy, libdevice.mul_rn(u, w))
            ),
            libdevice.mul_rn(gx, libdevice.mul_rn(u, u)),
        )
        cb = libdevice.add_rn(
            libdevice.add_rn(
                libdevice.mul_rn(gz, libdevice.mul_rn(libdevice.mul_rn(w, _f64(2.0)), z)),
                libdevice.mul_rn(gy, libdevice.add_rn(libdevice.mul_rn(u, z), libdevice.mul_rn(v, w))),
            ),
            libdevice.mul_rn(gx, libdevice.mul_rn(libdevice.mul_rn(u, _f64(2.0)), v)),
        )
        cd = libdevice.add_rn(
            libdevice.add_rn(
                libdevice.mul_rn(gz, libdevice.mul_rn(z, z)), libdevice.mul_rn(gy, libdevice.mul_rn(v, z))
            ),
            libdevice.mul_rn(gx, libdevice.mul_rn(v, v)),
        )
        tl.store(geometry + i, px, valid)
        tl.store(geometry + N + i, py, valid)
        tl.store(geometry + 2 * N + i, ca, valid)
        tl.store(geometry + 3 * N + i, cb, valid)
        tl.store(geometry + 4 * N + i, cd, valid)
        tl.store(amp_pair + i, libdevice.mul_rn(divided, tl.load(deriv + 5 * N + i, valid, 0)), valid)


class _FullPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, cx, cy, cz, amp, x, y, band, point, active, exposure, inverse, detj, *tables):
        n = point.numel()
        strict = eigen_strict_vjp_enabled()
        output = pos.new_empty(n)
        derivatives = pos.new_empty((6, n))
        if n:
            block = eigen_full_block()
            cblock = eigen_full_cblock()
            warps = eigen_full_warps()
            _forward[(triton.cdiv(n, block),)](
                pos.contiguous(),
                cx.contiguous(),
                cy.contiguous(),
                cz.contiguous(),
                amp.contiguous(),
                x.contiguous(),
                y.contiguous(),
                band.contiguous(),
                point.contiguous(),
                active.contiguous(),
                exposure.contiguous(),
                inverse.contiguous(),
                detj.contiguous(),
                *[t.contiguous() for t in tables],
                output,
                derivatives,
                n,
                amp.shape[1],
                tables[0].shape[1],
                eigen_exp_poly_enabled(),
                eigen_shared_recip_enabled(),
                strict,
                block,
                cblock,
                enable_fp_fusion=eigen_full_fma_enabled(),
                num_warps=warps,
            )
        ctx.save_for_backward(derivatives, active, point, band, amp, inverse, detj)
        ctx.strict = strict
        ctx.amp_shape = amp.shape
        ctx.amp_dtype = amp.dtype
        ctx.rows = pos.shape[0]
        return output

    @staticmethod
    def backward(ctx, gout):
        derivatives, active, point, band, amp, inverse, detj = ctx.saved_tensors
        if ctx.strict:
            n = active.numel()
            pair_grad = gout.new_empty((5, n))
            amp_pair = torch.empty(n, device=gout.device, dtype=ctx.amp_dtype)
            if n:
                _strict_vjp[(triton.cdiv(n, 256),)](
                    derivatives,
                    gout.contiguous(),
                    active,
                    point,
                    band,
                    amp,
                    inverse,
                    detj,
                    pair_grad,
                    amp_pair,
                    n,
                    ctx.amp_shape[1],
                    256,
                    enable_fp_fusion=False,
                )
            pos = gout.new_zeros((ctx.rows, 2)).index_add_(0, active, pair_grad[:2].T.contiguous())
            cov = [gout.new_zeros(ctx.rows).index_add_(0, active, pair_grad[k]) for k in [2, 3, 4]]
            amp_index = active * ctx.amp_shape[1] + band[point]
            amp_grad = torch.zeros(ctx.amp_shape.numel(), device=gout.device, dtype=ctx.amp_dtype).index_add_(
                0, amp_index, amp_pair
            )
            return (pos, *cov, amp_grad.reshape(ctx.amp_shape), *[None] * 14)
        if eigen_full_backward_mode() == "triton":
            geometry = gout.new_zeros((5, ctx.rows))
            amp_grad = torch.zeros(ctx.amp_shape, dtype=ctx.amp_dtype, device=gout.device)
            n = active.numel()
            if n:
                _backward[(triton.cdiv(n, 256),)](
                    derivatives,
                    gout.contiguous(),
                    active,
                    point,
                    band,
                    geometry,
                    amp_grad,
                    n,
                    ctx.rows,
                    ctx.amp_shape[1],
                    256,
                )
            return (
                geometry[:2].T.contiguous(),
                geometry[2],
                geometry[3],
                geometry[4],
                amp_grad,
                *[None] * 14,
            )
        grads = derivatives * gout[None, :]
        geometry = grads.new_zeros((5, ctx.rows)).index_add_(1, active, grads[:5])
        # Match the existing gather.double() backward: round each amplitude
        # contribution to the amplitude dtype BEFORE accumulating duplicates.
        amp_index = active * ctx.amp_shape[1] + band[point]
        amp_grad = torch.zeros(ctx.amp_shape.numel(), dtype=ctx.amp_dtype, device=gout.device)
        amp_grad.index_add_(0, amp_index, grads[5].to(ctx.amp_dtype))
        return (
            geometry[:2].T.contiguous(),
            geometry[2],
            geometry[3],
            geometry[4],
            amp_grad.reshape(ctx.amp_shape),
            *[None] * 14,
        )


def full_pairs(positions, cxx, cxy, cyy, amplitudes, x, y, bands, point, active, context):
    return _FullPairs.apply(
        positions,
        cxx,
        cxy,
        cyy,
        amplitudes,
        x,
        y,
        bands,
        point,
        active,
        context["exposure_row"],
        context["inverse"],
        context["det"],
        *[context[k] for k in ["weights", "mx", "my", "sxx", "sxy", "syy"]],
    )
