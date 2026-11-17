# ruff: noqa: I001
"""Exact-rounding covariance kernels generated from the eager autograd graph."""
import triton
import triton.language as tl
from triton.language.extra import libdevice
from .raster_pair_kernels import _add, _mul, _sub

@triton.jit
def _neg(x):
    return tl.inline_asm_elementwise("neg.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _cov_forward(p_theta_1, p_logs_1, o_a, o_b, o_d, N, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = r < N
    v0 = tl.load(p_theta_1 + r, mask=keep, other=0.0)
    v1 = libdevice.cos(v0)
    v2 = _mul(v1, v1)
    v3 = tl.load(p_logs_1 + r * 2 + 0, mask=keep, other=0.0)
    v4 = v3
    v5 = _mul(v4, tl.full((BLOCK,), 2.0, tl.float32))
    v6 = libdevice.exp(v5)
    v7 = _mul(v2, v6)
    v8 = libdevice.sin(v0)
    v9 = _mul(v8, v8)
    v10 = tl.load(p_logs_1 + r * 2 + 1, mask=keep, other=0.0)
    v11 = v10
    v12 = _mul(v11, tl.full((BLOCK,), 2.0, tl.float32))
    v13 = libdevice.exp(v12)
    v14 = _mul(v9, v13)
    v15 = _add(v7, v14)
    v16 = _mul(v1, v8)
    v17 = _sub(v6, v13)
    v18 = _mul(v16, v17)
    v19 = _mul(v8, v8)
    v20 = _mul(v19, v6)
    v21 = _mul(v1, v1)
    v22 = _mul(v21, v13)
    v23 = _add(v20, v22)
    tl.store(o_a + r, v15, mask=keep)
    tl.store(o_b + r, v18, mask=keep)
    tl.store(o_d + r, v23, mask=keep)

@triton.jit
def _cov_backward(p_gd_1, p_theta_1, p_gb_1, p_ga_1, p_logs_1, o_gl, o_gt, N, BLOCK: tl.constexpr):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = r < N
    v0 = tl.full((BLOCK,), 0.0, tl.float32)
    v1 = tl.load(p_gd_1 + r, mask=keep, other=0.0)
    v2 = tl.load(p_theta_1 + r, mask=keep, other=0.0)
    v3 = libdevice.sin(v2)
    v4 = _mul(v3, v3)
    v5 = _mul(v1, v4)
    v6 = tl.load(p_gb_1 + r, mask=keep, other=0.0)
    v7 = libdevice.cos(v2)
    v8 = _mul(v7, v3)
    v9 = _mul(v6, v8)
    v10 = _add(v5, v9)
    v11 = tl.load(p_ga_1 + r, mask=keep, other=0.0)
    v12 = _mul(v7, v7)
    v13 = _mul(v11, v12)
    v14 = _add(v10, v13)
    v15 = tl.load(p_logs_1 + r * 2 + 0, mask=keep, other=0.0)
    v16 = v15
    v17 = _mul(v16, tl.full((BLOCK,), 2.0, tl.float32))
    v18 = libdevice.exp(v17)
    v19 = v18
    v20 = v19
    v21 = _mul(v14, v20)
    v22 = _mul(v21, tl.full((BLOCK,), 2.0, tl.float32))
    v23 = v22
    v24 = _add(v0, v23)
    v25 = _mul(v7, v7)
    v26 = _mul(v1, v25)
    v27 = _neg(v9)
    v28 = _add(v26, v27)
    v29 = _mul(v3, v3)
    v30 = _mul(v11, v29)
    v31 = _add(v28, v30)
    v32 = tl.load(p_logs_1 + r * 2 + 1, mask=keep, other=0.0)
    v33 = v32
    v34 = _mul(v33, tl.full((BLOCK,), 2.0, tl.float32))
    v35 = libdevice.exp(v34)
    v36 = v35
    v37 = v36
    v38 = _mul(v31, v37)
    v39 = _mul(v38, tl.full((BLOCK,), 2.0, tl.float32))
    v40 = v39
    v41 = tl.full((BLOCK,), 0.0, tl.float32)
    v42 = _add(v40, v41)
    v43 = _mul(v1, v18)
    v44 = _mul(v43, v3)
    v45 = _mul(v43, v3)
    v46 = _add(v44, v45)
    v47 = _sub(v18, v35)
    v48 = _mul(v6, v47)
    v49 = _mul(v48, v7)
    v50 = _add(v46, v49)
    v51 = _mul(v11, v35)
    v52 = _mul(v51, v3)
    v53 = _add(v50, v52)
    v54 = _mul(v51, v3)
    v55 = _add(v53, v54)
    v56 = libdevice.cos(v2)
    v57 = _mul(v55, v56)
    v58 = _mul(v1, v35)
    v59 = _mul(v58, v7)
    v60 = _mul(v58, v7)
    v61 = _add(v59, v60)
    v62 = _mul(v48, v3)
    v63 = _add(v61, v62)
    v64 = _mul(v11, v18)
    v65 = _mul(v64, v7)
    v66 = _add(v63, v65)
    v67 = _mul(v64, v7)
    v68 = _add(v66, v67)
    v69 = libdevice.sin(v2)
    v70 = _neg(v69)
    v71 = _mul(v68, v70)
    v72 = _add(v57, v71)
    tl.store(o_gl + r * 2, v24, mask=keep)
    tl.store(o_gl + r * 2 + 1, v42, mask=keep)
    tl.store(o_gt + r, v72, mask=keep)
