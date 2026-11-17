"""Validated dispatcher for generated exact covariance kernels."""

from __future__ import annotations

import torch
import triton

from .covariance_generated import _cov_backward, _cov_forward


def applies(log_scales, theta):
    return (
        log_scales.is_cuda
        and theta.device == log_scales.device
        and log_scales.dtype == theta.dtype == torch.float32
        and log_scales.is_contiguous()
        and theta.is_contiguous()
        and log_scales.ndim == 2
        and log_scales.shape[1] == 2
        and theta.shape == log_scales.shape[:1]
        and torch.__version__.split("+")[0] == "2.11.0"
    )


class Covariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logs, theta):
        ctx.save_for_backward(logs, theta)
        ctx.set_materialize_grads(False)
        a, b, d = (torch.empty_like(theta) for _ in range(3))
        if theta.numel():
            _cov_forward[(triton.cdiv(theta.numel(), 256),)](
                theta,
                logs,
                a,
                b,
                d,
                theta.numel(),
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return a, b, d

    @staticmethod
    def backward(ctx, ga, gb, gd):
        logs, theta = ctx.saved_tensors
        if (
            all(g is not None for g in (ga, gb, gd))
            and ctx.needs_input_grad == (True, True)
            and not torch.is_grad_enabled()
        ):
            gl, gt = torch.empty_like(logs), torch.empty_like(theta)
            if theta.numel():
                _cov_backward[(triton.cdiv(theta.numel(), 256),)](
                    gd.contiguous(),
                    theta,
                    gb.contiguous(),
                    ga.contiguous(),
                    logs,
                    gl,
                    gt,
                    theta.numel(),
                    BLOCK=256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return gl, gt
        from survey_fusion.models.splat_field import SplatField

        higher_order = torch.is_grad_enabled()
        with torch.enable_grad():
            outputs = SplatField._covariance_components_eager_from_rows(logs, theta)
            active = [(o, g) for o, g in zip(outputs, (ga, gb, gd)) if g is not None]
            if not active:
                return None, None
            inputs = [t for t, needed in zip((logs, theta), ctx.needs_input_grad) if needed]
            gradients = torch.autograd.grad(
                tuple(o for o, _ in active), inputs, tuple(g for _, g in active), create_graph=higher_order
            )
        iterator = iter(gradients)
        return tuple(next(iterator) if needed else None for needed in ctx.needs_input_grad)
