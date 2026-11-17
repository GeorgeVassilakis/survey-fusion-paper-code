"""Fused row gather for the six per-splat parameter tensors."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_BLOCK_ROWS = 256
_INDEX_LIMIT = 2 ** 31


def applies(indices: torch.Tensor, *parameters: torch.Tensor) -> bool:
    """Use the fused kernel only for its supported storage and index range."""
    return (
        indices.is_cuda
        and indices.dtype == torch.long
        and indices.ndim == 1
        and indices.numel() < _INDEX_LIMIT
        and all(
            value.device == indices.device
            and value.dtype == torch.float32
            and value.is_contiguous()
            and value.numel() < _INDEX_LIMIT
            and indices.numel() * math.prod(value.shape[1:]) < _INDEX_LIMIT
            for value in parameters
        )
    )


@triton.jit
def _gather_splat_rows_kernel(
    indices_ptr,
    mu_ptr,
    log_scales_ptr,
    theta_ptr,
    raw_flux_ptr,
    sed_coeffs_ptr,
    sed_shift_ptr,
    out_mu_ptr,
    out_log_scales_ptr,
    out_theta_ptr,
    out_raw_flux_ptr,
    out_sed_coeffs_ptr,
    out_sed_shift_ptr,
    n_rows,
    N_TEMPLATES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    keep = row < n_rows
    source = tl.load(indices_ptr + row, mask=keep, other=0).to(tl.int32)

    for column in tl.static_range(0, 2):
        tl.store(
            out_mu_ptr + row * 2 + column,
            tl.load(mu_ptr + source * 2 + column, mask=keep),
            mask=keep,
        )
        tl.store(
            out_log_scales_ptr + row * 2 + column,
            tl.load(log_scales_ptr + source * 2 + column, mask=keep),
            mask=keep,
        )
    tl.store(out_theta_ptr + row, tl.load(theta_ptr + source, mask=keep), mask=keep)
    tl.store(
        out_raw_flux_ptr + row,
        tl.load(raw_flux_ptr + source, mask=keep),
        mask=keep,
    )
    for column in tl.static_range(0, N_TEMPLATES):
        tl.store(
            out_sed_coeffs_ptr + row * N_TEMPLATES + column,
            tl.load(
                sed_coeffs_ptr + source * N_TEMPLATES + column, mask=keep
            ),
            mask=keep,
        )
    tl.store(
        out_sed_shift_ptr + row,
        tl.load(sed_shift_ptr + source, mask=keep),
        mask=keep,
    )


def gather_splat_rows(
    indices: torch.Tensor,
    mu: torch.Tensor,
    log_scales: torch.Tensor,
    theta: torch.Tensor,
    raw_flux: torch.Tensor,
    sed_coeffs: torch.Tensor,
    sed_shift: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Gather all per-splat parameters in one CUDA kernel."""
    if not indices.is_cuda:
        raise ValueError("Fused splat row gather requires CUDA indices.")
    if indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError("Fused splat row gather requires one-dimensional int64 indices.")
    # The kernel reads indices_ptr + row. Preserve strided and offset views by
    # staging their logical values; contiguous production indices are reused.
    indices = indices.contiguous()
    parameters = (mu, log_scales, theta, raw_flux, sed_coeffs, sed_shift)
    if any(not value.is_cuda for value in parameters):
        raise ValueError("Fused splat row gather requires CUDA parameters.")
    if any(value.dtype != torch.float32 for value in parameters):
        raise ValueError("Fused splat row gather requires float32 parameters.")
    if any(not value.is_contiguous() for value in parameters):
        raise ValueError("Fused splat row gather requires contiguous parameters.")
    if mu.ndim != 2 or mu.shape[1] != 2:
        raise ValueError("mu must have shape [rows, 2].")
    if log_scales.shape != mu.shape:
        raise ValueError("log_scales must have the same shape as mu.")
    n_splats = int(mu.shape[0])
    if (
        theta.shape != (n_splats,)
        or raw_flux.shape != (n_splats,)
        or sed_shift.shape != (n_splats,)
        or sed_coeffs.ndim != 2
        or int(sed_coeffs.shape[0]) != n_splats
    ):
        raise ValueError("Per-splat parameters must share one row count.")
    if int(indices.numel()) >= _INDEX_LIMIT or n_splats >= _INDEX_LIMIT:
        raise ValueError("Fused splat row gather exceeds its int32 index limit.")

    n_rows = int(indices.numel())
    outputs = (
        torch.empty((n_rows, 2), dtype=mu.dtype, device=mu.device),
        torch.empty((n_rows, 2), dtype=mu.dtype, device=mu.device),
        torch.empty(n_rows, dtype=mu.dtype, device=mu.device),
        torch.empty(n_rows, dtype=mu.dtype, device=mu.device),
        torch.empty(
            (n_rows, int(sed_coeffs.shape[1])),
            dtype=mu.dtype,
            device=mu.device,
        ),
        torch.empty(n_rows, dtype=mu.dtype, device=mu.device),
    )
    if n_rows:
        grid = (triton.cdiv(n_rows, _BLOCK_ROWS),)
        _gather_splat_rows_kernel[grid](
            indices,
            *parameters,
            *outputs,
            n_rows,
            N_TEMPLATES=int(sed_coeffs.shape[1]),
            BLOCK=_BLOCK_ROWS,
            enable_fp_fusion=False,
        )
    return outputs
