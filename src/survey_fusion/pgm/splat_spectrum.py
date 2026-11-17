from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache, partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.weak import WeakTensorKeyDictionary

from survey_fusion.models.splat_field import SplatField
from survey_fusion.utils.perf_env import (
    AMP_CHUNK,
    activation_checkpointing_disabled,
    compact_all_bands_enabled,
    compact_band_weights_enabled,
    env_chunk,
    exact_band_contraction_enabled,
    exact_band_reduction_rows_supported,
    fiber_bins_mode,
    fused_band_weights_enabled,
    fused_fiber_bins_backward_enabled,
    fused_quadrature_enabled,
    fused_splat_gather_enabled,
    sparse_band_contraction_enabled,
    validated_cublas_runtime,
)
from survey_fusion.utils.torch_utils import (
    as_tensor_on_device,
    copy_tensor_to_device,
    tensor_on_device,
)
from survey_fusion.utils.wcs import radec_deg_to_omega_xyz

from .contracts import (
    ExposureState, SkyDiskApertureSpec, aperture_area_arcsec2,
    validated_spectral_source_scale, spectral_source_geometry_key, validate_spectral_source_psf,
)
from .splat_raster import build_pair_index


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _normal_pdf(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)


def _integrated_positive_part(t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = t / sigma
    return 0.5 * (
        (t.square() + sigma.square()) * _normal_cdf(z)
        + t * sigma * _normal_pdf(z)
    )


def _truncated_linear_bin_integral(
    bin_lower: torch.Tensor,
    bin_upper: torch.Tensor,
    bin_width: torch.Tensor,
    lsf_sigma: torch.Tensor,
    source_lower: torch.Tensor,
    source_upper: torch.Tensor,
    slope: torch.Tensor,
    intercept: torch.Tensor,
    lsf_mode: str | None,
) -> torch.Tensor:
    """Integrate a linear source segment over a span-truncated bin.

    The source segment is zero outside [source_lower, source_upper], and the
    bin integration is zero outside [bin_lower, bin_upper]. Both truncations
    are part of the integral; bin_width remains the original full-bin width.
    """
    if bin_lower.ndim == 2:
        bin_lower = bin_lower.unsqueeze(-1)
        bin_upper = bin_upper.unsqueeze(-1)
        bin_width = bin_width.unsqueeze(-1)
    if source_lower.ndim == 2:
        source_lower = source_lower.unsqueeze(1)
        source_upper = source_upper.unsqueeze(1)
        slope = slope.unsqueeze(1)
        intercept = intercept.unsqueeze(1)

    def no_lsf() -> torch.Tensor:
        lower = torch.maximum(bin_lower, source_lower)
        upper = torch.minimum(bin_upper, source_upper)
        active = (upper > lower) & (source_upper > source_lower)
        result = (
            (0.5 * slope * (lower + upper) + intercept)
            * (upper - lower)
            / bin_width
        )
        return torch.where(active, result, torch.zeros_like(result))

    def positive_lsf(sigma: torch.Tensor) -> torch.Tensor:
        def cdf_integral(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = (q - y) / sigma
            return (y - q) * _normal_cdf(z) - sigma * _normal_pdf(z)

        def first_moment_integral(
            q: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            delta = q - y
            z = delta / sigma
            cdf = _normal_cdf(z)
            pdf = _normal_pdf(z)
            a0 = (y - q) * cdf - sigma * pdf
            b0 = 0.5 * (
                (delta.square() + sigma.square()) * cdf
                + sigma * delta * pdf
            )
            return y * a0 - b0

        def sigma_pdf_integral(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -sigma.square() * _normal_cdf((q - y) / sigma)

        source_lo = source_lower
        source_hi = source_upper
        p0 = (
            cdf_integral(source_hi, bin_upper)
            - cdf_integral(source_lo, bin_upper)
            - cdf_integral(source_hi, bin_lower)
            + cdf_integral(source_lo, bin_lower)
        )
        p1 = (
            first_moment_integral(source_hi, bin_upper)
            - first_moment_integral(source_lo, bin_upper)
            - first_moment_integral(source_hi, bin_lower)
            + first_moment_integral(source_lo, bin_lower)
        )
        q = (
            sigma_pdf_integral(source_lo, bin_upper)
            - sigma_pdf_integral(source_hi, bin_upper)
            - sigma_pdf_integral(source_lo, bin_lower)
            + sigma_pdf_integral(source_hi, bin_lower)
        )
        result = (slope * (p1 + q) + intercept * p0) / bin_width
        active = (bin_upper > bin_lower) & (source_upper > source_lower)
        return torch.where(active, result, torch.zeros_like(result))

    if lsf_mode == "zero":
        return no_lsf()
    if lsf_mode == "positive":
        return positive_lsf(lsf_sigma[:, None, None])
    positive = lsf_sigma > 0
    safe_sigma = torch.where(positive, lsf_sigma, torch.ones_like(lsf_sigma))
    return torch.where(
        positive[:, None, None],
        positive_lsf(safe_sigma[:, None, None]),
        no_lsf(),
    )


def piecewise_linear_bin_average(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    *,
    validate: bool = True,
    lsf_mode: str | None = None,
) -> torch.Tensor:
    """Average clamped piecewise-linear spectra over top-hat bins and Gaussian LSF.

    knot_lam and knot_values have shape [J, N]. lower, upper, and lsf_sigma
    have shape [M]. The result has shape [M, J].

    validate=False skips the data-dependent input checks (each is a host
    sync); pass it only when the caller has already validated the inputs,
    e.g. at spectrum-cache build time. lsf_mode="zero" or "positive"
    declares the sign pattern of lsf_sigma so the matching branch runs on
    the full tensors without the two any() syncs and mask gathers; values
    are element-for-element identical to the masked path. None means the
    pattern is unknown or mixed.
    """
    if knot_lam.ndim != 2 or knot_values.shape != knot_lam.shape:
        raise ValueError("knot_lam and knot_values must have matching [J,N] shapes.")
    if knot_lam.shape[1] < 2:
        raise ValueError("At least two wavelength knots are required.")
    if lower.ndim != 1 or upper.shape != lower.shape or lsf_sigma.shape != lower.shape:
        raise ValueError("lower, upper, and lsf_sigma must have matching [M] shapes.")
    if validate:
        if torch.any(knot_lam[:, 1:] <= knot_lam[:, :-1]):
            raise ValueError("Wavelength knots must be strictly increasing.")
        if torch.any(upper <= lower):
            raise ValueError("Spectral bin upper edges must exceed lower edges.")
        if torch.any(lsf_sigma < 0):
            raise ValueError("LSF sigma must be non-negative.")

    slopes = (knot_values[:, 1:] - knot_values[:, :-1]) / (
        knot_lam[:, 1:] - knot_lam[:, :-1]
    )
    hinge_coeff = torch.cat(
        (
            slopes[:, :1],
            slopes[:, 1:] - slopes[:, :-1],
            -slopes[:, -1:],
        ),
        dim=1,
    )
    hinge_average = _hinge_bin_averages(
        knot_lam, lower, upper, lsf_sigma, lsf_mode, off_grid="clamp"
    )
    # Multiply-reduce instead of einsum: the hinge expansion cancels huge
    # mixed-sign terms, and TF32 tensor-core matmuls (forward or backward)
    # lose enough mantissa to leave large negative bin values on real SEDs.
    return knot_values[None, :, 0] + (
        hinge_average * hinge_coeff[None, :, :]
    ).sum(dim=2)


def _hinge_bin_averages(
    knot_lam: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    lsf_mode: str | None,
    off_grid: str = "clamp",
) -> torch.Tensor:
    """Per-knot LSF-smoothed hinge integrals averaged over each bin, [M,J,N]."""
    if off_grid not in ("clamp", "zero"):
        raise ValueError("off_grid must be 'clamp' or 'zero'.")
    if off_grid == "zero":
        width = (upper - lower)[:, None, None]
        bin_lower = torch.maximum(
            lower[:, None, None], knot_lam[None, :, :1]
        )
        bin_upper = torch.minimum(
            upper[:, None, None], knot_lam[None, :, -1:]
        )
        return _truncated_linear_bin_integral(
            bin_lower,
            bin_upper,
            width,
            lsf_sigma,
            knot_lam[None, :, :],
            knot_lam[None, :, -1:],
            torch.ones_like(knot_lam)[None, :, :],
            -knot_lam[None, :, :],
            lsf_mode,
        )

    width = (upper - lower)[:, None, None]
    lo = lower[:, None, None] - knot_lam[None, :, :]
    hi = upper[:, None, None] - knot_lam[None, :, :]
    use_reverse = knot_lam[None, :, :] < 0.5 * (
        lower[:, None, None] + upper[:, None, None]
    )

    if lsf_mode == "zero":
        direct = (F.relu(hi).square() - F.relu(lo).square()) / (2.0 * width)
        reverse = (
            0.5 * (lower + upper)[:, None, None]
            - knot_lam[None, :, :]
            + (F.relu(-lo).square() - F.relu(-hi).square()) / (2.0 * width)
        )
        hinge_average = torch.where(use_reverse, reverse, direct)
    elif lsf_mode == "positive":
        sigma = lsf_sigma[:, None, None]
        direct = (
            _integrated_positive_part(hi, sigma)
            - _integrated_positive_part(lo, sigma)
        ) / width
        reverse = (
            0.5 * (lower + upper)[:, None, None]
            - knot_lam[None, :, :]
            + (
                _integrated_positive_part(-lo, sigma)
                - _integrated_positive_part(-hi, sigma)
            )
            / width
        )
        hinge_average = torch.where(use_reverse, reverse, direct)
    else:
        hinge_average = torch.empty_like(lo)
        zero_lsf = lsf_sigma == 0
        if bool(torch.any(zero_lsf)):
            lo_zero = lo[zero_lsf]
            hi_zero = hi[zero_lsf]
            direct = (
                F.relu(hi_zero).square() - F.relu(lo_zero).square()
            ) / (2.0 * width[zero_lsf])
            reverse = (
                0.5 * (lower[zero_lsf] + upper[zero_lsf])[:, None, None]
                - knot_lam[None, :, :]
                + (
                    F.relu(-lo_zero).square() - F.relu(-hi_zero).square()
                )
                / (2.0 * width[zero_lsf])
            )
            hinge_average[zero_lsf] = torch.where(
                use_reverse[zero_lsf], reverse, direct
            )
        positive_lsf = ~zero_lsf
        if bool(torch.any(positive_lsf)):
            sigma = lsf_sigma[positive_lsf, None, None]
            direct = (
                _integrated_positive_part(hi[positive_lsf], sigma)
                - _integrated_positive_part(lo[positive_lsf], sigma)
            ) / width[positive_lsf]
            reverse = (
                0.5 * (lower[positive_lsf] + upper[positive_lsf])[:, None, None]
                - knot_lam[None, :, :]
                + (
                    _integrated_positive_part(-lo[positive_lsf], sigma)
                    - _integrated_positive_part(-hi[positive_lsf], sigma)
                )
                / width[positive_lsf]
            )
            hinge_average[positive_lsf] = torch.where(
                use_reverse[positive_lsf], reverse, direct
            )

    return hinge_average


def _paired_hinge_bin_averages(
    knot_lam: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    lsf_mode: str | None,
    off_grid: str = "clamp",
) -> torch.Tensor:
    """_hinge_bin_averages for pre-paired rows: [P,N] from [P,N] and [P].

    Same expressions as the [M,J,N] outer-product form, with the bin axis and
    the knot-row axis already collapsed into one pair axis by the caller. Every
    operation here is elementwise, so a pair row computes exactly the arithmetic
    its (bin, splat) cell computes in the outer-product form.
    """
    if off_grid not in ("clamp", "zero"):
        raise ValueError("off_grid must be 'clamp' or 'zero'.")
    if off_grid == "zero":
        width = (upper - lower)[:, None]
        bin_lower = torch.maximum(lower[:, None], knot_lam[:, :1])
        bin_upper = torch.minimum(upper[:, None], knot_lam[:, -1:])
        return _truncated_linear_bin_integral(
            bin_lower,
            bin_upper,
            width,
            lsf_sigma,
            knot_lam,
            knot_lam[:, -1:],
            torch.ones_like(knot_lam),
            -knot_lam,
            lsf_mode,
        ).squeeze(1)

    width = (upper - lower)[:, None]
    lo = lower[:, None] - knot_lam
    hi = upper[:, None] - knot_lam
    use_reverse = knot_lam < 0.5 * (lower + upper)[:, None]

    if lsf_mode == "zero":
        direct = (F.relu(hi).square() - F.relu(lo).square()) / (2.0 * width)
        reverse = (
            0.5 * (lower + upper)[:, None]
            - knot_lam
            + (F.relu(-lo).square() - F.relu(-hi).square()) / (2.0 * width)
        )
        return torch.where(use_reverse, reverse, direct)
    if lsf_mode == "positive":
        sigma = lsf_sigma[:, None]
        direct = (
            _integrated_positive_part(hi, sigma)
            - _integrated_positive_part(lo, sigma)
        ) / width
        reverse = (
            0.5 * (lower + upper)[:, None]
            - knot_lam
            + (
                _integrated_positive_part(-lo, sigma)
                - _integrated_positive_part(-hi, sigma)
            )
            / width
        )
        return torch.where(use_reverse, reverse, direct)

    hinge_average = torch.empty_like(lo)
    zero_lsf = lsf_sigma == 0
    if bool(torch.any(zero_lsf)):
        lo_zero = lo[zero_lsf]
        hi_zero = hi[zero_lsf]
        direct = (
            F.relu(hi_zero).square() - F.relu(lo_zero).square()
        ) / (2.0 * width[zero_lsf])
        reverse = (
            0.5 * (lower[zero_lsf] + upper[zero_lsf])[:, None]
            - knot_lam[zero_lsf]
            + (
                F.relu(-lo_zero).square() - F.relu(-hi_zero).square()
            )
            / (2.0 * width[zero_lsf])
        )
        hinge_average[zero_lsf] = torch.where(
            use_reverse[zero_lsf], reverse, direct
        )
    positive_lsf = ~zero_lsf
    if bool(torch.any(positive_lsf)):
        sigma = lsf_sigma[positive_lsf, None]
        direct = (
            _integrated_positive_part(hi[positive_lsf], sigma)
            - _integrated_positive_part(lo[positive_lsf], sigma)
        ) / width[positive_lsf]
        reverse = (
            0.5 * (lower[positive_lsf] + upper[positive_lsf])[:, None]
            - knot_lam[positive_lsf]
            + (
                _integrated_positive_part(-lo[positive_lsf], sigma)
                - _integrated_positive_part(-hi[positive_lsf], sigma)
            )
            / width[positive_lsf]
        )
        hinge_average[positive_lsf] = torch.where(
            use_reverse[positive_lsf], reverse, direct
        )
    return hinge_average


def _paired_zero_weights(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
) -> torch.Tensor:
    """Return the eager positive-LSF zero-extension weights."""
    del knot_values
    hinge_average = _paired_hinge_bin_averages(
        knot_lam, lower, upper, lsf_sigma, "positive", "zero"
    )
    scaled = (hinge_average[:, :-1] - hinge_average[:, 1:]) / (
        knot_lam[:, 1:] - knot_lam[:, :-1]
    )
    width = (upper - lower)[:, None]
    bin_lower = torch.maximum(lower[:, None], knot_lam[:, :1])
    bin_upper = torch.minimum(upper[:, None], knot_lam[:, -1:])
    base = _truncated_linear_bin_integral(
        bin_lower,
        bin_upper,
        width,
        lsf_sigma,
        knot_lam[:, :1],
        knot_lam[:, -1:],
        torch.zeros_like(knot_lam[:, :1]),
        torch.ones_like(knot_lam[:, :1]),
        "positive",
    ).squeeze(1)
    return torch.cat(
        (
            base - scaled[:, :1],
            scaled[:, :-1] - scaled[:, 1:],
            scaled[:, -1:],
        ),
        dim=1,
    ).clamp_min(0.0)


_GRAPHED_FIBER_BUCKETS = (131_072, 262_144, 393_216)
_GRAPHED_FIBER_CALLABLES: dict[tuple[int, int], object] = {}


def _graphed_zero_weights(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    rows, knots = inputs[0].shape
    if rows == 0 or knots != 96:
        return None
    bucket = next((size for size in _GRAPHED_FIBER_BUCKETS if rows <= size), None)
    if bucket is None:
        return None
    key = (inputs[0].device.index, bucket)
    graphed = _GRAPHED_FIBER_CALLABLES.get(key)
    if graphed is None:
        device = inputs[0].device
        sample_lam = torch.linspace(
            3000.0, 11000.0, knots, dtype=torch.float64, device=device
        ).expand(bucket, knots).clone().requires_grad_()
        sample_values = torch.zeros_like(sample_lam, requires_grad=True)
        sample_lower = torch.full(
            (bucket,), 3000.0, dtype=torch.float64, device=device
        )
        sample_upper = torch.full(
            (bucket,), 3100.0, dtype=torch.float64, device=device
        )
        sample_sigma = torch.ones(bucket, dtype=torch.float64, device=device)
        graphed = torch.cuda.make_graphed_callables(
            _paired_zero_weights,
            (
                sample_lam,
                sample_values,
                sample_lower,
                sample_upper,
                sample_sigma,
            ),
            num_warmup_iters=1,
            allow_unused_input=True,
        )
        _GRAPHED_FIBER_CALLABLES[key] = graphed

    padded = []
    for index, value in enumerate(inputs):
        shape = (bucket, knots) if value.ndim == 2 else (bucket,)
        target = torch.empty(shape, dtype=value.dtype, device=value.device)
        if index == 0:
            target.copy_(value[-1:].detach().expand(shape))
        elif index == 1:
            target.zero_()
        elif index == 2:
            target.copy_(inputs[0][-1, 0].detach().expand(shape))
        elif index == 3:
            target.copy_(inputs[0][-1, 1].detach().expand(shape))
        else:
            target.fill_(1.0)
        target[:rows].copy_(value)
        padded.append(target)
    return graphed(*padded)[:rows].clone()


def _paired_positive_hat_bin_average_eager(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    *,
    lsf_mode: str | None = None,
    off_grid: str = "clamp",
    blocks: Sequence[tuple[int, int]],
    graph_elementwise: bool = False,
) -> torch.Tensor:
    """positive_hat_bin_average for pre-paired rows: [P] from [P,N] and [P].

    The outer-product form builds an [M,J,N] weight tensor and contracts the
    knot axis; this builds the [P,N] rows the caller actually needs and
    contracts the same axis, so a pair row computes exactly the arithmetic its
    (bin, splat) cell computes.

    `blocks` gives the row ranges the caller would otherwise have passed as
    separate calls. The elementwise pipeline runs once over all P rows, but
    both reductions over the knot axis are taken block by block: a reduction
    picks its accumulation order from the shape it is handed, and a block of a
    few dozen rows does not reduce the same way as one of several thousand.
    """
    if knot_lam.ndim != 2 or knot_values.shape != knot_lam.shape:
        raise ValueError("knot_lam and knot_values must have matching [P,N] shapes.")
    if knot_lam.shape[1] < 2:
        raise ValueError("At least two wavelength knots are required.")
    if lower.ndim != 1 or upper.shape != lower.shape or lsf_sigma.shape != lower.shape:
        raise ValueError("lower, upper, and lsf_sigma must have matching [P] shapes.")
    if lower.shape[0] != knot_lam.shape[0]:
        raise ValueError("Paired hat averaging needs one bin edge pair per knot row.")
    if off_grid not in ("clamp", "zero"):
        raise ValueError("off_grid must be 'clamp' or 'zero'.")

    # Zero extension uses truncated moments at absolute wavelengths. Float32
    # cancellation can assign positive mass to distant knots after clipping.
    # Evaluate the moments and contraction in float64, retaining the public dtype.
    result_dtype = None
    if off_grid == "zero":
        tensors = (knot_lam, knot_values, lower, upper, lsf_sigma)
        result_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            result_dtype = torch.promote_types(result_dtype, tensor.dtype)
        knot_lam, knot_values, lower, upper, lsf_sigma = (
            tensor.to(torch.float64) for tensor in tensors
        )

    if off_grid == "zero" and lsf_mode == "positive":
        inputs64 = (knot_lam, knot_values, lower, upper, lsf_sigma)
        weights = (
            _graphed_zero_weights(inputs64) if graph_elementwise else None
        )
        if weights is None:
            weights = _paired_zero_weights(*inputs64)
        result = torch.cat(
            [(weights[a:b] * knot_values[a:b]).sum(dim=1) for a, b in blocks]
        )
        return result if result_dtype is None else result.to(result_dtype)

    hinge_average = _paired_hinge_bin_averages(
        knot_lam, lower, upper, lsf_sigma, lsf_mode, off_grid
    )
    scaled = (hinge_average[:, :-1] - hinge_average[:, 1:]) / (
        knot_lam[:, 1:] - knot_lam[:, :-1]
    )
    if off_grid == "zero":
        width = (upper - lower)[:, None]
        bin_lower = torch.maximum(lower[:, None], knot_lam[:, :1])
        bin_upper = torch.minimum(upper[:, None], knot_lam[:, -1:])
        base = _truncated_linear_bin_integral(
            bin_lower,
            bin_upper,
            width,
            lsf_sigma,
            knot_lam[:, :1],
            knot_lam[:, -1:],
            torch.zeros_like(knot_lam[:, :1]),
            torch.ones_like(knot_lam[:, :1]),
            lsf_mode,
        ).squeeze(1)
        weights = torch.cat(
            (
                base - scaled[:, :1],
                scaled[:, :-1] - scaled[:, 1:],
                scaled[:, -1:],
            ),
            dim=1,
        )
    else:
        weights = torch.cat(
            (
                1.0 - scaled[:, :1],
                scaled[:, :-1] - scaled[:, 1:],
                scaled[:, -1:],
            ),
            dim=1,
        )
    weights = weights.clamp_min(0.0)
    if off_grid == "clamp":
        weights = weights / torch.cat(
            [weights[a:b].sum(dim=1, keepdim=True) for a, b in blocks]
        ).clamp_min(torch.finfo(weights.dtype).tiny)
    result = torch.cat(
        [(weights[a:b] * knot_values[a:b]).sum(dim=1) for a, b in blocks]
    )
    return result if result_dtype is None else result.to(result_dtype)


@lru_cache(maxsize=1)
def _fiber_bin_kernels():
    """Import the Triton fiber-bin kernels only for an eligible CUDA call."""
    from . import fiber_bin_kernels

    return fiber_bin_kernels


class _FusedPairedPositiveHatBinAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, knot_lam, knot_values, lower, upper, lsf_sigma, blocks):
        ctx.save_for_backward(knot_lam, knot_values, lower, upper, lsf_sigma)
        ctx.blocks = blocks
        contribution = _fiber_bin_kernels().contributions(
            knot_lam, knot_values, lower, upper, lsf_sigma
        )
        block_results = [contribution[a:b].sum(dim=1) for a, b in blocks]
        result = (
            block_results[0]
            if len(block_results) == 1
            else torch.cat(block_results)
        )
        return result.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_result):
        saved = ctx.saved_tensors
        if (
            fused_fiber_bins_backward_enabled()
            and ctx.needs_input_grad[:5] == (True, True, False, False, False)
            and torch.__version__.split("+")[0] == "2.11.0"
            and saved[0].shape[0] > 0
            and saved[0].shape[1] == 96
            and ctx.blocks == ((0, saved[0].shape[0]),)
        ):
            from .fiber_bin_vjp_kernels import gradients

            grad_lam, grad_values = gradients(*saved, grad_result.contiguous())
            return grad_lam, grad_values, None, None, None, None
        with torch.enable_grad():
            references = tuple(
                tensor.detach().requires_grad_(True) for tensor in saved
            )
            result = _paired_positive_hat_bin_average_eager(
                *references,
                lsf_mode="positive",
                off_grid="zero",
                blocks=ctx.blocks,
            )
            gradients = torch.autograd.grad(result, references, grad_result)
        return (*gradients, None)


def paired_positive_hat_bin_average(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    *,
    lsf_mode: str | None = None,
    off_grid: str = "clamp",
    blocks: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Evaluate paired hat rows through the selected exact fiber-bin path."""
    mode = fiber_bins_mode()
    if mode == "graphed":
        eligible = (
            knot_lam.is_cuda
            and torch.is_grad_enabled()
            and lsf_mode == "positive"
            and off_grid == "zero"
            and tuple(
                tensor.requires_grad
                for tensor in (knot_lam, knot_values, lower, upper, lsf_sigma)
            )
            == (True, True, False, False, False)
        )
        return _paired_positive_hat_bin_average_eager(
            knot_lam,
            knot_values,
            lower,
            upper,
            lsf_sigma,
            lsf_mode=lsf_mode,
            off_grid=off_grid,
            blocks=blocks,
            graph_elementwise=eligible,
        )
    if mode == "fused" and knot_lam.is_cuda:
        kernels = _fiber_bin_kernels()
        if kernels.applies(
            knot_lam, knot_values, lower, upper, lsf_sigma, lsf_mode, off_grid
        ):
            # Preserve the eager function's validation before entering a custom
            # autograd node. The production caller supplies one complete block.
            if knot_lam.ndim != 2 or knot_values.shape != knot_lam.shape:
                raise ValueError(
                    "knot_lam and knot_values must have matching [P,N] shapes."
                )
            if knot_lam.shape[1] < 2:
                raise ValueError("At least two wavelength knots are required.")
            if (
                lower.ndim != 1
                or upper.shape != lower.shape
                or lsf_sigma.shape != lower.shape
            ):
                raise ValueError(
                    "lower, upper, and lsf_sigma must have matching [P] shapes."
                )
            if lower.shape[0] != knot_lam.shape[0]:
                raise ValueError(
                    "Paired hat averaging needs one bin edge pair per knot row."
                )
            return _FusedPairedPositiveHatBinAverage.apply(
                knot_lam, knot_values, lower, upper, lsf_sigma, tuple(blocks)
            )
    return _paired_positive_hat_bin_average_eager(
        knot_lam,
        knot_values,
        lower,
        upper,
        lsf_sigma,
        lsf_mode=lsf_mode,
        off_grid=off_grid,
        blocks=blocks,
    )


def positive_hat_bin_average(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lsf_sigma: torch.Tensor,
    *,
    validate: bool = True,
    lsf_mode: str | None = None,
    off_grid: str = "clamp",
) -> torch.Tensor:
    """Nonnegative reformulation of piecewise_linear_bin_average.

    Same integral, factored as a hat-basis contraction: each bin average is
    a nonnegative weighted average of the knot values, so nonnegative
    spectra yield nonnegative bin averages in any precision. The weights
    are adjacent-knot differences of the smoothed hinge integrals,
    clamped at zero against roundoff and
    renormalized to a partition of unity for off_grid="clamp". For
    off_grid="zero", the truncated weights are left unnormalized and computed
    in float64 to avoid cancellation at physical wavelengths. The result is
    cast back to the promoted input dtype.
    """
    if knot_lam.ndim != 2 or knot_values.shape != knot_lam.shape:
        raise ValueError("knot_lam and knot_values must have matching [J,N] shapes.")
    if knot_lam.shape[1] < 2:
        raise ValueError("At least two wavelength knots are required.")
    if lower.ndim != 1 or upper.shape != lower.shape or lsf_sigma.shape != lower.shape:
        raise ValueError("lower, upper, and lsf_sigma must have matching [M] shapes.")
    if off_grid not in ("clamp", "zero"):
        raise ValueError("off_grid must be 'clamp' or 'zero'.")
    if validate:
        if torch.any(knot_lam[:, 1:] <= knot_lam[:, :-1]):
            raise ValueError("Wavelength knots must be strictly increasing.")
        if torch.any(upper <= lower):
            raise ValueError("Spectral bin upper edges must exceed lower edges.")
        if torch.any(lsf_sigma < 0):
            raise ValueError("LSF sigma must be non-negative.")

    # Zero extension uses truncated moments at absolute wavelengths. Float32
    # cancellation can assign positive mass to distant knots after clipping.
    # Evaluate the moments and contraction in float64, retaining the public dtype.
    result_dtype = None
    if off_grid == "zero":
        tensors = (knot_lam, knot_values, lower, upper, lsf_sigma)
        result_dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            result_dtype = torch.promote_types(result_dtype, tensor.dtype)
        knot_lam, knot_values, lower, upper, lsf_sigma = (
            tensor.to(torch.float64) for tensor in tensors
        )

    hinge_average = _hinge_bin_averages(
        knot_lam, lower, upper, lsf_sigma, lsf_mode, off_grid
    )
    scaled = (hinge_average[:, :, :-1] - hinge_average[:, :, 1:]) / (
        knot_lam[:, 1:] - knot_lam[:, :-1]
    )[None, :, :]
    if off_grid == "zero":
        width = (upper - lower)[:, None, None]
        bin_lower = torch.maximum(
            lower[:, None, None], knot_lam[None, :, :1]
        )
        bin_upper = torch.minimum(
            upper[:, None, None], knot_lam[None, :, -1:]
        )
        base = _truncated_linear_bin_integral(
            bin_lower,
            bin_upper,
            width,
            lsf_sigma,
            knot_lam[None, :, :1],
            knot_lam[None, :, -1:],
            torch.zeros_like(knot_lam)[None, :, :1],
            torch.ones_like(knot_lam)[None, :, :1],
            lsf_mode,
        )
        weights = torch.cat(
            (
                base - scaled[:, :, :1],
                scaled[:, :, :-1] - scaled[:, :, 1:],
                scaled[:, :, -1:],
            ),
            dim=2,
        )
    else:
        weights = torch.cat(
            (
                1.0 - scaled[:, :, :1],
                scaled[:, :, :-1] - scaled[:, :, 1:],
                scaled[:, :, -1:],
            ),
            dim=2,
        )
    weights = weights.clamp_min(0.0)
    if off_grid == "clamp":
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
    result = (weights * knot_values[None, :, :]).sum(dim=2)
    return result if result_dtype is None else result.to(result_dtype)


def _piecewise_linear_response_moments(
    x: torch.Tensor,
    response_lam: torch.Tensor,
    throughput: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative integrals of throughput*lambda and throughput*lambda**2."""
    clamped_x = x.clamp(response_lam[0], response_lam[-1])
    segment = torch.searchsorted(
        response_lam, clamped_x.contiguous(), right=True
    ).sub(1).clamp(0, response_lam.numel() - 2)
    lam0 = response_lam[segment]
    response0 = throughput[segment]
    slope = (
        throughput[1:] - throughput[:-1]
    ) / (response_lam[1:] - response_lam[:-1])
    local_slope = slope[segment]
    u = clamped_x - lam0

    segment_moment0 = (
        response0 * lam0 * u
        + 0.5 * (response0 + local_slope * lam0) * u.square()
        + (local_slope / 3.0) * u.pow(3)
    )
    segment_moment1 = (
        response0 * lam0.square() * u
        + (response0 * lam0 + 0.5 * local_slope * lam0.square()) * u.square()
        + (response0 / 3.0 + 2.0 * local_slope * lam0 / 3.0) * u.pow(3)
        + 0.25 * local_slope * u.pow(4)
    )

    widths = response_lam[1:] - response_lam[:-1]
    left = response_lam[:-1]
    response_left = throughput[:-1]
    segment_moment0_full = (
        response_left * left * widths
        + 0.5 * (response_left + slope * left) * widths.square()
        + (slope / 3.0) * widths.pow(3)
    )
    segment_moment1_full = (
        response_left * left.square() * widths
        + (response_left * left + 0.5 * slope * left.square()) * widths.square()
        + (response_left / 3.0 + 2.0 * slope * left / 3.0) * widths.pow(3)
        + 0.25 * slope * widths.pow(4)
    )
    prefix0 = torch.cat(
        (torch.zeros_like(segment_moment0_full[:1]), segment_moment0_full.cumsum(0))
    )
    prefix1 = torch.cat(
        (torch.zeros_like(segment_moment1_full[:1]), segment_moment1_full.cumsum(0))
    )
    return prefix0[segment] + segment_moment0, prefix1[segment] + segment_moment1


def piecewise_linear_band_average(
    knot_lam: torch.Tensor,
    knot_values: torch.Tensor,
    response_lam: torch.Tensor,
    throughput: torch.Tensor,
) -> torch.Tensor:
    """Exactly average clamped piecewise-linear spectra through a bandpass.

    The bandpass is linearly interpolated in wavelength and photon weighted by
    wavelength. Each spectrum is linearly interpolated between its own knots
    and held constant beyond its first and last knot.
    """
    if knot_lam.ndim != 2 or knot_values.shape != knot_lam.shape:
        raise ValueError("knot_lam and knot_values must have matching [J,N] shapes.")
    if knot_lam.shape[1] < 2:
        raise ValueError("At least two wavelength knots are required.")
    if response_lam.ndim != 1 or throughput.shape != response_lam.shape:
        raise ValueError(
            "response_lam and throughput must have matching one-dimensional shapes."
        )
    if response_lam.numel() < 2:
        raise ValueError("At least two bandpass samples are required.")
    if torch.any(knot_lam[:, 1:] <= knot_lam[:, :-1]):
        raise ValueError("Wavelength knots must be strictly increasing.")
    if torch.any(response_lam[1:] <= response_lam[:-1]):
        raise ValueError("Bandpass wavelengths must be strictly increasing.")
    if not torch.isfinite(knot_lam).all() or not torch.isfinite(knot_values).all():
        raise ValueError("Spectrum knots and values must be finite.")
    if not torch.isfinite(response_lam).all() or not torch.isfinite(throughput).all():
        raise ValueError("Bandpass wavelengths and throughput must be finite.")

    weights = _piecewise_linear_band_weights(
        knot_lam, response_lam, throughput
    )
    return torch.sum(knot_values * weights, dim=1)


def _piecewise_linear_band_weights(
    knot_lam: torch.Tensor,
    response_lam: torch.Tensor,
    throughput: torch.Tensor,
) -> torch.Tensor:
    throughput = throughput.clamp_min(0.0)
    response_start = response_lam[0]
    response_stop = response_lam[-1]
    total_weight, _ = _piecewise_linear_response_moments(
        response_stop, response_lam, throughput
    )
    if bool(total_weight <= 0):
        raise ValueError("Bandpass must have positive photon-weighted throughput.")

    clamped = knot_lam.clamp(response_start, response_stop)
    moment0, moment1 = _piecewise_linear_response_moments(
        clamped, response_lam, throughput
    )
    interval_mask = clamped[:, 1:] > clamped[:, :-1]
    delta_moment0 = moment0[:, 1:] - moment0[:, :-1]
    delta_moment1 = moment1[:, 1:] - moment1[:, :-1]
    right_weight = (
        delta_moment1 - knot_lam[:, :-1] * delta_moment0
    ) / (
        knot_lam[:, 1:] - knot_lam[:, :-1]
    )
    left_weight = delta_moment0 - right_weight
    left_weight = torch.where(
        interval_mask, left_weight, torch.zeros_like(left_weight)
    )
    right_weight = torch.where(
        interval_mask, right_weight, torch.zeros_like(right_weight)
    )
    weights = F.pad(left_weight, (0, 1)) + F.pad(right_weight, (1, 0))

    first_weight = moment0[:, 0]
    last_weight = moment0[:, -1]
    edge_weights = F.pad(first_weight[:, None], (0, knot_lam.shape[1] - 1))
    edge_weights = edge_weights + F.pad(
        (total_weight - last_weight)[:, None],
        (knot_lam.shape[1] - 1, 0),
    )
    return (weights + edge_weights) / total_weight


def _delta_moment_tables(
    response_lam: torch.Tensor, throughput: torch.Tensor, C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Piecewise polynomials for the response moments at knot_lam = C_g * t.

    Written in delta = t - 1 (t stays near 1, so this basis is well
    conditioned in float32). Bucket b of knot g covers
    tau[g, b-1] <= t < tau[g, b]; bucket 0 is identically zero and the last
    bucket holds the full-response totals.
    """
    L = response_lam.detach().cpu().to(torch.float64).numpy()
    T = np.clip(throughput.detach().cpu().to(torch.float64).numpy(), 0.0, None)
    Cn = C.detach().cpu().to(torch.float64).numpy()
    S = L.size - 1
    G = Cn.size
    slope = (T[1:] - T[:-1]) / (L[1:] - L[:-1])
    widths = L[1:] - L[:-1]
    m0_full = (
        T[:-1] * L[:-1] * widths
        + 0.5 * (T[:-1] + slope * L[:-1]) * widths**2
        + (slope / 3.0) * widths**3
    )
    m1_full = (
        T[:-1] * L[:-1] ** 2 * widths
        + (T[:-1] * L[:-1] + 0.5 * slope * L[:-1] ** 2) * widths**2
        + (T[:-1] / 3.0 + 2.0 * slope * L[:-1] / 3.0) * widths**3
        + 0.25 * slope * widths**4
    )
    prefix0 = np.concatenate(([0.0], np.cumsum(m0_full)))
    prefix1 = np.concatenate(([0.0], np.cumsum(m1_full)))

    tau = L[None, :] / Cn[:, None]
    coef0 = np.zeros((G, S + 2, 4))
    coef1 = np.zeros((G, S + 2, 5))
    for g in range(G):
        # u = C_g t - L_i = (C_g - L_i) + C_g delta
        u_base = np.polynomial.Polynomial([0.0, Cn[g]])
        for i in range(S):
            a = np.array([
                prefix0[i],
                T[i] * L[i],
                0.5 * (T[i] + slope[i] * L[i]),
                slope[i] / 3.0,
            ])
            b = np.array([
                prefix1[i],
                T[i] * L[i] ** 2,
                T[i] * L[i] + 0.5 * slope[i] * L[i] ** 2,
                T[i] / 3.0 + 2.0 * slope[i] * L[i] / 3.0,
                0.25 * slope[i],
            ])
            u_of_delta = np.polynomial.Polynomial([Cn[g] - L[i]]) + u_base
            p0 = np.polynomial.Polynomial(a)(u_of_delta).coef
            p1 = np.polynomial.Polynomial(b)(u_of_delta).coef
            coef0[g, i + 1, : p0.size] = p0
            coef1[g, i + 1, : p1.size] = p1
        coef0[g, S + 1, 0] = prefix0[-1]
        coef1[g, S + 1, 0] = prefix1[-1]
    return (
        torch.from_numpy(tau),
        torch.from_numpy(coef0),
        torch.from_numpy(coef1),
        float(prefix0[-1]),
    )


_BAND_WEIGHT_TABLE_CACHE: dict = {}
_TENSOR_CONTENT_HASH = WeakTensorKeyDictionary()
_INTEGRATION_GRID_CACHE = WeakTensorKeyDictionary()


def _band_integration_grid(
    grid: torch.Tensor, log_mid: float, scale: float
) -> torch.Tensor:
    """exp(log_mid + scale * grid), cached per (grid, bounds).

    The grid buffer and wavelength bounds are fixed for a run; rebuilding C
    every step would defeat the content-hash memo below by producing a fresh
    tensor object each call.
    """
    key = (float(log_mid), float(scale), grid._version)
    cached = _INTEGRATION_GRID_CACHE.get(grid)
    if cached is not None and cached[0] == key:
        return cached[1]
    C = torch.exp(log_mid + scale * grid)
    # Keep only the latest (bounds, version) per grid so in-place grid
    # mutation or varying bounds cannot grow the cache without bound.
    _INTEGRATION_GRID_CACHE[grid] = (key, C)
    return C


def _tensor_content_hash(tensor: torch.Tensor) -> int:
    """Content hash memoized per tensor object, so the device-to-host copy
    happens once per distinct tensor instead of once per training step."""
    version = tensor._version
    cached = _TENSOR_CONTENT_HASH.get(tensor)
    if cached is not None and cached[0] == version:
        return cached[1]
    value = hash(tensor.detach().cpu().to(torch.float64).numpy().tobytes())
    _TENSOR_CONTENT_HASH[tensor] = (version, value)
    return value


def _band_weight_tables(
    response_lam: torch.Tensor, throughput: torch.Tensor, C: torch.Tensor,
    *, dtype: torch.dtype, device: torch.device,
) -> tuple:
    key = (
        _tensor_content_hash(response_lam),
        _tensor_content_hash(throughput),
        _tensor_content_hash(C),
        dtype,
        str(device),
    )
    cached = _BAND_WEIGHT_TABLE_CACHE.get(key)
    if cached is None:
        tau, coef0, coef1, total0 = _delta_moment_tables(response_lam, throughput, C)
        cached = (
            tau.to(device=device),
            coef0.to(device=device, dtype=dtype),
            coef1.to(device=device, dtype=dtype),
            total0,
            float(response_lam[0]),
            float(response_lam[-1]),
        )
        _BAND_WEIGHT_TABLE_CACHE[key] = cached
    return cached


def _classify_band_knots(
    tau: torch.Tensor, t_lo: float, t_hi: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split knots into static (one bucket over [t_lo, t_hi]) and dynamic.

    Hoisted out of the per-chunk weight evaluation: classifying against the
    global t range instead of each chunk's costs nothing mathematically (a
    knot static per-chunk but dynamic globally evaluates the same polynomial
    through the gather path) and removes the min/max/nonzero host syncs from
    every chunk x band call.
    """
    G = tau.shape[0]
    lo_b = torch.searchsorted(
        tau, torch.full((G, 1), t_lo, dtype=tau.dtype, device=tau.device), right=True
    )[:, 0]
    hi_b = torch.searchsorted(
        tau, torch.full((G, 1), t_hi, dtype=tau.dtype, device=tau.device), right=True
    )[:, 0]
    static = lo_b == hi_b
    s_idx = torch.nonzero(static, as_tuple=True)[0]
    d_idx = torch.nonzero(~static, as_tuple=True)[0]
    return s_idx, d_idx, lo_b


def _tabulated_weights_and_grad(
    t: torch.Tensor, C: torch.Tensor, tau: torch.Tensor,
    coef0: torch.Tensor, coef1: torch.Tensor, total0: float,
    R0: float, R1: float, *, need_grad: bool,
    classification: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Band weights [J, G] and, when needed, their derivative in t.

    Knots whose response-segment bucket is constant over the batch's t range
    evaluate one fixed polynomial as a matmul over the delta power basis; the
    remaining knots gather their per-splat bucket coefficients. Exact for all
    t > 0.
    """
    dtype = t.dtype
    G = tau.shape[0]
    delta = t - 1.0
    tt = t[:, None]
    if classification is None:
        classification = _classify_band_knots(tau, float(t.min()), float(t.max()))
    s_idx, d_idx, lo_b = classification

    J = t.shape[0]
    M0 = torch.empty(J, G, dtype=dtype, device=t.device)
    M1 = torch.empty(J, G, dtype=dtype, device=t.device)
    dM0 = torch.empty(J, G, dtype=dtype, device=t.device) if need_grad else None
    dM1 = torch.empty(J, G, dtype=dtype, device=t.device) if need_grad else None

    one = torch.ones_like(delta)
    pow5 = torch.stack([one, delta, delta * delta, delta**3, delta**4], dim=1)
    if need_grad:
        dpow5 = torch.stack(
            [torch.zeros_like(delta), one, 2 * delta, 3 * delta * delta, 4 * delta**3],
            dim=1,
        )

    if s_idx.numel():
        c0 = coef0[s_idx, lo_b[s_idx]]
        c1 = coef1[s_idx, lo_b[s_idx]]
        M0[:, s_idx] = pow5[:, :4] @ c0.T
        M1[:, s_idx] = pow5 @ c1.T
        if need_grad:
            dM0[:, s_idx] = dpow5[:, :4] @ c0.T
            dM1[:, s_idx] = dpow5 @ c1.T

    if d_idx.numel():
        Gd = d_idx.numel()
        bucket = torch.searchsorted(
            tau[d_idx], tt.expand(J, Gd).T.contiguous().to(tau.dtype), right=True
        ).T
        gexp = d_idx[None, :].expand(J, Gd)
        c0 = coef0[gexp, bucket]
        c1 = coef1[gexp, bucket]
        dd = delta[:, None]
        m0 = c0[..., 3]
        for k in (2, 1, 0):
            m0 = m0 * dd + c0[..., k]
        m1 = c1[..., 4]
        for k in (3, 2, 1, 0):
            m1 = m1 * dd + c1[..., k]
        M0[:, d_idx] = m0
        M1[:, d_idx] = m1
        if need_grad:
            dm0 = c0[..., 3] * 3
            for k in (2, 1):
                dm0 = dm0 * dd + c0[..., k] * k
            dm1 = c1[..., 4] * 4
            for k in (3, 2, 1):
                dm1 = dm1 * dd + c1[..., k] * k
            dM0[:, d_idx] = dm0
            dM1[:, d_idx] = dm1

    knot = C[None, :] * tt
    dC = (C[1:] - C[:-1])[None, :]
    d0 = M0[:, 1:] - M0[:, :-1]
    d1 = M1[:, 1:] - M1[:, :-1]
    denom = dC * tt
    right = (d1 - knot[:, :-1] * d0) / denom
    left = d0 - right
    lower = knot[:, :-1].clamp(R0, R1)
    upper = knot[:, 1:].clamp(R0, R1)
    mask = upper > lower
    zero = torch.zeros((), dtype=dtype, device=t.device)
    right_masked = torch.where(mask, right, zero)
    left_masked = torch.where(mask, left, zero)
    w = F.pad(left_masked, (0, 1)) + F.pad(right_masked, (1, 0))
    w[:, 0] += M0[:, 0]
    w[:, -1] += total0 - M0[:, -1]
    w = w / total0

    if not need_grad:
        return w, None

    dd0 = dM0[:, 1:] - dM0[:, :-1]
    dd1 = dM1[:, 1:] - dM1[:, :-1]
    Cg = C[None, :-1]
    dright = (dd1 - Cg * d0 - knot[:, :-1] * dd0) / denom - right / tt
    dleft = dd0 - dright
    dright = torch.where(mask, dright, zero)
    dleft = torch.where(mask, dleft, zero)
    dw = F.pad(dleft, (0, 1)) + F.pad(dright, (1, 0))
    dw[:, 0] += dM0[:, 0]
    dw[:, -1] -= dM0[:, -1]
    return w, dw / total0


class _TabulatedBandWeights(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, t, C, tau, coef0, coef1, total0, R0, R1,
        s_idx=None, d_idx=None, lo_b=None,
    ):
        need_grad = bool(ctx.needs_input_grad[0])
        classification = (
            None if s_idx is None else (s_idx, d_idx, lo_b)
        )
        w, dw = _tabulated_weights_and_grad(
            t, C, tau, coef0, coef1, total0, R0, R1, need_grad=need_grad,
            classification=classification,
        )
        if need_grad:
            ctx.save_for_backward(dw)
        return w

    @staticmethod
    def backward(ctx, grad_w):
        (dw,) = ctx.saved_tensors
        return (
            (grad_w * dw).sum(dim=1),
            None, None, None, None, None, None, None, None, None, None,
        )


_HAT_BAND_TABLE_CACHE: dict = {}


def _hat_band_tables(
    response_lam: torch.Tensor, throughput: torch.Tensor,
    *, dtype: torch.dtype, device: torch.device,
) -> tuple:
    """Local per-segment tables for the sign-safe hat band weights.

    Everything is built in float64 from the response curve alone (the knot
    grid enters only at evaluation): segment left edges/values/slopes of the
    photon-weighted response R(lam) = lam * T(lam), and prefix sums of the
    per-segment integrals of R and lam*R. Prefix sums are stored as
    double-single (hi + lo residual) pairs so adjacent-prefix differences
    keep float64-grade accuracy in float32 arithmetic; under float64 the lo
    parts are exactly zero and the pairs degrade to plain prefixes.
    """
    key = (
        _tensor_content_hash(response_lam),
        _tensor_content_hash(throughput),
        dtype,
        str(device),
    )
    cached = _HAT_BAND_TABLE_CACHE.get(key)
    if cached is None:
        L = response_lam.detach().cpu().to(torch.float64)
        T = throughput.detach().cpu().to(torch.float64).clamp_min(0.0)
        if L.numel() < 2:
            raise ValueError("At least two bandpass samples are required.")
        if not torch.isfinite(L).all() or not torch.isfinite(T).all():
            raise ValueError("Bandpass wavelengths and throughput must be finite.")
        if torch.any(L[1:] <= L[:-1]):
            raise ValueError("Bandpass wavelengths must be strictly increasing.")
        widths = L[1:] - L[:-1]
        slope = (T[1:] - T[:-1]) / widths
        left = L[:-1]
        t_left = T[:-1]
        m0 = (
            t_left * left * widths
            + 0.5 * (t_left + slope * left) * widths.square()
            + (slope / 3.0) * widths.pow(3)
        )
        m1 = (
            t_left * left.square() * widths
            + (t_left * left + 0.5 * slope * left.square()) * widths.square()
            + (t_left / 3.0 + 2.0 * slope * left / 3.0) * widths.pow(3)
            + 0.25 * slope * widths.pow(4)
        )
        prefix0 = torch.cat((torch.zeros(1, dtype=torch.float64), m0.cumsum(0)))
        prefix1 = torch.cat((torch.zeros(1, dtype=torch.float64), m1.cumsum(0)))
        if not float(prefix0[-1]) > 0.0:
            raise ValueError("Bandpass must have positive photon-weighted throughput.")

        def _split(prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hi = prefix.to(dtype)
            lo = (prefix - hi.to(torch.float64)).to(dtype)
            return hi.to(device), lo.to(device)

        p0_hi, p0_lo = _split(prefix0)
        p1_hi, p1_lo = _split(prefix1)
        cached = (
            L.to(device),
            L.to(device=device, dtype=dtype),
            T.to(device=device, dtype=dtype),
            slope.to(device=device, dtype=dtype),
            p0_hi, p0_lo, p1_hi, p1_lo,
            float(prefix0[-1]),
            float(L[0]),
            float(L[-1]),
        )
        _HAT_BAND_TABLE_CACHE[key] = cached
    return cached


_SORTED_BANDPASS_CACHE = WeakTensorKeyDictionary()


def _sorted_bandpass(
    lam_tensor: torch.Tensor, throughput_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """argsort-ordered (lam, throughput), memoized per source tensor pair.

    The sort itself is cheap, but its outputs are fresh tensor objects every
    call, which makes the content-hash memo behind the band tables miss and
    forces a device-to-host copy per bandpass per step. Callers pass the same
    cached bandpass tensors on every step, so key on those objects (and their
    versions) and hand back the same sorted tensors.
    """
    key = (
        lam_tensor._version,
        id(throughput_tensor),
        throughput_tensor._version,
    )
    cached = _SORTED_BANDPASS_CACHE.get(lam_tensor)
    if cached is not None and cached[0] == key:
        return cached[2]
    order = torch.argsort(lam_tensor)
    sorted_pair = (lam_tensor[order], throughput_tensor[order])
    # Keep a strong reference to the throughput tensor so its id cannot be
    # reused by a different tensor while this entry is live.
    _SORTED_BANDPASS_CACHE[lam_tensor] = (key, throughput_tensor, sorted_pair)
    return sorted_pair


_HAT_STACK_CACHE = WeakTensorKeyDictionary()


def _hat_stacked_table(*parts: torch.Tensor) -> torch.Tensor:
    """torch.stack(parts), memoized on the part tensors.

    The parts come from the per-band table cache and never change, so the
    stack is built once per band per run. Gathering every row in one indexing
    call then reads the int64 segment index once instead of once per row, and
    that index is the widest operand in the band evaluation.
    """
    head = parts[0]
    key = tuple((id(part), part._version) for part in parts)
    cached = _HAT_STACK_CACHE.get(head)
    if cached is not None and cached[0] == key:
        return cached[2]
    table = torch.stack(parts)
    # Hold the tail parts so their ids cannot be reused while this entry is
    # live; the head is the weak key and must not be referenced here.
    _HAT_STACK_CACHE[head] = (key, parts[1:], table)
    return table


_HAT_INTERIOR_BOUNDS_CACHE = WeakTensorKeyDictionary()


def _hat_interior_bounds(L64: torch.Tensor) -> torch.Tensor:
    """The response boundaries without the first one, contiguous and cached."""
    cached = _HAT_INTERIOR_BOUNDS_CACHE.get(L64)
    if cached is not None and cached[0] == L64._version:
        return cached[1]
    interior = L64[1:].contiguous()
    _HAT_INTERIOR_BOUNDS_CACHE[L64] = (L64._version, interior)
    return interior


_HAT_SEGMENT_TABLE_CACHE = WeakTensorKeyDictionary()


def _hat_segment_table(
    Ld: torch.Tensor, Td: torch.Tensor, sloped: torch.Tensor
) -> torch.Tensor:
    """[3, S+1] stack of segment left edge, response value and slope.

    The slope array has one entry per segment against one per node for the
    other two, so it carries one unused trailing entry here; segment indices
    are clamped to S-1 and never reach it.
    """
    key = (Ld._version, id(Td), Td._version, id(sloped), sloped._version)
    cached = _HAT_SEGMENT_TABLE_CACHE.get(Ld)
    if cached is not None and cached[0] == key:
        return cached[2]
    table = torch.stack((Ld, Td, F.pad(sloped, (0, 1))))
    _HAT_SEGMENT_TABLE_CACHE[Ld] = (key, (Td, sloped), table)
    return table


_HAT_WINDOW_CACHE: dict = {}


def _hat_window_width(C: torch.Tensor, R0: float, R1: float) -> int:
    """Fixed per-splat knot-window width for a band.

    For any warp t, the knots whose hats can overlap [R0, R1] span at most
    the largest run of knots inside a wavelength ratio R1/R0, independent of
    t. Three knots of slack absorb the searchsorted boundary and float32
    rounding of the per-splat start index.
    """
    key = (_tensor_content_hash(C), float(R0), float(R1))
    cached = _HAT_WINDOW_CACHE.get(key)
    if cached is None:
        c = C.detach().double().cpu()
        ratio = float(R1) / float(R0)
        ends = torch.searchsorted(c, c * ratio, right=True)
        width = int((ends - torch.arange(c.numel())).max()) + 3
        cached = min(width, int(c.numel()))
        _HAT_WINDOW_CACHE[key] = cached
    return cached


def _hat_edge_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """[n, W] stagger-sum of two [n, W-1] interval arrays.

    Same values as F.pad(left, (0, 1)) + F.pad(right, (1, 0)), written
    directly into one buffer instead of materializing two padded copies and
    adding them. The end columns are pure copies where the padded form added
    an exact zero.
    """
    out = torch.empty(
        left.shape[0], left.shape[1] + 1, dtype=left.dtype, device=left.device
    )
    out[:, 0] = left[:, 0]
    torch.add(left[:, 1:], right[:, :-1], out=out[:, 1:-1])
    out[:, -1] = right[:, -1]
    return out


@lru_cache(maxsize=1)
def _hat_band_kernels():
    """The fused Triton band-weight kernels, imported on first CUDA use."""
    from . import hat_band_kernels

    return hat_band_kernels


def _hat_fused_backend(t: torch.Tensor, window: int, G: int):
    """The fused kernels when they reproduce the eager chain, else None."""
    if not fused_band_weights_enabled() or not t.is_cuda:
        return None
    kernels = _hat_band_kernels()
    return kernels if kernels.applies(t, window, G) else None


def _hat_window_index(
    t: torch.Tensor, G: int, start: torch.Tensor | None, window: int
) -> torch.Tensor:
    """[J, window] knot indices covering each splat's evaluation window."""
    if start is None:
        return torch.arange(G, device=t.device)[None, :].expand(t.shape[0], G)
    return start[:, None] + torch.arange(window, device=t.device)[None, :]


def _hat_weights_and_grad(
    t: torch.Tensor, C: torch.Tensor,
    L64: torch.Tensor, Ld: torch.Tensor, Td: torch.Tensor, sloped: torch.Tensor,
    p0_hi: torch.Tensor, p0_lo: torch.Tensor,
    p1_hi: torch.Tensor, p1_lo: torch.Tensor,
    total0: float, R0: float, R1: float, *, need_grad: bool,
    window: int | None = None, drop_off_grid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Hat-basis band weights [J, G] and, when needed, their derivative in t.

    Same operator as the tabulated path (photon-weighted piecewise-linear
    bandpass, spectra linear between knots C_g * t and constant beyond, all
    integrals clamped to the response range), but assembled from local
    nonnegative parts: exact Simpson pieces inside the boundary response
    segments plus double-single prefix differences over complete segments.
    No cumulative moment is ever evaluated at an arbitrary knot position;
    the only remaining subtraction is the centered first moment, whose
    float32 amplification is bounded by the knot-spacing-to-wavelength
    ratio (~1e-4 worst case, versus 1e-1 for the cumulative tables), and
    the result is clamped and renormalized to an exact partition of unity. The
    derivative comes from the Leibniz rule on each interval integral: only
    boundary response values and the already-computed masses appear.

    Each splat only evaluates the fixed-width knot window that can overlap
    the response for its own t (from _hat_window_width); the skipped knots
    have exactly zero weight, so results are identical to a full evaluation.

    drop_off_grid changes what happens to the response mass the spectrum's
    knot span does not cover. By default that mass rides the constant
    extrapolation onto the terminal knots and the weights are a partition of
    unity. With it set, the mass is dropped instead of dumped, and the weights
    are left unnormalized: they sum to the fraction of the band's photon
    response that lands on the knot span, which is the fraction of the band
    the model actually models.
    """
    dtype = t.dtype
    G = C.shape[0]
    S = Ld.shape[0] - 1
    zero = torch.zeros((), dtype=dtype, device=t.device)
    if window is None:
        window = _hat_window_width(C, R0, R1)
    if window >= G:
        window = G
        start = None
    else:
        start = torch.searchsorted(
            C.detach(), (R0 / t).to(C.dtype).contiguous(), right=True
        )
        start = (start - 2).clamp(0, G - window)

    segment_table = _hat_segment_table(Ld, Td, sloped)

    def gather_segment(k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # One gather over the stacked table rather than three: the segment
        # index is the widest operand here (int64, twice the payload width),
        # and three separate gathers read it three times.
        rows = segment_table[:, k]
        return rows[0], rows[1], rows[2]

    def response_at(x: torch.Tensor, sg: tuple[torch.Tensor, ...]) -> torch.Tensor:
        lk, tk, sk = sg
        return x * (tk + sk * (x - lk))

    def simpson_panel(
        p: torch.Tensor, q: torch.Tensor, sg: tuple[torch.Tensor, ...],
        *, r_p: torch.Tensor | None = None, r_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Panel scale, midpoint, and the three photon-response samples.

        Split out of `simpson` because every panel is integrated twice, once
        against each hat ramp, and the response samples are the same both
        times; the derivative below reuses two of them a third time. These are
        the widest tensors in the band evaluation, so computing each once
        instead of two or three times is most of this routine's traffic.
        """
        m = 0.5 * (p + q)
        return (
            (q - p) / 6.0,
            m,
            response_at(p, sg) if r_p is None else r_p,
            response_at(m, sg),
            response_at(q, sg) if r_q is None else r_q,
        )

    def simpson(
        panel: tuple[torch.Tensor, ...],
        p: torch.Tensor, q: torch.Tensor,
        ramp_ref: torch.Tensor | None, negate_ramp: bool = False,
    ) -> torch.Tensor:
        h6, m, r_p, r_m, r_q = panel
        if ramp_ref is None:
            return h6 * (r_p + 4.0 * r_m + r_q)
        f_p = r_p * (p - ramp_ref)
        f_m = r_m * (m - ramp_ref)
        f_q = r_q * (q - ramp_ref)
        # The reversed ramp negates all three samples. Carrying the sign on
        # the panel scale instead costs one pass rather than three, and IEEE
        # negation is exact, so (-h6) * s is bit for bit h6 * (-s).
        scale = -h6 if negate_ramp else h6
        return scale * (f_p + 4.0 * f_m + f_q)

    fused = _hat_fused_backend(t, window, G)
    if fused is not None:
        left, right, d_left, d_right = fused.interval_integrals(
            t, C, start, window, _hat_interior_bounds(L64), Ld, Td, sloped,
            p0_hi, p0_lo, p1_hi, p1_lo, R0, R1, need_grad=need_grad,
        )
        if drop_off_grid:
            return fused.dropped_knot_weights(
                left, right, d_left, d_right, start, G, total0,
                need_grad=need_grad,
            )
        idxK = _hat_window_index(t, G, start, window)
    else:
        idxK = _hat_window_index(t, G, start, window)
        Cg = C[None, :].expand(t.shape[0], G) if start is None else C[idxK]
        lam = t[:, None] * Cg
        ac = lam.clamp(R0, R1)
        # Searching the interior boundaries returns the segment index directly.
        # ac is clamped into [R0, R1] = [L64[0], L64[-1]], so the count of full
        # boundaries at or below ac is one more than the count of interior ones
        # whenever ac reaches L64[0], and both forms floor at zero when float32
        # clamping leaves ac a hair below it. That makes this exactly the old
        # searchsorted(L64, ...) - 1 clamped to [0, S-1], one pass cheaper.
        seg = torch.searchsorted(
            _hat_interior_bounds(L64), ac.to(L64.dtype).contiguous(), right=True
        ).clamp_max_(S - 1)
        # The interval ends are read five to seven times each below. A strided
        # slice of the knot grid blocks vectorised loads in every one of those
        # kernels, and one compacting copy is cheaper than the penalty; the
        # copies are exact.
        a, b = ac[:, :-1].contiguous(), ac[:, 1:].contiguous()
        ka, kb = seg[:, :-1], seg[:, 1:]
        # Every clamped knot is the left end of one interval and the right end
        # of the next, with the same segment either way, so gather the segment
        # rows and evaluate the photon response once over the whole knot grid
        # and take the two interval views of each. Element for element these
        # are the same expressions the per-interval form evaluated twice.
        knot_segment = gather_segment(seg)
        knot_response = response_at(ac, knot_segment)
        seg_a = tuple(row[:, :-1].contiguous() for row in knot_segment)
        seg_b = tuple(row[:, 1:].contiguous() for row in knot_segment)
        Ra = knot_response[:, :-1].contiguous()
        Rb = knot_response[:, 1:].contiguous()
        lam_lo, lam_hi = lam[:, :-1].contiguous(), lam[:, 1:].contiguous()
        active = b > a
        dC = Cg[:, 1:] - Cg[:, :-1]
        D = t[:, None] * dC

        ka_next = ka + 1
        aU = torch.minimum(Ld[ka_next], b)
        bL = torch.maximum(seg_b[0], aU)
        panel_a = simpson_panel(a, aU, seg_a, r_p=Ra)
        panel_b = simpson_panel(bL, b, seg_b, r_q=Rb)
        I_r = (
            simpson(panel_a, a, aU, lam_lo)
            + simpson(panel_b, bL, b, lam_lo)
        )
        I_l = (
            simpson(panel_a, a, aU, lam_hi, True)
            + simpson(panel_b, bL, b, lam_hi, True)
        )
        del panel_a, panel_b

        khi = torch.maximum(kb, ka_next)
        klo = ka_next
        prefix_table = _hat_stacked_table(p0_hi, p0_lo, p1_hi, p1_lo)
        prefix_hi = prefix_table[:, khi]
        prefix_lo = prefix_table[:, klo]
        p0d = (prefix_hi[0] - prefix_lo[0]) + (prefix_hi[1] - prefix_lo[1])
        p1d = (prefix_hi[2] - prefix_lo[2]) + (prefix_hi[3] - prefix_lo[3])
        del prefix_hi, prefix_lo
        I_r = I_r + (p1d - lam_lo * p0d)
        I_l = I_l + (lam_hi * p0d - p1d)

        right = torch.where(active, I_r / D, zero)
        left = torch.where(active, I_l / D, zero)

        d_left = d_right = None
        if need_grad:
            tt = t[:, None]
            mass = right + left
            # Same interval-view trick as the response: the interior indicator
            # is a property of each knot, and each knot serves as both ends.
            alpha = ((lam > R0) & (lam < R1)).to(dtype)
            alpha_a, alpha_b = alpha[:, :-1], alpha[:, 1:]
            Cp, Cn = Cg[:, :-1].contiguous(), Cg[:, 1:].contiguous()
            d_right = alpha_b * Cn * Rb - (Cp / D) * mass - right / tt
            d_left = -alpha_a * Cp * Ra + (Cn / D) * mass - left / tt
            d_right = torch.where(active, d_right, zero)
            d_left = torch.where(active, d_left, zero)

    if not drop_off_grid:
        lam_0 = t * C[0]
        lam_G = t * C[-1]
        a0 = lam_0.clamp(R0, R1)
        bG = lam_G.clamp(R0, R1)
        edge_pos = torch.stack([a0, bG], dim=1)
        edge_seg = torch.searchsorted(
            _hat_interior_bounds(L64), edge_pos.to(L64.dtype).contiguous(),
            right=True,
        ).clamp_max_(S - 1)
        k0, kG = edge_seg[:, 0], edge_seg[:, 1]
        kG_next = kG + 1
        sg0 = gather_segment(k0)
        sgG = gather_segment(kG)
        last_edge = Ld[kG_next]
        panel_first = simpson_panel(sg0[0], a0, sg0)
        panel_last = simpson_panel(bG, last_edge, sgG)
        e_first = (p0_hi[k0] + p0_lo[k0]) + simpson(
            panel_first, sg0[0], a0, None
        )
        e_last = (
            (p0_hi[S] - p0_hi[kG_next]) + (p0_lo[S] - p0_lo[kG_next])
        ) + simpson(panel_last, bG, last_edge, None)

    w = t.new_zeros(t.shape[0], G)
    w.scatter_add_(1, idxK, _hat_edge_sum(left, right))
    if not drop_off_grid:
        w[:, 0] += e_first
        w[:, -1] += e_last
    w = w / total0
    positive = w > 0
    w = torch.where(positive, w, zero)
    if drop_off_grid:
        W = w
    else:
        total = w.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        W = w / total

    if not need_grad:
        return W, None

    if not drop_off_grid:
        alpha_0 = ((lam_0 > R0) & (lam_0 < R1)).to(dtype)
        alpha_G = ((lam_G > R0) & (lam_G < R1)).to(dtype)
    dw = t.new_zeros(t.shape[0], G)
    dw.scatter_add_(1, idxK, _hat_edge_sum(d_left, d_right))
    if not drop_off_grid:
        dw[:, 0] += alpha_0 * C[0] * panel_first[4]
        dw[:, -1] -= alpha_G * C[-1] * panel_last[2]
    dw = torch.where(positive, dw / total0, zero)
    if drop_off_grid:
        # With no renormalization the positivity-clamped weights are the
        # output itself, so their derivative is already the output's.
        return W, dw
    dW = (dw - W * dw.sum(dim=1, keepdim=True)) / total
    return W, dW


class _HatBandWeights(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, t, C, L64, Ld, Td, sloped,
        p0_hi, p0_lo, p1_hi, p1_lo, total0, R0, R1,
        window=None, drop_off_grid=False,
    ):
        need_grad = bool(ctx.needs_input_grad[0])
        w, dw = _hat_weights_and_grad(
            t, C, L64, Ld, Td, sloped, p0_hi, p0_lo, p1_hi, p1_lo,
            total0, R0, R1, need_grad=need_grad, window=window,
            drop_off_grid=drop_off_grid,
        )
        if need_grad:
            ctx.save_for_backward(dw)
        return w

    @staticmethod
    def backward(ctx, grad_w):
        (dw,) = ctx.saved_tensors
        return (
            (grad_w * dw).sum(dim=1),
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None,
        )


class _DenseBandContraction(torch.autograd.Function):
    """Contract per-band weight blocks against the per-splat spectrum.

    The dense [rows, bands, knots] block has to exist in the forward: the band
    average is one batched contraction over it, and no per-band formulation
    reproduces that contraction's reduction bit for bit (measured: a per-band
    matmul over only that band's rows differs, and so does a broadcast
    multiply-reduce). Its *gradient* need not exist. Left to autograd, filling
    the block band by band makes every band after the first carry a gradient
    for the whole partially filled block, and the contraction then builds the
    full dense gradient as an outer product only for the scatter to gather
    each band's rows back out of it. At production shape that is gigabytes
    written and re-read to recover under a third of it.

    Every element of the block gradient is a plain product of two saved
    values, with no reduction and hence no reassociation, so each band's block
    gradient is formed directly. The gradient in the spectrum is the same
    batched contraction against the same block that autograd would run.
    """

    @staticmethod
    def forward(ctx, values, n_bands, placements, *band_weights):
        with torch.no_grad():
            weights = values.new_zeros(values.shape[0], n_bands, values.shape[1])
            for (band_index, rows), block in zip(placements, band_weights):
                weights[:, band_index].index_copy_(0, rows, block)
            averages = torch.einsum("jg,jbg->jb", values, weights)
        ctx.save_for_backward(values, weights)
        ctx.placements = placements
        return averages

    @staticmethod
    def backward(ctx, grad_averages):
        values, weights = ctx.saved_tensors
        grad_averages = grad_averages.contiguous()
        grad_values = None
        if ctx.needs_input_grad[0]:
            grad_values = torch.bmm(
                grad_averages.unsqueeze(1), weights
            ).squeeze(1)
        return (grad_values, None, None) + tuple(
            values.index_select(0, rows)
            * grad_averages.index_select(0, rows)[:, band_index:band_index + 1]
            for band_index, rows in ctx.placements
        )


@lru_cache(maxsize=1)
def _band_contraction_kernels():
    """The sparse Triton band contraction, imported only when selected."""
    from . import band_contraction_kernels

    return band_contraction_kernels


class _SparseBandContraction(torch.autograd.Function):
    """Contract retained row-band blocks without materialising dense zeros.

    Forward uses compensated fp32 summation over knots sequentially from 0
    through G - 1. The values gradient uses the same scheme over present bands
    sequentially from band 0 upward. Every arithmetic operation is an explicit
    round-to-nearest intrinsic. This fixed order is reproducible, but differs
    from the cuBLAS dense path.
    """

    @staticmethod
    def forward(ctx, values, n_bands, placements, *band_weights):
        kernels = _band_contraction_kernels()
        with torch.no_grad():
            averages, positions = kernels.forward(
                values, n_bands, placements, band_weights
            )
        ctx.save_for_backward(values, positions, *band_weights)
        ctx.n_bands = n_bands
        ctx.placements = placements
        return averages

    @staticmethod
    def backward(ctx, grad_averages):
        values, positions, *band_weights = ctx.saved_tensors
        grad_averages = grad_averages.contiguous()
        grad_values = None
        if ctx.needs_input_grad[0]:
            grad_values = _band_contraction_kernels().backward_values(
                grad_averages,
                positions,
                ctx.n_bands,
                ctx.placements,
                band_weights,
                values.shape[1],
            )
        return (grad_values, None, None) + tuple(
            values.index_select(0, rows)
            * grad_averages.index_select(0, rows)[:, band_index:band_index + 1]
            for band_index, rows in ctx.placements
        )


class _ExactBandContraction(torch.autograd.Function):
    """Production-shape sparse contraction using the verified cuBLAS order."""

    @staticmethod
    def forward(ctx, values, n_bands, placements, *band_weights):
        from . import exact_band_contraction_kernels as kernels

        averages, positions = kernels.forward(values, n_bands, placements, band_weights)
        ctx.save_for_backward(values, positions, *band_weights)
        ctx.n_bands = n_bands
        ctx.placements = placements
        return averages

    @staticmethod
    def backward(ctx, grad_averages):
        from . import exact_band_contraction_kernels as kernels

        values, positions, *band_weights = ctx.saved_tensors
        grad_averages = grad_averages.contiguous()
        grad_values = None
        if ctx.needs_input_grad[0]:
            grad_values = kernels.backward_values(
                grad_averages, positions, ctx.n_bands, ctx.placements, band_weights, values.shape[1]
            )
        return (grad_values, None, None) + tuple(
            values.index_select(0, rows)
            * grad_averages.index_select(0, rows)[:, band_index:band_index + 1]
            for band_index, rows in ctx.placements
        )


def splat_band_amplitudes(
    field,
    bandpasses,
    *,
    loglam_min: float,
    loglam_max: float,
    splat_chunk: int = AMP_CHUNK,
    indices: torch.Tensor | None = None,
    indices_sorted_unique: bool = False,
    include_flux: bool = True,
    band_row_indices: Sequence[torch.Tensor] | None = None,
    gathered_parameters: tuple[torch.Tensor, ...] | None = None,
) -> dict:
    """Exactly integrate selected splat SEDs through one or more bandpasses.

    Pass indices_sorted_unique=True only when `indices` is ascending and
    duplicate-free (e.g. built by torch.unique); it lets the sparse-gather
    backward skip a per-parameter sort.

    include_flux=False returns band amplitudes PER UNIT FLUX (the SED shape
    integral alone). Gauge-sensitive consumers must use this instead of
    dividing the full amplitude by softplus(raw_flux): the division needs a
    clamp that breaks gauge invariance for near-zero raw flux.

    band_row_indices optionally gives ascending-unique positions into the
    selected union rows for each bandpass. Rows omitted for a band return
    exact zero amplitudes in that band.
    """
    if band_row_indices is not None and len(band_row_indices) != len(bandpasses):
        raise ValueError(
            "band_row_indices must contain one tensor per bandpass."
        )
    if not bandpasses:
        return {}
    if not math.isfinite(loglam_min) or not math.isfinite(loglam_max):
        raise ValueError("Wavelength bounds must be finite.")
    if loglam_max <= loglam_min:
        raise ValueError("loglam_max must exceed loglam_min.")
    if splat_chunk <= 0:
        raise ValueError("splat_chunk must be positive.")

    device = field.mu.device
    dtype = field.mu.dtype
    prepared_bandpasses = []
    for key, (lam, throughput) in bandpasses.items():
        lam_tensor = torch.as_tensor(lam, dtype=dtype, device=device)
        throughput_tensor = torch.as_tensor(
            throughput, dtype=dtype, device=device
        )
        if lam_tensor.ndim != 1 or throughput_tensor.shape != lam_tensor.shape:
            raise ValueError(
                f"Bandpass {key!r} wavelengths and throughput must be matching vectors."
            )
        sorted_lam, sorted_throughput = _sorted_bandpass(
            lam_tensor, throughput_tensor
        )
        prepared_bandpasses.append((key, sorted_lam, sorted_throughput))
    if band_row_indices is not None:
        for rows in band_row_indices:
            if not torch.is_tensor(rows) or rows.dtype != torch.long:
                raise ValueError(
                    "band_row_indices entries must be long-dtype tensors."
                )

    scale = 0.5 * float(loglam_max - loglam_min)
    log_mid = 0.5 * float(loglam_max + loglam_min)
    grid = field.sed.loglam_grid
    templates = field.sed.templates()
    output = {key: [] for key, _, _ in prepared_bandpasses}

    C = _band_integration_grid(grid, log_mid, scale)
    band_tables = [
        _hat_band_tables(
            response_lam, throughput, dtype=dtype, device=device,
        )
        for _, response_lam, throughput in prepared_bandpasses
    ]

    if gathered_parameters is not None:
        if indices is None:
            raise ValueError("gathered_parameters requires indices.")
        if len(gathered_parameters) != 6:
            raise ValueError("gathered_parameters must contain six tensors.")
        selected = indices.to(device=device, dtype=torch.long)
        n_splats = int(selected.numel())
        _, _, _, all_raw_flux, all_coeffs, all_shifts = gathered_parameters
    elif indices is None:
        selected = None
        n_splats = int(field.mu.shape[0])
        all_coeffs = field.sed_coeffs
        all_raw_flux = field.raw_flux
        all_shifts = field.sed_shift
    else:
        # Gather each parameter once for the whole active set instead of once
        # per chunk: under sparse gradients that is one _SparseRowGather (one
        # coalesce in backward) per parameter per step instead of one per
        # chunk, and under dense gradients one index_select backward instead
        # of a full-size zero-fill per chunk.
        selected = indices.to(device=device, dtype=torch.long)
        n_splats = int(selected.numel())
        all_coeffs = field.sed_coefficients(
            selected, sorted_unique=indices_sorted_unique
        )
        all_raw_flux = field.raw_flux_values(
            selected, sorted_unique=indices_sorted_unique
        )
        all_shifts = field.spectral_shifts(
            selected, sorted_unique=indices_sorted_unique
        )

    band_widths = [
        _hat_window_width(C, tables[-2], tables[-1]) for tables in band_tables
    ]
    drop_off_grid = field.sed.off_grid == "zero"

    def integrate_chunk(
        coeffs: torch.Tensor,
        raw_flux: torch.Tensor,
        shift: torch.Tensor,
        template_values: torch.Tensor,
        *,
        local_band_rows: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        values = F.softplus(coeffs) @ template_values
        t = torch.exp(scale * shift)
        if local_band_rows is None and not torch.is_grad_enabled() and compact_all_bands_enabled():
            # The pruning path needs all rows in every band. Reuse the existing
            # verified compact contraction instead of materializing N*5*96
            # mostly-zero weights. No thresholds or retained rows change.
            all_rows=torch.arange(values.shape[0],device=values.device,dtype=torch.long)
            local_band_rows=[all_rows]*len(prepared_bandpasses)
        if (
            compact_band_weights_enabled()
            and exact_band_contraction_enabled()
            and local_band_rows is not None
            and drop_off_grid
            and values.is_cuda
            and values.dtype == torch.float32
            and exact_band_reduction_rows_supported(values.shape[0])
            and values.shape[1] == 96
            and len(prepared_bandpasses) == 5
            and validated_cublas_runtime()
            and torch.cuda.get_device_capability(values.device) == (10, 0)
        ):
            from .compact_band_kernels import CompactBandContraction, CompactHatWeights

            compact_placements, compact_blocks, compact_starts = [], [], []
            for band_index, (tables, width, local_rows) in enumerate(
                zip(band_tables, band_widths, local_band_rows)
            ):
                if local_rows.numel():
                    compact_weight, compact_start = CompactHatWeights.apply(t[local_rows], C, tables, width)
                    compact_placements.append((band_index, local_rows))
                    compact_blocks.append(compact_weight)
                    compact_starts.append(compact_start)
            if compact_blocks:
                averages = CompactBandContraction.apply(
                    values, tuple(compact_placements), tuple(compact_starts), *compact_blocks
                )
                if not include_flux:
                    return tuple(averages[:, band_index] for band_index in range(len(prepared_bandpasses)))
                flux = F.softplus(raw_flux)
                return tuple(flux * averages[:, band_index] for band_index in range(len(prepared_bandpasses)))
        placements = []
        band_blocks = []
        if local_band_rows is None:
            weights = torch.stack(
                [
                    _HatBandWeights.apply(t, C, *tables, width, drop_off_grid)
                    for tables, width in zip(band_tables, band_widths)
                ],
                dim=1,
            )
        else:
            for band_index, (tables, width, local_rows) in enumerate(
                zip(band_tables, band_widths, local_band_rows)
            ):
                if local_rows.numel():
                    placements.append((band_index, local_rows))
                    band_blocks.append(
                        _HatBandWeights.apply(
                            t[local_rows], C, *tables, width, drop_off_grid
                        )
                    )
            if not band_blocks:
                weights = values.new_zeros(
                    values.shape[0], len(prepared_bandpasses), values.shape[1]
                )
        if band_blocks:
            contraction = _DenseBandContraction
            if sparse_band_contraction_enabled():
                kernels = _band_contraction_kernels()
                if kernels.applies(
                    values, len(prepared_bandpasses), placements, band_blocks
                ):
                    contraction = _SparseBandContraction
            elif exact_band_contraction_enabled() and values.is_cuda:
                from . import exact_band_contraction_kernels as exact_kernels

                if exact_kernels.applies(values, len(prepared_bandpasses), placements, band_blocks):
                    contraction = _ExactBandContraction
            averages = contraction.apply(
                values, len(prepared_bandpasses), tuple(placements), *band_blocks,
            )
        else:
            averages = torch.einsum("jg,jbg->jb", values, weights)
        if not include_flux:
            return tuple(
                averages[:, band_index]
                for band_index in range(len(prepared_bandpasses))
            )
        flux = F.softplus(raw_flux)
        return tuple(
            flux * averages[:, band_index]
            for band_index in range(len(prepared_bandpasses))
        )

    splat_chunk = env_chunk("SPLAT_AMP_CHUNK", int(splat_chunk))
    use_checkpoint = (
        torch.is_grad_enabled() and not activation_checkpointing_disabled()
    )
    chunk_starts = list(range(0, n_splats, int(splat_chunk)))
    row_bounds = None
    if band_row_indices is not None and chunk_starts:
        # One host round trip for every chunk boundary of every band. Slicing
        # by a device searchsorted result costs a device sync per boundary,
        # and those add up to a stall per band per chunk on the hot path; the
        # boundaries themselves are the same integers either way.
        with torch.no_grad():
            edges = torch.tensor(
                chunk_starts + [n_splats], dtype=torch.long, device=device
            )
            row_bounds = torch.stack(
                [torch.searchsorted(rows, edges) for rows in band_row_indices]
            ).tolist()
    for chunk_index, start in enumerate(chunk_starts):
        stop = min(start + int(splat_chunk), n_splats)
        args = (
            all_coeffs[start:stop],
            all_raw_flux[start:stop],
            all_shifts[start:stop],
            templates,
        )
        chunk_integrator = integrate_chunk
        if band_row_indices is not None:
            with torch.no_grad():
                local_band_rows = [
                    rows[bounds[chunk_index]:bounds[chunk_index + 1]] - start
                    for rows, bounds in zip(band_row_indices, row_bounds)
                ]
            chunk_integrator = partial(
                integrate_chunk, local_band_rows=tuple(local_band_rows)
            )
        if use_checkpoint:
            amplitudes = checkpoint(chunk_integrator, *args, use_reentrant=False)
        else:
            amplitudes = chunk_integrator(*args)
        for (key, _, _), amplitude in zip(prepared_bandpasses, amplitudes):
            output[key].append(amplitude)
    return {
        key: (
            torch.cat(chunks)
            if chunks
            else torch.empty((0,), dtype=dtype, device=device)
        )
        for key, chunks in output.items()
    }


def _disk_quadrature(
    exposure: ExposureState,
    aperture: SkyDiskApertureSpec,
    *,
    radial_nodes: int,
    angular_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radial_x, radial_w = np.polynomial.legendre.leggauss(radial_nodes)
    radial_u = 0.5 * (radial_x + 1.0)
    radial_w = 0.5 * radial_w
    theta = (
        np.arange(angular_nodes, dtype=np.float64) + 0.5
    ) * (2.0 * np.pi / float(angular_nodes))
    radius = float(aperture.radius_arcsec) * np.sqrt(radial_u)
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    dx = (rr * np.cos(tt)).reshape(-1)
    dy = (rr * np.sin(tt)).reshape(-1)
    weights = np.repeat(radial_w / float(angular_nodes), angular_nodes)

    center_ra, center_dec = aperture.center_radec_deg
    cos_dec = math.cos(math.radians(float(center_dec)))
    if abs(cos_dec) < 1.0e-12:
        raise ValueError("SkyDiskApertureSpec center is too close to a pole.")
    ra = float(center_ra) + dx / (3600.0 * cos_dec)
    dec = float(center_dec) + dy / 3600.0
    cos_dec_t = math.cos(math.radians(float(center_dec)))
    probe_ra = np.array(
        [
            float(center_ra),
            float(center_ra) + 1.0 / (3600.0 * cos_dec_t),
            float(center_ra),
        ],
        dtype=np.float64,
    )
    probe_dec = np.array(
        [float(center_dec), float(center_dec), float(center_dec) + 1.0 / 3600.0],
        dtype=np.float64,
    )

    omega = radec_deg_to_omega_xyz(ra, dec).reshape(-1, 3)
    probe = radec_deg_to_omega_xyz(probe_ra, probe_dec).reshape(-1, 3)
    x, y = exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(omega, dtype=dtype, device=device)
    )
    probe_x, probe_y = exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(probe, dtype=dtype, device=device)
    )
    jacobian = torch.stack(
        (
            torch.stack((probe_x[1] - probe_x[0], probe_x[2] - probe_x[0])),
            torch.stack((probe_y[1] - probe_y[0], probe_y[2] - probe_y[0])),
        )
    )
    return (
        torch.stack((x, y), dim=1),
        torch.as_tensor(weights, dtype=dtype, device=device),
        jacobian,
    )


@dataclass(frozen=True, slots=True)
class SplatSpectrumCache:
    exposure_identity: int
    apply_spatial_psf: bool
    radial_nodes: int
    angular_nodes: int
    dtype: torch.dtype
    device: torch.device
    lam_edges: torch.Tensor
    points: torch.Tensor
    weights: torch.Tensor
    psf_cov: torch.Tensor
    psf_eigen_max: torch.Tensor
    aperture_area_arcsec2: float
    scalar_lsf_sigma: float | None
    lsf_sigma_by_bin: torch.Tensor | None
    # Hoisted invariants: quadrature-point bounds for the candidate reach
    # test, the LSF sign pattern ("zero" | "positive" | None for mixed) so the
    # hot path can skip data-dependent host syncs, and the per-bin LSF sigma
    # materialized for the scalar case as well, so the flat path can read every
    # exposure's LSF with one concatenation. None defaults keep older call sites
    # valid; the hot path falls back to computing them.
    point_bounds: tuple[float, float, float, float] | None = None
    lsf_mode: str | None = None
    lsf_sigma_full: torch.Tensor | None = None
    spectral_source_scale: torch.Tensor | None = None
    source_scale_snapshot: np.ndarray | None = None
    source_scale_exposure: ExposureState | None = None
    source_geometry_snapshot: tuple | None = None


@dataclass(frozen=True, slots=True)
class SplatSpectrumBatchContext:
    """Differentiable work shared only within one spectral training batch."""

    templates: torch.Tensor
    splat_eigen_max: torch.Tensor


def build_spectrum_batch_context(field) -> SplatSpectrumBatchContext:
    return SplatSpectrumBatchContext(
        templates=field.sed.templates(),
        splat_eigen_max=torch.exp(
            2.0 * field.log_scales.detach().amax(dim=1)
        ),
    )


def build_spectrum_cache(
    exposure: ExposureState,
    aperture: SkyDiskApertureSpec,
    *,
    apply_spatial_psf: bool,
    radial_nodes: int,
    angular_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SplatSpectrumCache:
    validate_spectral_source_psf(exposure, apply_spatial_psf=apply_spatial_psf)
    if exposure.lam_edges is None:
        raise ValueError("Deterministic splat spectra require wavelength-bin edges.")
    points, weights, sky_to_norm = _disk_quadrature(
        exposure,
        aperture,
        radial_nodes=int(radial_nodes),
        angular_nodes=int(angular_nodes),
        device=device,
        dtype=dtype,
    )
    psf_cov = torch.zeros((2, 2), dtype=dtype, device=device)
    if apply_spatial_psf and exposure.psf_fwhm_arcsec is not None:
        psf_sigma_arcsec = float(exposure.psf_fwhm_arcsec) / 2.355
        if psf_sigma_arcsec > 0.0:
            psf_cov = sky_to_norm @ sky_to_norm.T * (psf_sigma_arcsec**2)

    lsf_fwhm = exposure.lsf_fwhm_angstrom
    scalar_lsf_sigma: float | None
    lsf_sigma_by_bin: torch.Tensor | None
    if lsf_fwhm is None:
        scalar_lsf_sigma = 0.0
        lsf_sigma_by_bin = None
        lsf_mode = "zero"
    elif np.isscalar(lsf_fwhm) or (
        isinstance(lsf_fwhm, torch.Tensor) and lsf_fwhm.ndim == 0
    ):
        scalar_lsf_sigma = float(lsf_fwhm) / 2.355
        lsf_sigma_by_bin = None
        if scalar_lsf_sigma < 0.0:
            raise ValueError("LSF sigma must be non-negative.")
        lsf_mode = "zero" if scalar_lsf_sigma == 0.0 else "positive"
    else:
        scalar_lsf_sigma = None
        lsf_sigma_by_bin = torch.as_tensor(lsf_fwhm, dtype=dtype, device=device) / 2.355
        if bool(torch.any(lsf_sigma_by_bin < 0)):
            raise ValueError("LSF sigma must be non-negative.")
        if bool(torch.all(lsf_sigma_by_bin > 0)):
            lsf_mode = "positive"
        elif bool(torch.all(lsf_sigma_by_bin == 0)):
            lsf_mode = "zero"
        else:
            lsf_mode = None

    lam_edges = torch.as_tensor(exposure.lam_edges, dtype=dtype, device=device)
    if lam_edges.ndim != 2 or lam_edges.shape[-1] != 2:
        raise ValueError("lam_edges must have shape (n_bins, 2).")
    if bool(torch.any(lam_edges[:, 1] <= lam_edges[:, 0])):
        raise ValueError("Spectral bin upper edges must exceed lower edges.")
    if lsf_sigma_by_bin is not None and lsf_sigma_by_bin.shape != (
        lam_edges.shape[0],
    ):
        raise ValueError(
            "Vector LSF must provide exactly one sigma per spectral bin; got "
            f"{tuple(lsf_sigma_by_bin.shape)} for {int(lam_edges.shape[0])} bins."
        )

    source_scale = validated_spectral_source_scale(exposure)
    if source_scale is not None:
        cast_scale = source_scale.astype(np.float64 if dtype == torch.float64 else np.float32)
        if not np.isfinite(cast_scale).all() or np.any(cast_scale <= 0):
            raise ValueError("spectral_source_scale is not representable in the output dtype.")
    if source_scale is not None and source_scale.shape != (int(lam_edges.shape[0]),):
        raise ValueError("spectral_source_scale must match wavelength-bin edges.")
    return SplatSpectrumCache(
        exposure_identity=id(exposure),
        spectral_source_scale=(None if source_scale is None else torch.tensor(source_scale, dtype=dtype, device=device)),
        source_scale_snapshot=source_scale,
        source_scale_exposure=exposure,
        source_geometry_snapshot=spectral_source_geometry_key(exposure),
        apply_spatial_psf=bool(apply_spatial_psf),
        radial_nodes=int(radial_nodes),
        angular_nodes=int(angular_nodes),
        dtype=dtype,
        device=device,
        lam_edges=lam_edges,
        points=points,
        weights=weights,
        psf_cov=psf_cov,
        psf_eigen_max=_largest_eigenvalue_symmetric_2x2(psf_cov),
        aperture_area_arcsec2=aperture_area_arcsec2(
            aperture, pixel_scale_arcsec=exposure.pixel_scale_arcsec
        ),
        scalar_lsf_sigma=scalar_lsf_sigma,
        lsf_sigma_by_bin=lsf_sigma_by_bin,
        lsf_sigma_full=(
            lsf_sigma_by_bin
            if lsf_sigma_by_bin is not None
            else torch.full(
                (int(lam_edges.shape[0]),),
                scalar_lsf_sigma,
                dtype=dtype,
                device=device,
            )
        ),
        # One host transfer for all four bounds rather than four separate
        # stalls; a run builds one cache per fiber exposure, 889 of them. The
        # reduction kernels are unchanged, so each bound is bitwise what the
        # separate reads produced.
        point_bounds=tuple(
            torch.stack(
                (
                    points[:, 0].min(),
                    points[:, 0].max(),
                    points[:, 1].min(),
                    points[:, 1].max(),
                )
            ).tolist()
        ),
        lsf_mode=lsf_mode,
    )


def stacked_fiber_bounds(
    caches: Sequence[SplatSpectrumCache],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """[F, 4] quadrature-point bounds (x_min, x_max, y_min, y_max) per fiber."""
    bounds: list[tuple[float, float, float, float] | torch.Tensor] = []
    for cache in caches:
        if cache.point_bounds is not None:
            bounds.append(cache.point_bounds)
        else:
            bounds.append(
                torch.stack(
                    (
                        cache.points[:, 0].min(),
                        cache.points[:, 0].max(),
                        cache.points[:, 1].min(),
                        cache.points[:, 1].max(),
                    )
                )
            )
    if all(not isinstance(bound, torch.Tensor) for bound in bounds):
        return tensor_on_device(bounds, dtype=dtype, device=device)
    return torch.stack(
        [
            bound
            if isinstance(bound, torch.Tensor)
            else tensor_on_device(bound, dtype=dtype, device=device)
            for bound in bounds
        ]
    )


def _fiber_candidate_chunk(
    mu: torch.Tensor,
    splat_eigen_max: torch.Tensor,
    psf_eigen_max: torch.Tensor,
    bounds: torch.Tensor,
    n_sigma: float,
) -> list[torch.Tensor]:
    n_fibers = int(bounds.shape[0])
    x_min, x_max, y_min, y_max = bounds.unbind(dim=1)
    # Superset proposal: the largest reach any fiber in the chunk can ask of a
    # splat, plus the largest box half-extent, so a centre-to-centre box test
    # covers every box-to-box pair.
    half_extent = 0.5 * torch.maximum(x_max - x_min, y_max - y_min).amax()
    radius = (
        float(n_sigma) * torch.sqrt(splat_eigen_max + psf_eigen_max.amax())
        + half_extent
    )
    # The guard has to absorb the fp32 rounding that separates the two forms of
    # the test and the tile arithmetic underneath, which is a few ulps of the
    # coordinates involved. It only has to hold where the forms nearly agree,
    # and there the fiber box sits within one reach of the splat, so the splat's
    # own position bounds the fiber side too. Keeping it per splat also keeps a
    # non-finite position local: that splat is dropped, exactly as the dense
    # form's comparisons drop it, instead of the reduction turning every radius
    # into a NaN. A non-finite position carries no magnitude to guard against.
    guard = 64.0 * torch.finfo(mu.dtype).eps
    radius = radius + guard * (
        radius + torch.nan_to_num(mu.abs().amax(dim=1), nan=0.0, posinf=0.0)
    )
    pair_fiber, pair_splat = build_pair_index(
        mu, radius, 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    )

    # The exact test, in the dense implementation's own fp32 expressions: the
    # reach is recomputed per pair from the same two operands.
    reach = float(n_sigma) * torch.sqrt(
        splat_eigen_max[pair_splat] + psf_eigen_max[pair_fiber]
    )
    mu_x = mu[pair_splat, 0]
    mu_y = mu[pair_splat, 1]
    keep = (
        (mu_x + reach >= x_min[pair_fiber])
        & (mu_x - reach <= x_max[pair_fiber])
        & (mu_y + reach >= y_min[pair_fiber])
        & (mu_y - reach <= y_max[pair_fiber])
    )
    kept = keep.nonzero(as_tuple=True)[0]
    counts = torch.bincount(pair_fiber[kept], minlength=n_fibers)
    return list(torch.split(pair_splat[kept], counts.tolist()))


def batched_fiber_candidates(
    field,
    caches: Sequence[SplatSpectrumCache],
    *,
    n_sigma: float,
    splat_eigen_max: torch.Tensor,
    fiber_chunk: int = 64,
) -> list[torch.Tensor]:
    """Ascending candidate splat ids per fiber exposure.

    The test is the fiber's quadrature bounding box against the splat's
    +-n_sigma reach box. Deciding it densely costs an [n_fibers, n_splats]
    boolean and float intermediates of the same shape -- 2.5 GB each at 62
    fibers and a 10M-splat census. Instead the tile index the image path
    already uses proposes a superset from the fiber centres, and the identical
    fp32 expressions then decide each proposed pair.

    build_pair_index emits its pairs ordered by (point, splat) and the nonzero
    compaction of the exact test keeps that order, so each fiber's list comes
    out ascending exactly as the dense row-major nonzero produced it.
    """
    with torch.no_grad():
        if int(splat_eigen_max.shape[0]) != int(field.mu.shape[0]):
            raise ValueError("Spectrum batch context has a stale splat count.")
        mu = field.mu.detach()
        candidates: list[torch.Tensor] = []
        for start in range(0, len(caches), int(fiber_chunk)):
            cache_chunk = caches[start : start + int(fiber_chunk)]
            candidates.extend(
                _fiber_candidate_chunk(
                    mu,
                    splat_eigen_max,
                    torch.stack([cache.psf_eigen_max for cache in cache_chunk]),
                    stacked_fiber_bounds(
                        cache_chunk, dtype=mu.dtype, device=mu.device
                    ),
                    n_sigma,
                )
            )
        return candidates


def convolved_gaussian_coefficients(
    log_scales: torch.Tensor,
    theta: torch.Tensor,
    psf_cov: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-splat coefficients of the PSF-convolved Gaussian in whitened form.

    For C = Sigma_j + Psi with entries [[a, b], [b, d]] the Mahalanobis form is
    evaluated through the Cholesky factor of C rather than through C^-1:

        m = dx^2 * inv_a + (dy - shear * dx)^2 * coef
        inv_a = 1 / a,  shear = b / a,  coef = a / det C

    Returns (inv_a, shear, coef, norm) with norm = 1 / (2 pi sqrt(det C)).

    Why not the precision matrix. An elongated splat (axes 1.34 by 194 arcsec,
    fiber S3154-54821-0498 of exposure 993) has precision entries of order
    1/lambda_min that must cancel to 1/lambda_max along its long axis. The
    previous contraction ran that cancellation inside a batched GEMM, and with
    TF32 enabled cuBLAS rounds the GEMM inputs to 10 mantissa bits, which
    leaves a rounded precision that is no longer positive definite: the
    captured form was -27.3 where the same-point reference was +21.9, and the
    fiber's predicted flux came out 3378 times the observation. Every operation
    here is elementwise, so TF32 does not affect this stage, and m is a
    sum of two squares, so exp(-m / 2) cannot exceed one.

    Why the determinant comes from the rows. det Sigma_j = exp(2 (s0 + s1))
    exactly, whereas a*d - b*b in fp32 loses the minor eigenvalue once
    lambda_max / lambda_min approaches 1 / eps: it is 0.06% off at axis ratio
    145, 4% off at 1000 and non-positive by 5000, and the old LU determinant
    shared that loss. det C = det Sigma + det Psi + tr(adj(Sigma) Psi) has three
    non-negative terms, so the splat's own conditioning no longer enters. What
    remains is the PSF's: det Psi and the trace term are sums whose relative
    error is eps times cond(Psi), because the fp32 entries of Psi carry its
    minor eigenvalue no better than Sigma's entries carried theirs. The
    WCS-derived PSF covariances here are near isotropic (cond about 1.07), where
    that error is a few 1e-7. It only becomes visible when the splat's minor
    variance falls below the PSF's minor variance, since det Sigma otherwise
    dominates det C: a PSF with cond 1e6 aligned with a 194 by 0.0001 arcsec
    splat is 2% off in fp32, and would need Psi supplied in a factored form
    instead. Psi is
    symmetric by construction; its off-diagonal is averaged so that a last-bit
    asymmetry from the WCS product cannot make the factorization inconsistent.
    With apply_spatial_psf off, Psi is zero and det C is exactly det Sigma.
    """
    a, b, d = SplatField.covariance_components_from_rows(log_scales, theta)
    psf_a = psf_cov[0, 0]
    psf_d = psf_cov[1, 1]
    psf_b = 0.5 * (psf_cov[0, 1] + psf_cov[1, 0])
    total_a = a + psf_a
    total_b = b + psf_b
    determinant = (
        torch.exp(2.0 * (log_scales[:, 0] + log_scales[:, 1]))
        + (psf_a * psf_d - psf_b * psf_b)
        + (a * psf_d + d * psf_a - 2.0 * b * psf_b)
    )
    inv_a = 1.0 / total_a
    shear = total_b * inv_a
    coef = total_a / determinant
    norm = 1.0 / (2.0 * math.pi * torch.sqrt(determinant))
    return inv_a, shear, coef, norm


def candidate_aperture_weights(
    field,
    candidate: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    psf_cov: torch.Tensor,
    area_arcsec2: float,
) -> torch.Tensor:
    """[J] aperture-integrated PSF-convolved density of each candidate splat.

    The quadrature sum keeps its [Q, J] layout and leading-axis reduction, so
    the per-exposure loop and the blocked batch stay cell for cell identical.
    """
    log_scales, theta = field.shape_rows(candidate, sorted_unique=True)
    inv_a, shear, coef, norm = convolved_gaussian_coefficients(
        log_scales, theta, psf_cov
    )
    mu = field.positions(candidate, sorted_unique=True)
    delta_x = points[:, 0, None] - mu[None, :, 0]
    delta_y = points[:, 1, None] - mu[None, :, 1]
    sheared = torch.addcmul(delta_y, delta_x, shear[None, :], value=-1.0)
    mahalanobis = torch.addcmul(
        delta_x * delta_x * inv_a[None, :], sheared * sheared, coef[None, :]
    )
    density = norm[None, :] * torch.exp(-0.5 * mahalanobis)
    return torch.sum(density * weights[:, None], dim=0) * area_arcsec2


def paired_precision_and_norm(
    cov: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Inverse entries and Gaussian normalization of each pair's 2x2 covariance.

    The adjugate rather than torch.linalg.inv_ex and torch.linalg.det. Both of
    those lower to a batched LU whose backward runs batched 2x2 GEMMs: 13.3 ms
    of forward and backward over 206k pairs against 5.3 ms here. An LU of a 2x2
    forms the same determinant with the same cancellation between a*d and b*c,
    so no conditioning is given up. Written for a general 2x2 because the PSF
    term is a matrix product and need not be exactly symmetric.
    """
    c00, c01, c10, c11 = cov.reshape(-1, 4).unbind(-1)
    determinant = c00 * c11 - c01 * c10
    reciprocal = 1.0 / determinant
    return (
        (
            c11 * reciprocal,
            -c01 * reciprocal,
            -c10 * reciprocal,
            c00 * reciprocal,
        ),
        1.0 / (2.0 * math.pi * torch.sqrt(determinant)),
    )


def paired_gaussian_quadrature(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    quadrature: torch.Tensor,
    pair_fiber: torch.Tensor,
    pair_mu: torch.Tensor,
    precision: tuple[torch.Tensor, ...],
    norm: torch.Tensor,
) -> torch.Tensor:
    if fused_quadrature_enabled() and pair_mu.is_cuda:
        from . import quadrature_kernels

        if quadrature_kernels.applies(points_x, points_y, quadrature, pair_fiber, pair_mu, precision, norm):
            return quadrature_kernels.GaussianQuadrature.apply(
                points_x, points_y, quadrature, pair_fiber, pair_mu, *precision, norm
            )
    return _paired_gaussian_quadrature_eager(
        points_x, points_y, quadrature, pair_fiber, pair_mu, precision, norm
    )


def _paired_gaussian_quadrature_eager(
    points_x: torch.Tensor,
    points_y: torch.Tensor,
    quadrature: torch.Tensor,
    pair_fiber: torch.Tensor,
    pair_mu: torch.Tensor,
    precision: tuple[torch.Tensor, ...],
    norm: torch.Tensor,
) -> torch.Tensor:
    """[P] quadrature sum of each pair's PSF-convolved Gaussian.

    The quadratic form is written out rather than run as a batch of 2x2 GEMMs:
    at one flat pair axis cuBLAS measures 3.6 ms on 206k of them against 1.3 ms
    for the chain below, which evaluates the same products in the same order
    with the multiply-adds unfused. Being elementwise it is also outside the
    reach of TF32, which is what broke the GEMM contraction the single and
    blocked paths used to run (see convolved_gaussian_coefficients). Its
    precision entries still come from the fp32 adjugate of the total
    covariance, so it keeps that route's conditioning limit for extreme
    needles; the fused Triton kernel mirrors this chain and must change with it.
    """
    p00, p01, p10, p11 = precision
    delta_x = points_x[pair_fiber] - pair_mu[:, 0, None]
    delta_y = points_y[pair_fiber] - pair_mu[:, 1, None]
    first = torch.addcmul(delta_x * p00[:, None], delta_y, p10[:, None])
    second = torch.addcmul(delta_x * p01[:, None], delta_y, p11[:, None])
    mahalanobis = torch.addcmul(first * delta_x, second, delta_y)
    density = norm[:, None] * torch.exp(-0.5 * mahalanobis)
    return torch.sum(density * quadrature[pair_fiber], dim=1)


def _candidate_spatial_weights(
    field,
    exposure: ExposureState,
    aperture: SkyDiskApertureSpec,
    *,
    apply_spatial_psf: bool,
    radial_nodes: int,
    angular_nodes: int,
    n_sigma: float,
    cache: SplatSpectrumCache | None = None,
    splat_eigen_max: torch.Tensor | None = None,
    candidate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = field.mu.device
    dtype = field.mu.dtype
    if cache is None:
        points, weights, sky_to_norm = _disk_quadrature(
            exposure,
            aperture,
            radial_nodes=radial_nodes,
            angular_nodes=angular_nodes,
            device=device,
            dtype=dtype,
        )
        psf_cov = torch.zeros((2, 2), dtype=dtype, device=device)
        if apply_spatial_psf and exposure.psf_fwhm_arcsec is not None:
            psf_sigma_arcsec = float(exposure.psf_fwhm_arcsec) / 2.355
            if psf_sigma_arcsec > 0.0:
                psf_cov = sky_to_norm @ sky_to_norm.T * (psf_sigma_arcsec**2)
        area_arcsec2 = aperture_area_arcsec2(
            aperture, pixel_scale_arcsec=exposure.pixel_scale_arcsec
        )
    else:
        if cache.exposure_identity != id(exposure):
            raise ValueError("Spectrum cache exposure mismatch for this call.")
        if cache.radial_nodes != int(radial_nodes):
            raise ValueError("Spectrum cache radial_nodes mismatch for this call.")
        if cache.angular_nodes != int(angular_nodes):
            raise ValueError("Spectrum cache angular_nodes mismatch for this call.")
        if cache.apply_spatial_psf != bool(apply_spatial_psf):
            raise ValueError("Spectrum cache apply_spatial_psf mismatch for this call.")
        if cache.dtype != dtype or cache.device != device:
            raise ValueError("Spectrum cache device/dtype mismatch for this call.")
        points = cache.points
        weights = cache.weights
        psf_cov = cache.psf_cov
        psf_eigen_max = cache.psf_eigen_max
        area_arcsec2 = cache.aperture_area_arcsec2

    with torch.no_grad():
        if splat_eigen_max is None:
            splat_eigen_max = torch.exp(2.0 * field.log_scales.detach().amax(dim=1))
        if int(splat_eigen_max.shape[0]) != int(field.mu.shape[0]):
            raise ValueError("Spectrum batch context has a stale splat count.")
        if candidate is None:
            if cache is None:
                psf_eigen_max = _largest_eigenvalue_symmetric_2x2(psf_cov)
            if cache is not None and cache.point_bounds is not None:
                x_min, x_max, y_min, y_max = cache.point_bounds
            else:
                x_min = points[:, 0].min()
                x_max = points[:, 0].max()
                y_min = points[:, 1].min()
                y_max = points[:, 1].max()
            eigen_max = splat_eigen_max + psf_eigen_max
            reach = float(n_sigma) * torch.sqrt(eigen_max)
            keep = (
                (field.mu[:, 0] + reach >= x_min)
                & (field.mu[:, 0] - reach <= x_max)
                & (field.mu[:, 1] + reach >= y_min)
                & (field.mu[:, 1] - reach <= y_max)
            )
            candidate = torch.nonzero(keep, as_tuple=True)[0]
    if candidate.numel() == 0:
        return candidate, torch.empty((0,), dtype=dtype, device=device)

    # Per-fiber nonzero and row-major batched-nonzero slices are both
    # ascending unique, so sparse-gather backwards can skip coalesce sorts.
    return candidate, candidate_aperture_weights(
        field, candidate, points, weights, psf_cov, area_arcsec2
    )


def _largest_eigenvalue_symmetric_2x2(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must have trailing shape (2, 2).")
    diagonal_mean = 0.5 * (matrix[..., 0, 0] + matrix[..., 1, 1])
    diagonal_half_difference = 0.5 * (
        matrix[..., 0, 0] - matrix[..., 1, 1]
    )
    off_diagonal = 0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0])
    return diagonal_mean + torch.sqrt(
        diagonal_half_difference.square() + off_diagonal.square()
    )


def splat_fiber_spectrum(
    field,
    exposure: ExposureState,
    bin_indices: torch.Tensor,
    *,
    loglam_min: float,
    loglam_max: float,
    apply_spatial_psf: bool,
    radial_nodes: int = 8,
    angular_nodes: int = 32,
    n_sigma: float = 6.0,
    splat_chunk: int = 2048,
    cache: SplatSpectrumCache | None = None,
    batch_context: SplatSpectrumBatchContext | None = None,
    candidate: torch.Tensor | None = None,
    validate_bins: bool = True,
) -> torch.Tensor:
    """Deterministic SplatField prediction for selected fiber-spectrum bins.

    validate_bins=False skips the bin-index bounds check (two host syncs);
    pass it only when the caller has already range-checked bin_indices.
    """
    if not isinstance(exposure.aperture, SkyDiskApertureSpec):
        raise TypeError("Deterministic splat spectra require SkyDiskApertureSpec.")
    validate_spectral_source_psf(exposure, apply_spatial_psf=apply_spatial_psf)
    if exposure.lam_edges is None:
        raise ValueError("Deterministic splat spectra require wavelength-bin edges.")
    local = bin_indices.to(dtype=torch.long, device=field.mu.device)
    if cache is None:
        cache = build_spectrum_cache(
            exposure,
            exposure.aperture,
            apply_spatial_psf=apply_spatial_psf,
            radial_nodes=radial_nodes,
            angular_nodes=angular_nodes,
            device=field.mu.device,
            dtype=field.mu.dtype,
        )
    elif cache.exposure_identity != id(exposure):
        raise ValueError("Spectrum cache exposure mismatch for this call.")
    _validate_source_scale_cache(cache)
    edges = cache.lam_edges
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("Spectrum cache lam_edges must have shape (n_bins, 2).")
    if validate_bins and (
        torch.any(local < 0) or torch.any(local >= edges.shape[0])
    ):
        raise ValueError("Spectral bin index is out of bounds.")
    lower = edges[local, 0]
    upper = edges[local, 1]

    if cache.scalar_lsf_sigma is not None:
        lsf_sigma = torch.full_like(lower, cache.scalar_lsf_sigma)
    elif cache.lsf_sigma_by_bin is not None:
        lsf_sigma = cache.lsf_sigma_by_bin[local]
    else:
        raise ValueError("Invalid spectrum cache: missing scalar and vector LSF.")

    candidate, spatial = _candidate_spatial_weights(
        field,
        exposure,
        exposure.aperture,
        apply_spatial_psf=apply_spatial_psf,
        radial_nodes=int(radial_nodes),
        angular_nodes=int(angular_nodes),
        n_sigma=float(n_sigma),
        cache=cache,
        splat_eigen_max=(
            None if batch_context is None else batch_context.splat_eigen_max
        ),
        candidate=candidate,
    )
    output = torch.zeros(local.shape, dtype=field.mu.dtype, device=field.mu.device)
    if candidate.numel() == 0:
        return _apply_cached_source_scale(output, cache, local)

    scale = 0.5 * float(loglam_max - loglam_min)
    log_mid = 0.5 * float(loglam_max + loglam_min)
    if scale <= 0.0:
        raise ValueError("loglam_max must exceed loglam_min.")
    templates = (
        field.sed.templates()
        if batch_context is None
        else batch_context.templates
    )
    grid = field.sed.loglam_grid
    # Candidate chunks are ascending unique (from nonzero), so the sparse
    # gathers skip their coalesce sorts. The cache validated edges and LSF
    # at build time, so the bin integrator runs sync-free.
    for start in range(0, int(candidate.numel()), int(splat_chunk)):
        idx = candidate[start : start + int(splat_chunk)]
        values = (
            F.softplus(field.sed_coefficients(idx, sorted_unique=True))
            @ templates
        )
        knot_lam = torch.exp(
            log_mid + scale * (
                grid[None, :]
                + field.spectral_shifts(idx, sorted_unique=True)[:, None]
            )
        )
        spectral = positive_hat_bin_average(
            knot_lam, values, lower, upper, lsf_sigma,
            validate=False, lsf_mode=cache.lsf_mode,
            off_grid=field.sed.off_grid,
        )
        amplitude = (
            field.fluxes(idx, sorted_unique=True)
            * spatial[start : start + idx.numel()]
        )
        output = output + torch.sum(spectral * amplitude[None, :], dim=1)
    return _apply_cached_source_scale(output, cache, local)


def _exclusive_cumsum(counts: Sequence[int]) -> np.ndarray:
    return np.concatenate(
        ([0], np.cumsum(np.asarray(counts, dtype=np.int64))[:-1])
    )


def _validate_source_scale_cache(cache: SplatSpectrumCache) -> None:
    exposure = cache.source_scale_exposure
    if exposure is None:
        return  # Legacy explicitly constructed caches carry aperture-only response.
    validate_spectral_source_psf(exposure, apply_spatial_psf=cache.apply_spatial_psf)
    if spectral_source_geometry_key(exposure) != cache.source_geometry_snapshot:
        raise ValueError("Spectrum cache response geometry is stale; rebuild the cache.")
    # Scalars are immutable; avoid allocating/copying a full native-bin vector
    # on each training forward. Array values retain full content checks.
    value = exposure.spectral_source_scale
    snapshot = cache.source_scale_snapshot
    if value is not None and np.isscalar(value):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar <= 0:
            raise ValueError("spectral_source_scale must be positive and finite.")
        expected = (exposure.spectral_response_metadata or {}).get("resolved_source_scale")
        if cache.source_geometry_snapshot is not None and scalar != float(expected):
            raise ValueError("spectral_source_scale disagrees with resolved response metadata.")
        if snapshot is None or not np.all(snapshot == scalar):
            raise ValueError("Spectrum cache spectral_source_scale is stale; rebuild the cache.")
        return
    current = validated_spectral_source_scale(exposure)
    snapshot = cache.source_scale_snapshot
    if (current is None) != (snapshot is None) or (
        current is not None and not np.array_equal(current, snapshot)
    ):
        raise ValueError("Spectrum cache spectral_source_scale is stale; rebuild the cache.")


def _apply_cached_source_scale(output, cache, local):
    if cache.spectral_source_scale is None:
        return output
    return output * cache.spectral_source_scale[local.to(device=output.device, dtype=torch.long)]


def _validate_fiber_cache_batch(
    caches: Sequence[SplatSpectrumCache],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    reference = caches[0]
    for cache in caches:
        _validate_source_scale_cache(cache)
        if cache.lsf_mode != reference.lsf_mode:
            raise ValueError("Batched fiber spectra require one shared LSF mode.")
        if cache.points.shape != reference.points.shape:
            raise ValueError("Batched fiber spectra require one shared quadrature.")
        if cache.dtype != dtype or cache.device != device:
            raise ValueError("Spectrum cache device/dtype mismatch for this call.")
        if cache.apply_spatial_psf != reference.apply_spatial_psf:
            raise ValueError("Batched fiber spectra require one shared PSF setting.")


def _cache_lsf_sigma_full(
    cache: SplatSpectrumCache, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if cache.lsf_sigma_full is not None:
        return cache.lsf_sigma_full
    if cache.lsf_sigma_by_bin is not None:
        return cache.lsf_sigma_by_bin
    if cache.scalar_lsf_sigma is None:
        raise ValueError("Invalid spectrum cache: missing scalar and vector LSF.")
    return torch.full(
        (int(cache.lam_edges.shape[0]),),
        cache.scalar_lsf_sigma,
        dtype=dtype,
        device=device,
    )


def batched_fiber_spectra(
    field,
    caches: Sequence[SplatSpectrumCache],
    candidates: Sequence[torch.Tensor],
    bin_indices: Sequence[torch.Tensor],
    *,
    loglam_min: float,
    loglam_max: float,
    batch_context: SplatSpectrumBatchContext,
    splat_chunk: int = 2048,
) -> list[torch.Tensor]:
    """splat_fiber_spectrum for a whole spectral batch in one set of kernels.

    A training batch of 64 fiber datums spans ~62 distinct exposures, so the
    per-exposure loop pays the full fixed cost of the fiber machinery to
    produce about one bin at a time and the step becomes kernel-launch bound.
    This evaluates every exposure's candidates together: the candidate index
    sets are concatenated, the quadrature/PSF state is gathered per candidate
    from its owning exposure, and the SED bin integration runs over maximal
    runs of exposures rather than one exposure at a time.

    The arithmetic is deliberately kept cell for cell identical to the loop.
    Every operation on the concatenated axis is elementwise or a gather; the
    two reductions the loop performs -- the quadrature sum over points and the
    per-chunk sum over splats -- keep their original axis, length and layout,
    and the SED template GEMM keeps its per-exposure shape so cuBLAS cannot
    pick a different algorithm for a different row count.

    Returns one prediction tensor per exposure, matching the loop's outputs.
    """
    n_exposures = len(caches)
    if n_exposures != len(candidates) or n_exposures != len(bin_indices):
        raise ValueError("caches, candidates and bin_indices must be parallel.")
    if n_exposures == 0:
        return []
    device = field.mu.device
    dtype = field.mu.dtype

    _validate_fiber_cache_batch(caches, dtype, device)

    counts = [int(candidate.numel()) for candidate in candidates]
    bin_counts = [int(index.numel()) for index in bin_indices]
    outputs: list[torch.Tensor] = [
        torch.zeros(int(count), dtype=dtype, device=device) for count in bin_counts
    ]
    live = [i for i, count in enumerate(counts) if count > 0]
    if not live:
        return outputs

    live_counts = [counts[i] for i in live]
    offsets = [0]
    for count in live_counts:
        offsets.append(offsets[-1] + count)
    # The per-splat parameter gathers keep the loop's own grouping and are
    # concatenated. A single gather over the concatenated index would be
    # cheaper still, but its backward folds every block's contribution to a
    # shared splat into one coalesce instead of the loop's separate
    # accumulations, which reassociates the gradient sum. torch.cat's backward
    # is a plain slice, so each gather node receives exactly the gradient the
    # loop hands it. The spatial stage gathers once per exposure and the SED
    # stage once per splat_chunk block, because that is what the loop does:
    # production candidate sets run to ~5900 splats, so the chunking is live.
    chunk_blocks = [
        candidates[i][start: start + int(splat_chunk)]
        for i in live
        for start in range(0, counts[i], int(splat_chunk))
    ]


    # --- per-candidate spatial weight ---------------------------------------
    # This stage stays per exposure. Its forward merges cleanly, but its
    # backward reduces over the quadrature axis of a [Q, K] tensor -- for the
    # Mahalanobis contraction and for the density's norm broadcast -- and a
    # reduction over a non-trailing axis takes its accumulation order from the
    # size of the trailing one. Merging exposures here changes each one's
    # gradient in the last bit.
    spatial = torch.cat(
        [
            candidate_aperture_weights(
                field,
                candidates[i],
                caches[i].points,
                caches[i].weights,
                caches[i].psf_cov,
                caches[i].aperture_area_arcsec2,
            )
            for i in live
        ]
    )

    # --- per-candidate SED knots -------------------------------------------
    scale = 0.5 * float(loglam_max - loglam_min)
    log_mid = 0.5 * float(loglam_max + loglam_min)
    if scale <= 0.0:
        raise ValueError("loglam_max must exceed loglam_min.")
    grid = field.sed.loglam_grid
    templates = batch_context.templates

    # softplus and the template GEMM keep the loop's blocks. softplus is the
    # one elementwise op here whose result depends on the shape it is given:
    # its vectorised CPU path leaves a scalar tail, so a merged call changes
    # the last few elements. The GEMM keeps its row count so cuBLAS cannot
    # select a different algorithm. exp and the products below are safe to
    # merge and are merged.
    chunk_offsets = [0]
    for index in chunk_blocks:
        chunk_offsets.append(chunk_offsets[-1] + int(index.numel()))
    values = torch.cat([
        F.softplus(field.sed_coefficients(index, sorted_unique=True)) @ templates
        for index in chunk_blocks
    ])
    fluxes = torch.cat([
        field.fluxes(index, sorted_unique=True) for index in chunk_blocks
    ])
    shifts = torch.cat([
        field.spectral_shifts(index, sorted_unique=True) for index in chunk_blocks
    ])
    knot_lam = torch.exp(log_mid + scale * (grid[None, :] + shifts[:, None]))
    amplitude = fluxes * spatial

    # --- bin edges and LSF, per exposure ------------------------------------
    lower_by_exposure = []
    upper_by_exposure = []
    lsf_by_exposure = []
    for i in live:
        cache = caches[i]
        local = bin_indices[i].to(dtype=torch.long, device=device)
        edges = cache.lam_edges[local]
        lower_by_exposure.append(edges[:, 0])
        upper_by_exposure.append(edges[:, 1])
        if cache.scalar_lsf_sigma is not None:
            lsf_by_exposure.append(
                torch.full(
                    (int(local.numel()),),
                    cache.scalar_lsf_sigma,
                    dtype=dtype,
                    device=device,
                )
            )
        elif cache.lsf_sigma_by_bin is not None:
            lsf_by_exposure.append(cache.lsf_sigma_by_bin[local])
        else:
            raise ValueError("Invalid spectrum cache: missing scalar and vector LSF.")

    # --- bin integration ----------------------------------------------------
    # Exposures carrying a single bin -- 60 or so out of 62 -- have one hat row
    # per candidate, so consecutive ones are merged into a single elementwise
    # pass with the loop's blocks preserved for the two knot-axis reductions.
    # An exposure carrying several bins broadcasts one knot row against each of
    # them, and the loop reduces that bin axis inside the hat weights before
    # autograd sums the knot-row partials, which the row form cannot reproduce;
    # those take the original outer-product call on their own block.
    chunk_owner = [
        j for j, _ in enumerate(live)
        for _ in range(0, live_counts[j], int(splat_chunk))
    ]
    spectral_by_chunk: list[torch.Tensor | None] = [None] * len(chunk_blocks)
    run_start = 0
    while run_start < len(chunk_blocks):
        j = chunk_owner[run_start]
        if bin_counts[live[j]] > 1:
            a, b = chunk_offsets[run_start], chunk_offsets[run_start + 1]
            spectral_by_chunk[run_start] = positive_hat_bin_average(
                knot_lam[a:b],
                values[a:b],
                lower_by_exposure[j],
                upper_by_exposure[j],
                lsf_by_exposure[j],
                validate=False,
                lsf_mode=caches[live[j]].lsf_mode,
                off_grid=field.sed.off_grid,
            )
            run_start += 1
            continue
        run_stop = run_start
        while (
            run_stop < len(chunk_blocks)
            and bin_counts[live[chunk_owner[run_stop]]] == 1
        ):
            run_stop += 1
        a = chunk_offsets[run_start]
        b = chunk_offsets[run_stop]
        row_blocks = [
            (chunk_offsets[c] - a, chunk_offsets[c + 1] - a)
            for c in range(run_start, run_stop)
        ]
        rows = [chunk_offsets[c + 1] - chunk_offsets[c] for c in range(run_start, run_stop)]
        owners = [chunk_owner[c] for c in range(run_start, run_stop)]
        merged = paired_positive_hat_bin_average(
            knot_lam[a:b],
            values[a:b],
            torch.cat([
                lower_by_exposure[o].expand(n) for o, n in zip(owners, rows)
            ]),
            torch.cat([
                upper_by_exposure[o].expand(n) for o, n in zip(owners, rows)
            ]),
            torch.cat([
                lsf_by_exposure[o].expand(n) for o, n in zip(owners, rows)
            ]),
            lsf_mode=caches[live[owners[0]]].lsf_mode,
            off_grid=field.sed.off_grid,
            blocks=row_blocks,
        )
        for c, (block_start, block_stop) in zip(
            range(run_start, run_stop), row_blocks
        ):
            spectral_by_chunk[c] = merged[block_start:block_stop].view(1, -1)
        run_start = run_stop

    for c, spectral in enumerate(spectral_by_chunk):
        j = chunk_owner[c]
        i = live[j]
        a, b = chunk_offsets[c], chunk_offsets[c + 1]
        outputs[i] = outputs[i] + torch.sum(
            spectral * amplitude[a:b][None, :], dim=1
        )
    return [
        _apply_cached_source_scale(output, cache, local)
        for output, cache, local in zip(outputs, caches, bin_indices)
    ]


# Transient budgets for the flat fiber path, in tensor elements. The pair
# stage's [P, Q] intermediates and the row stage's [R, N] ones are the only
# tensors that grow with the batch, so these bound the peak instead of the batch
# composition doing it. Pair chunking is value-neutral: the matrix inverse and
# determinant are taken once over all pairs, and every chunked operation is
# elementwise or reduces over the trailing quadrature axis. Row chunking is not,
# for the same reason the blocked path's `blocks` are not: a knot-axis reduction
# picks its accumulation order from the row count it is handed.
_FIBER_PAIR_ELEMENTS = 64_000_000
_FIBER_ROW_ELEMENTS = 32_000_000


def flat_fiber_spectra(
    field,
    caches: Sequence[SplatSpectrumCache],
    candidates: Sequence[torch.Tensor],
    bin_indices: Sequence[torch.Tensor],
    *,
    loglam_min: float,
    loglam_max: float,
    batch_context: SplatSpectrumBatchContext,
) -> torch.Tensor:
    """Every fiber bin of a spectral batch in one flat pass over pairs.

    batched_fiber_spectra keeps the per-exposure loop's arithmetic cell for
    cell, which costs it one set of kernels per exposure and per splat chunk:
    8,561 launches for 28 ms of kernel work at the paper run's shape, so the
    stage is host-bound by an order of magnitude. This evaluates the whole
    batch as one flat list of (fiber, candidate splat) pairs instead.

    Each per-splat parameter is gathered exactly once, over the sorted-unique
    union of every exposure's candidates, and the pairs then index those dense
    tensors: six sparse gathers per step rather than one per exposure and per
    chunk, which is what the backward's launch count follows.

    The terms are unchanged -- the same Gaussian quadrature per (pair, node),
    the same hat-basis bin integration per (bin, splat) row -- but three
    reductions are regrouped: the quadrature sum now runs over the trailing
    axis of a [P, Q] tensor rather than the leading axis of a [Q, K] one, the
    knot-axis sums see one block of rows rather than one per exposure and
    chunk, and the sum over candidates into a bin is taken over a padded matrix.
    So this agrees with the blocked path to fp32 reassociation, not bitwise, and
    SPLAT_FIBER_PATH selects between them.

    Returns the predictions for every exposure's bins concatenated in exposure
    order, matching torch.cat over batched_fiber_spectra's outputs.
    """
    n_exposures = len(caches)
    if n_exposures != len(candidates) or n_exposures != len(bin_indices):
        raise ValueError("caches, candidates and bin_indices must be parallel.")
    device = field.mu.device
    dtype = field.mu.dtype
    counts = [int(candidate.numel()) for candidate in candidates]
    bin_counts = [int(index.numel()) for index in bin_indices]
    n_out = sum(bin_counts)
    n_pairs = sum(counts)
    n_rows = sum(count * bins for count, bins in zip(counts, bin_counts))
    if caches:
        _validate_fiber_cache_batch(caches, dtype, device)
    if n_rows == 0:
        return torch.zeros(n_out, dtype=dtype, device=device)

    scale = 0.5 * float(loglam_max - loglam_min)
    log_mid = 0.5 * float(loglam_max + loglam_min)
    if scale <= 0.0:
        raise ValueError("loglam_max must exceed loglam_min.")

    # --- per-exposure state, stacked once ------------------------------------
    points_x, points_y = torch.stack(
        [cache.points for cache in caches]
    ).unbind(dim=-1)
    points_x = points_x.contiguous()
    points_y = points_y.contiguous()
    quadrature = torch.stack([cache.weights for cache in caches])
    psf_cov = torch.stack([cache.psf_cov for cache in caches])
    area = tensor_on_device(
        [cache.aperture_area_arcsec2 for cache in caches],
        dtype=dtype,
        device=device,
    )
    edges = torch.cat([cache.lam_edges for cache in caches])
    lsf = torch.cat(
        [_cache_lsf_sigma_full(cache, dtype, device) for cache in caches]
    )

    # --- pair and row bookkeeping -------------------------------------------
    # A pair is (exposure, candidate splat); a row is (exposure, bin, candidate
    # splat). Both are laid out in exposure order, so every per-slot quantity is
    # a gather from a small host-built table and no host sync is needed.
    n_bins = [int(cache.lam_edges.shape[0]) for cache in caches]
    owner = np.repeat(np.arange(n_exposures), bin_counts)
    rows_per_slot = np.asarray(counts, dtype=np.int64)[owner]
    slot_table = torch.as_tensor(
        np.stack(
            (
                rows_per_slot,
                _exclusive_cumsum(counts)[owner],
                _exclusive_cumsum(n_bins)[owner],
            )
        ),
        dtype=torch.long,
    )
    slot_table = copy_tensor_to_device(
        slot_table, device=device, non_blocking=True
    )
    slot_rows, slot_pair_base, slot_edge_base = slot_table.unbind(dim=0)
    pair_fiber = torch.repeat_interleave(
        torch.arange(n_exposures, device=device),
        as_tensor_on_device(counts, dtype=torch.long, device=device),
        output_size=n_pairs,
    )
    row_slot = torch.repeat_interleave(
        torch.arange(n_out, device=device), slot_rows, output_size=n_rows
    )
    row_within = (
        torch.arange(n_rows, device=device)
        - (slot_rows.cumsum(0) - slot_rows)[row_slot]
    )
    row_pair = slot_pair_base[row_slot] + row_within
    slot_bin = torch.cat(bin_indices).to(dtype=torch.long) + slot_edge_base
    slot_edges = edges[slot_bin]
    slot_lower = slot_edges[:, 0]
    slot_upper = slot_edges[:, 1]
    slot_lsf = lsf[slot_bin]

    # --- one gather per per-splat parameter, over the candidate union --------
    pair_splat = torch.cat(list(candidates))
    union, pair_row = torch.unique(pair_splat, return_inverse=True)
    if fused_splat_gather_enabled() and field.sparse_gradients_enabled:
        gathered = field.gather_splat_parameters(union, sorted_unique=True)
        union_mu, log_scales, theta, raw_flux, sed_coeffs, sed_shift = gathered
        a, b, d = field.covariance_components_from_rows(log_scales, theta)
        union_cov = torch.stack(
            (torch.stack((a, b), dim=-1), torch.stack((b, d), dim=-1)), dim=-2
        )
        union_flux = F.softplus(raw_flux)
        union_values = F.softplus(sed_coeffs) @ batch_context.templates
        union_knot_lam = torch.exp(
            log_mid
            + scale
            * (field.sed.loglam_grid[None, :] + sed_shift[:, None])
        )
    else:
        union_mu = field.positions(union, sorted_unique=True)
        # covariance_components is the same matrix without the batched 2x2 GEMMs
        # covariances() runs: 0.13 ms against 7.6 ms over a 206k-splat union, and
        # the same two gathers underneath.
        a, b, d = field.covariance_components(union, sorted_unique=True)
        union_cov = torch.stack(
            (torch.stack((a, b), dim=-1), torch.stack((b, d), dim=-1)), dim=-2
        )
        union_flux = field.fluxes(union, sorted_unique=True)
        union_values = (
            F.softplus(field.sed_coefficients(union, sorted_unique=True))
            @ batch_context.templates
        )
        union_knot_lam = torch.exp(
            log_mid
            + scale
            * (
                field.sed.loglam_grid[None, :]
                + field.spectral_shifts(union, sorted_unique=True)[:, None]
            )
        )

    # --- per-pair spatial weight --------------------------------------------
    precision, norm = paired_precision_and_norm(
        union_cov[pair_row] + psf_cov[pair_fiber]
    )
    pair_mu = union_mu[pair_row]
    pair_chunk = max(1, _FIBER_PAIR_ELEMENTS // int(points_x.shape[1]))
    spatial = torch.cat(
        [
            paired_gaussian_quadrature(
                points_x,
                points_y,
                quadrature,
                pair_fiber[start : start + pair_chunk],
                pair_mu[start : start + pair_chunk],
                tuple(entry[start : start + pair_chunk] for entry in precision),
                norm[start : start + pair_chunk],
            )
            for start in range(0, n_pairs, pair_chunk)
        ]
    )
    amplitude = union_flux[pair_row] * (spatial * area[pair_fiber])

    # --- per-row bin integration --------------------------------------------
    n_knots = int(union_knot_lam.shape[1])
    row_chunk = max(1, _FIBER_ROW_ELEMENTS // n_knots)
    row_splat = pair_row[row_pair]
    contribution = torch.cat(
        [
            paired_positive_hat_bin_average(
                union_knot_lam[row_splat[start : start + row_chunk]],
                union_values[row_splat[start : start + row_chunk]],
                slot_lower[row_slot[start : start + row_chunk]],
                slot_upper[row_slot[start : start + row_chunk]],
                slot_lsf[row_slot[start : start + row_chunk]],
                lsf_mode=caches[0].lsf_mode,
                off_grid=field.sed.off_grid,
                blocks=[(0, min(row_chunk, n_rows - start))],
            )
            * amplitude[row_pair[start : start + row_chunk]]
            for start in range(0, n_rows, row_chunk)
        ]
    )

    # --- sum over each bin's candidates -------------------------------------
    # Rows of one bin are contiguous, so scattering them into a padded matrix
    # and reducing its trailing axis keeps the sum deterministic; an index_add
    # over bins would accumulate with atomics.
    width = int(rows_per_slot.max())
    output = (
        torch.zeros(n_out * width, dtype=dtype, device=device)
        .scatter(0, row_slot * width + row_within, contribution)
        .view(n_out, width)
        .sum(dim=1)
    )
    if any(cache.spectral_source_scale is not None for cache in caches):
        response = torch.cat([
            (torch.ones(int(local.numel()), dtype=dtype, device=device)
             if cache.spectral_source_scale is None
             else cache.spectral_source_scale[local.to(device=device, dtype=torch.long)])
            for cache, local in zip(caches, bin_indices)
        ])
        output = output * response
    return output


def splat_rendered_significance(
    field,
    bandpasses,
    band_sigma: dict,
    band_psf_variance: dict,
    *,
    loglam_min: float,
    loglam_max: float,
) -> torch.Tensor:
    """Per-splat rendered peak significance, max over bands.

    For band b: flux * A_b / (2 pi sqrt(det(Sigma + v_b I))) / sigma_b, where
    A_b is the band-average SED amplitude per unit flux, v_b the band's PSF
    variance in model normalized units, and sigma_b its image noise. This is
    the splat's peak surface brightness AS RENDERED over the noise: a
    shrinking compact splat saturates at its PSF-limited peak instead of
    claiming divergent intrinsic significance. Gauge-invariant by
    construction — flux * A_b is the rendered band amplitude, independent of
    the raw_flux/SED split, with no flux division (and so no clamp) anywhere.
    """
    if set(bandpasses) != set(band_sigma) or set(bandpasses) != set(band_psf_variance):
        raise ValueError(
            "bandpasses, band_sigma, and band_psf_variance must share keys."
        )
    for key in bandpasses:
        sigma = float(band_sigma[key])
        variance = float(band_psf_variance[key])
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"band_sigma[{key!r}] must be finite and positive.")
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError(
                f"band_psf_variance[{key!r}] must be finite and non-negative."
            )
    with torch.no_grad():
        unit_amplitudes = splat_band_amplitudes(
            field,
            bandpasses,
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            include_flux=False,
        )
        flux = F.softplus(field.raw_flux.detach())
        scales_sq = torch.exp(2.0 * field.log_scales.detach())
        # det(R diag(s^2) R^T + v I) = (s1^2 + v)(s2^2 + v): rotation commutes
        # with the isotropic PSF term, so theta drops out of the determinant.
        best = None
        for key in bandpasses:
            variance = float(band_psf_variance[key])
            determinant = (scales_sq[:, 0] + variance) * (scales_sq[:, 1] + variance)
            peak = flux * unit_amplitudes[key] / (
                2.0 * math.pi * torch.sqrt(determinant)
            )
            significance = peak / float(band_sigma[key])
            best = significance if best is None else torch.maximum(best, significance)
    return best


def template_band_averages(
    sed,
    bandpasses,
    *,
    loglam_min: float,
    loglam_max: float,
) -> tuple[list, torch.Tensor]:
    """Band-average of each raw SED template through each bandpass.

    Returns (band_keys, (K, B) tensor): the linear map from softplus
    coefficient space to per-unit-flux band amplitudes, used to project
    observed band ratios onto the dictionary when initializing born splats.
    Same integration internals as splat_band_amplitudes at zero shift.
    """
    if not bandpasses:
        raise ValueError("At least one bandpass is required.")
    grid = sed.loglam_grid
    device = grid.device
    dtype = grid.dtype
    scale = 0.5 * float(loglam_max - loglam_min)
    log_mid = 0.5 * float(loglam_max + loglam_min)
    C = _band_integration_grid(grid, log_mid, scale)
    templates = sed.templates()  # (K, n_grid), aligned with C's grid points
    band_keys = []
    columns = []
    drop_off_grid = sed.off_grid == "zero"
    ones = torch.ones(1, device=device, dtype=dtype)
    for key, (lam, throughput) in bandpasses.items():
        lam_tensor = torch.as_tensor(lam, dtype=dtype, device=device)
        throughput_tensor = torch.as_tensor(throughput, dtype=dtype, device=device)
        order = torch.argsort(lam_tensor)
        tables = _hat_band_tables(
            lam_tensor[order], throughput_tensor[order], dtype=dtype, device=device
        )
        width = _hat_window_width(C, tables[-2], tables[-1])
        weights = _HatBandWeights.apply(
            ones, C, *tables, width, drop_off_grid
        )  # (1, n_grid)
        band_keys.append(key)
        columns.append((templates * weights).sum(dim=1))
    return band_keys, torch.stack(columns, dim=1)
