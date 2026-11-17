"""Environment toggles for hot-path performance.

SPLAT_NO_CHECKPOINT=0 recomputes hot-path chunk activations in backward
instead of storing them (the default stores them). Gradients are
bitwise-identical; storing costs peak memory (~20 GiB at a 10M-splat census
with a 10k-pixel batch), so a smaller GPU may need the recompute path.

SPLAT_EXACT_PAIR_PRUNE=0 restores the legacy tile-truncated support, which
kept some pairs beyond n_sigma and cut others at tile boundaries. The exact
Mahalanobis cutoff is the documented operator and is on by default; disable
it only for comparison against runs trained with the tile-shaped support.

SPLAT_AMP_TRITON=0 restores the eager ATen chain behind the band-amplitude
hat weights instead of the fused Triton kernels. The two produce bitwise
identical weights and derivatives (tests/test_hat_band_kernels.py), so this
is a kill switch for a machine whose Triton install is broken, not a
scientific choice.

SPLAT_PAIR_CHUNK / SPLAT_AMP_CHUNK override the pair-evaluation and
band-amplitude chunk sizes to cut per-chunk kernel-launch overhead.
Changing a chunk size moves chunk boundaries and therefore the grouping of
duplicate-index accumulations, so results differ at floating-point
reassociation level (within the nondeterminism the CUDA index_add path
already has). Hold these fixed across arms of a controlled comparison.

SPLAT_FIBER_PATH=blocked restores the per-exposure/per-chunk fiber spectral
evaluation. The default "flat" path evaluates every (exposure, candidate)
pair in one set of kernels, which keeps the same terms but regroups the
quadrature and knot-axis reductions, so the two agree only to floating-point
reassociation level. Hold this fixed across arms of a controlled comparison.

SPLAT_FIBER_STREAM=default restores serial image and fiber evaluation on the
current CUDA stream. The default "side" path issues the image work first and
then evaluates the independent fiber forward on a side stream. Both streams
are joined before their results are combined and after backward.
SPLAT_FUSED_GATHER=0 restores the six independent per-parameter row gathers.
The default fused path copies the same float32 values in one CUDA kernel and
returns the same sparse COO gradient blocks.
SPLAT_ADAM_PATH=eager restores the eager row-gathered SparseAdam update. The
default "fused" path performs the same fp32 operations and roundings in one
Triton kernel per parameter.
SPLAT_PINNED_H2D=0 restores direct transfers from pageable host memory. The
default stages per-step host values in pinned memory and enqueues non-blocking
copies, preserving every byte while avoiding host-side pipeline drains.
SPLAT_RASTER_PAIRS=eager restores the eager ATen raster pair evaluation. The
default "fused" path combines its gathers and scalar arithmetic in one Triton
kernel while preserving every float32 rounding point. Checkpointed raster
chunks retain the eager path because their purpose is to recompute it.

SPLAT_RASTER_PAIRS_BACKWARD=autograd recomputes the original eager expression
and uses its exact autograd graph. The default "fused" path evaluates the same
VJP with explicit rounding, then uses the original ATen scatter operations.

SPLAT_BAND_CONTRACTION defaults to "exact": sparse storage with the verified
cuBLAS operation order on the recorded B200/PyTorch/CUDA runtime. Other runtimes,
shapes and short 65,535-row batch tails retain the dense implementation.
"dense" is the reference kill switch. "sparse" remains an opt-in
compensated reassociation.

SPLAT_FIBER_BINS defaults to "fused" and SPLAT_FIBER_BINS_BACKWARD to "fused".
The generated VJP preserves all arithmetic nodes and all 18 ATen backward
reductions. It supports the production 96-knot, single-block case with gradients
only in wavelength knots and values; other cases retain eager recomputation.
"eager" and "autograd" restore the reference paths. CUDA graph replay remains
available as "graphed" but was slower.

SPLAT_PAIR_INDEX=fused keeps the reference tile grid, candidate census and
ordered pair list while fusing index construction and exact support tests.
SPLAT_BATCHED_CALIBRATION=1 batches immutable-table gathers and returns each
exposure's log-scale gradient in the original autograd accumulation order.
SPLAT_COMPACT_BANDS=1 keeps only active knot windows with the verified contraction
and reduction orders. It follows the same runtime and short-tail fallbacks.
SPLAT_FUSED_COVARIANCE=1 and SPLAT_FUSED_QUADRATURE=1 fuse exact elementwise
forward/backward work. Single-node quadrature and differentiable quadrature
data keep the eager path. SPLAT_DEVICE_INDEX_FILL=1 avoids CPU scalar transfers
for boolean membership writes. These are the validated defaults; setting their
boolean switches to 0 (or the pair index to "eager") restores the references.

The eigen-MoG and maintenance switches below default to the execution path
the paper runs used. On the tested source and shapes, a strict state
comparison found that path equal to the reference over 32 updates (all 36
state tensors and the prune masks); that is an empirical result, not a
guarantee, and FMA in general changes rounding. Setting any of them to 0
selects the reference:
SPLAT_EIGEN_FAST_CONTEXT, SPLAT_EIGEN_SHAPE_CACHE, SPLAT_EIGEN_FUSED_INDEX,
SPLAT_EIGEN_REUSE_GATHER, SPLAT_EIGEN_FUSED_PREDICT, SPLAT_EIGEN_STRICT_VJP,
SPLAT_EIGEN_FULL_FMA, SPLAT_EIGEN_ASYNC_VALIDATE, SPLAT_COALESCE_DIRECT,
SPLAT_COMPACT_ALL_BANDS, SPLAT_PRUNE_ONLY_FAST, SPLAT_FAST_SPECTRAL_GROUPS.
execution_flags() records every SPLAT_* knob a run resolved, for checkpoints.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _binary_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1'.")
    return value == "1"


def activation_checkpointing_disabled() -> bool:
    return _binary_flag("SPLAT_NO_CHECKPOINT", "1")


def eigen_fast_context_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_FAST_CONTEXT", "1")


def eigen_shape_cache_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_SHAPE_CACHE", "1")


def eigen_fused_index_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_FUSED_INDEX", "1")


def eigen_reuse_gather_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_REUSE_GATHER", "1")


def eigen_fused_predict_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_FUSED_PREDICT", "1")


def eigen_strict_vjp_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_STRICT_VJP", "1")


def eigen_full_fma_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_FULL_FMA", "1")


def eigen_async_validate_enabled() -> bool:
    return _binary_flag("SPLAT_EIGEN_ASYNC_VALIDATE", "1")


def coalesce_direct_enabled() -> bool:
    return _binary_flag("SPLAT_COALESCE_DIRECT", "1")


def compact_all_bands_enabled() -> bool:
    return _binary_flag("SPLAT_COMPACT_ALL_BANDS", "1")


def prune_only_fast_enabled() -> bool:
    return _binary_flag("SPLAT_PRUNE_ONLY_FAST", "1")


def fast_spectral_groups_enabled() -> bool:
    return _binary_flag("SPLAT_FAST_SPECTRAL_GROUPS", "1")


def exact_pair_pruning_enabled() -> bool:
    return os.environ.get("SPLAT_EXACT_PAIR_PRUNE", "1") == "1"


def fused_band_weights_enabled() -> bool:
    return os.environ.get("SPLAT_AMP_TRITON", "1") == "1"


def flat_fiber_path_enabled() -> bool:
    value = os.environ.get("SPLAT_FIBER_PATH", "flat")
    if value not in ("flat", "blocked"):
        raise ValueError("SPLAT_FIBER_PATH must be 'flat' or 'blocked'.")
    return value == "flat"


def fiber_side_stream_enabled() -> bool:
    value = os.environ.get("SPLAT_FIBER_STREAM", "side")
    if value not in ("side", "default"):
        raise ValueError("SPLAT_FIBER_STREAM must be 'side' or 'default'.")
    return value == "side"


def fused_splat_gather_enabled() -> bool:
    return os.environ.get("SPLAT_FUSED_GATHER", "1") == "1"


def fused_row_sparse_adam_enabled() -> bool:
    value = os.environ.get("SPLAT_ADAM_PATH", "fused")
    if value not in ("fused", "eager"):
        raise ValueError("SPLAT_ADAM_PATH must be 'fused' or 'eager'.")
    return value == "fused"


def pinned_h2d_enabled() -> bool:
    value = os.environ.get("SPLAT_PINNED_H2D", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_PINNED_H2D must be '0' or '1'.")
    return value == "1"


def fused_raster_pairs_enabled() -> bool:
    value = os.environ.get("SPLAT_RASTER_PAIRS", "fused")
    if value not in ("fused", "eager"):
        raise ValueError("SPLAT_RASTER_PAIRS must be 'fused' or 'eager'.")
    return value == "fused"


def fused_raster_pairs_backward_enabled() -> bool:
    value = os.environ.get("SPLAT_RASTER_PAIRS_BACKWARD", "fused")
    if value not in ("fused", "autograd"):
        raise ValueError(
            "SPLAT_RASTER_PAIRS_BACKWARD must be 'fused' or 'autograd'."
        )
    return value == "fused"


def band_contraction_mode() -> str:
    value = os.environ.get("SPLAT_BAND_CONTRACTION", "exact")
    if value not in ("dense", "sparse", "exact"):
        raise ValueError("SPLAT_BAND_CONTRACTION must be 'dense' or 'sparse' or 'exact'.")
    return value


def sparse_band_contraction_enabled() -> bool:
    return band_contraction_mode() == "sparse"


def exact_band_contraction_enabled() -> bool:
    return band_contraction_mode() == "exact"


def exact_band_reduction_rows_supported(rows: int) -> bool:
    """cuBLAS splits batches at 65,535; short tails use another reduction."""
    tail = int(rows) % 65_535
    return rows >= 4096 and (tail == 0 or tail >= 4096)


def fused_pair_index_enabled() -> bool:
    value = os.environ.get("SPLAT_PAIR_INDEX", "fused")
    if value not in ("fused", "eager"):
        raise ValueError("SPLAT_PAIR_INDEX must be 'fused' or 'eager'.")
    return value == "fused"


def batched_spectral_calibration_enabled() -> bool:
    value = os.environ.get("SPLAT_BATCHED_CALIBRATION", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_BATCHED_CALIBRATION must be '0' or '1'.")
    return value == "1"


def compact_band_weights_enabled() -> bool:
    value = os.environ.get("SPLAT_COMPACT_BANDS", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_COMPACT_BANDS must be '0' or '1'.")
    return value == "1"


def fused_covariance_enabled() -> bool:
    value = os.environ.get("SPLAT_FUSED_COVARIANCE", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_FUSED_COVARIANCE must be '0' or '1'.")
    return value == "1"


def fused_quadrature_enabled() -> bool:
    value = os.environ.get("SPLAT_FUSED_QUADRATURE", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_FUSED_QUADRATURE must be '0' or '1'.")
    return value == "1"


def device_index_fill_enabled() -> bool:
    value = os.environ.get("SPLAT_DEVICE_INDEX_FILL", "1")
    if value not in ("0", "1"):
        raise ValueError("SPLAT_DEVICE_INDEX_FILL must be '0' or '1'.")
    return value == "1"


def fiber_bins_mode() -> str:
    value = os.environ.get("SPLAT_FIBER_BINS", "fused")
    if value not in ("graphed", "fused", "eager"):
        raise ValueError(
            "SPLAT_FIBER_BINS must be 'graphed', 'fused', or 'eager'."
        )
    return value


def fused_fiber_bins_enabled() -> bool:
    return fiber_bins_mode() == "fused"


def fused_fiber_bins_backward_enabled() -> bool:
    value = os.environ.get("SPLAT_FIBER_BINS_BACKWARD", "fused")
    if value not in ("fused", "autograd"):
        raise ValueError("SPLAT_FIBER_BINS_BACKWARD must be 'fused' or 'autograd'.")
    return value == "fused"


# Chunk sizes the paper runs used;
# they are the defaults of every call site that reads SPLAT_AMP_CHUNK or
# SPLAT_PAIR_CHUNK. 65,536 amplitude rows would also miss the exact band
# reduction (exact_band_reduction_rows_supported) and take the dense path.
# Each chunk is a large block on the GPU; a smaller card may need smaller
# values through the environment variables.
AMP_CHUNK = 8_000_000
PAIR_CHUNK = 16_000_000


def env_chunk(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive chunk size.")
    return value


def _positive_int(name: str, default: str) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def eigen_block() -> int:
    return _positive_int("SPLAT_EIGEN_BLOCK", "32")


def eigen_cblock() -> int:
    return _positive_int("SPLAT_EIGEN_CBLOCK", "8")


def eigen_full_block() -> int:
    return _positive_int("SPLAT_EIGEN_FULL_BLOCK", "32")


def eigen_full_cblock() -> int:
    return _positive_int("SPLAT_EIGEN_FULL_CBLOCK", "8")


def eigen_full_warps() -> int:
    return _positive_int("SPLAT_EIGEN_FULL_WARPS", "4")


def eigen_exp_poly_enabled() -> bool:
    """Polynomial exponential in the full-pairs kernel: an approximation, off by default."""
    return _binary_flag("SPLAT_EIGEN_EXP_POLY", "0")


def eigen_shared_recip_enabled() -> bool:
    """Shared reciprocal in the full-pairs kernel: changes rounding, off by default."""
    return _binary_flag("SPLAT_EIGEN_SHARED_RECIP", "0")


def eigen_full_backward_mode() -> str:
    value = os.environ.get("SPLAT_EIGEN_FULL_BACKWARD", "torch")
    if value not in ("torch", "triton"):
        raise ValueError("SPLAT_EIGEN_FULL_BACKWARD must be 'torch' or 'triton'.")
    return value


def calibration_singleton_mode() -> str:
    return os.environ.get("SPLAT_CALIB_SINGLETONS", "0")


def packed_calibration_enabled() -> bool:
    return os.environ.get("SPLAT_CALIB_PACKED", "0") == "1"


@lru_cache(maxsize=1)
def validated_cublas_runtime() -> bool:
    """The exact cuBLAS-order kernels are tied to this verified runtime."""
    from importlib import metadata

    import torch

    try:
        return (
            torch.__version__ == "2.11.0+cu130"
            and torch.version.git_version == "70d99e998b4955e0049d13a98d77ae1b14db1f45"
            and metadata.version("triton") == "3.6.0"
            and metadata.version("nvidia-cublas") == "13.1.0.3"
        )
    except metadata.PackageNotFoundError:
        return False


_RESOLVED_FLAGS = {
    "SPLAT_NO_CHECKPOINT": activation_checkpointing_disabled,
    "SPLAT_EXACT_PAIR_PRUNE": exact_pair_pruning_enabled,
    "SPLAT_AMP_TRITON": fused_band_weights_enabled,
    "SPLAT_FIBER_PATH": flat_fiber_path_enabled,
    "SPLAT_FIBER_STREAM": fiber_side_stream_enabled,
    "SPLAT_FUSED_GATHER": fused_splat_gather_enabled,
    "SPLAT_ADAM_PATH": fused_row_sparse_adam_enabled,
    "SPLAT_PINNED_H2D": pinned_h2d_enabled,
    "SPLAT_RASTER_PAIRS": fused_raster_pairs_enabled,
    "SPLAT_RASTER_PAIRS_BACKWARD": fused_raster_pairs_backward_enabled,
    "SPLAT_BAND_CONTRACTION": band_contraction_mode,
    "SPLAT_PAIR_INDEX": fused_pair_index_enabled,
    "SPLAT_BATCHED_CALIBRATION": batched_spectral_calibration_enabled,
    "SPLAT_COMPACT_BANDS": compact_band_weights_enabled,
    "SPLAT_FUSED_COVARIANCE": fused_covariance_enabled,
    "SPLAT_FUSED_QUADRATURE": fused_quadrature_enabled,
    "SPLAT_DEVICE_INDEX_FILL": device_index_fill_enabled,
    "SPLAT_FIBER_BINS": fiber_bins_mode,
    "SPLAT_FIBER_BINS_BACKWARD": fused_fiber_bins_backward_enabled,
    "SPLAT_EIGEN_FAST_CONTEXT": eigen_fast_context_enabled,
    "SPLAT_EIGEN_SHAPE_CACHE": eigen_shape_cache_enabled,
    "SPLAT_EIGEN_FUSED_INDEX": eigen_fused_index_enabled,
    "SPLAT_EIGEN_REUSE_GATHER": eigen_reuse_gather_enabled,
    "SPLAT_EIGEN_FUSED_PREDICT": eigen_fused_predict_enabled,
    "SPLAT_EIGEN_STRICT_VJP": eigen_strict_vjp_enabled,
    "SPLAT_EIGEN_FULL_FMA": eigen_full_fma_enabled,
    "SPLAT_EIGEN_ASYNC_VALIDATE": eigen_async_validate_enabled,
    "SPLAT_COALESCE_DIRECT": coalesce_direct_enabled,
    "SPLAT_COMPACT_ALL_BANDS": compact_all_bands_enabled,
    "SPLAT_PRUNE_ONLY_FAST": prune_only_fast_enabled,
    "SPLAT_FAST_SPECTRAL_GROUPS": fast_spectral_groups_enabled,
    "SPLAT_EIGEN_BLOCK": eigen_block,
    "SPLAT_EIGEN_CBLOCK": eigen_cblock,
    "SPLAT_EIGEN_FULL_BLOCK": eigen_full_block,
    "SPLAT_EIGEN_FULL_CBLOCK": eigen_full_cblock,
    "SPLAT_EIGEN_FULL_WARPS": eigen_full_warps,
    "SPLAT_EIGEN_EXP_POLY": eigen_exp_poly_enabled,
    "SPLAT_EIGEN_SHARED_RECIP": eigen_shared_recip_enabled,
    "SPLAT_EIGEN_FULL_BACKWARD": eigen_full_backward_mode,
    "SPLAT_CALIB_SINGLETONS": calibration_singleton_mode,
    "SPLAT_CALIB_PACKED": packed_calibration_enabled,
    "SPLAT_AMP_CHUNK": lambda: env_chunk("SPLAT_AMP_CHUNK", AMP_CHUNK),
    "SPLAT_PAIR_CHUNK": lambda: env_chunk("SPLAT_PAIR_CHUNK", PAIR_CHUNK),
}


def execution_flags() -> dict[str, object]:
    """The resolved value of every SPLAT_* knob the code reads, plus each SPLAT_* variable set verbatim."""
    return {
        "resolved": {name: accessor() for name, accessor in _RESOLVED_FLAGS.items()},
        "environment": {
            name: value for name, value in sorted(os.environ.items()) if name.startswith("SPLAT_")
        },
    }
