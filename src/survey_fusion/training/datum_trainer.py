from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from ..data.grids import SpatialGrid
from ..data.image_variance import (
    SIGMA_RECONSTRUCTION_RTOL,
    image_variance_outlier_pull,
    sigma_reconstruction_error,
)
from ..data.random_holdout import random_holdout_local_indices
from ..pgm.contracts import (
    DatumKind,
    ExposureState,
    GroupedKernelSamples,
    MaskApertureSpec,
    PSFMixtureComponents,
    SkyDiskApertureSpec,
    aperture_area_arcsec2,
    validated_spectral_source_scale, validate_spectral_source_psf,
    validate_batch_datum_kinds,
    validate_batch_exposure_links,
)
from ..pgm.likelihoods import (
    current_model_image_sigma,
    datum_unbiased_gaussian_nll,
    population_weighted_modality_nll,
)
from ..pgm.splat_raster import (
    DEFAULT_PAIR_N_SIGMA,
    PointPSFMixture,
    build_mixture_packed_point_index,
    build_packed_point_index,
    packed_mixture_multiband_point_prediction,
    packed_multiband_point_prediction,
    tiled_mixture_point_prediction,
    tiled_point_prediction,
)
from ..pgm.splat_spectrum import (
    SplatSpectrumCache,
    batched_fiber_candidates,
    batched_fiber_spectra,
    build_spectrum_batch_context,
    build_spectrum_cache,
    flat_fiber_spectra,
    splat_band_amplitudes,
    splat_fiber_spectrum,
)
from ..utils.distributed import DistributedContext, is_distributed_active
from ..utils.perf_env import (
    PAIR_CHUNK,
    batched_spectral_calibration_enabled,
    coalesce_direct_enabled,
    device_index_fill_enabled,
    eigen_fast_context_enabled,
    execution_flags,
    fast_spectral_groups_enabled,
    fiber_side_stream_enabled,
    flat_fiber_path_enabled,
    fused_splat_gather_enabled,
)
from ..utils.torch_utils import (
    as_tensor_on_device,
    copy_tensor_to_device,
    resolve_amp_settings,
    startup_worker_count,
    tensor_on_device,
)
from ..utils.wcs import radec_deg_to_omega_xyz
from .checkpoints import (
    PSF_JACOBIAN_ALGORITHM,
    PSF_JACOBIAN_CAP_POLICY,
    PSF_JACOBIAN_MAX_TILES_PER_AXIS,
    PSF_MIXTURE_PROVENANCE_CONVENTION,
)
from .renderer import (
    SpectralRenderer,
    convolve_flat_image,
    gaussian_mixture_psf_kernel,
    gaussian_psf_kernel,
    integrate_aperture,
)
from .row_sparse_adam import RowSparseAdam
from .spectral_calibration import SpectralCalibrationConfig, SpectralNuisanceCalibrator


def transform_psf_mixture_to_normalized(
    mixture: PSFMixtureComponents,
    jacobian: torch.Tensor,
    *,
    pixel_variance: float = 0.0,
    isotropy_tolerance: float = 1e-3,
) -> PointPSFMixture:
    """Transform native detector-pixel covariances through a full 2x2 Jacobian."""
    if jacobian.shape != (2, 2):
        raise ValueError("Pixel-to-normalized Jacobian must have shape (2, 2).")
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("Pixel-to-normalized Jacobian contains non-finite values.")
    if not math.isfinite(float(pixel_variance)) or float(pixel_variance) < 0.0:
        raise ValueError("pixel_variance must be finite and non-negative.")
    rows = torch.as_tensor(mixture.components, dtype=jacobian.dtype, device=jacobian.device)
    weights = rows[:, 0].unsqueeze(0)
    native = torch.zeros((rows.shape[0], 2, 2), dtype=jacobian.dtype, device=jacobian.device)
    native[:, 0, 0] = rows[:, 1]
    native[:, 0, 1] = rows[:, 2]
    native[:, 1, 0] = rows[:, 2]
    native[:, 1, 1] = rows[:, 3]
    if pixel_variance:
        native = native + float(pixel_variance) * torch.eye(
            2, dtype=jacobian.dtype, device=jacobian.device
        ).unsqueeze(0)
    transformed = jacobian.unsqueeze(0) @ native @ jacobian.T.unsqueeze(0)
    native_isotropic = bool(torch.allclose(
        native[:, 0, 0], native[:, 1, 1], rtol=0.0, atol=1e-12
    )) and bool(torch.all(native[:, 0, 1].abs() <= 1e-12))
    metric = jacobian @ jacobian.T
    mean_scale = 0.5 * torch.trace(metric)
    relative_deviation = torch.linalg.matrix_norm(
        metric - mean_scale * torch.eye(2, dtype=metric.dtype, device=metric.device)
    ) / mean_scale.clamp_min(torch.finfo(metric.dtype).tiny)
    if native_isotropic and float(relative_deviation) < float(isotropy_tolerance):
        variances = 0.5 * (transformed[:, 0, 0] + transformed[:, 1, 1])
        return PointPSFMixture(weights, variances.unsqueeze(0))
    compact = torch.stack(
        (transformed[:, 0, 0], transformed[:, 0, 1], transformed[:, 1, 1]), dim=-1
    )
    return PointPSFMixture(weights, compact.unsqueeze(0))


def _stack_point_psf_mixtures(
    mixtures: list[PointPSFMixture],
) -> PointPSFMixture:
    """Stack one-row mixtures, promoting isotropic rows when necessary."""
    if not mixtures:
        raise ValueError("At least one PSF mixture row is required.")
    weights = torch.cat([mixture.weights for mixture in mixtures], dim=0)
    all_isotropic = all(mixture.isotropic for mixture in mixtures)

    def covariance_rows(mixture: PointPSFMixture) -> torch.Tensor:
        if all_isotropic or not mixture.isotropic:
            return mixture.covariances
        zero = torch.zeros_like(mixture.covariances)
        return torch.stack(
            (mixture.covariances, zero, mixture.covariances), dim=-1
        )

    return PointPSFMixture(
        weights,
        torch.cat([covariance_rows(mixture) for mixture in mixtures], dim=0),
    )


@dataclass(slots=True)
class TrainerState:
    history: Dict[str, List[float]]
    band_losses: Dict[str, float]
    amplitude: float
    epochs_ran: int = 0
    steps_ran: int = 0
    steps_ran_this_invocation: int = 0


@dataclass(slots=True)
class ExposureRuntime:
    exposure: ExposureState
    renderer: SpectralRenderer
    psf_kernel: torch.Tensor | None
    spec_projector: Any | None
    omega_xyz: torch.Tensor | None
    fiber_indices: torch.Tensor | None
    aperture_weights: torch.Tensor | None
    aperture_support: torch.Tensor | None
    height: int
    width: int


@dataclass(slots=True)
class ImageImportanceSamplingConfig:
    enabled: bool = False
    max_importance_weight: float = 20.0


class _NullProgressBar:
    def __init__(self) -> None:
        self.format_dict: dict[str, Any] = {}

    def write(self, _text: str) -> None:
        return

    def set_postfix(self, **_kwargs: Any) -> None:
        return

    def update(self, _n: int = 1) -> None:
        return

    def __enter__(self) -> "_NullProgressBar":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool | None:
        return None


_SPEED_PROFILE_TIMING_KEYS: tuple[str, ...] = (
    "kernel_sampling_sec",
    "kernel_plan_sec",
    "kernel_sample_pair_sec",
    "model_forward_sec",
    "backward_sec",
    "optimizer_sec",
    "image_forward_index_sec",
    "image_forward_amplitude_sec",
    "image_forward_raster_sec",
)


class _IndexDataset(Dataset[torch.Tensor]):
    """A minimal Dataset that yields global datum indices.

    Notes
    -----
    The backing tensor is always stored on CPU to keep DataLoader worker
    processes CUDA-free.
    """

    def __init__(self, indices: torch.Tensor):
        if indices.ndim != 1:
            raise ValueError("indices must be a 1D tensor.")
        self._indices = indices.detach().to(dtype=torch.long).cpu()

    def __len__(self) -> int: # type: ignore[override]
        return int(self._indices.numel())

    def __getitem__(self, idx: int) -> torch.Tensor: # type: ignore[override]
        return self._indices[idx]


class _CheckpointEpochSampler(Sampler[int]):
    """Yield a reproducible epoch permutation without consuming global RNG."""

    def __init__(self, size: int, *, base_seed: int, shuffle: bool):
        self._size = int(size)
        self._base_seed = int(base_seed)
        self._shuffle = bool(shuffle)
        self._epoch = 0
        self._cycle = 0
        self._skip_datums = 0

    def __len__(self) -> int:
        return self._size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        self._epoch = int(epoch)
        self._cycle = 0
        self._skip_datums = 0

    def set_resume_cursor(
        self, consumed_batches: int, *, batch_size: int, drop_last: bool,
    ) -> None:
        if consumed_batches < 0:
            raise ValueError(
                f"consumed_batches must be non-negative, got {consumed_batches}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        batches_per_cycle = (
            self._size // batch_size
            if drop_last
            else (self._size + batch_size - 1) // batch_size
        )
        if batches_per_cycle <= 0:
            raise ValueError(
                "Cannot resume a non-empty checkpoint sampler with no complete "
                f"batches (size={self._size}, batch_size={batch_size}, "
                f"drop_last={drop_last})."
            )
        self._cycle, remainder = divmod(consumed_batches, batches_per_cycle)
        self._skip_datums = int(remainder * batch_size)
        if self._skip_datums >= self._size:
            raise ValueError(
                "Checkpoint sampler cursor skips beyond the current permutation: "
                f"skip_datums={self._skip_datums}, size={self._size}."
            )

    def _iterator_seed(self, cycle: int) -> int:
        packed = struct.pack(
            "<QQQ",
            self._base_seed & ((1 << 64) - 1),
            int(self._epoch),
            int(cycle),
        )
        digest = hashlib.blake2b(packed, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)

    def iterator_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._iterator_seed(self._cycle))
        return generator

    def __iter__(self):
        cycle = int(self._cycle)
        skip_datums = int(self._skip_datums)
        self._cycle += 1
        self._skip_datums = 0
        if self._shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._iterator_seed(cycle))
            order = torch.randperm(self._size, generator=generator)
        else:
            order = torch.arange(self._size, dtype=torch.long)
        if skip_datums < 0 or skip_datums > int(order.numel()):
            raise RuntimeError(
                "Checkpoint sampler produced an invalid skip cursor: "
                f"skip_datums={skip_datums}, permutation_size={order.numel()}."
            )
        yield from order[skip_datums:].tolist()


def _fiber_mask_sum(mask) -> float:
    """Total aperture weight, for dense arrays and sparse FiberMask alike."""
    if hasattr(mask, "weights"):
        return float(np.asarray(mask.weights, dtype=np.float64).sum())
    return float(np.asarray(mask, dtype=np.float64).sum())


def _optional_axis_ratio(value: object) -> float | None:
    """trainer.refine.prune_axis_ratio: None disables the shape prune."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("trainer.refine.prune_axis_ratio must be a number or null.")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise ValueError(
            f"trainer.refine.prune_axis_ratio must be finite and above 1, got {value!r}."
        )
    return ratio


class DatumTrainer:
    def __init__(
        self, model: nn.Module, grid: SpatialGrid, data: Any, training_bands: List[str],
        held_out_band: str | None, lambda_step: float, trainer_cfg, device: torch.device,
        precision_policy: str = "portable_fp32", distributed_context: DistributedContext | None = None,
        wandb_logger=None, speed_logger=None, wandb_log_every_n_steps: int = 1,
        epoch_log_every: int | None = None, exclude_image_exposure_ids=None,
        image_variance_columns: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
        prepared_store: Mapping[str, object] | None = None,
    ):
        self.device = device
        # Identity of the prepared datum store (digest and seed), recorded in
        # checkpoints; None for stores built in memory without a cache key.
        self.prepared_store = None if prepared_store is None else dict(prepared_store)
        self._recipe_trainer_cfg = json.loads(json.dumps(trainer_cfg, sort_keys=True, default=str))
        self.distributed_context = distributed_context or DistributedContext()
        self.distributed_enabled = bool(self.distributed_context.enabled)
        self.rank = int(self.distributed_context.rank)
        self.local_rank = int(self.distributed_context.local_rank)
        self.world_size = int(self.distributed_context.world_size)
        self.is_primary_rank = bool(self.distributed_context.is_primary)
        if isinstance(model, DistributedDataParallel):
            self.model = model
        else:
            self.model = model.to(device)
        self.grid = grid
        self.data = data
        self.psf_mixture_digest = getattr(data, "psf_mixture_digest", None)
        self.psf_mixture_digest_version = getattr(
            data, "psf_mixture_digest_version", None
        )
        self.psf_mixture_provenance_convention = (
            PSF_MIXTURE_PROVENANCE_CONVENTION
            if self.psf_mixture_digest_version == 2
            else None
        )
        self.training_bands = training_bands
        self.held_out_band = held_out_band
        self.precision_policy = str(precision_policy or "portable_fp32")
        amp_settings = resolve_amp_settings(self.precision_policy, device)
        self.amp_enabled = bool(amp_settings.enabled)
        self.amp_device_type = amp_settings.device_type
        self.amp_dtype = amp_settings.dtype
        self.amp_dtype_name = (
            str(self.amp_dtype).replace("torch.", "") if self.amp_dtype is not None else None
        )
        self.use_flux_density = True
        # Image exposures withheld from training (novel-view holdout): their
        # datums never enter the training indices or the importance sampler,
        # but they stay in the store and runtime maps so the deterministic
        # render path can still evaluate them.
        self.exclude_image_exposure_ids = frozenset(
            int(e) for e in (exclude_image_exposure_ids or ())
        )
        self.exclude_exposure_labels = sorted(
            str(label) for label in (trainer_cfg.get("exclude_exposure_labels") or [])
        )
        # Spectra withheld from the objective by manifest label: they never
        # enter the spectral training indices or the calibration, but stay in
        # the store and runtime maps, where the evaluation scripts score them.
        raw_spectrum_labels = [
            str(label) for label in (trainer_cfg.get("exclude_spectrum_labels") or [])
        ]
        if len(set(raw_spectrum_labels)) != len(raw_spectrum_labels):
            raise ValueError(
                f"trainer.exclude_spectrum_labels contains duplicates: {raw_spectrum_labels}."
            )
        self.exclude_spectrum_labels = sorted(raw_spectrum_labels)
        self.excluded_spectrum_ids: tuple[int, ...] = ()
        objective_denominator = trainer_cfg.get("objective_denominator")
        self.objective_denominator = None if objective_denominator is None else int(objective_denominator)
        if self.objective_denominator is not None and self.objective_denominator <= 0:
            raise ValueError("trainer.objective_denominator must be a positive datum count.")
        # fpM-flagged pixels of fitted exposures are absent from the fit:
        # neither trainable nor held, so an overlap twin keeps its own status.
        image_pixel_mask = trainer_cfg.get("image_pixel_mask")
        self.image_pixel_mask_absent_by_exposure: Dict[int, np.ndarray] | None = (
            None if image_pixel_mask is None else self._load_image_pixel_mask(image_pixel_mask)
        )
        self.replica_samples_k = int(trainer_cfg.get("replica_samples_k", 1) or 0)
        if self.replica_samples_k < 1:
            raise ValueError("trainer.replica_samples_k must be an integer >= 1.")
        self.history = {"loss": [], "image_loss": [], "spec_loss": [], "n_spec": [], "calibration_prior": []}
        self.wandb_logger = wandb_logger
        self.speed_logger = speed_logger
        try:
            wandb_log_every = int(wandb_log_every_n_steps)
        except (TypeError, ValueError):
            wandb_log_every = 1
        self.wandb_log_every_n_steps = max(1, wandb_log_every)
        self._last_wandb_step_logged: int | None = None
        self.epoch_log_every = (
            epoch_log_every if epoch_log_every is not None and epoch_log_every > 0 else None
        )
        self.apply_spatial_psf = bool(trainer_cfg.get("apply_spatial_psf", True))
        # A 1% covariance change is small relative to the PSF-model and pixel-
        # response approximations, while still catching projection curvature
        # large enough to move or stretch mixture components visibly.
        self.psf_jacobian_rel_tol = float(
            trainer_cfg.get("psf_jacobian_rel_tol", 1e-2)
        )
        self.psf_jacobian_algorithm = PSF_JACOBIAN_ALGORITHM
        self.psf_jacobian_max_tiles_per_axis = PSF_JACOBIAN_MAX_TILES_PER_AXIS
        self.psf_jacobian_cap_policy = PSF_JACOBIAN_CAP_POLICY
        if (
            not math.isfinite(self.psf_jacobian_rel_tol)
            or self.psf_jacobian_rel_tol <= 0.0
        ):
            raise ValueError(
                "trainer.psf_jacobian_rel_tol must be a finite positive value."
            )
        self.psf_model = str(trainer_cfg.get("psf_model", "gaussian")).strip().lower()
        if self.psf_model not in {"gaussian", "dgpsf", "mog", "eigen_mog"}:
            raise ValueError(
                "trainer.psf_model must be one of 'gaussian', 'dgpsf', 'mog', or 'eigen_mog'."
            )
        if self.psf_model in {"dgpsf", "mog"} and (
            not isinstance(self.psf_mixture_digest, str)
            or len(self.psf_mixture_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.psf_mixture_digest.lower())
        ):
            raise ValueError(
                "trainer.psf_model requires a valid manifest psf_mixture_digest; "
                "run scripts/backfill_psf_mixture.py on the manifest."
            )
        if (
            self.psf_model in {"dgpsf", "mog"}
            and self.psf_mixture_digest_version != 2
        ):
            raise ValueError(
                "trainer.psf_model requires PSF mixture digest version 2 so "
                "the operator QA, fallback, deconvolution, and pixel-response "
                "provenance are bound to the checkpoint identity; rebuild the "
                "manifest with scripts/backfill_psf_mixture.py."
            )
        self.allow_psf_model_change = bool(trainer_cfg.get("allow_psf_model_change", False))
        if self.allow_psf_model_change:
            print(
                "WARNING: trainer.allow_psf_model_change=true permits resume across "
                "PSF operator provenance mismatches.",
                flush=True,
            )
        self.eigen_mog_config = None
        self._eigen_mog_renderer = None
        if self.psf_model == "eigen_mog":
            from .eigen_mog_renderer import eigen_mog_identity

            if not self.apply_spatial_psf or not bool(trainer_cfg.get("analytic_images", False)):
                raise ValueError("eigen_mog requires analytic_images=true and apply_spatial_psf=true")
            base = self._base_model()
            if not all(callable(getattr(base, name, None)) for name in (
                "covariance_components", "positions", "_gather_parameter"
            )) or not all(hasattr(base, name) for name in ("log_scales", "theta")):
                raise ValueError("eigen_mog requires SplatField geometry with row-gather support")
            if int(trainer_cfg.get("analytic_image_quadrature_order", 1)) != 1:
                raise ValueError("eigen_mog native stamps include the pixel response; quadrature order must be 1")
            self.eigen_mog_sidecar_path, self.eigen_mog_config = eigen_mog_identity(trainer_cfg)
            self.eigen_mog_pair_chunk = int(trainer_cfg.get("eigen_mog_pair_chunk", PAIR_CHUNK))
            self.eigen_mog_execution = str(trainer_cfg.get("eigen_mog_execution", "reference"))
        self.analytic_images_enabled = bool(trainer_cfg.get("analytic_images", False))
        self.analytic_spectra_enabled = bool(trainer_cfg.get("analytic_spectra", False))
        self.image_variance_outlier_pull = image_variance_outlier_pull(trainer_cfg)
        if self.image_variance_outlier_pull is None and image_variance_columns is not None:
            raise ValueError(
                "Per-column image variance coefficients require trainer.image_variance=poisson_current."
            )
        if self.image_variance_outlier_pull is not None and image_variance_columns is None:
            raise ValueError(
                "trainer.image_variance=poisson_current requires the per-column variance coefficients "
                "(survey_fusion.data.image_variance.image_variance_columns)."
            )
        self.amp_band_subsets = bool(trainer_cfg.get("amp_band_subsets", True))
        self.sparse_splat_gradients = bool(
            trainer_cfg.get("sparse_splat_gradients", False)
        )
        self.analytic_image_quadrature_order = int(
            trainer_cfg.get("analytic_image_quadrature_order", 1)
        )
        if self.analytic_image_quadrature_order < 1:
            raise ValueError("trainer.analytic_image_quadrature_order must be positive.")
        # Dense analytic evaluation is O(pixels x splats): fine at 10^4 splats,
        # impossible at the 10^5-10^6 needed to cover the full COSMOS field.
        # The patch rasterizer is O(splats x patch_area), so it renders the whole
        # exposure and the batch is gathered from it.
        self.patch_raster_enabled = bool(trainer_cfg.get("patch_raster", False))
        self.patch_raster_px = int(trainer_cfg.get("patch_raster_px", 25))
        # Break-even batch size for rasterizing an exposure; defaults to
        # patch^2 from the cost comparison above.
        self.patch_raster_min_pixels = int(
            trainer_cfg.get("patch_raster_min_pixels", 0)
        ) or self.patch_raster_px ** 2
        self._analytic_image_cache: Dict[int, dict] = {}
        self._pixel_norm_x: torch.Tensor | None = None
        self._pixel_norm_y: torch.Tensor | None = None
        self._pixel_norm_base: np.ndarray | None = None
        self._pixel_norm_filled: np.ndarray | None = None
        self._spectrum_cache_by_exposure: Dict[int, SplatSpectrumCache] = {}
        self.splat_flux_l1 = float(trainer_cfg.get("splat_flux_l1", 0.0) or 0.0)
        # Pair-support truncation width for the point-splat operators, in
        # sigma of the PSF-convolved splat. An operator knob, not a refine
        # knob: it shapes every image render, with or without refinement.
        # 8.0 sits on the dose-curve plateau (discarded pair weight
        # exp(-32) ~ 1.3e-14, below fp32 resolution); the historical 3.5
        # severed gradients between shrinking splats and faint sources.
        pair_n_sigma = (trainer_cfg.get("refine") or {}).get(
            "pair_n_sigma", DEFAULT_PAIR_N_SIGMA
        )
        self.pair_n_sigma = float(pair_n_sigma)
        if not math.isfinite(self.pair_n_sigma) or self.pair_n_sigma <= 0.0:
            raise ValueError(
                "trainer.refine.pair_n_sigma must be a finite positive "
                f"support width in sigma, got {pair_n_sigma!r}."
            )
        self.gauge_projection_enabled = bool(
            (trainer_cfg.get("refine") or {}).get("gauge_projection", False)
        )
        self.checkpoint_rng_state_enabled = bool(
            trainer_cfg.get("checkpoint_rng_state", False)
        )
        self.checkpoint_gauge_sidecar_enabled = bool(
            trainer_cfg.get("checkpoint_gauge_sidecar", False)
        )
        if self.checkpoint_gauge_sidecar_enabled:
            sidecar_model = self.model.module if hasattr(self.model, "module") else self.model
            if not callable(
                getattr(getattr(sidecar_model, "sed", None), "templates", None)
            ):
                raise ValueError(
                    "trainer.checkpoint_gauge_sidecar requires a splat model with "
                    "an SED dictionary; a bandless model has no gauge to "
                    "canonicalize."
                )
        # Absolute sampling position: epochs finished across every invocation
        # of this run, plus batches drawn into the current one. Restored on
        # resume so the draw stream continues instead of replaying its prefix.
        self._epochs_completed = 0
        self._steps_into_epoch = 0
        self._resume_epoch_offset = 0
        self._resume_epoch_batch_offset = 0
        self._checkpoint_loader_base_seed = int(torch.initial_seed())
        birth_cfg = (trainer_cfg.get("refine") or {}).get("birth") or {}
        self.birth_enabled = bool(birth_cfg.get("enabled", False))
        self.birth_z_min = float(birth_cfg.get("z_min", 8.0))
        self.birth_n_eff_min = float(birth_cfg.get("n_eff_min", 25.0))
        self.birth_kappa = float(birth_cfg.get("kappa", 0.2))
        self.birth_cap_per_event = int(birth_cfg.get("cap_per_event", 64))
        self.birth_cell_scale = float(birth_cfg.get("cell_scale", 2.0))
        if self.birth_enabled and not self.analytic_images_enabled:
            raise ValueError(
                "refine.birth requires trainer.analytic_images=true: the "
                "residual accumulator reads the analytic image forward."
            )
        self._birth_accumulator_state = None
        self._birth_capture = False
        self._pending_birth = None
        self.grad_clip = trainer_cfg.get("grad_clip", 1.0)
        self.epochs = int(trainer_cfg.get("epochs", 1))
        self.max_steps = int(trainer_cfg.get("max_steps", 0) or 0)
        # Each fitted spectral bin's NLL term counts this many times an image
        # pixel's; the objective stays divided by the unweighted datum count.
        self.spectral_bin_weight = float(trainer_cfg.get("spectral_bin_weight", 5.0))
        if not math.isfinite(self.spectral_bin_weight) or self.spectral_bin_weight <= 0.0:
            raise ValueError("trainer.spectral_bin_weight must be finite and positive.")
        self.adam_eps = float(trainer_cfg.get("adam_eps", 1e-8))
        if not math.isfinite(self.adam_eps) or self.adam_eps <= 0.0:
            raise ValueError("trainer.adam_eps must be finite and positive.")
        self.eval_interval = int(trainer_cfg.get("eval_interval", 100))
        self.image_batch_size = int(trainer_cfg.get("pixel_batch_size", 0) or 0)
        self.spec_batch_size = int(trainer_cfg.get("spec_batch_size", 0) or 0)
        random_holdout_cfg = trainer_cfg.get("random_holdout", {})
        if not isinstance(random_holdout_cfg, dict):
            random_holdout_cfg = {}
        self.spectral_holdout_fraction = float(
            random_holdout_cfg.get("fraction", 0.0) or 0.0
        )
        self.spectral_holdout_seed = int(
            random_holdout_cfg.get("seed", 0) or 0
        )
        self.fixed_holdout_max_image_per_band = int(
            trainer_cfg.get("fixed_holdout_max_image_per_band", 20_000) or 0
        )
        self.fixed_holdout_max_spec = int(
            trainer_cfg.get("fixed_holdout_max_spec", 10_000) or 0
        )
        if not 0.0 <= self.spectral_holdout_fraction <= 1.0:
            raise ValueError("random_holdout.fraction must be between 0 and 1.")
        if (
            self.spectral_holdout_fraction > 0.0
            and self.spectral_holdout_seed == 0
        ):
            raise ValueError(
                "random_holdout.fraction > 0 requires an explicit "
                "random_holdout.seed."
            )
        self.dataloader_shuffle = bool(trainer_cfg.get("dataloader_shuffle", True))
        self.dataloader_num_workers = int(trainer_cfg.get("dataloader_num_workers", 0) or 0)
        default_pin_memory = bool(device.type == "cuda")
        self.dataloader_pin_memory = bool(trainer_cfg.get("dataloader_pin_memory", default_pin_memory))
        self.dataloader_drop_last = bool(trainer_cfg.get("dataloader_drop_last", False))
        raw_band_loss_every = trainer_cfg.get("band_loss_log_every_n_steps", 1)
        if raw_band_loss_every is None:
            self.band_loss_log_every_n_steps = 0
        else:
            self.band_loss_log_every_n_steps = max(0, int(raw_band_loss_every))
        self.image_importance_sampling_cfg = self._parse_image_importance_sampling_cfg(
            trainer_cfg.get("image_importance_sampling", {})
        )
        calibration_cfg = trainer_cfg.get("calibration", {})
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        spectral_cfg = calibration_cfg.get("spectral_nuisance", {})
        if not isinstance(spectral_cfg, dict):
            spectral_cfg = {}
        self.spectral_calibration_cfg = SpectralCalibrationConfig.from_dict(spectral_cfg)
        self.spectral_calibrator: SpectralNuisanceCalibrator | None = None
        self.params: List[nn.Parameter] = list(self.model.parameters())
        beta1 = float(trainer_cfg.get("beta1", 0.9))
        beta2 = float(trainer_cfg.get("beta2", 0.999))
        self._base_learning_rate = float(trainer_cfg.get("learning_rate", 1e-3))
        self._beta1 = beta1
        self._beta2 = beta2
        self.lr_scheduler_cfg = self._parse_lr_scheduler_cfg(
            trainer_cfg.get("lr_scheduler", {})
        )
        base_model_for_opt = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.splat_optimizer: RowSparseAdam | None = None
        if self.sparse_splat_gradients:
            if is_distributed_active(self.distributed_context):
                raise ValueError(
                    "distributed splat training requires dense gradients."
                )
            required_methods = (
                "set_sparse_gradients",
                "sparse_param_groups_gsplat",
                "shared_param_groups_gsplat",
            )
            missing = [
                name
                for name in required_methods
                if not callable(getattr(base_model_for_opt, name, None))
            ]
            if missing:
                raise ValueError(
                    "sparse_splat_gradients requires a splat model implementing "
                    + ", ".join(missing)
                )
            if not self.analytic_images_enabled or not self.analytic_spectra_enabled:
                raise ValueError(
                    "sparse_splat_gradients requires deterministic analytic image "
                    "and spectrum operators."
                )
            if self.splat_flux_l1 != 0.0:
                raise ValueError(
                    "sparse_splat_gradients requires splat_flux_l1=0 because a "
                    "whole-field flux prior produces dense gradients."
                )
            if self.lr_scheduler_cfg["enabled"]:
                raise ValueError(
                    "sparse_splat_gradients does not support lr_scheduler yet."
                )
            base_model_for_opt.set_sparse_gradients(True)
        if callable(getattr(base_model_for_opt, "param_groups_gsplat", None)):
            position_lr_multiplier = float(trainer_cfg.get("position_lr_multiplier", 1.0))
            if not math.isfinite(position_lr_multiplier) or position_lr_multiplier <= 0.0:
                raise ValueError("trainer.position_lr_multiplier must be finite and positive.")
            widths = [
                int(
                    getattr(getattr(exp, "wcs", None), "width", 0)
                    or getattr(exp, "image_width", 0)
                    or 0
                )
                for exp in getattr(self.data, "exposures", {}).values()
            ]
            field_width = max(widths) if widths else 1024
            self._field_width = int(field_width)
            self._position_lr_horizon = int(
                trainer_cfg.get("position_lr_horizon", 30_000)
            )
            if self._position_lr_horizon <= 0:
                raise ValueError("trainer.position_lr_horizon must be positive.")
            if 0 < self.max_steps < self._position_lr_horizon:
                warnings.warn(
                    (
                        f"trainer.max_steps={self.max_steps} is shorter than "
                        "trainer.position_lr_horizon="
                        f"{self._position_lr_horizon}; the position-LR schedule "
                        "will not complete."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._position_lr_gamma = 0.01 ** (1.0 / self._position_lr_horizon)
            if self.sparse_splat_gradients:
                self.optimizer = torch.optim.Adam(
                    base_model_for_opt.shared_param_groups_gsplat(),
                    betas=(beta1, beta2), eps=self.adam_eps,
                )
                self.splat_optimizer = RowSparseAdam(
                    base_model_for_opt.sparse_param_groups_gsplat(),
                    betas=(beta1, beta2), eps=self.adam_eps,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    base_model_for_opt.param_groups_gsplat(),
                    betas=(beta1, beta2), eps=self.adam_eps,
                )
            self._position_group = next(
                (
                    g
                    for opt in (self.splat_optimizer, self.optimizer)
                    if opt is not None
                    for g in opt.param_groups
                    if any(p is base_model_for_opt.mu for p in g["params"])
                ),
                None,
            )
            if self._position_group is not None:
                self._position_group["lr"] *= position_lr_multiplier
            self._position_lr_init = (
                float(self._position_group["lr"]) if self._position_group else 0.0
            )
            self.refine_strategy = self._build_refine_strategy(
                base_model_for_opt, trainer_cfg
            )
        else:
            if self.sparse_splat_gradients:
                raise ValueError(
                    "sparse_splat_gradients is only available for splat models."
                )
            self.optimizer = torch.optim.Adam(
                self.params, lr=self._base_learning_rate, betas=(beta1, beta2), eps=self.adam_eps,
            )
        if not hasattr(self, "refine_strategy"):
            self.refine_strategy = None
            self._position_group = None
            self._position_lr_init = 0.0
            self._position_lr_gamma = 1.0
        self.lr_scheduler: Any | None = None
        self.kernel_samples_k = int(trainer_cfg.get("kernel_samples_k", 0) or 0)
        self.image_sampler: DistributedSampler | None = None
        self.spec_sampler: DistributedSampler | None = None
        self._checkpoint_epoch_samplers: list[_CheckpointEpochSampler] = []
        self.image_train_indices_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_train_local_positions_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_importance_sampling_enabled = False
        self.image_importance_exposure_ids: list[int] = []
        self.image_importance_exposure_probs: torch.Tensor | None = None
        self.image_importance_exposure_cdf: torch.Tensor | None = None
        self.image_importance_local_probs_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_importance_local_cdf_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_importance_correction_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_importance_generator: torch.Generator | None = None
        self._pending_image_importance_weights: torch.Tensor | None = None
        self._restored_importance_pending_batch: tuple[torch.Tensor, torch.Tensor] | None = None

        self.exposures = {int(exp_id): exposure for exp_id, exposure in self.data.exposures.items()}
        validate_batch_exposure_links(self.data.batch, self.exposures)
        validate_batch_datum_kinds(self.data.batch)
        self.n_image_datums = int(getattr(self.data, "n_image", 0) or 0)
        self.image_band_by_exposure = {
            int(exp_id): str(band)
            for exp_id, band in getattr(self.data, "image_band_by_exposure", {}).items()
        }
        self.image_source_by_exposure = {
            int(exp_id): np.asarray(image, dtype=np.float32)
            for exp_id, image in getattr(self.data, "image_source_by_exposure", {}).items()
        }
        self.image_additive_by_exposure = {
            int(exp_id): np.asarray(image, dtype=np.float32)
            for exp_id, image in getattr(self.data, "image_additive_by_exposure", {}).items()
        }
        self.image_total_by_exposure = {
            int(exp_id): np.asarray(image, dtype=np.float32)
            for exp_id, image in getattr(self.data, "image_total_by_exposure", {}).items()
        }
        self.spec_source_by_exposure = {
            int(exp_id): np.asarray(spec, dtype=np.float32)
            for exp_id, spec in getattr(self.data, "spec_source_by_exposure", {}).items()
        }
        self.spec_additive_by_exposure = {
            int(exp_id): np.asarray(spec, dtype=np.float32)
            for exp_id, spec in getattr(self.data, "spec_additive_by_exposure", {}).items()
        }
        self.spec_total_by_exposure = {
            int(exp_id): np.asarray(spec, dtype=np.float32)
            for exp_id, spec in getattr(self.data, "spec_total_by_exposure", {}).items()
        }
        self.partial_holdout_enabled = bool(getattr(self.data, "partial_holdout_enabled", False))
        raw_excluded = getattr(self.data, "image_excluded_local_indices_by_exposure", {})
        raw_random_holdout = getattr(
            self.data, "image_random_holdout_indices_by_exposure", {}
        )
        self.image_excluded_local_indices_by_exposure: Dict[int, torch.Tensor] = {}
        self.image_random_holdout_indices_by_exposure: Dict[int, torch.Tensor] = {}
        if isinstance(raw_excluded, dict):
            for exp_id, local_indices in raw_excluded.items():
                local = torch.as_tensor(local_indices, dtype=torch.long)
                if local.ndim != 1:
                    raise ValueError("image_excluded_local_indices_by_exposure values must be 1D arrays.")
                if local.numel() > 0:
                    local = torch.unique(local, sorted=True)
                self.image_excluded_local_indices_by_exposure[int(exp_id)] = local
        if isinstance(raw_random_holdout, dict):
            for exp_id, local_indices in raw_random_holdout.items():
                local = torch.as_tensor(local_indices, dtype=torch.long)
                if local.ndim != 1:
                    raise ValueError(
                        "image_random_holdout_indices_by_exposure values must "
                        "be 1D arrays."
                    )
                if local.numel() > 0:
                    local = torch.unique(local, sorted=True)
                self.image_random_holdout_indices_by_exposure[int(exp_id)] = local
        image_kind = int(DatumKind.IMAGE_PIXEL)
        spec_kind = int(DatumKind.SPECTRAL_BIN)
        self.global_indices_by_exposure_kind = self._build_global_index_cache()
        all_exposure_ids = {
            int(exposure_id)
            for exposure_id, _ in self.global_indices_by_exposure_kind
        }
        all_exposure_ids.update(int(exp_id) for exp_id in self.image_band_by_exposure)
        self.image_exposure_ids = sorted(
            int(exposure_id)
            for exposure_id, kind in self.global_indices_by_exposure_kind
            if int(kind) == image_kind
        )
        self.spec_exposure_ids = sorted(
            int(exposure_id)
            for exposure_id, kind in self.global_indices_by_exposure_kind
            if int(kind) == spec_kind
        )
        missing_bands = sorted(
            exp_id
            for exp_id in self.image_exposure_ids
            if exp_id not in self.image_band_by_exposure
        )
        if missing_bands:
            raise ValueError(f"Missing image band mappings for image exposures: {missing_bands}")
        unknown_band_exposures = sorted(
            exp_id
            for exp_id in self.image_band_by_exposure
            if exp_id not in all_exposure_ids
        )
        if unknown_band_exposures:
            raise ValueError(
                "image_band_by_exposure contains exposures not referenced by batch datums: "
                f"{unknown_band_exposures}"
            )
        self.primary_exposure_id = min(self.exposures.keys())
        if self.image_exposure_ids:
            self.primary_image_exposure_id = min(self.image_exposure_ids)
        else:
            self.primary_image_exposure_id = self.primary_exposure_id
        self.band_slices: List[Tuple[str, int, int]] = []

        self.loglam_min, self.loglam_max = self._compute_global_loglam_bounds()
        sed = getattr(self._base_model(), "sed", None)
        set_loglam_scale = getattr(sed, "set_loglam_scale", None)
        if callable(set_loglam_scale):
            set_loglam_scale(0.5 * (self.loglam_max - self.loglam_min))
        self.renderer_bandpasses = self._collect_renderer_bandpasses()
        self.runtime_by_exposure = self._build_exposure_runtime(
            lambda_step=lambda_step, psf_truncate=float(trainer_cfg.get("psf_truncate", 4.0)),
            include_diagnostics=False,
        )
        for runtime in self.runtime_by_exposure.values():
            validate_spectral_source_psf(runtime.exposure, apply_spatial_psf=self.apply_spatial_psf)
        primary_runtime = self.runtime_by_exposure[self.primary_image_exposure_id]
        self.renderer = primary_runtime.renderer
        self.diagnostic_renderer = SpectralRenderer(self.renderer_bandpasses, lambda_step, self.device)
        self.psf_kernel = primary_runtime.psf_kernel
        self.aperture_support = primary_runtime.aperture_support
        if image_variance_columns is not None:
            self._init_image_variance_columns(image_variance_columns)

        if self.image_pixel_mask_absent_by_exposure is not None:
            self._check_image_pixel_mask_store()
        field_overlap_context = None
        field_overlap_table = trainer_cfg.get("field_overlap_table")
        if field_overlap_table is not None:
            field_overlap_context = self._apply_field_overlap_exclusions(
                field_overlap_table
            )
        training_indices_started = time.perf_counter()
        self._init_training_indices()
        if field_overlap_context is not None:
            overlap_pairs_for_check, original_overlap_exclusions = field_overlap_context
            self._check_field_overlap_postcondition(
                overlap_pairs_for_check, original_overlap_exclusions
            )
        training_indices_sec = time.perf_counter() - training_indices_started
        importance_started = time.perf_counter()
        self._init_image_importance_sampling()
        importance_sampling_sec = time.perf_counter() - importance_started
        spectrum_cache_started = time.perf_counter()
        self._prebuild_spectrum_caches()
        spectrum_cache_sec = time.perf_counter() - spectrum_cache_started
        if self.is_primary_rank:
            print(
                "[survey_fusion] trainer initialization: "
                f"training_indices={training_indices_sec:.3f}s, "
                f"importance_sampling={importance_sampling_sec:.3f}s, "
                f"spectrum_caches={spectrum_cache_sec:.3f}s",
                flush=True,
            )
            if self.wandb_logger:
                modality_summary = self.scientific_modality_summary()
                self.wandb_logger.log(
                    {
                        "init/training_indices_sec": training_indices_sec,
                        "init/importance_sampling_sec": importance_sampling_sec,
                        "objective/image_population_count": modality_summary[
                            "image_population_count"
                        ],
                        "objective/spectrum_population_count": modality_summary[
                            "spectrum_population_count"
                        ],
                        "objective/image_weight": modality_summary["image_weight"],
                        "objective/spectrum_weight": modality_summary[
                            "spectrum_weight"
                        ],
                    },
                    step=0,
                )
        self._init_dataloaders()
        if self.kernel_samples_k <= 0:
            raise ValueError("trainer.kernel_samples_k must be a positive integer.")
        if getattr(self.data, "kernel_factory", None) is None:
            raise ValueError("MC training requires data.kernel_factory.")
        self._init_spectral_calibrator()
        if self.spectral_calibrator is not None:
            calibration_params = list(self.spectral_calibrator.parameters())
            if calibration_params:
                self.params.extend(calibration_params)
                self.optimizer.add_param_group(
                    {
                        "params": calibration_params,
                        "lr": self._base_learning_rate * float(self.spectral_calibration_cfg.lr_multiplier),
                        "betas": (self._beta1, self._beta2),
                    }
                )
        self._refresh_clip_params()
        self._init_lr_scheduler()
        self._speed_profile_enabled = bool(self.speed_logger)
        self._speed_profile_wall: Dict[str, float] = {}
        self._speed_profile_cuda: Dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self._fiber_side_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and fiber_side_stream_enabled()
            else None
        )
        self._fiber_side_stream_used = False

        base_model = self._base_model()
        if type(base_model).forward is nn.Module.forward:
            raise ValueError("Model must implement standard forward(x, y, log_lam).")
        missing_norm_support = [
            int(exp_id)
            for exp_id in sorted(set(self.image_exposure_ids + self.spec_exposure_ids))
            if not callable(getattr(self.exposures[int(exp_id)].wcs, "omega_xyz_to_norm_torch", None))
        ]
        if missing_norm_support:
            raise ValueError(
                "Every active exposure WCS must implement omega_xyz_to_norm_torch; "
                f"missing support for exposures {missing_norm_support}."
            )

    def _reset_speed_profile(self) -> None:
        self._speed_profile_wall = {}
        self._speed_profile_cuda = {}

    def _record_speed_wall(self, key: str, duration_sec: float) -> None:
        if not self._speed_profile_enabled:
            return
        self._speed_profile_wall[key] = self._speed_profile_wall.get(key, 0.0) + float(duration_sec)

    def _record_speed_cuda_events(
        self, key: str, start_event: torch.cuda.Event, end_event: torch.cuda.Event,
    ) -> None:
        if not self._speed_profile_enabled:
            return
        self._speed_profile_cuda.setdefault(key, []).append((start_event, end_event))

    def _finalize_speed_profile(self) -> Dict[str, float]:
        if not self._speed_profile_enabled:
            return {}
        timings = dict(self._speed_profile_wall)
        if self.device.type == "cuda":
            for key, event_pairs in self._speed_profile_cuda.items():
                total_sec = 0.0
                for start_event, end_event in event_pairs:
                    total_sec += float(start_event.elapsed_time(end_event)) / 1000.0
                timings[key] = timings.get(key, 0.0) + total_sec
        self._reset_speed_profile()
        return timings

    def _base_model(self) -> nn.Module:
        model = self.model
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def _gather_batch_rows(
        self, values: torch.Tensor, indices: torch.Tensor, *, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        idx_cpu = indices.detach().to(dtype=torch.long, device="cpu")
        gathered = torch.index_select(values, 0, idx_cpu)
        if dtype is not None:
            gathered = gathered.to(dtype=dtype)
        target_device = self.device if device is None else torch.device(device)
        if gathered.device != target_device:
            gathered = copy_tensor_to_device(
                gathered,
                device=target_device,
                non_blocking=bool(values.device.type == "cpu" and target_device.type == "cuda"),
            )
        return gathered

    def _gather_named_rows(
        self, field: str, indices: torch.Tensor, *, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        gather_rows = getattr(self.data, "gather_rows", None)
        if callable(gather_rows):
            gathered = gather_rows(str(field), indices)
            if dtype is not None:
                gathered = gathered.to(dtype=dtype)
            target_device = self.device if device is None else torch.device(device)
            if gathered.device != target_device:
                gathered = copy_tensor_to_device(
                    gathered,
                    device=target_device,
                    non_blocking=bool(gathered.device.type == "cpu" and target_device.type == "cuda"),
                )
            return gathered
        values = getattr(self.data.batch, str(field))
        if values is None:
            raise KeyError(f"DatumBatch field {field!r} is not present.")
        return self._gather_batch_rows(values, indices, device=device, dtype=dtype)

    @staticmethod
    def _assemble_coalesced_rows(
        gathered: List[torch.Tensor], parameters: List[nn.Parameter]
    ) -> None:
        """Rebuild each parameter's gradient over the union of `gathered`.

        `gathered` are the index tensors the parameters were gathered with, so
        their union is exactly the row set the gradients hold. One membership
        mask and one `nonzero` serve every parameter in the list.
        """
        n_splats = int(parameters[0].shape[0])
        device = parameters[0].device
        if any(int(p.shape[0]) != n_splats for p in parameters):
            # Sharing an index tensor means sharing a row space; a mismatch
            # would put rows past the end of a shorter parameter, and the
            # assembled COO carries no invariant check.
            raise RuntimeError(
                "Parameters gathered with the same rows must share a row count."
            )
        if device.type=='cuda' and coalesce_direct_enabled():
            from ..pgm.coalesce_pair_kernels import coalesce_pair
            if coalesce_pair(gathered,parameters):
                return
        mask = torch.zeros(n_splats, dtype=torch.bool, device=device)
        for indices in gathered:
            if device_index_fill_enabled() and mask.is_cuda:
                mask.index_fill_(0, indices, True)
            else:
                mask[indices] = True
        # The one synchronise: nonzero sizes its output on the host. Its rows
        # come back ascending and duplicate-free, which is what makes the
        # assembled tensor coalesced by construction.
        rows = mask.nonzero(as_tuple=True)[0]
        n_rows = int(rows.numel())
        inverse = torch.empty(n_splats, dtype=torch.long, device=device)
        inverse[rows] = torch.arange(n_rows, device=device)
        for parameter in parameters:
            gradient = parameter.grad
            values = gradient._values()
            dense = values.new_zeros((n_rows,) + values.shape[1:])
            dense.index_add_(0, inverse[gradient._indices()[0]], values)
            parameter.grad = torch.sparse_coo_tensor(
                rows.unsqueeze(0),
                dense,
                gradient.shape,
                device=device,
                dtype=values.dtype,
                check_invariants=False,
                is_coalesced=True,
            )

    def _coalesce_sparse_gradients(self) -> None:
        """Coalesce the per-splat sparse gradients in place, without sorting.

        Under sparse_splat_gradients each per-splat parameter is gathered once
        by the image path and once by the fiber path, over an ascending
        duplicate-free row set each time, so the accumulated gradient is a
        concatenation of coalesced COO blocks and a row carries one value per
        block it appears in. `coalesce()` finds those groups by sorting the
        whole index array, and thrust's sort synchronises about five times per
        call: thirty host stalls per step over six parameters.

        The rows are recoverable without a sort. Mark the gathered indices in a
        membership mask over the census, read the marked rows off in ascending
        order with one `nonzero`, and add the values into the compacted rows.
        With at most two sorted-unique gather blocks, each row receives at
        most two values. Their sum is independent of order. Three or more
        contributions can reassociate differently, so those parameters keep
        ordinary coalescence, as do gathers without a sorted-unique guarantee.

        Parameters gathered with the same index tensors share one mask, so a
        step pays a single synchronise rather than one per parameter. The
        gather log is what makes that safe: the row set is the union of the
        logged index sets, and a parameter whose logged gathers do not account
        for every entry of its gradient is coalesced the ordinary way.
        """
        model = self._base_model()
        take_log = getattr(model, "take_sparse_gather_log", None)
        log = take_log() if callable(take_log) else None
        groups: Dict[tuple[int, ...], tuple[List[torch.Tensor], List[nn.Parameter]]] = {}
        for parameter in self.params:
            gradient = parameter.grad
            if gradient is None or not gradient.is_sparse:
                continue
            if gradient.is_coalesced():
                continue
            gathered = None if log is None else log.get(id(parameter))
            if (
                gathered is None
                or not 1 <= len(gathered) <= 2
                or sum(int(indices.numel()) for indices in gathered)
                != int(gradient._nnz())
            ):
                parameter.grad = gradient.coalesce()
                continue
            groups.setdefault(
                tuple(id(indices) for indices in gathered), (gathered, [])
            )[1].append(parameter)
        for gathered, parameters in groups.values():
            self._assemble_coalesced_rows(gathered, parameters)

    def _gradient_norm(self) -> torch.Tensor:
        """Global norm over the clipped parameters, left on the device.

        Returning the reduced scalar rather than its host value keeps the
        synchronise out of the step: the caller scales by it and the value is
        only read once the step's other scalars are read anyway.
        """
        squared_norms = []
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                # Coalesce in place so clip, refine accumulation and
                # SparseAdam reuse the sorted result instead of each
                # re-sorting the same gradient. A no-op on the gradients
                # _coalesce_sparse_gradients already assembled.
                grad = grad.coalesce()
                param.grad = grad
                values = grad.values()
            else:
                values = grad
            if id(param) not in self._clip_param_ids:
                continue
            # Promote the reduced scalar to float64 before squaring: two
            # individually finite float32 norms can overflow float32 when
            # their squares are summed, which would zero every gradient via
            # a scale of max_norm/inf.
            squared_norms.append(
                torch.linalg.vector_norm(values.detach())
                .to(torch.float64)
                .square()
            )
        if not squared_norms:
            return torch.zeros((), dtype=torch.float64)
        return torch.stack(squared_norms).sum().sqrt()

    def _sync_model_gradients(self) -> None:
        if not is_distributed_active(self.distributed_context):
            return
        base_model = self._base_model()
        if not callable(getattr(base_model, "param_groups_gsplat", None)):
            return
        for parameter in base_model.parameters():
            gradient = parameter.grad
            if gradient is not None and gradient.layout != torch.strided:
                raise ValueError(
                    "distributed splat training requires dense gradients."
                )
            if gradient is None:
                gradient = torch.zeros_like(parameter)
                parameter.grad = gradient
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(float(self.world_size))

    def _clip_gradient_norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip the shared/calibration gradients to grad_clip; returns
        (preclip, postclip) norms over that clipped set as device scalars.

        Per-splat parameters are excluded: their rows are quasi-independent
        local objects whose updates SparseAdam/Adam already bound per
        coordinate, and a global norm over a growing census scales with the
        census size, so including them dilutes every row's update as the
        field grows and pushes faint-row gradients below Adam's eps. The
        postclip norm is scale * preclip by linearity, so it does not need a
        second full sweep over every gradient.

        Taking the `scale < 1` decision on the host would synchronise here,
        right after backward and right before the optimizer step, draining the
        queue the step is about to refill. Clamping on the device and
        multiplying unconditionally decides the same thing: outside the
        clipping regime the scale is exactly 1.0, and a float multiply by 1.0
        returns its operand bitwise.
        """
        total_norm = self._gradient_norm()
        max_norm = float(self.grad_clip)
        scale = (max_norm / (total_norm + 1.0e-6)).clamp(max=1.0)
        for parameter in self.params:
            if id(parameter) not in self._clip_param_ids:
                continue
            gradient = parameter.grad
            if gradient is None:
                continue
            values = gradient._values() if gradient.is_sparse else gradient
            # float64 scale narrowed to the gradient's dtype, which is what
            # passing the host float to mul_ does inside the kernel.
            values.mul_(scale.to(values.dtype))
        return total_norm, total_norm * scale

    def _init_lr_scheduler(self) -> None:
        if not self.lr_scheduler_cfg["enabled"]:
            return
        scheduler_type = self.lr_scheduler_cfg["type"]
        if scheduler_type == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.lr_scheduler_cfg["mode"], factor=self.lr_scheduler_cfg["factor"],
                patience=self.lr_scheduler_cfg["patience"], threshold=self.lr_scheduler_cfg["threshold"],
                threshold_mode=self.lr_scheduler_cfg["threshold_mode"],
                cooldown=self.lr_scheduler_cfg["cooldown"], min_lr=self.lr_scheduler_cfg["min_lr"],
                eps=self.lr_scheduler_cfg["eps"],
            )
            return
        if scheduler_type == "OneCycleLR":
            steps_per_epoch = self._steps_per_epoch()
            max_possible_steps = self.epochs * steps_per_epoch
            if self.max_steps > 0:
                total_steps = (min(self.max_steps, max_possible_steps) if max_possible_steps > 0 else 0)
            else:
                total_steps = max_possible_steps
            if total_steps <= 0:
                raise ValueError(
                    "trainer.lr_scheduler OneCycleLR requires at least one planned training step."
                )
            base_lrs = [
                float(group.get("lr", self._base_learning_rate))
                for group in self.optimizer.param_groups
            ]
            max_lr_cfg = self.lr_scheduler_cfg["max_lr"]
            if max_lr_cfg is None:
                max_lr: float | list[float] = base_lrs
            elif isinstance(max_lr_cfg, list):
                if len(max_lr_cfg) != len(base_lrs):
                    raise ValueError(
                        "trainer.lr_scheduler.max_lr list length must match optimizer param groups."
                    )
                max_lr = [float(value) for value in max_lr_cfg]
            else:
                base_lr = max(float(self._base_learning_rate), 1e-30)
                max_lr = [float(max_lr_cfg) * (group_lr / base_lr) for group_lr in base_lrs]
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lr, total_steps=int(total_steps),
                pct_start=self.lr_scheduler_cfg["pct_start"],
                anneal_strategy=self.lr_scheduler_cfg["anneal_strategy"],
                cycle_momentum=self.lr_scheduler_cfg["cycle_momentum"],
                div_factor=self.lr_scheduler_cfg["div_factor"],
                final_div_factor=self.lr_scheduler_cfg["final_div_factor"],
                three_phase=self.lr_scheduler_cfg["three_phase"],
            )
            return
        raise ValueError(f"Unsupported lr scheduler type: {scheduler_type}")

    def _build_point_index(self, model, xs, ys, sig, point_group=None):
        """Single production entry to the point-splat pair index; the resolved
        support width must reach the operator through this call."""
        if isinstance(sig, PointPSFMixture):
            return build_mixture_packed_point_index(
                model, xs, ys, sig, n_sigma=self.pair_n_sigma,
                point_group=point_group,
            )
        return build_packed_point_index(model, xs, ys, sig, n_sigma=self.pair_n_sigma)

    def _build_refine_strategy(self, base_model, trainer_cfg: dict):
        refine_cfg = trainer_cfg.get("refine", {}) or {}
        if not bool(refine_cfg.get("enabled", False)):
            return None
        from survey_fusion.models.splat_strategy import (
            SplatRefineConfig,
            SplatRefineStrategy,
            psf_based_grow_scale,
        )

        sample_idx = torch.arange(
            0,
            int(self.data.batch.y.shape[0]),
            max(1, int(self.data.batch.y.shape[0]) // 100_000),
            dtype=torch.long,
        )
        sigma = self._gather_named_rows(
            "sigma", sample_idx, device=torch.device("cpu"), dtype=torch.float32,
        )
        lifecycle_statistics = str(refine_cfg.get("lifecycle_statistics", "fitted_images"))
        self.lifecycle_statistics = lifecycle_statistics
        if lifecycle_statistics not in ("legacy", "fitted_images"):
            raise ValueError(
                "trainer.refine.lifecycle_statistics must be 'legacy' or 'fitted_images', "
                f"got {lifecycle_statistics!r}."
            )
        fitted_image_ids = {
            int(exp_id) for exp_id in getattr(self.data, "image_band_by_exposure", {})
        } - set(self.exclude_image_exposure_ids)
        if self.image_pixel_mask_absent_by_exposure is not None:
            present = ~self._image_pixel_absent_rows(sample_idx, fitted_image_ids)
            sample_idx, sigma = sample_idx[present], sigma[present]
        if lifecycle_statistics == "fitted_images":
            # The thresholds describe the imaging: spectra carry guider seeing
            # and per-fibre sigma, and no held datum may steer the fit.
            sigma = sigma[self._fitted_image_rows(sample_idx, fitted_image_ids)]
        elif self.exclude_image_exposure_ids:
            # Held-out exposures must not steer the refine thresholds: their
            # sigma rows carry observed-count information through the Poisson
            # noise model.
            row_exposure = self._gather_named_rows(
                "exposure_id", sample_idx, device=torch.device("cpu"),
                dtype=torch.long,
            )
            excluded_rows = torch.isin(
                row_exposure,
                torch.tensor(sorted(self.exclude_image_exposure_ids), dtype=torch.long),
            )
            if bool((~excluded_rows).any()):
                sigma = sigma[~excluded_rows]
        # The lifecycle reduces a broad mixture operator to legacy scalar FWHM
        # for census growth.
        lifecycle_exposures = {
            int(exp_id): exp for exp_id, exp in self.data.exposures.items()
            if int(exp_id) not in self.exclude_image_exposure_ids
            and (lifecycle_statistics == "legacy" or int(exp_id) in fitted_image_ids)
        }
        psf_values = [
            float(getattr(exp, "psf_fwhm_arcsec", 0.0) or 0.0)
            for exp in lifecycle_exposures.values()
            if getattr(exp, "psf_fwhm_arcsec", None)
        ]
        median_psf = float(np.median(psf_values)) if psf_values else 1.4
        pixel_scales = [
            float(getattr(getattr(exp, "wcs", None), "pixel_scale_arcsec", 0.0) or 0.0)
            for exp in lifecycle_exposures.values()
        ]
        positive_scales = [p for p in pixel_scales if p > 0]
        model_pixel_scale = float(np.median(positive_scales)) if positive_scales else 0.396
        prune_band_snr = bool(refine_cfg.get("prune_band_snr", False))
        band_snr_sigma = refine_cfg.get("prune_band_snr_sigma", None)
        config = SplatRefineConfig(
            grow_grad2d=float(refine_cfg.get("grow_grad2d", 0.0)),
            grow_target_frac=float(refine_cfg.get("grow_target_frac", 0.05)),
            grow_scale=float(refine_cfg.get("grow_scale", 0.0))
            or psf_based_grow_scale(median_psf, self._field_width, model_pixel_scale),
            prune_scale=float(refine_cfg.get("prune_scale", 0.1)),
            prune_axis_ratio=_optional_axis_ratio(refine_cfg.get("prune_axis_ratio")),
            prune_peak_sb_frac=float(refine_cfg.get("prune_peak_sb_frac", 0.005)),
            prune_band_snr=prune_band_snr,
            prune_band_snr_sigma=(
                None if band_snr_sigma is None else float(band_snr_sigma)
            ),
            refine_start=int(refine_cfg.get("start", 500)),
            refine_stop=int(refine_cfg.get("stop", 15_000)),
            refine_every=int(refine_cfg.get("every", 100)),
            max_splats=int(refine_cfg.get("max_splats", 0) or 0),
            birth_cap_per_event=(
                self.birth_cap_per_event if self.birth_enabled else 0
            ),
            birth_grace_steps=int(
                (refine_cfg.get("birth") or {}).get(
                    "grace_steps", int(refine_cfg.get("every", 100))
                )
            ),
        )
        return SplatRefineStrategy(
            config,
            field_width=self._field_width,
            reference_sigma=float(sigma.median()),
            band_snr_fn=self._build_band_snr_fn(base_model) if prune_band_snr else None,
        )

    def _prepare_band_tables(
        self, *, missing_band_policy: str = "raise", consumer: str = "band-SNR pruning",
    ) -> dict:
        """Per-band bandpass tuples, median image noise, and PSF variance in
        model normalized units, for rendered-significance pruning and gauge
        projection. Needs image_band_by_exposure, the global index cache, and
        renderer bandpasses, none of which exist yet at trainer init; built
        lazily on first use and cached."""
        if missing_band_policy not in {"raise", "warn"}:
            raise ValueError(
                "missing_band_policy must be 'raise' or 'warn'; got "
                f"{missing_band_policy!r}."
            )

        def handle_missing(message: str) -> None:
            if missing_band_policy == "raise":
                raise ValueError(message)
            warnings.warn(
                f"{consumer}: {message} Proceeding over the remaining bands.",
                RuntimeWarning,
                stacklevel=2,
            )

        cached = getattr(self, "_band_table_state", None)
        if cached is not None:
            missing_bands = getattr(self, "_band_table_missing_bands", ())
            if missing_bands:
                handle_missing(
                    f"Bands {list(missing_bands)} were excluded from the cached "
                    "band tables because they have no usable image data or "
                    "renderer bandpass."
                )
            return cached
        band_sigma: Dict[str, float] = {}
        band_psf_variance: Dict[str, float] = {}
        pixel_scales = [
            float(getattr(getattr(exp, "wcs", None), "pixel_scale_arcsec", 0.0) or 0.0)
            for exp_id, exp in self.data.exposures.items()
            if int(exp_id) not in self.exclude_image_exposure_ids
        ]
        positive_scales = [p for p in pixel_scales if p > 0]
        pixel_scale = float(np.median(positive_scales)) if positive_scales else 0.396
        # match TangentPlaneWCS: width-1 intervals span the [-1, 1] field
        norm_per_arcsec = 2.0 / (float(self._field_width - 1) * pixel_scale)
        # the renderer convolves with psf_sigma^2 + pixel^2/12 (quadrature
        # order 1 pixel-response term); the significance must use the same
        # effective variance or compact-splat peaks are overestimated
        pixel_variance = (pixel_scale * norm_per_arcsec) ** 2 / 12.0
        for band in sorted(set(self.image_band_by_exposure.values())):
            rows = []
            fwhms = []
            for exp_id, exp_band in self.image_band_by_exposure.items():
                if str(exp_band) != str(band):
                    continue
                if int(exp_id) in self.exclude_image_exposure_ids:
                    continue
                idx = self._global_indices_for(int(exp_id), DatumKind.IMAGE_PIXEL)
                if idx.numel() == 0:
                    continue
                step = max(1, int(idx.numel()) // 20_000)
                sample = idx[::step]
                if self.image_pixel_mask_absent_by_exposure is not None:
                    sample = sample[~self._image_pixel_absent_rows(sample, {int(exp_id)})]
                rows.append(
                    self._gather_named_rows(
                        "sigma", sample, device=torch.device("cpu"),
                        dtype=torch.float32,
                    )
                )
                exposure = self.exposures[int(exp_id)]
                # The lifecycle also uses legacy scalar FWHM for morphology
                # pruning.
                fwhm = float(getattr(exposure, "psf_fwhm_arcsec", 0.0) or 0.0)
                if fwhm > 0:
                    fwhms.append(fwhm)
            if rows:
                band_sigma[str(band)] = float(torch.cat(rows).median())
                median_fwhm = float(np.median(fwhms)) if fwhms else 1.4
                sigma_norm = median_fwhm / 2.355 * norm_per_arcsec
                band_psf_variance[str(band)] = sigma_norm ** 2 + pixel_variance
        if not band_sigma:
            raise ValueError(
                "refine.prune_band_snr found no image datums to estimate "
                "per-band noise from."
            )
        # Every band this run has image data in must reach the tables for the
        # rendered-significance prune. Other consumers may proceed over the
        # bands that still have usable tables.
        missing_sigma = sorted(
            {str(band) for band in self.image_band_by_exposure.values()}
            - set(band_sigma)
        )
        if missing_sigma:
            handle_missing(
                f"Bands {missing_sigma} have image exposures but no usable "
                "sigma rows (every exposure excluded, or no cached image "
                "datums). The rendered-significance tables would ignore them."
            )
        bandpass_tuples = {
            band: (bp.lam, bp.throughput)
            for band, bp in self.renderer_bandpasses.items()
            if band in band_sigma
        }
        if not bandpass_tuples:
            raise ValueError(
                "refine.prune_band_snr found no bandpasses matching the "
                "image bands."
            )
        missing_bandpass = sorted(set(band_sigma) - set(bandpass_tuples))
        if missing_bandpass:
            handle_missing(
                f"Bands {missing_bandpass} have image data but no renderer "
                "bandpass; their rendered amplitude cannot be computed."
            )
        self._band_table_missing_bands = tuple(
            sorted(set(missing_sigma) | set(missing_bandpass))
        )
        self._band_table_state = {
            "bandpass_tuples": bandpass_tuples,
            "band_sigma": {b: band_sigma[b] for b in bandpass_tuples},
            "band_psf_variance": {b: band_psf_variance[b] for b in bandpass_tuples},
            "pixel_norm": pixel_scale * norm_per_arcsec,
        }
        return self._band_table_state

    def _birth_accumulator(self):
        """Lazy per-run residual grid: PSF-scale cells over the model plane."""
        if self._birth_accumulator_state is None:
            from survey_fusion.models.splat_birth import ResidualBirthAccumulator

            # one grid = one model plane; a multi-WCS run must fail at
            # construction, not silently merge tangent planes cell-by-cell
            wcs_ids = {
                id(getattr(exp, "wcs", None))
                for exp in self.data.exposures.values()
                if getattr(exp, "wcs", None) is not None
            }
            if len(wcs_ids) > 1:
                raise RuntimeError(
                    "refine.birth accumulates on a single model-WCS plane; "
                    f"this run spans {len(wcs_ids)} WCS identities. The check "
                    "is identity-based: scene paths that build one (even "
                    "numerically identical) tangent plane per exposure also "
                    "refuse; birth requires the shared scene.model_wcs path."
                )
            tables = self._prepare_band_tables(
                missing_band_policy="warn", consumer="residual birth"
            )
            median_psf_sigma = float(
                np.median(
                    [math.sqrt(v) for v in tables["band_psf_variance"].values()]
                )
            )
            self._birth_accumulator_state = ResidualBirthAccumulator(
                cell_size=self.birth_cell_scale * median_psf_sigma,
                bands=[str(b) for b in tables["bandpass_tuples"]],
                device=self.device,
            )
        return self._birth_accumulator_state

    def _build_birth_rows(self, model) -> dict | None:
        """Rank residual cells and initialize complete splat rows for them."""
        accumulator = self._birth_accumulator_state
        if accumulator is None:
            return None
        candidates = accumulator.candidates(
            z_min=self.birth_z_min,
            n_eff_min=self.birth_n_eff_min,
            kappa=self.birth_kappa,
            top_k=self.birth_cap_per_event,
        )
        accumulator.reset()
        if candidates is None:
            return None
        from survey_fusion.pgm.splat_spectrum import template_band_averages

        tables = self._prepare_band_tables(
            missing_band_policy="warn", consumer="residual birth"
        )
        band_keys, template_map = template_band_averages(
            model.sed,
            tables["bandpass_tuples"],
            loglam_min=self.loglam_min,
            loglam_max=self.loglam_max,
        )
        # rank-revealing CPU solve: the candidate count is tiny (<= cap) and
        # CUDA gels assumes full rank of a learned, possibly collapsing map
        template_map = template_map.detach().to("cpu", torch.float64)  # (K, B)
        order = [accumulator.band_index[str(key)] for key in band_keys]
        cell_area_px = (accumulator.cell_size / tables["pixel_norm"]) ** 2
        # Rendered pixel values are density samples in data units (peak =
        # flux * A / (2 pi sqrt det), no pixel-area factor), so a newborn's
        # prediction summed over the cell's pixels is flux * A / pixel_norm^2.
        # Inverting the cell's aperture residual sum onto model flux therefore
        # carries pixel_norm^2; without it the init lands ~1/pixel_norm^2
        # (~2.5e7 at full84 scale) too bright, enough to make the loss
        # diverge at the first birth event.
        aperture = (
            candidates["mean_residual_flux"][order, :].to("cpu", torch.float64)
            * cell_area_px
            * float(tables["pixel_norm"]) ** 2
        ).clamp_min(0.0)  # (B, k) in band_keys order, model flux units
        eligible = candidates["eligible"][order, :].cpu()
        z = candidates["z"][order, :].to("cpu", torch.float64)
        best = torch.where(eligible, z, -torch.inf).argmax(dim=0)  # (k,)
        k = int(best.numel())
        rows = torch.arange(k)
        # Flux/SED separation: sed_coeffs carry the band SHAPE (ratios to the
        # best band), raw_flux carries the luminosity. Solving on absolute
        # fluxes would park brightness in the coefficients and make every
        # birth render the same raw flux. Only bands the candidate actually
        # triggered in constrain the solve: an unsampled band is unknown,
        # not an observed zero. Approximate by design; grace lets it train.
        # The positivity floor must scale with the data units (scenes may
        # apply training_flux_scale): floor at the flux of a 1e-3-sigma cell
        # in the triggering band, deeply sub-noise in any unit system, rather
        # than at an absolute constant.
        sigma_by_band = torch.tensor(
            [float(tables["band_sigma"][str(key)]) for key in band_keys],
            dtype=torch.float64,
        )
        flux_floor = (
            sigma_by_band[best] * float(accumulator.cell_size) ** 2 * 1e-3
        )
        flux = torch.maximum(aperture[best, rows], flux_floor)
        ratios = aperture / flux[None, :]
        solution = torch.empty(
            (template_map.shape[0], k), dtype=torch.float64
        )
        design = template_map.T  # (B, K)
        for index in range(k):
            bands = eligible[:, index]
            fit = torch.linalg.lstsq(
                design[bands], ratios[bands, index : index + 1],
                driver="gelsd",
            ).solution.clamp_(min=1e-6)
            # renormalize so the best band renders ratio 1 exactly and the
            # luminosity stays entirely in raw_flux
            rendered = float(design[best[index]] @ fit[:, 0])
            solution[:, index] = fit[:, 0] / max(rendered, 1e-12)
        median_psf_sigma = accumulator.cell_size / self.birth_cell_scale

        def inv_softplus(value: torch.Tensor) -> torch.Tensor:
            return value + torch.log(-torch.expm1(-value))

        device = model.mu.device
        return {
            "mu": candidates["centers"].to(device=device, dtype=model.mu.dtype),
            "log_scales": torch.full(
                (k, 2), math.log(median_psf_sigma),
                device=device, dtype=model.log_scales.dtype,
            ),
            "theta": torch.zeros(k, device=device, dtype=model.theta.dtype),
            "raw_flux": inv_softplus(flux).to(
                device=device, dtype=model.raw_flux.dtype
            ),
            "sed_coeffs": inv_softplus(solution.T).to(
                device=device, dtype=model.sed_coeffs.dtype
            ),
            # a newborn has no redshift estimate; grace exists so sed_shift
            # can train before the first verdict
            "sed_shift": torch.zeros(
                k, device=device, dtype=model.sed_shift.dtype
            ),
        }

    def _build_band_snr_fn(self, base_model):
        """Per-splat PSF-convolved rendered peak significance, max over bands
        (splat_rendered_significance). The raw_flux/SED split is gauge-
        degenerate, so pruning must test the rendered product; the intrinsic
        peak diverges as a compact splat shrinks, so the determinant must
        carry the band PSF variance the image renderer applies."""
        from survey_fusion.pgm.splat_spectrum import splat_rendered_significance

        if not callable(getattr(getattr(base_model, "sed", None), "templates", None)):
            raise ValueError(
                "refine.prune_band_snr requires a splat model with an SED "
                "dictionary; the bandless prune has no rendered amplitude."
            )

        def band_snr_fn(model) -> torch.Tensor:
            tables = self._prepare_band_tables(
                missing_band_policy="raise", consumer="band-SNR pruning"
            )
            return splat_rendered_significance(
                model,
                tables["bandpass_tuples"],
                tables["band_sigma"],
                tables["band_psf_variance"],
                loglam_min=self.loglam_min,
                loglam_max=self.loglam_max,
            )

        return band_snr_fn

    def _splat_gauge_factors(self, model) -> torch.Tensor:
        """Band-max per-unit-flux band amplitude: the canonical-gauge scale."""
        from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes

        tables = self._prepare_band_tables(
            missing_band_policy="warn", consumer="gauge projection"
        )
        with torch.no_grad():
            unit_amplitudes = splat_band_amplitudes(
                model,
                tables["bandpass_tuples"],
                loglam_min=self.loglam_min,
                loglam_max=self.loglam_max,
                include_flux=False,
            )
            stacked = torch.stack(
                [unit_amplitudes[b] for b in sorted(unit_amplitudes)], dim=0
            )
            return stacked.amax(dim=0)

    def refine_state_dict(self) -> dict | None:
        if self.refine_strategy is None:
            return None
        return self.refine_strategy.state_dict()

    def load_refine_state_dict(self, state: dict | None) -> bool:
        if self.refine_strategy is None or state is None:
            return False
        self.refine_strategy.load_state_dict(state, model=self._base_model())
        return True

    def _refresh_trainable_params(self) -> None:
        """Rebuild the trainable-parameter list after refinement surgery.

        Surgery replaces every per-splat nn.Parameter with a new object, so a
        list captured at construction keeps pointing at the orphaned tensors:
        gradient clipping and gradient-norm logging would silently operate on
        parameters the optimizer no longer updates."""
        self.params = list(self.model.parameters())
        if self.spectral_calibrator is not None:
            self.params.extend(self.spectral_calibrator.parameters())
        self._refresh_clip_params()

    def _refresh_clip_params(self) -> None:
        """Recompute the set of parameters subject to global gradient
        clipping: everything in self.params except the per-splat tensors.
        Must be called whenever self.params is rebuilt, since refinement
        surgery replaces the per-splat nn.Parameter objects."""
        splat_ids: set[int] = set()
        base_model = self._base_model()
        groups_fn = getattr(base_model, "sparse_param_groups_gsplat", None)
        if callable(groups_fn):
            for group in groups_fn():
                for param in group["params"]:
                    splat_ids.add(id(param))
        self._clip_param_ids = {
            id(param) for param in self.params if id(param) not in splat_ids
        }

    @staticmethod
    def _signature_update_bytes(digest, value: bytes) -> None:
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    @classmethod
    def _signature_update_value(cls, digest, value: object) -> None:
        if torch.is_tensor(value):
            cls._signature_update_bytes(digest, b"tensor")
            if value.layout != torch.strided:
                raise ValueError(
                    "Distributed state signatures require strided tensors."
                )
            cls._signature_update_value(
                digest, tuple(int(size) for size in value.shape)
            )
            cls._signature_update_bytes(digest, str(value.dtype).encode("utf-8"))
            cpu_value = value.detach().contiguous().cpu().reshape(-1)
            byte_view = cpu_value.view(torch.uint8)
            if sys.byteorder == "big" and cpu_value.element_size() > 1:
                byte_view = (
                    byte_view.reshape(-1, cpu_value.element_size())
                    .flip(1)
                    .reshape(-1)
                )
            cls._signature_update_bytes(
                digest, byte_view.numpy().tobytes()
            )
            return
        if value is None:
            cls._signature_update_bytes(digest, b"none")
            return
        if isinstance(value, bool):
            cls._signature_update_bytes(digest, b"bool")
            digest.update(b"\x01" if value else b"\x00")
            return
        if isinstance(value, int):
            cls._signature_update_bytes(digest, b"int")
            cls._signature_update_bytes(digest, str(value).encode("ascii"))
            return
        if isinstance(value, float):
            cls._signature_update_bytes(digest, b"float")
            digest.update(struct.pack("<d", value))
            return
        if isinstance(value, str):
            cls._signature_update_bytes(digest, b"str")
            cls._signature_update_bytes(digest, value.encode("utf-8"))
            return
        if isinstance(value, (torch.dtype, torch.device)):
            cls._signature_update_bytes(digest, b"torch")
            cls._signature_update_bytes(digest, str(value).encode("utf-8"))
            return
        if isinstance(value, Mapping):
            cls._signature_update_bytes(digest, b"mapping")
            items = sorted(value.items(), key=lambda item: str(item[0]))
            cls._signature_update_value(digest, len(items))
            for key, item in items:
                cls._signature_update_value(digest, str(key))
                cls._signature_update_value(digest, item)
            return
        if isinstance(value, tuple):
            cls._signature_update_bytes(digest, b"tuple")
            cls._signature_update_value(digest, len(value))
            for item in value:
                cls._signature_update_value(digest, item)
            return
        if isinstance(value, list):
            cls._signature_update_bytes(digest, b"list")
            cls._signature_update_value(digest, len(value))
            for item in value:
                cls._signature_update_value(digest, item)
            return
        raise TypeError(
            f"Unsupported distributed signature value {type(value).__qualname__}."
        )

    def _distributed_state_signature(self) -> str:
        from survey_fusion.models.splat_strategy import PER_SPLAT_PARAMS

        digest = hashlib.sha256()
        base_model = self._base_model()
        named_parameters = tuple(base_model.named_parameters())
        self._signature_update_value(digest, len(named_parameters))
        for name, parameter in named_parameters:
            self._signature_update_value(digest, name)
            self._signature_update_value(digest, parameter)

        named_buffers = tuple(base_model.named_buffers())
        self._signature_update_value(digest, len(named_buffers))
        for name, buffer in named_buffers:
            self._signature_update_value(digest, name)
            self._signature_update_value(digest, buffer)

        optimizers = (self.optimizer, self.splat_optimizer)
        for optimizer_index, optimizer in enumerate(optimizers):
            if optimizer is None:
                continue
            self._signature_update_value(digest, optimizer_index)
            self._signature_update_value(
                digest, type(optimizer).__qualname__
            )
            self._signature_update_value(digest, len(optimizer.param_groups))
            for group_index, group in enumerate(optimizer.param_groups):
                self._signature_update_value(digest, group_index)
                hyperparameters = {
                    key: value for key, value in group.items() if key != "params"
                }
                self._signature_update_value(digest, hyperparameters)
            for name in PER_SPLAT_PARAMS:
                parameter = getattr(base_model, name)
                self._signature_update_value(digest, name)
                state = optimizer.state.get(parameter, {})
                self._signature_update_value(digest, len(state))
                for key, value in sorted(
                    state.items(), key=lambda item: str(item[0])
                ):
                    self._signature_update_value(digest, str(key))
                    self._signature_update_value(digest, value)
        return digest.hexdigest()

    def _verify_distributed_state_signatures(self, stage: str) -> str | None:
        if not is_distributed_active(self.distributed_context):
            return None
        if not callable(
            getattr(self._base_model(), "param_groups_gsplat", None)
        ):
            return None
        local_error = None
        signature = None
        try:
            signature = self._distributed_state_signature()
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            {
                "rank": self.rank,
                "error": local_error,
                "signature": signature,
            },
        )
        errors = [
            f"rank {entry.get('rank')}: {entry.get('error')}"
            for entry in gathered
            if isinstance(entry, Mapping) and entry.get("error") is not None
        ]
        if errors:
            raise RuntimeError(
                f"Distributed splat {stage} signature failed: "
                + "; ".join(errors)
            )
        if any(not isinstance(entry, Mapping) for entry in gathered):
            raise RuntimeError(
                f"Distributed splat {stage} signature exchange was malformed."
            )
        signatures = [
            entry.get("signature")
            for entry in gathered
            if isinstance(entry, Mapping)
        ]
        if not signatures or any(
            candidate != signatures[0] for candidate in signatures[1:]
        ):
            details = ", ".join(
                f"rank {entry.get('rank')}: {entry.get('signature')}"
                for entry in gathered
                if isinstance(entry, Mapping)
            )
            raise RuntimeError(
                f"Distributed splat {stage} state signature mismatch: {details}"
            )
        return signature

    def _distributed_refine(self, global_step: int) -> dict:
        from survey_fusion.models.splat_strategy import SplatRefinePlan

        strategy = self.refine_strategy
        if strategy is None or not strategy._cadence_matches(
            strategy.config, global_step
        ):
            return {}
        if getattr(self, "gauge_projection_enabled", False) or getattr(
            self, "birth_enabled", False
        ):
            # rank-0-only planning with deliberately-unreduced statistics
            # cannot mutate parameters consistently across ranks, and
            # rank-local residual grids would propose rank-0-only births;
            # failing loudly beats silent rank divergence
            raise RuntimeError(
                "refine.gauge_projection and refine.birth are not "
                "distributed-correct; disable them for distributed training."
            )

        envelope_list: list[object] = [None]
        if self.rank == 0:
            try:
                plan = strategy.make_plan(self._base_model(), global_step)
                envelope_list[0] = {
                    "error": None,
                    "plan": None if plan is None else plan.to_payload(),
                }
            except Exception as error:
                envelope_list[0] = {
                    "error": f"{type(error).__name__}: {error}",
                    "plan": None,
                }
        dist.broadcast_object_list(envelope_list, src=0)
        envelope = envelope_list[0]
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "error",
            "plan",
        }:
            raise RuntimeError(
                "Distributed refinement received an invalid plan envelope."
            )
        if envelope["error"] is not None:
            raise RuntimeError(
                f"Distributed refinement planning failed: {envelope['error']}"
            )
        payload = envelope["plan"]
        if payload is None:
            strategy.grad_accum = None
            strategy.count = None
            return {}

        local_error = None
        report: dict = {}
        signature = None
        try:
            plan = SplatRefinePlan.from_payload(
                payload, self._base_model().mu.device
            )
            optimizers = [
                optimizer
                for optimizer in (self.optimizer, self.splat_optimizer)
                if optimizer is not None
            ]
            report = strategy.apply_plan(
                self._base_model(), optimizers, plan
            )
            self._refresh_trainable_params()
            self._position_group = next(
                (
                    group
                    for optimizer in optimizers
                    for group in optimizer.param_groups
                    if any(
                        parameter is self._base_model().mu
                        for parameter in group["params"]
                    )
                ),
                self._position_group,
            )
            signature = self._distributed_state_signature()
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(
            gathered,
            {
                "rank": self.rank,
                "error": local_error,
                "signature": signature,
            },
        )
        malformed = [
            rank for rank, entry in enumerate(gathered)
            if not isinstance(entry, Mapping)
        ]
        errors = [
            f"rank {entry.get('rank')}: {entry.get('error')}"
            for entry in gathered
            if isinstance(entry, Mapping) and entry.get("error") is not None
        ]
        if malformed:
            errors.extend(
                f"rank {rank}: malformed application result"
                for rank in malformed
            )
        if errors:
            raise RuntimeError(
                "Distributed refinement application failed: "
                + "; ".join(errors)
            )
        signatures = [
            entry.get("signature")
            for entry in gathered
            if isinstance(entry, Mapping)
        ]
        if not signatures or any(
            candidate != signatures[0] for candidate in signatures[1:]
        ):
            details = ", ".join(
                f"rank {entry.get('rank')}: {entry.get('signature')}"
                for entry in gathered
                if isinstance(entry, Mapping)
            )
            raise RuntimeError(
                "Distributed refinement state signature mismatch: " + details
            )
        return report

    def _set_calibration_lr_for_update(self, update: int) -> None:
        """Hold the spectral calibration at its initial value for the first
        freeze_until_step updates, then release it at its configured rate: a
        free global nuisance otherwise absorbs the field's early misfit."""
        calibrator = self.spectral_calibrator
        if calibrator is None or calibrator.cfg.freeze_until_step <= 0:
            return
        group = next(
            g for g in self.optimizer.param_groups
            if any(p is calibrator.log_scales for p in g["params"])
        )
        group["lr"] = (
            0.0 if int(update) <= calibrator.cfg.freeze_until_step
            else self._base_learning_rate * float(calibrator.cfg.lr_multiplier)
        )

    def _post_optimizer_step(self, global_step: int) -> dict:
        """gsplat step order: optimizer step, then position-lr decay, then
        densify/prune."""
        self._set_calibration_lr_for_update(int(global_step) + 1)
        if self._position_group is not None and self._position_lr_gamma != 1.0:
            decay_steps = min(int(global_step), int(self._position_lr_horizon))
            self._position_group["lr"] = self._position_lr_init * (
                self._position_lr_gamma ** decay_steps
            )
        if self.refine_strategy is None:
            return {}
        optimizers = [
            opt for opt in (self.optimizer, self.splat_optimizer) if opt is not None
        ]
        distributed_active = is_distributed_active(self.distributed_context)
        if distributed_active:
            report = self._distributed_refine(global_step)
        else:
            base_model = self._base_model()
            strategy = self.refine_strategy
            if getattr(
                self, "gauge_projection_enabled", False
            ) and strategy._cadence_matches(
                strategy.config, global_step
            ):
                projected = strategy.project_gauge(
                    base_model, optimizers, self._splat_gauge_factors(base_model)
                )
                if projected:
                    print(
                        f"[refine] step {global_step}: gauge-projected "
                        f"{projected} splats",
                        flush=True,
                    )
            birth_rows = None
            if getattr(self, "birth_enabled", False) and strategy._cadence_matches(
                strategy.config, global_step
            ):
                birth_rows = self._build_birth_rows(base_model)
            plan = strategy.make_plan(
                base_model, global_step, birth_rows=birth_rows
            )
            report = (
                {} if plan is None else strategy.apply_plan(
                    base_model, optimizers, plan
                )
            )
        if report and not distributed_active:
            self._position_group = next(
                (
                    g
                    for opt in optimizers
                    for g in opt.param_groups
                    if any(p is self._base_model().mu for p in g["params"])
                ),
                self._position_group,
            )
            self._refresh_trainable_params()
        if report:
            strategy_cfg = self.refine_strategy.config
            band_snr = bool(strategy_cfg.prune_band_snr)
            reasons = report.get("prune_reasons")
            reason_note = (
                f" ({'band_snr' if band_snr else 'flux'} {reasons['flux']}, "
                f"axis {reasons['axis_ratio']}, scale {reasons['scale']})"
                if reasons
                else ""
            )
            # The census trajectory of a band-SNR arm is only auditable against
            # the threshold that produced it, and that threshold is set per run.
            criterion_note = (
                f" [threshold {strategy_cfg.prune_band_snr_sigma:g} sigma "
                "peak SNR]"
                if band_snr
                else ""
            )
            birth_note = (
                f" +{report['birth']} born" if report.get("birth") else ""
            )
            print(
                f"[refine] step {global_step}: +{report['dupli']} dupli, "
                f"+{report['split']} split{birth_note}, -{report['prune']} "
                f"pruned{reason_note}{criterion_note}, "
                f"n_splats={report['n_splats']}",
                flush=True,
            )
        return report

    def optimizer_state_dict(self) -> dict[str, Any]:
        if self.splat_optimizer is None:
            return self.optimizer.state_dict()
        return {
            "format": "hybrid_sparse_v1",
            "dense": self.optimizer.state_dict(),
            "splat": self.splat_optimizer.state_dict(),
        }

    @staticmethod
    def _host_optimizer_step_counters(optimizer: Any) -> None:
        """Put Adam's step counters back on the host, where torch puts them.

        torch hosts `step` on the CPU whenever capturable and fused are both
        off, deliberately: the bias corrections read it as a Python number
        every step, so a device-resident counter is a device-to-host copy in
        the middle of the optimizer, four of them here. `load_state_dict`
        preserves whatever device the counter arrives on, and resume loads the
        whole checkpoint with map_location=device, so a resumed run runs with
        the counters on the GPU while a fresh one does not. The value is a
        small integer in float32, exact on either device.
        """
        for state in optimizer.state.values():
            step = state.get("step")
            if torch.is_tensor(step) and step.device.type != "cpu":
                state["step"] = step.cpu()

    def load_optimizer_state_dict(self, state: dict[str, Any]) -> None:
        if self.splat_optimizer is None:
            if state.get("format") == "hybrid_sparse_v1":
                raise ValueError(
                    "A sparse-splat optimizer checkpoint cannot initialize dense training."
                )
            self.optimizer.load_state_dict(state)
            self._host_optimizer_step_counters(self.optimizer)
            self._rebind_position_group()
            return
        if state.get("format") != "hybrid_sparse_v1":
            raise ValueError(
                "Sparse-splat training requires a hybrid_sparse_v1 optimizer checkpoint."
            )
        dense_state = state.get("dense")
        splat_state = state.get("splat")
        if not isinstance(dense_state, dict) or not isinstance(splat_state, dict):
            raise ValueError(
                "hybrid_sparse_v1 optimizer state requires dense and splat mappings."
            )
        self.optimizer.load_state_dict(dense_state)
        self._host_optimizer_step_counters(self.optimizer)
        self.splat_optimizer.load_state_dict(splat_state)
        self._rebind_position_group()

    def _rebind_position_group(self) -> None:
        """Point the position-rate schedule at the live group: load_state_dict replaces the group dicts."""
        if getattr(self, "_position_group", None) is None:
            return
        mu = self._base_model().mu
        self._position_group = next(
            (
                group
                for optimizer in (self.splat_optimizer, self.optimizer)
                if optimizer is not None
                for group in optimizer.param_groups
                if any(parameter is mu for parameter in group["params"])
            ),
            self._position_group,
        )

    @staticmethod
    def _parse_lr_scheduler_cfg(raw_cfg: Any) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "enabled": False, "type": "ReduceLROnPlateau", "monitor": "epoch_mean_total_loss", "mode": "min",
            "factor": 0.5, "patience": 6, "threshold": 1e-3, "threshold_mode": "rel", "cooldown": 2,
            "min_lr": 1e-5, "eps": 1e-8, "max_lr": None, "pct_start": 0.3, "anneal_strategy": "cos",
            "cycle_momentum": False, "div_factor": 25.0, "final_div_factor": 1e4, "three_phase": False,
        }
        if raw_cfg is None:
            cfg_map: Dict[str, Any] = {}
        elif isinstance(raw_cfg, dict):
            cfg_map = dict(raw_cfg)
        else:
            raise ValueError("trainer.lr_scheduler must be a mapping.")
        cfg: Dict[str, Any] = {**defaults, **cfg_map}

        enabled = cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("trainer.lr_scheduler.enabled must be a bool.")
        scheduler_type = cfg.get("type", "ReduceLROnPlateau")
        if scheduler_type not in {"ReduceLROnPlateau", "OneCycleLR"}:
            raise ValueError(
                "trainer.lr_scheduler.type must be one of {'ReduceLROnPlateau','OneCycleLR'}."
            )
        monitor = str(cfg.get("monitor", "epoch_mean_total_loss"))
        if scheduler_type == "ReduceLROnPlateau" and monitor != "epoch_mean_total_loss":
            raise ValueError("trainer.lr_scheduler.monitor must be exactly 'epoch_mean_total_loss'.")
        mode = str(cfg.get("mode", "min"))
        if mode not in {"min", "max"}:
            raise ValueError("trainer.lr_scheduler.mode must be one of {'min','max'}.")
        threshold_mode = str(cfg.get("threshold_mode", "rel"))
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(
                "trainer.lr_scheduler.threshold_mode must be one of {'rel','abs'}."
            )

        try:
            factor = float(cfg.get("factor", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.factor must be a float in (0, 1).") from exc
        if not (0.0 < factor < 1.0):
            raise ValueError("trainer.lr_scheduler.factor must be in (0, 1).")

        try:
            patience = int(cfg.get("patience", 6))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.patience must be an integer >= 0.") from exc
        if patience < 0:
            raise ValueError("trainer.lr_scheduler.patience must be >= 0.")

        try:
            threshold = float(cfg.get("threshold", 1e-3))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.threshold must be a float.") from exc

        try:
            cooldown = int(cfg.get("cooldown", 2))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.cooldown must be an integer >= 0.") from exc
        if cooldown < 0:
            raise ValueError("trainer.lr_scheduler.cooldown must be >= 0.")

        try:
            min_lr = float(cfg.get("min_lr", 1e-5))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.min_lr must be a float >= 0.") from exc
        if min_lr < 0.0:
            raise ValueError("trainer.lr_scheduler.min_lr must be >= 0.")

        try:
            eps = float(cfg.get("eps", 1e-8))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.eps must be a float > 0.") from exc
        if eps <= 0.0:
            raise ValueError("trainer.lr_scheduler.eps must be > 0.")

        max_lr_raw = cfg.get("max_lr")
        max_lr: float | list[float] | None
        if max_lr_raw is None:
            max_lr = None
        elif isinstance(max_lr_raw, (list, tuple)):
            max_lr = []
            for value in max_lr_raw:
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("trainer.lr_scheduler.max_lr values must be floats > 0.") from exc
                if parsed <= 0.0:
                    raise ValueError("trainer.lr_scheduler.max_lr values must be > 0.")
                max_lr.append(parsed)
        else:
            try:
                max_lr = float(max_lr_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("trainer.lr_scheduler.max_lr must be a float > 0.") from exc
            if max_lr <= 0.0:
                raise ValueError("trainer.lr_scheduler.max_lr must be > 0.")

        try:
            pct_start = float(cfg.get("pct_start", 0.3))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.pct_start must be a float in (0, 1).") from exc
        if not (0.0 < pct_start < 1.0):
            raise ValueError("trainer.lr_scheduler.pct_start must be in (0, 1).")

        anneal_strategy = str(cfg.get("anneal_strategy", "cos"))
        if anneal_strategy not in {"cos", "linear"}:
            raise ValueError(
                "trainer.lr_scheduler.anneal_strategy must be one of {'cos','linear'}."
            )
        cycle_momentum = cfg.get("cycle_momentum", False)
        if not isinstance(cycle_momentum, bool):
            raise ValueError("trainer.lr_scheduler.cycle_momentum must be a bool.")
        three_phase = cfg.get("three_phase", False)
        if not isinstance(three_phase, bool):
            raise ValueError("trainer.lr_scheduler.three_phase must be a bool.")

        try:
            div_factor = float(cfg.get("div_factor", 25.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.div_factor must be a float > 0.") from exc
        if div_factor <= 0.0:
            raise ValueError("trainer.lr_scheduler.div_factor must be > 0.")

        try:
            final_div_factor = float(cfg.get("final_div_factor", 1e4))
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer.lr_scheduler.final_div_factor must be a float > 0.") from exc
        if final_div_factor <= 0.0:
            raise ValueError("trainer.lr_scheduler.final_div_factor must be > 0.")

        return {
            "enabled": enabled, "type": scheduler_type, "monitor": monitor, "mode": mode, "factor": factor,
            "patience": patience, "threshold": threshold, "threshold_mode": threshold_mode,
            "cooldown": cooldown, "min_lr": min_lr, "eps": eps, "max_lr": max_lr, "pct_start": pct_start,
            "anneal_strategy": anneal_strategy, "cycle_momentum": cycle_momentum, "div_factor": div_factor,
            "final_div_factor": final_div_factor, "three_phase": three_phase,
        }

    @staticmethod
    def _parse_image_importance_sampling_cfg(raw_cfg: Any) -> ImageImportanceSamplingConfig:
        defaults = ImageImportanceSamplingConfig()
        if raw_cfg is None:
            cfg_map: Dict[str, Any] = {}
        elif isinstance(raw_cfg, Mapping):
            cfg_map = dict(raw_cfg)
        else:
            raise ValueError("trainer.image_importance_sampling must be a mapping.")

        supported_keys = {"enabled", "max_importance_weight"}
        unsupported = sorted(str(key) for key in cfg_map.keys() if str(key) not in supported_keys)
        if unsupported:
            raise ValueError(
                "trainer.image_importance_sampling contains removed/unsupported keys: "
                f"{unsupported}. Retained keys are {sorted(supported_keys)}."
            )

        enabled = cfg_map.get("enabled", defaults.enabled)
        if not isinstance(enabled, bool):
            raise ValueError("trainer.image_importance_sampling.enabled must be a bool.")
        try:
            max_importance_weight = float(
                cfg_map.get("max_importance_weight", defaults.max_importance_weight)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trainer.image_importance_sampling.max_importance_weight must be a float >= 1."
            ) from exc
        if max_importance_weight < 1.0:
            raise ValueError("trainer.image_importance_sampling.max_importance_weight must be >= 1.")
        return ImageImportanceSamplingConfig(enabled=enabled, max_importance_weight=max_importance_weight)

    def _build_global_index_cache(self) -> Dict[tuple[int, int], torch.Tensor]:
        batch_exposure = self.data.batch.exposure_id
        batch_kind = self.data.batch.datum_kind
        batch_local = self.data.batch.datum_index
        n_rows = int(batch_exposure.numel())
        if n_rows == 0:
            return {}
        if batch_kind.numel() != n_rows or batch_local.numel() != n_rows:
            raise ValueError(
                "Datum exposure, kind, and local-index columns must have equal length."
            )

        changes = torch.nonzero(
            (batch_exposure[1:] != batch_exposure[:-1])
            | (batch_kind[1:] != batch_kind[:-1]),
            as_tuple=True,
        )[0]
        boundaries = [0, *(int(value) + 1 for value in changes.tolist()), n_rows]
        parts: Dict[tuple[int, int], list[torch.Tensor]] = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            key = (
                int(batch_exposure[start]),
                int(batch_kind[start]),
            )
            indices = torch.arange(start, end, dtype=torch.long)
            local = batch_local[start:end]
            if local.numel() > 1 and bool(torch.any(local[1:] < local[:-1])):
                indices = indices[torch.argsort(local)]
            parts.setdefault(key, []).append(indices)

        cache: Dict[tuple[int, int], torch.Tensor] = {}
        for key, key_parts in parts.items():
            indices = key_parts[0] if len(key_parts) == 1 else torch.cat(key_parts)
            if len(key_parts) > 1:
                local = torch.index_select(batch_local, 0, indices)
                indices = indices[torch.argsort(local)]
            cache[key] = indices
        return cache

    def _init_spectral_calibrator(self) -> None:
        cfg = self.spectral_calibration_cfg
        if not cfg.enabled:
            self.spectral_calibrator = None
            return
        if not self.spec_exposure_ids:
            self.spectral_calibrator = None
            return
        lam_by_exposure = {
            int(exp_id): np.asarray(self.exposures[int(exp_id)].lam_spec, dtype=np.float64)
            for exp_id in self.spec_exposure_ids
        }
        self.spectral_calibrator = SpectralNuisanceCalibrator(
            spectral_lam_by_exposure=lam_by_exposure, bandpasses=self.renderer_bandpasses, cfg=cfg,
            device=self.device,
        )

    def _apply_spectral_nuisance(
        self, exposure_id: int, local_spec_indices: torch.Tensor, spec_mu: torch.Tensor, *,
        apply_spectral_nuisance: bool = True,
    ) -> torch.Tensor:
        if not apply_spectral_nuisance or self.spectral_calibrator is None:
            return spec_mu
        scales = self.spectral_calibrator.scale_for_indices(exposure_id, local_spec_indices)
        return spec_mu * scales.to(dtype=spec_mu.dtype)

    def _compute_global_loglam_bounds(self) -> Tuple[float, float]:
        lam_min: float | None = None
        lam_max: float | None = None
        for exposure in self.exposures.values():
            for bp in exposure.bandpasses.values():
                lam_arr = np.asarray(bp.lam, dtype=np.float64)
                lam_arr = lam_arr[np.isfinite(lam_arr) & (lam_arr > 0)]
                if lam_arr.size == 0:
                    continue
                lo = float(np.min(lam_arr))
                hi = float(np.max(lam_arr))
                lam_min = lo if lam_min is None else min(lam_min, lo)
                lam_max = hi if lam_max is None else max(lam_max, hi)
            lam_spec = np.asarray(exposure.lam_spec, dtype=np.float64)
            lam_spec = lam_spec[np.isfinite(lam_spec) & (lam_spec > 0)]
            if lam_spec.size > 0:
                lo = float(np.min(lam_spec))
                hi = float(np.max(lam_spec))
                lam_min = lo if lam_min is None else min(lam_min, lo)
                lam_max = hi if lam_max is None else max(lam_max, hi)
        if lam_min is None or lam_max is None or lam_max <= lam_min:
            raise ValueError("Unable to determine valid global wavelength bounds for normalization.")
        return float(np.log(lam_min)), float(np.log(lam_max))

    def _collect_renderer_bandpasses(self) -> Dict[str, Any]:
        bandpasses: Dict[str, Any] = {}
        for exposure in self.exposures.values():
            for band, bp in exposure.bandpasses.items():
                if band not in bandpasses:
                    bandpasses[band] = bp
        if not bandpasses:
            raise ValueError("At least one image bandpass is required to construct spectral renderers.")
        return bandpasses

    @staticmethod
    def _deterministic_lsf_fwhm_value(lsf_fwhm_angstrom: float | np.ndarray | None) -> float:
        """Reduce scalar/array LSF metadata to a deterministic scalar for legacy projector code."""
        if lsf_fwhm_angstrom is None:
            return 0.0
        if np.isscalar(lsf_fwhm_angstrom):
            value = float(lsf_fwhm_angstrom)
            return value if np.isfinite(value) and value > 0 else 0.0
        arr = np.asarray(lsf_fwhm_angstrom, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            return 0.0
        return float(np.median(arr))

    @staticmethod
    def _apply_diagnostic_lsf(
        spectrum: torch.Tensor, lam_spec: np.ndarray, lsf_fwhm_angstrom: float | np.ndarray | None, *,
        lam_edges: np.ndarray | None = None, truncate_sigma: float = 4.0,
    ) -> torch.Tensor:
        if lsf_fwhm_angstrom is None:
            return spectrum
        lam = np.asarray(lam_spec, dtype=np.float64)
        if lam.ndim != 1 or lam.size == 0:
            return spectrum
        if np.isscalar(lsf_fwhm_angstrom):
            fwhm = np.full(lam.shape, float(lsf_fwhm_angstrom), dtype=np.float64)
        else:
            fwhm = np.asarray(lsf_fwhm_angstrom, dtype=np.float64)
            if fwhm.shape != lam.shape:
                raise ValueError("lsf_fwhm_angstrom must align with lam_spec for diagnostics.")
        if lam_edges is not None:
            edges = np.asarray(lam_edges, dtype=np.float64)
            if edges.shape != (lam.shape[0], 2):
                raise ValueError("lam_edges must have shape [len(lam_spec), 2] for diagnostics.")
            bin_width = edges[:, 1] - edges[:, 0]
        elif lam.size == 1:
            bin_width = np.ones((1,), dtype=np.float64)
        else:
            midpoints = 0.5 * (lam[1:] + lam[:-1])
            edges = np.empty((lam.shape[0] + 1,), dtype=np.float64)
            edges[1:-1] = midpoints
            edges[0] = lam[0] - 0.5 * (lam[1] - lam[0])
            edges[-1] = lam[-1] + 0.5 * (lam[-1] - lam[-2])
            bin_width = np.diff(edges)
        if np.any(~np.isfinite(bin_width)) or np.any(bin_width <= 0.0):
            raise ValueError("Diagnostic spectral bin widths must be positive and finite.")
        sigma_angstrom = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if not np.any(np.isfinite(sigma_angstrom) & (sigma_angstrom > 0.0)):
            return spectrum

        lam_t = torch.tensor(lam, dtype=spectrum.dtype, device=spectrum.device)
        bin_width_t = torch.tensor(bin_width, dtype=spectrum.dtype, device=spectrum.device)
        output = torch.empty_like(spectrum)
        for idx, sigma_value in enumerate(sigma_angstrom):
            sigma = float(sigma_value)
            if not np.isfinite(sigma) or sigma <= 0.0:
                output[idx] = spectrum[idx]
                continue
            support = torch.abs(lam_t - lam_t[idx]) <= float(truncate_sigma) * sigma
            if not bool(torch.any(support).item()):
                output[idx] = spectrum[idx]
                continue
            delta = (lam_t[support] - lam_t[idx]) / sigma
            weights = torch.exp(-0.5 * delta * delta) * bin_width_t[support]
            weights_sum = torch.sum(weights)
            if not torch.isfinite(weights_sum) or float(weights_sum.item()) <= 0.0:
                output[idx] = spectrum[idx]
                continue
            output[idx] = torch.sum(spectrum[support] * weights) / weights_sum
        return output

    def _fiber_geometry(self, exposure, pixel_wcs, height: int, width: int, psf_kernel):
        """Sky directions and aperture weights for one exposure.

        Only the pixels the fiber actually covers are evaluated. A full
        (height, width) omega map costs 1.9 GB on an 84 arcmin model grid, per
        spectral exposure, for an aperture of about fifty pixels; with 910
        spectra that alone exceeds any node.

        Image exposures keep the dense path: their grid is a single cutout and
        the diagnostics genuinely want every pixel.
        """
        mask = exposure.fiber_mask
        if not hasattr(mask, "indices"):
            y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            omega_xyz = pixel_wcs.pixel_to_omega_xyz(x_coords, y_coords)
            omega_t = torch.tensor(
                omega_xyz.reshape(-1, 3), dtype=torch.float32, device=self.device
            )
            mask_t = torch.tensor(np.asarray(mask, dtype=np.float32), device=self.device)
            fiber_indices = torch.nonzero(mask_t.flatten() > 0, as_tuple=True)[0]
            if self.apply_spatial_psf and psf_kernel is not None:
                aperture_weights = convolve_flat_image(mask_t.flatten(), height, width, psf_kernel)
                aperture_support = torch.nonzero(aperture_weights > 0, as_tuple=True)[0]
            else:
                aperture_weights = mask_t.flatten()
                aperture_support = fiber_indices
            return omega_t, fiber_indices, aperture_weights, aperture_support

        rows, cols = mask.rowcol()
        if rows.size == 0:
            raise ValueError("Fiber aperture covers no pixels of the model grid.")
        omega_xyz = pixel_wcs.pixel_to_omega_xyz(
            cols.reshape(-1, 1).astype(np.float64), rows.reshape(-1, 1).astype(np.float64)
        )
        omega_t = torch.tensor(
            np.asarray(omega_xyz, dtype=np.float64).reshape(-1, 3),
            dtype=torch.float32, device=self.device,
        )
        # omega_t is indexed positionally by aperture pixel, so the fiber
        # indices are the identity over the covered pixels.
        fiber_indices = torch.arange(int(rows.size), dtype=torch.long, device=self.device)
        aperture_weights = torch.tensor(
            np.asarray(mask.weights, dtype=np.float32), device=self.device
        )
        return omega_t, fiber_indices, aperture_weights, fiber_indices

    def _build_exposure_runtime(
        self, *, lambda_step: float, psf_truncate: float, include_diagnostics: bool,
    ) -> Dict[int, ExposureRuntime]:
        runtimes: Dict[int, ExposureRuntime] = {}
        for exp_id, exposure in self.exposures.items():
            renderer_bandpasses = exposure.bandpasses or self.renderer_bandpasses
            renderer = SpectralRenderer(renderer_bandpasses, lambda_step, self.device)
            renderer.default_normalize = True
            psf_kernel = None
            if self.apply_spatial_psf and exposure.psf_fwhm_arcsec is not None:
                psf_kernel = gaussian_psf_kernel(
                    float(exposure.psf_fwhm_arcsec), float(exposure.pixel_scale_arcsec), self.device,
                    truncate=psf_truncate,
                )

            if exp_id in self.image_exposure_ids:
                if exposure.image_height is None or exposure.image_width is None:
                    raise ValueError(f"Image exposure {exp_id} must define image_height and image_width.")
                height = int(exposure.image_height)
                width = int(exposure.image_width)
            else:
                height, width = exposure.fiber_mask.shape
            omega_t: torch.Tensor | None = None
            fiber_indices: torch.Tensor | None = None
            aperture_weights: torch.Tensor | None = None
            aperture_support: torch.Tensor | None = None
            if include_diagnostics:
                if self.psf_model != "gaussian" and exp_id in self.image_exposure_ids:
                    raise NotImplementedError(
                        "Mask-aperture image diagnostics do not support PSF mixtures; "
                        "use the analytic deterministic image renderer."
                    )
                pixel_wcs = exposure.wcs
                if exp_id in self.image_exposure_ids and exposure.obs_wcs is not None:
                    pixel_wcs = exposure.obs_wcs
                omega_t, fiber_indices, aperture_weights, aperture_support = (
                    self._fiber_geometry(exposure, pixel_wcs, height, width, psf_kernel)
                )

            spec_projector = None
            if exp_id in self.spec_exposure_ids:
                lsf_fwhm = self._deterministic_lsf_fwhm_value(exposure.lsf_fwhm_angstrom)
                spec_projector = renderer.build_spectrum_projector(
                    exposure.lam_spec, None, lsf_fwhm, lam_edges=exposure.lam_edges,
                )
                if spec_projector is None:
                    raise ValueError(f"Spectrum projector could not be built for exposure {exp_id}.")

            runtimes[exp_id] = ExposureRuntime(
                exposure=exposure, renderer=renderer, psf_kernel=psf_kernel, spec_projector=spec_projector,
                omega_xyz=omega_t, fiber_indices=fiber_indices, aperture_weights=aperture_weights,
                aperture_support=aperture_support, height=int(height), width=int(width),
            )
        return runtimes

    def _global_indices_for(self, exposure_id: int, kind: DatumKind) -> torch.Tensor:
        return self.global_indices_by_exposure_kind.get(
            (int(exposure_id), int(kind)), torch.empty((0,), dtype=torch.long),
        )

    def _field_overlap_training_bands(self) -> set[str]:
        if self.training_bands:
            train_set = set(self.training_bands)
        else:
            available_bands = sorted(set(self.image_band_by_exposure.values()))
            if self.partial_holdout_enabled:
                train_set = set(available_bands)
            else:
                train_set = {
                    band for band in available_bands if band != self.held_out_band
                }
        if self.partial_holdout_enabled and self.held_out_band not in train_set:
            raise ValueError(
                "Partial holdout is enabled, but held_out_band is not in training_bands. "
                f"Add {self.held_out_band!r} to physics.training_bands."
            )
        return train_set

    @staticmethod
    def _numpy_indices(indices: torch.Tensor | None) -> np.ndarray:
        if indices is None or indices.numel() == 0:
            return np.empty((0,), dtype=np.int64)
        values = indices.detach().to(dtype=torch.long, device="cpu").numpy()
        if values.size < 2 or bool(np.all(values[1:] > values[:-1])):
            return values.astype(np.int64, copy=False)
        return np.unique(values).astype(np.int64, copy=False)

    def _field_overlap_state(
        self,
        exposure_id: int,
        train_set: set[str],
        *,
        excluded_by_exposure: Mapping[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        exposure_id = int(exposure_id)
        exposure = self.exposures[exposure_id]
        width = getattr(exposure, "image_width", None)
        height = getattr(exposure, "image_height", None)
        if width is None or height is None:
            raise ValueError(
                f"Image exposure {exposure_id} is missing image dimensions."
            )
        global_indices = self._global_indices_for(
            exposure_id, DatumKind.IMAGE_PIXEL
        )
        valid_local = self._numpy_indices(
            self._gather_named_rows(
                "datum_index",
                global_indices,
                device=torch.device("cpu"),
                dtype=torch.long,
            )
        )
        excluded_map = (
            self.image_excluded_local_indices_by_exposure
            if excluded_by_exposure is None
            else excluded_by_exposure
        )
        excluded = self._numpy_indices(excluded_map.get(exposure_id))
        random_holdout = self._numpy_indices(
            self.image_random_holdout_indices_by_exposure.get(exposure_id)
        )
        held_local = np.union1d(excluded, random_holdout).astype(
            np.int64, copy=False
        )
        exposure_held = (
            exposure_id in self.exclude_image_exposure_ids
            or self.image_band_by_exposure[exposure_id] not in train_set
        )
        absent = (self.image_pixel_mask_absent_by_exposure or {}).get(exposure_id)
        if absent is not None and absent.size and not exposure_held:
            valid_local = valid_local[
                ~np.isin(valid_local, absent, assume_unique=True)
            ]
        return {
            "width": int(width),
            "height": int(height),
            "valid_local": valid_local,
            "held_local": held_local,
            "exposure_held": exposure_held,
        }

    def _validate_field_overlap_pair(
        self,
        pair: Mapping[str, Any],
        pair_index: int,
        identity_by_exposure: dict[int, tuple[str, str]],
        exposure_by_label: dict[str, int],
    ) -> None:
        band = str(pair["band"])
        n_rows = int(pair["n_rows"])
        n_cols = int(pair["n_cols"])
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError(
                f"field overlap pair {pair_index} has a non-positive rectangle."
            )
        for side in ("lower", "upper"):
            exposure_id = int(pair[f"{side}_exposure_id"])
            label = str(pair[f"{side}_label"])
            if not label:
                raise ValueError(
                    f"field overlap pair {pair_index} has an empty {side}_label."
                )
            identity = (label, band)
            previous = identity_by_exposure.setdefault(exposure_id, identity)
            if previous != identity:
                raise ValueError(
                    f"Field overlap table gives inconsistent identities for exposure "
                    f"{exposure_id}: {previous!r} versus {identity!r}."
                )
            previous_exposure_id = exposure_by_label.setdefault(label, exposure_id)
            if previous_exposure_id != exposure_id:
                raise ValueError(
                    f"Field overlap label {label!r} maps to both exposure "
                    f"{previous_exposure_id} and {exposure_id}."
                )
            store_band = self.image_band_by_exposure.get(exposure_id)
            if store_band is None or str(store_band) != band:
                raise RuntimeError(
                    f"Manifest/store exposure order drift: overlap-table entry "
                    f"{exposure_id} ({label!r}, band {band!r}) maps to store band "
                    f"{store_band!r}. Refusing to deduplicate by a misaligned id."
                )
            exposure = self.exposures[exposure_id]
            height = getattr(exposure, "image_height", None)
            width = getattr(exposure, "image_width", None)
            y0 = int(pair[f"{side}_y0"])
            x0 = int(pair[f"{side}_x0"])
            if (
                height is None
                or width is None
                or y0 < 0
                or x0 < 0
                or y0 + n_rows > int(height)
                or x0 + n_cols > int(width)
            ):
                raise ValueError(
                    f"field overlap pair {pair_index} rectangle falls outside "
                    f"exposure {exposure_id}."
                )
        if int(pair["lower_exposure_id"]) == int(pair["upper_exposure_id"]):
            raise ValueError(
                f"field overlap pair {pair_index} references the same exposure twice."
            )

    def _apply_field_overlap_exclusions(
        self, table_path_value: object,
    ) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
        from ..data.field_overlaps import (
            FIELD_ROW_SPACING,
            OVERLAP_ROWS,
            resolve_overlap_drops,
        )

        table_path = Path(str(table_path_value))
        table_bytes = table_path.read_bytes()
        table = json.loads(table_bytes)
        if not isinstance(table, dict):
            raise ValueError(f"Field overlap table {table_path} is not a JSON object.")
        if table.get("schema") != 1:
            raise ValueError(
                f"Field overlap table {table_path} has unsupported schema "
                f"{table.get('schema')!r}; expected 1."
            )
        if int(table.get("frame_row_spacing", -1)) != FIELD_ROW_SPACING:
            raise ValueError(
                f"Field overlap table {table_path} has frame_row_spacing="
                f"{table.get('frame_row_spacing')!r}; expected {FIELD_ROW_SPACING}."
            )
        if int(table.get("overlap_rows", -1)) != OVERLAP_ROWS:
            raise ValueError(
                f"Field overlap table {table_path} has overlap_rows="
                f"{table.get('overlap_rows')!r}; expected {OVERLAP_ROWS}."
            )
        raw_pairs = table.get("pairs")
        if not isinstance(raw_pairs, list):
            raise ValueError(f"Field overlap table {table_path} has no pairs list.")
        pairs: list[dict[str, Any]] = []
        identity_by_exposure: dict[int, tuple[str, str]] = {}
        exposure_by_label: dict[str, int] = {}
        for pair_index, raw_pair in enumerate(raw_pairs):
            if not isinstance(raw_pair, dict):
                raise ValueError(
                    f"Field overlap table pair {pair_index} is not a JSON object."
                )
            pair = dict(raw_pair)
            self._validate_field_overlap_pair(
                pair, pair_index, identity_by_exposure, exposure_by_label
            )
            pairs.append(pair)

        train_set = self._field_overlap_training_bands()
        drop_parts: dict[int, list[np.ndarray]] = {}
        branch_totals = {
            "trainable_trainable": 0,
            "trainable_lower_held_upper": 0,
            "held_lower_trainable_upper": 0,
            "anything_else": 0,
        }
        state_cache: dict[int, dict[str, Any]] = {}

        def state(exposure_id: int) -> dict[str, Any]:
            if exposure_id not in state_cache:
                state_cache[exposure_id] = self._field_overlap_state(exposure_id, train_set)
            return state_cache[exposure_id]

        for pair in pairs:
            lower_id = int(pair["lower_exposure_id"])
            upper_id = int(pair["upper_exposure_id"])
            states = {lower_id: state(lower_id), upper_id: state(upper_id)}
            pair_drops, pair_summary = resolve_overlap_drops([pair], states)
            for branch in branch_totals:
                branch_totals[branch] += int(pair_summary[branch])
            for exposure_id, local_indices in pair_drops.items():
                drop_parts.setdefault(int(exposure_id), []).append(local_indices)

        drops_by_exposure = {
            exposure_id: np.unique(np.concatenate(parts)).astype(
                np.int64, copy=False
            )
            for exposure_id, parts in sorted(drop_parts.items())
        }
        original_exclusions = {
            exposure_id: self.image_excluded_local_indices_by_exposure[exposure_id]
            for exposure_id in identity_by_exposure
            if exposure_id in self.image_excluded_local_indices_by_exposure
        }
        for exposure_id, drops in drops_by_exposure.items():
            existing = self.image_excluded_local_indices_by_exposure.get(
                exposure_id
            )
            drop_tensor = torch.from_numpy(drops)
            if existing is None or existing.numel() == 0:
                merged = drop_tensor
            else:
                merged = torch.unique(
                    torch.cat(
                        (
                            existing.detach().to(dtype=torch.long, device="cpu"),
                            drop_tensor,
                        )
                    ),
                    sorted=True,
                )
            self.image_excluded_local_indices_by_exposure[exposure_id] = merged

        table_sha256 = hashlib.sha256(table_bytes).hexdigest()
        self.field_overlap_summary = {
            "table_sha256": table_sha256,
            "pair_count": len(pairs),
            "pixel_pairs_per_rule_branch": branch_totals,
            "dropped_pixels_by_exposure": {
                exposure_id: int(drops.size)
                for exposure_id, drops in sorted(drops_by_exposure.items())
            },
        }
        if self.is_primary_rank:
            n_pixel_pairs = sum(branch_totals.values())
            n_dropped = sum(
                self.field_overlap_summary["dropped_pixels_by_exposure"].values()
            )
            print(
                f"[field-overlaps] pairs={len(pairs)} pixel_pairs={n_pixel_pairs} "
                f"dropped={n_dropped} rules={branch_totals} "
                f"table_sha256={table_sha256}",
                flush=True,
            )
        return pairs, original_exclusions

    def _check_field_overlap_postcondition(
        self,
        pairs: Sequence[Mapping[str, Any]],
        original_exclusions: Mapping[int, torch.Tensor],
    ) -> None:
        from ..data.field_overlaps import _local_indices_for_pair, _membership

        train_set = self._field_overlap_training_bands()
        state_cache: dict[int, dict[str, Any]] = {}
        train_local_cache: dict[int, np.ndarray] = {}

        def state(exposure_id: int) -> dict[str, Any]:
            if exposure_id not in state_cache:
                state_cache[exposure_id] = self._field_overlap_state(
                    exposure_id, train_set, excluded_by_exposure=original_exclusions,
                )
            return state_cache[exposure_id]

        def train_local_for(exposure_id: int) -> np.ndarray:
            if exposure_id not in train_local_cache:
                train_global = self.image_train_indices_by_exposure.get(
                    exposure_id, torch.empty((0,), dtype=torch.long)
                )
                train_local_cache[exposure_id] = self._numpy_indices(
                    self._gather_named_rows(
                        "datum_index",
                        train_global,
                        device=torch.device("cpu"),
                        dtype=torch.long,
                    )
                )
            return train_local_cache[exposure_id]

        for pair_index, pair in enumerate(pairs):
            lower_id = int(pair["lower_exposure_id"])
            upper_id = int(pair["upper_exposure_id"])
            states = {lower_id: state(lower_id), upper_id: state(upper_id)}
            local_by_exposure = dict(
                zip(
                    (lower_id, upper_id),
                    _local_indices_for_pair(
                        pair,
                        int(states[lower_id]["width"]),
                        int(states[upper_id]["width"]),
                    ),
                )
            )
            trainable_by_exposure: dict[int, np.ndarray] = {}
            held_by_exposure: dict[int, np.ndarray] = {}
            for exposure_id, local_indices in local_by_exposure.items():
                trainable_by_exposure[exposure_id] = _membership(
                    train_local_for(exposure_id), local_indices
                )
                present = _membership(
                    states[exposure_id]["valid_local"], local_indices
                )
                held_by_exposure[exposure_id] = present & (
                    bool(states[exposure_id]["exposure_held"])
                    | _membership(
                        states[exposure_id]["held_local"], local_indices
                    )
                )

            lower_trainable = trainable_by_exposure[lower_id]
            upper_trainable = trainable_by_exposure[upper_id]
            n_duplicate = int(
                np.count_nonzero(lower_trainable & upper_trainable)
            )
            n_leaked = int(
                np.count_nonzero(
                    (lower_trainable & held_by_exposure[upper_id])
                    | (upper_trainable & held_by_exposure[lower_id])
                )
            )
            if n_duplicate or n_leaked:
                raise RuntimeError(
                    "Field-overlap training post-condition failed for pair "
                    f"{pair_index} ({lower_id}->{upper_id}): "
                    f"duplicate_training={n_duplicate}, "
                    f"training_with_held_twin={n_leaked}."
                )

    def _load_image_pixel_mask(self, manifest_path_value: object) -> Dict[int, np.ndarray]:
        """Absent pixels per image exposure: sidecar-masked, not excluded, not random holdout."""
        from ..data.fpm_masks import SIDECAR_KIND, SIDECAR_SCHEMA

        manifest_path = Path(str(manifest_path_value))
        manifest_bytes = manifest_path.read_bytes()
        sidecar = json.loads(manifest_bytes)
        if (
            not isinstance(sidecar, dict)
            or sidecar.get("schema") != SIDECAR_SCHEMA
            or sidecar.get("kind") != SIDECAR_KIND
        ):
            raise ValueError(
                f"Image pixel mask {manifest_path} is not a schema-{SIDECAR_SCHEMA} "
                f"{SIDECAR_KIND} manifest."
            )
        arrays_path = manifest_path.parent / str(sidecar["arrays"])
        arrays_sha256 = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        if arrays_sha256 != sidecar["arrays_sha256"]:
            raise ValueError(
                f"Image pixel mask arrays {arrays_path} do not match the manifest sha256."
            )
        band_by_exposure = {
            int(exp_id): str(band)
            for exp_id, band in getattr(self.data, "image_band_by_exposure", {}).items()
        }
        rows = sidecar["exposures"]
        exposure_ids = [int(row["exposure_id"]) for row in rows]
        if len(set(exposure_ids)) != len(exposure_ids) or set(exposure_ids) != set(band_by_exposure):
            raise ValueError(
                f"Image pixel mask {manifest_path} does not list exactly the store's "
                "image exposures."
            )
        held_maps = [
            getattr(self.data, name, {}) or {}
            for name in (
                "image_excluded_local_indices_by_exposure",
                "image_random_holdout_indices_by_exposure",
            )
        ]
        absent_by_exposure: Dict[int, np.ndarray] = {}
        self._image_pixel_mask_store_pixels: Dict[int, int] = {}
        n_masked = 0
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {f"e{exp_id}" for exp_id in exposure_ids}:
                raise ValueError(
                    f"Image pixel mask arrays {arrays_path} do not match the manifest exposures."
                )
            for row in rows:
                exposure_id = int(row["exposure_id"])
                if str(row["band"]) != band_by_exposure[exposure_id]:
                    raise RuntimeError(
                        f"Manifest/store exposure order drift: image pixel mask entry "
                        f"{exposure_id} ({row.get('label')!r}, band {row['band']!r}) maps to "
                        f"store band {band_by_exposure[exposure_id]!r}."
                    )
                exposure = self.data.exposures[exposure_id]
                height, width = int(exposure.image_height), int(exposure.image_width)
                if (int(row["height"]), int(row["width"])) != (height, width):
                    raise ValueError(
                        f"Image pixel mask entry {exposure_id} is {row['height']}x{row['width']}; "
                        f"the store exposure is {height}x{width}."
                    )
                masked = np.array(arrays[f"e{exposure_id}"])
                if (
                    masked.dtype != np.int64
                    or masked.ndim != 1
                    or masked.size != int(row["n_masked"])
                    or (masked.size and (masked[0] < 0 or masked[-1] >= height * width))
                    or (masked.size > 1 and np.any(masked[1:] <= masked[:-1]))
                ):
                    raise ValueError(
                        f"Image pixel mask entry {exposure_id} is not {row['n_masked']} sorted, "
                        "unique in-image int64 pixel indices."
                    )
                held = np.union1d(
                    *(
                        np.asarray(held_map.get(exposure_id, ()), dtype=np.int64)
                        for held_map in held_maps
                    )
                )
                absent_by_exposure[exposure_id] = (
                    masked[~np.isin(masked, held)] if held.size else masked
                )
                self._image_pixel_mask_store_pixels[exposure_id] = int(row["n_store_pixels"])
                n_masked += int(masked.size)
        self.image_pixel_mask_summary = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "arrays_sha256": arrays_sha256,
            "planes": [str(plane) for plane in sidecar["planes"]],
            "masked_pixels": n_masked,
        }
        return absent_by_exposure

    def _check_image_pixel_mask_store(self) -> None:
        for exposure_id, n_store_pixels in sorted(self._image_pixel_mask_store_pixels.items()):
            n_rows = int(self._global_indices_for(exposure_id, DatumKind.IMAGE_PIXEL).numel())
            if n_rows != n_store_pixels:
                raise RuntimeError(
                    f"Image pixel mask was built for {n_store_pixels} pixels of exposure "
                    f"{exposure_id}; the store has {n_rows}. Rebuild it for this store."
                )

    def _image_pixel_absent_rows(
        self, rows: torch.Tensor, exposure_ids: set[int],
    ) -> torch.Tensor:
        """Mask of rows that are absent (fpM-masked) pixels of the given image exposures."""
        cpu = torch.device("cpu")
        row_exposure = self._gather_named_rows("exposure_id", rows, device=cpu, dtype=torch.long)
        row_kind = self._gather_named_rows("datum_kind", rows, device=cpu, dtype=torch.long)
        local = self._gather_named_rows("datum_index", rows, device=cpu, dtype=torch.long)
        image_rows = row_kind == int(DatumKind.IMAGE_PIXEL)
        absent_rows = torch.zeros(row_exposure.shape, dtype=torch.bool)
        for exposure_id, absent in sorted(self.image_pixel_mask_absent_by_exposure.items()):
            if exposure_id not in exposure_ids or absent.size == 0:
                continue
            on_exposure = image_rows & (row_exposure == exposure_id)
            absent_rows[on_exposure] = torch.from_numpy(
                np.isin(local[on_exposure].numpy(), absent)
            )
        return absent_rows

    def _ensure_diagnostic_runtime(self, exposure_id: int) -> ExposureRuntime:
        if self.psf_model != "gaussian" and int(exposure_id) in self.image_exposure_ids:
            raise NotImplementedError(
                "Mask-aperture image diagnostics do not support PSF mixtures; "
                "use the analytic deterministic image renderer."
            )
        runtime = self.runtime_by_exposure[int(exposure_id)]
        if runtime.omega_xyz is not None and runtime.fiber_indices is not None:
            return runtime

        exposure = runtime.exposure
        height = int(runtime.height)
        width = int(runtime.width)
        pixel_wcs = exposure.wcs
        if int(exposure_id) in self.image_exposure_ids and exposure.obs_wcs is not None:
            pixel_wcs = exposure.obs_wcs
        (
            runtime.omega_xyz, runtime.fiber_indices, runtime.aperture_weights,
            runtime.aperture_support,
        ) = self._fiber_geometry(exposure, pixel_wcs, height, width, runtime.psf_kernel)
        return runtime

    def _fitted_image_rows(
        self, rows: torch.Tensor, fitted_image_ids: set[int],
    ) -> torch.Tensor:
        """Mask of rows that are fitted image pixels (no held or excluded pixel)."""
        row_exposure = self._gather_named_rows(
            "exposure_id", rows, device=torch.device("cpu"), dtype=torch.long,
        )
        keep = torch.isin(row_exposure, torch.tensor(sorted(fitted_image_ids), dtype=torch.long))
        local = self._gather_named_rows(
            "datum_index", rows, device=torch.device("cpu"), dtype=torch.long,
        )
        for source in ("image_random_holdout_indices_by_exposure", "image_excluded_local_indices_by_exposure"):
            for exp_id, held_local in (getattr(self.data, source, {}) or {}).items():
                held = torch.as_tensor(held_local, dtype=torch.long)
                if held.numel() == 0:
                    continue
                on_exposure = keep & (row_exposure == int(exp_id))
                keep[on_exposure] = ~torch.isin(local[on_exposure], held)
        if not bool(keep.any()):
            raise RuntimeError("No fitted image rows remain for the lifecycle statistics.")
        return keep

    def _objective_denominator(self) -> int:
        """Datum count N of the objective (sum NLL + negative log prior) / N."""
        if self.objective_denominator is not None:
            return self.objective_denominator
        return int(self.image_train_indices.numel()) + int(self.spec_train_indices.numel())

    def _exclude_spectra_by_label(self) -> None:
        if not self.exclude_spectrum_labels:
            return
        label_by_id = {
            int(exp_id): str(label)
            for exp_id, label in (getattr(self.data, "spec_label_by_exposure", None) or {}).items()
        }
        unknown = sorted(set(self.exclude_spectrum_labels) - set(label_by_id.values()))
        if unknown:
            raise ValueError(
                "trainer.exclude_spectrum_labels: "
                f"{unknown} are not labels of spectra loaded in the store."
            )
        excluded_labels = set(self.exclude_spectrum_labels)
        excluded_ids = {exp_id for exp_id, label in label_by_id.items() if label in excluded_labels}
        self.excluded_spectrum_ids = tuple(sorted(excluded_ids))
        self.spec_exposure_ids = [
            exp_id for exp_id in self.spec_exposure_ids if exp_id not in excluded_ids
        ]

    def _init_training_indices(self) -> None:
        self._exclude_spectra_by_label()
        available_bands = sorted(set(self.image_band_by_exposure.values()))
        if not self.training_bands:
            if self.partial_holdout_enabled:
                self.training_bands = list(available_bands)
            else:
                self.training_bands = [band for band in available_bands if band != self.held_out_band]
        train_set = set(self.training_bands)
        if self.partial_holdout_enabled and self.held_out_band not in train_set:
            raise ValueError(
                "Partial holdout is enabled, but held_out_band is not in training_bands. "
                f"Add {self.held_out_band!r} to physics.training_bands."
            )
        unknown_excluded = self.exclude_image_exposure_ids - {
            int(e) for e in self.image_band_by_exposure
        }
        if unknown_excluded:
            raise ValueError(
                "exclude_image_exposure_ids contains ids that are not image "
                f"exposures in the store: {sorted(unknown_excluded)}. Known image "
                f"exposure ids: {sorted(int(e) for e in self.image_band_by_exposure)}."
            )
        for excluded_id in sorted(self.exclude_image_exposure_ids):
            excluded_band = str(self.image_band_by_exposure[excluded_id])
            if excluded_band not in train_set and self.is_primary_rank:
                print(
                    f"Warning: excluded exposure {excluded_id} is in band "
                    f"{excluded_band!r}, which is not a training band; the "
                    "exclusion is a no-op and the exposure is NOT a trained "
                    "control either.",
                    flush=True,
                )
        image_indices = []
        image_holdout_parts_by_band: Dict[str, list[torch.Tensor]] = {}
        self.image_train_indices_by_exposure = {}
        self.image_train_local_positions_by_exposure = {}
        self.image_train_counts_by_exposure: Dict[int, int] = {}
        masked_rows_by_exposure: Dict[int, int] = {}
        for exp_id, band in sorted(self.image_band_by_exposure.items()):
            if int(exp_id) in self.exclude_image_exposure_ids:
                continue
            if band in train_set:
                idx = self._global_indices_for(exp_id, DatumKind.IMAGE_PIXEL)
                excluded_local = self.image_excluded_local_indices_by_exposure.get(int(exp_id), None)
                random_local = self.image_random_holdout_indices_by_exposure.get(
                    int(exp_id), None
                )
                if (
                    idx.numel() > 0
                    and (
                        (excluded_local is not None and excluded_local.numel() > 0)
                        or (random_local is not None and random_local.numel() > 0)
                    )
                ):
                    local = self._gather_named_rows(
                        "datum_index", idx, device=torch.device("cpu"), dtype=torch.long
                    )
                    if random_local is not None and random_local.numel() > 0:
                        random_mask = self._local_membership_mask(
                            local,
                            random_local,
                            int(self.exposures[int(exp_id)].image_height)
                            * int(self.exposures[int(exp_id)].image_width),
                        )
                        held_out = idx[random_mask]
                        if held_out.numel() > 0:
                            image_holdout_parts_by_band.setdefault(
                                str(band), []
                            ).append(held_out)
                    if excluded_local is not None and excluded_local.numel() > 0:
                        excluded_mask = self._local_membership_mask(
                            local,
                            excluded_local,
                            int(self.exposures[int(exp_id)].image_height)
                            * int(self.exposures[int(exp_id)].image_width),
                        )
                        idx = idx[~excluded_mask]
                absent_local = (self.image_pixel_mask_absent_by_exposure or {}).get(int(exp_id))
                if absent_local is not None and absent_local.size:
                    absent_mask = self._local_membership_mask(
                        self._gather_named_rows(
                            "datum_index", idx, device=torch.device("cpu"), dtype=torch.long
                        ),
                        torch.from_numpy(absent_local),
                        int(self.exposures[int(exp_id)].image_height)
                        * int(self.exposures[int(exp_id)].image_width),
                    )
                    if int(absent_mask.sum()) != int(absent_local.size):
                        raise RuntimeError(
                            f"Image pixel mask: exposure {exp_id} has {absent_local.size} masked "
                            f"pixels but {int(absent_mask.sum())} of them are trainable rows."
                        )
                    idx = idx[~absent_mask]
                    masked_rows_by_exposure[int(exp_id)] = int(absent_local.size)
                if idx.numel() > 0:
                    image_indices.append(idx)
                    self.image_train_indices_by_exposure[int(exp_id)] = idx
        if image_indices:
            self.image_train_indices = torch.cat(image_indices, dim=0)
        else:
            self.image_train_indices = torch.empty((0,), dtype=torch.long)
        if self.image_pixel_mask_absent_by_exposure is not None:
            self.image_pixel_mask_summary["removed_rows_by_exposure"] = masked_rows_by_exposure
            self.image_pixel_mask_summary["removed_rows"] = sum(masked_rows_by_exposure.values())
            if self.is_primary_rank:
                print(
                    f"[fpm-mask] planes={self.image_pixel_mask_summary['planes']} "
                    f"removed_rows={self.image_pixel_mask_summary['removed_rows']} "
                    f"image_population={int(self.image_train_indices.numel())} "
                    f"manifest_sha256={self.image_pixel_mask_summary['manifest_sha256']}",
                    flush=True,
                )
        if self.exclude_image_exposure_ids and self.is_primary_rank:
            excluded_bands = {
                int(e): str(self.image_band_by_exposure[e])
                for e in sorted(self.exclude_image_exposure_ids)
            }
            print(
                f"[holdout] {len(excluded_bands)} image exposures excluded from "
                f"training (id: band): {excluded_bands}",
                flush=True,
            )
        self.image_holdout_indices_by_band = {
            band: self._fixed_index_subset(
                torch.cat(parts, dim=0),
                self.fixed_holdout_max_image_per_band,
            )
            for band, parts in sorted(image_holdout_parts_by_band.items())
            if parts
        }

        spec_indices = []
        spec_holdout_indices = []
        self.spec_holdout_indices_by_exposure: Dict[int, torch.Tensor] = {}
        source_gate = getattr(self.data.batch, "source_gate", None)
        source_gate_cpu = (
            source_gate.detach().to(dtype=torch.float32, device="cpu")
            if source_gate is not None
            else None
        )
        for exp_id in self.spec_exposure_ids:
            idx = self._global_indices_for(exp_id, DatumKind.SPECTRAL_BIN)
            if source_gate_cpu is not None and idx.numel() > 0:
                gate_idx = idx.to(dtype=torch.long, device="cpu")
                gate = torch.index_select(source_gate_cpu, 0, gate_idx)
                idx = idx[gate > 0.0]
            if self.spectral_holdout_fraction > 0.0 and idx.numel() > 0:
                local = self._gather_named_rows(
                    "datum_index",
                    idx,
                    device=torch.device("cpu"),
                    dtype=torch.long,
                )
                exposure = self.exposures[int(exp_id)]
                n_local = int(np.asarray(exposure.lam_spec).size)
                held_out_local = torch.as_tensor(
                    random_holdout_local_indices(
                        n_local,
                        self.spectral_holdout_fraction,
                        self.spectral_holdout_seed,
                        salt=int(exp_id),
                    ),
                    dtype=torch.long,
                )
                is_holdout = self._local_membership_mask(
                    local, held_out_local, n_local
                )
                held_out = idx[is_holdout]
                if held_out.numel() > 0:
                    self.spec_holdout_indices_by_exposure[int(exp_id)] = held_out
                    spec_holdout_indices.append(held_out)
                idx = idx[~is_holdout]
            if idx.numel() > 0:
                spec_indices.append(idx)
        if spec_indices:
            self.spec_train_indices = torch.cat(spec_indices, dim=0)
        else:
            self.spec_train_indices = torch.empty((0,), dtype=torch.long)
        if spec_holdout_indices:
            self.spec_holdout_indices = self._fixed_index_subset(
                torch.cat(spec_holdout_indices, dim=0),
                self.fixed_holdout_max_spec,
            )
        else:
            self.spec_holdout_indices = torch.empty((0,), dtype=torch.long)

    @staticmethod
    def _fixed_index_subset(indices: torch.Tensor, max_count: int) -> torch.Tensor:
        values = indices.detach().to(dtype=torch.long, device="cpu")
        if max_count <= 0 or values.numel() <= max_count:
            return values
        positions = torch.linspace(
            0,
            int(values.numel()) - 1,
            steps=int(max_count),
            dtype=torch.float64,
        ).round().to(dtype=torch.long)
        return values[positions]

    @staticmethod
    def _local_membership_mask(
        local_indices: torch.Tensor,
        selected_local_indices: torch.Tensor,
        n_local: int,
    ) -> torch.Tensor:
        local = local_indices.detach().to(dtype=torch.long, device="cpu")
        selected = selected_local_indices.detach().to(
            dtype=torch.long, device="cpu"
        )
        if n_local < 0:
            raise ValueError("n_local must be non-negative.")
        if local.numel() > 0 and (
            int(local.min()) < 0 or int(local.max()) >= int(n_local)
        ):
            raise ValueError("Datum-local indices are out of bounds.")
        if selected.numel() > 0 and (
            int(selected.min()) < 0 or int(selected.max()) >= int(n_local)
        ):
            raise ValueError("Selected local indices are out of bounds.")
        lookup = torch.zeros(int(n_local), dtype=torch.bool)
        lookup[selected] = True
        return lookup[local]

    def _init_image_importance_sampling(self) -> None:
        self.image_importance_sampling_enabled = False
        self.image_importance_exposure_ids = []
        self.image_importance_exposure_probs = None
        self.image_importance_exposure_cdf = None
        self.image_importance_local_probs_by_exposure = {}
        self.image_importance_local_cdf_by_exposure = {}
        self.image_importance_correction_by_exposure = {}
        self.image_importance_generator = None
        self._image_importance_cdf_matrix = None
        self._image_importance_correction_flat = None
        self._image_importance_index_flat = None
        self._image_importance_local_counts = None
        self._image_importance_offsets = None
        self._pending_image_importance_weights = None
        cfg = self.image_importance_sampling_cfg
        if not cfg.enabled:
            return
        image_train_total = int(self.image_train_indices.numel())
        if image_train_total == 0:
            return

        exposure_ids = sorted(self.image_train_indices_by_exposure.keys())
        if not exposure_ids:
            return
        active_ids = [
            int(exp_id)
            for exp_id in exposure_ids
            if self.image_train_indices_by_exposure[int(exp_id)].numel() > 0
        ]
        if not active_ids:
            return
        # The per-exposure tables live in one buffer each, with every
        # exposure's entry a view into it, so the per-step sampler can search
        # all the CDFs in one batched call and gather every draw's index and
        # weight with one index. The values are unchanged: each exposure writes
        # its own slice of the buffer instead of its own tensor.
        counts = [
            int(self.image_train_indices_by_exposure[exp_id].numel())
            for exp_id in active_ids
        ]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        cdf_matrix = torch.empty((len(active_ids), max(counts)), dtype=torch.float64)
        correction_flat = torch.empty((int(offsets[-1]),), dtype=torch.float64)
        index_flat = torch.empty((int(offsets[-1]),), dtype=torch.long)

        def _build_exposure(slot: int) -> torch.Tensor:
            exp_id = active_ids[slot]
            idx = self.image_train_indices_by_exposure[exp_id]
            _, local_score, local_indices = self._build_image_importance_score_full_for_exposure(
                exp_id, idx,
            )
            probs = self._build_image_importance_probs_from_scores(
                local_indices=local_indices, local_score=local_score,
            )
            n_local = int(local_indices.numel())
            correction = 1.0 / (float(n_local) * probs)
            start, stop = int(offsets[slot]), int(offsets[slot + 1])
            cdf_matrix[slot, :n_local].copy_(self._probs_to_cdf(probs))
            # Past every draw, so padding is never selected and never read.
            cdf_matrix[slot, n_local:].fill_(2.0)
            correction_flat[start:stop].copy_(correction)
            index_flat[start:stop].copy_(idx)
            return probs

        # Exposures are independent: each task reads its own rows out of the
        # store, consumes no RNG and writes its own disjoint slice, and the
        # numpy and torch array work releases the GIL.
        with ThreadPoolExecutor(max_workers=startup_worker_count(len(active_ids))) as pool:
            probs_by_slot = list(pool.map(_build_exposure, range(len(active_ids))))
        for slot, exp_id in enumerate(active_ids):
            start, stop = int(offsets[slot]), int(offsets[slot + 1])
            self.image_importance_local_probs_by_exposure[exp_id] = probs_by_slot[slot]
            self.image_importance_local_cdf_by_exposure[exp_id] = cdf_matrix[slot, : counts[slot]]
            self.image_importance_correction_by_exposure[exp_id] = correction_flat[start:stop]
            self.image_train_indices_by_exposure[exp_id] = index_flat[start:stop]
        self._image_importance_cdf_matrix = cdf_matrix
        self._image_importance_correction_flat = correction_flat
        self._image_importance_index_flat = index_flat
        self._image_importance_local_counts = torch.as_tensor(counts, dtype=torch.long)
        self._image_importance_offsets = torch.as_tensor(offsets[:-1], dtype=torch.long)
        self.image_importance_exposure_ids = sorted(self.image_importance_local_probs_by_exposure.keys())
        exposure_count_tensor = torch.tensor(
            [
                float(self.image_train_indices_by_exposure[int(exp_id)].numel())
                for exp_id in self.image_importance_exposure_ids
            ],
            dtype=torch.float64,
        )
        self.image_importance_exposure_probs = exposure_count_tensor / torch.clamp(
            exposure_count_tensor.sum(), min=1.0,
        )
        self.image_importance_exposure_cdf = self._probs_to_cdf(self.image_importance_exposure_probs)
        generator_seed = int(torch.initial_seed()) + 104729 * int(self.rank)
        self.image_importance_generator = torch.Generator(device="cpu")
        self.image_importance_generator.manual_seed(generator_seed)
        self.image_importance_sampling_enabled = True

    @staticmethod
    def _probs_to_cdf(probs: torch.Tensor) -> torch.Tensor:
        cdf = torch.cumsum(probs.to(dtype=torch.float64, device="cpu"), dim=0)
        if cdf.numel() > 0:
            cdf[-1] = 1.0
        return cdf

    @staticmethod
    def _sample_cdf(cdf: torch.Tensor, count: int, *, generator: torch.Generator) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0,), dtype=torch.long)
        u = torch.rand((int(count),), generator=generator, dtype=cdf.dtype)
        return torch.searchsorted(cdf, u, right=True).clamp(max=cdf.numel() - 1)

    def _build_image_importance_score_full_for_exposure(
        self, exposure_id: int, global_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exposure = self.exposures[int(exposure_id)]
        height = exposure.image_height
        width = exposure.image_width
        if height is None or width is None:
            raise ValueError("Image importance sampling requires image dimensions for every exposure.")
        idx_cpu = global_indices.to(dtype=torch.long, device=self.data.batch.y.device)
        y = torch.index_select(self.data.batch.y, 0, idx_cpu).detach().to(dtype=torch.float32, device="cpu")
        sigma = torch.index_select(self.data.batch.sigma, 0, idx_cpu).detach().to(
            dtype=torch.float32, device="cpu",
        )
        local = torch.index_select(self.data.batch.datum_index, 0, idx_cpu).detach().to(
            dtype=torch.long, device="cpu",
        )
        values = y.numpy()
        sigma_np = sigma.numpy()
        additive_np = None
        if getattr(self.data.batch, "additive", None) is not None:
            additive_np = (
                torch.index_select(self.data.batch.additive, 0, idx_cpu)
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .numpy()
            )
        score_local = self._build_source_proxy_importance_score(values, sigma_np, additive=additive_np)
        score_full = np.zeros((int(height) * int(width),), dtype=np.float32)
        score_full[local.numpy()] = score_local.astype(np.float32, copy=False)
        score_full = np.clip(score_full, 0.0, None)
        score_full = np.where(np.isfinite(score_full), score_full, 0.0)
        score_local = score_full[local.numpy()]

        return (
            torch.tensor(score_full, dtype=torch.float64), torch.tensor(score_local, dtype=torch.float64),
            local,
        )

    @staticmethod
    def _build_source_proxy_importance_score(
        y: np.ndarray, sigma: np.ndarray, *, additive: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(y, dtype=np.float32).reshape(-1)
        sigma_values = np.asarray(sigma, dtype=np.float32).reshape(-1)
        n = min(int(values.size), int(sigma_values.size))
        if additive is not None:
            additive_values = np.asarray(additive, dtype=np.float32).reshape(-1)
            n = min(n, int(additive_values.size))
        else:
            additive_values = None
        if n <= 0:
            return np.empty((0,), dtype=np.float64)
        values = values[:n]
        sigma_values = sigma_values[:n]
        source_proxy = values if additive_values is None else values - additive_values[:n]
        finite = np.isfinite(source_proxy) & np.isfinite(sigma_values) & (sigma_values > 0.0)
        if not np.any(finite):
            return np.zeros((n,), dtype=np.float64)
        background = float(np.median(source_proxy[finite]))
        standardized = np.zeros((n,), dtype=np.float64)
        standardized[finite] = (
            source_proxy[finite].astype(np.float64) - background
        ) / sigma_values[finite].astype(np.float64)
        score = np.clip(standardized, 0.0, None)
        return np.asarray(score, dtype=np.float64)

    def _build_image_importance_probs_from_scores(
        self, *, local_indices: torch.Tensor, local_score: torch.Tensor,
    ) -> torch.Tensor:
        n_local = int(local_indices.numel())
        if n_local <= 0:
            return torch.empty((0,), dtype=torch.float64)
        uniform = np.full((n_local,), 1.0 / float(n_local), dtype=np.float64)
        local_score_np = np.clip(np.asarray(local_score.detach().cpu().numpy(), dtype=np.float64), 0.0, None)
        local_sum = float(np.sum(local_score_np))
        if local_sum > 0.0:
            proposal = local_score_np / local_sum
        else:
            proposal = uniform.copy()
        return self._apply_image_importance_uniform_floor(proposal, uniform=uniform)

    def _apply_image_importance_uniform_floor(
        self, proposal: np.ndarray | torch.Tensor, *, uniform: np.ndarray | None = None,
    ) -> torch.Tensor:
        proposal_np = np.asarray(proposal, dtype=np.float64)
        n_local = int(proposal_np.shape[0])
        if n_local <= 0:
            return torch.empty((0,), dtype=torch.float64)
        if uniform is None:
            uniform = np.full((n_local,), 1.0 / float(n_local), dtype=np.float64)
        proposal_sum = float(np.sum(proposal_np))
        if proposal_sum > 0.0:
            proposal_np = proposal_np / proposal_sum
        else:
            proposal_np = np.asarray(uniform, dtype=np.float64).copy()
        uniform_mass = 1.0 / float(self.image_importance_sampling_cfg.max_importance_weight)
        probs = (1.0 - uniform_mass) * proposal_np + uniform_mass * uniform
        probs = probs / max(float(np.sum(probs)), 1.0e-12)
        return torch.tensor(probs, dtype=torch.float64)

    def _resolve_image_batch_size(self) -> int:
        total = int(self.image_train_indices.numel())
        if total <= 0:
            return 0
        if self.image_batch_size <= 0:
            return total
        return min(int(self.image_batch_size), total)

    def _resolve_effective_global_image_batch_size(self) -> int:
        batch_size = self._resolve_image_batch_size()
        if batch_size <= 0:
            return 0
        if self.distributed_enabled and int(self.world_size) > 1:
            return batch_size * int(self.world_size)
        return batch_size

    def training_image_datum_count(self) -> int:
        return int(self.image_train_indices.numel())

    def image_training_summary(self) -> Dict[str, Any]:
        total_image_datums = int(self.training_image_datum_count())
        return {
            "n_image_datums": total_image_datums,
            "n_materialized_image_datums": int(self.image_train_indices.numel()),
            "n_total_image_datums": total_image_datums,
        }

    def image_sampling_summary(self) -> Dict[str, Any]:
        cfg = self.image_importance_sampling_cfg
        return {
            "importance_sampling_enabled": bool(self.image_importance_sampling_enabled),
            "mode": "materialized", "max_importance_weight": float(cfg.max_importance_weight),
        }

    def recipe_record(self) -> Dict[str, Any]:
        """The objective and data settings a checkpoint was trained under, for resume checks."""
        mask = getattr(self, "image_pixel_mask_summary", None)
        overlap = getattr(self, "field_overlap_summary", None)
        refine = getattr(self, "refine_strategy", None)
        return {
            "schema": 1,
            "spectral_bin_weight": float(self.spectral_bin_weight),
            "objective_population": {
                "image": int(self.image_train_indices.numel()),
                "spectrum": int(self.spec_train_indices.numel()),
            },
            "image_variance": {
                "mode": "data" if self.image_variance_outlier_pull is None else "poisson_current",
                "outlier_pull": self.image_variance_outlier_pull,
            },
            "image_pixel_mask": (
                None if mask is None
                else {key: mask[key] for key in ("manifest_sha256", "arrays_sha256", "planes")}
            ),
            "field_overlap_table_sha256": None if overlap is None else overlap["table_sha256"],
            "lifecycle_statistics": getattr(self, "lifecycle_statistics", None),
            "spectral_calibration": bool(self.spectral_calibration_cfg.enabled),
            "exclude_exposure_labels": list(self.exclude_exposure_labels),
            "exclude_spectrum_labels": list(self.exclude_spectrum_labels),
            "random_holdout": {
                "fraction": float(self.spectral_holdout_fraction),
                "seed": int(self.spectral_holdout_seed),
            },
            "adam_eps": float(self.adam_eps),
            "prune_axis_ratio": None if refine is None else refine.config.prune_axis_ratio,
            "splat_flux_l1": float(self.splat_flux_l1),
            "objective_denominator": int(self._objective_denominator()),
            "prepared_store": self.prepared_store,
            "trainer_config": self._recipe_trainer_cfg,
            "execution_flags": execution_flags(),
        }

    def scientific_modality_summary(self) -> Dict[str, Any]:
        image_population_count = int(self.image_train_indices.numel())
        spectrum_population_count = int(self.spec_train_indices.numel())
        population_count = image_population_count + spectrum_population_count
        if population_count <= 0:
            image_weight = 0.0
            spectrum_weight = 0.0
        else:
            image_weight = image_population_count / population_count
            spectrum_weight = (
                self.spectral_bin_weight * spectrum_population_count / population_count
            )
        return {
            "mode": "full_training_population_mean_nll",
            "image_population_count": image_population_count,
            "spectrum_population_count": spectrum_population_count,
            "spectral_bin_weight": self.spectral_bin_weight,
            "image_weight": image_weight,
            "spectrum_weight": spectrum_weight,
        }

    def _sample_importance_image_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.image_importance_sampling_enabled:
            raise RuntimeError("Importance image sampling is not enabled.")
        if self.image_importance_exposure_cdf is None or self.image_importance_generator is None:
            raise RuntimeError("Importance image sampling state is not initialized.")
        batch_size = self._resolve_image_batch_size()
        if batch_size <= 0:
            return (torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.float32))
        generator = self.image_importance_generator
        exposure_choice = self._sample_cdf(
            self.image_importance_exposure_cdf, batch_size, generator=generator,
        )
        n_exposures = len(self.image_importance_exposure_ids)
        counts = torch.bincount(exposure_choice, minlength=n_exposures)
        # One draw for the whole batch. The per-exposure form drew `count`
        # float64 uniforms per non-empty exposure, in ascending exposure order,
        # from this generator; torch's CPU float64 stream is chunk-invariant,
        # so a single call of the same total length produces that same
        # concatenation and leaves the generator in the same state.
        draws = torch.rand((batch_size,), generator=generator, dtype=torch.float64)
        exposure_of_draw = torch.repeat_interleave(
            torch.arange(n_exposures, dtype=torch.long), counts
        )
        slot_of_draw = torch.arange(batch_size, dtype=torch.long) - (
            torch.cumsum(counts, dim=0) - counts
        )[exposure_of_draw]
        # searchsorted needs a rectangular right-hand side to search every
        # exposure's CDF in one call. The unused slots hold zero and are never
        # gathered back; the CDFs themselves are untouched, so each draw is
        # compared against exactly the float64 values it was compared against
        # before, and lands on the same local index.
        padded = torch.zeros((n_exposures, int(counts.max())), dtype=torch.float64)
        padded[exposure_of_draw, slot_of_draw] = draws
        local_choice = torch.minimum(
            torch.searchsorted(self._image_importance_cdf_matrix, padded, right=True),
            (self._image_importance_local_counts - 1).unsqueeze(1),
        )[exposure_of_draw, slot_of_draw]
        flat = self._image_importance_offsets[exposure_of_draw] + local_choice
        image_indices = self._image_importance_index_flat[flat]
        image_weights = self._image_importance_correction_flat[flat].to(dtype=torch.float32)
        return image_indices, image_weights

    def _init_dataloaders(self) -> None:
        """Construct DataLoaders that yield batches of global datum indices."""

        def _validate_distributed_loader_size(indices: torch.Tensor, *, label: str) -> None:
            if not self.distributed_enabled or self.dataloader_drop_last:
                return
            n = int(indices.numel())
            if n == 0:
                return
            if n % int(self.world_size) != 0:
                raise ValueError(
                    f"Distributed training with trainer.dataloader_drop_last=false requires "
                    f"{label} count ({n}) to be divisible by world_size ({self.world_size}) "
                    "to avoid sampler padding changing the training objective."
                )

        def _make_loader(indices: torch.Tensor, batch_size: int) -> DataLoader | None:
            n = int(indices.numel())
            if n == 0:
                return None
            bs = n if batch_size <= 0 else min(int(batch_size), n)
            dataset = _IndexDataset(indices)
            num_workers = max(0, int(self.dataloader_num_workers))
            persistent_workers = bool(num_workers > 0)
            sampler = None
            loader_generator = None
            if self.distributed_enabled:
                sampler = DistributedSampler(
                    dataset, num_replicas=self.world_size, rank=self.rank,
                    shuffle=bool(self.dataloader_shuffle), drop_last=bool(self.dataloader_drop_last),
                )
            elif self.checkpoint_rng_state_enabled:
                sampler = _CheckpointEpochSampler(
                    n, base_seed=self._checkpoint_loader_base_seed,
                    shuffle=bool(self.dataloader_shuffle),
                )
                self._checkpoint_epoch_samplers.append(sampler)
                loader_generator = sampler.iterator_generator()
            return DataLoader(
                dataset, batch_size=bs, shuffle=bool(self.dataloader_shuffle) if sampler is None else False,
                sampler=sampler, num_workers=num_workers, pin_memory=bool(self.dataloader_pin_memory),
                drop_last=bool(self.dataloader_drop_last), persistent_workers=persistent_workers,
                generator=loader_generator,
            )

        if not self.image_importance_sampling_enabled:
            _validate_distributed_loader_size(self.image_train_indices, label="image datum")
        _validate_distributed_loader_size(self.spec_train_indices, label="spectral datum")

        self.image_loader = (
            None
            if self.image_importance_sampling_enabled
            else _make_loader(self.image_train_indices, self.image_batch_size)
        )
        self.spec_loader = _make_loader(self.spec_train_indices, self.spec_batch_size)
        self.image_sampler = (
            self.image_loader.sampler
            if isinstance(getattr(self.image_loader, "sampler", None), DistributedSampler)
            else None
        )
        self.spec_sampler = (
            self.spec_loader.sampler
            if isinstance(getattr(self.spec_loader, "sampler", None), DistributedSampler)
            else None
        )

    def _steps_per_epoch(self) -> int:
        if self.image_importance_sampling_enabled:
            image_batch_size = self._resolve_effective_global_image_batch_size()
            n_total = int(self.image_train_indices.numel())
            if image_batch_size > 0:
                if self.dataloader_drop_last:
                    n_img = int(n_total // image_batch_size)
                else:
                    n_img = int((n_total + image_batch_size - 1) // image_batch_size)
            else:
                n_img = 0
        else:
            n_img = len(self.image_loader) if self.image_loader is not None else 0
        n_spec = len(self.spec_loader) if self.spec_loader is not None else 0
        return max(n_img, n_spec)

    def epoch_position(self) -> Dict[str, int]:
        """Absolute sampling position for the checkpoint.

        Normalized so steps_into_epoch always names an unfinished epoch, which
        is the invariant load_epoch_position validates against."""
        steps_per_epoch = int(self._steps_per_epoch())
        epochs_completed = int(self._epochs_completed)
        steps_into_epoch = int(self._steps_into_epoch)
        if steps_per_epoch > 0 and steps_into_epoch >= steps_per_epoch:
            epochs_completed += steps_into_epoch // steps_per_epoch
            steps_into_epoch = steps_into_epoch % steps_per_epoch
        return {
            "steps_per_epoch": steps_per_epoch,
            "epochs_completed": epochs_completed,
            "steps_into_epoch": steps_into_epoch,
        }

    def load_epoch_position(self, position: Any) -> bool:
        """Continue the epoch cycle a checkpoint stopped inside.

        Refused loudly when the epoch length changed, because the recorded
        intra-epoch offset then names a different fraction of the data than it
        did when it was written."""
        if not isinstance(position, Mapping):
            print(
                "WARNING: resume checkpoint carries no epoch position; the "
                "resumed run restarts its epoch cycle at 0.",
                flush=True,
            )
            return False
        steps_per_epoch = int(self._steps_per_epoch())
        saved_steps_per_epoch = int(position.get("steps_per_epoch", -1))
        epochs_completed = int(position.get("epochs_completed", 0))
        steps_into_epoch = int(position.get("steps_into_epoch", 0))
        if saved_steps_per_epoch != steps_per_epoch:
            print(
                "WARNING: resume checkpoint recorded "
                f"steps_per_epoch={saved_steps_per_epoch} but this trainer has "
                f"{steps_per_epoch}; the epoch position was NOT restored and the "
                "epoch cycle restarts at 0.",
                flush=True,
            )
            return False
        if epochs_completed < 0 or not 0 <= steps_into_epoch < max(1, steps_per_epoch):
            print(
                "WARNING: resume checkpoint epoch position "
                f"(epochs_completed={epochs_completed}, "
                f"steps_into_epoch={steps_into_epoch}) is out of range for "
                f"steps_per_epoch={steps_per_epoch}; it was NOT restored.",
                flush=True,
            )
            return False
        self._resume_epoch_offset = epochs_completed
        self._resume_epoch_batch_offset = steps_into_epoch
        self._epochs_completed = epochs_completed
        self._steps_into_epoch = steps_into_epoch
        return True

    @staticmethod
    def _iter_loader(loader: DataLoader | None):
        if loader is None:
            return None
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, _CheckpointEpochSampler):
            loader.generator = sampler.iterator_generator()
        return iter(loader)

    @staticmethod
    def _loader_skip_batches(
        loader: DataLoader | None, skip_batches: int, *, cycle_shorter: bool,
    ) -> int:
        if loader is None:
            return 0
        return int(skip_batches) if cycle_shorter else min(int(skip_batches), len(loader))

    def _set_loader_resume_cursor(
        self, loader: DataLoader | None, consumed_batches: int,
    ) -> None:
        if loader is None:
            return
        sampler = getattr(loader, "sampler", None)
        if not isinstance(sampler, _CheckpointEpochSampler):
            return
        batch_size = loader.batch_size
        if batch_size is None:
            raise RuntimeError("Checkpoint sampler DataLoader has no batch size.")
        sampler.set_resume_cursor(
            int(consumed_batches), batch_size=int(batch_size),
            drop_last=bool(loader.drop_last),
        )

    def _iter_epoch_batches(self, *, skip_batches: int = 0):
        """Yield (image_indices_cpu, spec_indices_cpu) pairs for one epoch.

        skip_batches shortens a resumed epoch by the batches the interrupted
        invocation already drew from it, so later epoch boundaries land where
        an uninterrupted run's would."""
        image_loader = self.image_loader
        spec_loader = self.spec_loader
        steps = self._steps_per_epoch()
        if steps <= 0:
            return
        cycle_image = image_loader is not None and spec_loader is not None and len(image_loader) < steps
        cycle_spec = spec_loader is not None and len(spec_loader) > 0 and len(spec_loader) < steps
        if self.checkpoint_rng_state_enabled:
            if skip_batches < 0 or skip_batches > steps:
                raise ValueError(
                    f"skip_batches must be between 0 and {steps}, got {skip_batches}."
                )
            self._set_loader_resume_cursor(
                image_loader,
                self._loader_skip_batches(
                    image_loader, int(skip_batches), cycle_shorter=cycle_image,
                ),
            )
            self._set_loader_resume_cursor(
                spec_loader,
                self._loader_skip_batches(
                    spec_loader, int(skip_batches), cycle_shorter=cycle_spec,
                ),
            )
            steps = steps - int(skip_batches)
            if steps <= 0:
                return
            img_iter = self._iter_loader(image_loader)
            spec_iter = self._iter_loader(spec_loader)
        else:
            img_iter = iter(image_loader) if image_loader is not None else None
            spec_iter = iter(spec_loader) if spec_loader is not None else None
            # the cycling decisions above describe the full epoch; only the number
            # of batches still owed to it shrinks when an epoch resumes mid-way
            steps = steps - int(skip_batches)
            if steps <= 0:
                return

        def _next_batch(loader, iterator, *, cycle_shorter: bool):
            if loader is None or iterator is None:
                return None, iterator
            try:
                return next(iterator), iterator
            except StopIteration:
                if not cycle_shorter:
                    return None, None
                iterator = self._iter_loader(loader)
                return next(iterator), iterator

        for _ in range(steps):
            if self.image_importance_sampling_enabled:
                # One-ahead producer thread: the next batch's CPU inverse-CDF
                # draws overlap the current step's GPU work. A single worker
                # owns the generator and futures are consumed FIFO, so the
                # realized draw sequence is identical to serial sampling.
                pool = getattr(self, "_importance_sample_pool", None)
                if pool is None:
                    pool = ThreadPoolExecutor(max_workers=1)
                    self._importance_sample_pool = pool
                restored = self._restored_importance_pending_batch
                self._restored_importance_pending_batch = None
                if restored is not None:
                    img_batch, image_weights = restored
                else:
                    pending = getattr(self, "_importance_pending_batch", None)
                    if pending is None:
                        pending = pool.submit(self._sample_importance_image_batch)
                    img_batch, image_weights = pending.result()
                self._importance_pending_batch = pool.submit(
                    self._sample_importance_image_batch
                )
                self._pending_image_importance_weights = image_weights
            else:
                img_batch, img_iter = _next_batch(image_loader, img_iter, cycle_shorter=cycle_image)
            spec_batch, spec_iter = _next_batch(spec_loader, spec_iter, cycle_shorter=cycle_spec)
            yield img_batch, spec_batch

    def _evaluate_model_samples(
        self, omega_xyz: torch.Tensor, loglam: torch.Tensor, *, exposure_id: int, use_ddp: bool = True,
    ) -> torch.Tensor:
        if omega_xyz.ndim != 2 or omega_xyz.shape[-1] != 3:
            raise ValueError("omega_xyz must have shape [N,3].")
        if loglam.ndim != 1:
            raise ValueError("loglam must have shape [N].")
        if omega_xyz.shape[0] != loglam.shape[0]:
            raise ValueError("omega_xyz and loglam must have the same leading dimension.")
        runtime = self.runtime_by_exposure[int(exposure_id)]
        active_model = self.model if use_ddp else self._base_model()
        outputs = []
        batch_size = 65536
        for start in range(0, omega_xyz.shape[0], batch_size):
            end = min(start + batch_size, omega_xyz.shape[0])
            normalize = runtime.exposure.wcs.omega_xyz_to_norm_torch
            x_norm, y_norm = normalize(omega_xyz[start:end])
            denom = self.loglam_max - self.loglam_min
            if denom <= 0:
                raise ValueError("loglam_max must be greater than loglam_min.")
            loglam_norm = 2.0 * (loglam[start:end] - self.loglam_min) / denom - 1.0
            if self._speed_profile_enabled and self.device.type == "cuda":
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                with self._autocast_context():
                    flux = active_model(x_norm, y_norm, loglam_norm)
                event_end.record()
                self._record_speed_cuda_events("model_forward_sec", event_start, event_end)
            else:
                wall_start = time.perf_counter()
                with self._autocast_context():
                    flux = active_model(x_norm, y_norm, loglam_norm)
                self._record_speed_wall("model_forward_sec", time.perf_counter() - wall_start)
            outputs.append(flux)
        return torch.cat(outputs, dim=0)

    def _autocast_context(self):
        if not self.amp_enabled or self.amp_device_type is None or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype)

    def _evaluate_field_for_pixels(self, exposure_id: int, pixel_indices: torch.Tensor) -> torch.Tensor:
        if pixel_indices.ndim != 1:
            raise ValueError("pixel_indices must be a 1D tensor.")
        runtime = self._ensure_diagnostic_runtime(int(exposure_id))
        if pixel_indices.numel() == 0:
            return torch.empty((0, runtime.renderer.grid.log_lam.shape[0]), device=self.device)
        if runtime.omega_xyz is None:
            raise RuntimeError("Diagnostic runtime omega_xyz was not initialized.")
        omega = torch.index_select(runtime.omega_xyz, 0, pixel_indices.to(device=self.device))
        loglam = runtime.renderer.grid.log_lam
        n_pixels = omega.shape[0]
        L = loglam.shape[0]
        omega_rep = omega.unsqueeze(1).expand(n_pixels, L, 3).reshape(-1, 3)
        loglam_rep = loglam.unsqueeze(0).expand(n_pixels, L).reshape(-1)
        flux = self._evaluate_model_samples(omega_rep, loglam_rep, exposure_id=exposure_id, use_ddp=False)
        return flux.view(n_pixels, L)

    def _render_band_for_omega(
        self, exposure_id: int, band: str, omega_xyz: np.ndarray, *, chunk_size: int,
        profile: Dict[str, Any] | None = None,
    ) -> torch.Tensor:
        runtime = self.runtime_by_exposure[int(exposure_id)]
        loglam = runtime.renderer.grid.log_lam
        n_lambda = int(loglam.shape[0])
        n_rows = int(omega_xyz.shape[0])
        if n_rows == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        if profile is not None:
            profile["n_rows"] = int(n_rows)
            profile["n_lambda"] = int(n_lambda)
            profile["total_model_points"] = int(n_rows * n_lambda)
            profile["chunk_size"] = int(chunk_size)
            profile["chunks"] = []
            profile["timings_sec"] = {
                "omega_tensor": 0.0, "model_eval": 0.0, "render_band": 0.0, "total": 0.0,
            }
        outputs: List[torch.Tensor] = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk_profile: Dict[str, Any] | None = None
            if profile is not None:
                chunk_profile = {"row_start": int(start), "row_end": int(end), "rows": int(end - start)}
                chunk_start = time.perf_counter()
            omega_tensor_start = time.perf_counter()
            omega_t = torch.tensor(omega_xyz[start:end], dtype=torch.float32, device=self.device)
            omega_tensor_sec = time.perf_counter() - omega_tensor_start
            rows = end - start
            omega_rep = omega_t.unsqueeze(1).expand(rows, n_lambda, 3).reshape(-1, 3)
            loglam_rep = loglam.unsqueeze(0).expand(rows, n_lambda).reshape(-1)
            model_eval_start = time.perf_counter()
            flux = self._evaluate_model_samples(
                omega_rep, loglam_rep, exposure_id=int(exposure_id), use_ddp=False,
            ).view(rows, n_lambda)
            model_eval_sec = time.perf_counter() - model_eval_start
            render_band_start = time.perf_counter()
            outputs.append(runtime.renderer.render_band(flux, band, normalize=True))
            render_band_sec = time.perf_counter() - render_band_start
            if profile is not None and chunk_profile is not None:
                chunk_profile["timings_sec"] = {
                    "omega_tensor": omega_tensor_sec, "model_eval": model_eval_sec,
                    "render_band": render_band_sec, "total": time.perf_counter() - chunk_start,
                }
                profile["chunks"].append(chunk_profile)
                profile["timings_sec"]["omega_tensor"] += omega_tensor_sec
                profile["timings_sec"]["model_eval"] += model_eval_sec
                profile["timings_sec"]["render_band"] += render_band_sec
                profile["timings_sec"]["total"] += chunk_profile["timings_sec"]["total"]
        return torch.cat(outputs, dim=0)

    def _spectra_over_splat_support(
        self, exposure_id: int, omega_t: torch.Tensor, loglam: torch.Tensor,
    ) -> torch.Tensor | None:
        """Model spectra summed over the splats whose support reaches each point.

        Returns None for models without the splat interface, so the caller
        falls back to the dense evaluation.
        """
        model = self._base_model()
        if not callable(getattr(model, "covariance_components", None)):
            return None
        runtime = self.runtime_by_exposure[int(exposure_id)]
        denominator = self.loglam_max - self.loglam_min
        if denominator <= 0:
            raise ValueError("loglam_max must be greater than loglam_min.")
        n_points = int(omega_t.shape[0])
        n_lambda = int(loglam.shape[0])
        with torch.no_grad():
            x, y = runtime.exposure.wcs.omega_xyz_to_norm_torch(omega_t)
            loglam_norm = 2.0 * (loglam - self.loglam_min) / denominator - 1.0
            # The field carries no PSF, so the support is the splat's own
            # covariance at the trainer's configured truncation width.
            index = build_packed_point_index(
                model, x, y, 0.0, n_sigma=self.pair_n_sigma
            )
            active = index.active_splat
            if active.numel() == 0:
                return torch.zeros(
                    (n_points, n_lambda), device=self.device, dtype=model.mu.dtype
                )
            a, b, d = model.covariance_components(active, sorted_unique=True)
            centre = model.positions(active, sorted_unique=True)
            determinant = (a * d - b * b).clamp_min(1e-30)
            dx = x[:, None] - centre[None, :, 0]
            dy = y[:, None] - centre[None, :, 1]
            q = (
                d * dx.square() - (2.0 * b) * dx * dy + a * dy.square()
            ) / determinant
            gaussian = (1.0 / (2 * math.pi * determinant.sqrt())) * torch.exp(-0.5 * q)
            scaled_sed = model.fluxes(active, sorted_unique=True)[:, None] * model.sed.eval(
                model.sed_coefficients(active, sorted_unique=True),
                loglam_norm,
                model.spectral_shifts(active, sorted_unique=True),
            )
            spectra = torch.zeros(
                (n_points, n_lambda), device=gaussian.device, dtype=gaussian.dtype
            )
            # (n_points, chunk, n_lambda) is the shape of the sum, so chunk the
            # splat axis to bound it the way SplatField.forward bounds its own.
            step = max(1, 100_000_000 // max(n_points * n_lambda, 1))
            for start in range(0, int(active.numel()), step):
                stop = start + step
                spectra += (
                    gaussian[:, start:stop, None] * scaled_sed[None, start:stop, :]
                ).sum(dim=1)
            return spectra

    def _evaluate_spectra_for_omega(
        self, exposure_id: int, omega_xyz: np.ndarray | torch.Tensor, *, renderer: SpectralRenderer,
    ) -> torch.Tensor:
        """Model spectra at a handful of sky positions.

        DIAGNOSTIC PATH, not bitwise with the dense field evaluation. The dense
        sum runs every splat against every (position, wavelength) query, which
        at the paper run's 10M-splat census costs ~200 s for the dozen pixels
        the point-SED figure asks for. Splats are gathered here through the
        same truncated support the production image operator uses
        (`trainer.refine.pair_n_sigma`), so only the splats that reach a
        requested position enter the sum. A dropped splat contributes at most
        exp(-n_sigma^2/2) of its own peak - 1.3e-14 at the shipped n_sigma = 8,
        five orders of magnitude under float32's 6e-8 resolution - and the
        terms that remain are the same values added in a different order, with
        the quadratic form written out from the covariance components instead
        of through a matrix inverse. Both are float32 rounding effects, so the
        spectra are close to but not bitwise equal to the dense path's.
        """
        omega_t = torch.as_tensor(omega_xyz, dtype=torch.float32, device=self.device)
        if omega_t.ndim != 2 or omega_t.shape[-1] != 3:
            raise ValueError("omega_xyz must have shape [N,3].")
        n_points = int(omega_t.shape[0])
        if n_points == 0:
            return torch.empty((0, int(renderer.grid.log_lam.shape[0])), device=self.device)
        loglam = renderer.grid.log_lam
        n_lambda = int(loglam.shape[0])
        supported = self._spectra_over_splat_support(
            int(exposure_id), omega_t, loglam
        )
        if supported is not None:
            return supported
        omega_rep = omega_t.unsqueeze(1).expand(n_points, n_lambda, 3).reshape(-1, 3)
        loglam_rep = loglam.unsqueeze(0).expand(n_points, n_lambda).reshape(-1)
        flux = self._evaluate_model_samples(
            omega_rep, loglam_rep, exposure_id=int(exposure_id), use_ddp=False,
        )
        return flux.view(n_points, n_lambda)

    def _project_spectra_to_band_fluxes(
        self, spectra: torch.Tensor, *, renderer: SpectralRenderer | None = None,
    ) -> Dict[str, np.ndarray]:
        active_renderer = self.diagnostic_renderer if renderer is None else renderer
        outputs: Dict[str, np.ndarray] = {}
        for band in sorted(self.renderer_bandpasses.keys()):
            outputs[band] = (
                active_renderer.render_band(spectra, band, normalize=True)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
        return outputs

    def _evaluate_weighted_field_for_omega(
        self, exposure_id: int, omega_xyz: np.ndarray, weights: np.ndarray, *, chunk_size: int,
    ) -> torch.Tensor:
        runtime = self.runtime_by_exposure[int(exposure_id)]
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        omega_arr = np.asarray(omega_xyz, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        if omega_arr.ndim != 2 or omega_arr.shape[-1] != 3:
            raise ValueError("omega_xyz must have shape [N,3].")
        if weights_arr.ndim != 1 or weights_arr.shape[0] != omega_arr.shape[0]:
            raise ValueError("weights must be a 1D array matching omega_xyz rows.")
        if omega_arr.shape[0] == 0:
            return torch.empty((runtime.renderer.grid.log_lam.shape[0],), device=self.device)
        weight_sum = float(np.sum(weights_arr))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("weights must have positive finite total weight.")
        loglam = runtime.renderer.grid.log_lam
        n_lambda = int(loglam.shape[0])
        field_sum = torch.zeros((n_lambda,), dtype=torch.float32, device=self.device)
        for start in range(0, omega_arr.shape[0], chunk_size):
            end = min(start + chunk_size, omega_arr.shape[0])
            omega_t = torch.tensor(omega_arr[start:end], dtype=torch.float32, device=self.device)
            weight_t = torch.tensor(weights_arr[start:end], dtype=torch.float32, device=self.device)
            rows = end - start
            omega_rep = omega_t.unsqueeze(1).expand(rows, n_lambda, 3).reshape(-1, 3)
            loglam_rep = loglam.unsqueeze(0).expand(rows, n_lambda).reshape(-1)
            flux = self._evaluate_model_samples(
                omega_rep, loglam_rep, exposure_id=int(exposure_id), use_ddp=False,
            ).view(rows, n_lambda)
            field_sum = field_sum + (flux * weight_t.unsqueeze(1)).sum(dim=0)
        return field_sum / float(weight_sum)

    def _sky_disk_quadrature(
        self, aperture: SkyDiskApertureSpec, *, psf_fwhm_arcsec: float | None, radial_nodes: int = 4,
        angular_nodes: int = 8, psf_nodes: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        if radial_nodes <= 0:
            raise ValueError("radial_nodes must be positive.")
        if angular_nodes <= 0:
            raise ValueError("angular_nodes must be positive.")
        if psf_nodes <= 0:
            raise ValueError("psf_nodes must be positive.")
        center_ra_deg, center_dec_deg = aperture.center_radec_deg
        radius_arcsec = float(aperture.radius_arcsec)
        if radius_arcsec <= 0.0 or not np.isfinite(radius_arcsec):
            raise ValueError("SkyDiskApertureSpec.radius_arcsec must be positive and finite.")
        cos_dec = float(np.cos(np.deg2rad(center_dec_deg)))
        if abs(cos_dec) < 1e-12:
            raise ValueError("SkyDiskApertureSpec center is too close to a pole for small-angle RA offsets.")

        radial_u = (np.arange(radial_nodes, dtype=np.float64) + 0.5) / float(radial_nodes)
        radii = radius_arcsec * np.sqrt(radial_u)
        theta = 2.0 * np.pi * (np.arange(angular_nodes, dtype=np.float64) + 0.5) / float(angular_nodes)
        rr, tt = np.meshgrid(radii, theta, indexing="ij")
        dx_base = (rr * np.cos(tt)).reshape(-1)
        dy_base = (rr * np.sin(tt)).reshape(-1)
        base_weights = np.full(dx_base.shape, 1.0 / float(dx_base.size), dtype=np.float64)

        sigma_arcsec = 0.0
        if self.apply_spatial_psf and psf_fwhm_arcsec is not None:
            psf_fwhm = float(psf_fwhm_arcsec)
            if np.isfinite(psf_fwhm) and psf_fwhm > 0.0:
                sigma_arcsec = psf_fwhm / 2.355

        if sigma_arcsec > 0.0:
            gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(psf_nodes)
            offset = np.sqrt(2.0) * sigma_arcsec * gh_nodes
            weight_1d = gh_weights / np.sqrt(np.pi)
            dx_psf, dy_psf = np.meshgrid(offset, offset, indexing="ij")
            psf_weights = np.outer(weight_1d, weight_1d).reshape(-1)
            dx_total = dx_base[:, None] + dx_psf.reshape(1, -1)
            dy_total = dy_base[:, None] + dy_psf.reshape(1, -1)
            weights = (base_weights[:, None] * psf_weights[None, :]).reshape(-1)
            dx_total = dx_total.reshape(-1)
            dy_total = dy_total.reshape(-1)
        else:
            dx_total = dx_base
            dy_total = dy_base
            weights = base_weights

        ra = float(center_ra_deg) + dx_total / (3600.0 * cos_dec)
        dec = float(center_dec_deg) + dy_total / 3600.0
        omega_xyz = radec_deg_to_omega_xyz(ra, dec).reshape(-1, 3)
        weights = weights / np.sum(weights)
        return omega_xyz, weights

    def _build_mc_kernel_samples(self, global_indices: torch.Tensor, *, k: int) -> GroupedKernelSamples:
        if self.psf_model != "gaussian":
            kinds = self._gather_named_rows(
                "datum_kind", global_indices, device=torch.device("cpu"), dtype=torch.long,
            )
            if bool(torch.any(kinds == int(DatumKind.IMAGE_PIXEL))):
                raise NotImplementedError(
                    "MC image kernel builders do not support trainer.psf_model != "
                    "'gaussian'; use trainer.analytic_images=true."
                )
        start = time.perf_counter()
        samples = self.data.kernel_factory.build_kernel_samples_for_indices(
            global_indices, k=int(k), device=self.device,
        )
        self._record_speed_wall("kernel_sampling_sec", time.perf_counter() - start)
        return samples

    def _build_mc_kernel_sample_pair(
        self, global_indices: torch.Tensor, *, k: int,
    ) -> tuple[GroupedKernelSamples, GroupedKernelSamples]:
        if self.psf_model != "gaussian":
            kinds = self._gather_named_rows(
                "datum_kind", global_indices, device=torch.device("cpu"), dtype=torch.long,
            )
            if bool(torch.any(kinds == int(DatumKind.IMAGE_PIXEL))):
                raise NotImplementedError(
                    "MC image kernel builders do not support trainer.psf_model != "
                    "'gaussian'; use trainer.analytic_images=true."
                )
        factory = self.data.kernel_factory
        start = time.perf_counter()
        plan_start = time.perf_counter()
        plan = factory.build_kernel_sample_plan_for_indices(global_indices)
        self._record_speed_wall("kernel_plan_sec", time.perf_counter() - plan_start)
        sample_pair_start = time.perf_counter()
        samples_a, samples_b = factory.build_kernel_sample_pair_from_plan(plan, k=int(k), device=self.device)
        self._record_speed_wall("kernel_sample_pair_sec", time.perf_counter() - sample_pair_start)
        self._record_speed_wall("kernel_sampling_sec", time.perf_counter() - start)
        return samples_a, samples_b

    def _compute_source_mu_mc_from_samples(
        self, global_indices: torch.Tensor, samples: GroupedKernelSamples, *,
        apply_spectral_nuisance: bool = True,
    ) -> torch.Tensor:
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        grouped = samples
        sample_tensors = grouped.samples
        if sample_tensors.w.shape[0] != global_indices.numel():
            raise ValueError("KernelSamples row count must match the number of global indices.")
        mu_sorted = torch.empty((global_indices.numel(), ), device=self.device, dtype=torch.float32)
        local_indices = grouped.row_datum_index
        exposure_blocks = grouped.exposure_blocks
        if not exposure_blocks and int(grouped.row_exposure_id.numel()) > 0:
            exposure_blocks = self._infer_grouped_exposure_blocks(grouped.row_exposure_id)
        for exp_id, start, end in exposure_blocks:
            omega = sample_tensors.omega_xyz[start:end]
            loglam = sample_tensors.loglam[start:end]
            w = sample_tensors.w[start:end]
            Z = sample_tensors.Z[start:end]
            bsz, ksz, _ = omega.shape
            vals = self._evaluate_model_samples(
                omega.reshape(-1, 3), loglam.reshape(-1), exposure_id=exp_id,
            ).view(bsz, ksz)
            mu_exp = (vals * w).sum(dim=1) * Z
            if exp_id in self.spec_exposure_ids:
                mu_exp = self._apply_spectral_nuisance(
                    exp_id, local_indices[start:end], mu_exp,
                    apply_spectral_nuisance=apply_spectral_nuisance,
                )
            mu_sorted[start:end] = mu_exp
        mu = torch.empty_like(mu_sorted)
        mu[grouped.row_original_index] = mu_sorted
        return mu

    def _apply_observation_contract(
        self, global_indices: torch.Tensor, source_mu: torch.Tensor,
    ) -> torch.Tensor:
        batch = self.data.batch
        mu = source_mu
        if getattr(batch, "source_gate", None) is not None:
            gate = self._gather_named_rows(
                "source_gate", global_indices, device=self.device, dtype=source_mu.dtype,
            )
            mu = mu * gate
        if getattr(batch, "additive", None) is not None:
            additive = self._gather_named_rows(
                "additive", global_indices, device=self.device, dtype=source_mu.dtype,
            )
            mu = mu + additive
        return mu

    def _compute_mu_mc_from_samples(
        self, global_indices: torch.Tensor, samples: GroupedKernelSamples, *,
        apply_spectral_nuisance: bool = True,
    ) -> torch.Tensor:
        source_mu = self._compute_source_mu_mc_from_samples(
            global_indices, samples, apply_spectral_nuisance=apply_spectral_nuisance,
        )
        return self._apply_observation_contract(global_indices, source_mu)

    def _compute_source_mu_mc_pair_from_samples(
        self, global_indices: torch.Tensor, samples_a: GroupedKernelSamples, samples_b: GroupedKernelSamples,
        *, apply_spectral_nuisance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        grouped_a = samples_a
        grouped_b = samples_b
        sample_tensors_a = grouped_a.samples
        sample_tensors_b = grouped_b.samples
        n_rows = int(global_indices.numel())
        if sample_tensors_a.w.shape[0] != n_rows or sample_tensors_b.w.shape[0] != n_rows:
            raise ValueError("KernelSamples row count must match the number of global indices.")
        local_indices = grouped_a.row_datum_index
        exposure_blocks = grouped_a.exposure_blocks
        if not exposure_blocks and int(grouped_a.row_exposure_id.numel()) > 0:
            exposure_blocks = self._infer_grouped_exposure_blocks(grouped_a.row_exposure_id)
        vals_a, vals_b = self._evaluate_grouped_model_sample_pair(
            sample_tensors_a, sample_tensors_b, exposure_blocks=exposure_blocks,
        )
        mu_base_a = (vals_a * sample_tensors_a.w).sum(dim=1) * sample_tensors_a.Z
        mu_base_b = (vals_b * sample_tensors_b.w).sum(dim=1) * sample_tensors_b.Z
        mu_parts_a: list[torch.Tensor] = []
        mu_parts_b: list[torch.Tensor] = []
        for exp_id, start, end in exposure_blocks:
            mu_exp_a = mu_base_a[start:end]
            mu_exp_b = mu_base_b[start:end]
            if exp_id in self.spec_exposure_ids:
                local = local_indices[start:end]
                mu_exp_a = self._apply_spectral_nuisance(
                    exp_id, local, mu_exp_a, apply_spectral_nuisance=apply_spectral_nuisance,
                )
                mu_exp_b = self._apply_spectral_nuisance(
                    exp_id, local, mu_exp_b, apply_spectral_nuisance=apply_spectral_nuisance,
                )
            mu_parts_a.append(mu_exp_a)
            mu_parts_b.append(mu_exp_b)
        mu_sorted_a = torch.cat(mu_parts_a, dim=0) if mu_parts_a else mu_base_a
        mu_sorted_b = torch.cat(mu_parts_b, dim=0) if mu_parts_b else mu_base_b
        mu_a = torch.empty_like(mu_sorted_a)
        mu_b = torch.empty_like(mu_sorted_b)
        mu_a[grouped_a.row_original_index] = mu_sorted_a
        mu_b[grouped_a.row_original_index] = mu_sorted_b
        return mu_a, mu_b

    def _evaluate_grouped_model_sample_pair(
        self, samples_a, samples_b, *, exposure_blocks: tuple[tuple[int, int, int], ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if samples_a.omega_xyz.shape != samples_b.omega_xyz.shape:
            raise ValueError("Paired KernelSamples must have matching omega shapes.")
        if samples_a.loglam.shape != samples_b.loglam.shape:
            raise ValueError("Paired KernelSamples must have matching loglam shapes.")
        n_rows = int(samples_a.loglam.shape[0])
        k = int(samples_a.loglam.shape[1])
        flat_count = int(n_rows * k)
        total_count = int(2 * flat_count)
        if total_count == 0:
            empty = torch.empty((n_rows, k), dtype=torch.float32, device=self.device)
            return empty, empty
        denom = self.loglam_max - self.loglam_min
        if denom <= 0:
            raise ValueError("loglam_max must be greater than loglam_min.")
        x_norm = torch.empty((total_count,), dtype=torch.float32, device=self.device)
        y_norm = torch.empty((total_count,), dtype=torch.float32, device=self.device)
        loglam_norm = torch.empty((total_count,), dtype=torch.float32, device=self.device)

        def _fill_block(sample_tensors, offset: int, exp_id: int, start: int, end: int) -> None:
            runtime = self.runtime_by_exposure[int(exp_id)]
            normalize = getattr(runtime.exposure.wcs, "omega_xyz_to_norm_torch", None)
            if not callable(normalize):
                raise ValueError(
                    "Grouped paired MC evaluation requires every active exposure WCS "
                    "to implement omega_xyz_to_norm_torch."
                )
            dst_start = int(offset + start * k)
            dst_end = int(offset + end * k)
            omega = sample_tensors.omega_xyz[start:end].reshape(-1, 3)
            block_loglam = sample_tensors.loglam[start:end].reshape(-1)
            block_x, block_y = normalize(omega)
            x_norm[dst_start:dst_end] = block_x.to(dtype=torch.float32)
            y_norm[dst_start:dst_end] = block_y.to(dtype=torch.float32)
            loglam_norm[dst_start:dst_end] = (
                2.0 * (block_loglam.to(dtype=torch.float32) - self.loglam_min) / denom - 1.0
            )

        if self._speed_profile_enabled and self.device.type == "cuda":
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            with self._autocast_context():
                for exp_id, start, end in exposure_blocks:
                    _fill_block(samples_a, 0, exp_id, start, end)
                    _fill_block(samples_b, flat_count, exp_id, start, end)
                vals = self.model(x_norm, y_norm, loglam_norm)
            event_end.record()
            self._record_speed_cuda_events("model_forward_sec", event_start, event_end)
        else:
            wall_start = time.perf_counter()
            with self._autocast_context():
                for exp_id, start, end in exposure_blocks:
                    _fill_block(samples_a, 0, exp_id, start, end)
                    _fill_block(samples_b, flat_count, exp_id, start, end)
                vals = self.model(x_norm, y_norm, loglam_norm)
            self._record_speed_wall("model_forward_sec", time.perf_counter() - wall_start)
        return vals[:flat_count].view(n_rows, k), vals[flat_count:].view(n_rows, k)

    def _compute_mu_mc_pair_from_samples(
        self, global_indices: torch.Tensor, samples_a: GroupedKernelSamples, samples_b: GroupedKernelSamples,
        *, apply_spectral_nuisance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_a, source_b = self._compute_source_mu_mc_pair_from_samples(
            global_indices, samples_a, samples_b, apply_spectral_nuisance=apply_spectral_nuisance,
        )
        return (
            self._apply_observation_contract(global_indices, source_a),
            self._apply_observation_contract(global_indices, source_b),
        )

    @staticmethod
    def _infer_grouped_exposure_blocks(exposure_ids: torch.Tensor,) -> tuple[tuple[int, int, int], ...]:
        ids = exposure_ids.detach().to(dtype=torch.long, device="cpu").tolist()
        if not ids:
            return ()
        blocks: list[tuple[int, int, int]] = []
        start = 0
        current = int(ids[0])
        for idx in range(1, len(ids)):
            exp_id = int(ids[idx])
            if exp_id != current:
                blocks.append((current, start, idx))
                start = idx
                current = exp_id
        blocks.append((current, start, len(ids)))
        return tuple(blocks)

    def _compute_mu_mc_for_indices(
        self, global_indices: torch.Tensor, *, apply_spectral_nuisance: bool = True
    ) -> torch.Tensor:
        samples = self._build_mc_kernel_samples(global_indices, k=self.kernel_samples_k)
        return self._compute_mu_mc_from_samples(
            global_indices, samples, apply_spectral_nuisance=apply_spectral_nuisance,
        )

    def _compute_source_mu_mc_for_indices(
        self, global_indices: torch.Tensor, *, apply_spectral_nuisance: bool = True
    ) -> torch.Tensor:
        samples = self._build_mc_kernel_samples(global_indices, k=self.kernel_samples_k)
        return self._compute_source_mu_mc_from_samples(
            global_indices, samples, apply_spectral_nuisance=apply_spectral_nuisance,
        )

    def _compute_field_sum_for_fiber(self, exposure_id: int) -> torch.Tensor | None:
        runtime = self._ensure_diagnostic_runtime(int(exposure_id))
        if runtime.fiber_indices is None or runtime.fiber_indices.numel() == 0:
            return None
        if runtime.aperture_weights is not None:
            if runtime.aperture_support is None or runtime.aperture_support.numel() == 0:
                return None
            field_pixels = self._evaluate_field_for_pixels(exposure_id, runtime.aperture_support)
            weights = torch.index_select(runtime.aperture_weights, 0, runtime.aperture_support)
            return integrate_aperture(field_pixels, weights, normalize=False)
        field_pixels = self._evaluate_field_for_pixels(exposure_id, runtime.fiber_indices)
        return field_pixels.sum(dim=0)

    def compute_mu(
        self, global_indices: torch.Tensor | None = None, *, apply_spectral_nuisance: bool = True,
    ) -> torch.Tensor:
        if global_indices is None:
            global_indices = torch.arange(int(self.data.batch.y.shape[0]), dtype=torch.long)
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        return self._compute_mu_mc_for_indices(
            global_indices, apply_spectral_nuisance=apply_spectral_nuisance
        )

    def compute_source_mu(
        self, global_indices: torch.Tensor | None = None, *, apply_spectral_nuisance: bool = True,
    ) -> torch.Tensor:
        if global_indices is None:
            global_indices = torch.arange(int(self.data.batch.y.shape[0]), dtype=torch.long)
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        return self._compute_source_mu_mc_for_indices(
            global_indices, apply_spectral_nuisance=apply_spectral_nuisance
        )

    def _compute_mc_unbiased_gaussian_terms(self, global_indices: torch.Tensor) -> torch.Tensor:
        if global_indices.ndim != 1:
            raise ValueError("global_indices must be a 1D tensor.")
        if global_indices.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        samples_a, samples_b = self._build_mc_kernel_sample_pair(global_indices, k=self.replica_samples_k)
        mu_a, mu_b = self._compute_mu_mc_pair_from_samples(global_indices, samples_a, samples_b)
        y = self._gather_named_rows("y", global_indices, device=self.device, dtype=torch.float32)
        sigma = self._gather_named_rows("sigma", global_indices, device=self.device, dtype=torch.float32)
        return datum_unbiased_gaussian_nll(y, mu_a, mu_b, sigma)

    def _pixel_to_norm_jacobians(
        self,
        exposure,
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
    ) -> torch.Tensor:
        """Forward one-pixel Jacobians at detector coordinates."""
        x = np.asarray(pixel_x, dtype=np.float64).reshape(-1)
        y = np.asarray(pixel_y, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError("Jacobian probe x and y coordinates must align.")
        probe_x = np.stack((x, x + 1.0, x), axis=1).reshape(-1, 1)
        probe_y = np.stack((y, y, y + 1.0), axis=1).reshape(-1, 1)
        probe = np.asarray(
            exposure.obs_wcs.pixel_to_omega_xyz(probe_x, probe_y),
            dtype=np.float64,
        ).reshape(-1, 3)
        # One-pixel differencing of float32 normalized coordinates loses the
        # step to cancellation (~2 percent Jacobian error at full84 scales,
        # measured); probe in float64 and cast only the finished Jacobian.
        x_norm, y_norm = exposure.wcs.omega_xyz_to_norm_torch(
            torch.as_tensor(probe, dtype=torch.float64, device=self.device)
        )
        x_norm = x_norm.reshape(-1, 3)
        y_norm = y_norm.reshape(-1, 3)
        derivative_x = torch.stack(
            (x_norm[:, 1] - x_norm[:, 0], y_norm[:, 1] - y_norm[:, 0]),
            dim=1,
        )
        derivative_y = torch.stack(
            (x_norm[:, 2] - x_norm[:, 0], y_norm[:, 2] - y_norm[:, 0]),
            dim=1,
        )
        return torch.stack((derivative_x, derivative_y), dim=2).to(torch.float32)

    @staticmethod
    def _native_psf_covariances(
        components: PSFMixtureComponents,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.as_tensor(
            components.components,
            dtype=reference.dtype,
            device=reference.device,
        )
        covariance = torch.empty(
            (rows.shape[0], 2, 2), dtype=rows.dtype, device=rows.device
        )
        covariance[:, 0, 0] = rows[:, 1]
        covariance[:, 0, 1] = rows[:, 2]
        covariance[:, 1, 0] = rows[:, 2]
        covariance[:, 1, 1] = rows[:, 3]
        return covariance

    @staticmethod
    def _psf_covariance_relative_variation(
        native_covariance: torch.Tensor,
        reference_jacobians: torch.Tensor,
        probe_jacobians: torch.Tensor,
    ) -> float:
        """Maximum component Frobenius change relative to each reference row."""
        reference = (
            reference_jacobians.unsqueeze(-3)
            @ native_covariance
            @ reference_jacobians.transpose(-1, -2).unsqueeze(-3)
        )
        probes = (
            probe_jacobians.unsqueeze(-3)
            @ native_covariance
            @ probe_jacobians.transpose(-1, -2).unsqueeze(-3)
        )
        denominator = torch.linalg.matrix_norm(reference, dim=(-2, -1)).clamp_min(
            torch.finfo(reference.dtype).tiny
        )
        relative = torch.linalg.matrix_norm(
            probes - reference, dim=(-2, -1)
        ) / denominator
        return float(relative.max())

    @staticmethod
    def _frame_jacobian_probe_coordinates(
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cx = 0.5 * float(width - 1)
        cy = 0.5 * float(height - 1)
        left = min(1.0, float(max(0, width - 2)))
        right = float(max(0, width - 2))
        top = min(1.0, float(max(0, height - 2)))
        bottom = float(max(0, height - 2))
        return (
            np.asarray(
                [cx, left, cx, right, left, right, left, cx, right],
                dtype=np.float64,
            ),
            np.asarray(
                [cy, top, top, top, cy, cy, bottom, bottom, bottom],
                dtype=np.float64,
            ),
        )

    def _jacobian_tile_plan(
        self,
        exposure,
        native_covariance: torch.Tensor,
    ) -> tuple[tuple[int, int], torch.Tensor, float]:
        """Choose a sampled-variation-bounded tile grid, capped at 16x16."""
        width = int(exposure.image_width)
        height = int(exposure.image_height)
        max_tiles_per_axis = self.psf_jacobian_max_tiles_per_axis
        tile_rows = tile_cols = 2
        while True:
            centers_x: list[float] = []
            centers_y: list[float] = []
            probes_x: list[list[float]] = []
            probes_y: list[list[float]] = []
            max_probe_x = float(max(0, width - 2))
            max_probe_y = float(max(0, height - 2))
            for tile_y in range(tile_rows):
                y0 = tile_y * float(height) / tile_rows - 0.5
                y1 = (tile_y + 1) * float(height) / tile_rows - 0.5
                cy = min(max(0.5 * (y0 + y1), 0.0), max_probe_y)
                py0 = min(max(y0, 0.0), max_probe_y)
                py1 = min(max(y1, 0.0), max_probe_y)
                for tile_x in range(tile_cols):
                    x0 = tile_x * float(width) / tile_cols - 0.5
                    x1 = (tile_x + 1) * float(width) / tile_cols - 0.5
                    cx = min(max(0.5 * (x0 + x1), 0.0), max_probe_x)
                    px0 = min(max(x0, 0.0), max_probe_x)
                    px1 = min(max(x1, 0.0), max_probe_x)
                    centers_x.append(cx)
                    centers_y.append(cy)
                    probes_x.append(
                        [cx, px0, cx, px1, px0, px1, px0, cx, px1]
                    )
                    probes_y.append(
                        [cy, py0, py0, py0, cy, cy, py1, py1, py1]
                    )
            center_jacobians = self._pixel_to_norm_jacobians(
                exposure, np.asarray(centers_x), np.asarray(centers_y)
            )
            probe_jacobians = self._pixel_to_norm_jacobians(
                exposure, np.asarray(probes_x).reshape(-1), np.asarray(probes_y).reshape(-1)
            ).reshape(len(centers_x), 9, 2, 2)
            tile_variation = self._psf_covariance_relative_variation(
                native_covariance,
                center_jacobians[:, None, :, :],
                probe_jacobians,
            )
            if tile_variation <= self.psf_jacobian_rel_tol:
                return (tile_rows, tile_cols), center_jacobians, tile_variation
            if (
                tile_rows >= max_tiles_per_axis
                or tile_cols >= max_tiles_per_axis
            ):
                label = getattr(exposure, "label", None) or "unlabelled exposure"
                raise RuntimeError(
                    f"PSF Jacobian {label}: tile grid {tile_rows}x{tile_cols} "
                    f"cannot meet requested covariance variation tolerance "
                    f"{self.psf_jacobian_rel_tol:.6g}; achieved "
                    f"{tile_variation:.6g}. Cap policy is "
                    f"{self.psf_jacobian_cap_policy!r}, so rendering is "
                    "aborted instead of silently accepting the approximation."
                )
            tile_rows = min(max_tiles_per_axis, tile_rows * 2)
            tile_cols = min(max_tiles_per_axis, tile_cols * 2)

    def _point_jacobians(
        self,
        cache: Mapping[str, Any],
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
    ) -> torch.Tensor:
        """Per-pixel float64 pixel-to-normalized Jacobians from validated tables."""
        x = np.asarray(pixel_x, dtype=np.float64).reshape(-1)
        y = np.asarray(pixel_y, dtype=np.float64).reshape(-1)
        if "psf_jacobian_path" not in cache or "pixel_to_norm_jacobian" not in cache:
            raise RuntimeError(
                "Analytic exposure cache lacks Jacobian tables; clear the cache after changing psf_model."
            )
        if cache["psf_jacobian_path"] == "tiled":
            tile_rows, tile_cols = cache["psf_jacobian_tile_grid"]
            tile_x = np.clip(
                np.floor((x + 0.5) * tile_cols / cache["width"]).astype(np.int64),
                0, tile_cols - 1,
            )
            tile_y = np.clip(
                np.floor((y + 0.5) * tile_rows / cache["height"]).astype(np.int64),
                0, tile_rows - 1,
            )
            tile_index = torch.as_tensor(
                tile_y * tile_cols + tile_x, dtype=torch.long, device=self.device
            )
            return cache["pixel_to_norm_jacobian_tiles"].to(torch.float64)[tile_index]
        return cache["pixel_to_norm_jacobian"].to(torch.float64).expand(x.size, 2, 2)

    def _point_psf_mixture(
        self,
        cache: Mapping[str, Any],
        mixture_key: str,
        pixel_x: np.ndarray,
        pixel_y: np.ndarray,
    ) -> PointPSFMixture:
        x = np.asarray(pixel_x, dtype=np.float64).reshape(-1)
        y = np.asarray(pixel_y, dtype=np.float64).reshape(-1)
        if cache["psf_jacobian_path"] == "tiled":
            tile_rows, tile_cols = cache["psf_jacobian_tile_grid"]
            tile_x = np.floor((x + 0.5) * tile_cols / cache["width"]).astype(np.int64)
            tile_y = np.floor((y + 0.5) * tile_rows / cache["height"]).astype(np.int64)
            tile_x = np.clip(tile_x, 0, tile_cols - 1)
            tile_y = np.clip(tile_y, 0, tile_rows - 1)
            tile_index = as_tensor_on_device(
                tile_y * tile_cols + tile_x, dtype=torch.long, device=self.device
            )
            tiled = cache[f"{mixture_key}_tiles"]
            return PointPSFMixture(
                tiled.weights[tile_index], tiled.covariances[tile_index]
            )
        mixture = cache[mixture_key]
        return PointPSFMixture(
            mixture.weights.expand(x.size, -1),
            mixture.covariances.expand(
                (x.size,) + mixture.covariances.shape[1:]
            ),
        )

    def _analytic_cache_for_exposure(self, exp_id: int) -> dict:
        cache = self._analytic_image_cache.get(int(exp_id))
        if cache is not None:
            return cache
        exposure = self.runtime_by_exposure[int(exp_id)].exposure
        band = next(iter(exposure.bandpasses))
        bp = exposure.bandpasses[band]
        lam_tab = np.asarray(bp.lam, dtype=np.float64)
        thr_tab = np.asarray(bp.throughput, dtype=np.float64)
        order = np.argsort(lam_tab)
        lam_tab, thr_tab = lam_tab[order], thr_tab[order]
        # dense resampling with trapezoid endpoint weights, matching the
        # continuous photon-weighted CDF the MC BandpassSampler draws from
        lam_np = np.linspace(lam_tab[0], lam_tab[-1], 129)
        w_np = np.maximum(np.interp(lam_np, lam_tab, thr_tab), 0.0) * lam_np
        trap = np.ones_like(w_np)
        trap[0] = trap[-1] = 0.5
        lam = torch.as_tensor(lam_np, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(w_np * trap, dtype=torch.float32, device=self.device)
        denom = self.loglam_max - self.loglam_min
        if denom <= 0:
            raise ValueError("loglam_max must be greater than loglam_min.")
        ll_norm = 2.0 * (torch.log(lam.clamp_min(1e-6)) - self.loglam_min) / denom - 1.0
        cx = 0.5 * float(exposure.image_width - 1)
        cy = 0.5 * float(exposure.image_height - 1)
        if self.psf_model == "gaussian":
            probe_x = np.array([[cx], [cx + 1.0]])
            probe_y = np.array([[cy], [cy]])
        else:
            probe_x = np.array([[cx], [cx + 1.0], [cx]])
            probe_y = np.array([[cy], [cy], [cy + 1.0]])
        probe = np.asarray(
            exposure.obs_wcs.pixel_to_omega_xyz(probe_x, probe_y),
            dtype=np.float64,
        ).reshape(-1, 3)
        xn, yn = exposure.wcs.omega_xyz_to_norm_torch(
            torch.as_tensor(probe, dtype=torch.float32, device=self.device)
        )
        mixture_center_jacobian = None
        if self.psf_model == "gaussian" or not self.apply_spatial_psf:
            norm_per_pixel = float(torch.hypot(xn[1] - xn[0], yn[1] - yn[0]))
        else:
            mixture_center_jacobian = self._pixel_to_norm_jacobians(
                exposure, np.asarray([cx]), np.asarray([cy])
            )[0]
            norm_per_pixel = float(torch.linalg.vector_norm(
                mixture_center_jacobian[:, 0]
            ))
        sigma_px = (
            float(exposure.psf_fwhm_arcsec)
            / 2.355
            / float(exposure.pixel_scale_arcsec)
            if exposure.psf_fwhm_arcsec is not None
            else 0.0
        )
        psf_sigma_norm = sigma_px * norm_per_pixel
        sigma_norm = math.sqrt(psf_sigma_norm**2 + (norm_per_pixel**2) / 12.0)
        cache = {
            "ll": ll_norm, "w": weights / weights.sum(), "sigma": sigma_norm,
            "band_lam": torch.as_tensor(
                lam_tab, dtype=torch.float32, device=self.device
            ),
            "band_throughput": torch.as_tensor(
                thr_tab, dtype=torch.float32, device=self.device
            ),
            "psf_sigma": psf_sigma_norm,
            "width": int(exposure.image_width), "norm_per_pixel": norm_per_pixel,
            "height": int(exposure.image_height),
            "pixel_sigma": norm_per_pixel / math.sqrt(12.0),
        }
        if self.psf_model == "eigen_mog":
            jacobian = mixture_center_jacobian
            if jacobian is None:
                raise RuntimeError("Missing eigen_mog center Jacobian")
            label = getattr(exposure, "label", None) or f"exposure_id={int(exp_id)}"
            operator = self._eigen_mog_operator()
            sidecar_row = operator.sidecar_row(int(exp_id))
            components = operator.representation.covariances[sidecar_row].reshape(-1, 3)
            component_tensor = torch.as_tensor(
                components, dtype=jacobian.dtype, device=jacobian.device
            )
            native_covariance = torch.stack(
                (component_tensor[:, 0], component_tensor[:, 1],
                 component_tensor[:, 1], component_tensor[:, 2]), -1
            ).reshape(-1, 2, 2)
            frame_x, frame_y = self._frame_jacobian_probe_coordinates(
                int(exposure.image_width), int(exposure.image_height)
            )
            frame_jacobians = torch.cat((
                jacobian.unsqueeze(0),
                self._pixel_to_norm_jacobians(exposure, frame_x[1:], frame_y[1:]),
            ), dim=0)
            max_variation = self._psf_covariance_relative_variation(
                native_covariance,
                jacobian.reshape(1, 1, 2, 2),
                frame_jacobians.reshape(1, 9, 2, 2),
            )
            cache.update({
                "pixel_to_norm_jacobian": jacobian,
                "psf_jacobian_max_rel_variation": max_variation,
                "psf_jacobian_path": "single",
                "psf_jacobian_tile_grid": (1, 1),
                "psf_jacobian_tile_max_rel_variation": max_variation,
                "psf_jacobian_rel_tol": self.psf_jacobian_rel_tol,
                "psf_jacobian_algorithm": self.psf_jacobian_algorithm,
                "psf_jacobian_max_tiles_per_axis": self.psf_jacobian_max_tiles_per_axis,
                "psf_jacobian_cap_policy": self.psf_jacobian_cap_policy,
                "psf_jacobian_cap_reached": False,
            })
            if max_variation > self.psf_jacobian_rel_tol:
                tile_grid, tile_jacobians, tile_variation = self._jacobian_tile_plan(
                    exposure, native_covariance
                )
                cache.update({
                    "psf_jacobian_path": "tiled",
                    "psf_jacobian_tile_grid": tile_grid,
                    "psf_jacobian_tile_max_rel_variation": tile_variation,
                    "pixel_to_norm_jacobian_tiles": tile_jacobians,
                    "psf_jacobian_cap_reached": (
                        tile_grid[0] >= self.psf_jacobian_max_tiles_per_axis
                        or tile_grid[1] >= self.psf_jacobian_max_tiles_per_axis
                    ),
                })
            tile_rows, tile_cols = cache["psf_jacobian_tile_grid"]
            print(
                f"WARNING: PSF Jacobian {label}: max_rel_variation={max_variation:.6g} "
                f"tolerance={self.psf_jacobian_rel_tol:.6g} path={cache['psf_jacobian_path']} "
                f"grid={tile_rows}x{tile_cols} tile_max_rel_variation="
                f"{cache['psf_jacobian_tile_max_rel_variation']:.6g} "
                f"algorithm={self.psf_jacobian_algorithm} cap_policy={self.psf_jacobian_cap_policy} "
                f"cap_reached={cache['psf_jacobian_cap_reached']}",
                flush=True,
            )
        if self.psf_model in {"dgpsf", "mog"}:
            label = getattr(exposure, "label", None) or f"exposure_id={int(exp_id)}"
            if not self.apply_spatial_psf:
                cache.update({
                    "psf_jacobian_max_rel_variation": None,
                    "psf_jacobian_path": "disabled",
                    "psf_jacobian_tile_grid": (1, 1),
                    "psf_jacobian_tile_max_rel_variation": None,
                    "psf_jacobian_rel_tol": self.psf_jacobian_rel_tol,
                    "psf_jacobian_algorithm": self.psf_jacobian_algorithm,
                    "psf_jacobian_max_tiles_per_axis": (
                        self.psf_jacobian_max_tiles_per_axis
                    ),
                    "psf_jacobian_cap_policy": self.psf_jacobian_cap_policy,
                    "psf_jacobian_cap_reached": False,
                })
                print(
                    f"WARNING: PSF Jacobian {label}: path=disabled "
                    "(apply_spatial_psf=false)",
                    flush=True,
                )
            else:
                field_name = (
                    "psf_mixture_dgpsf" if self.psf_model == "dgpsf"
                    else "psf_mixture_kl"
                )
                components = getattr(exposure, field_name, None)
                if components is None:
                    raise ValueError(
                        f"Image exposure {label!r} lacks {field_name}; run "
                        "scripts/backfill_psf_mixture.py on the manifest."
                    )
                # Center and tile Jacobians must use the same float64 one-pixel
                # differencing algorithm; float32 coordinate differencing can
                # lose percent-level derivatives through cancellation.
                jacobian = mixture_center_jacobian
                if jacobian is None:  # guarded by the non-Gaussian branch
                    raise RuntimeError("Missing non-Gaussian center Jacobian.")
                cache["pixel_to_norm_jacobian"] = jacobian
                cache["psf_mixture"] = transform_psf_mixture_to_normalized(
                    components, jacobian, pixel_variance=1.0 / 12.0,
                )
                cache["psf_mixture_no_pixel"] = transform_psf_mixture_to_normalized(
                    components, jacobian,
                )
                native_covariance = self._native_psf_covariances(
                    components, jacobian
                )
                frame_x, frame_y = self._frame_jacobian_probe_coordinates(
                    int(exposure.image_width), int(exposure.image_height)
                )
                frame_jacobians = torch.cat((
                    jacobian.unsqueeze(0),
                    self._pixel_to_norm_jacobians(
                        exposure, frame_x[1:], frame_y[1:]
                    ),
                ), dim=0)
                max_variation = self._psf_covariance_relative_variation(
                    native_covariance,
                    jacobian.reshape(1, 1, 2, 2),
                    frame_jacobians.reshape(1, 9, 2, 2),
                )
                cache["psf_jacobian_max_rel_variation"] = max_variation
                cache["psf_jacobian_path"] = "single"
                cache["psf_jacobian_tile_grid"] = (1, 1)
                cache["psf_jacobian_tile_max_rel_variation"] = max_variation
                cache["psf_jacobian_rel_tol"] = self.psf_jacobian_rel_tol
                cache["psf_jacobian_algorithm"] = self.psf_jacobian_algorithm
                cache["psf_jacobian_max_tiles_per_axis"] = (
                    self.psf_jacobian_max_tiles_per_axis
                )
                cache["psf_jacobian_cap_policy"] = self.psf_jacobian_cap_policy
                cache["psf_jacobian_cap_reached"] = False
                if max_variation > self.psf_jacobian_rel_tol:
                    tile_grid, tile_jacobians, tile_variation = (
                        self._jacobian_tile_plan(exposure, native_covariance)
                    )
                    cache["psf_jacobian_path"] = "tiled"
                    cache["psf_jacobian_tile_grid"] = tile_grid
                    cache["psf_jacobian_tile_max_rel_variation"] = tile_variation
                    cache["psf_jacobian_cap_reached"] = (
                        tile_grid[0] >= self.psf_jacobian_max_tiles_per_axis
                        or tile_grid[1] >= self.psf_jacobian_max_tiles_per_axis
                    )
                    cache["pixel_to_norm_jacobian_tiles"] = tile_jacobians
                    cache["psf_mixture_tiles"] = _stack_point_psf_mixtures([
                        transform_psf_mixture_to_normalized(
                            components, tile_jacobian, pixel_variance=1.0 / 12.0
                        )
                        for tile_jacobian in tile_jacobians
                    ])
                    cache["psf_mixture_no_pixel_tiles"] = _stack_point_psf_mixtures([
                        transform_psf_mixture_to_normalized(
                            components, tile_jacobian
                        )
                        for tile_jacobian in tile_jacobians
                    ])
                tile_rows, tile_cols = cache["psf_jacobian_tile_grid"]
                print(
                    f"WARNING: PSF Jacobian {label}: "
                    f"max_rel_variation={max_variation:.6g} "
                    f"tolerance={self.psf_jacobian_rel_tol:.6g} "
                    f"path={cache['psf_jacobian_path']} "
                    f"grid={tile_rows}x{tile_cols} "
                    f"tile_max_rel_variation="
                    f"{cache['psf_jacobian_tile_max_rel_variation']:.6g} "
                    f"algorithm={self.psf_jacobian_algorithm} "
                    f"cap_policy={self.psf_jacobian_cap_policy} "
                    f"cap_reached={cache['psf_jacobian_cap_reached']}",
                    flush=True,
                )
        if self.patch_raster_enabled:
            h, w = int(exposure.image_height), int(exposure.image_width)
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            omega_all = np.asarray(
                exposure.obs_wcs.pixel_to_omega_xyz(
                    cols.reshape(-1, 1).astype(np.float64),
                    rows.reshape(-1, 1).astype(np.float64),
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            xg, yg = exposure.wcs.omega_xyz_to_norm_torch(
                torch.as_tensor(omega_all, dtype=torch.float32, device=self.device)
            )
            cache["xgrid"] = xg.reshape(h, w)
            cache["ygrid"] = yg.reshape(h, w)
        self._analytic_image_cache[int(exp_id)] = cache
        return cache

    def _spectrum_cache_for_exposure(self, exp_id: int) -> SplatSpectrumCache:
        exposure = self.runtime_by_exposure[int(exp_id)].exposure
        model = self._base_model()
        cache = self._spectrum_cache_by_exposure.get(int(exp_id))
        if cache is not None:
            if (
                cache.exposure_identity != id(exposure)
                or cache.device != model.mu.device
                or cache.dtype != model.mu.dtype
            ):
                raise ValueError(
                    "Existing spectrum cache does not match the current exposure or model."
                )
            return cache
        if not isinstance(exposure.aperture, SkyDiskApertureSpec):
            raise TypeError("Deterministic splat spectra require SkyDiskApertureSpec.")
        cache = build_spectrum_cache(
            exposure,
            exposure.aperture,
            apply_spatial_psf=self.apply_spatial_psf,
            radial_nodes=8,
            angular_nodes=32,
            device=model.mu.device,
            dtype=model.mu.dtype,
        )
        self._spectrum_cache_by_exposure[int(exp_id)] = cache
        return cache

    def _prebuild_spectrum_caches(self) -> None:
        """Build every fiber's spectrum cache before training starts.

        A cache is a handful of small host-to-device copies and a few scalar
        reads, about 7.5 ms of host time. Built on first touch, that lands
        inside a training step, and with 889 fiber exposures a resume drags
        those builds through its first few hundred steps; it is what puts the
        mean step above the median there. The caches are a few tens of
        kilobytes each and are keyed on the exposure identity, so building
        them all up front is the same work in a place where it costs nothing.

        Exposures without a disk aperture are left to the deterministic
        spectral path to reject when it reaches them, exactly as before.
        """
        if not self.analytic_spectra_enabled:
            return
        for exp_id in self.spec_exposure_ids:
            exposure = self.runtime_by_exposure[int(exp_id)].exposure
            if isinstance(exposure.aperture, SkyDiskApertureSpec):
                self._spectrum_cache_for_exposure(int(exp_id))

    def _init_image_variance_columns(
        self, columns: Mapping[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Device tables of per-column (alpha, beta) for poisson_current.

        Every stored image sigma is rebuilt as sqrt(alpha * max(y, 0) + beta)
        before training: coefficients from another store, exposure order or
        column registration, or a store whose sigma is not the data variance,
        stop the run here instead of training on wrong variances.
        """
        started = time.perf_counter()
        image_ids = list(self.image_exposure_ids)
        if sorted(int(exp_id) for exp_id in columns) != image_ids:
            raise ValueError(
                "Per-column image variance coefficients must cover exactly the store's image exposures."
            )
        tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for exp_id in image_ids:
            alpha, beta = (np.asarray(values, dtype=np.float64) for values in columns[exp_id])
            width = int(self.exposures[exp_id].image_width)
            if alpha.shape != (width,) or beta.shape != (width,):
                raise ValueError(
                    f"Image exposure {exp_id} has {width} columns; its variance coefficients have shapes "
                    f"{alpha.shape} and {beta.shape}."
                )
            if not (np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta)) and alpha.min() > 0.0
                    and beta.min() > 0.0):
                raise ValueError(
                    f"Image exposure {exp_id} variance coefficients must be finite and positive."
                )
            tables[exp_id] = (alpha, beta)

        def reconstruction_error(exp_id: int) -> float:
            idx = self._global_indices_for(exp_id, DatumKind.IMAGE_PIXEL)
            cpu = torch.device("cpu")
            return sigma_reconstruction_error(
                self._gather_named_rows("y", idx, device=cpu, dtype=torch.float32).numpy(),
                self._gather_named_rows("sigma", idx, device=cpu, dtype=torch.float32).numpy(),
                self._gather_named_rows("datum_index", idx, device=cpu, dtype=torch.long).numpy(),
                int(self.exposures[exp_id].image_width), *tables[exp_id],
            )

        with ThreadPoolExecutor(max_workers=startup_worker_count(len(image_ids))) as pool:
            errors = list(pool.map(reconstruction_error, image_ids))
        worst = int(np.argmax(errors))
        if not errors[worst] <= SIGMA_RECONSTRUCTION_RTOL:
            raise ValueError(
                "Per-column variance coefficients do not reproduce the stored image sigma of exposure "
                f"{image_ids[worst]}: max relative error {errors[worst]:.3e} > "
                f"{SIGMA_RECONSTRUCTION_RTOL:.0e}."
            )
        widths = [int(self.exposures[exp_id].image_width) for exp_id in image_ids]
        self._image_variance_widths = torch.zeros(image_ids[-1] + 1, dtype=torch.long)
        self._image_variance_offsets = torch.full((image_ids[-1] + 1,), -1, dtype=torch.long)
        self._image_variance_widths[image_ids] = torch.as_tensor(widths, dtype=torch.long)
        self._image_variance_offsets[image_ids] = torch.as_tensor(
            np.concatenate(([0], np.cumsum(widths)[:-1])), dtype=torch.long
        )
        self._image_variance_alpha, self._image_variance_beta = (
            as_tensor_on_device(
                np.concatenate([tables[exp_id][part] for exp_id in image_ids]),
                device=self.device, dtype=torch.float32,
            )
            for part in (0, 1)
        )
        if self.is_primary_rank:
            print(
                f"[image-variance] poisson_current outlier_pull={self.image_variance_outlier_pull} "
                f"exposures={len(image_ids)} max_sigma_reconstruction_rel_err={errors[worst]:.3e} "
                f"(gate {SIGMA_RECONSTRUCTION_RTOL:.0e}) in {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    def _current_model_image_sigma(
        self, exp_ids: torch.Tensor, local: torch.Tensor, y: torch.Tensor, mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        flat = self._image_variance_offsets[exp_ids] + local % self._image_variance_widths[exp_ids]
        flat = copy_tensor_to_device(flat, device=self.device)
        return current_model_image_sigma(
            y, mu, sigma,
            torch.index_select(self._image_variance_alpha, 0, flat),
            torch.index_select(self._image_variance_beta, 0, flat),
            self.image_variance_outlier_pull,
        )

    def _compute_analytic_image_gaussian_terms(self, global_indices: torch.Tensor) -> torch.Tensor:
        """Closed-form image datum terms for splat-type fields (no MC kernels).
        Requires the base model to expose covariances()/positions()/fluxes()."""
        if global_indices.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        model = self._base_model()
        if not callable(getattr(model, "covariances", None)):
            raise ValueError(
                "trainer.analytic_images=true requires a splat-type model exposing "
                "covariances(); got " + type(model).__name__
            )
        exp_ids = self._gather_named_rows(
            "exposure_id", global_indices, device=torch.device("cpu"), dtype=torch.long
        )
        local = self._gather_named_rows(
            "datum_index", global_indices, device=torch.device("cpu"), dtype=torch.long
        )
        capture = bool(getattr(self, "birth_enabled", False)) and torch.is_grad_enabled()
        self._birth_capture = capture
        self._birth_batch_coords = None
        # a stale pending batch from an interrupted earlier call must never
        # be consumed against a later batch that merely matches its length
        self._pending_birth = None
        try:
            mu = self._compute_analytic_image_source_mu(global_indices, exp_ids, local)
        finally:
            self._birth_capture = False
        mu = self._apply_observation_contract(global_indices, mu)
        y = self._gather_named_rows("y", global_indices, device=self.device, dtype=torch.float32)
        sigma = self._gather_named_rows(
            "sigma", global_indices, device=self.device, dtype=torch.float32
        )
        if capture and self._birth_batch_coords is not None:
            with torch.no_grad():
                if getattr(self.data.batch, "additive", None) is not None:
                    additive = self._gather_named_rows(
                        "additive", global_indices,
                        device=self.device, dtype=torch.float32,
                    )
                else:
                    additive = torch.zeros_like(y)
                pixel_x, pixel_y, pixel_band, band_keys = self._birth_batch_coords
                self._birth_batch_coords = None
                mu_detached = mu.detach()
                self._pending_birth = {
                    "px": pixel_x,
                    "py": pixel_y,
                    "band_local": pixel_band,
                    "band_keys": band_keys,
                    "r": (y - mu_detached) / sigma,
                    "m": (mu_detached - additive) / sigma,
                    "f": y - mu_detached,
                }
        if self.image_variance_outlier_pull is not None:
            return datum_unbiased_gaussian_nll(
                y, mu, mu, self._current_model_image_sigma(exp_ids, local, y, mu, sigma)
            )
        return datum_unbiased_gaussian_nll(y, mu, mu, sigma)

    def _compute_analytic_spectral_gaussian_terms(
        self, global_indices: torch.Tensor
    ) -> torch.Tensor:
        if global_indices.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        source_mu = self._compute_analytic_spectral_source_mu(global_indices)
        mu = self._apply_observation_contract(global_indices, source_mu)
        y = self._gather_named_rows("y", global_indices, device=self.device, dtype=torch.float32)
        sigma = self._gather_named_rows(
            "sigma", global_indices, device=self.device, dtype=torch.float32
        )
        return datum_unbiased_gaussian_nll(y, mu, mu, sigma)

    def _compute_analytic_spectral_source_mu(
        self, global_indices: torch.Tensor
    ) -> torch.Tensor:
        if global_indices.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        model = self._base_model()
        if not callable(getattr(getattr(model, "sed", None), "templates", None)):
            raise ValueError(
                "trainer.analytic_spectra=true requires a splat-type model with an SED dictionary."
            )
        batch_context = build_spectrum_batch_context(model)
        exp_ids = self._gather_named_rows(
            "exposure_id", global_indices, device=torch.device("cpu"), dtype=torch.long
        )
        local = self._gather_named_rows(
            "datum_index", global_indices, device=torch.device("cpu"), dtype=torch.long
        )
        source_mu = torch.empty(
            int(global_indices.numel()), device=self.device, dtype=torch.float32
        )
        fast_groups = fast_spectral_groups_enabled()
        if fast_groups:
            order=np.argsort(exp_ids.numpy(),kind='stable')
            sorted_ids=exp_ids.numpy()[order]
            boundaries=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1,len(order)]
            exposure_ids=sorted_ids[boundaries[:-1]].tolist()
        else:
            exposure_ids = exp_ids.unique().tolist()
        caches = [
            self._spectrum_cache_for_exposure(int(exp_id))
            for exp_id in exposure_ids
        ]
        candidates = batched_fiber_candidates(
            model,
            caches,
            n_sigma=6.0,
            splat_eigen_max=batch_context.splat_eigen_max,
        )
        selected_cpu_by_exposure = []
        local_cpu_by_exposure = []
        bin_counts = []
        offsets = [0]
        for exp_id, cache in ([] if fast_groups else zip(exposure_ids, caches)):
            selected = (exp_ids == exp_id).nonzero(as_tuple=True)[0]
            local_cpu = local[selected]
            selected_cpu_by_exposure.append(selected)
            local_cpu_by_exposure.append(local_cpu)
            bin_counts.append(int(cache.lam_edges.shape[0]))
            offsets.append(offsets[-1] + int(local_cpu.numel()))
        if fast_groups:
            selected_staged=torch.from_numpy(order)
            local_staged=torch.from_numpy(local.numpy()[order])
            offsets=boundaries.tolist()
            bin_counts=[int(cache.lam_edges.shape[0]) for cache in caches]
        else:
            selected_staged = torch.cat(selected_cpu_by_exposure)
            local_staged = torch.cat(local_cpu_by_exposure)
        # Range-check the bin indices while they are still CPU tensors, so the
        # fiber evaluation itself can skip its device-synced check. One
        # comparison over the whole batch against each datum's own bin count:
        # a 64-datum batch spans about 62 exposures, and per exposure this was
        # two reductions and two host scalar reads.
        bin_limit = torch.from_numpy(
            np.repeat(np.asarray(bin_counts, dtype=np.int64), np.diff(offsets))
        )
        if bool(
            torch.any((local_staged < 0) | (local_staged >= bin_limit))
        ):
            raise ValueError("Spectral bin index is out of bounds.")
        if self.device.type == "cuda":
            selected_staged = selected_staged.pin_memory()
            local_staged = local_staged.pin_memory()
        selected_device = selected_staged.to(
            device=self.device, non_blocking=True
        )
        local_device = local_staged.to(device=self.device, non_blocking=True)
        # A 64-datum spectral batch spans about 62 distinct fiber exposures, so
        # evaluating them one at a time paid the fiber machinery's whole fixed
        # cost per datum and made the training step kernel-launch bound.
        bins_by_exposure = [
            local_device[offsets[i]: offsets[i + 1]]
            for i in range(len(exposure_ids))
        ]
        if flat_fiber_path_enabled():
            predictions = flat_fiber_spectra(
                model,
                caches,
                candidates,
                bins_by_exposure,
                loglam_min=self.loglam_min,
                loglam_max=self.loglam_max,
                batch_context=batch_context,
            )
        else:
            predictions = torch.cat(
                batched_fiber_spectra(
                    model,
                    caches,
                    candidates,
                    bins_by_exposure,
                    loglam_min=self.loglam_min,
                    loglam_max=self.loglam_max,
                    batch_context=batch_context,
                )
            )
        # The per-exposure blocks are disjoint slices of one destination, so
        # one scatter over the concatenated batch writes exactly what a scatter
        # per exposure wrote, without a kernel per exposure.
        source_mu[selected_device] = self._apply_spectral_nuisance_batch(
            exposure_ids, offsets, local_device, predictions
        ).to(torch.float32)
        return source_mu

    def _apply_spectral_nuisance_batch(
        self,
        exposure_ids: Sequence[int],
        offsets: Sequence[int],
        local_spec_indices: torch.Tensor,
        spec_mu: torch.Tensor,
    ) -> torch.Tensor:
        """_apply_spectral_nuisance over a batch's concatenated exposures.

        The calibrator keeps its per-exposure throughput-mix tables, so the
        scales are still gathered one exposure at a time; the prediction itself
        is scaled once for the whole batch.
        """
        if self.spectral_calibrator is None:
            return spec_mu
        if batched_spectral_calibration_enabled():
            scales = self.spectral_calibrator.scale_for_batch(
                exposure_ids, offsets, local_spec_indices
            )
            return spec_mu * scales.to(dtype=spec_mu.dtype)
        scales = torch.cat(
            [
                self.spectral_calibrator.scale_for_indices(
                    int(exp_id), local_spec_indices[offsets[i]: offsets[i + 1]]
                )
                for i, exp_id in enumerate(exposure_ids)
            ]
        )
        return spec_mu * scales.to(dtype=spec_mu.dtype)

    @staticmethod
    def _summarize_fixed_residuals(
        y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> Dict[str, float | int]:
        residual = y.to(torch.float64) - mu.to(torch.float64)
        sigma64 = sigma.to(torch.float64)
        normalized = residual / sigma64
        gaussian_nll = (
            0.5 * normalized.square()
            + torch.log(sigma64)
            + 0.5 * math.log(2.0 * math.pi)
        )
        return {
            "n": int(y.numel()),
            "gaussian_nll_mean": float(gaussian_nll.mean().item()),
            "chi2_mean": float(normalized.square().mean().item()),
            "normalized_residual_mean": float(normalized.mean().item()),
            "normalized_residual_std": float(
                normalized.std(unbiased=False).item()
            ),
            "mae": float(residual.abs().mean().item()),
            "rmse": float(residual.square().mean().sqrt().item()),
        }

    @torch.no_grad()
    def evaluate_fixed_holdout(self) -> Dict[str, Dict[str, Dict[str, float | int]]]:
        if not self.analytic_images_enabled or not self.analytic_spectra_enabled:
            raise ValueError(
                "Fixed holdout evaluation requires deterministic analytic image "
                "and spectral operators."
            )
        result: Dict[str, Dict[str, Dict[str, float | int]]] = {
            "image": {},
            "spectrum": {},
        }
        for band, indices_cpu in self.image_holdout_indices_by_band.items():
            if indices_cpu.numel() == 0:
                continue
            indices = indices_cpu.to(dtype=torch.long, device="cpu")
            exp_ids = self._gather_named_rows(
                "exposure_id",
                indices,
                device=torch.device("cpu"),
                dtype=torch.long,
            )
            local = self._gather_named_rows(
                "datum_index",
                indices,
                device=torch.device("cpu"),
                dtype=torch.long,
            )
            source_mu = self._compute_analytic_image_source_mu(
                indices, exp_ids, local
            )
            mu = self._apply_observation_contract(indices, source_mu)
            y = self._gather_named_rows(
                "y", indices, device=self.device, dtype=torch.float32
            )
            sigma = self._gather_named_rows(
                "sigma", indices, device=self.device, dtype=torch.float32
            )
            result["image"][str(band)] = self._summarize_fixed_residuals(
                y, mu, sigma
            )

        if self.spec_holdout_indices.numel() > 0:
            indices = self.spec_holdout_indices.to(
                dtype=torch.long, device="cpu"
            )
            source_mu = self._compute_analytic_spectral_source_mu(indices)
            mu = self._apply_observation_contract(indices, source_mu)
            y = self._gather_named_rows(
                "y", indices, device=self.device, dtype=torch.float32
            )
            sigma = self._gather_named_rows(
                "sigma", indices, device=self.device, dtype=torch.float32
            )
            result["spectrum"]["all"] = self._summarize_fixed_residuals(
                y, mu, sigma
            )
        return result

    def _analytic_image_quadrature_rule(
        self,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, str]:
        quadrature_order = self.analytic_image_quadrature_order
        if quadrature_order == 1:
            offsets_x = np.zeros((1, 1), dtype=np.float64)
            offsets_y = np.zeros((1, 1), dtype=np.float64)
            quadrature_weights = torch.ones(
                1, dtype=torch.float32, device=self.device
            )
            sigma_key = "sigma"
        else:
            nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
            nodes = 0.5 * nodes
            weights = 0.5 * weights
            offsets_x, offsets_y = np.meshgrid(nodes, nodes, indexing="xy")
            offsets_x = offsets_x.reshape(1, -1)
            offsets_y = offsets_y.reshape(1, -1)
            quadrature_weights = as_tensor_on_device(
                np.outer(weights, weights).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            sigma_key = "psf_sigma"
        return offsets_x, offsets_y, quadrature_weights, sigma_key

    def _ensure_pixel_norm_tables(
        self,
        run_exp_ids: np.ndarray,
        offsets_x: np.ndarray,
        offsets_y: np.ndarray,
        samples_per_pixel: int,
    ) -> bool:
        """Tabulate whole frames of normalized model coordinates on demand.

        The per-step path ran astropy's pixel->sky transform on the batch's own
        pixels, one serial call per exposure per step, for coordinates that
        never change. Both halves of that map are per-point - WCSLIB transforms
        each pixel independently and omega_xyz_to_norm_torch is elementwise -
        so a whole-frame table computed once through the same calls holds
        exactly the float32 values the per-batch call produced, and the step
        becomes one gather.

        Only the one-node quadrature is tabulated: a higher order needs a
        separate entry per node per pixel, which no longer fits, and keeps the
        per-batch transform. Frames are filled the first time a batch touches
        them, so an exposure the run never trains on is never transformed.
        """
        if int(samples_per_pixel) != 1:
            return False
        if self._pixel_norm_x is None:
            exposure_ids = [int(exp_id) for exp_id in self.image_exposure_ids]
            if not exposure_ids:
                return False
            sizes = [
                int(self.runtime_by_exposure[exp_id].exposure.image_height)
                * int(self.runtime_by_exposure[exp_id].exposure.image_width)
                for exp_id in exposure_ids
            ]
            bases = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
            self._pixel_norm_base = np.zeros(max(exposure_ids) + 1, dtype=np.int64)
            self._pixel_norm_base[exposure_ids] = bases[:-1]
            self._pixel_norm_filled = np.zeros(max(exposure_ids) + 1, dtype=bool)
            self._pixel_norm_x = torch.empty(
                int(bases[-1]), dtype=torch.float32, device=self.device
            )
            self._pixel_norm_y = torch.empty_like(self._pixel_norm_x)
        # Every step asks this question about ~400 exposures, so ask it of
        # numpy rather than of a python loop.
        missing = run_exp_ids[~self._pixel_norm_filled[run_exp_ids]]
        if missing.size == 0:
            return True

        def _fill(exp_id: int) -> None:
            exp_id = int(exp_id)
            exposure = self.runtime_by_exposure[exp_id].exposure
            width = int(exposure.image_width)
            height = int(exposure.image_height)
            columns = np.tile(np.arange(width, dtype=np.float64), height)[:, None]
            rows = np.repeat(np.arange(height, dtype=np.float64), width)[:, None]
            omega = np.asarray(
                exposure.obs_wcs.pixel_to_omega_xyz(
                    columns + offsets_x, rows + offsets_y
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            x_norm, y_norm = exposure.wcs.omega_xyz_to_norm_torch(
                torch.as_tensor(omega, dtype=torch.float32, device=self.device)
            )
            base = int(self._pixel_norm_base[exp_id])
            self._pixel_norm_x[base : base + x_norm.numel()] = x_norm
            self._pixel_norm_y[base : base + y_norm.numel()] = y_norm

        # Exposures are independent and write disjoint slices; the transforms
        # release the GIL, so this only shortens the wall clock.
        with ThreadPoolExecutor(max_workers=startup_worker_count(len(missing))) as pool:
            list(pool.map(_fill, missing))
        self._pixel_norm_filled[missing] = True
        return True

    def _eigen_mog_operator(self):
        if self._eigen_mog_renderer is None:
            from .eigen_mog_renderer import EigenMogRenderer

            self._eigen_mog_renderer = EigenMogRenderer(self)
        return self._eigen_mog_renderer

    def _compute_analytic_image_source_mu(
        self, global_indices: torch.Tensor, exp_ids: torch.Tensor, local: torch.Tensor
    ) -> torch.Tensor:
        model = self._base_model()
        mu = torch.empty(int(global_indices.numel()), device=self.device, dtype=torch.float32)
        offsets_x, offsets_y, quadrature_weights, sigma_key = (
            self._analytic_image_quadrature_rule()
        )
        samples_per_pixel = int(quadrature_weights.numel())
        # Group the batch by band, not exposure: exposures of one band share
        # bandpass amplitudes, so the tile index and pair evaluation run once
        # per band per step (5x) instead of once per exposure (410x on full84).
        # Per-exposure seeing enters as a per-point PSF sigma.
        #
        # The per-exposure bookkeeping is CPU-bound and used to serialize with
        # the GPU: one device sync per exposure plus a serial astropy
        # transform. Do one device-to-host transfer for the whole batch,
        # group in numpy, fan the astropy pixel->sky transforms over a thread
        # pool, and evaluate omega->norm once per distinct model-plane WCS
        # instead of once per exposure. The parallelism assumes each
        # exposure's obs_wcs is a distinct object (true for the SDSS
        # adapter); a backend sharing one stateful obs_wcs across exposures
        # would need those calls serialized.
        exp_np = exp_ids.detach().cpu().numpy()
        local_np = local.detach().cpu().numpy()
        if bool(np.all(np.diff(exp_np) >= 0)):
            # The hierarchical sampler emits exposure-major batches, so the
            # stable argsort is usually the identity: skip the sort and the
            # destination scatter entirely.
            order_np = None
            sorted_exp = exp_np
        else:
            order_np = np.argsort(exp_np, kind="stable")
            sorted_exp = exp_np[order_np]
        sorted_local = local_np if order_np is None else local_np[order_np]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_exp)) + 1))
        stops = np.concatenate((starts[1:], [int(sorted_exp.size)]))
        run_exp_ids = [int(e) for e in sorted_exp[starts]]
        for exp_id in run_exp_ids:
            self._analytic_cache_for_exposure(exp_id)

        tabulated = self._ensure_pixel_norm_tables(
            sorted_exp[starts], offsets_x, offsets_y, samples_per_pixel
        )
        omega_all = None
        if not tabulated:
            def _exposure_omega(start: int, stop: int, exp_id: int) -> np.ndarray:
                cache = self._analytic_cache_for_exposure(exp_id)
                exposure = self.runtime_by_exposure[exp_id].exposure
                idx = sorted_local[start:stop]
                px = np.mod(idx, cache["width"]).astype(np.float64)[:, None]
                py = np.floor_divide(idx, cache["width"]).astype(np.float64)[:, None]
                return np.asarray(
                    exposure.obs_wcs.pixel_to_omega_xyz(
                        px + offsets_x, py + offsets_y
                    ),
                    dtype=np.float64,
                ).reshape(-1, 3)

            # Serial on purpose. A 10,000-pixel batch spread over 406 exposures
            # produces roughly 400 runs, so a thread pool here submits and awaits
            # ~400 futures per step for tasks of about 70 us each, and the queue
            # and GIL handoffs cost more than the work. `map` yields the same
            # results in the same order, so the arrays are unchanged.
            omega_parts = list(
                map(
                    _exposure_omega,
                    starts.tolist(), stops.tolist(), run_exp_ids,
                )
            )
            omega_all = as_tensor_on_device(
                np.concatenate(omega_parts, axis=0),
                dtype=torch.float32, device=self.device,
            )

        band_keys: list = []
        band_id_by_key: Dict[Any, int] = {}
        band_lam_throughput: Dict[Any, tuple] = {}
        run_band_ids = []
        run_sigmas = []
        run_mixtures = []
        eigen_mog_jacobians = []
        use_psf_mixture = self.apply_spatial_psf and self.psf_model in {"dgpsf", "mog"}
        wcs_groups: Dict[int, tuple] = {}
        point_counts = (stops - starts).astype(np.int64)
        fast_eigen_context = self.psf_model == 'eigen_mog' and tabulated and eigen_fast_context_enabled()
        if fast_eigen_context:
            band_keys,band_lam_throughput,run_band_ids=self._eigen_mog_operator().batch_metadata(run_exp_ids)
        for run_index, exp_id in enumerate([] if fast_eigen_context else run_exp_ids):
            cache = self._analytic_cache_for_exposure(exp_id)
            exposure = self.runtime_by_exposure[exp_id].exposure
            band_key = self.image_band_by_exposure.get(exp_id, exp_id)
            if band_key not in band_id_by_key:
                band_id_by_key[band_key] = len(band_keys)
                band_keys.append(band_key)
                band_lam_throughput[band_key] = (
                    cache["band_lam"], cache["band_throughput"]
                )
            run_band_ids.append(band_id_by_key[band_key])
            if not use_psf_mixture:
                # The non-analytic path always retains detector-pixel
                # integration when PSF convolution is disabled. Analytic order
                # one uses the established covariance fold; higher orders
                # integrate the square pixel explicitly at their quadrature
                # nodes and therefore add no covariance here.
                run_sigmas.append(float(
                    cache[sigma_key]
                    if self.apply_spatial_psf
                    else cache["pixel_sigma"] if sigma_key == "sigma" else 0.0
                ))
            else:
                mixture_key = "psf_mixture" if sigma_key == "sigma" else "psf_mixture_no_pixel"
                idx = sorted_local[starts[run_index]:stops[run_index]]
                pixel_x = (
                    np.mod(idx, cache["width"]).astype(np.float64)[:, None]
                    + offsets_x
                ).reshape(-1)
                pixel_y = (
                    np.floor_divide(idx, cache["width"]).astype(np.float64)[:, None]
                    + offsets_y
                ).reshape(-1)
                run_mixtures.append(self._point_psf_mixture(
                    cache, mixture_key, pixel_x, pixel_y
                ))
            if self.psf_model == "eigen_mog":
                idx = sorted_local[starts[run_index]:stops[run_index]]
                eigen_mog_jacobians.append(self._point_jacobians(
                    cache,
                    np.mod(idx, cache["width"]).astype(np.float64),
                    np.floor_divide(idx, cache["width"]).astype(np.float64),
                ))
            wcs_groups.setdefault(
                id(exposure.wcs), (exposure.wcs, [])
            )[1].append(run_index)

        sample_counts = point_counts * samples_per_pixel
        point_group = None
        if use_psf_mixture:
            point_group = torch.repeat_interleave(
                torch.arange(len(sample_counts), device=self.device),
                as_tensor_on_device(sample_counts, device=self.device),
            )
        # Range-check the band ids here, on the ~one-per-run host list they
        # are built from, so the packed prediction can skip its device-side
        # check. That check is two blocking host syncs landing in the middle
        # of the image forward, where the host is otherwise running ahead of
        # the GPU. np.repeat cannot introduce a value that is not already in
        # run_band_ids, so this is the same test on the same values.
        run_band_ids_np = np.asarray(run_band_ids, dtype=np.int64)
        present_band_ids = run_band_ids_np[np.asarray(sample_counts) > 0]
        if present_band_ids.size and (
            int(present_band_ids.min()) < 0
            or int(present_band_ids.max()) >= len(band_keys)
        ):
            raise ValueError("point_band contains an out-of-range band index.")
        point_band = as_tensor_on_device(
            np.repeat(run_band_ids_np, sample_counts),
            device=self.device,
        )
        if fast_eigen_context:
            sig = None
        elif not use_psf_mixture:
            sig = as_tensor_on_device(
                np.repeat(np.asarray(run_sigmas, dtype=np.float32), sample_counts),
                device=self.device,
            )
        else:
            mixture_weights = torch.cat([
                mixture.weights.expand(int(count), -1)
                if mixture.weights.shape[0] == 1 else mixture.weights
                for mixture, count in zip(run_mixtures, sample_counts)
            ], dim=0)
            all_isotropic = all(mixture.isotropic for mixture in run_mixtures)

            def covariance_rows(mixture: PointPSFMixture) -> torch.Tensor:
                if all_isotropic or not mixture.isotropic:
                    return mixture.covariances
                zero = torch.zeros_like(mixture.covariances)
                return torch.stack(
                    (mixture.covariances, zero, mixture.covariances), dim=-1
                )
            mixture_covariances = torch.cat([
                covariance_rows(mixture).expand(
                    (int(count),) + covariance_rows(mixture).shape[1:]
                )
                if covariance_rows(mixture).shape[0] == 1
                else covariance_rows(mixture)
                for mixture, count in zip(run_mixtures, sample_counts)
            ], dim=0)
            sig = PointPSFMixture(mixture_weights, mixture_covariances)
        if tabulated:
            table_index = as_tensor_on_device(
                self._pixel_norm_base[sorted_exp] + sorted_local, device=self.device
            )
            xs = self._pixel_norm_x[table_index]
            ys = self._pixel_norm_y[table_index]
        elif len(wcs_groups) == 1:
            ((model_wcs, _),) = wcs_groups.values()
            xs, ys = model_wcs.omega_xyz_to_norm_torch(omega_all)
        else:
            sample_offsets = np.concatenate(([0], np.cumsum(sample_counts)))
            xs = torch.empty(
                omega_all.shape[0], device=self.device, dtype=omega_all.dtype
            )
            ys = torch.empty_like(xs)
            for model_wcs, run_list in wcs_groups.values():
                seg = as_tensor_on_device(
                    np.concatenate([
                        np.arange(sample_offsets[r], sample_offsets[r + 1])
                        for r in run_list
                    ]),
                    device=self.device,
                )
                x_seg, y_seg = model_wcs.omega_xyz_to_norm_torch(omega_all[seg])
                xs[seg] = x_seg
                ys[seg] = y_seg

        use_cuda_events = self._speed_profile_enabled and self.device.type == "cuda"
        if use_cuda_events:
            section_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            section_events[0].record()
        section_wall_start = time.perf_counter()
        eigen_mog_context = None
        if self.psf_model == "eigen_mog":
            eigen_mog_operator = self._eigen_mog_operator()
            eigen_mog_context = eigen_mog_operator.prepare(
                sorted_exp, sorted_local, None if fast_eigen_context else torch.cat(eigen_mog_jacobians, dim=0)
            )
            packed_index = eigen_mog_operator.index(
                model, xs, ys, eigen_mog_context, self.pair_n_sigma
            )
        else:
            packed_index = self._build_point_index(
                model, xs, ys, sig, point_group=point_group
            )
        if use_cuda_events:
            section_events[1].record()
            self._record_speed_cuda_events(
                "image_forward_index_sec", section_events[0], section_events[1]
            )
        else:
            self._record_speed_wall(
                "image_forward_index_sec", time.perf_counter() - section_wall_start
            )
        section_wall_start = time.perf_counter()
        gathered_parameters = None
        if fused_splat_gather_enabled() and model.sparse_gradients_enabled:
            gathered_parameters = model.gather_splat_parameters(
                packed_index.active_splat, sorted_unique=True
            )
        band_rows = None
        if self.amp_band_subsets:
            with torch.no_grad():
                member = torch.zeros(
                    packed_index.active_splat.numel(),
                    len(band_keys),
                    dtype=torch.bool,
                    device=self.device,
                )
                if device_index_fill_enabled() and member.is_cuda:
                    flat_membership = (
                        packed_index.pair_active_splat * len(band_keys)
                        + point_band[packed_index.pair_point]
                    )
                    member.view(-1).index_fill_(0, flat_membership, True)
                else:
                    member[
                        packed_index.pair_active_splat,
                        point_band[packed_index.pair_point],
                    ] = True
                # One nonzero over the band-major membership instead of one per
                # band: nonzero has to synchronize to size its output, so the
                # per-band form stalls the pipeline five times per step. Band
                # b owns flat positions [b * n_active, (b + 1) * n_active), and
                # nonzero returns them in ascending flat order, so splitting by
                # the per-band counts reproduces each column's ascending unique
                # rows exactly.
                n_active = packed_index.active_splat.numel()
                band_counts = member.sum(dim=0)
                flat_rows = member.t().contiguous().view(-1).nonzero(
                    as_tuple=True
                )[0]
                band_rows = list(torch.split(
                    flat_rows - (flat_rows // n_active) * n_active,
                    band_counts.tolist(),
                ))
        amplitudes_by_band = splat_band_amplitudes(
            model,
            {band_key: band_lam_throughput[band_key] for band_key in band_keys},
            loglam_min=self.loglam_min,
            loglam_max=self.loglam_max,
            indices=packed_index.active_splat,
            indices_sorted_unique=True,
            band_row_indices=band_rows,
            gathered_parameters=gathered_parameters,
        )
        amplitude_matrix = torch.stack(
            [amplitudes_by_band[band_key] for band_key in band_keys], dim=1
        )
        if use_cuda_events:
            section_events[2].record()
            self._record_speed_cuda_events(
                "image_forward_amplitude_sec", section_events[1], section_events[2]
            )
        else:
            self._record_speed_wall(
                "image_forward_amplitude_sec", time.perf_counter() - section_wall_start
            )
        section_wall_start = time.perf_counter()
        mixture_support_stats: Dict[str, int] = {}
        if eigen_mog_context is not None:
            values = eigen_mog_operator.predict(
                model, xs, ys, point_band, amplitude_matrix, packed_index, eigen_mog_context,
                gathered_parameters=gathered_parameters,
            )
        elif not use_psf_mixture:
            values = packed_multiband_point_prediction(
                model, xs, ys, sig, point_band, amplitude_matrix, packed_index,
                validate_bands=False,
                gathered_parameters=gathered_parameters,
            )
        else:
            values = packed_mixture_multiband_point_prediction(
                model, xs, ys, sig, point_band, amplitude_matrix, packed_index,
                n_sigma=self.pair_n_sigma,
                support_stats=mixture_support_stats,
                validate_bands=False,
                gathered_parameters=gathered_parameters,
            )
        values = values.reshape(-1, samples_per_pixel)
        values = (values * quadrature_weights.unsqueeze(0)).sum(dim=1)
        if use_cuda_events:
            section_events[3].record()
            self._record_speed_cuda_events(
                "image_forward_raster_sec", section_events[2], section_events[3]
            )
        else:
            self._record_speed_wall(
                "image_forward_raster_sec", time.perf_counter() - section_wall_start
            )
        if order_np is None:
            mu = values.to(torch.float32)
        else:
            destination = as_tensor_on_device(order_np, device=self.device)
            mu[destination] = values.to(torch.float32)
        packed_image_stats = {
            "model_splats": int(model.mu.shape[0]),
            "active_splats": int(packed_index.active_splat.numel()),
            "candidate_pairs": int(
                packed_index.candidate_pairs
                if packed_index.candidate_pairs is not None
                else packed_index.pair_point.numel()
            ),
            "retained_pairs": int(
                mixture_support_stats.get(
                    "retained_pairs",
                    packed_index.retained_pairs
                    if packed_index.retained_pairs is not None
                    else packed_index.pair_point.numel(),
                )
            ),
            "points": int(xs.numel()),
        }
        if band_rows is not None:
            packed_image_stats["band_rows_total"] = sum(
                int(rows.numel()) for rows in band_rows
            )
        self._last_packed_image_stats = packed_image_stats
        if getattr(self, "_birth_capture", False):
            if len(wcs_groups) > 1:
                raise RuntimeError(
                    "Residual birth accumulates on a single model-WCS grid; "
                    f"this batch spans {len(wcs_groups)} groups."
                )
            with torch.no_grad():
                spp = int(samples_per_pixel)
                pixel_x = xs.detach().view(-1, spp).mean(dim=1)
                pixel_y = ys.detach().view(-1, spp).mean(dim=1)
                pixel_band = as_tensor_on_device(
                    np.repeat(
                        np.asarray(run_band_ids, dtype=np.int64), point_counts
                    ),
                    device=self.device,
                )
                if order_np is not None:
                    original = as_tensor_on_device(order_np, device=self.device)
                    for source, name in (
                        (pixel_x, "x"), (pixel_y, "y"), (pixel_band, "band"),
                    ):
                        restored = torch.empty_like(source)
                        restored[original] = source
                        if name == "x":
                            pixel_x = restored
                        elif name == "y":
                            pixel_y = restored
                        else:
                            pixel_band = restored
                self._birth_batch_coords = (
                    pixel_x, pixel_y, pixel_band, list(band_keys)
                )
        return mu

    def fit(self, on_step_end: Callable[[int, Dict[str, float]], None] | None = None):
        stats: Dict[str, float] = {}

        if (
            is_distributed_active(self.distributed_context)
            and callable(
                getattr(self._base_model(), "param_groups_gsplat", None)
            )
        ):
            self._verify_distributed_state_signatures("initialization")
        steps_per_epoch = self._steps_per_epoch()
        max_possible_steps = self.epochs * steps_per_epoch
        if self.max_steps > 0:
            planned_steps = min(self.max_steps, max_possible_steps) if max_possible_steps > 0 else 0
        else:
            planned_steps = max_possible_steps
        global_step = int(getattr(self, "resume_start_step", 0))
        start_step = global_step
        self._set_calibration_lr_for_update(global_step + 1)
        epochs_ran = 0
        print(
            "Training budget: "
            f"target absolute step={planned_steps}, current step={global_step}, "
            f"steps remaining this invocation={max(0, planned_steps - global_step)}",
            flush=True,
        )
        if global_step >= planned_steps:
            print(
                "Training budget already satisfied; exiting without optimizer steps.",
                flush=True,
            )
        progress = (
            tqdm(
                total=planned_steps, initial=global_step,
                desc="Training", unit="step",
            )
            if self.is_primary_rank
            else _NullProgressBar()
        )
        epoch_offset = int(self._resume_epoch_offset)
        with progress as pbar:
            for epoch in range(self.epochs):
                absolute_epoch = epoch_offset + epoch
                skip_batches = self._resume_epoch_batch_offset if epoch == 0 else 0
                if self.image_sampler is not None:
                    self.image_sampler.set_epoch(absolute_epoch)
                if self.spec_sampler is not None:
                    self.spec_sampler.set_epoch(absolute_epoch)
                if self.checkpoint_rng_state_enabled:
                    for sampler in self._checkpoint_epoch_samplers:
                        sampler.set_epoch(absolute_epoch)
                self._epochs_completed = absolute_epoch
                self._steps_into_epoch = int(skip_batches)
                epoch_had_steps = False
                epoch_loss_sum = 0.0
                epoch_loss_count = 0
                epoch_batches = self._iter_epoch_batches(skip_batches=skip_batches)
                while True:
                    if global_step >= planned_steps:
                        break
                    data_wait_start = time.perf_counter()
                    try:
                        img_batch_cpu, spec_batch_cpu = next(epoch_batches)
                    except StopIteration:
                        break
                    data_wait_sec = time.perf_counter() - data_wait_start
                    if self._speed_profile_enabled and self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                    transfer_start = time.perf_counter()
                    image_indices = (
                        img_batch_cpu.to(dtype=torch.long, device="cpu")
                        if img_batch_cpu is not None
                        else torch.empty((0,), dtype=torch.long)
                    )
                    spec_indices = (
                        spec_batch_cpu.to(dtype=torch.long, device="cpu")
                        if spec_batch_cpu is not None
                        else torch.empty((0,), dtype=torch.long)
                    )
                    image_weights = self._pending_image_importance_weights
                    self._pending_image_importance_weights = None
                    transfer_sec = time.perf_counter() - transfer_start
                    train_step_start = time.perf_counter()
                    compute_band_losses = (
                        self.band_loss_log_every_n_steps > 0
                        and global_step % self.band_loss_log_every_n_steps == 0
                    )
                    local_stats = self._train_step(
                        image_indices, spec_indices, image_weights=image_weights,
                        compute_band_losses=compute_band_losses,
                        global_step=global_step,
                    )
                    refine_report = self._post_optimizer_step(global_step + 1)
                    if refine_report:
                        local_stats["refine"] = refine_report
                    if (self.lr_scheduler is not None and self.lr_scheduler_cfg["type"] == "OneCycleLR"):
                        self.lr_scheduler.step()
                    local_train_step_sec = time.perf_counter() - train_step_start
                    local_total_step_sec = data_wait_sec + transfer_sec + local_train_step_sec
                    stats, reduced_times = self._reduce_step_logging(
                        local_stats, data_wait_sec=data_wait_sec, transfer_sec=transfer_sec,
                        train_step_sec=local_train_step_sec, total_step_sec=local_total_step_sec,
                    )
                    train_step_sec = float(reduced_times["train_step_sec"])
                    total_step_sec = float(reduced_times["total_step_sec"])
                    loss_value = float(stats.get("total_loss", float("nan")))
                    if np.isfinite(loss_value):
                        epoch_loss_sum += loss_value
                        epoch_loss_count += 1
                    self._record_history(stats)
                    if global_step % max(self.eval_interval, 20) == 0:
                        pbar.write(self._format_stats(epoch, global_step, stats))
                    self._log_wandb(global_step, stats)
                    pbar.set_postfix(
                        loss=stats.get("total_loss", 0.0), img=stats.get("image_loss", 0.0),
                        spec=stats.get("spec_loss", 0.0),
                    )
                    pbar.update(1)
                    global_step += 1
                    self._steps_into_epoch += 1
                    epoch_had_steps = True
                    self._log_speed(
                        epoch=epoch, global_step=global_step, stats=stats,
                        data_wait_sec=float(reduced_times["data_wait_sec"]),
                        transfer_sec=float(reduced_times["transfer_sec"]), train_step_sec=train_step_sec,
                        total_step_sec=total_step_sec,
                    )
                    if on_step_end is not None:
                        on_step_end(global_step, stats)

                if epoch_had_steps:
                    epochs_ran += 1
                    epoch_mean_total_loss = float("nan")
                    if epoch_loss_count > 0:
                        epoch_mean_total_loss = epoch_loss_sum / float(epoch_loss_count)
                    stats["epoch_mean_total_loss"] = epoch_mean_total_loss
                    epoch_payload: Dict[str, Any] | None = None
                    if np.isfinite(epoch_mean_total_loss):
                        epoch_payload = {"train/epoch_mean_total_loss": epoch_mean_total_loss}
                    if (
                        self.lr_scheduler is not None
                        and self.lr_scheduler_cfg["type"] == "ReduceLROnPlateau"
                    ):
                        if np.isfinite(epoch_mean_total_loss):
                            self.lr_scheduler.step(epoch_mean_total_loss)
                        else:
                            warnings.warn(
                                (
                                    "Skipping ReduceLROnPlateau step at epoch "
                                    f"{epoch + 1}: epoch_mean_total_loss is non-finite."
                                ),
                                RuntimeWarning,
                                stacklevel=2,
                            )
                    self._log_wandb(global_step - 1, stats, force=True, extra_payload=epoch_payload)
                    self._maybe_log_progress(epoch, global_step, planned_steps, stats, pbar)

                if global_step >= planned_steps:
                    break

        if global_step > 0:
            self._log_wandb(global_step - 1, stats, force=True)

        # A speculative importance draw submitted for a step that never ran
        # would otherwise swallow its exception; surface it rather than
        # returning a state whose sampler silently died.
        pending = getattr(self, "_importance_pending_batch", None)
        if pending is not None:
            if pending.done() and pending.exception() is not None:
                raise pending.exception()
            if not self.checkpoint_rng_state_enabled:
                self._importance_pending_batch = None
                pending.cancel()

        return TrainerState(
            history=self.history, band_losses=stats.get("band_losses", {}), amplitude=1.0,
            epochs_ran=epochs_ran, steps_ran=global_step,
            steps_ran_this_invocation=global_step - start_step,
        )

    def _maybe_log_progress(
        self, epoch: int, global_step: int, planned_steps: int, stats: Dict[str, float], pbar: tqdm,
    ) -> None:
        if self.epoch_log_every is None:
            return
        epoch_num = epoch + 1
        if epoch_num % self.epoch_log_every != 0:
            return
        pct = 100.0 * global_step / max(1, planned_steps)
        rate = pbar.format_dict.get("rate", None)
        if rate is None:
            rate_str = "?"
        else:
            rate_str = f"{rate:.2f}step/s"
        pbar.write(
            (
                f"Training: {pct:5.1f}% ({global_step}/{planned_steps}) "
                f"epoch={epoch_num}/{self.epochs} "
                f"{rate_str}, img={stats.get('image_loss', 0.0):.3g}, "
                f"loss={stats.get('total_loss', 0.0):.3g}, "
                f"spec={stats.get('spec_loss', 0.0):.3g}"
            )
        )

    def _compute_step_gaussian_terms(
        self, image_indices: torch.Tensor, spec_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Evaluate the image and spectral branches for one training step."""
        n_img = int(image_indices.numel())
        n_spec = int(spec_indices.numel())
        self._fiber_side_stream_used = False
        if n_img + n_spec == 0:
            return None
        if not (self.analytic_images_enabled or self.analytic_spectra_enabled):
            if n_img > 0 and n_spec > 0:
                term_indices = torch.cat((image_indices, spec_indices), dim=0)
            elif n_img > 0:
                term_indices = image_indices
            else:
                term_indices = spec_indices
            return self._compute_mc_unbiased_gaussian_terms(term_indices)

        parts = []
        use_side_stream = (
            self._fiber_side_stream is not None
            and n_img > 0
            and n_spec > 0
            and self.analytic_images_enabled
            and self.analytic_spectra_enabled
        )
        if use_side_stream:
            main_stream = torch.cuda.current_stream(self.device)
            self._fiber_side_stream.wait_stream(main_stream)
            parts.append(self._compute_analytic_image_gaussian_terms(image_indices))
            with torch.cuda.stream(self._fiber_side_stream):
                parts.append(
                    self._compute_analytic_spectral_gaussian_terms(spec_indices)
                )
            main_stream.wait_stream(self._fiber_side_stream)
            self._fiber_side_stream_used = True
        else:
            if n_img > 0:
                if self.analytic_images_enabled:
                    parts.append(
                        self._compute_analytic_image_gaussian_terms(image_indices)
                    )
                else:
                    parts.append(
                        self._compute_mc_unbiased_gaussian_terms(image_indices)
                    )
            if n_spec > 0:
                if self.analytic_spectra_enabled:
                    parts.append(
                        self._compute_analytic_spectral_gaussian_terms(spec_indices)
                    )
                else:
                    parts.append(
                        self._compute_mc_unbiased_gaussian_terms(spec_indices)
                    )
        return torch.cat(parts, dim=0)

    def _join_fiber_side_stream(self) -> None:
        """Make subsequent training-stream work wait for fiber backward."""
        if self._fiber_side_stream_used:
            torch.cuda.current_stream(self.device).wait_stream(
                self._fiber_side_stream
            )

    def _train_step(
        self, image_indices: torch.Tensor, spec_indices: torch.Tensor, *,
        image_weights: torch.Tensor | None = None, compute_band_losses: bool = True,
        global_step: int | None = None,
    ):
        self._reset_speed_profile()
        self.optimizer.zero_grad()
        if self.splat_optimizer is not None:
            self.splat_optimizer.zero_grad()
        start_gather_log = getattr(
            self._base_model(), "start_sparse_gather_log", None
        )
        if self.sparse_splat_gradients and callable(start_gather_log):
            start_gather_log()
        self._last_packed_image_stats = {}
        image_nll = torch.zeros((), device=self.device)
        spec_nll = torch.zeros((), device=self.device)
        calibration_prior = torch.zeros((), device=self.device)
        band_loss_sums: Dict[str, float] = {}
        band_loss_counts: Dict[str, int] = {}
        image_terms = None
        spec_count = int(spec_indices.numel())
        n_img = int(image_indices.numel())
        total_count = n_img + spec_count
        all_terms = self._compute_step_gaussian_terms(image_indices, spec_indices)
        if image_indices.numel() > 0:
            if all_terms is None:
                raise RuntimeError("MC terms were not computed for a non-empty image batch.")
            image_terms = all_terms[:n_img]
            if image_weights is not None:
                if image_weights.shape != image_terms.shape:
                    raise ValueError("image_weights must match image_terms shape.")
                image_terms = image_terms * copy_tensor_to_device(
                    image_weights, device=self.device, dtype=image_terms.dtype
                )
            pending_birth = getattr(self, "_pending_birth", None)
            if pending_birth is not None:
                self._pending_birth = None
                if int(pending_birth["r"].numel()) != int(n_img):
                    raise RuntimeError(
                        "Birth residual batch does not match the image batch "
                        f"({int(pending_birth['r'].numel())} vs {int(n_img)})."
                    )
                accumulator = self._birth_accumulator()
                band_map = as_tensor_on_device(
                    [
                        accumulator.band_index[str(key)]
                        for key in pending_birth["band_keys"]
                    ],
                    device=self.device,
                )
                sample_weights = (
                    copy_tensor_to_device(
                        image_weights, device=self.device, dtype=torch.float32
                    )
                    if image_weights is not None
                    else torch.ones(int(n_img), device=self.device)
                )
                accumulator.accumulate(
                    band_map[pending_birth["band_local"]],
                    pending_birth["px"],
                    pending_birth["py"],
                    pending_birth["r"],
                    pending_birth["m"],
                    pending_birth["f"],
                    sample_weights,
                )
            image_nll = image_terms.sum()
            if compute_band_losses:
                image_exp = self._gather_named_rows(
                    "exposure_id", image_indices, device=torch.device("cpu"), dtype=torch.long,
                )
                for exp_id_t in torch.unique(image_exp):
                    exp_id = int(exp_id_t.item())
                    band = self.image_band_by_exposure.get(exp_id)
                    if band is None:
                        continue
                    band_mask = image_exp == exp_id_t
                    band_count = int(band_mask.sum().item())
                    if band_count <= 0:
                        continue
                    key = f"e{exp_id}:{band}"
                    mask_device = copy_tensor_to_device(
                        band_mask, device=self.device
                    )
                    band_loss_sums[key] = float(image_terms[mask_device].sum().item())
                    band_loss_counts[key] = band_count
        spec_terms = None
        if spec_indices.numel() > 0:
            if all_terms is None:
                raise RuntimeError("MC terms were not computed for a non-empty spectral batch.")
            spec_terms = all_terms[n_img:n_img + spec_count]
            spec_nll = spec_terms.sum()
        total_nll = torch.zeros((), device=self.device)
        if image_terms is not None:
            total_nll = total_nll + image_nll
        if spec_terms is not None:
            total_nll = total_nll + spec_nll
        distributed_active = is_distributed_active(self.distributed_context)
        global_n_img = float(n_img)
        global_n_spec = float(spec_count)
        if distributed_active:
            count_tensor = tensor_on_device(
                [global_n_img, global_n_spec],
                dtype=torch.float32,
                device=self.device,
            )
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            global_n_img = float(count_tensor[0].item())
            global_n_spec = float(count_tensor[1].item())
        if total_count > 0:
            world_size_scale = float(self.world_size) if distributed_active else 1.0
            total_loss = population_weighted_modality_nll(
                image_nll_sum=image_nll * world_size_scale,
                image_sample_count=global_n_img,
                image_population_count=int(self.image_train_indices.numel()),
                spectrum_nll_sum=spec_nll * world_size_scale,
                spectrum_sample_count=global_n_spec,
                spectrum_population_count=int(self.spec_train_indices.numel()),
                spectrum_weight=self.spectral_bin_weight,
            )
        else:
            total_loss = torch.zeros((), device=self.device)
        if self.splat_flux_l1 > 0.0 and total_count > 0:
            base = self._base_model()
            if callable(getattr(base, "fluxes", None)):
                # Exponential(rate) prior on each splat flux, so the objective
                # is the log posterior (sum NLL + prior) / N. The rate is fixed
                # in config: a penalty whose strength is rescaled by the current
                # loss value is not a prior, it is a moving target that changes
                # the posterior every time the fit improves.
                population_count = (
                    int(self.image_train_indices.numel())
                    + int(self.spec_train_indices.numel())
                )
                total_loss = total_loss + self.splat_flux_l1 * base.fluxes().sum() / max(
                    population_count, 1
                )
        if self.spectral_calibrator is not None:
            calibration_prior = self.spectral_calibrator.prior_penalty()
            if self.spectral_calibrator.cfg.prior_normalization == "objective":
                calibration_prior = calibration_prior / self._objective_denominator()
            total_loss = total_loss + calibration_prior
        if self._speed_profile_enabled and self.device.type == "cuda":
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_start.record()
            total_loss.backward()
            backward_end.record()
            self._record_speed_cuda_events("backward_sec", backward_start, backward_end)
        else:
            backward_wall_start = time.perf_counter()
            total_loss.backward()
            self._record_speed_wall("backward_sec", time.perf_counter() - backward_wall_start)
        self._join_fiber_side_stream()
        self._sync_model_gradients()
        self._sync_spectral_calibrator_grads()
        # Before refine accumulation, which reads the positional gradient and
        # would otherwise coalesce it a second time.
        self._coalesce_sparse_gradients()
        if self.refine_strategy is not None:
            # Growth reads the accumulator only for plan steps below
            # refine_stop, and the plan following this step runs at
            # global_step + 1; past that every refine event is prune-only,
            # so the per-step norm/index_add bookkeeping is dead work.
            growth_possible = (
                global_step is None
                or global_step + 1 < int(self.refine_strategy.config.refine_stop)
            )
            if growth_possible:
                self.refine_strategy.accumulate(self._base_model())
        grad_norm_preclip, grad_norm_postclip = self._clip_gradient_norm()
        if self._speed_profile_enabled and self.device.type == "cuda":
            optimizer_start = torch.cuda.Event(enable_timing=True)
            optimizer_end = torch.cuda.Event(enable_timing=True)
            optimizer_start.record()
            self.optimizer.step()
            if self.splat_optimizer is not None:
                self.splat_optimizer.step()
            optimizer_end.record()
            self._record_speed_cuda_events("optimizer_sec", optimizer_start, optimizer_end)
        else:
            optimizer_wall_start = time.perf_counter()
            self.optimizer.step()
            if self.splat_optimizer is not None:
                self.splat_optimizer.step()
            self._record_speed_wall("optimizer_sec", time.perf_counter() - optimizer_wall_start)
        # One host read for every per-step scalar. Each was its own blocking
        # copy: four .item() here, both grad norms, and two .cpu() inside the
        # calibrator's report. float32 widens to float64 exactly and a Python
        # float is a float64, so every number below is bitwise what the
        # separate reads produced.
        scalars = [total_nll, image_nll, spec_nll, calibration_prior,
                   grad_norm_preclip, grad_norm_postclip]
        if self.spectral_calibrator is not None:
            scalars.append(self.spectral_calibrator.scales_by_band())
        read = torch.cat(
            [
                value.detach().reshape(-1).to(device=self.device, dtype=torch.float64)
                for value in scalars
            ]
        ).tolist()
        calibration_scales: Dict[str, float] = {}
        if self.spectral_calibrator is not None:
            calibration_scales = dict(
                zip(self.spectral_calibrator.band_names, read[6:])
            )
        return {
            "local_total_nll": read[0],
            "local_image_nll": read[1],
            "local_spec_nll": read[2], "calibration_prior": read[3],
            "calibration_scales": calibration_scales, "band_loss_sums": band_loss_sums,
            "band_loss_counts": band_loss_counts, "amplitude": 1.0, "n_spec": spec_count, "n_img": n_img,
            "total_datums": total_count, "mc_samples": 2 * total_count * int(self.replica_samples_k),
            # Read here rather than at the clip, so the norms ride the
            # synchronise the loss scalars above already pay for.
            "grad_norm_preclip": read[4],
            "grad_norm_postclip": read[5],
            "model_splats": int(self._last_packed_image_stats.get("model_splats", 0)),
            "active_splats": int(self._last_packed_image_stats.get("active_splats", 0)),
            "candidate_pairs": int(self._last_packed_image_stats.get("candidate_pairs", 0)),
            "retained_pairs": int(self._last_packed_image_stats.get("retained_pairs", 0)),
            "image_points": int(self._last_packed_image_stats.get("points", 0)),
            **(
                {"band_rows_total": int(self._last_packed_image_stats["band_rows_total"])}
                if "band_rows_total" in self._last_packed_image_stats
                else {}
            ),
            "timings": self._finalize_speed_profile(),
        }

    def _underlying_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _render_analytic_splat_image_window(
        self,
        exposure_id: int,
        *,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        chunk_size: int,
        include_additive: bool,
        profile: Dict[str, Any] | None,
    ) -> np.ndarray:
        model = self._base_model()
        runtime = self.runtime_by_exposure[int(exposure_id)]
        exposure = runtime.exposure
        band = self.image_band_by_exposure[int(exposure_id)]
        cache = self._analytic_cache_for_exposure(int(exposure_id))
        offsets_x, offsets_y, quadrature_weights, sigma_key = (
            self._analytic_image_quadrature_rule()
        )
        samples_per_pixel = int(quadrature_weights.numel())
        height = int(y1 - y0)
        width = int(x1 - x0)
        n_pixels = height * width
        analytic_chunk_size = min(
            n_pixels, max(65_536, int(chunk_size))
        )
        output = np.empty(n_pixels, dtype=np.float32)
        wcs_sec = 0.0
        render_sec = 0.0
        transfer_sec = 0.0
        started = time.perf_counter()

        with torch.no_grad():
            amps = splat_band_amplitudes(
                model,
                {band: (cache["band_lam"], cache["band_throughput"])},
                loglam_min=self.loglam_min,
                loglam_max=self.loglam_max,
            )[band]
            for start in range(0, n_pixels, analytic_chunk_size):
                stop = min(start + analytic_chunk_size, n_pixels)
                flat = torch.arange(start, stop, dtype=torch.long)
                rows = torch.div(flat, width, rounding_mode="floor") + int(y0)
                cols = torch.remainder(flat, width) + int(x0)

                wcs_started = time.perf_counter()
                omega = np.asarray(
                    exposure.obs_wcs.pixel_to_omega_xyz(
                        cols.to(torch.float64).unsqueeze(1).numpy() + offsets_x,
                        rows.to(torch.float64).unsqueeze(1).numpy() + offsets_y,
                    ),
                    dtype=np.float64,
                ).reshape(-1, 3)
                xn, yn = exposure.wcs.omega_xyz_to_norm_torch(
                    torch.as_tensor(
                        omega, dtype=torch.float32, device=self.device
                    )
                )
                wcs_sec += time.perf_counter() - wcs_started

                render_started = time.perf_counter()
                use_psf_mixture = (
                    self.apply_spatial_psf and self.psf_model in {"dgpsf", "mog"}
                )
                if self.psf_model == "eigen_mog":
                    operator = self._eigen_mog_operator()
                    local = (rows * int(exposure.image_width) + cols).numpy()
                    jacobians = self._point_jacobians(
                        cache,
                        cols.to(torch.float64).numpy(),
                        rows.to(torch.float64).numpy(),
                    )
                    context = operator.prepare(
                        np.full(len(local), exposure_id), local, jacobians
                    )
                    index = operator.index(model, xn, yn, context, self.pair_n_sigma)
                    values = operator.predict(
                        model, xn, yn, torch.zeros_like(xn, dtype=torch.long),
                        amps[index.active_splat, None], index, context,
                    )
                elif not use_psf_mixture:
                    sigma = float(
                        cache[sigma_key]
                        if self.apply_spatial_psf
                        else cache["pixel_sigma"] if sigma_key == "sigma" else 0.0
                    )
                    values = tiled_point_prediction(
                        model, xn, yn, sigma, amps,
                        n_sigma=self.pair_n_sigma,
                    )
                else:
                    mixture_key = "psf_mixture" if sigma_key == "sigma" else "psf_mixture_no_pixel"
                    pixel_x = (
                        cols.to(torch.float64).unsqueeze(1).numpy() + offsets_x
                    ).reshape(-1)
                    pixel_y = (
                        rows.to(torch.float64).unsqueeze(1).numpy() + offsets_y
                    ).reshape(-1)
                    mixture = self._point_psf_mixture(
                        cache, mixture_key, pixel_x, pixel_y
                    )
                    values = tiled_mixture_point_prediction(
                        model, xn, yn, mixture, amps, n_sigma=self.pair_n_sigma,
                    )
                values = values.reshape(-1, samples_per_pixel)
                values = (
                    values * quadrature_weights.unsqueeze(0)
                ).sum(dim=1)
                render_sec += time.perf_counter() - render_started

                transfer_started = time.perf_counter()
                output[start:stop] = values.to(torch.float32).cpu().numpy()
                transfer_sec += time.perf_counter() - transfer_started

        image = output.reshape(height, width)
        if include_additive:
            additive = self.image_additive_by_exposure.get(int(exposure_id))
            if additive is None:
                materialize_additive = getattr(
                    self.data, "materialize_image_additive", None
                )
                if callable(materialize_additive):
                    additive = np.asarray(
                        materialize_additive(int(exposure_id)),
                        dtype=np.float32,
                    )
                    self.image_additive_by_exposure[int(exposure_id)] = additive
            if additive is not None:
                additive_window = np.asarray(
                    additive, dtype=np.float32
                )[y0:y1, x0:x1]
                if additive_window.shape != (height, width):
                    raise ValueError(
                        "Image additive window shape does not match deterministic "
                        "render window."
                    )
                image = image + additive_window

        if profile is not None:
            profile["actual_mode"] = "analytic_splat"
            profile["exposure_id"] = int(exposure_id)
            profile["band"] = str(band)
            profile["window"] = {
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1),
            }
            profile["native_shape"] = [height, width]
            profile["quadrature_order"] = int(
                self.analytic_image_quadrature_order
            )
            profile["analytic_chunk_size"] = int(analytic_chunk_size)
            profile["n_chunks"] = int(
                (n_pixels + analytic_chunk_size - 1) // analytic_chunk_size
            )
            profile["timings_sec"] = {
                "wcs_and_grid": wcs_sec,
                "band_render": render_sec,
                "device_transfer": transfer_sec,
                "total": time.perf_counter() - started,
            }
            if self.device.type == "cuda":
                profile["gpu_peak_mem_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                )
        return np.asarray(image, dtype=np.float32).copy()

    def render_datum_image_window_deterministic(
        self, exposure_id: int, *, y0: int, y1: int, x0: int, x1: int, supersample: int = 2,
        chunk_size: int = 2048, include_additive: bool = True, profile: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Render a deterministic image patch using the datum-space image operator."""
        if supersample <= 0:
            raise ValueError("supersample must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        runtime = self.runtime_by_exposure[int(exposure_id)]
        exposure = runtime.exposure
        height = int(runtime.height)
        width = int(runtime.width)
        y0_i = int(y0)
        y1_i = int(y1)
        x0_i = int(x0)
        x1_i = int(x1)
        if y0_i < 0 or x0_i < 0 or y1_i > height or x1_i > width:
            raise ValueError("Requested deterministic image window is out of bounds.")
        if y1_i <= y0_i or x1_i <= x0_i:
            raise ValueError("Requested deterministic image window must have positive area.")

        band = self.image_band_by_exposure[int(exposure_id)]
        pixel_wcs = exposure.obs_wcs or exposure.wcs
        overall_start = time.perf_counter()
        if profile is not None and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        model = self._base_model()
        if (
            self.analytic_images_enabled
            and callable(getattr(model, "covariances", None))
        ):
            if profile is not None:
                profile["requested_supersample"] = int(supersample)
            return self._render_analytic_splat_image_window(
                int(exposure_id),
                y0=y0_i,
                y1=y1_i,
                x0=x0_i,
                x1=x1_i,
                chunk_size=int(chunk_size),
                include_additive=bool(include_additive),
                profile=profile,
            )

        if self.psf_model == "eigen_mog":
            raise NotImplementedError(
                "psf_model='eigen_mog' image windows require the analytic SplatField renderer"
            )

        with torch.no_grad():
            kernel_hr = None
            pad = 0
            kernel_build_start = time.perf_counter()
            if self.apply_spatial_psf and exposure.psf_fwhm_arcsec is not None:
                if self.psf_model == "gaussian":
                    kernel_hr = gaussian_psf_kernel(
                        float(exposure.psf_fwhm_arcsec),
                        float(exposure.pixel_scale_arcsec) / float(supersample), self.device,
                    )
                else:
                    components = (
                        exposure.psf_mixture_dgpsf if self.psf_model == "dgpsf"
                        else exposure.psf_mixture_kl
                    )
                    if components is None:
                        label = exposure.label or f"exposure_id={int(exposure_id)}"
                        raise ValueError(
                            f"Image exposure {label!r} lacks the requested PSF mixture; "
                            "run scripts/backfill_psf_mixture.py on the manifest."
                        )
                    kernel_hr = gaussian_mixture_psf_kernel(
                        components, supersample=int(supersample), device=self.device,
                    )
                pad = int(kernel_hr.shape[-1] // 2)
            kernel_build_sec = time.perf_counter() - kernel_build_start

            height_native = y1_i - y0_i
            width_native = x1_i - x0_i
            height_hr = height_native * supersample + 2 * pad
            width_hr = width_native * supersample + 2 * pad
            if profile is not None:
                profile["exposure_id"] = int(exposure_id)
                profile["band"] = str(band)
                profile["window"] = {"y0": int(y0_i), "y1": int(y1_i), "x0": int(x0_i), "x1": int(x1_i)}
                profile["native_shape"] = [int(height_native), int(width_native)]
                profile["supersample"] = int(supersample)
                profile["hr_shape"] = [int(height_hr), int(width_hr)]
                profile["psf_pad"] = int(pad)
                profile["timings_sec"] = {
                    "kernel_build": kernel_build_sec, "wcs_and_grid": 0.0, "band_render": 0.0,
                    "psf_conv": 0.0, "downsample": 0.0, "total": 0.0,
                }
            x_hr = (
                (np.arange(width_hr, dtype=np.float64) - pad + 0.5) / float(supersample)
                + float(x0_i)
                - 0.5
            )
            # Stream supersampled WCS evaluation in row tiles so diagnostics do
            # not materialize the full xx/yy/omega grids.
            target_points_per_tile = 1_048_576
            rows_per_tile = max(1, min(height_hr, target_points_per_tile // max(1, width_hr)))
            if profile is not None:
                profile["rows_per_tile"] = int(rows_per_tile)
                profile["n_tiles"] = int((height_hr + rows_per_tile - 1) // rows_per_tile)
                profile["tiles"] = []
            band_hr = torch.empty((height_hr, width_hr), dtype=torch.float32, device=self.device)
            for row_start in range(0, height_hr, rows_per_tile):
                row_stop = min(row_start + rows_per_tile, height_hr)
                tile_profile: Dict[str, Any] | None = None
                tile_start = time.perf_counter()
                if profile is not None:
                    tile_profile = {
                        "row_start": int(row_start), "row_stop": int(row_stop),
                        "rows": int(row_stop - row_start),
                    }
                wcs_start = time.perf_counter()
                y_hr = (
                    (np.arange(row_start, row_stop, dtype=np.float64) - pad + 0.5)
                    / float(supersample)
                    + float(y0_i)
                    - 0.5
                )
                yy_tile, xx_tile = np.meshgrid(y_hr, x_hr, indexing="ij")
                omega_tile = np.asarray(pixel_wcs.pixel_to_omega_xyz(xx_tile, yy_tile), dtype=np.float64)
                wcs_sec = time.perf_counter() - wcs_start
                expected_shape = (row_stop - row_start, width_hr, 3)
                if omega_tile.shape != expected_shape:
                    raise ValueError(
                        "Image WCS must return shape [H_tile, W_sup, 3] for supersampled inputs."
                    )
                render_profile: Dict[str, Any] | None = {} if profile is not None else None
                band_tile = self._render_band_for_omega(
                    int(exposure_id), band, omega_tile.reshape(-1, 3), chunk_size=int(chunk_size),
                    profile=render_profile,
                ).view(row_stop - row_start, width_hr)
                band_hr[row_start:row_stop] = band_tile
                if profile is not None and tile_profile is not None:
                    tile_profile["points"] = int((row_stop - row_start) * width_hr)
                    tile_profile["timings_sec"] = {
                        "wcs_and_grid": wcs_sec, "band_render": float(render_profile["timings_sec"]["total"]),
                        "total": time.perf_counter() - tile_start,
                    }
                    tile_profile["render"] = render_profile
                    profile["tiles"].append(tile_profile)
                    profile["timings_sec"]["wcs_and_grid"] += wcs_sec
                    profile["timings_sec"]["band_render"] += float(render_profile["timings_sec"]["total"])
            psf_conv_start = time.perf_counter()
            if kernel_hr is not None:
                band_hr = convolve_flat_image(
                    band_hr.reshape(-1), height_hr, width_hr, kernel_hr,
                ).view(height_hr, width_hr)
            psf_conv_sec = time.perf_counter() - psf_conv_start
            image_hr = band_hr
            if pad > 0:
                image_hr = image_hr[pad:-pad, pad:-pad]
            downsample_start = time.perf_counter()
            image_native = image_hr.view(
                height_native, supersample, width_native, supersample,
            ).mean(dim=(1, 3))
            if include_additive:
                additive = self.image_additive_by_exposure.get(int(exposure_id))
                if additive is None:
                    materialize_additive = getattr(self.data, "materialize_image_additive", None)
                    if callable(materialize_additive):
                        additive = np.asarray(materialize_additive(int(exposure_id)), dtype=np.float32)
                        self.image_additive_by_exposure[int(exposure_id)] = additive
                if additive is not None:
                    additive_window = np.asarray(additive, dtype=np.float32)[y0_i:y1_i, x0_i:x1_i]
                    if additive_window.shape != (height_native, width_native):
                        raise ValueError(
                            "Image additive window shape does not match deterministic render window."
                        )
                    image_native = image_native + torch.as_tensor(
                        additive_window, dtype=image_native.dtype, device=self.device,
                    )
            downsample_sec = time.perf_counter() - downsample_start
            if profile is not None:
                profile["timings_sec"]["psf_conv"] = psf_conv_sec
                profile["timings_sec"]["downsample"] = downsample_sec
                profile["timings_sec"]["total"] = time.perf_counter() - overall_start
                if self.device.type == "cuda":
                    profile["gpu_peak_mem_allocated_mb"] = float(
                        torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                    )
            return image_native.detach().cpu().numpy().copy()

    def compute_datum_image_predictions_deterministic(
        self, *, supersample: int = 2, chunk_size: int = 2048, include_additive: bool = True,
        exposure_ids: list[int] | tuple[int, ...] | None = None, profile: Dict[str, Any] | None = None,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Render deterministic image diagnostics using the image datum operator."""
        if supersample <= 0:
            raise ValueError("supersample must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        predictions: Dict[int, Dict[str, np.ndarray]] = {}
        overall_start = time.perf_counter()
        if profile is not None:
            profile["supersample"] = int(supersample)
            profile["chunk_size"] = int(chunk_size)
            profile["exposures"] = {}
        selected_exposure_ids = (
            sorted(int(exp_id) for exp_id in self.image_exposure_ids)
            if exposure_ids is None
            else [int(exp_id) for exp_id in exposure_ids]
        )
        unknown_exposure_ids = [
            int(exp_id)
            for exp_id in selected_exposure_ids
            if int(exp_id) not in self.image_exposure_ids
        ]
        if unknown_exposure_ids:
            raise ValueError(
                "Requested deterministic image diagnostics for non-image exposure ids: "
                f"{unknown_exposure_ids}."
            )
        if profile is not None:
            profile["selected_exposure_ids"] = selected_exposure_ids
        for exp_id in selected_exposure_ids:
            runtime = self.runtime_by_exposure[int(exp_id)]
            band = self.image_band_by_exposure[int(exp_id)]
            exposure_profile: Dict[str, Any] | None = {} if profile is not None else None
            exposure_start = time.perf_counter()
            predictions[int(exp_id)] = {
                band: self.render_datum_image_window_deterministic(
                    int(exp_id), y0=0, y1=int(runtime.height), x0=0, x1=int(runtime.width),
                    supersample=int(supersample), chunk_size=int(chunk_size),
                    include_additive=bool(include_additive), profile=exposure_profile,
                )
            }
            if profile is not None and exposure_profile is not None:
                exposure_profile["timings_sec"]["exposure_total"] = time.perf_counter() - exposure_start
                profile["exposures"][f"e{int(exp_id)}:{band}"] = exposure_profile
        if profile is not None:
            profile["timings_sec"] = {
                "compute_datum_image_predictions_deterministic": time.perf_counter() - overall_start
            }
        return predictions

    def _predict_fiber_spectrum_from_field_sum(
        self, exp_id: int, *, include_additive: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        runtime = self.runtime_by_exposure[int(exp_id)]
        if runtime.spec_projector is None:
            return None
        field_sum = self._compute_field_sum_for_fiber(exp_id)
        if field_sum is None:
            return None
        spec = runtime.renderer.project_field_to_spectrum(
            field_sum, runtime.spec_projector, apply_photon_weight=False, apply_throughput=False,
            apply_lsf=False,
        )
        spec = self._apply_diagnostic_lsf(
            spec, runtime.exposure.lam_spec, runtime.exposure.lsf_fwhm_angstrom,
            lam_edges=runtime.exposure.lam_edges,
        )
        validate_spectral_source_psf(runtime.exposure, apply_spatial_psf=self.apply_spatial_psf)
        source_scale = validated_spectral_source_scale(runtime.exposure)
        if source_scale is not None:
            spec = spec * torch.tensor(source_scale, dtype=spec.dtype, device=spec.device)
        spec = self._finalize_fiber_spectrum(exp_id, spec, include_additive=include_additive)
        lam = runtime.exposure.lam_spec
        return lam, spec.detach().cpu().numpy()

    def _predict_fiber_spectrum_from_aperture_quadrature(
        self, exp_id: int, *, include_additive: bool = True, radial_nodes: int = 4, angular_nodes: int = 8,
        psf_nodes: int = 3, chunk_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        runtime = self.runtime_by_exposure[int(exp_id)]
        if runtime.spec_projector is None:
            return None
        aperture = runtime.exposure.aperture
        if not isinstance(aperture, SkyDiskApertureSpec):
            raise ValueError("Deterministic aperture quadrature currently supports SkyDiskApertureSpec only.")
        omega_xyz, weights = self._sky_disk_quadrature(
            aperture, psf_fwhm_arcsec=runtime.exposure.psf_fwhm_arcsec, radial_nodes=int(radial_nodes),
            angular_nodes=int(angular_nodes), psf_nodes=int(psf_nodes),
        )
        field_mean = self._evaluate_weighted_field_for_omega(
            exp_id, omega_xyz, weights, chunk_size=int(chunk_size),
        )
        field_mean = field_mean * aperture_area_arcsec2(
            aperture, pixel_scale_arcsec=runtime.exposure.pixel_scale_arcsec
        )
        spec = runtime.renderer.project_field_to_spectrum(
            field_mean, runtime.spec_projector, apply_photon_weight=False, apply_throughput=False,
            apply_lsf=False,
        )
        spec = self._apply_diagnostic_lsf(
            spec, runtime.exposure.lam_spec, runtime.exposure.lsf_fwhm_angstrom,
            lam_edges=runtime.exposure.lam_edges,
        )
        validate_spectral_source_psf(runtime.exposure, apply_spatial_psf=self.apply_spatial_psf)
        source_scale = validated_spectral_source_scale(runtime.exposure)
        if source_scale is not None:
            spec = spec * torch.tensor(source_scale, dtype=spec.dtype, device=spec.device)
        spec = self._finalize_fiber_spectrum(exp_id, spec, include_additive=include_additive)
        lam = runtime.exposure.lam_spec
        return lam, spec.detach().cpu().numpy()

    def _predict_fiber_spectrum_from_splats(
        self, exp_id: int, *, include_additive: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        runtime = self.runtime_by_exposure[int(exp_id)]
        model = self._base_model()
        local = torch.arange(
            int(runtime.exposure.lam_spec.size), dtype=torch.long, device=self.device
        )
        spec = splat_fiber_spectrum(
            model,
            runtime.exposure,
            local,
            cache=self._spectrum_cache_for_exposure(int(exp_id)),
            loglam_min=self.loglam_min,
            loglam_max=self.loglam_max,
            apply_spatial_psf=self.apply_spatial_psf,
        )
        spec = self._finalize_fiber_spectrum(exp_id, spec, include_additive=include_additive)
        return runtime.exposure.lam_spec, spec.detach().cpu().numpy()

    def _spectral_source_gate(self, exp_id: int, n_bins: int) -> torch.Tensor | None:
        """Per-bin source gate for a spectral exposure, ordered like lam_spec.

        Training multiplies the rendered source by this gate before adding the
        additive (sky fibers carry gate 0); fiber predictions must match or a
        sky fiber's source-only prediction reports phantom source flux."""
        if getattr(self.data.batch, "source_gate", None) is None:
            return None
        idx = self._global_indices_for(int(exp_id), DatumKind.SPECTRAL_BIN)
        if int(idx.numel()) != int(n_bins):
            raise ValueError(
                f"Spectral exposure {int(exp_id)} has {int(idx.numel())} datum rows "
                f"but {int(n_bins)} wavelength bins; cannot align the source gate."
            )
        return self._gather_named_rows(
            "source_gate", idx, device=self.device, dtype=torch.float32,
        )

    def _finalize_fiber_spectrum(
        self, exp_id: int, spec: torch.Tensor, *, include_additive: bool,
    ) -> torch.Tensor:
        """Calibrate, source-gate, and optionally add sky, in the training
        observation-contract order (calibrated source * gate + additive)."""
        if (
            self.spectral_calibrator is not None
            and bool(self.spectral_calibration_cfg.apply_to_predictions)
        ):
            scales = self.spectral_calibrator.scale_for_full_exposure(exp_id).to(dtype=spec.dtype)
            spec = spec * scales
        gate = self._spectral_source_gate(int(exp_id), int(spec.shape[-1]))
        if gate is not None:
            spec = spec * gate.to(dtype=spec.dtype)
        if include_additive:
            spec = self._add_spectral_additive(exp_id, spec)
        return spec

    def _add_spectral_additive(self, exp_id: int, spec: torch.Tensor) -> torch.Tensor:
        additive = self.spec_additive_by_exposure.get(int(exp_id))
        if additive is None:
            return spec
        additive_arr = np.asarray(additive, dtype=np.float32)
        if tuple(additive_arr.shape) != tuple(spec.shape):
            raise ValueError(
                "Spectrum additive shape "
                f"{additive_arr.shape} does not match prediction shape {tuple(spec.shape)}."
            )
        return spec + torch.as_tensor(additive_arr, dtype=spec.dtype, device=self.device)

    def predict_fiber_spectra(
        self, *, include_additive: bool = True, max_spectra: int | None = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        outputs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        limit = None if max_spectra is None else max(0, int(max_spectra))
        if limit == 0:
            return outputs
        with torch.no_grad():
            for exp_id in sorted(self.spec_exposure_ids):
                if limit is not None and len(outputs) >= limit:
                    break
                runtime = self.runtime_by_exposure[int(exp_id)]
                aperture = runtime.exposure.aperture
                model = self._base_model()
                # Non-mask apertures (for example SDSS SkyDiskApertureSpec) use
                # a deterministic quadrature over the declared aperture so
                # diagnostics stay datum-aligned without fresh MC noise.
                if aperture is None or isinstance(aperture, MaskApertureSpec):
                    predictor = self._predict_fiber_spectrum_from_field_sum
                elif callable(getattr(getattr(model, "sed", None), "templates", None)):
                    predictor = self._predict_fiber_spectrum_from_splats
                else:
                    predictor = self._predict_fiber_spectrum_from_aperture_quadrature
                prediction = predictor(exp_id, include_additive=bool(include_additive))
                if prediction is not None:
                    outputs[int(exp_id)] = prediction
        return outputs

    def predict_fiber_spectra_variants(
        self, *, max_spectra: int | None = None,
    ) -> Tuple[
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        Dict[int, Tuple[np.ndarray, np.ndarray]],
    ]:
        """(source_only, with_additive) fiber spectra from ONE prediction pass.

        _finalize_fiber_spectrum adds the stored additive as its final step,
        so the with-additive variant is exactly the source-only prediction
        plus the stored additive; predicting twice would recompute identical
        source spectra.
        """
        source_only = self.predict_fiber_spectra(
            include_additive=False, max_spectra=max_spectra
        )
        with_additive: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        with torch.no_grad():
            for exp_id, (lam, values) in source_only.items():
                spec = torch.as_tensor(np.asarray(values), device=self.device)
                spec = self._add_spectral_additive(int(exp_id), spec)
                with_additive[int(exp_id)] = (lam, spec.cpu().numpy())
        return source_only, with_additive

    def get_calibration_state_dict(self) -> Dict[str, object] | None:
        if self.spectral_calibrator is None:
            return None
        return self.spectral_calibrator.get_state()

    def load_calibration_state_dict(self, state: Dict[str, object] | None) -> bool:
        if self.spectral_calibrator is None:
            return False
        if state is None:
            return False
        self.spectral_calibrator.load_state(state)
        return True

    def get_calibration_report(self) -> Dict[str, object]:
        if self.spectral_calibrator is None:
            return {"enabled": False}
        return self.spectral_calibrator.report()

    def sample_point_spectra(
        self, pixel_requests: List[Tuple[str, int, int]], *, exposure_id: int | None = None,
    ) -> List[Dict[str, object]]:
        """Model spectrum and band fluxes at a few labelled pixels.

        Diagnostic only: nothing here feeds training or the reported metrics.
        The spectra come from `_evaluate_spectra_for_omega`, whose docstring
        records why they are not bitwise with a dense whole-census evaluation.
        """
        if not pixel_requests:
            return []
        image_exposure_id = (self.primary_image_exposure_id if exposure_id is None else int(exposure_id))
        if image_exposure_id not in self.image_exposure_ids:
            raise ValueError(f"Exposure {image_exposure_id} is not an image exposure.")

        runtime = self.runtime_by_exposure[image_exposure_id]
        valid = []
        indices = []
        H, W = runtime.height, runtime.width
        for label, y, x in pixel_requests:
            if y < 0 or y >= H or x < 0 or x >= W:
                continue
            valid.append((label, y, x))
            indices.append(y * W + x)
        if not valid:
            return []

        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        runtime = self._ensure_diagnostic_runtime(int(image_exposure_id))
        if runtime.omega_xyz is None:
            raise RuntimeError("Diagnostic runtime omega_xyz was not initialized.")
        omega = torch.index_select(runtime.omega_xyz, 0, idx_tensor)
        spectra = self._evaluate_spectra_for_omega(
            int(image_exposure_id), omega, renderer=self.diagnostic_renderer,
        )
        band_fluxes = self._project_spectra_to_band_fluxes(spectra)
        samples = []
        for i, (label, y, x) in enumerate(valid):
            samples.append(
                {
                    "label": label, "coords": (int(y), int(x)),
                    "spectrum": spectra[i].detach().cpu().numpy().copy(),
                    "band_fluxes": {band: float(values[i]) for band, values in band_fluxes.items()},
                }
            )
        return samples

    def _sample_indices(self, n_total: int, batch_size: int) -> torch.Tensor:
        if batch_size <= 0 or batch_size >= n_total:
            return torch.arange(n_total, device=self.device)
        perm = torch.randperm(n_total, device=self.device)
        return perm[:batch_size]

    def _sample_from_indices(self, indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        if indices.numel() == 0:
            return indices
        if batch_size <= 0 or batch_size >= indices.numel():
            return indices
        perm = torch.randperm(indices.numel(), device=self.device)[:batch_size]
        return indices[perm]

    def _format_stats(self, epoch: int, global_step: int, stats: Dict[str, float]) -> str:
        band_losses = sorted(stats.get("band_losses", {}).items())
        max_band_losses = 12
        band_loss_str = ", ".join(
            f"{b}={v:.3e}" for b, v in band_losses[:max_band_losses]
        )
        if len(band_losses) > max_band_losses:
            if band_loss_str:
                band_loss_str += ", "
            band_loss_str += f"... +{len(band_losses) - max_band_losses} more"
        calibration_prior = float(stats.get("calibration_prior", 0.0))
        support = (
            f"support[model_splats={int(stats.get('model_splats', 0))}, "
            f"active_splats={int(stats.get('active_splats', 0))}, "
            f"candidate_pairs={int(stats.get('candidate_pairs', 0))}, "
            f"retained_pairs={int(stats.get('retained_pairs', 0))}]"
        )
        return (
            f"Step {global_step+1} (epoch {epoch+1}): loss={stats['total_loss']:.3e}, "
            f"image={stats['image_loss']:.3e}, spec={stats['spec_loss']:.3e}, "
            f"cal_prior={calibration_prior:.3e}; {support}; "
            f"band_losses[{band_loss_str}]"
        )

    def _sync_spectral_calibrator_grads(self) -> None:
        if not is_distributed_active(self.distributed_context) or self.spectral_calibrator is None:
            return
        for param in self.spectral_calibrator.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(float(self.world_size))

    def _record_history(self, stats: Dict[str, float]) -> None:
        self.history["loss"].append(float(stats.get("total_loss", float("nan"))))
        self.history["image_loss"].append(float(stats.get("image_loss", float("nan"))))
        self.history["spec_loss"].append(float(stats.get("spec_loss", float("nan"))))
        self.history["n_spec"].append(int(stats.get("n_spec", 0) or 0))
        self.history["calibration_prior"].append(float(stats.get("calibration_prior", 0.0)))

    def _reduce_step_logging(
        self, local_stats: Dict[str, Any], *, data_wait_sec: float, transfer_sec: float,
        train_step_sec: float, total_step_sec: float,
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        image_nll = float(local_stats.get("local_image_nll", 0.0) or 0.0)
        spec_nll = float(local_stats.get("local_spec_nll", 0.0) or 0.0)
        total_nll = float(local_stats.get("local_total_nll", 0.0) or 0.0)
        n_img = int(local_stats.get("n_img", 0) or 0)
        n_spec = int(local_stats.get("n_spec", 0) or 0)
        calibration_prior = float(local_stats.get("calibration_prior", 0.0) or 0.0)

        if is_distributed_active(self.distributed_context):
            sums = torch.tensor(
                [image_nll, spec_nll, total_nll, float(n_img), float(n_spec), calibration_prior],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(sums[:5], op=dist.ReduceOp.SUM)
            dist.all_reduce(sums[5:], op=dist.ReduceOp.SUM)
            sums[5] /= float(self.world_size)
            global_image_nll = float(sums[0].item())
            global_spec_nll = float(sums[1].item())
            global_n_img = int(round(float(sums[3].item())))
            global_n_spec = int(round(float(sums[4].item())))
            calibration_prior = float(sums[5].item())
        else:
            global_image_nll = image_nll
            global_spec_nll = spec_nll
            global_n_img = n_img
            global_n_spec = n_spec
        global_total_count = int(global_n_img + global_n_spec)

        timing_values = {
            "data_wait_sec": float(data_wait_sec), "transfer_sec": float(transfer_sec),
            "train_step_sec": float(train_step_sec), "total_step_sec": float(total_step_sec),
        }
        if is_distributed_active(self.distributed_context):
            timing_tensor = torch.tensor(
                [timing_values[key] for key in timing_values], dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
            timing_values = {
                key: float(value.item())
                for key, value in zip(timing_values.keys(), timing_tensor, strict=False)
            }

        reduced_timings = dict(local_stats.get("timings", {}) or {})
        if reduced_timings or is_distributed_active(self.distributed_context):
            timing_keys = _SPEED_PROFILE_TIMING_KEYS
            if is_distributed_active(self.distributed_context):
                timing_tensor = torch.tensor(
                    [float(reduced_timings.get(key, 0.0) or 0.0) for key in timing_keys],
                    dtype=torch.float64, device=self.device,
                )
                dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
                reduced_timings = {
                    key: float(value.item())
                    for key, value in zip(timing_keys, timing_tensor, strict=False)
                }
            else:
                reduced_timings = {
                    key: float(reduced_timings.get(key, 0.0) or 0.0)
                    for key in timing_keys
                }
            reduced_timings = {key: value for key, value in reduced_timings.items() if value > 0.0}

        grad_norm_preclip = float(local_stats.get("grad_norm_preclip", float("nan")))
        grad_norm_postclip = float(local_stats.get("grad_norm_postclip", float("nan")))
        if is_distributed_active(self.distributed_context):
            grad_tensor = torch.tensor(
                [grad_norm_preclip, grad_norm_postclip],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.MAX)
            grad_norm_preclip = float(grad_tensor[0].item())
            grad_norm_postclip = float(grad_tensor[1].item())

        band_keys = [
            f"e{int(exp_id)}:{band}"
            for exp_id, band in sorted(self.image_band_by_exposure.items())
        ]
        local_band_sums = local_stats.get("band_loss_sums", {}) or {}
        local_band_counts = local_stats.get("band_loss_counts", {}) or {}
        band_losses: Dict[str, float] = {}
        if band_keys and (local_band_sums or local_band_counts):
            if is_distributed_active(self.distributed_context):
                sum_tensor = torch.tensor(
                    [float(local_band_sums.get(key, 0.0) or 0.0) for key in band_keys], dtype=torch.float64,
                    device=self.device,
                )
                count_tensor = torch.tensor(
                    [float(local_band_counts.get(key, 0.0) or 0.0) for key in band_keys], dtype=torch.float64,
                    device=self.device,
                )
                dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                for idx, key in enumerate(band_keys):
                    count = float(count_tensor[idx].item())
                    if count > 0.0:
                        band_losses[key] = float(sum_tensor[idx].item() / count)
            else:
                for key in band_keys:
                    count = float(local_band_counts.get(key, 0.0) or 0.0)
                    if count > 0.0:
                        band_losses[key] = (
                            float(local_band_sums.get(key, 0.0) or 0.0) / count
                        )

        image_loss = (global_image_nll / float(global_n_img) if global_n_img > 0 else 0.0)
        spec_loss = (global_spec_nll / float(global_n_spec) if global_n_spec > 0 else float("nan"))
        image_population_count = int(self.image_train_indices.numel())
        spectrum_population_count = int(self.spec_train_indices.numel())
        population_count = image_population_count + spectrum_population_count
        total_loss = calibration_prior
        if population_count > 0:
            if image_population_count > 0:
                total_loss += image_population_count / population_count * image_loss
            if spectrum_population_count > 0:
                total_loss += spectrum_population_count / population_count * spec_loss

        stats = {
            "total_loss": total_loss, "image_loss": image_loss, "spec_loss": spec_loss,
            "calibration_prior": calibration_prior,
            "calibration_scales": local_stats.get("calibration_scales", {}), "band_losses": band_losses,
            "amplitude": 1.0, "n_spec": global_n_spec, "n_img": global_n_img,
            "total_datums": global_total_count,
            "mc_samples": 2 * global_total_count * int(self.replica_samples_k),
            "grad_norm_preclip": grad_norm_preclip, "grad_norm_postclip": grad_norm_postclip,
            "model_splats": int(local_stats.get("model_splats", 0) or 0),
            "active_splats": int(local_stats.get("active_splats", 0) or 0),
            "candidate_pairs": int(local_stats.get("candidate_pairs", 0) or 0),
            "retained_pairs": int(local_stats.get("retained_pairs", 0) or 0),
            "image_points": int(local_stats.get("image_points", 0) or 0),
            "timings": reduced_timings,
        }
        if "band_rows_total" in local_stats:
            stats["band_rows_total"] = int(local_stats["band_rows_total"])
        return stats, timing_values

    def _log_speed(
        self, *, epoch: int, global_step: int, stats: Dict[str, float], data_wait_sec: float,
        transfer_sec: float, train_step_sec: float, total_step_sec: float,
    ) -> None:
        if not self._speed_profile_enabled:
            return
        payload: Dict[str, Any] = {
            "step": int(global_step),
            "epoch": int(epoch + 1),
            "total_loss": float(stats.get("total_loss", float("nan"))),
            "image_loss": float(stats.get("image_loss", float("nan"))),
            "spec_loss": float(stats.get("spec_loss", float("nan"))),
            "calibration_prior": float(stats.get("calibration_prior", 0.0)),
            "n_img": int(stats.get("n_img", 0) or 0),
            "n_spec": int(stats.get("n_spec", 0) or 0),
            "total_datums": int(stats.get("total_datums", 0) or 0),
            "mc_samples": int(stats.get("mc_samples", 0) or 0),
            "data_wait_sec": float(data_wait_sec),
            "transfer_sec": float(transfer_sec),
            "train_step_sec": float(train_step_sec),
            "total_step_sec": float(total_step_sec),
            "steps_per_sec": (1.0 / total_step_sec) if total_step_sec > 0.0 else float("nan"),
            "datums_per_sec": (
                float(stats.get("total_datums", 0) or 0) / total_step_sec
                if total_step_sec > 0.0
                else float("nan")
            ),
            "mc_samples_per_sec": (
                float(stats.get("mc_samples", 0) or 0) / total_step_sec
                if total_step_sec > 0.0
                else float("nan")
            ),
            "grad_norm_preclip": float(stats.get("grad_norm_preclip", float("nan"))),
            "grad_norm_postclip": float(stats.get("grad_norm_postclip", float("nan"))),
            "model_splats": int(stats.get("model_splats", 0) or 0),
            "active_splats": int(stats.get("active_splats", 0) or 0),
            "candidate_pairs": int(stats.get("candidate_pairs", 0) or 0),
            "retained_pairs": int(stats.get("retained_pairs", 0) or 0),
            "image_points": int(stats.get("image_points", 0) or 0),
        }
        if "band_rows_total" in stats:
            payload["band_rows_total"] = int(stats["band_rows_total"])
        timings = stats.get("timings", {})
        if isinstance(timings, dict):
            for key, value in timings.items():
                payload[key] = float(value)
        optimizer_groups = list(self.optimizer.param_groups)
        if self.splat_optimizer is not None:
            optimizer_groups.extend(self.splat_optimizer.param_groups)
        lrs = [float(group.get("lr", float("nan"))) for group in optimizer_groups]
        if lrs:
            payload["lr"] = lrs[0]
            for idx, lr in enumerate(lrs):
                payload[f"lr_group_{idx}"] = lr
        if self.device.type == "cuda":
            # These four are already host-side Python floats. Only the
            # multi-rank maximum needs the device, so build the tensor only
            # when there is something to all-reduce: on one rank the round
            # trip is an H2D copy plus four device synchronisations per step
            # for numbers that never left the host.
            mem_mb = [
                torch.cuda.memory_allocated(self.device) / (1024.0 ** 2),
                torch.cuda.memory_reserved(self.device) / (1024.0 ** 2),
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2),
                torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2),
            ]
            if is_distributed_active(self.distributed_context):
                mem_tensor = torch.tensor(
                    mem_mb, dtype=torch.float64, device=self.device,
                )
                dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
                mem_mb = [float(mem_tensor[i].item()) for i in range(4)]
            payload["gpu_mem_allocated_mb"] = float(mem_mb[0])
            payload["gpu_mem_reserved_mb"] = float(mem_mb[1])
            payload["gpu_peak_mem_allocated_mb"] = float(mem_mb[2])
            payload["gpu_peak_mem_reserved_mb"] = float(mem_mb[3])
        if not self.speed_logger:
            return
        self.speed_logger.record_step(payload)

    def _log_wandb(
        self, global_step: int, stats: Dict[str, float], *, force: bool = False,
        extra_payload: Dict[str, Any] | None = None,
    ) -> None:
        if not self.wandb_logger:
            return
        step = global_step + 1
        if not force and step % self.wandb_log_every_n_steps != 0:
            return
        if self._last_wandb_step_logged == step and not extra_payload:
            return
        payload = {
            "train/total_loss": stats.get("total_loss"), "train/image_loss": stats.get("image_loss"),
            "train/spec_loss": stats.get("spec_loss"),
            "train/calibration_prior": stats.get("calibration_prior", 0.0),
            "train/n_spec": stats.get("n_spec"), "train/n_img": stats.get("n_img"),
            "train/model_splats": stats.get("model_splats", 0),
            "train/active_splats": stats.get("active_splats", 0),
            "train/candidate_pairs": stats.get("candidate_pairs", 0),
            "train/retained_pairs": stats.get("retained_pairs", 0),
            "train/image_points": stats.get("image_points", 0),
        }
        optimizer_groups = list(self.optimizer.param_groups)
        if self.splat_optimizer is not None:
            optimizer_groups.extend(self.splat_optimizer.param_groups)
        lrs = [float(group.get("lr", float("nan"))) for group in optimizer_groups]
        if lrs:
            payload["train/lr"] = lrs[0]
            for idx, lr in enumerate(lrs):
                payload[f"train/lr_group_{idx}"] = lr
        for band, loss in stats.get("band_losses", {}).items():
            payload[f"train/band_loss/{band}"] = loss
        for band, value in stats.get("calibration_scales", {}).items():
            payload[f"train/calibration_scale/{band}"] = value
        if extra_payload:
            payload.update(extra_payload)
        self.wandb_logger.log(payload, step=step)
        self._last_wandb_step_logged = step
