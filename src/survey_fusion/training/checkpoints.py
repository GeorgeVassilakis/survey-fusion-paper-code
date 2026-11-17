from __future__ import annotations

import math
import hashlib
import json
import random
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..models.splat_field import SplatField
from ..models.splat_strategy import canonicalize_gauge_rows
from ..utils.perf_env import exact_pair_pruning_enabled

# splat settings that must match between a checkpoint and the trainer resuming
# it; n_splats is validated too, but resume adapts the model to the checkpoint's
# population first
_SPLAT_CONFIG_KEYS = (
    "name", "n_splats", "n_templates", "n_loglam_grid",
    "sed_off_grid", "sed_pin_gauge_knots",
)
_SPLAT_SED_CONFIG_KEYS = ("sed_off_grid", "sed_pin_gauge_knots")
# Grid-padding keys are structural like n_loglam_grid, but checkpoints written
# before padding existed lack them; absence means an unpadded grid.
_SPLAT_PAD_CONFIG_KEYS = ("sed_pad_knots_blue", "sed_pad_knots_red")

OPERATOR_SCHEMA_VERSION = 3
PIXEL_RESPONSE_CONVENTION = "square_top_hat"
PSF_MIXTURE_PROVENANCE_CONVENTION = "digest_v2_operator_qa_pixel_response"
PSF_JACOBIAN_ALGORITHM = "float64_one_pixel_tiled_v2"
PSF_JACOBIAN_MAX_TILES_PER_AXIS = 16
PSF_JACOBIAN_CAP_POLICY = "error"
OPERATOR_IDENTITY_KEYS = (
    "operator_schema_version",
    "psf_model",
    "psf_mixture_digest",
    "psf_mixture_digest_version",
    "psf_mixture_provenance_convention",
    "apply_spatial_psf",
    "pair_n_sigma",
    "exact_pair_prune",
    "pixel_response_convention",
    "analytic_image_quadrature_order",
    "analytic_images_enabled",
    "analytic_spectra_enabled",
    "spectral_response_digest",
    "spectral_exposure_count",
    "psf_jacobian_rel_tol",
    "psf_jacobian_algorithm",
    "psf_jacobian_max_tiles_per_axis",
    "psf_jacobian_cap_policy",
)


def checkpoint_absolute_step(
    checkpoint: Mapping[str, object], *, checkpoint_path: Path,
) -> int:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Cannot determine absolute training step for {checkpoint_path}: "
            f"checkpoint is {type(checkpoint).__name__}, not a mapping."
        )
    resume_meta = checkpoint.get("resume_meta")
    top_level_fields = sorted(str(key) for key in checkpoint)
    if not isinstance(resume_meta, Mapping):
        raise ValueError(
            f"Cannot determine absolute training step for {checkpoint_path}: "
            f"resume_meta is {type(resume_meta).__name__}; fields present are "
            f"{top_level_fields}."
        )
    resume_fields = sorted(str(key) for key in resume_meta)
    # Precedence matters: checkpoints written since the budget-semantics fix
    # store the absolute step under "absolute_step" and demote
    # "steps_ran_this_invocation" to a genuinely invocation-relative count, so
    # the legacy field is only trustworthy as the absolute step when neither
    # newer field exists.
    if "global_step_completed" in resume_meta:
        field = "global_step_completed"
    elif "absolute_step" in resume_meta:
        field = "absolute_step"
    else:
        field = "steps_ran_this_invocation"
    value = resume_meta.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Cannot determine absolute training step for {checkpoint_path}: "
            f"{field}={value!r}; resume_meta fields present are {resume_fields}."
        )
    return value


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def model_config_for_checkpoint(model) -> dict[str, object]:
    base = unwrap_model(model)
    if isinstance(base, SplatField):
        pad_blue = int(getattr(base.sed, "pad_knots_blue", 0))
        pad_red = int(getattr(base.sed, "pad_knots_red", 0))
        return {
            "name": "splat",
            "n_splats": int(base.mu.shape[0]),
            "n_templates": int(base.sed_coeffs.shape[1]),
            # data-span knot count, matching the config key; the stored grid
            # buffer holds n_loglam_grid + both pads knots
            "n_loglam_grid": int(base.sed.loglam_grid.numel()) - pad_blue - pad_red,
            "sed_off_grid": str(base.sed.off_grid),
            "sed_pin_gauge_knots": bool(base.sed.pin_gauge_knots),
            "sed_pad_knots_blue": pad_blue,
            "sed_pad_knots_red": pad_red,
        }
    config: dict[str, object] = {}
    output_head = getattr(base, "output_head", None)
    if output_head is not None:
        config["output_head"] = str(output_head)
    return config


def validate_checkpoint_model_config(
    model, checkpoint: Mapping[str, object], *, checkpoint_path: Path, context: str,
) -> None:
    current = model_config_for_checkpoint(model)
    if not current:
        return
    if current.get("name") == "splat":
        raw_config = checkpoint.get("model_config")
        if not isinstance(raw_config, Mapping) or "name" not in raw_config:
            # legacy splat checkpoint written before provenance was recorded;
            # state_dict shapes are still enforced at load time. Its absent SED
            # keys are checked below against the legacy clamp/false semantics.
            print(
                f"Warning: {context} checkpoint {checkpoint_path} predates splat "
                "model_config provenance; skipping structural validation."
            )
            raw_config = raw_config if isinstance(raw_config, Mapping) else {}
        else:
            for key in _SPLAT_CONFIG_KEYS[:4] + _SPLAT_PAD_CONFIG_KEYS:
                if key in _SPLAT_PAD_CONFIG_KEYS:
                    saved = raw_config.get(key, 0)
                else:
                    saved = raw_config.get(key)
                wanted = current[key]
                if saved != wanted:
                    raise ValueError(
                        f"{context} checkpoint model_config.{key}={saved!r} does not "
                        f"match the current model ({wanted!r}). Use the original run "
                        "config or a matching checkpoint."
                    )
        missing_sed_keys = [
            key for key in _SPLAT_SED_CONFIG_KEYS if key not in raw_config
        ]
        if missing_sed_keys:
            if len(missing_sed_keys) != len(_SPLAT_SED_CONFIG_KEYS):
                raise ValueError(
                    f"{context} checkpoint {checkpoint_path} has incomplete SED "
                    "model_config provenance: both sed_off_grid and "
                    "sed_pin_gauge_knots must be present, or both must be absent."
                )
            legacy_sed_config = {
                "sed_off_grid": "clamp", "sed_pin_gauge_knots": False,
            }
            live_sed_config = {
                key: current[key] for key in _SPLAT_SED_CONFIG_KEYS
            }
            if live_sed_config != legacy_sed_config:
                raise ValueError(
                    f"{context} checkpoint {checkpoint_path} lacks the SED "
                    "provenance keys sed_off_grid and sed_pin_gauge_knots; "
                    "legacy checkpoints mean sed_off_grid='clamp' and "
                    "sed_pin_gauge_knots=False, but the current model uses "
                    f"sed_off_grid={live_sed_config['sed_off_grid']!r} and "
                    f"sed_pin_gauge_knots={live_sed_config['sed_pin_gauge_knots']!r}."
                )
            warnings.warn(
                f"{context} checkpoint {checkpoint_path} predates SED operator "
                "provenance; assuming sed_off_grid='clamp' and "
                "sed_pin_gauge_knots=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        sed_mismatches = [
            (key, raw_config[key], current[key])
            for key in _SPLAT_SED_CONFIG_KEYS
            if raw_config[key] != current[key]
        ]
        if sed_mismatches:
            details = "; ".join(
                f"model_config.{key}={saved!r} vs current {wanted!r}"
                for key, saved, wanted in sed_mismatches
            )
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} has SED operator "
                f"settings that do not match the current model: {details}. "
                "Use the original run config or a matching checkpoint."
            )
        return
    raw_checkpoint_config = checkpoint.get("model_config")
    if not isinstance(raw_checkpoint_config, Mapping):
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} is missing required model_config. "
            "Use a checkpoint written by the current training entrypoint."
        )
    saved_output_head = raw_checkpoint_config.get("output_head")
    current_output_head = current.get("output_head")
    if saved_output_head is None:
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} is missing required "
            "model_config.output_head. Use a checkpoint written by the current "
            "training entrypoint."
        )
    if current_output_head is not None and str(saved_output_head) != str(current_output_head):
        raise ValueError(
            f"{context} checkpoint model output_head={saved_output_head!r} does not match "
            f"current config output_head={current_output_head!r}. "
            "Use the original run config or override model.output_head to match the checkpoint."
        )


# Width of the pair support before the config knob existed. UNRELIABLE as a
# blanket legacy default: some earlier runs set other widths through
# environment knobs, and the oldest runs used tile-shaped support.
# Absent provenance therefore means UNKNOWN, not 3.5.
LEGACY_PAIR_N_SIGMA = 3.5

SPLAT_LINEAGE_BUFFER_DEFAULTS = {
    "halving_depth": lambda n: torch.zeros(n, dtype=torch.uint8),
    "last_halving_event": lambda n: torch.full((n,), -1, dtype=torch.int32),
    "birth_event": lambda n: torch.full((n,), -1, dtype=torch.int32),
}


def migrate_checkpoint_model_state(checkpoint: Mapping[str, object]) -> None:
    """Fill lineage buffers absent from pre-v4 splat checkpoints, in place.

    Only checkpoint_version <= 3 states are migrated: those predate the
    lineage buffers, so defaults are the truth. A v4+ state missing a lineage
    key is corrupt and must fail the strict load instead of being silently
    defaulted. Only states that look like a splat field (a top-level
    per-splat "mu") are touched."""
    version = checkpoint.get("checkpoint_version")
    if isinstance(version, int) and version >= 4:
        return
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or "mu" not in model_state:
        return
    n = int(model_state["mu"].shape[0])
    for name, default in SPLAT_LINEAGE_BUFFER_DEFAULTS.items():
        if name not in model_state:
            model_state[name] = default(n)


# The RNG payload is an OPTIONAL checkpoint key, so it carries its own schema
# version instead of moving CURRENT_CHECKPOINT_VERSION: every checkpoint the
# entrypoint has ever written stays loadable, and a run with
# trainer.checkpoint_rng_state off writes a byte-for-byte pre-change checkpoint.
RNG_STATE_KEY = "rng_state"
RNG_STATE_SCHEMA_VERSION = 2


def _capture_importance_pending_batch(trainer) -> dict[str, torch.Tensor] | None:
    pending = getattr(trainer, "_importance_pending_batch", None)
    if pending is None:
        return None
    result = pending.result()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "Importance sampler prefetch returned an invalid pending batch; "
            f"expected (indices, weights), got {type(result).__name__}."
        )
    indices, weights = result
    if not torch.is_tensor(indices) or not torch.is_tensor(weights):
        raise RuntimeError(
            "Importance sampler prefetch returned non-tensor indices or weights."
        )
    if indices.ndim != 1 or weights.ndim != 1 or indices.numel() != weights.numel():
        raise RuntimeError(
            "Importance sampler prefetch returned mismatched one-dimensional "
            f"indices and weights (shapes={tuple(indices.shape)}, "
            f"{tuple(weights.shape)})."
        )
    return {
        "indices": indices.detach().to(dtype=torch.long, device="cpu").clone(),
        "weights": weights.detach().to(dtype=torch.float32, device="cpu").clone(),
    }


def _encode_numpy_rng_state(state: tuple) -> dict[str, object]:
    # np.random.get_state() embeds a uint32 ndarray, which torch.load's
    # weights_only=True default refuses to unpickle. Checkpoints must stay
    # loadable through train.py's _load_checkpoint, so the ndarray travels as
    # an int64 torch tensor (every MT19937 key fits) and is rebuilt on restore.
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": str(name),
        "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng_state(payload: Mapping[str, object]) -> tuple:
    keys = payload["keys"]
    if torch.is_tensor(keys):
        keys = keys.cpu().numpy()
    return (
        str(payload["bit_generator"]),
        np.asarray(keys, dtype=np.uint32),
        int(payload["pos"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def capture_rng_state(trainer) -> dict[str, object]:
    """Snapshot every stateful sampler a training step draws from.

    Covers the three process-global streams (python, numpy, torch CPU), the
    torch CUDA stream of the trainer's device, the image importance sampler's
    private generator, the kernel factory's per-device jitter generators, and
    the epoch position. Anything omitted here silently replays its prefix when
    the run resumes.

    The importance prefetch is completed before its generator state is read; the
    completed but unconsumed batch is retained in the payload."""
    device = torch.device(getattr(trainer, "device", "cpu"))
    importance = getattr(trainer, "image_importance_generator", None)
    importance_pending_batch = _capture_importance_pending_batch(trainer)
    if importance is None and importance_pending_batch is not None:
        raise RuntimeError(
            "Importance sampler has a pending batch but no importance generator."
        )
    factory = getattr(getattr(trainer, "data", None), "kernel_factory", None)
    factory_state = (
        factory.rng_state_dict()
        if callable(getattr(factory, "rng_state_dict", None))
        else None
    )
    return {
        "schema": RNG_STATE_SCHEMA_VERSION,
        "world_size": int(getattr(trainer, "world_size", 1)),
        "device_type": device.type,
        "python": random.getstate(),
        "numpy": _encode_numpy_rng_state(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
        "image_importance_generator": (
            None if importance is None else importance.get_state()
        ),
        "importance_pending_batch": importance_pending_batch,
        "dataloader_base_seed": int(trainer._checkpoint_loader_base_seed),
        "kernel_factory": factory_state,
        "epoch_position": trainer.epoch_position(),
    }


def restore_rng_state(
    trainer,
    checkpoint: Mapping[str, object],
    *,
    checkpoint_path: Path,
    context: str = "Resume",
) -> bool:
    """Restore sampler state saved by capture_rng_state. True when applied.

    Never raises on an absent payload: checkpoints written before this existed
    must still resume, loudly, with the prefix-replay defect intact."""
    if not bool(getattr(trainer, "checkpoint_rng_state_enabled", False)):
        if RNG_STATE_KEY in checkpoint:
            print(
                f"WARNING: {context} checkpoint {checkpoint_path} carries sampler "
                "RNG state but trainer.checkpoint_rng_state is false; the resumed "
                "run REPLAYS the original draw prefix.",
                flush=True,
            )
        return False
    state = checkpoint.get(RNG_STATE_KEY)
    if state is None:
        print(
            f"WARNING: {context} checkpoint {checkpoint_path} predates sampler "
            "RNG-state checkpointing; RNG state was NOT restored and the resumed "
            "run REPLAYS the original run's draw prefix (python, numpy, torch, "
            "importance sampler and epoch position all restart).",
            flush=True,
        )
        return False
    if not isinstance(state, Mapping):
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} has a malformed "
            f"{RNG_STATE_KEY}: expected a mapping, got {type(state).__name__}."
        )
    schema = state.get("schema")
    if schema != RNG_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} has {RNG_STATE_KEY} schema "
            f"{schema!r}; this build writes and reads schema "
            f"{RNG_STATE_SCHEMA_VERSION}."
        )
    saved_world_size = int(state.get("world_size", 1))
    world_size = int(getattr(trainer, "world_size", 1))
    if saved_world_size > 1 or world_size > 1:
        print(
            f"WARNING: {context} checkpoint {checkpoint_path} RNG state was NOT "
            f"restored: distributed runs are excluded (checkpoint world_size="
            f"{saved_world_size}, current world_size={world_size}). Only rank 0's "
            "streams are checkpointed, so restoring them would desynchronize the "
            "other ranks. The resumed run replays its draw prefix.",
            flush=True,
        )
        return False

    random.setstate(state["python"])
    np.random.set_state(_decode_numpy_rng_state(state["numpy"]))
    # generator states are CPU byte tensors; torch.load's map_location will have
    # dragged them onto the training device with everything else
    torch.set_rng_state(state["torch_cpu"].cpu())
    device = torch.device(getattr(trainer, "device", "cpu"))
    saved_cuda = state.get("torch_cuda")
    if device.type == "cuda" and saved_cuda is not None:
        torch.cuda.set_rng_state(saved_cuda.cpu(), device)
    elif device.type == "cuda" or saved_cuda is not None:
        print(
            f"WARNING: {context} checkpoint {checkpoint_path} was written on a "
            f"{state.get('device_type')!r} device but resumes on "
            f"{device.type!r}; the CUDA RNG stream was NOT restored.",
            flush=True,
        )

    importance = getattr(trainer, "image_importance_generator", None)
    saved_importance = state.get("image_importance_generator")
    if importance is not None and "importance_pending_batch" not in state:
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} has an incomplete {RNG_STATE_KEY}: "
            "the importance prefetch payload is missing."
        )
    if importance is not None and saved_importance is not None:
        importance.set_state(saved_importance.cpu())
    elif importance is not None or saved_importance is not None:
        print(
            f"WARNING: {context} checkpoint {checkpoint_path} and this trainer "
            "disagree about image importance sampling "
            f"(checkpoint state present={saved_importance is not None}, trainer "
            f"generator present={importance is not None}); the importance draw "
            "stream was NOT restored.",
            flush=True,
        )
    saved_pending = state.get("importance_pending_batch")
    if saved_pending is None:
        trainer._restored_importance_pending_batch = None
    else:
        if not isinstance(saved_pending, Mapping):
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} has a malformed "
                "importance_pending_batch; expected a mapping."
            )
        saved_indices = saved_pending.get("indices")
        saved_weights = saved_pending.get("weights")
        if not torch.is_tensor(saved_indices) or not torch.is_tensor(saved_weights):
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} has a malformed "
                "importance_pending_batch; indices and weights must be tensors."
            )
        if (
            saved_indices.ndim != 1
            or saved_weights.ndim != 1
            or saved_indices.numel() != saved_weights.numel()
        ):
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} has a malformed "
                "importance_pending_batch; indices and weights must be matching "
                "one-dimensional tensors."
            )
        if importance is None:
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} carries a pending "
                "importance batch but this trainer has no importance sampler."
            )
        expected_batch_size = trainer._resolve_image_batch_size()
        if int(saved_indices.numel()) != int(expected_batch_size):
            raise ValueError(
                f"{context} checkpoint {checkpoint_path} carries an importance "
                f"pending batch of size {saved_indices.numel()}, but this trainer "
                f"expects {expected_batch_size}."
            )
        trainer._restored_importance_pending_batch = (
            saved_indices.detach().to(dtype=torch.long, device="cpu").clone(),
            saved_weights.detach().to(dtype=torch.float32, device="cpu").clone(),
        )

    saved_loader_seed = state.get("dataloader_base_seed")
    if not isinstance(saved_loader_seed, int) or isinstance(saved_loader_seed, bool):
        raise ValueError(
            f"{context} checkpoint {checkpoint_path} has an invalid "
            f"dataloader_base_seed: {saved_loader_seed!r}."
        )
    trainer._checkpoint_loader_base_seed = int(saved_loader_seed)

    factory = getattr(getattr(trainer, "data", None), "kernel_factory", None)
    saved_factory = state.get("kernel_factory")
    if saved_factory is not None and callable(
        getattr(factory, "load_rng_state_dict", None)
    ):
        skipped = factory.load_rng_state_dict(saved_factory)
        if skipped:
            print(
                f"WARNING: {context} checkpoint {checkpoint_path} kernel-jitter "
                f"generators for devices {sorted(skipped)} do not exist on this "
                "host; those streams restart.",
                flush=True,
            )

    restored_epoch = trainer.load_epoch_position(state.get("epoch_position"))
    position = trainer.epoch_position()
    print(
        "Restored sampler RNG state from resume checkpoint "
        f"(epoch position {'restored' if restored_epoch else 'NOT restored'}: "
        f"epochs_completed={position['epochs_completed']}, "
        f"steps_into_epoch={position['steps_into_epoch']}).",
        flush=True,
    )
    return True


def checkpoint_pair_n_sigma(checkpoint: Mapping[str, object]) -> float | None:
    """Pair-support width the checkpoint was trained with; None = unknown.

    Checkpoints written before operator_config existed carry no reliable
    width: most trained at the hardcoded 3.5, but some runs set 5-12
    through environment knobs."""
    operator_config = checkpoint.get("operator_config")
    if isinstance(operator_config, Mapping):
        value = operator_config.get("pair_n_sigma")
        if value is not None:
            return float(value)
    return None


def warn_on_pair_n_sigma_mismatch(
    checkpoint: Mapping[str, object], current: float, *, context: str,
) -> bool:
    """Print a loud provenance warning when rendering a checkpoint through a
    different (or unknown) pair-support width than it trained with. Returns
    True when a warning was printed."""
    trained = checkpoint_pair_n_sigma(checkpoint)
    if trained is None:
        print(
            f"WARNING: {context} renders at pair_n_sigma={float(current)} but "
            "the checkpoint predates operator provenance (most such runs "
            f"trained at {LEGACY_PAIR_N_SIGMA}; some varied). "
            "Predictions may reinterpret the model under a different support.",
            flush=True,
        )
        return True
    if math.isclose(trained, float(current), rel_tol=1e-9):
        return False
    print(
        f"WARNING: {context} renders at pair_n_sigma={float(current)} but the "
        f"checkpoint trained at {trained}; predictions are a reinterpretation, "
        "not the run's archived operator. Set trainer.refine.pair_n_sigma to match for "
        "provenance-faithful renders.",
        flush=True,
    )
    return True


GAUGE_KEY = "gauge"
GAUGE_CANONICAL = "canonical"
GAUGE_SIDECAR_FILENAME = "model_canonical.pt"
# rows per gauge chunk: the fp64 transform allocates ~3 buffers of
# chunk x n_templates, so a 9.8M-splat census never materializes a full extra
# copy of the SED coefficients
GAUGE_SIDECAR_CHUNK = 262_144


def checkpoint_is_gauge_canonical(checkpoint: Mapping[str, object]) -> bool:
    return str(checkpoint.get(GAUGE_KEY, "")) == GAUGE_CANONICAL


def build_gauge_canonical_sidecar(
    checkpoint: Mapping[str, object],
    *,
    gauge_factors: torch.Tensor,
    source_step: int,
    source_name: str,
) -> dict[str, object]:
    """A gauge-canonical COPY of a checkpoint's model state, for offline work.

    Nothing the run depends on is touched: the live model, its Adam moments and
    the checkpoint file itself are left exactly as training produced them.
    Projecting the LIVE model instead is destructive,
    because every projected row loses its optimizer moments (no exact transport
    exists through softplus). The sidecar carries no optimizer state at all, so
    it cannot be resumed from, and it is derived: the raw checkpoint plus the
    band tables reproduce it exactly, which is why it is never archived."""
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("Gauge sidecar requires a model_state mapping.")
    missing = [key for key in ("mu", "sed_coeffs", "raw_flux") if key not in model_state]
    if missing:
        raise ValueError(
            f"Gauge sidecar requires splat model_state keys {missing}; the model "
            "has no band gauge to canonicalize."
        )
    n_splats = int(model_state["sed_coeffs"].shape[0])
    if not torch.is_tensor(gauge_factors) or gauge_factors.shape != (n_splats,):
        raise ValueError(
            "gauge_factors must be a per-splat vector "
            f"(got shape {getattr(gauge_factors, 'shape', None)} for {n_splats} splats)."
        )
    if not bool(torch.isfinite(gauge_factors).all()) or bool((gauge_factors < 0).any()):
        raise ValueError("gauge_factors must be finite and non-negative.")

    factors = gauge_factors.detach().to("cpu")
    canonical_state = {
        key: (value.detach().to("cpu", copy=True) if torch.is_tensor(value) else value)
        for key, value in model_state.items()
    }
    projected = _canonicalize_model_state(canonical_state, factors)
    resume_meta = checkpoint.get("resume_meta")
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping):
        sidecar_model_config = dict(model_config)
    else:
        sidecar_model_config = {}
    sidecar_model_config.setdefault("sed_off_grid", "clamp")
    sidecar_model_config.setdefault("sed_pin_gauge_knots", False)
    return {
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        GAUGE_KEY: GAUGE_CANONICAL,
        "gauge_drift_log": 0.0,
        "gauge_projected_splats": int(projected),
        "gauge_source_step": int(source_step),
        "gauge_source_checkpoint": str(source_name),
        "operator_config": checkpoint.get("operator_config"),
        "model_config": sidecar_model_config,
        "model_state": canonical_state,
        "calibration_state": checkpoint.get("calibration_state"),
        "optimizer_state": None,
        "lr_scheduler_state": checkpoint.get("lr_scheduler_state"),
        "resume_meta": dict(resume_meta) if isinstance(resume_meta, Mapping) else {},
    }


def _canonicalize_model_state(
    model_state: dict[str, object], gauge_factors: torch.Tensor,
) -> int:
    """Fold the band gauge into raw_flux in place, chunked. Returns row count."""
    sed_coeffs = model_state["sed_coeffs"]
    raw_flux = model_state["raw_flux"]
    n_splats = int(sed_coeffs.shape[0])
    projected = 0
    for start in range(0, n_splats, GAUGE_SIDECAR_CHUNK):
        end = min(start + GAUGE_SIDECAR_CHUNK, n_splats)
        # slices are views, so the row writes below land in canonical_state
        coeffs_chunk = sed_coeffs[start:end]
        flux_chunk = raw_flux[start:end]
        rows, new_coeffs, new_flux = canonicalize_gauge_rows(
            coeffs_chunk, flux_chunk, gauge_factors[start:end], drift_log=0.0,
        )
        if rows.numel() == 0:
            continue
        coeffs_chunk[rows] = new_coeffs
        flux_chunk[rows] = new_flux
        projected += int(rows.numel())
    return projected


def checkpoint_psf_model(checkpoint: Mapping[str, object]) -> str:
    """Recorded image PSF operator; old checkpoints are legacy Gaussian."""
    operator_config = checkpoint.get("operator_config")
    if isinstance(operator_config, Mapping):
        value = operator_config.get("psf_model")
        if value is not None:
            return str(value).strip().lower()
    return "gaussian"


def spectral_operator_identity(trainer) -> tuple[str, int]:
    """Hash actual spectral geometry and response, including declared provenance."""
    from ..pgm.contracts import SkyDiskApertureSpec, validated_spectral_source_scale

    digest = hashlib.sha256()

    def scalar(value) -> None:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def array(value) -> None:
        if value is None:
            scalar(None)
            return
        values = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        scalar(values.shape)
        digest.update(values.tobytes())

    data = getattr(trainer, "data", None)
    exposures = getattr(trainer, "exposures", getattr(data, "exposures", {})) or {}
    labels = getattr(data, "spec_label_by_exposure", {}) or {}
    roles = getattr(data, "spec_role_by_exposure", {}) or {}
    blocks = getattr(data, "spec_block_by_exposure", {}) or {}
    spectral_ids = set(labels) or set(blocks)
    summary = getattr(data, "holdout_region_summary", {}) or {}
    scalar({key: summary.get(key) for key in (
        "spectral_response_mode", "spectral_lsf_sampling", "spectral_response_missing_policy",
        "missing_spectral_response_labels", "missing_spectral_response_count",
    )})
    count = 0
    for exposure_id, exposure in sorted(exposures.items()):
        if spectral_ids and exposure_id not in spectral_ids:
            continue
        # Imaging exposures may retain the scene wavelength grid as metadata.
        # A nonempty lam_spec alone does not make them spectral observations.
        if not spectral_ids and exposure.lam_edges is None and exposure.bandpasses:
            continue
        if np.asarray(exposure.lam_spec).size == 0:
            continue
        count += 1
        scalar({"exposure_id": int(exposure_id), "label": labels.get(exposure_id, exposure.label),
                "role": roles.get(exposure_id), "fwhm_arcsec": exposure.psf_fwhm_arcsec,
                "pixel_scale_arcsec": exposure.pixel_scale_arcsec,
                "metadata": exposure.spectral_response_metadata})
        array(exposure.lam_spec)
        array(exposure.lam_edges)
        array(exposure.lsf_fwhm_angstrom)
        array(validated_spectral_source_scale(exposure))
        aperture = exposure.aperture
        if isinstance(aperture, SkyDiskApertureSpec):
            scalar({"aperture": "sky_disk", "center": aperture.center_radec_deg,
                    "radius": aperture.radius_arcsec})
        else:
            scalar({"aperture": "mask"})
            array(getattr(aperture, "mask", exposure.fiber_mask))
    return digest.hexdigest(), count


def operator_config_for_trainer(trainer) -> dict[str, object]:
    """Return the complete image and spectral operator identity for a live trainer."""
    spectral_digest, spectral_count = spectral_operator_identity(trainer)
    return {
        "operator_schema_version": OPERATOR_SCHEMA_VERSION,
        "psf_model": str(getattr(trainer, "psf_model", "gaussian")).strip().lower(),
        "eigen_mog_config": getattr(trainer, "eigen_mog_config", None),
        "psf_mixture_digest": getattr(trainer, "psf_mixture_digest", None),
        "psf_mixture_digest_version": getattr(
            trainer, "psf_mixture_digest_version", None
        ),
        "psf_mixture_provenance_convention": getattr(
            trainer, "psf_mixture_provenance_convention", None
        ),
        "apply_spatial_psf": bool(getattr(trainer, "apply_spatial_psf", True)),
        "pair_n_sigma": float(getattr(trainer, "pair_n_sigma", 8.0)),
        "exact_pair_prune": bool(exact_pair_pruning_enabled()),
        "pixel_response_convention": (
            "native_effective_samples"
            if getattr(trainer, "psf_model", "gaussian") == "eigen_mog"
            else PIXEL_RESPONSE_CONVENTION
        ),
        "analytic_image_quadrature_order": int(
            getattr(trainer, "analytic_image_quadrature_order", 1)
        ),
        "spectral_response_digest": spectral_digest,
        "spectral_exposure_count": spectral_count,
        "analytic_spectra_enabled": bool(getattr(trainer, "analytic_spectra_enabled", False)),
        "analytic_images_enabled": bool(
            getattr(trainer, "analytic_images_enabled", False)
        ),
        "psf_jacobian_rel_tol": float(
            getattr(trainer, "psf_jacobian_rel_tol", 1e-2)
        ),
        "psf_jacobian_algorithm": str(
            getattr(trainer, "psf_jacobian_algorithm", PSF_JACOBIAN_ALGORITHM)
        ),
        "psf_jacobian_max_tiles_per_axis": int(
            getattr(
                trainer,
                "psf_jacobian_max_tiles_per_axis",
                PSF_JACOBIAN_MAX_TILES_PER_AXIS,
            )
        ),
        "psf_jacobian_cap_policy": str(
            getattr(trainer, "psf_jacobian_cap_policy", PSF_JACOBIAN_CAP_POLICY)
        ),
    }


def checkpoint_operator_identity(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Return a JSON-safe checkpoint identity, marking absent fields unknown."""
    raw = checkpoint.get("operator_config")
    config = raw if isinstance(raw, Mapping) else {}
    return {key: config.get(key, "unknown") for key in OPERATOR_IDENTITY_KEYS}


def _operator_values_match(key: str, trained: object, current: object) -> bool:
    if key in {"pair_n_sigma", "psf_jacobian_rel_tol"}:
        try:
            return math.isclose(float(trained), float(current), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return trained == current


def validate_checkpoint_operator_config(
    checkpoint: Mapping[str, object],
    *,
    current_operator_config: Mapping[str, object],
    context: str,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    allow_psf_model_change: bool = False,
    allow_operator_change: bool = False,
    allow_operator_reinterpretation: bool = False,
) -> bool:
    """Validate complete operator provenance for resume and evaluation."""
    current = dict(current_operator_config)
    missing_current = [key for key in OPERATOR_IDENTITY_KEYS if key not in current]
    if missing_current:
        raise ValueError(
            f"Current operator identity is missing required fields {missing_current}."
        )
    current_model = str(current["psf_model"]).strip().lower()
    if current_model not in {"gaussian", "dgpsf", "mog", "eigen_mog"}:
        raise ValueError(f"Invalid current PSF model {current_model!r}.")
    current["psf_model"] = current_model

    raw_operator_config = checkpoint.get("operator_config")
    operator_config = (
        raw_operator_config if isinstance(raw_operator_config, Mapping) else {}
    )
    saved_schema = operator_config.get("operator_schema_version")
    legacy = not isinstance(saved_schema, int) or saved_schema < OPERATOR_SCHEMA_VERSION
    unknown = [
        key for key in OPERATOR_IDENTITY_KEYS
        if key not in operator_config
    ]
    incomplete = legacy or bool(unknown)
    location = f" {checkpoint_path}" if checkpoint_path is not None else ""
    if incomplete:
        incomplete_parts = []
        if legacy:
            incomplete_parts.append(
                f"schema {saved_schema!r} predates required schema "
                f"{OPERATOR_SCHEMA_VERSION}"
            )
        if unknown:
            incomplete_parts.append(f"missing fields {unknown}")
        detail = "incomplete operator identity: " + "; ".join(incomplete_parts)
        if resume and not allow_operator_change:
            raise ValueError(
                f"{context} checkpoint{location} has {detail}. Resume is "
                "fail-closed; set resume.allow_operator_change=true to accept "
                "the unverified operator explicitly."
            )
        if not resume and not allow_operator_reinterpretation:
            raise ValueError(
                f"{context} checkpoint{location} has {detail}. Evaluation is "
                "fail-closed; set eval.allow_operator_reinterpretation=true "
                "or the equivalent CLI flag to opt in explicitly."
            )
        action = "resume operator override" if resume else "operator reinterpretation"
        print(
            f"WARNING: explicit {action} for {context} checkpoint{location}: "
            f"{detail}. Operator provenance cannot be verified.",
            flush=True,
        )
        return True

    mismatches: list[tuple[str, object, object]] = []
    for key in OPERATOR_IDENTITY_KEYS:
        trained = operator_config.get(key, "unknown")
        wanted = current[key]
        if not _operator_values_match(key, trained, wanted):
            mismatches.append((key, trained, wanted))
    if "eigen_mog" in {current_model, str(operator_config.get("psf_model", ""))}:
        trained = operator_config.get("eigen_mog_config")
        wanted = current.get("eigen_mog_config")
        if current_model == "eigen_mog" and not isinstance(wanted, Mapping):
            raise ValueError("eigen_mog operator identity lacks representation configuration")
        if trained != wanted:
            mismatches.append(("eigen_mog_config", trained, wanted))
    if not mismatches:
        return False
    detail = "; ".join(
        f"{key}={trained!r} vs {wanted!r}"
        for key, trained, wanted in mismatches
    )
    mismatch_keys = {key for key, _trained, _wanted in mismatches}
    psf_fields = {
        "psf_model",
        "eigen_mog_config",
        "psf_mixture_digest",
        "psf_mixture_digest_version",
        "psf_mixture_provenance_convention",
    }
    if "eigen_mog" in {current_model, str(operator_config.get("psf_model", ""))}:
        psf_fields.add("pixel_response_convention")
    psf_only = mismatch_keys <= psf_fields
    resume_allowed = allow_operator_change or (
        allow_psf_model_change and psf_only
    )
    if resume and not resume_allowed:
        raise ValueError(
            f"{context} checkpoint{location} operator mismatch: {detail}. "
            "Match the checkpoint operator, set trainer.allow_psf_model_change=true "
            "for a PSF-only change, or set resume.allow_operator_change=true to "
            "accept a broader operator change explicitly."
        )
    if resume:
        print(
            "WARNING: explicit resume operator override: resuming across "
            f"operator mismatch ({detail}).",
            flush=True,
        )
        return True
    if not allow_operator_reinterpretation:
        raise ValueError(
            f"{context} checkpoint{location} operator mismatch: {detail}. "
            "Evaluation is fail-closed; set eval.allow_operator_reinterpretation=true "
            "or the equivalent CLI flag to opt in explicitly."
        )
    print(
        f"WARNING: explicit operator reinterpretation for {context} checkpoint"
        f"{location}: {detail}. Predictions reinterpret the archived model.",
        flush=True,
    )
    return True
