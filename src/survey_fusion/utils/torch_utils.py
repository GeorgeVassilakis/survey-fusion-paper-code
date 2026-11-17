"""PyTorch device selection and reproducibility utilities.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Literal, Mapping
import warnings

import numpy as np
import torch

from .perf_env import pinned_h2d_enabled

__all__ = (
    "AmpSettings", "apply_precision_policy", "describe_runtime", "resolve_amp_settings", "select_device",
    "as_tensor_on_device", "copy_tensor_to_device", "select_runtime_device",
    "set_seed", "tensor_on_device", "validate_runtime_environment",
)

DeviceLiteral = Literal["cpu", "cuda", "auto"]
PrecisionPolicyLiteral = Literal["portable_fp32", "tf32", "ampere_bf16"]

# Earlier runs recorded the TF32 policy as "ampere_tf32";
# reading those configs must keep selecting it.
_HISTORICAL_POLICY_NAMES = {"ampere_tf32": "tf32"}


def _canonical_policy(precision_policy: object) -> str:
    policy = str(precision_policy or "portable_fp32").strip().lower()
    return _HISTORICAL_POLICY_NAMES.get(policy, policy)


@dataclass(frozen=True, slots=True)
class AmpSettings:
    enabled: bool
    device_type: str | None
    dtype: torch.dtype | None


def as_tensor_on_device(
    data: Any, *, device: torch.device | str, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a tensor on ``device``, pinning host storage for CUDA copies."""
    target = torch.device(device)
    if target.type == "cuda" and pinned_h2d_enabled():
        host = torch.as_tensor(data, dtype=dtype, device="cpu")
        if not host.is_pinned():
            host = host.pin_memory()
        return host.to(device=target, non_blocking=True)
    return torch.as_tensor(data, dtype=dtype, device=target)


def tensor_on_device(
    data: Any, *, device: torch.device | str, dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy host values into a new tensor, pinning storage for CUDA copies."""
    target = torch.device(device)
    if target.type == "cuda" and pinned_h2d_enabled():
        host = torch.tensor(data, dtype=dtype, device="cpu", pin_memory=True)
        return host.to(device=target, non_blocking=True)
    return torch.tensor(data, dtype=dtype, device=target)


def copy_tensor_to_device(
    tensor: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy a tensor to ``device``, staging pageable CPU input when enabled."""
    target = torch.device(device)
    if (
        tensor.device.type == "cpu"
        and target.type == "cuda"
        and pinned_h2d_enabled()
    ):
        staged = tensor if tensor.is_pinned() else tensor.pin_memory()
        return staged.to(device=target, dtype=dtype, non_blocking=True)
    return tensor.to(device=target, dtype=dtype, non_blocking=non_blocking)


def _device_is_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    raise ValueError(f"Unsupported device {device!r}.")


def select_device(device: DeviceLiteral = "auto") -> torch.device:
    """Select the PyTorch device for computation.

    Parameters
    ----------
    device : `str`, optional
        Target device name ("cpu", "cuda", or "auto"). If "auto",
        attempts to select CUDA before falling back to CPU.

    Returns
    -------
    device : `torch.device`
        The selected PyTorch device.
    """
    if device == "auto":
        if _device_is_available("cuda"):
            return torch.device("cuda")
        return torch.device("cpu")
    if not _device_is_available(device):
        raise RuntimeError(f"Requested device '{device}' is not available.")
    return torch.device(device)


def select_runtime_device(
    device: DeviceLiteral = "auto", platform_cfg: Mapping[str, Any] | None = None,
) -> torch.device:
    platform_cfg = dict(platform_cfg or {})
    platform_name = str(platform_cfg.get("name", "default"))
    expected_device = str(platform_cfg.get("expected_device", "") or "").strip().lower()
    allow_cpu_fallback = bool(platform_cfg.get("allow_cpu_fallback", False))

    if device != "auto":
        resolved = select_device(device)
    elif expected_device in {"cpu", "cuda"}:
        if _device_is_available(expected_device):
            resolved = torch.device(expected_device)
        elif allow_cpu_fallback:
            warnings.warn(
                f"Platform '{platform_name}' expected {expected_device} but it is unavailable; "
                "falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved = torch.device("cpu")
        else:
            raise RuntimeError(
                f"Platform '{platform_name}' requires device '{expected_device}', but it is unavailable."
            )
    else:
        resolved = select_device(device)

    if expected_device and resolved.type != expected_device:
        if not (allow_cpu_fallback and resolved.type == "cpu"):
            raise RuntimeError(
                f"Platform '{platform_name}' requires device '{expected_device}', "
                f"but resolved '{resolved.type}'."
            )
    return resolved


def apply_precision_policy(
    precision_policy: PrecisionPolicyLiteral | str = "portable_fp32", device: torch.device | None = None,
) -> str:
    policy = _canonical_policy(precision_policy)
    if policy == "portable_fp32":
        torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return policy
    if policy == "tf32":
        if device is None or device.type != "cuda":
            raise RuntimeError("precision_policy='tf32' requires a CUDA device.")
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return policy
    if policy == "ampere_bf16":
        if device is None or device.type != "cuda":
            return policy
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return policy
    raise ValueError(
        f"Unsupported precision_policy {precision_policy!r}. "
        "Expected one of {'portable_fp32', 'tf32', 'ampere_bf16'}."
    )


def resolve_amp_settings(
    precision_policy: PrecisionPolicyLiteral | str = "portable_fp32", device: torch.device | None = None,
) -> AmpSettings:
    policy = _canonical_policy(precision_policy)
    if policy == "ampere_bf16" and device is not None and device.type == "cuda":
        return AmpSettings(enabled=True, device_type="cuda", dtype=torch.bfloat16)
    return AmpSettings(enabled=False, device_type=None, dtype=None)


def validate_runtime_environment(
    device: torch.device, platform_cfg: Mapping[str, Any] | None = None,
    *, env: Mapping[str, str] | None = None,
) -> None:
    platform_cfg = dict(platform_cfg or {})
    platform_name = str(platform_cfg.get("name", "default"))
    if not bool(platform_cfg.get("require_slurm", False)):
        return

    env_map: Mapping[str, str] = env if env is not None else os.environ
    if not env_map.get("SLURM_JOB_ID"):
        raise RuntimeError(
            f"Platform '{platform_name}' requires a Slurm allocation. "
            "Launch this profile with sbatch or srun."
        )

    required_partitions_cfg = platform_cfg.get("required_slurm_partitions")
    required_partitions: set[str] = set()
    if isinstance(required_partitions_cfg, (list, tuple, set)):
        required_partitions = {str(part).strip() for part in required_partitions_cfg if str(part).strip()}
    active_partition = str(
        env_map.get("SLURM_JOB_PARTITION") or env_map.get("SLURM_STEP_PARTITION") or ""
    ).strip()
    if required_partitions and active_partition:
        active_parts = {part.strip() for part in active_partition.split(",") if part.strip()}
        if required_partitions.isdisjoint(active_parts):
            allowed = ", ".join(sorted(required_partitions))
            raise RuntimeError(
                f"Platform '{platform_name}' requires one of Slurm partitions '{allowed}', "
                f"but job partition is '{active_partition}'."
            )

    if device.type == "cuda":
        visible_gpus = (
            env_map.get("CUDA_VISIBLE_DEVICES")
            or env_map.get("SLURM_STEP_GPUS")
            or env_map.get("SLURM_JOB_GPUS")
            or env_map.get("SLURM_GPUS_ON_NODE")
        )
        if not visible_gpus:
            warnings.warn(
                f"Platform '{platform_name}' resolved CUDA but no Slurm GPU visibility env var "
                "was detected. Verify the job requested a GPU explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )


def describe_runtime(
    device: torch.device, platform_cfg: Mapping[str, Any] | None = None,
    *, precision_policy: str | None = None,
) -> str:
    platform_cfg = dict(platform_cfg or {})
    platform_name = str(platform_cfg.get("name", "default"))
    policy = precision_policy or str(platform_cfg.get("precision_policy", "portable_fp32"))
    return f"Platform: {platform_name} | device={device.type} | precision_policy={policy}"


def startup_worker_count(n_tasks: int) -> int:
    """Threads to fan an independent per-exposure startup task over.

    The startup loops spend their time inside numpy and torch array calls,
    which release the GIL, so threads shorten the wall clock without touching a
    single result. The cap keeps the intra-op pools of many workers from
    oversubscribing a shared box.
    """
    return max(1, min(int(n_tasks), os.cpu_count() or 1, 32))


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : `int`
        The random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _device_is_available("cuda"):
        torch.cuda.manual_seed_all(seed)
