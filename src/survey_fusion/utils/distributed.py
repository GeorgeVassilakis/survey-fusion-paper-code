from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist

__all__ = (
    "DistributedContext", "broadcast_object", "destroy_distributed", "initialize_distributed",
    "is_distributed_active", "resolve_distributed_batch_size", "resolve_distributed_context",
)


_OBJECT_BROADCAST_GROUP: dist.ProcessGroup | None = None


@dataclass(frozen=True, slots=True)
class DistributedContext:
    enabled: bool = False
    backend: str | None = None
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str | None = None
    master_port: int | None = None
    init_method: str = "env://"

    @property
    def is_primary(self) -> bool:
        return int(self.rank) == 0


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _deterministic_master_port(env_map: Mapping[str, str]) -> int:
    raw_job_id = str(env_map.get("SLURM_JOB_ID", "") or "").strip()
    if raw_job_id.isdigit():
        return 15000 + (int(raw_job_id) % 20000)
    return 29500


def _slurm_num_nodes(env_map: Mapping[str, str]) -> int | None:
    for key in ("SLURM_STEP_NUM_NODES", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        raw = str(env_map.get(key, "") or "").strip()
        if raw.isdigit():
            return int(raw)
    return None


def resolve_distributed_context(
    platform_cfg: Mapping[str, Any] | None = None, *, env: Mapping[str, str] | None = None,
) -> DistributedContext:
    platform_map = dict(platform_cfg or {})
    dist_cfg = _as_mapping(platform_map.get("distributed"))
    enabled = bool(dist_cfg.get("enabled", False))
    if not enabled:
        return DistributedContext()

    env_map = env if env is not None else os.environ
    backend = str(dist_cfg.get("backend", "nccl") or "nccl").strip().lower()
    world_size = int(
        env_map.get("WORLD_SIZE") or env_map.get("SLURM_NTASKS") or dist_cfg.get("world_size", 1) or 1
    )
    rank = int(env_map.get("RANK") or env_map.get("SLURM_PROCID") or dist_cfg.get("rank", 0) or 0)
    local_rank = int(
        env_map.get("LOCAL_RANK") or env_map.get("SLURM_LOCALID") or dist_cfg.get("local_rank", rank) or rank
    )
    master_addr = str(env_map.get("MASTER_ADDR") or dist_cfg.get("master_addr") or "127.0.0.1").strip()
    raw_port = env_map.get("MASTER_PORT") or dist_cfg.get("master_port")
    master_port = _deterministic_master_port(env_map) if raw_port in (None, "", "null") else int(raw_port)

    if world_size < 1:
        raise ValueError("Distributed world_size must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Distributed rank {rank} must be in [0, {world_size}).")
    if local_rank < 0:
        raise ValueError("Distributed local_rank must be non-negative.")
    slurm_num_nodes = _slurm_num_nodes(env_map)
    if slurm_num_nodes is not None and slurm_num_nodes > 1:
        raise ValueError(
            "Distributed training currently supports only single-node Slurm allocations; "
            f"got {slurm_num_nodes} nodes."
        )

    # Treat single-rank launches as eager non-distributed execution while
    # preserving resolved metadata for logging/tests.
    active = world_size > 1
    return DistributedContext(
        enabled=active,
        backend=backend if active else None,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        init_method="env://",
    )


def resolve_distributed_batch_size(
    batch_size: int, ctx: DistributedContext, *, mode: str = "per_gpu", label: str = "batch_size",
) -> int:
    resolved = int(batch_size)
    if resolved <= 0 or not ctx.enabled or int(ctx.world_size) <= 1:
        return resolved
    mode_key = str(mode or "per_gpu").strip().lower()
    if mode_key == "per_gpu":
        return resolved
    if mode_key != "global":
        raise ValueError("platform.distributed.batch_mode must be 'per_gpu' or 'global'.")
    if resolved < int(ctx.world_size):
        raise ValueError(f"{label}={resolved} must be >= world_size={ctx.world_size} in global batch mode.")
    if resolved % int(ctx.world_size) != 0:
        raise ValueError(
            f"{label}={resolved} must be divisible by world_size={ctx.world_size} " "in global batch mode."
        )
    return resolved // int(ctx.world_size)


def initialize_distributed(ctx: DistributedContext) -> None:
    global _OBJECT_BROADCAST_GROUP
    if not ctx.enabled or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", str(ctx.master_addr or "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", str(int(ctx.master_port or 29500)))
    os.environ.setdefault("RANK", str(int(ctx.rank)))
    os.environ.setdefault("LOCAL_RANK", str(int(ctx.local_rank)))
    os.environ.setdefault("WORLD_SIZE", str(int(ctx.world_size)))
    device_id: torch.device | None = None
    if str(ctx.backend).lower() == "nccl" and torch.cuda.is_available():
        current_idx = int(torch.cuda.current_device())
        device_id = torch.device(f"cuda:{current_idx}")
    init_kwargs: dict[str, Any] = {
        "backend": str(ctx.backend), "init_method": str(ctx.init_method), "rank": int(ctx.rank),
        "world_size": int(ctx.world_size),
    }
    if device_id is not None:
        init_kwargs["device_id"] = device_id
    dist.init_process_group(**init_kwargs)
    if str(ctx.backend).lower() != "gloo":
        _OBJECT_BROADCAST_GROUP = dist.new_group(backend="gloo")


def synchronize_distributed(ctx: DistributedContext, device: torch.device | None = None) -> None:
    if not is_distributed_active(ctx):
        return
    if str(ctx.backend).lower() == "nccl" and device is not None and device.type == "cuda":
        dist.barrier(device_ids=[int(device.index or 0)])
        return
    dist.barrier()


def destroy_distributed(ctx: DistributedContext) -> None:
    global _OBJECT_BROADCAST_GROUP
    if not ctx.enabled or not dist.is_initialized():
        return
    if _OBJECT_BROADCAST_GROUP is not None:
        dist.destroy_process_group(_OBJECT_BROADCAST_GROUP)
        _OBJECT_BROADCAST_GROUP = None
    dist.destroy_process_group()


def is_distributed_active(ctx: DistributedContext) -> bool:
    return bool(ctx.enabled and dist.is_initialized())


def broadcast_object(value: Any, ctx: DistributedContext) -> Any:
    if not is_distributed_active(ctx):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=0, group=_OBJECT_BROADCAST_GROUP)
    return payload[0]
