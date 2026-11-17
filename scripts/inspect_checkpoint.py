"""Print the canonical shell-readable summary of a training checkpoint.

Usage: python scripts/inspect_checkpoint.py <checkpoint.pt>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _plain_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable fields used by shell-side checkpoint inspection."""
    resume_meta = checkpoint.get("resume_meta")
    if not isinstance(resume_meta, Mapping):
        resume_meta = {}
    global_step = resume_meta.get("global_step_completed")
    if global_step is None:
        global_step = resume_meta.get("steps_ran_this_invocation")

    model_state = checkpoint.get("model_state")
    mu = model_state.get("mu") if isinstance(model_state, Mapping) else None
    mu_rows = int(mu.shape[0]) if torch.is_tensor(mu) and mu.ndim >= 1 else "missing"

    summary: dict[str, Any] = {
        "checkpoint_version": checkpoint.get("checkpoint_version", "missing"),
        "global_step": "missing" if global_step is None else int(global_step),
        "epochs": resume_meta.get("epochs_ran_this_invocation", "missing"),
        "saved_at": resume_meta.get("saved_at_utc", "missing"),
        "mu_rows": mu_rows,
    }
    operator_config = checkpoint.get("operator_config")
    if isinstance(operator_config, Mapping):
        for key in sorted(operator_config):
            summary[str(key)] = operator_config[key]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    for key, value in checkpoint_summary(checkpoint).items():
        print(f"{key}={_plain_value(value)}")


if __name__ == "__main__":
    main()
