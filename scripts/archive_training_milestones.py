"""Preserve fixed-evaluation checkpoints while a training run is active."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from survey_fusion.training.checkpoints import checkpoint_absolute_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cadence", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def _checkpoint_step(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint_absolute_step(state, checkpoint_path=path)


def _load_fixed_history(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return payload


def archive_available_milestone(run_dir: Path, *, cadence: int) -> int | None:
    history_path = run_dir / "fixed_holdout_history.json"
    if not history_path.is_file():
        return None
    history = _load_fixed_history(history_path)
    if not history:
        return None
    latest_step = int(history[-1]["step"])
    if latest_step <= 0 or latest_step % cadence:
        raise ValueError(
            f"Fixed held-out step {latest_step} is not a positive multiple of {cadence}."
        )
    destination = run_dir / f"checkpoint_step_{latest_step}.pt"
    image_snapshot = run_dir / f"model_cutouts_step_{latest_step}.png"
    history_snapshot = run_dir / f"fixed_holdout_history_step_{latest_step}.json"
    if destination.is_file():
        if _checkpoint_step(destination) != latest_step:
            raise ValueError(f"{destination} has the wrong checkpoint step.")
        if not image_snapshot.is_file() or not history_snapshot.is_file():
            raise RuntimeError(
                f"Archived checkpoint step {latest_step} is missing its diagnostics."
            )
        return latest_step

    checkpoint = run_dir / "model.pt"
    checkpoint_step = _checkpoint_step(checkpoint)
    if checkpoint_step < latest_step:
        return None
    if checkpoint_step > latest_step:
        raise RuntimeError(
            f"Checkpoint advanced to step {checkpoint_step} before fixed step "
            f"{latest_step} was archived."
        )

    image = run_dir / "model_cutouts.png"
    if not image.is_file():
        raise FileNotFoundError(image)
    shutil.copy2(image, image_snapshot)
    with history_snapshot.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")
    os.link(checkpoint, destination)
    return latest_step


def _training_completed(speed_log: Path) -> bool:
    if not speed_log.is_file():
        return False
    with speed_log.open(encoding="utf-8") as handle:
        return any(
            json.loads(line).get("type") == "run_end"
            for line in handle
            if line.strip()
        )


def main() -> None:
    args = parse_args()
    archived: set[int] = set()
    while True:
        step = archive_available_milestone(args.run_dir, cadence=args.cadence)
        if step is not None and step not in archived:
            archived.add(step)
            print(
                json.dumps(
                    {
                        "type": "milestone_archived",
                        "step": step,
                        "run_dir": str(args.run_dir),
                    }
                ),
                flush=True,
            )
        if _training_completed(args.run_dir / "speed.log"):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
