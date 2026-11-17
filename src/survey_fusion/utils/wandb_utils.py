"""Lightweight Weights & Biases integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping


class WandbRun:
    """Thin wrapper around a wandb run with graceful opt-out."""

    def __init__(
        self, cfg: Mapping[str, Any], run_dir: Path, full_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ):
        self.enabled = bool(cfg.get("enabled", False))
        self.run = None
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"wandb not available ({exc}); disabling wandb logging.")
            self.enabled = False
            return

        self._wandb = wandb
        settings = None
        try:
            settings = wandb.Settings(save_code=bool(cfg.get("save_code", False)))
        except Exception as exc:  # pragma: no cover - optional
            print(f"wandb.Settings init failed ({exc}); continuing without custom settings.")
        run_name = cfg.get("run_name")
        init_kwargs = {
            "project": cfg.get("project"), "entity": cfg.get("entity"), "name": run_name,
            "group": cfg.get("group"), "notes": cfg.get("notes"), "config": full_config, "dir": str(run_dir),
            "mode": cfg.get("mode", "online"),
        }
        if settings is not None:
            init_kwargs["settings"] = settings
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        self.run = wandb.init(**init_kwargs)
        if cfg.get("watch_model", False) and model is not None:
            self.watch(model, log=cfg.get("watch", "gradients"))

    def watch(self, model: Any, *, log: str = "gradients") -> None:
        if not self.run:
            return
        try:
            self.run.watch(model, log=log, log_freq=0)
        except Exception as exc:  # pragma: no cover - optional
            print(f"wandb.watch failed; continuing without watching (error: {exc})")

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        if not self.run:
            return
        self.run.log(data, step=step)

    def log_images(self, images: Mapping[str, Path], step: int | None = None) -> None:
        if not self.run or not self._wandb:
            return
        payload = {}
        for key, path in images.items():
            path = Path(path)
            if path.exists():
                payload[key] = self._wandb.Image(str(path))
        if payload:
            self.run.log(payload, step=step)

    def log_artifacts(self, name: str, files: Mapping[str, Path]) -> None:
        if not self.run or not self._wandb:
            return
        artifact = self._wandb.Artifact(name=name, type="run_artifacts")
        added = 0
        for alias, path in files.items():
            path = Path(path)
            if path.exists():
                artifact.add_file(str(path), name=alias)
                added += 1
        if added > 0:
            self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run:
            self.run.finish()
        self.run = None
        self._wandb = None

    def __bool__(self) -> bool: # pragma: no cover - trivial
        return bool(self.enabled and self.run is not None)
