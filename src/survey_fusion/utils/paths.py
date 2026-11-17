"""Path resolution and directory management utilities.

This module provides helpers for locating the project root, resolving
relative paths, and managing output directories for runs.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict

__all__ = (
    "find_project_root", "resolve_path", "ensure_dir", "build_run_dir", "dump_config", "paths_from_config",
)


def find_project_root(marker: str = ".project-root") -> Path:
    """Find the project root directory by searching for a marker file.

    Walks up the directory tree from the current file until the marker
    file is found.

    Parameters
    ----------
    marker : `str`, optional
        Name of the marker file identifying the project root.

    Returns
    -------
    root : `pathlib.Path`
        Path to the project root directory.

    Raises
    ------
    FileNotFoundError
        If the marker file is not found in any parent directory.
    """
    current = Path(__file__).resolve()
    for candidate in [current] + list(current.parents):
        marker_path = candidate / marker
        if marker_path.exists():
            return marker_path.parent
    raise FileNotFoundError(f"Could not locate project root marker '{marker}' from {current}")


def resolve_path(value: str, project_root: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root.

    If the path is absolute, it is returned as is. If relative, it is
    resolved against the project root.

    Parameters
    ----------
    value : `str`
        The path string to resolve.
    project_root : `pathlib.Path`, optional
        The project root directory. If ``None``, it is discovered
        automatically.

    Returns
    -------
    path : `pathlib.Path`
        The resolved absolute path.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    root = project_root or find_project_root()
    return (root / path).resolve()


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : `pathlib.Path`
        The directory path to ensure.

    Returns
    -------
    path : `pathlib.Path`
        The verified directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_dir(
    outdir: Path, run_name: str | None = None, *, append_timestamp: bool = True,
    use_outdir_directly: bool = False, timestamp_format: str = "%Y%m%d-%H%M%S",
    suffix: str = "",
) -> Path:
    """Create and return a run-specific output directory.

    Parameters
    ----------
    outdir : `pathlib.Path`
        Base output directory.
    run_name : `str`, optional
        Name of the run. Defaults to "run" if ``None``.
    append_timestamp : `bool`, optional
        Whether to append a timestamp to the directory name.
    use_outdir_directly : `bool`, optional
        If `True`, returns ``outdir`` without creating a subdirectory.
    timestamp_format : `str`, optional
        Format string for the timestamp (strftime).

    Returns
    -------
    run_dir : `pathlib.Path`
        Path to the created run directory.
    """
    outdir = ensure_dir(outdir)
    if use_outdir_directly:
        target = outdir.with_name(f"{outdir.name}{suffix}") if suffix else outdir
        return ensure_dir(target)
    safe_name = (run_name or "run").replace(" ", "_")
    if append_timestamp:
        timestamp = _dt.datetime.utcnow().strftime(timestamp_format)
        target = outdir / f"{safe_name}_{timestamp}{suffix}"
    else:
        target = outdir / f"{safe_name}{suffix}"
    return ensure_dir(target)


def dump_config(cfg: Any, path: Path) -> None:
    """Write a configuration object to a YAML file.

    Parameters
    ----------
    cfg : `Any`
        Configuration object (typically an OmegaConf DictConfig).
    path : `pathlib.Path`
        Destination file path.
    """
    from omegaconf import OmegaConf

    with path.open("w", encoding="utf-8") as fh:
        fh.write(OmegaConf.to_yaml(cfg, resolve=True))


def paths_from_config(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve all paths in a configuration dictionary.

    Parameters
    ----------
    cfg : `dict`
        Configuration dictionary containing a "paths" section or being
        the paths section itself.

    Returns
    -------
    resolved : `dict`
        Dictionary mapping path keys to resolved absolute `pathlib.Path` objects.
        Always includes "project_root".
    """
    paths_cfg = cfg.get("paths", cfg)
    project_root = find_project_root(paths_cfg.get("project_root_marker", ".project-root"))
    resolved = {}
    for key, value in paths_cfg.items():
        if key == "project_root_marker":
            continue
        resolved[key] = resolve_path(value, project_root)
    resolved["project_root"] = project_root
    return resolved
