"""One-time preprocessing cache for the scene + datum store.

Parsing 1300+ FITS files and running ~1e9 WCS transforms produces exactly the
same scene and datum store every run: none of it depends on the model. Save
the built objects once and reload them.

The cache is a DIRECTORY: a small metadata pickle plus one raw .npy sidecar
per large array. A monolithic pickle materializes every numpy column by value,
which cost ~10 minutes to parse on load and far longer to serialize on save at
COSMOS scale. Sidecar arrays instead reload as copy-on-write memory maps: the
open is milliseconds and pages fault in from local disk as training touches
them. Copy-on-write keeps accidental writes private instead of corrupting the
shared cache.

The cache is keyed by a digest of everything that feeds the build: the data
config, seed (random holdout draws from it), and the manifest file's identity.
Any change rebuilds; a schema constant guards against code changes to the
store layout.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Spectral response metadata and calibration operators are part of the store.
# Reject pre-response stores even when an archival config selects legacy seeing.
PREPARED_SCHEMA = 9

_NON_CONTENT_KEYS = (
    "datum_spill_dir", "loader_workers", "prepared_dir", "prepared_cache",
    "prepare_seed", "expected_prepared_digest",
)

# Seed of the prepared store (it keys the cache and seeds the Monte-Carlo
# kernel factory; the random holdout has its own seed). Fixed rather than
# tied to run.seed, so runs at different model seeds share one store.
DEFAULT_PREPARE_SEED = 42


def prepare_seed(data_cfg: dict | None) -> int:
    value = dict(data_cfg or {}).get("prepare_seed", DEFAULT_PREPARE_SEED)
    if value is None or isinstance(value, bool):
        raise ValueError("data.prepare_seed must be an integer.")
    return int(value)

# arrays below this size stay inline in the metadata pickle; sidecar files
# have per-file open overhead that only pays off for real columns
_SIDECAR_MIN_BYTES = 1_000_000

_META_NAME = "meta.pkl"
_ARRAY_DIR = "arrays"


def prepared_digest(
    data_cfg: dict, *, seed: int, bands: list[str],
    manifest_path: str | os.PathLike | None, holdout_band: str | None = None,
) -> str:
    cfg = {k: v for k, v in dict(data_cfg).items() if k not in _NON_CONTENT_KEYS}
    manifest: dict[str, Any] = {"path": str(manifest_path)}
    if manifest_path is not None and Path(manifest_path).exists():
        content_digest = hashlib.sha256()
        with Path(manifest_path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                content_digest.update(chunk)
        manifest["sha256"] = content_digest.hexdigest()
    # Sidecar geometry/calibration provenance is content, not a filename alone.
    response_path = cfg.get("spectral_response_metadata_path")
    response_identity = None
    if response_path:
        response_file = Path(str(response_path))
        if not response_file.is_absolute() and cfg.get("observation_root"):
            response_file = Path(str(cfg["observation_root"])) / response_file
        response_identity = {"path": str(response_file),
                             "sha256": hashlib.sha256(response_file.read_bytes()).hexdigest()}
    payload = json.dumps(
        {
            "schema": PREPARED_SCHEMA, "cfg": cfg, "seed": int(seed),
            "bands": sorted(str(b) for b in bands),
            "manifest": manifest, "holdout_band": holdout_band,
            "spectral_response_metadata": response_identity,
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _prepared_digest_inputs(
    data_cfg: dict | None, *, seed: int, bands: list[str],
    manifest_path: str | os.PathLike | None, holdout_band: str | None,
) -> dict[str, Any]:
    cfg = {
        k: v for k, v in dict(data_cfg or {}).items()
        if k not in _NON_CONTENT_KEYS
    }
    # Store only the canonical JSON-compatible values used by the digest.
    cfg = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    return {
        "data_cfg": cfg,
        "seed": int(seed),
        "bands": sorted(str(band) for band in bands),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "holdout_band": holdout_band,
    }


def _digest_from_inputs(inputs: dict[str, Any]) -> str:
    return prepared_digest(
        inputs["data_cfg"],
        seed=int(inputs["seed"]),
        bands=list(inputs["bands"]),
        manifest_path=inputs.get("manifest_path"),
        holdout_band=inputs.get("holdout_band"),
    )


def prepared_path(prepared_dir: str | os.PathLike, digest: str) -> Path:
    return Path(prepared_dir) / f"prepared_{digest}"


class _SidecarPickler(pickle.Pickler):
    """Externalize large arrays to raw .npy sidecars, by persistent_id."""

    def __init__(self, file, array_dir: Path):
        super().__init__(file, protocol=4)
        self.array_dir = array_dir
        self.counter = 0

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor) and obj.nbytes >= _SIDECAR_MIN_BYTES:
            if obj.device.type != "cpu":
                raise ValueError("Prepared caches must contain CPU tensors only.")
            name = f"{self.counter:06d}.npy"
            self.counter += 1
            np.save(self.array_dir / name, obj.detach().contiguous().numpy())
            return ("torch", name)
        if (
            isinstance(obj, np.ndarray)
            and obj.dtype != object
            and obj.nbytes >= _SIDECAR_MIN_BYTES
        ):
            name = f"{self.counter:06d}.npy"
            self.counter += 1
            np.save(self.array_dir / name, np.ascontiguousarray(obj))
            return ("numpy", name)
        return None


class _SidecarUnpickler(pickle.Unpickler):
    def __init__(self, file, array_dir: Path):
        super().__init__(file)
        self.array_dir = array_dir

    def persistent_load(self, pid):
        kind, name = pid
        array = np.load(self.array_dir / name, mmap_mode="c")
        if kind == "torch":
            return torch.from_numpy(array)
        if kind == "numpy":
            return array
        raise pickle.UnpicklingError(f"Unknown persistent id kind {kind!r}.")


def save_prepared(
    path: str | os.PathLike, scene, store, *, data_cfg: dict | None = None,
    seed: int = 0, bands: list[str] | None = None,
    manifest_path: str | os.PathLike | None = None,
    holdout_band: str | None = None,
    expected_digest: str | None = None,
) -> None:
    digest_inputs = _prepared_digest_inputs(
        data_cfg, seed=seed, bands=list(bands or []),
        manifest_path=manifest_path, holdout_band=holdout_band,
    )
    recomputed_digest = _digest_from_inputs(digest_inputs)
    if expected_digest is not None and expected_digest != recomputed_digest:
        raise RuntimeError(
            "Prepared-cache inputs changed while the scene was being built: "
            f"expected digest {expected_digest}, recomputed {recomputed_digest}. "
            "Refusing to save data under a stale cache key."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / _ARRAY_DIR).mkdir(parents=True)
    buffer = io.BytesIO()
    pickler = _SidecarPickler(buffer, tmp / _ARRAY_DIR)
    pickler.dump({
        "schema": PREPARED_SCHEMA,
        "prepared_digest": recomputed_digest,
        "digest_inputs": digest_inputs,
        "scene": scene,
        "store": store,
    })
    (tmp / _META_NAME).write_bytes(buffer.getvalue())
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)


def load_prepared(
    path: str | os.PathLike, *, data_cfg: dict | None = None,
    seed: int = 0, bands: list[str] | None = None,
    manifest_path: str | os.PathLike | None = None,
    holdout_band: str | None = None,
):
    """Returns (scene, store), or None if the cache is absent or unreadable.

    An unreadable cache (schema drift, torch version change, truncated write)
    must fall back to a rebuild, not kill the run. A transient resource
    failure in the CALLER's environment (fd exhaustion while mmapping the
    ~4k sidecars, out of memory) must raise instead: the cache is healthy,
    and the rebuild path deletes it in place under concurrent readers.
    """
    path = Path(path)
    meta = path / _META_NAME
    if not meta.exists():
        return None
    try:
        with open(meta, "rb") as fh:
            payload = _SidecarUnpickler(fh, path / _ARRAY_DIR).load()
    except OSError as exc:
        if exc.errno in (
            errno.EMFILE, errno.ENFILE, errno.ENOMEM,
            errno.EAGAIN, errno.ESTALE, errno.ETIMEDOUT,
        ):
            raise RuntimeError(
                f"Transient resource failure reading prepared cache at {path} "
                f"({exc!r}); the cache is not at fault. Retry the load (fd "
                "exhaustion needs a higher ulimit -n) instead of rebuilding."
            ) from exc
        print(f"[prepared] cache at {path} unreadable ({exc!r}); rebuilding.", flush=True)
        return None
    except (
        # the failure modes of a stale or truncated cache: pickle stream
        # damage, missing sidecars, and classes renamed since the save
        RuntimeError, EOFError, ValueError, pickle.UnpicklingError,
        ModuleNotFoundError, AttributeError, KeyError,
    ) as exc:
        print(f"[prepared] cache at {path} unreadable ({exc!r}); rebuilding.", flush=True)
        return None
    if payload.get("schema") != PREPARED_SCHEMA:
        print(f"[prepared] cache schema {payload.get('schema')} != {PREPARED_SCHEMA}; rebuilding.", flush=True)
        return None
    if data_cfg is None:
        digest_inputs = payload.get("digest_inputs")
        if not isinstance(digest_inputs, dict):
            raise RuntimeError(
                f"Prepared cache at {path} lacks digest inputs for schema "
                f"{PREPARED_SCHEMA}; refusing an unverifiable load."
            )
    else:
        digest_inputs = _prepared_digest_inputs(
            data_cfg, seed=seed, bands=list(bands or []),
            manifest_path=manifest_path, holdout_band=holdout_band,
        )
    expected_digest = _digest_from_inputs(digest_inputs)
    stored_digest = payload.get("prepared_digest")
    if stored_digest != expected_digest:
        raise RuntimeError(
            f"Prepared cache digest mismatch at {path}: stored "
            f"{stored_digest!r}, recomputed {expected_digest!r} from the current "
            "config and manifest bytes. Refusing to load stale prepared data."
        )
    scene = payload["scene"]
    store = payload["store"]
    store_digest = getattr(store, "psf_mixture_digest", None)
    store_digest_version = getattr(store, "psf_mixture_digest_version", None)
    if isinstance(store_digest, str) and store_digest_version is None:
        bound_manifest_path = digest_inputs.get("manifest_path")
        if bound_manifest_path is not None:
            manifest_payload = json.loads(
                Path(bound_manifest_path).read_text(encoding="utf-8")
            )
            manifest_digest = manifest_payload.get("psf_mixture_digest")
            manifest_digest_version = manifest_payload.get(
                "psf_mixture_digest_version", 1
            )
            if (
                str(manifest_digest).lower() == store_digest.lower()
                and isinstance(manifest_digest_version, int)
                and not isinstance(manifest_digest_version, bool)
                and manifest_digest_version in (1, 2)
            ):
                store.psf_mixture_digest_version = manifest_digest_version
                print(
                    "[prepared] restored PSF mixture digest version "
                    f"{manifest_digest_version} from the content-bound scene "
                    "manifest.",
                    flush=True,
                )
    return scene, store
