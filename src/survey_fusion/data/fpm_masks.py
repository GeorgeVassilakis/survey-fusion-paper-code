from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits

from .field_overlaps import geometry_from_manifest

FPM_ROWS = 1489
FPM_COLS = 2048
# The planes the Tractor's SDSS loader masks.
FPM_MASK_PLANES = ("INTERP", "SATUR", "CR", "GHOST")
SIDECAR_SCHEMA = 1
SIDECAR_KIND = "sdss_fpm_image_pixel_mask"


def fpm_plane_hdus(hdul: fits.HDUList) -> dict[str, int]:
    """Mask plane name -> HDU index, from the S_MASKTYPE table in the last HDU."""
    table = hdul[-1].data
    planes: dict[str, int] = {}
    n_types = None
    for def_name, attribute, value in zip(table["defName"], table["attributeName"], table["Value"]):
        if str(def_name).strip() != "S_MASKTYPE":
            continue
        attribute = str(attribute).strip()
        if attribute == "S_NMASK_TYPES":
            n_types = int(value)
        elif attribute.startswith("S_MASK_"):
            planes[attribute[len("S_MASK_"):]] = int(value)
        else:
            raise ValueError(f"Unexpected S_MASKTYPE attribute {attribute!r}.")
    if n_types is None or sorted(planes.values()) != list(range(n_types)) or len(hdul) != n_types + 2:
        raise ValueError(
            f"fpM mask-type table is inconsistent: S_NMASK_TYPES={n_types}, "
            f"planes={planes}, n_hdu={len(hdul)}."
        )
    return {name: 1 + value for name, value in planes.items()}


def fpm_spans(record: Any) -> np.ndarray:
    """One object's spans as (row, first column, last column), the last column inclusive."""
    raw = np.asarray(record["s"], dtype=np.uint8).tobytes()
    return np.frombuffer(raw, dtype=">i2").reshape(-1, 3).astype(np.int64)


def _plane_mask(table: Any, *, rows: int, cols: int, where: str) -> np.ndarray:
    span_parts = []
    for object_index, record in enumerate(table if table is not None else ()):
        spans = fpm_spans(record)
        if int(record["row0"]) != 0 or int(record["col0"]) != 0:
            raise ValueError(
                f"{where} object {object_index} has a non-zero mask origin "
                f"({int(record['row0'])}, {int(record['col0'])}); only frame-origin masks are supported."
            )
        y, x1, x2 = spans[:, 0], spans[:, 1], spans[:, 2]
        if (
            spans.shape[0] != int(record["nspan"])
            or spans.shape[0] == 0
            or np.any(x2 < x1)
            or int(np.sum(x2 - x1 + 1)) != int(record["npix"])
            or (int(y.min()), int(y.max()), int(x1.min()), int(x2.max()))
            != (int(record["rmin"]), int(record["rmax"]), int(record["cmin"]), int(record["cmax"]))
        ):
            raise ValueError(f"{where} object {object_index} spans disagree with its nspan/npix/bounding box.")
        if int(y.min()) < 0 or int(y.max()) >= rows or int(x1.min()) < 0 or int(x2.max()) >= cols:
            raise ValueError(f"{where} object {object_index} extends outside the {rows}x{cols} frame.")
        span_parts.append(spans)
    edges = np.zeros((rows, cols + 1), dtype=np.int32)
    if span_parts:
        spans = np.concatenate(span_parts)
        np.add.at(edges, (spans[:, 0], spans[:, 1]), 1)
        np.add.at(edges, (spans[:, 0], spans[:, 2] + 1), -1)
    return np.cumsum(edges, axis=1)[:, :cols] > 0


def read_fpm_masks(path: str | Path, planes: Sequence[str] = FPM_MASK_PLANES) -> dict[str, np.ndarray]:
    """Boolean frame-pixel masks [row, column] (1489 x 2048) for each requested fpM plane."""
    with fits.open(path) as hdul:
        header = hdul[0].header
        rows, cols = int(header["MASKROWS"]), int(header["MASKCOLS"])
        if (rows, cols) != (FPM_ROWS, FPM_COLS):
            raise ValueError(f"{path} masks a {rows}x{cols} frame; expected {FPM_ROWS}x{FPM_COLS}.")
        hdus = fpm_plane_hdus(hdul)
        masks = {}
        for plane in planes:
            if plane not in hdus:
                raise ValueError(f"{path} has no {plane!r} mask plane; planes are {sorted(hdus)}.")
            masks[plane] = _plane_mask(hdul[hdus[plane]].data, rows=rows, cols=cols, where=f"{path} {plane}")
    return masks


def cutout_local_indices(frame_mask: np.ndarray, *, r0: int, c0: int, height: int, width: int) -> np.ndarray:
    """Flat cutout indices (row * width + column) of the masked frame pixels inside the cutout.

    Cutout pixel (y, x) is frame pixel (y + r0, x + c0)."""
    rows, cols = frame_mask.shape
    if r0 < 0 or c0 < 0 or r0 + height > rows or c0 + width > cols:
        raise ValueError(f"Cutout rows [{r0}, {r0 + height}) x columns [{c0}, {c0 + width}) leave the frame.")
    window = frame_mask[r0 : r0 + height, c0 : c0 + width]
    return np.flatnonzero(window.reshape(-1)).astype(np.int64)


def store_image_pixels(store: Any) -> dict[int, np.ndarray]:
    """Sorted local pixel indices of every image exposure's datums in a prepared store."""
    from ..pgm.contracts import DatumKind

    exposure = np.asarray(store.batch.exposure_id)
    kind = np.asarray(store.batch.datum_kind)
    local = np.asarray(store.batch.datum_index)
    image_kind = int(DatumKind.IMAGE_PIXEL)
    starts = [0]
    chunk = 1 << 26
    for start in range(1, exposure.size, chunk):
        stop = min(start + chunk, exposure.size)
        changed = (exposure[start:stop] != exposure[start - 1 : stop - 1]) | (
            kind[start:stop] != kind[start - 1 : stop - 1]
        )
        starts.extend((start + np.flatnonzero(changed)).tolist())
    starts.append(exposure.size)
    pixels: dict[int, list[np.ndarray]] = {}
    for start, stop in zip(starts[:-1], starts[1:]):
        if int(kind[start]) == image_kind:
            pixels.setdefault(int(exposure[start]), []).append(np.asarray(local[start:stop], dtype=np.int64))
    result = {}
    for exposure_id, parts in sorted(pixels.items()):
        values = np.sort(np.concatenate(parts))
        if values.size > 1 and np.any(values[1:] == values[:-1]):
            raise ValueError(f"Image exposure {exposure_id} has duplicate pixel datums.")
        result[exposure_id] = values
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sidecar_binding(
    manifest_path: str | Path, *, store_digest: str | None, image_labels: Mapping[int, str],
) -> None:
    """Refuse a sidecar built for another prepared store or another exposure order.

    store_digest is None only for a store built without a cache key, which has
    no name to compare; the exposure labels are compared regardless.
    """
    sidecar = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    store = str(sidecar.get("store") or "")
    if store_digest is not None and store != f"prepared_{store_digest}":
        raise ValueError(
            f"Image pixel mask {manifest_path} was built for store {store or '(unnamed)'!r}; "
            f"this run loads prepared_{store_digest}."
        )
    sidecar_labels = {int(row["exposure_id"]): str(row["label"]) for row in sidecar["exposures"]}
    scene_labels = {int(exp_id): str(label) for exp_id, label in image_labels.items()}
    if sidecar_labels != scene_labels:
        drift = sorted(
            exp_id for exp_id in set(sidecar_labels) | set(scene_labels)
            if sidecar_labels.get(exp_id) != scene_labels.get(exp_id)
        )
        raise ValueError(
            f"Image pixel mask {manifest_path} labels differ from the scene's image exposures "
            f"at ids {drift[:10]}."
        )


def build_fpm_mask_sidecar(
    manifest_path: str | Path,
    observation_root: str | Path,
    fpm_manifest_path: str | Path,
    store_pixels: Mapping[int, np.ndarray],
    store_bands: Mapping[int, str],
    out_dir: str | Path,
    *,
    planes: Sequence[str] = FPM_MASK_PLANES,
    store_label: str = "",
) -> dict[str, Any]:
    """Write per-exposure masked store pixels (npz) and their JSON manifest; return the manifest."""
    manifest_path, fpm_manifest_path, out_dir = Path(manifest_path), Path(fpm_manifest_path), Path(out_dir)
    fpm_manifest = json.loads(fpm_manifest_path.read_text(encoding="utf-8"))
    fpm_by_key = {
        (int(row["run"]), int(row["camcol"]), int(row["field"]), str(row["band"])): row
        for row in fpm_manifest["files"]
    }
    geometry = geometry_from_manifest(manifest_path, observation_root)
    if {int(entry["exposure_id"]) for entry in geometry} != {int(e) for e in store_pixels}:
        raise ValueError("Manifest image exposures and store image exposures differ.")
    arrays: dict[str, np.ndarray] = {}
    exposures = []
    for entry in geometry:
        exposure_id = int(entry["exposure_id"])
        if str(store_bands[exposure_id]) != str(entry["band"]):
            raise RuntimeError(
                f"Manifest/store exposure order drift: exposure {exposure_id} is band "
                f"{entry['band']!r} in the manifest and {store_bands[exposure_id]!r} in the store."
            )
        fpm = fpm_by_key[(int(entry["run"]), int(entry["camcol"]), int(entry["field"]), str(entry["band"]))]
        fpm_path = fpm_manifest_path.parent / str(fpm["file"])
        if _sha256(fpm_path) != str(fpm["sha256"]):
            raise ValueError(f"{fpm_path} does not match its manifest sha256.")
        masks = read_fpm_masks(fpm_path, planes)
        geometry_kwargs = {key: int(entry[key]) for key in ("r0", "c0", "height", "width")}
        valid = np.asarray(store_pixels[exposure_id], dtype=np.int64)
        if valid.size and (int(valid[0]) < 0 or int(valid[-1]) >= entry["height"] * entry["width"]):
            raise ValueError(f"Store pixels of exposure {exposure_id} fall outside its cutout.")
        by_plane = {plane: cutout_local_indices(masks[plane], **geometry_kwargs) for plane in planes}
        union = np.unique(np.concatenate(list(by_plane.values())))
        masked = union[np.isin(union, valid, assume_unique=True)]
        arrays[f"e{exposure_id}"] = masked
        exposures.append(
            {
                **{key: entry[key] for key in ("exposure_id", "label", "band", "run", "camcol", "field")},
                **geometry_kwargs,
                "fpm_file": str(fpm["file"]),
                "fpm_sha256": str(fpm["sha256"]),
                "n_store_pixels": int(valid.size),
                "n_masked": int(masked.size),
                "n_cutout_masked_by_plane": {plane: int(values.size) for plane, values in by_plane.items()},
                "n_masked_by_plane": {
                    plane: int(np.count_nonzero(np.isin(values, valid, assume_unique=True)))
                    for plane, values in by_plane.items()
                },
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "fpm_mask.npz"
    np.savez(arrays_path, **arrays)
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "kind": SIDECAR_KIND,
        "planes": list(planes),
        "arrays": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "image_manifest_sha256": _sha256(manifest_path),
        "fpm_manifest_sha256": _sha256(fpm_manifest_path),
        "store": store_label,
        "n_store_pixels": int(sum(row["n_store_pixels"] for row in exposures)),
        "n_masked": int(sum(row["n_masked"] for row in exposures)),
        "exposures": exposures,
    }
    (out_dir / "fpm_mask.json").write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    return sidecar


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SDSS fpM image-pixel mask sidecar for a prepared store.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--fpm-manifest", required=True)
    parser.add_argument("--prepared", required=True, help="prepared_<digest> directory")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    import resource

    from .prepared import load_prepared

    # The prepared store mmaps ~4k array files.
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    loaded = load_prepared(args.prepared)
    if loaded is None:
        raise RuntimeError(f"No readable prepared store at {args.prepared}.")
    _, store = loaded
    sidecar = build_fpm_mask_sidecar(
        args.manifest, args.observation_root, args.fpm_manifest, store_image_pixels(store),
        {int(e): str(b) for e, b in store.image_band_by_exposure.items()}, args.out_dir,
        store_label=str(Path(args.prepared).name),
    )
    print(
        f"[fpm-mask] exposures={len(sidecar['exposures'])} masked={sidecar['n_masked']} "
        f"of {sidecar['n_store_pixels']} store pixels arrays_sha256={sidecar['arrays_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
