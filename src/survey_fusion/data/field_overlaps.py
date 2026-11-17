from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits

FIELD_ROW_SPACING = 1361
OVERLAP_ROWS = 128

_RULE_BRANCHES = (
    "trainable_trainable",
    "trainable_lower_held_upper",
    "held_lower_trainable_upper",
    "anything_else",
)


def overlap_pairs(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the non-empty consecutive-field overlap rectangles."""
    by_scan: dict[tuple[str, int, int], dict[int, list[Mapping[str, Any]]]] = {}
    seen_exposure_ids: set[int] = set()
    for entry in entries:
        exposure_id = int(entry["exposure_id"])
        if exposure_id in seen_exposure_ids:
            raise ValueError(f"Duplicate exposure_id {exposure_id} in overlap geometry.")
        seen_exposure_ids.add(exposure_id)
        height = int(entry["height"])
        width = int(entry["width"])
        if height <= 0 or width <= 0:
            raise ValueError(f"Exposure {exposure_id} has non-positive cutout shape {(height, width)}.")
        key = (str(entry["band"]), int(entry["run"]), int(entry["camcol"]))
        by_scan.setdefault(key, {}).setdefault(int(entry["field"]), []).append(entry)

    pairs: list[dict[str, Any]] = []
    for (band, _run, _camcol), by_field in sorted(by_scan.items()):
        for lower_field in sorted(by_field):
            upper_entries = by_field.get(lower_field + 1, ())
            for lower in sorted(by_field[lower_field], key=lambda item: int(item["exposure_id"])):
                for upper in sorted(upper_entries, key=lambda item: int(item["exposure_id"])):
                    lower_r0 = int(lower["r0"])
                    upper_r0 = int(upper["r0"])
                    row_lo = max(
                        FIELD_ROW_SPACING,
                        lower_r0,
                        FIELD_ROW_SPACING + upper_r0,
                    )
                    row_hi = min(
                        FIELD_ROW_SPACING + OVERLAP_ROWS,
                        lower_r0 + int(lower["height"]),
                        FIELD_ROW_SPACING + upper_r0 + int(upper["height"]),
                    )
                    lower_c0 = int(lower["c0"])
                    upper_c0 = int(upper["c0"])
                    col_lo = max(lower_c0, upper_c0)
                    col_hi = min(
                        lower_c0 + int(lower["width"]),
                        upper_c0 + int(upper["width"]),
                    )
                    if row_hi <= row_lo or col_hi <= col_lo:
                        continue
                    pairs.append(
                        {
                            "lower_exposure_id": int(lower["exposure_id"]),
                            "upper_exposure_id": int(upper["exposure_id"]),
                            "lower_label": str(lower["label"]),
                            "upper_label": str(upper["label"]),
                            "band": band,
                            "lower_y0": row_lo - lower_r0,
                            "lower_x0": col_lo - lower_c0,
                            "upper_y0": row_lo - FIELD_ROW_SPACING - upper_r0,
                            "upper_x0": col_lo - upper_c0,
                            "n_rows": row_hi - row_lo,
                            "n_cols": col_hi - col_lo,
                        }
                    )
    return pairs


def _integer_offset(value: object, *, key: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if not np.isfinite(numeric) or abs(numeric - rounded) > 1e-6:
        raise ValueError(f"{key}={numeric!r} is not an integer to within 1e-6.")
    return int(rounded)


def geometry_from_manifest(
    manifest_path: str | Path,
    observation_root: str | Path,
) -> list[dict[str, Any]]:
    """Read manifest FITS headers and return cutout geometry in frame pixels."""
    manifest_path = Path(manifest_path)
    observation_root = Path(observation_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_exposures = manifest.get("image_exposures")
    if not isinstance(image_exposures, list):
        raise ValueError(f"{manifest_path} does not contain an image_exposures list.")

    geometry: list[dict[str, Any]] = []
    for exposure_id, entry in enumerate(image_exposures):
        if not isinstance(entry, dict):
            raise ValueError(f"image_exposures[{exposure_id}] is not an object.")
        cutout_path = observation_root / str(entry["cutout_path"])
        frame_path = observation_root / str(entry["frame_path"])
        cutout_header = fits.getheader(cutout_path, 0)
        frame_header = fits.getheader(frame_path, 0)
        c0 = _integer_offset(
            frame_header["CRPIX1"] - cutout_header["CRPIX1"],
            key=f"image_exposures[{exposure_id}] c0",
        )
        r0 = _integer_offset(
            frame_header["CRPIX2"] - cutout_header["CRPIX2"],
            key=f"image_exposures[{exposure_id}] r0",
        )
        geometry.append(
            {
                "exposure_id": exposure_id,
                "label": str(entry["label"]),
                "band": str(entry["band"]),
                "run": int(entry["run"]),
                "camcol": int(entry["camcol"]),
                "field": int(entry["field"]),
                "r0": r0,
                "c0": c0,
                "height": int(cutout_header["NAXIS2"]),
                "width": int(cutout_header["NAXIS1"]),
            }
        )
    return geometry


def _rank_correlation(lower: np.ndarray, upper: np.ndarray) -> float | None:
    """Spearman correlation: twins agree in the noise, whatever a few edited pixels do."""
    if lower.size < 2:
        return None
    lower = np.argsort(np.argsort(lower, kind="stable"), kind="stable").astype(np.float64)
    upper = np.argsort(np.argsort(upper, kind="stable"), kind="stable").astype(np.float64)
    lower_centered = lower - np.mean(lower, dtype=np.float64)
    upper_centered = upper - np.mean(upper, dtype=np.float64)
    denominator = float(
        np.sqrt(
            np.sum(lower_centered * lower_centered, dtype=np.float64)
            * np.sum(upper_centered * upper_centered, dtype=np.float64)
        )
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    return float(np.sum(lower_centered * upper_centered, dtype=np.float64) / denominator)


def build_overlap_table(
    manifest_path: str | Path,
    observation_root: str | Path,
    *,
    verify_pixels: bool = True,
) -> dict[str, Any]:
    """Build a versioned overlap table and optionally verify cutout pixels."""
    manifest_path = Path(manifest_path)
    observation_root = Path(observation_root)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    image_exposures = manifest.get("image_exposures")
    if not isinstance(image_exposures, list):
        raise ValueError(f"{manifest_path} does not contain an image_exposures list.")
    pairs = overlap_pairs(geometry_from_manifest(manifest_path, observation_root))
    verification: list[dict[str, Any]] = []
    if verify_pixels:
        for pair in pairs:
            lower_entry = image_exposures[int(pair["lower_exposure_id"])]
            upper_entry = image_exposures[int(pair["upper_exposure_id"])]
            lower_data = np.asarray(
                fits.getdata(observation_root / str(lower_entry["cutout_path"]), 0),
                dtype=np.float64,
            )
            upper_data = np.asarray(
                fits.getdata(observation_root / str(upper_entry["cutout_path"]), 0),
                dtype=np.float64,
            )
            lower_y0 = int(pair["lower_y0"])
            lower_x0 = int(pair["lower_x0"])
            upper_y0 = int(pair["upper_y0"])
            upper_x0 = int(pair["upper_x0"])
            n_rows = int(pair["n_rows"])
            n_cols = int(pair["n_cols"])
            lower_pixels = lower_data[lower_y0 : lower_y0 + n_rows, lower_x0 : lower_x0 + n_cols]
            upper_pixels = upper_data[upper_y0 : upper_y0 + n_rows, upper_x0 : upper_x0 + n_cols]
            expected_shape = (n_rows, n_cols)
            if lower_pixels.shape != expected_shape or upper_pixels.shape != expected_shape:
                raise ValueError(
                    "Overlap rectangle falls outside cutout data for exposure pair "
                    f"{pair['lower_exposure_id']}->{pair['upper_exposure_id']}."
                )
            finite = np.isfinite(lower_pixels) & np.isfinite(upper_pixels)
            correlation = _rank_correlation(lower_pixels[finite], upper_pixels[finite])
            # The same rectangle offset by one row: a misregistered twin
            # correlates there instead (sky noise is independent row to row).
            shifted = [
                _rank_correlation(a[ok], b[ok])
                for a, b in ((lower_pixels[1:], upper_pixels[:-1]), (lower_pixels[:-1], upper_pixels[1:]))
                for ok in [np.isfinite(a) & np.isfinite(b)]
            ]
            shifted = [value for value in shifted if value is not None]
            record = {
                "lower_exposure_id": int(pair["lower_exposure_id"]),
                "upper_exposure_id": int(pair["upper_exposure_id"]),
                "n_finite": int(finite.sum()),
                "rank_correlation": correlation,
                "shifted_rank_correlation": max(shifted) if shifted else None,
            }
            verification.append(record)
            aligned = correlation is not None and correlation >= 0.95 and (
                not shifted or correlation - max(shifted) >= 0.5
            )
            if record["n_finite"] >= 1000 and not aligned:
                raise ValueError(
                    "Overlap pixel verification failed for exposure pair "
                    f"{pair['lower_exposure_id']}->{pair['upper_exposure_id']}: "
                    f"n_finite={record['n_finite']}, rank correlation={correlation}, "
                    f"one-row-shifted={record['shifted_rank_correlation']}."
                )

    return {
        "schema": 1,
        "frame_row_spacing": FIELD_ROW_SPACING,
        "overlap_rows": OVERLAP_ROWS,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pairs": pairs,
        "verification": verification,
    }


def _validated_sorted_unique(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int64) or array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 numpy array.")
    if array.size > 1 and np.any(array[1:] <= array[:-1]):
        raise ValueError(f"{name} must be sorted and unique.")
    return array


def _membership(sorted_values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, candidates)
    present = positions < sorted_values.size
    if np.any(present):
        present[present] = sorted_values[positions[present]] == candidates[present]
    return present


def _local_indices_for_pair(
    pair: Mapping[str, Any],
    lower_width: int,
    upper_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(pair["n_rows"])
    n_cols = int(pair["n_cols"])
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("Overlap pairs must have positive n_rows and n_cols.")
    lower_y0 = int(pair["lower_y0"])
    lower_x0 = int(pair["lower_x0"])
    upper_y0 = int(pair["upper_y0"])
    upper_x0 = int(pair["upper_x0"])
    lower_width = int(lower_width)
    upper_width = int(upper_width)
    if (
        lower_width <= 0
        or upper_width <= 0
        or min(lower_y0, lower_x0, upper_y0, upper_x0) < 0
        or lower_x0 + n_cols > lower_width
        or upper_x0 + n_cols > upper_width
    ):
        raise ValueError("Overlap pair rectangle is inconsistent with exposure widths.")
    row_offsets = np.arange(n_rows, dtype=np.int64)
    column_offsets = np.arange(n_cols, dtype=np.int64)
    lower_local = (
        (lower_y0 + row_offsets[:, None]) * lower_width + lower_x0 + column_offsets[None, :]
    ).reshape(-1)
    upper_local = (
        (upper_y0 + row_offsets[:, None]) * upper_width + upper_x0 + column_offsets[None, :]
    ).reshape(-1)
    return lower_local, upper_local


def resolve_overlap_drops(
    pairs: Sequence[Mapping[str, Any]],
    state_by_exposure: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Apply the training/holdout-safe rule to overlap pixel pairs."""
    drop_parts: dict[int, list[np.ndarray]] = {}
    totals = {branch: 0 for branch in _RULE_BRANCHES}
    per_pair: list[dict[str, Any]] = []
    for pair in pairs:
        lower_id = int(pair["lower_exposure_id"])
        upper_id = int(pair["upper_exposure_id"])
        try:
            lower_state = state_by_exposure[lower_id]
            upper_state = state_by_exposure[upper_id]
        except KeyError as exc:
            raise ValueError(f"Missing overlap state for exposure {exc.args[0]}.") from exc
        lower_valid = _validated_sorted_unique(
            lower_state["valid_local"], name=f"state[{lower_id}].valid_local"
        )
        upper_valid = _validated_sorted_unique(
            upper_state["valid_local"], name=f"state[{upper_id}].valid_local"
        )
        lower_held_local = _validated_sorted_unique(
            lower_state["held_local"], name=f"state[{lower_id}].held_local"
        )
        upper_held_local = _validated_sorted_unique(
            upper_state["held_local"], name=f"state[{upper_id}].held_local"
        )
        lower_local, upper_local = _local_indices_for_pair(
            pair, int(lower_state["width"]), int(upper_state["width"])
        )
        lower_present = _membership(lower_valid, lower_local)
        upper_present = _membership(upper_valid, upper_local)
        lower_held = lower_present & (
            bool(lower_state["exposure_held"]) | _membership(lower_held_local, lower_local)
        )
        upper_held = upper_present & (
            bool(upper_state["exposure_held"]) | _membership(upper_held_local, upper_local)
        )
        lower_trainable = lower_present & ~lower_held
        upper_trainable = upper_present & ~upper_held

        masks = {
            "trainable_trainable": lower_trainable & upper_trainable,
            "trainable_lower_held_upper": lower_trainable & upper_held,
            "held_lower_trainable_upper": lower_held & upper_trainable,
        }
        matched = (
            masks["trainable_trainable"]
            | masks["trainable_lower_held_upper"]
            | masks["held_lower_trainable_upper"]
        )
        masks["anything_else"] = ~matched
        counts = {branch: int(np.count_nonzero(masks[branch])) for branch in _RULE_BRANCHES}
        for branch, count in counts.items():
            totals[branch] += count
        per_pair.append(
            {
                "lower_exposure_id": lower_id,
                "upper_exposure_id": upper_id,
                "pixel_pairs": int(lower_local.size),
                **counts,
            }
        )

        lower_drop = lower_local[masks["trainable_lower_held_upper"]]
        upper_drop = upper_local[masks["trainable_trainable"] | masks["held_lower_trainable_upper"]]
        if lower_drop.size:
            drop_parts.setdefault(lower_id, []).append(lower_drop)
        if upper_drop.size:
            drop_parts.setdefault(upper_id, []).append(upper_drop)

    drops = {
        exposure_id: np.unique(np.concatenate(parts)).astype(np.int64, copy=False)
        for exposure_id, parts in sorted(drop_parts.items())
    }
    return drops, {**totals, "per_pair": per_pair}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SDSS consecutive-field overlap geometry.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observation-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    table = build_overlap_table(args.manifest, args.observation_root)
    output_path = Path(args.out)
    output_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    pixel_pairs = sum(int(pair["n_rows"]) * int(pair["n_cols"]) for pair in table["pairs"])
    print(f"[field-overlaps] pairs={len(table['pairs'])} pixel_pairs={pixel_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
