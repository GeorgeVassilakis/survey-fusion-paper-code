#!/usr/bin/env python3
"""Declare a conditional Gaussian calibration response from immutable survey metadata.

This is a centered nominal Gaussian standard with the same spectroscopic seeing,
not an exact inverse of the SDSS calibration pipeline. Missing seeing remains
missing: strict metadata training fails unless exclusion is explicitly requested.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

CALIBRATION = {
    "mode": "gaussian_point_source",
    "approximation": "centered_gaussian_standard_same_seeing",
    "provenance": {
        "source": "https://www.sdss3.org/dr9/algorithms/spectrophotometry.php",
        "field": "point-source flux convention",
        "method": "declared centered nominal Gaussian standard with the same spectroscopic seeing; not exact pipeline inverse",
    },
}


def _provenance(value: object, context: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key].strip() for key in ("source", "field", "method")
    ):
        raise ValueError(f"{context}: source, field, method provenance required")


def _hash(value: object, context: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context}: SHA256 required")


def build_response(rows: object) -> list[dict]:
    """Validate identities and declare calibration without changing measured seeing."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a nonempty metadata list")
    result = copy.deepcopy(rows)
    seen = set()
    for row in result:
        if not isinstance(row, dict):
            raise TypeError("Each metadata row must be an object")
        label = row.get("label")
        match = re.fullmatch(r"S(\d{4})-(\d{5})-(\d{4})", label if isinstance(label, str) else "")
        if match is None or label in seen:
            raise ValueError(f"Invalid or duplicate spectrum label: {label!r}")
        seen.add(label)
        for field, expected in zip(("plate", "mjd", "fiberid"), map(int, match.groups())):
            if type(row.get(field)) is not int or row[field] != expected:
                raise ValueError(f"{label}: inconsistent {field} identity")
        for field in ("run2d", "vers2d"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{label}: {field} processing identity required")
        _hash(row.get("plate_header_sha256"), label)
        components = row.get("exposure_components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"{label}: exposure components required")
        exposures = set()
        for component in components:
            if not isinstance(component, dict):
                raise TypeError(f"{label}: invalid exposure component")
            eid = component.get("expid")
            if (
                not isinstance(eid, str)
                or not re.fullmatch(r"[br][12]-\d{8}-\d{8}-\d{8}", eid)
                or eid in exposures
            ):
                raise ValueError(f"{label}: invalid or duplicate exposure identity")
            exposures.add(eid)
            source = component.get("source")
            frame = "spCFrame-" + "-".join(eid.split("-")[:2]) + ".fits"
            if not isinstance(source, str) or not source.endswith("/" + frame):
                raise ValueError(f"{label}: exposure source does not match identity")
            _hash(component.get("sha256"), f"{label}/{eid}")
        response = row.get("spatial_response")
        if response is None:
            if row.get("metadata_status") != "missing_or_invalid":
                raise ValueError(f"{label}: missing response requires explicit missing status")
            continue
        if (
            not isinstance(response, dict)
            or response.get("schema_version") != 1
            or response.get("model") != "gaussian"
        ):
            raise ValueError(f"{label}: schema_version=1, model=gaussian required")
        fwhm = response.get("fwhm_arcsec")
        if (
            isinstance(fwhm, bool)
            or not isinstance(fwhm, (int, float))
            or not math.isfinite(fwhm)
            or fwhm <= 0
        ):
            raise ValueError(f"{label}: positive finite fwhm_arcsec required")
        _provenance(response.get("provenance"), label)
        if response.get("flux_convention") != "published_point_source_calibrated":
            raise ValueError(f"{label}: published_point_source_calibrated flux convention required")
        response["calibration"] = copy.deepcopy(CALIBRATION)
    return result


def generate(input_path: Path, output_path: Path) -> dict:
    """Write new sidecar and hash receipt; refuse existing output or receipt."""
    receipt_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Input metadata is immutable; choose a different output")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Output or provenance receipt already exists")
    original = input_path.read_bytes()
    rows = build_response(json.loads(original))
    encoded = (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode()
    missing = [row["label"] for row in rows if row.get("spatial_response") is None]
    receipt = {
        "schema_version": 1,
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(original).hexdigest(),
        "output_path": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "count": len(rows),
        "response_count": len(rows) - len(missing),
        "missing_count": len(missing),
        "missing_labels": missing,
        "calibration": CALIBRATION,
        "missing_policy": "preserved; strict metadata training fails unless explicit exclude is chosen",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(encoded)
    try:
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        output_path.unlink()
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        required=True,
        choices=["gaussian-point-source"],
        help="Explicitly acknowledge the conditional Gaussian approximation; not an exact survey response",
    )
    args = parser.parse_args()
    try:
        receipt = generate(args.input, args.output)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
