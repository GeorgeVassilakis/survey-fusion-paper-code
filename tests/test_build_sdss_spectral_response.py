"""Scientific contract and immutable-output checks for the sidecar generator."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_sdss_spectral_response.py"
SPEC = importlib.util.spec_from_file_location("build_sdss_spectral_response", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row():
    return {
        "label": "S4737-55630-0367",
        "plate": 4737,
        "mjd": 55630,
        "fiberid": 367,
        "run2d": "v5_13_2",
        "vers2d": "v5_13_2",
        "plate_header_sha256": "a" * 64,
        "metadata_status": "measured_approximate",
        "spatial_response": {
            "schema_version": 1,
            "model": "gaussian",
            "fwhm_arcsec": 1.47023,
            "provenance": {"source": "survey", "field": "SEEING50", "method": "unweighted exposure mean"},
            "flux_convention": "published_point_source_calibrated",
        },
        "exposure_components": [
            {
                "expid": "b1-00127886-00127885-00127884",
                "source": "https://survey/4737/spCFrame-b1-00127886.fits",
                "sha256": "b" * 64,
                "seeing50": 1.4663,
            }
        ],
    }


def test_preserves_measurements_and_missing_without_mutating_input():
    valid = row()
    missing = row()
    missing.update(
        label="S0481-51908-0640",
        plate=481,
        mjd=51908,
        fiberid=640,
        spatial_response=None,
        metadata_status="missing_or_invalid",
    )
    rows = [valid, missing]
    before = copy.deepcopy(rows)
    result = MODULE.build_response(rows)
    assert rows == before
    expected = copy.deepcopy(before)
    expected[0]["spatial_response"]["calibration"] = MODULE.CALIBRATION
    assert result == expected
    assert result[1]["spatial_response"] is None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1.4"])
def test_rejects_invalid_seeing(value):
    data = row()
    data["spatial_response"]["fwhm_arcsec"] = value
    with pytest.raises(ValueError, match="positive finite"):
        MODULE.build_response([data])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(plate=4738),
        lambda r: r.update(vers2d=""),
        lambda r: r.update(plate_header_sha256="unknown"),
        lambda r: r["spatial_response"]["provenance"].update(method=""),
        lambda r: r["spatial_response"].update(flux_convention="finite_aperture"),
        lambda r: r["exposure_components"][0].update(source="wrong-exposure"),
        lambda r: r["exposure_components"].append(copy.deepcopy(r["exposure_components"][0])),
    ],
)
def test_rejects_untraceable_identity_or_convention(mutation):
    data = row()
    mutation(data)
    with pytest.raises(ValueError):
        MODULE.build_response([data])


def test_rejects_duplicate_labels_and_silent_missing():
    with pytest.raises(ValueError, match="duplicate spectrum label"):
        MODULE.build_response([row(), row()])
    data = row()
    data["spatial_response"] = None
    with pytest.raises(ValueError, match="explicit missing status"):
        MODULE.build_response([data])


def test_hash_receipt_and_output_refusal(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([row()]))
    original = source.read_bytes()
    target = tmp_path / "response.json"
    receipt = MODULE.generate(source, target)
    assert source.read_bytes() == original
    assert receipt["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert (receipt["count"], receipt["response_count"], receipt["missing_count"]) == (1, 1, 0)
    output = target.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.generate(source, target)
    assert target.read_bytes() == output
    with pytest.raises(ValueError, match="immutable"):
        MODULE.generate(source, source)


def test_cli_requires_explicit_approximation_acknowledgement(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([row()]))
    output = tmp_path / "response.json"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "--calibration" in run.stderr
    assert not output.exists()
