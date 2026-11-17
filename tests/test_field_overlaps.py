from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from astropy.io import fits
from support.synthetic_trainer import (
    ConstantKernelFactory,
    OnesModel,
    SyntheticStore,
    image_exposure,
    make_batch,
)

import survey_fusion.data.field_overlaps as field_overlaps_module
from survey_fusion.data.field_overlaps import (
    FIELD_ROW_SPACING,
    OVERLAP_ROWS,
    build_overlap_table,
    geometry_from_manifest,
    overlap_pairs,
    resolve_overlap_drops,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer


def _write_fits(path, data, *, frame_crpix, r0=0, c0=0) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header["CRPIX1"] = float(frame_crpix[0] - c0)
    hdu.header["CRPIX2"] = float(frame_crpix[1] - r0)
    hdu.writeto(path)


def _synthetic_manifest(tmp_path):
    frame_crpix = (100.0, 200.0)
    lower = np.arange(30 * 8, dtype=np.float32).reshape(30, 8)
    upper = np.full((20, 7), -1.0, dtype=np.float32)
    upper[:14, :6] = lower[16:30, 2:8] * 1.01 + 7.0
    distant = np.zeros((10, 5), dtype=np.float32)
    entries = [
        {
            "label": "r-lower",
            "band": "r",
            "run": 12,
            "camcol": 3,
            "field": 40,
            "cutout_path": "lower.fits",
            "frame_path": "lower-frame.fits",
        },
        {
            "label": "r-upper",
            "band": "r",
            "run": 12,
            "camcol": 3,
            "field": 41,
            "cutout_path": "upper.fits",
            "frame_path": "upper-frame.fits",
        },
        {
            "label": "r-distant",
            "band": "r",
            "run": 12,
            "camcol": 3,
            "field": 43,
            "cutout_path": "distant.fits",
            "frame_path": "distant-frame.fits",
        },
    ]
    for entry in entries:
        _write_fits(
            tmp_path / entry["frame_path"],
            np.zeros((1, 1), dtype=np.float32),
            frame_crpix=frame_crpix,
        )
    _write_fits(tmp_path / "lower.fits", lower, frame_crpix=frame_crpix, r0=1350, c0=10)
    _write_fits(tmp_path / "upper.fits", upper, frame_crpix=frame_crpix, r0=5, c0=12)
    _write_fits(tmp_path / "distant.fits", distant, frame_crpix=frame_crpix, r0=0, c0=0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"image_exposures": entries}), encoding="utf-8")
    return manifest_path


def test_geometry_from_manifest_and_clipped_overlap(tmp_path) -> None:
    manifest_path = _synthetic_manifest(tmp_path)
    geometry = geometry_from_manifest(manifest_path, tmp_path)
    assert geometry[0]["r0"] == 1350
    assert geometry[0]["c0"] == 10
    assert geometry[1]["r0"] == 5
    assert [(entry["field"], entry["height"], entry["width"]) for entry in geometry] == [
        (40, 30, 8),
        (41, 20, 7),
        (43, 10, 5),
    ]

    pairs = overlap_pairs(geometry)
    assert pairs == [
        {
            "lower_exposure_id": 0,
            "upper_exposure_id": 1,
            "lower_label": "r-lower",
            "upper_label": "r-upper",
            "band": "r",
            "lower_y0": 16,
            "lower_x0": 2,
            "upper_y0": 0,
            "upper_x0": 0,
            "n_rows": 14,
            "n_cols": 6,
        }
    ]


def test_geometry_rejects_non_integer_offset(tmp_path) -> None:
    manifest_path = _synthetic_manifest(tmp_path)
    with fits.open(tmp_path / "upper.fits", mode="update") as hdus:
        hdus[0].header["CRPIX2"] += 0.01
    with pytest.raises(ValueError, match="not an integer"):
        geometry_from_manifest(manifest_path, tmp_path)


def _state(valid, held=(), *, width=8, exposure_held=False):
    return {
        "width": width,
        "valid_local": np.asarray(valid, dtype=np.int64),
        "held_local": np.asarray(held, dtype=np.int64),
        "exposure_held": exposure_held,
    }


def test_resolve_overlap_drops_rule_truth_table() -> None:
    pair = {
        "lower_exposure_id": 10,
        "upper_exposure_id": 11,
        "lower_y0": 0,
        "lower_x0": 0,
        "upper_y0": 0,
        "upper_x0": 0,
        "n_rows": 1,
        "n_cols": 8,
    }
    states = {
        10: _state([0, 1, 2, 3, 4, 7], held=[2, 3, 7]),
        11: _state([0, 1, 2, 3, 5], held=[1, 3]),
    }
    drops, summary = resolve_overlap_drops([pair], states)
    assert np.array_equal(drops[10], np.array([1], dtype=np.int64))
    assert np.array_equal(drops[11], np.array([0, 2], dtype=np.int64))
    assert {key: summary[key] for key in summary if key != "per_pair"} == {
        "trainable_trainable": 1,
        "trainable_lower_held_upper": 1,
        "held_lower_trainable_upper": 1,
        "anything_else": 5,
    }
    assert summary["per_pair"][0]["pixel_pairs"] == 8

    whole_exposure_states = {
        10: _state(range(8), exposure_held=True),
        11: _state(range(8)),
    }
    drops, summary = resolve_overlap_drops([pair], whole_exposure_states)
    assert set(drops) == {11}
    assert np.array_equal(drops[11], np.arange(8, dtype=np.int64))
    assert summary["held_lower_trainable_upper"] == 8


def test_cli_round_trip(tmp_path) -> None:
    manifest_path = _synthetic_manifest(tmp_path)
    output_path = tmp_path / "overlaps.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "survey_fusion.data.field_overlaps",
            "--manifest",
            str(manifest_path),
            "--observation-root",
            str(tmp_path),
            "--out",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["schema"] == 1
    assert output["frame_row_spacing"] == FIELD_ROW_SPACING
    assert output["overlap_rows"] == OVERLAP_ROWS
    assert len(output["pairs"]) == 1
    assert output["verification"][0]["n_finite"] == 84
    assert output["verification"][0]["rank_correlation"] == pytest.approx(1.0)
    assert "pairs=1 pixel_pairs=84" in completed.stdout
    assert output == build_overlap_table(manifest_path, tmp_path)


def test_pixel_verification_rejects_low_correlation(tmp_path) -> None:
    frame_crpix = (100.0, 200.0)
    entries = []
    rng = np.random.default_rng(7)
    for exposure_id, (field, r0) in enumerate(((10, FIELD_ROW_SPACING), (11, 0))):
        cutout_path = f"cutout-{exposure_id}.fits"
        frame_path = f"frame-{exposure_id}.fits"
        entries.append(
            {
                "label": f"r-{field}",
                "band": "r",
                "run": 1,
                "camcol": 1,
                "field": field,
                "cutout_path": cutout_path,
                "frame_path": frame_path,
            }
        )
        _write_fits(
            tmp_path / cutout_path,
            rng.normal(size=(20, 60)),
            frame_crpix=frame_crpix,
            r0=r0,
        )
        _write_fits(
            tmp_path / frame_path,
            np.zeros((1, 1)),
            frame_crpix=frame_crpix,
        )
    manifest_path = tmp_path / "bad-manifest.json"
    manifest_path.write_text(json.dumps({"image_exposures": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="pixel verification failed"):
        build_overlap_table(manifest_path, tmp_path)


TRAINER_CFG = {
    "kernel_samples_k": 8,
    "replica_samples_k": 1,
    "epochs": 0,
    "learning_rate": 1e-3,
    "apply_spatial_psf": False,
}


def _trainer_store() -> SyntheticStore:
    batch = make_batch(
        y=torch.zeros(8, dtype=torch.float32),
        exposure_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        datum_kind=torch.zeros(8, dtype=torch.long),
        datum_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    return SyntheticStore(
        exposures={
            0: image_exposure(band="r", height=1, width=4),
            1: image_exposure(band="r", height=1, width=4),
        },
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=8,
        n_spec=0,
        image_excluded_local_indices_by_exposure={
            0: np.array([2], dtype=np.int64),
            1: np.array([1], dtype=np.int64),
        },
        image_random_holdout_indices_by_exposure={
            1: np.array([1], dtype=np.int64),
        },
    )


def _overlap_table(tmp_path):
    path = tmp_path / "trainer-overlaps.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "frame_row_spacing": FIELD_ROW_SPACING,
                "overlap_rows": OVERLAP_ROWS,
                "manifest_sha256": "a" * 64,
                "pairs": [
                    {
                        "lower_exposure_id": 0,
                        "upper_exposure_id": 1,
                        "lower_label": "r-lower",
                        "upper_label": "r-upper",
                        "band": "r",
                        "lower_y0": 0,
                        "lower_x0": 0,
                        "upper_y0": 0,
                        "upper_x0": 0,
                        "n_rows": 1,
                        "n_cols": 4,
                    }
                ],
                "verification": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _make_trainer(store, trainer_cfg) -> DatumTrainer:
    device = torch.device("cpu")
    return DatumTrainer(
        model=OnesModel(),
        grid=build_spatial_grid(2, 4, device),
        data=store,
        training_bands=["r"],
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg=trainer_cfg,
        device=device,
    )


def _training_local_by_exposure(trainer) -> dict[int, list[int]]:
    result = {}
    for exposure_id, global_indices in trainer.image_train_indices_by_exposure.items():
        result[exposure_id] = trainer._gather_named_rows(
            "datum_index", global_indices, device=torch.device("cpu")
        ).tolist()
    return result


def test_trainer_field_overlap_integration_is_opt_in_and_preserves_holdout(
    tmp_path,
) -> None:
    baseline = _make_trainer(_trainer_store(), dict(TRAINER_CFG))
    assert _training_local_by_exposure(baseline) == {0: [0, 1, 3], 1: [0, 2, 3]}
    assert not hasattr(baseline, "field_overlap_summary")
    explicit_none = _make_trainer(_trainer_store(), dict(TRAINER_CFG, field_overlap_table=None))
    assert torch.equal(baseline.image_train_indices, explicit_none.image_train_indices)
    assert not hasattr(explicit_none, "field_overlap_summary")

    store = _trainer_store()
    random_holdout_before = {
        exposure_id: values.copy()
        for exposure_id, values in store.image_random_holdout_indices_by_exposure.items()
    }
    trainer = _make_trainer(
        store,
        dict(TRAINER_CFG, field_overlap_table=str(_overlap_table(tmp_path))),
    )
    assert _training_local_by_exposure(trainer) == {0: [0, 3]}
    assert np.array_equal(store.image_random_holdout_indices_by_exposure[1], random_holdout_before[1])
    assert torch.equal(trainer.image_random_holdout_indices_by_exposure[1], torch.tensor([1]))
    assert torch.equal(trainer.image_excluded_local_indices_by_exposure[0], torch.tensor([1, 2]))
    assert torch.equal(
        trainer.image_excluded_local_indices_by_exposure[1],
        torch.tensor([0, 1, 2, 3]),
    )
    assert trainer.field_overlap_summary["pixel_pairs_per_rule_branch"] == {
        "trainable_trainable": 2,
        "trainable_lower_held_upper": 1,
        "held_lower_trainable_upper": 1,
        "anything_else": 0,
    }
    assert trainer.field_overlap_summary["dropped_pixels_by_exposure"] == {
        0: 1,
        1: 3,
    }


def test_trainer_field_overlap_postcondition_catches_corrupt_drops(
    tmp_path,
    monkeypatch,
) -> None:
    real_resolver = field_overlaps_module.resolve_overlap_drops
    call_count = 0

    def corrupt_resolver(pairs, states):
        nonlocal call_count
        call_count += 1
        drops, summary = real_resolver(pairs, states)
        return ({}, summary) if call_count == 1 else (drops, summary)

    monkeypatch.setattr(field_overlaps_module, "resolve_overlap_drops", corrupt_resolver)
    with pytest.raises(RuntimeError, match="post-condition failed"):
        _make_trainer(
            _trainer_store(),
            dict(TRAINER_CFG, field_overlap_table=str(_overlap_table(tmp_path))),
        )
