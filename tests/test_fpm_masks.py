from __future__ import annotations

import hashlib
import json

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
    spec_exposure,
)

from survey_fusion.data.field_overlaps import FIELD_ROW_SPACING, OVERLAP_ROWS
from survey_fusion.data.fpm_masks import (
    FPM_MASK_PLANES,
    SIDECAR_KIND,
    build_fpm_mask_sidecar,
    cutout_local_indices,
    fpm_plane_hdus,
    read_fpm_masks,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.contracts import DatumKind
from survey_fusion.training.datum_trainer import DatumTrainer

PLANE_NAMES = (
    "INTERP", "SATUR", "NOTCHECKED", "OBJECT", "BRIGHTOBJECT",
    "BINOBJECT", "CATOBJECT", "SUBTRACTED", "GHOST", "CR",
)
IMAGE, SPECTRUM = int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)


def _objmask_hdu(objects) -> fits.BinTableHDU:
    records = []
    for obj in objects:
        spans = np.asarray(obj["spans"], dtype=np.int64).reshape(-1, 3)
        record = {
            "nspan": spans.shape[0],
            "row0": 0,
            "col0": 0,
            "rmin": int(spans[:, 0].min()),
            "rmax": int(spans[:, 0].max()),
            "cmin": int(spans[:, 1].min()),
            "cmax": int(spans[:, 2].max()),
            "npix": int(np.sum(spans[:, 2] - spans[:, 1] + 1)),
        }
        record.update(obj.get("override", {}))
        record["s"] = np.frombuffer(spans.astype(">i2").tobytes(), dtype=np.uint8)
        records.append(record)
    names = ("nspan", "row0", "col0", "rmin", "rmax", "cmin", "cmax", "npix")
    columns = [fits.Column(name="refcntr", format="1J", array=np.ones(len(records), dtype=np.int32))]
    columns += [
        fits.Column(name=name, format="1J", array=np.array([r[name] for r in records], dtype=np.int32))
        for name in names
    ]
    spans_column = np.empty(len(records), dtype=object)
    for index, record in enumerate(records):
        spans_column[index] = record["s"]
    columns.append(fits.Column(name="s", format="PB()", array=spans_column))
    return fits.BinTableHDU.from_columns(columns)


def _write_fpm(path, objects_by_plane) -> None:
    primary = fits.PrimaryHDU()
    primary.header["MASKROWS"] = 1489
    primary.header["MASKCOLS"] = 2048
    primary.header["NPLANE"] = len(PLANE_NAMES)
    types = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="defName", format="31A", array=["S_MASKTYPE"] * (len(PLANE_NAMES) + 1)),
            fits.Column(
                name="attributeName",
                format="31A",
                array=[f"S_MASK_{name}" for name in PLANE_NAMES] + ["S_NMASK_TYPES"],
            ),
            fits.Column(name="Value", format="1J", array=np.arange(len(PLANE_NAMES) + 1, dtype=np.int32)),
        ]
    )
    hdus = [primary] + [_objmask_hdu(objects_by_plane.get(name, [])) for name in PLANE_NAMES] + [types]
    fits.HDUList(hdus).writeto(path)


def _pixels(mask) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pixel) for pixel in np.argwhere(mask)]


def test_read_fpm_masks_decodes_inclusive_spans_in_frame_coordinates(tmp_path) -> None:
    path = tmp_path / "fpM-000012-r3-0040.fit.gz"
    _write_fpm(
        path,
        {
            "SATUR": [{"spans": [(10, 5, 7), (11, 6, 6)]}, {"spans": [(10, 7, 8)]}],
            "CR": [{"spans": [(0, 0, 0)]}, {"spans": [(1488, 2040, 2047)]}],
            "INTERP": [{"spans": [(row, 3, 3) for row in range(1489)]}],
            "NOTCHECKED": [{"spans": [(1489, 1, 2), (-1, 4, 4)]}],
        },
    )
    with fits.open(path) as hdul:
        assert {plane: fpm_plane_hdus(hdul)[plane] for plane in FPM_MASK_PLANES} == {
            "INTERP": 1, "SATUR": 2, "CR": 10, "GHOST": 9,
        }
    masks = read_fpm_masks(path)
    assert set(masks) == set(FPM_MASK_PLANES)
    assert all(mask.shape == (1489, 2048) and mask.dtype == bool for mask in masks.values())
    assert _pixels(masks["SATUR"]) == [(10, 5), (10, 6), (10, 7), (10, 8), (11, 6)]
    assert _pixels(masks["CR"]) == [(0, 0)] + [(1488, col) for col in range(2040, 2048)]
    assert _pixels(masks["INTERP"]) == [(row, 3) for row in range(1489)]
    assert not masks["GHOST"].any()


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"row0": 1}, "non-zero mask origin"),
        ({"col0": -2}, "non-zero mask origin"),
        ({"npix": 5}, "disagree"),
        ({"rmax": 12}, "disagree"),
        ({"cmin": 4}, "disagree"),
        ({"nspan": 1}, "disagree"),
    ],
)
def test_read_fpm_masks_rejects_inconsistent_objects(tmp_path, override, match) -> None:
    path = tmp_path / "bad.fit"
    _write_fpm(path, {"SATUR": [{"spans": [(10, 5, 7), (11, 6, 6)], "override": override}]})
    with pytest.raises(ValueError, match=match):
        read_fpm_masks(path)


def test_read_fpm_masks_rejects_out_of_frame_spans_and_unknown_planes(tmp_path) -> None:
    path = tmp_path / "edge.fit"
    _write_fpm(path, {"CR": [{"spans": [(5, 2046, 2048)]}]})
    with pytest.raises(ValueError, match="outside"):
        read_fpm_masks(path)
    assert not read_fpm_masks(path, ("SATUR",))["SATUR"].any()
    with pytest.raises(ValueError, match="no 'BOGUS' mask plane"):
        read_fpm_masks(path, ("BOGUS",))


def test_cutout_local_indices_offset_the_frame_by_the_cutout_origin() -> None:
    frame = np.zeros((1489, 2048), dtype=bool)
    for row, col in [(3, 10), (5, 14), (3, 9), (5, 15), (1, 12), (6, 12)]:
        frame[row, col] = True
    local = cutout_local_indices(frame, r0=2, c0=10, height=4, width=5)
    assert local.dtype == np.int64
    assert local.tolist() == [1 * 5 + 0, 3 * 5 + 4]
    with pytest.raises(ValueError, match="leave the frame"):
        cutout_local_indices(frame, r0=1486, c0=0, height=4, width=5)
    with pytest.raises(ValueError, match="leave the frame"):
        cutout_local_indices(frame, r0=0, c0=2045, height=4, width=5)


def _write_cutout(path, shape, *, r0, c0, frame_crpix=(100.0, 200.0)) -> None:
    hdu = fits.PrimaryHDU(np.zeros(shape, dtype=np.float32))
    hdu.header["CRPIX1"] = float(frame_crpix[0] - c0)
    hdu.header["CRPIX2"] = float(frame_crpix[1] - r0)
    hdu.writeto(path)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_inputs(tmp_path):
    entries = [
        {"label": "r-a", "band": "r", "run": 12, "camcol": 3, "field": 40,
         "cutout_path": "a.fits", "frame_path": "a-frame.fits"},
        {"label": "g-b", "band": "g", "run": 12, "camcol": 3, "field": 41,
         "cutout_path": "b.fits", "frame_path": "b-frame.fits"},
    ]
    for entry in entries:
        _write_cutout(tmp_path / entry["frame_path"], (1, 1), r0=0, c0=0)
    _write_cutout(tmp_path / "a.fits", (3, 4), r0=1486, c0=2044)
    _write_cutout(tmp_path / "b.fits", (2, 3), r0=0, c0=0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"image_exposures": entries}), encoding="utf-8")
    _write_fpm(
        tmp_path / "fpM-a.fit.gz",
        {
            "SATUR": [{"spans": [(1487, 2045, 2047)]}],
            "CR": [{"spans": [(1488, 2044, 2044)]}],
            "INTERP": [{"spans": [(0, 0, 0)]}],
            "NOTCHECKED": [{"spans": [(1489, 2000, 2001)]}],
        },
    )
    _write_fpm(
        tmp_path / "fpM-b.fit.gz",
        {"INTERP": [{"spans": [(0, 0, 0)]}], "GHOST": [{"spans": [(1, 2, 2)]}]},
    )
    fpm_manifest = tmp_path / "fpm_manifest.json"
    fpm_manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"run": 12, "camcol": 3, "field": 40, "band": "r", "file": "fpM-a.fit.gz",
                     "sha256": _sha256(tmp_path / "fpM-a.fit.gz")},
                    {"run": 12, "camcol": 3, "field": 41, "band": "g", "file": "fpM-b.fit.gz",
                     "sha256": _sha256(tmp_path / "fpM-b.fit.gz")},
                ]
            }
        ),
        encoding="utf-8",
    )
    store_pixels = {0: np.array([p for p in range(12) if p != 7], dtype=np.int64), 1: np.arange(6, dtype=np.int64)}
    return manifest_path, fpm_manifest, store_pixels


def _store_from_pixels(store_pixels, shapes, bands, **maps) -> SyntheticStore:
    exposure_ids = [e for e in sorted(store_pixels) for _ in store_pixels[e]]
    local = [int(p) for e in sorted(store_pixels) for p in store_pixels[e]]
    batch = make_batch(
        y=torch.zeros(len(local), dtype=torch.float32),
        exposure_id=torch.tensor(exposure_ids),
        datum_kind=torch.full((len(local),), IMAGE, dtype=torch.long),
        datum_index=torch.tensor(local),
    )
    return SyntheticStore(
        exposures={e: image_exposure(band=bands[e], height=shapes[e][0], width=shapes[e][1]) for e in store_pixels},
        batch=batch,
        image_band_by_exposure=dict(bands),
        kernel_factory=ConstantKernelFactory(batch),
        n_image=len(local),
        n_spec=0,
        **maps,
    )


TRAINER_CFG = {
    "kernel_samples_k": 1,
    "replica_samples_k": 1,
    "epochs": 0,
    "learning_rate": 1e-3,
    "apply_spatial_psf": False,
}


def _make_trainer(store, trainer_cfg, *, bands=("r",), exclude=None) -> DatumTrainer:
    device = torch.device("cpu")
    return DatumTrainer(
        model=OnesModel(),
        grid=build_spatial_grid(2, 4, device),
        data=store,
        training_bands=list(bands),
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg=trainer_cfg,
        device=device,
        exclude_image_exposure_ids=exclude,
    )


def _training_local(trainer) -> dict[int, list[int]]:
    return {
        exposure_id: trainer._gather_named_rows("datum_index", rows, device=torch.device("cpu")).tolist()
        for exposure_id, rows in sorted(trainer.image_train_indices_by_exposure.items())
    }


def test_build_sidecar_keeps_masked_store_pixels_and_trains_without_them(tmp_path) -> None:
    manifest_path, fpm_manifest, store_pixels = _builder_inputs(tmp_path)
    sidecar = build_fpm_mask_sidecar(
        manifest_path, tmp_path, fpm_manifest, store_pixels, {0: "r", 1: "g"}, tmp_path / "out",
        store_label="synthetic",
    )
    with np.load(tmp_path / "out" / "fpm_mask.npz") as arrays:
        assert arrays["e0"].tolist() == [5, 6, 8]
        assert arrays["e1"].tolist() == [0, 5]
        assert arrays["e0"].dtype == np.int64
    on_disk = json.loads((tmp_path / "out" / "fpm_mask.json").read_text())
    assert on_disk == sidecar
    assert sidecar["schema"] == 1 and sidecar["kind"] == SIDECAR_KIND
    assert sidecar["planes"] == list(FPM_MASK_PLANES)
    assert sidecar["arrays_sha256"] == _sha256(tmp_path / "out" / "fpm_mask.npz")
    a, b = sidecar["exposures"]
    assert (a["r0"], a["c0"], a["height"], a["width"]) == (1486, 2044, 3, 4)
    assert a["n_cutout_masked_by_plane"] == {"INTERP": 0, "SATUR": 3, "CR": 1, "GHOST": 0}
    assert a["n_masked_by_plane"] == {"INTERP": 0, "SATUR": 2, "CR": 1, "GHOST": 0}
    assert (a["n_store_pixels"], a["n_masked"], b["n_store_pixels"], b["n_masked"]) == (11, 3, 6, 2)
    assert (sidecar["n_store_pixels"], sidecar["n_masked"]) == (17, 5)

    store = _store_from_pixels(store_pixels, {0: (3, 4), 1: (2, 3)}, {0: "r", 1: "g"})
    trainer = _make_trainer(
        store, dict(TRAINER_CFG, image_pixel_mask=str(tmp_path / "out" / "fpm_mask.json")), bands=("r", "g"),
    )
    assert _training_local(trainer) == {0: [0, 1, 2, 3, 4, 9, 10, 11], 1: [1, 2, 3, 4]}
    assert trainer.image_pixel_mask_summary["removed_rows"] == 5
    assert trainer.image_pixel_mask_summary["removed_rows_by_exposure"] == {0: 3, 1: 2}
    assert trainer.image_pixel_mask_summary["masked_pixels"] == 5
    assert trainer.image_pixel_mask_summary["planes"] == list(FPM_MASK_PLANES)


def test_build_sidecar_refuses_store_band_drift(tmp_path) -> None:
    manifest_path, fpm_manifest, store_pixels = _builder_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="order drift"):
        build_fpm_mask_sidecar(manifest_path, tmp_path, fpm_manifest, store_pixels, {0: "g", 1: "r"}, tmp_path / "out")


def _write_sidecar(tmp_path, store, masked_by_exposure, *, edit_rows=None, **edits):
    arrays = {
        f"e{e}": np.asarray(masked_by_exposure.get(e, []), dtype=np.int64)
        for e in sorted(store.image_band_by_exposure)
    }
    arrays_path = tmp_path / "fpm_mask.npz"
    np.savez(arrays_path, **arrays)
    rows = []
    for e, band in sorted(store.image_band_by_exposure.items()):
        exposure = store.exposures[e]
        image_rows = (store.batch.exposure_id == e) & (store.batch.datum_kind == IMAGE)
        rows.append(
            {
                "exposure_id": e, "label": f"{band}-{e}", "band": band,
                "height": exposure.image_height, "width": exposure.image_width,
                "n_store_pixels": int(image_rows.sum()), "n_masked": int(arrays[f"e{e}"].size),
            }
        )
    if edit_rows is not None:
        rows = edit_rows(rows)
    manifest = {
        "schema": 1, "kind": SIDECAR_KIND, "planes": list(FPM_MASK_PLANES),
        "arrays": arrays_path.name, "arrays_sha256": _sha256(arrays_path), "exposures": rows,
    }
    manifest.update(edits)
    path = tmp_path / "fpm_mask.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def _pair_store() -> SyntheticStore:
    """Lower 1x6 and upper 1x4 exposures whose first four pixels are overlap twins."""
    batch = make_batch(
        y=torch.zeros(10, dtype=torch.float32),
        exposure_id=torch.tensor([0] * 6 + [1] * 4),
        datum_kind=torch.full((10,), IMAGE, dtype=torch.long),
        datum_index=torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3]),
    )
    return SyntheticStore(
        exposures={0: image_exposure(band="r", height=1, width=6), 1: image_exposure(band="r", height=1, width=4)},
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=10,
        n_spec=0,
        image_excluded_local_indices_by_exposure={0: np.array([2], dtype=np.int64), 1: np.array([1], dtype=np.int64)},
        image_random_holdout_indices_by_exposure={1: np.array([1], dtype=np.int64)},
    )


def _pair_table(tmp_path) -> str:
    path = tmp_path / "overlaps.json"
    pair = {
        "lower_exposure_id": 0, "upper_exposure_id": 1, "lower_label": "r-lower", "upper_label": "r-upper",
        "band": "r", "lower_y0": 0, "lower_x0": 0, "upper_y0": 0, "upper_x0": 0, "n_rows": 1, "n_cols": 4,
    }
    path.write_text(
        json.dumps(
            {"schema": 1, "frame_row_spacing": FIELD_ROW_SPACING, "overlap_rows": OVERLAP_ROWS,
             "manifest_sha256": "a" * 64, "pairs": [pair], "verification": []}
        ),
        encoding="utf-8",
    )
    return str(path)


def _pair_trainer(tmp_path, masked=None, *, exclude=None) -> DatumTrainer:
    store = _pair_store()
    cfg = dict(TRAINER_CFG, field_overlap_table=_pair_table(tmp_path))
    if masked is not None:
        cfg["image_pixel_mask"] = _write_sidecar(tmp_path, store, masked)
    return _make_trainer(store, cfg, exclude=exclude)


def _holdout_rows(trainer) -> dict[str, list[int]]:
    return {band: rows.tolist() for band, rows in trainer.image_holdout_indices_by_band.items()}


def test_feature_off_and_empty_mask_leave_the_population_unchanged(tmp_path) -> None:
    baseline = _pair_trainer(tmp_path)
    assert _training_local(baseline) == {0: [0, 3, 4, 5]}
    assert not hasattr(baseline, "image_pixel_mask_summary")
    assert baseline.image_pixel_mask_absent_by_exposure is None
    explicit_none = _make_trainer(
        _pair_store(), dict(TRAINER_CFG, field_overlap_table=_pair_table(tmp_path), image_pixel_mask=None),
    )
    empty = _pair_trainer(tmp_path, {})
    for other in (explicit_none, empty):
        assert torch.equal(other.image_train_indices, baseline.image_train_indices)
        assert _holdout_rows(other) == _holdout_rows(baseline)
        assert other.field_overlap_summary == baseline.field_overlap_summary
    assert empty.image_pixel_mask_summary["removed_rows"] == 0


def test_masked_lower_copy_keeps_the_upper_copy(tmp_path) -> None:
    trainer = _pair_trainer(tmp_path, {0: [0]})
    assert _training_local(trainer) == {0: [3, 4, 5], 1: [0]}
    assert trainer.image_pixel_mask_summary["removed_rows_by_exposure"] == {0: 1}


def test_masked_upper_copy_is_not_a_held_twin(tmp_path) -> None:
    trainer = _pair_trainer(tmp_path, {1: [3]})
    assert _training_local(trainer) == {0: [0, 3, 4, 5]}
    assert trainer.field_overlap_summary["pixel_pairs_per_rule_branch"]["anything_else"] == 1


def test_masked_copy_next_to_a_held_twin_leaves_the_holdout_alone(tmp_path) -> None:
    baseline = _pair_trainer(tmp_path)
    for masked, removed in (({0: [1]}, 1), ({1: [2]}, 1), ({1: [1]}, 0), ({0: [2]}, 0)):
        trainer = _pair_trainer(tmp_path, masked)
        assert _training_local(trainer) == {0: [0, 3, 4, 5]}, masked
        assert _holdout_rows(trainer) == _holdout_rows(baseline) == {"r": [7]}, masked
        assert trainer.image_pixel_mask_summary["removed_rows"] == removed, masked


def test_masked_pixel_outside_the_overlap_leaves_training(tmp_path) -> None:
    trainer = _pair_trainer(tmp_path, {0: [4]})
    assert _training_local(trainer) == {0: [0, 3, 5]}


def test_mask_inside_a_held_exposure_has_no_effect(tmp_path) -> None:
    baseline = _pair_trainer(tmp_path, exclude=[1])
    assert _training_local(baseline) == {0: [4, 5]}
    trainer = _pair_trainer(tmp_path, {1: [0, 1, 2, 3]}, exclude=[1])
    assert torch.equal(trainer.image_train_indices, baseline.image_train_indices)
    assert _holdout_rows(trainer) == _holdout_rows(baseline)
    assert trainer.image_pixel_mask_summary["removed_rows"] == 0


def test_masked_pixel_that_is_not_a_store_pixel_is_refused(tmp_path) -> None:
    store = _store_from_pixels({0: np.array([0, 1, 3], dtype=np.int64)}, {0: (1, 4)}, {0: "r"})
    with pytest.raises(RuntimeError, match="trainable rows"):
        _make_trainer(store, dict(TRAINER_CFG, image_pixel_mask=_write_sidecar(tmp_path, store, {0: [2]})))


def test_sidecar_for_another_store_is_refused(tmp_path) -> None:
    store = _pair_store()

    def recount(rows):
        rows[0]["n_store_pixels"] += 1
        return rows

    path = _write_sidecar(tmp_path, store, {0: [4]}, edit_rows=recount)
    with pytest.raises(RuntimeError, match="Rebuild"):
        _make_trainer(store, dict(TRAINER_CFG, image_pixel_mask=path))


def _edit_first(**changes):
    def edit(rows):
        rows[0].update(changes)
        return rows
    return edit


@pytest.mark.parametrize(
    ("kwargs", "masked", "error", "match"),
    [
        ({"schema": 2}, {}, ValueError, "schema-1"),
        ({"kind": "other"}, {}, ValueError, "schema-1"),
        ({"arrays_sha256": "0" * 64}, {}, ValueError, "sha256"),
        ({"edit_rows": _edit_first(band="g")}, {}, RuntimeError, "order drift"),
        ({"edit_rows": _edit_first(width=5)}, {}, ValueError, "store exposure is 1x6"),
        ({"edit_rows": lambda rows: rows[:1]}, {}, ValueError, "exactly the store"),
        ({"edit_rows": _edit_first(n_masked=2)}, {0: [4]}, ValueError, "sorted, unique"),
        ({}, {0: [4, 3]}, ValueError, "sorted, unique"),
        ({}, {0: [3, 3]}, ValueError, "sorted, unique"),
        ({}, {0: [6]}, ValueError, "sorted, unique"),
        ({}, {0: [-1]}, ValueError, "sorted, unique"),
    ],
)
def test_image_pixel_mask_config_is_validated(tmp_path, kwargs, masked, error, match) -> None:
    store = _pair_store()
    path = _write_sidecar(tmp_path, store, masked, **kwargs)
    with pytest.raises(error, match=match):
        _make_trainer(store, dict(TRAINER_CFG, image_pixel_mask=path))


def test_missing_sidecar_arrays_entry_is_refused(tmp_path) -> None:
    store = _pair_store()
    path = _write_sidecar(tmp_path, store, {})
    np.savez(tmp_path / "fpm_mask.npz", e0=np.zeros(0, dtype=np.int64))
    manifest = json.loads((tmp_path / "fpm_mask.json").read_text())
    manifest["arrays_sha256"] = _sha256(tmp_path / "fpm_mask.npz")
    (tmp_path / "fpm_mask.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="do not match the manifest exposures"):
        _make_trainer(store, dict(TRAINER_CFG, image_pixel_mask=path))


def _lifecycle_store() -> SyntheticStore:
    batch = make_batch(
        y=torch.zeros(12),
        sigma=torch.tensor([0.01, 0.01, 1000.0, 0.02, 0.02, 0.02] + [50.0] * 6),
        exposure_id=torch.tensor([0, 0, 0, 1, 1, 1] + [2] * 6),
        datum_kind=torch.tensor([IMAGE] * 6 + [SPECTRUM] * 6),
        datum_index=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5]),
    )
    return SyntheticStore(
        exposures={
            0: image_exposure(band="r", psf_fwhm_arcsec=1.0),
            1: image_exposure(band="r", psf_fwhm_arcsec=2.0),
            2: spec_exposure(psf_fwhm_arcsec=5.0, lam_spec=np.linspace(5000.0, 5500.0, 6)),
        },
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=6,
        n_spec=6,
        image_random_holdout_indices_by_exposure={0: np.array([2])},
        image_excluded_local_indices_by_exposure={0: np.array([2])},
    )


def _lifecycle_trainer(tmp_path, masked=None) -> DatumTrainer:
    torch.manual_seed(3)
    store = _lifecycle_store()
    refine = {"enabled": True, "start": 0, "stop": 100, "every": 10, "lifecycle_statistics": "fitted_images"}
    cfg = {
        "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": 0, "learning_rate": 1e-3,
        "apply_spatial_psf": False, "refine": refine,
    }
    if masked is not None:
        cfg["image_pixel_mask"] = _write_sidecar(tmp_path, store, masked)
    return DatumTrainer(
        model=SplatField(n_splats=4, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4)),
        grid=build_spatial_grid(2, 2, torch.device("cpu")), data=store,
        training_bands=["r"], held_out_band=None, lambda_step=20.0, trainer_cfg=cfg,
        device=torch.device("cpu"),
    )


def test_masked_pixels_leave_the_lifecycle_and_band_noise_statistics(tmp_path) -> None:
    baseline = _lifecycle_trainer(tmp_path)
    assert baseline.refine_strategy.reference_sigma == pytest.approx(0.02)
    assert baseline._prepare_band_tables()["band_sigma"]["r"] == pytest.approx(0.02)
    masked = _lifecycle_trainer(tmp_path, {1: [0, 1]})
    assert masked.refine_strategy.reference_sigma == pytest.approx(0.01)
    assert masked._prepare_band_tables()["band_sigma"]["r"] == pytest.approx(0.01)
    held = _lifecycle_trainer(tmp_path, {0: [2]})
    assert held.refine_strategy.reference_sigma == pytest.approx(0.02)
    assert held._prepare_band_tables()["band_sigma"]["r"] == pytest.approx(0.02)


def _binding_sidecar(tmp_path, *, store="prepared_" + "a" * 64, labels=("r-0", "g-1")):
    path = tmp_path / "binding.json"
    path.write_text(json.dumps({
        "store": store,
        "exposures": [{"exposure_id": i, "label": label} for i, label in enumerate(labels)],
    }), encoding="utf-8")
    return path


def test_sidecar_binding_accepts_its_own_store_and_labels(tmp_path) -> None:
    from survey_fusion.data.fpm_masks import check_sidecar_binding

    check_sidecar_binding(
        _binding_sidecar(tmp_path), store_digest="a" * 64, image_labels={0: "r-0", 1: "g-1"},
    )


@pytest.mark.parametrize("store", ["prepared_" + "b" * 64, ""])
def test_sidecar_binding_refuses_another_or_unnamed_store(tmp_path, store) -> None:
    from survey_fusion.data.fpm_masks import check_sidecar_binding

    with pytest.raises(ValueError, match="was built for store"):
        check_sidecar_binding(
            _binding_sidecar(tmp_path, store=store), store_digest="a" * 64,
            image_labels={0: "r-0", 1: "g-1"},
        )


def test_sidecar_binding_refuses_exposure_label_drift(tmp_path) -> None:
    from survey_fusion.data.fpm_masks import check_sidecar_binding

    with pytest.raises(ValueError, match=r"labels differ .* ids \[1\]"):
        check_sidecar_binding(
            _binding_sidecar(tmp_path), store_digest="a" * 64, image_labels={0: "r-0", 1: "i-1"},
        )


def test_sidecar_binding_without_a_store_name_still_checks_labels(tmp_path) -> None:
    from survey_fusion.data.fpm_masks import check_sidecar_binding

    check_sidecar_binding(
        _binding_sidecar(tmp_path, store=""), store_digest=None, image_labels={0: "r-0", 1: "g-1"},
    )
    with pytest.raises(ValueError, match="labels differ"):
        check_sidecar_binding(_binding_sidecar(tmp_path), store_digest=None, image_labels={0: "r-0"})
