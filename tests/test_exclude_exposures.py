from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory, OnesModel as ExposureValueModel, SyntheticStore as MockStore,
    image_exposure, make_batch,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer
from train import _resolve_excluded_image_exposure_ids

TRAINER_CFG = {
    "kernel_samples_k": 8,
    "replica_samples_k": 1,
    "epochs": 0,
    "learning_rate": 1e-3,
    "apply_spatial_psf": False,
}


def _make_store(device: torch.device) -> MockStore:
    batch = make_batch(
        y=torch.zeros(8, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=device),
        datum_kind=torch.zeros(8, dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long, device=device),
    )
    return MockStore(
        exposures={0: image_exposure(band="g"), 1: image_exposure(band="i")},
        batch=batch,
        image_band_by_exposure={0: "g", 1: "i"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=8,
        n_spec=0,
    )


def _make_trainer(device: torch.device, exclude) -> DatumTrainer:
    return DatumTrainer(
        model=ExposureValueModel(),
        grid=build_spatial_grid(2, 2, device),
        data=_make_store(device),
        training_bands=["g", "i"],
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg=dict(TRAINER_CFG),
        device=device,
        exclude_image_exposure_ids=exclude,
    )


def test_excluded_exposure_removed_from_training_but_still_renderable() -> None:
    device = torch.device("cpu")
    trainer = _make_trainer(device, exclude=[1])
    assert int(trainer.image_train_indices.numel()) == 4
    trained_exposures = torch.index_select(
        trainer.data.batch.exposure_id, 0, trainer.image_train_indices
    )
    assert set(trained_exposures.tolist()) == {0}
    assert 1 not in trainer.image_train_indices_by_exposure
    # The excluded exposure keeps its runtime and its place in the exposure
    # list, so the deterministic render path can still evaluate it.
    assert 1 in trainer.image_exposure_ids
    assert 1 in trainer.runtime_by_exposure


def test_no_exclusion_matches_baseline() -> None:
    device = torch.device("cpu")
    excluded_none = _make_trainer(device, exclude=None)
    excluded_empty = _make_trainer(device, exclude=())
    assert int(excluded_none.image_train_indices.numel()) == 8
    assert torch.equal(
        excluded_none.image_train_indices, excluded_empty.image_train_indices
    )


def test_unknown_excluded_id_raises() -> None:
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="not image"):
        _make_trainer(device, exclude=[7])


class _ObservationWCS:
    @staticmethod
    def pixel_to_omega_xyz(x, y):
        return np.concatenate([x, y, np.ones_like(x)], axis=1)


class _ModelWCS:
    @staticmethod
    def omega_xyz_to_norm_torch(omega):
        return 0.001 * omega[:, 0], 0.001 * omega[:, 1]


def _peaky_image(peak_yx: tuple[int, int]) -> torch.Tensor:
    torch.manual_seed(11)
    image = 0.01 * torch.randn(16, 16)
    image[peak_yx] = 3.0
    return image


def _seed_store() -> SimpleNamespace:
    wcs = SimpleNamespace(obs_wcs=_ObservationWCS(), wcs=_ModelWCS())
    return SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r"},
        image_source_by_exposure={0: _peaky_image((4, 5)), 1: _peaky_image((11, 12))},
        exposures={0: wcs, 1: wcs},
    )


def test_seed_from_store_skips_excluded_exposures() -> None:
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )
    n_used = seed_from_store(field, _seed_store(), exclude_exposure_ids=[1])
    assert n_used == 1


def test_seed_from_store_all_excluded_raises() -> None:
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )
    with pytest.raises(ValueError, match="every image exposure is excluded"):
        seed_from_store(field, _seed_store(), exclude_exposure_ids=[0, 1])


def test_seed_from_store_rejects_removed_floor_frac() -> None:
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )
    with pytest.raises(TypeError, match="floor_frac"):
        seed_from_store(field, _seed_store(), floor_frac=0.25)


def test_multiband_seeding_covers_excluded_r_footprint_from_g_band() -> None:
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    class OffsetObservationWCS:
        def __init__(self, x_offset):
            self.x_offset = float(x_offset)

        def pixel_to_omega_xyz(self, x, y):
            return np.concatenate([x + self.x_offset, y, np.ones_like(x)], axis=1)

    def exposure(x_offset):
        return SimpleNamespace(
            obs_wcs=OffsetObservationWCS(x_offset), wcs=_ModelWCS()
        )

    def image(peak_yx):
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        y, x = peak_yx
        return 5.0 - (yy - y).float() ** 2 - (xx - x).float() ** 2

    store = SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r", 2: "g"},
        image_source_by_exposure={
            0: image((4, 5)),
            1: image((4, 5)),
            2: image((4, 5)),
        },
        exposures={0: exposure(0.0), 1: exposure(100.0), 2: exposure(0.0)},
    )
    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    seed_from_store(field, store, prefer_band=None, exclude_exposure_ids=[0])

    excluded_r_footprint_peak = torch.tensor([[0.005, 0.004]])
    assert torch.cdist(field.mu, excluded_r_footprint_peak).min() < 1.0e-7


def _write_manifest(tmp_path, entries):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"image_exposures": entries}), encoding="utf-8")
    return path


def _manifest_entries():
    return [
        {"label": "g-001462-5-0167", "band": "g"},
        {"label": "r-001462-5-0167", "band": "r"},
        {"label": "i-002583-2-0136", "band": "i"},
    ]


def _resolver_store():
    return SimpleNamespace(image_band_by_exposure={0: "g", 1: "r", 2: "i"})


def test_resolver_maps_labels_to_manifest_positions(tmp_path) -> None:
    path = _write_manifest(tmp_path, _manifest_entries())
    ids = _resolve_excluded_image_exposure_ids(
        {"scene_manifest_path": str(path)},
        {"exclude_exposure_labels": ["i-002583-2-0136", "g-001462-5-0167"]},
        _resolver_store(),
    )
    assert ids == (0, 2)


def test_resolver_empty_labels_is_noop(tmp_path) -> None:
    assert (
        _resolve_excluded_image_exposure_ids({}, {"exclude_exposure_labels": []}, None)
        == ()
    )


def test_resolver_unknown_label_raises(tmp_path) -> None:
    path = _write_manifest(tmp_path, _manifest_entries())
    with pytest.raises(ValueError, match="not an"):
        _resolve_excluded_image_exposure_ids(
            {"scene_manifest_path": str(path)},
            {"exclude_exposure_labels": ["z-000000-0-0000"]},
            _resolver_store(),
        )


def test_resolver_duplicate_request_raises(tmp_path) -> None:
    path = _write_manifest(tmp_path, _manifest_entries())
    with pytest.raises(ValueError, match="duplicates"):
        _resolve_excluded_image_exposure_ids(
            {"scene_manifest_path": str(path)},
            {"exclude_exposure_labels": ["g-001462-5-0167", "g-001462-5-0167"]},
            _resolver_store(),
        )


def test_resolver_band_drift_raises(tmp_path) -> None:
    path = _write_manifest(tmp_path, _manifest_entries())
    store = SimpleNamespace(image_band_by_exposure={0: "r", 1: "g", 2: "i"})
    with pytest.raises(RuntimeError, match="order drift"):
        _resolve_excluded_image_exposure_ids(
            {"scene_manifest_path": str(path)},
            {"exclude_exposure_labels": ["g-001462-5-0167"]},
            store,
        )


def _splat_refine_trainer(exclude) -> DatumTrainer:
    from survey_fusion.models.splat_field import SplatField

    device = torch.device("cpu")
    torch.manual_seed(3)
    field = SplatField(
        n_splats=4, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4)
    )
    batch = make_batch(
        y=torch.zeros(8, dtype=torch.float32),
        sigma=torch.tensor([0.01, 0.01, 0.01, 100.0, 100.0, 100.0, 100.0, 100.0]),
        exposure_id=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long),
        datum_kind=torch.zeros(8, dtype=torch.long),
        datum_index=torch.tensor([0, 1, 2, 0, 1, 2, 3, 0], dtype=torch.long),
    )
    store = MockStore(
        exposures={
            0: image_exposure(band="r", psf_fwhm_arcsec=1.0),
            1: image_exposure(band="r", psf_fwhm_arcsec=3.0),
        },
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=8,
        n_spec=0,
    )
    return DatumTrainer(
        model=field,
        grid=build_spatial_grid(2, 2, device),
        data=store,
        training_bands=["r"],
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg=dict(
            TRAINER_CFG,
            refine={"enabled": True, "start": 0, "stop": 100, "every": 10},
        ),
        device=device,
        exclude_image_exposure_ids=exclude,
    )


def test_refine_reference_sigma_ignores_excluded_exposures() -> None:
    contaminated = _splat_refine_trainer(exclude=None)
    clean = _splat_refine_trainer(exclude=[1])
    assert contaminated.refine_strategy.reference_sigma == pytest.approx(100.0)
    assert clean.refine_strategy.reference_sigma == pytest.approx(0.01)


def test_checkpoint_records_and_validates_exclusion_provenance(tmp_path) -> None:
    import copy

    from train import _build_training_checkpoint, _load_resume_state_into_trainer

    device = torch.device("cpu")

    def build(labels):
        return DatumTrainer(
            model=ExposureValueModel(),
            grid=build_spatial_grid(2, 2, device),
            data=_make_store(device),
            training_bands=["g", "i"],
            held_out_band=None,
            lambda_step=20.0,
            trainer_cfg=dict(TRAINER_CFG, exclude_exposure_labels=list(labels)),
            device=device,
            exclude_image_exposure_ids=[1] if labels else None,
        )

    trainer = build(["i-002583-2-0136"])
    checkpoint = _build_training_checkpoint(
        trainer, SimpleNamespace(epochs_ran=0, steps_ran=0)
    )
    assert checkpoint["operator_config"]["exclude_exposure_labels"] == [
        "i-002583-2-0136"
    ]

    tampered = copy.deepcopy(checkpoint)
    tampered["operator_config"]["exclude_exposure_labels"] = []
    with pytest.raises(ValueError, match="exclude_exposure_labels"):
        _load_resume_state_into_trainer(
            trainer, tampered, checkpoint_path=tmp_path / "model.pt"
        )


def _load_eval_module():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "eval_holdout_exposures.py"
    )
    spec = importlib.util.spec_from_file_location("eval_holdout_exposures", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pick_controls_only_from_trained_exposures() -> None:
    module = _load_eval_module()
    store = SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r", 2: "r", 3: "r"},
        exposures={
            i: SimpleNamespace(psf_fwhm_arcsec=1.0 + 0.1 * i) for i in range(4)
        },
    )
    controls, matching = module.pick_controls(
        store, [0], per_band=2, trained_ids={2, 3}, psf_model="gaussian",
    )
    assert controls == [2, 3]
    assert matching["preferred_width_field"] == "psf_fwhm_arcsec"
    with pytest.raises(RuntimeError, match="No trained same-band control"):
        module.pick_controls(
            store, [0], per_band=1, trained_ids=set(), psf_model="gaussian",
        )


@pytest.mark.parametrize(
    ("psf_model", "expected", "field"),
    [
        ("gaussian", 1, "psf_fwhm_arcsec"),
        ("dgpsf", 2, "psf_effective_fwhm_dg_arcsec"),
        ("mog", 3, "psf_effective_fwhm_kl_arcsec"),
        ("kl", 3, "psf_effective_fwhm_kl_arcsec"),
    ],
)
def test_pick_controls_uses_active_operator_width(
    psf_model: str, expected: int, field: str,
) -> None:
    module = _load_eval_module()
    store = SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r", 2: "r", 3: "r"},
        exposures={
            0: SimpleNamespace(
                psf_fwhm_arcsec=1.0, psf_effective_fwhm_dg_arcsec=2.0,
                psf_effective_fwhm_kl_arcsec=3.0,
            ),
            1: SimpleNamespace(
                psf_fwhm_arcsec=1.1, psf_effective_fwhm_dg_arcsec=4.0,
                psf_effective_fwhm_kl_arcsec=5.0,
            ),
            2: SimpleNamespace(
                psf_fwhm_arcsec=1.9, psf_effective_fwhm_dg_arcsec=2.1,
                psf_effective_fwhm_kl_arcsec=6.0,
            ),
            3: SimpleNamespace(
                psf_fwhm_arcsec=2.5, psf_effective_fwhm_dg_arcsec=5.0,
                psf_effective_fwhm_kl_arcsec=3.1,
            ),
        },
    )
    controls, matching = module.pick_controls(
        store, [0], per_band=1, trained_ids={1, 2, 3}, psf_model=psf_model,
    )
    assert controls == [expected]
    assert matching["preferred_width_field"] == field
    assert set(matching["width_field_by_exposure"].values()) == {field}


def test_pick_controls_missing_model_width_warns_and_uses_legacy() -> None:
    module = _load_eval_module()
    store = SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r", 2: "r"},
        exposures={
            0: SimpleNamespace(psf_fwhm_arcsec=1.0),
            1: SimpleNamespace(psf_fwhm_arcsec=1.1),
            2: SimpleNamespace(psf_fwhm_arcsec=1.8),
        },
    )
    with pytest.warns(RuntimeWarning, match="psf_effective_fwhm_dg_arcsec"):
        controls, matching = module.pick_controls(
            store, [0], per_band=1, trained_ids={1, 2}, psf_model="dgpsf",
        )
    assert controls == [1]
    assert set(matching["width_field_by_exposure"].values()) == {"psf_fwhm_arcsec"}
