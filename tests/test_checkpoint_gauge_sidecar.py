from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import survey_fusion.training.checkpoints as checkpoints_module
from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.models.splat_field import SplatField
from survey_fusion.training.checkpoints import (
    GAUGE_SIDECAR_FILENAME, build_gauge_canonical_sidecar, checkpoint_is_gauge_canonical,
    model_config_for_checkpoint,
)
from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes


def _load_entrypoint_module(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_entrypoint_module("train.py", "survey_fusion_train_gauge_sidecar_entry")

LOGLAM_MIN = math.log(3400.0)
LOGLAM_MAX = math.log(9200.0)
BANDPASSES = {
    "x": (
        torch.tensor([3800.0, 4200.0, 5100.0, 6200.0, 7100.0]),
        torch.tensor([0.0, 0.75, 1.0, 0.4, 0.0]),
    ),
    "y": (
        torch.tensor([6800.0, 7600.0, 8600.0]),
        torch.tensor([0.0, 1.0, 0.0]),
    ),
}


def _field(n_splats: int = 6) -> SplatField:
    torch.manual_seed(7)
    field = SplatField(
        n_splats=n_splats,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.002,
    )
    field.sed.set_loglam_scale(0.5 * (LOGLAM_MAX - LOGLAM_MIN))
    with torch.no_grad():
        field.raw_flux.uniform_(-2.0, 1.0)
        field.sed_coeffs.add_(3.0)  # drive band-max unit amplitude far from 1
    return field


def _band_amplitudes(field: SplatField, *, include_flux: bool = True) -> dict:
    return splat_band_amplitudes(
        field, BANDPASSES, loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX,
        include_flux=include_flux,
    )


def _gauge_factors(field: SplatField) -> torch.Tensor:
    unit = _band_amplitudes(field, include_flux=False)
    return torch.stack([unit[b] for b in sorted(unit)], dim=0).amax(dim=0).detach()


def _checkpoint(field: SplatField, *, step: int) -> dict[str, object]:
    return {
        "checkpoint_version": train_entry.CURRENT_CHECKPOINT_VERSION,
        "operator_config": {"psf_model": "gaussian"},
        "model_config": model_config_for_checkpoint(field),
        "model_state": field.state_dict(),
        "calibration_state": None,
        "optimizer_state": {"format": "hybrid_sparse_v1"},
        "lr_scheduler_state": None,
        "resume_meta": {"global_step_completed": int(step)},
    }


def _loaded(sidecar: dict[str, object], field: SplatField) -> SplatField:
    reloaded = SplatField(
        n_splats=int(field.mu.shape[0]),
        n_templates=int(field.sed_coeffs.shape[1]),
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.002,
    )
    reloaded.sed.set_loglam_scale(0.5 * (LOGLAM_MAX - LOGLAM_MIN))
    reloaded.load_state_dict(sidecar["model_state"])
    return reloaded


def test_sidecar_is_prediction_preserving_and_canonicalizes_the_gauge() -> None:
    field = _field()
    before = _band_amplitudes(field)

    sidecar = build_gauge_canonical_sidecar(
        _checkpoint(field, step=1200), gauge_factors=_gauge_factors(field),
        source_step=1200, source_name="model.pt",
    )

    canonical = _loaded(sidecar, field)
    after = _band_amplitudes(canonical)
    for band in BANDPASSES:
        torch.testing.assert_close(after[band], before[band], rtol=5e-5, atol=0.0)
    band_max = _gauge_factors(canonical)
    torch.testing.assert_close(
        band_max, torch.ones_like(band_max), rtol=5e-5, atol=0.0
    )
    assert sidecar["gauge_projected_splats"] == int(field.mu.shape[0])


def test_sidecar_stamps_its_provenance_and_carries_no_optimizer_state() -> None:
    field = _field()

    sidecar = build_gauge_canonical_sidecar(
        _checkpoint(field, step=97_638), gauge_factors=_gauge_factors(field),
        source_step=97_638, source_name="model.pt",
    )

    assert checkpoint_is_gauge_canonical(sidecar)
    assert sidecar["gauge_drift_log"] == 0.0
    assert sidecar["gauge_source_step"] == 97_638
    assert sidecar["gauge_source_checkpoint"] == "model.pt"
    assert sidecar["optimizer_state"] is None
    assert sidecar["checkpoint_version"] == train_entry.CURRENT_CHECKPOINT_VERSION


def test_sidecar_stamps_sed_operator_provenance() -> None:
    field = SplatField(
        n_splats=6, n_templates=4, loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.002, sed_off_grid="zero", sed_pin_gauge_knots=True,
    )
    field.sed.set_loglam_scale(0.5 * (LOGLAM_MAX - LOGLAM_MIN))

    sidecar = build_gauge_canonical_sidecar(
        _checkpoint(field, step=12), gauge_factors=_gauge_factors(field),
        source_step=12, source_name="model.pt",
    )

    assert sidecar["model_config"]["sed_off_grid"] == "zero"
    assert sidecar["model_config"]["sed_pin_gauge_knots"] is True


def test_sidecar_leaves_the_source_checkpoint_and_live_model_untouched() -> None:
    field = _field()
    checkpoint = _checkpoint(field, step=10)
    original_coeffs = field.sed_coeffs.detach().clone()
    original_flux = field.raw_flux.detach().clone()

    build_gauge_canonical_sidecar(
        checkpoint, gauge_factors=_gauge_factors(field), source_step=10,
        source_name="model.pt",
    )

    assert torch.equal(field.sed_coeffs.detach(), original_coeffs)
    assert torch.equal(field.raw_flux.detach(), original_flux)
    assert torch.equal(checkpoint["model_state"]["sed_coeffs"], original_coeffs)
    assert torch.equal(checkpoint["model_state"]["raw_flux"], original_flux)


def test_sidecar_chunking_matches_a_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    field = _field(n_splats=9)
    factors = _gauge_factors(field)
    whole = build_gauge_canonical_sidecar(
        _checkpoint(field, step=1), gauge_factors=factors, source_step=1,
        source_name="model.pt",
    )

    monkeypatch.setattr(checkpoints_module, "GAUGE_SIDECAR_CHUNK", 2)
    chunked = build_gauge_canonical_sidecar(
        _checkpoint(field, step=1), gauge_factors=factors, source_step=1,
        source_name="model.pt",
    )

    assert chunked["gauge_projected_splats"] == whole["gauge_projected_splats"]
    for key, value in whole["model_state"].items():
        torch.testing.assert_close(chunked["model_state"][key], value, rtol=0.0, atol=0.0)


def test_sidecar_rejects_gauge_factors_that_do_not_match_the_census() -> None:
    field = _field()

    with pytest.raises(ValueError, match="per-splat vector"):
        build_gauge_canonical_sidecar(
            _checkpoint(field, step=1), gauge_factors=_gauge_factors(field)[:-1],
            source_step=1, source_name="model.pt",
        )


class _SidecarTrainer:
    def __init__(self, field: SplatField, *, enabled: bool) -> None:
        self.model = field
        self.checkpoint_gauge_sidecar_enabled = bool(enabled)

    def _splat_gauge_factors(self, model) -> torch.Tensor:
        return _gauge_factors(model)


def test_sidecar_file_is_written_only_when_the_flag_is_on(tmp_path: Path) -> None:
    field = _field()
    checkpoint = _checkpoint(field, step=500)

    assert train_entry._maybe_save_gauge_canonical_sidecar(
        _SidecarTrainer(field, enabled=False), checkpoint, tmp_path, step=500,
    ) is None
    assert not (tmp_path / GAUGE_SIDECAR_FILENAME).exists()

    path = train_entry._maybe_save_gauge_canonical_sidecar(
        _SidecarTrainer(field, enabled=True), checkpoint, tmp_path, step=500,
    )

    assert path == tmp_path / GAUGE_SIDECAR_FILENAME
    assert not (tmp_path / f"{GAUGE_SIDECAR_FILENAME}.partial").exists()
    written = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint_is_gauge_canonical(written)
    assert written["gauge_source_step"] == 500


def test_sidecar_filename_stays_clear_of_the_archiver_retention_glob(
    tmp_path: Path,
) -> None:
    field = _field()
    train_entry._maybe_save_gauge_canonical_sidecar(
        _SidecarTrainer(field, enabled=True), _checkpoint(field, step=500), tmp_path,
        step=500,
    )

    assert list(tmp_path.glob("checkpoint_step_*.pt")) == []


def test_resume_refuses_a_gauge_canonical_sidecar(tmp_path: Path) -> None:
    field = _field()
    sidecar = build_gauge_canonical_sidecar(
        _checkpoint(field, step=10), gauge_factors=_gauge_factors(field), source_step=10,
        source_name="model.pt",
    )
    trainer = make_image_spec_trainer(model=_field(), trainer_cfg={"epochs": 0})

    with pytest.raises(ValueError, match="gauge-canonical sidecar"):
        train_entry._load_resume_state_into_trainer(
            trainer, sidecar, checkpoint_path=tmp_path / "model_canonical.pt",
        )


def test_gauge_sidecar_flag_requires_a_splat_model() -> None:
    with pytest.raises(ValueError, match="checkpoint_gauge_sidecar"):
        make_image_spec_trainer(trainer_cfg={"checkpoint_gauge_sidecar": True})


def test_gauge_sidecar_flag_is_accepted_for_a_splat_model() -> None:
    trainer = make_image_spec_trainer(
        model=_field(), trainer_cfg={"checkpoint_gauge_sidecar": True},
    )

    assert trainer.checkpoint_gauge_sidecar_enabled is True


def test_checkpoint_build_is_unchanged_when_the_sidecar_flag_is_off(
    tmp_path: Path,
) -> None:
    trainer = make_image_spec_trainer(model=_field(), trainer_cfg={"epochs": 0})
    fit_state = SimpleNamespace(epochs_ran=0, steps_ran=0)

    train_entry._save_training_checkpoint(trainer, fit_state, tmp_path)

    assert trainer.checkpoint_gauge_sidecar_enabled is False
    assert (tmp_path / "model.pt").exists()
    assert not (tmp_path / GAUGE_SIDECAR_FILENAME).exists()
