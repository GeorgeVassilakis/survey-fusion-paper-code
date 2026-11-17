from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_module(
    Path(__file__).resolve().parents[1] / "src" / "train.py", "survey_fusion_train_finalization_resilience",
)
eval_entry = train_entry


def test_fiber_sigma_by_exposure_uses_nll_tensor_and_datum_order() -> None:
    data_store = SimpleNamespace(
        batch=SimpleNamespace(
            exposure_id=torch.tensor([3, 3, 2, 3], dtype=torch.long),
            datum_index=torch.tensor([2, 0, 0, 1], dtype=torch.long),
            sigma=torch.tensor([0.3, 0.1, 9.0, 0.2], dtype=torch.float32),
        ),
        spec_diagnostics_by_exposure={
            3: SimpleNamespace(lam=np.array([5000.0, 5100.0, 5200.0]))
        },
    )

    sigma = train_entry._fiber_sigma_by_exposure(data_store)

    np.testing.assert_allclose(sigma[3], np.array([0.1, 0.2, 0.3]), rtol=1e-6)


class FakeTrainer:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.lr_scheduler = None

    def get_calibration_state_dict(self):
        return None

    def optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def refine_state_dict(self):
        return None

    def recipe_record(self):
        return {"schema": 1}


def test_save_training_checkpoint_persists_model_file(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    fit_state = SimpleNamespace(epochs_ran=3, steps_ran=17)

    ckpt_path = train_entry._save_training_checkpoint(trainer, fit_state, tmp_path)

    assert ckpt_path == tmp_path / "model.pt"
    assert ckpt_path.exists()
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    assert checkpoint["checkpoint_version"] == train_entry.CURRENT_CHECKPOINT_VERSION
    assert "model_state" in checkpoint
    assert "mode" not in checkpoint["resume_meta"]
    assert checkpoint["resume_meta"]["epochs_ran_this_invocation"] == 3
    assert checkpoint["resume_meta"]["steps_ran_this_invocation"] == 17


def test_no_holdout_primary_image_exposure_prefers_i_band() -> None:
    data_store = SimpleNamespace(image_band_by_exposure={0: "u", 1: "g", 2: "r", 3: "i", 4: "z"})

    assert train_entry._primary_image_exposure_id(data_store, None) == 3
    assert eval_entry._primary_image_exposure_id(data_store, None) == 3


def test_no_holdout_primary_image_exposure_falls_back_to_first_available_band() -> None:
    data_store = SimpleNamespace(image_band_by_exposure={2: "g", 5: "r", 9: "z"})

    assert train_entry._primary_image_exposure_id(data_store, None) == 2
    assert eval_entry._primary_image_exposure_id(data_store, None) == 2


def test_resolve_diagnostic_outputs_accepts_total_source_and_both() -> None:
    assert train_entry._resolve_diagnostic_outputs(None) == ("total", "source")
    assert eval_entry._resolve_diagnostic_outputs({}) == ("total", "source")
    assert train_entry._resolve_diagnostic_outputs({"diagnostic_outputs": "total"}) == ("total",)
    assert eval_entry._resolve_diagnostic_outputs({"diagnostic_outputs": ["source", "total"]}) == (
        "source", "total",
    )
    assert train_entry._resolve_diagnostic_outputs({"diagnostic_outputs": ["total", "total"]}) == ("total",)


def test_resolve_diagnostic_outputs_rejects_unknown_or_empty_values() -> None:
    with pytest.raises(ValueError, match="total.*source"):
        train_entry._resolve_diagnostic_outputs({"diagnostic_outputs": ["model"]})
    with pytest.raises(ValueError, match="at least one"):
        eval_entry._resolve_diagnostic_outputs({"diagnostic_outputs": []})


def test_prediction_npz_loader_restores_nested_exposure_images(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    np.savez(path, e3__i=image)

    nested = train_entry._load_nested_images_npz(path)

    assert set(nested) == {3}
    np.testing.assert_allclose(nested[3]["i"], image)


def test_prediction_npz_loader_rejects_legacy_flat_keys(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    np.savez(path, i=np.ones((1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="expected"):
        train_entry._load_nested_images_npz(path)


def test_train_postprocessing_failure_keeps_checkpoint_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = FakeTrainer()
    fit_state = SimpleNamespace(epochs_ran=1, steps_ran=2)
    ckpt_path = train_entry._save_training_checkpoint(trainer, fit_state, tmp_path)

    def _boom(**kwargs) -> None:
        del kwargs
        raise RuntimeError("plot boom")

    monkeypatch.setattr(train_entry, "_run_train_postprocessing", _boom)

    with pytest.raises(RuntimeError, match="Postprocessing failed closed"):
        train_entry._run_train_postprocessing_safely(
            run_dir=tmp_path, ckpt_path=ckpt_path,
        )

    output = capsys.readouterr().out
    assert ckpt_path.exists()
    assert "post-training diagnostics failed after checkpoint save" in output
    assert "kept checkpoint" in output
    status = __import__("json").loads(
        (tmp_path / "postprocess_status.json").read_text(encoding="utf-8")
    )
    assert status["overall"] == "failed"
    assert set(status["missing_mandatory_outputs"]) == {"metrics.json", "history.json"}


def test_train_postprocessing_success_path_can_write_optional_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def _fake_postprocess(**kwargs) -> None:
        run_dir = Path(kwargs["run_dir"])
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")
        (run_dir / "point_seds.png").write_bytes(b"png")
        (run_dir / "predictions.npz").write_bytes(b"npz")

    monkeypatch.setattr(train_entry, "_run_train_postprocessing", _fake_postprocess)

    train_entry._run_train_postprocessing_safely(run_dir=tmp_path, ckpt_path=tmp_path / "model.pt")

    output = capsys.readouterr().out
    assert "Warning:" not in output
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "point_seds.png").exists()
    assert (tmp_path / "predictions.npz").exists()


def test_train_postprocessing_logs_wandb_with_final_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Metric:
        mse = 1.0
        mae = 2.0
        rmse = 3.0
        flux_error_pct = 4.0
        nrmse_pct = 5.0

    class _RecordingWandb:
        def __init__(self) -> None:
            self.scalar_steps: list[int | None] = []
            self.image_steps: list[int | None] = []

        def log(self, payload, step: int | None = None) -> None:
            assert payload
            self.scalar_steps.append(step)

        def log_images(self, payload, step: int | None = None) -> None:
            assert payload == {}
            self.image_steps.append(step)

        def log_artifacts(self, name: str, files) -> None:
            del name, files

    trainer = SimpleNamespace(
        held_out_band="i",
        history={"loss": [1.0]},
        psf_model="gaussian",
        use_flux_density=True,
        diagnostic_renderer=SimpleNamespace(
            grid=SimpleNamespace(lam=torch.tensor([5000.0], dtype=torch.float32))
        ),
        get_calibration_report=lambda: {},
        sample_point_spectra=lambda points, **kwargs: {},
        predict_fiber_spectra=lambda **kwargs: {},
    )
    fit_state = SimpleNamespace(steps_ran=17)
    image = np.ones((1, 1), dtype=np.float32)
    data_store = SimpleNamespace(
        exposures={0: SimpleNamespace(fiber_mask=np.ones((1, 1), dtype=np.float32))},
        image_band_by_exposure={0: "i"}, spec_diagnostics_by_exposure={},
    )
    wandb = _RecordingWandb()

    monkeypatch.setattr(
        train_entry, "_compute_image_diagnostic_predictions", lambda *args, **kwargs: {0: {"i": image}},
    )
    monkeypatch.setattr(
        train_entry.metrics_mod,
        "compute_metrics",
        lambda *args, **kwargs: {"e0:i": _Metric()},
    )
    monkeypatch.setattr(train_entry.metrics_mod, "compute_region_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_entry.metrics_mod, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "log_region_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(train_entry.metrics_mod, "region_metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks_by_exposure", lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks", lambda *args, **kwargs: ({}, None, 0, {}),
    )
    monkeypatch.setattr(train_entry.visualization, "plot_band_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.visualization, "plot_true_band_with_fiber_mask", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_entry.visualization, "plot_holdout_region_overlay", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.visualization, "plot_region_cutout_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_radial_profiles", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.diagnostics, "write_point_sed_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.diagnostics, "write_fiber_spectrum_diagnostics", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.diagnostics, "log_radial_diagnostics", lambda *args, **kwargs: None)

    train_entry._run_train_postprocessing(
        trainer=trainer, fit_state=fit_state, bandpasses={}, scene=SimpleNamespace(), data_store=data_store,
        observations={0: {"i": image}}, region_masks_by_exposure={}, primary_region=None,
        holdout_summary=None, run_dir=tmp_path, wandb_run=wandb, wandb_cfg={"log_artifacts": False},
        run_name="test_run", ckpt_path=tmp_path / "model.pt",
    )

    assert wandb.scalar_steps == [17]
    assert wandb.image_steps == [17]


def test_train_postprocessing_region_metrics_use_all_resolved_image_exposures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace(
        held_out_band="i",
        history={"loss": [1.0]},
        psf_model="gaussian",
        use_flux_density=True,
        diagnostic_renderer=SimpleNamespace(
            grid=SimpleNamespace(lam=torch.tensor([5000.0], dtype=torch.float32))
        ),
        get_calibration_report=lambda: {},
        sample_point_spectra=lambda points, **kwargs: {},
        predict_fiber_spectra=lambda **kwargs: {},
    )
    fit_state = SimpleNamespace(steps_ran=1)
    image = np.ones((1, 1), dtype=np.float32)
    data_store = SimpleNamespace(
        exposures={
            0: SimpleNamespace(fiber_mask=np.ones((1, 1), dtype=np.float32)),
            1: SimpleNamespace(fiber_mask=np.ones((1, 1), dtype=np.float32)),
        },
        image_band_by_exposure={0: "g", 1: "i"},
        spec_diagnostics_by_exposure={},
    )
    recorded_masks: list[dict[int, dict[str, np.ndarray]]] = []

    monkeypatch.setattr(
        train_entry, "_compute_image_diagnostic_predictions", lambda *args,
        **kwargs: {0: {"g": image}, 1: {"i": image}},
    )
    monkeypatch.setattr(train_entry.metrics_mod, "compute_metrics", lambda *args, **kwargs: {})

    def _capture_region_metrics(_predictions, _observations, region_masks_by_exposure, **kwargs):
        del kwargs
        recorded_masks.append(region_masks_by_exposure)
        return {}

    monkeypatch.setattr(train_entry.metrics_mod, "compute_region_metrics", _capture_region_metrics)
    monkeypatch.setattr(train_entry.metrics_mod, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "log_region_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(train_entry.metrics_mod, "region_metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(
        train_entry.visualization,
        "resolve_region_masks_by_exposure",
        lambda *args, **kwargs: {
            0: {"G1": np.ones((1, 1), dtype=bool)}, 1: {"G1": np.ones((1, 1), dtype=bool)},
        },
    )
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks", lambda *args,
        **kwargs: ({"G1": np.ones((1, 1), dtype=bool)}, None, 0, {}),
    )
    monkeypatch.setattr(train_entry.visualization, "plot_band_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.visualization, "plot_true_band_with_fiber_mask", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_entry.visualization, "plot_holdout_region_overlay", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.visualization, "plot_region_cutout_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_radial_profiles", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.diagnostics, "write_point_sed_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.diagnostics, "write_fiber_spectrum_diagnostics", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.diagnostics, "log_radial_diagnostics", lambda *args, **kwargs: None)

    train_entry._run_train_postprocessing(
        trainer=trainer, fit_state=fit_state, bandpasses={}, scene=SimpleNamespace(), data_store=data_store,
        observations={0: {"g": image}, 1: {"i": image}}, region_masks_by_exposure={}, primary_region=None,
        holdout_summary=None, run_dir=tmp_path, wandb_run=None, wandb_cfg={}, run_name="test_run",
        ckpt_path=tmp_path / "model.pt",
    )

    assert len(recorded_masks) == 2
    assert all(set(masks.keys()) == {0, 1} for masks in recorded_masks)


def test_train_postprocessing_skips_region_outputs_when_region_metrics_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = SimpleNamespace(
        held_out_band="i",
        history={"loss": [1.0]},
        psf_model="gaussian",
        use_flux_density=True,
        diagnostic_renderer=SimpleNamespace(
            grid=SimpleNamespace(lam=torch.tensor([5000.0], dtype=torch.float32))
        ),
        get_calibration_report=lambda: {},
        sample_point_spectra=lambda points, **kwargs: {},
        predict_fiber_spectra=lambda **kwargs: {},
    )
    fit_state = SimpleNamespace(steps_ran=1)
    image = np.ones((1, 1), dtype=np.float32)
    data_store = SimpleNamespace(
        exposures={0: SimpleNamespace(fiber_mask=np.ones((1, 1), dtype=np.float32))},
        image_band_by_exposure={0: "i"}, spec_diagnostics_by_exposure={}, region_metrics_enabled=False,
    )

    monkeypatch.setattr(
        train_entry, "_compute_image_diagnostic_predictions", lambda *args, **kwargs: {0: {"i": image}},
    )
    monkeypatch.setattr(train_entry.metrics_mod, "compute_metrics", lambda *args, **kwargs: {})

    def _unexpected_region_metrics(*args, **kwargs):
        raise AssertionError("region metrics should be skipped when disabled")

    def _unexpected_plot_region_grid(*args, **kwargs):
        raise AssertionError("region grid plotting should be skipped when disabled")

    monkeypatch.setattr(train_entry.metrics_mod, "compute_region_metrics", _unexpected_region_metrics)
    monkeypatch.setattr(train_entry.metrics_mod, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "log_region_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(train_entry.metrics_mod, "region_metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks_by_exposure", lambda *args,
        **kwargs: {0: {"G1": np.ones((1, 1), dtype=bool)}},
    )
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks", lambda *args,
        **kwargs: ({"G1": np.ones((1, 1), dtype=bool)}, None, 0, {}),
    )
    monkeypatch.setattr(train_entry.visualization, "plot_band_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.visualization, "plot_true_band_with_fiber_mask", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_entry.visualization, "plot_holdout_region_overlay", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.visualization, "plot_region_cutout_grid", _unexpected_plot_region_grid)
    monkeypatch.setattr(train_entry.visualization, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_radial_profiles", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.diagnostics, "write_point_sed_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.diagnostics, "write_fiber_spectrum_diagnostics", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.diagnostics, "log_radial_diagnostics", lambda *args, **kwargs: None)

    train_entry._run_train_postprocessing(
        trainer=trainer, fit_state=fit_state, bandpasses={}, scene=SimpleNamespace(), data_store=data_store,
        observations={0: {"i": image}}, region_masks_by_exposure={}, primary_region=None,
        holdout_summary=None, run_dir=tmp_path, wandb_run=None, wandb_cfg={}, run_name="test_run",
        ckpt_path=tmp_path / "model.pt",
    )

    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "region_metrics.json").exists()


def test_eval_postprocessing_failure_preserves_metadata_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")

    def _boom(**kwargs) -> None:
        del kwargs
        raise RuntimeError("plot boom")

    monkeypatch.setattr(eval_entry, "_run_eval_postprocessing", _boom)

    with pytest.raises(RuntimeError, match="Postprocessing failed closed"):
        eval_entry._run_eval_postprocessing_safely(run_dir=tmp_path)

    output = capsys.readouterr().out
    assert calibration_path.exists()
    assert "eval diagnostics failed after writing initial metadata" in output


def test_optional_postprocessing_failure_is_recorded_and_later_stage_runs(
    tmp_path: Path,
) -> None:
    later = tmp_path / "later.txt"

    def _callback(**kwargs) -> None:
        run_dir = Path(kwargs["run_dir"])
        status = kwargs["_postprocess_status"]
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")

        def _boom() -> None:
            raise RuntimeError("optional plot failed")

        status.run_optional("band_grid", _boom)
        status.run_optional(
            "radial_profiles", lambda: later.write_text("ran", encoding="utf-8"),
        )

    train_entry._run_postprocessing_safely(_callback, "warning", run_dir=tmp_path)

    status = __import__("json").loads(
        (tmp_path / "postprocess_status.json").read_text(encoding="utf-8")
    )
    by_name = {entry["name"]: entry for entry in status["stages"]}
    assert set(by_name) == set(train_entry._POSTPROCESS_STAGE_NAMES)
    assert by_name["band_grid"]["status"] == "failed"
    assert by_name["radial_profiles"]["status"] == "ok"
    assert status["overall"] == "partial"
    assert later.read_text(encoding="utf-8") == "ran"


def test_missing_mandatory_fiber_metrics_exits_nonzero(tmp_path: Path) -> None:
    def _callback(**kwargs) -> None:
        run_dir = Path(kwargs["run_dir"])
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")

    data_store = SimpleNamespace(spec_diagnostics_by_exposure={7: object()})
    with pytest.raises(RuntimeError, match="fiber_metrics.json"):
        train_entry._run_postprocessing_safely(
            _callback, "warning", run_dir=tmp_path, data_store=data_store,
        )


def test_unclassified_failure_after_mandatory_outputs_still_exits_nonzero(
    tmp_path: Path,
) -> None:
    def _callback(**kwargs) -> None:
        run_dir = Path(kwargs["run_dir"])
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "history.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("failed after mandatory stage boundary")

    with pytest.raises(RuntimeError, match="Postprocessing failed closed"):
        train_entry._run_postprocessing_safely(
            _callback, "warning", run_dir=tmp_path,
        )

    status = __import__("json").loads(
        (tmp_path / "postprocess_status.json").read_text(encoding="utf-8")
    )
    assert status["overall"] == "failed"
    assert status["missing_mandatory_outputs"] == []


def test_reinterpreted_metrics_carry_both_operator_identities(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text('{"r": {}}', encoding="utf-8")
    (tmp_path / "fiber_metrics.json").write_text('{"fiber": {}}', encoding="utf-8")
    provenance = {"trained": {"psf_model": "mog"}, "evaluating": {"psf_model": "dgpsf"}}

    train_entry._annotate_metrics_operator_provenance(tmp_path, provenance)

    for name in ("metrics.json", "fiber_metrics.json"):
        payload = __import__("json").loads((tmp_path / name).read_text(encoding="utf-8"))
        assert payload["operator_provenance"] == provenance


def test_train_postprocessing_skips_point_sed_diagnostics_for_psf_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class _Metric:
        mse = 1.0
        mae = 2.0
        rmse = 3.0
        flux_error_pct = 4.0
        nrmse_pct = 5.0

    class _RecordingWandb:
        def log(self, payload, step: int | None = None) -> None:
            del payload, step

        def log_images(self, payload, step: int | None = None) -> None:
            del payload, step

        def log_artifacts(self, name: str, files) -> None:
            del name, files

    trainer = SimpleNamespace(
        held_out_band="i",
        history={"loss": [1.0]},
        psf_model="mog",
        use_flux_density=True,
        diagnostic_renderer=SimpleNamespace(
            grid=SimpleNamespace(lam=torch.tensor([5000.0], dtype=torch.float32))
        ),
        get_calibration_report=lambda: {},
        sample_point_spectra=lambda points, **kwargs: {},
        predict_fiber_spectra=lambda **kwargs: {},
    )
    fit_state = SimpleNamespace(steps_ran=17)
    image = np.ones((1, 1), dtype=np.float32)
    data_store = SimpleNamespace(
        exposures={0: SimpleNamespace(fiber_mask=np.ones((1, 1), dtype=np.float32))},
        image_band_by_exposure={0: "i"}, spec_diagnostics_by_exposure={},
    )

    def _mask_aperture_boom(*args, **kwargs) -> None:
        raise NotImplementedError(
            "Mask-aperture image diagnostics do not support PSF mixtures; "
            "use the analytic deterministic image renderer."
        )

    monkeypatch.setattr(
        train_entry, "_compute_image_diagnostic_predictions", lambda *args, **kwargs: {0: {"i": image}},
    )
    monkeypatch.setattr(
        train_entry.metrics_mod,
        "compute_metrics",
        lambda *args, **kwargs: {"e0:i": _Metric()},
    )
    monkeypatch.setattr(train_entry.metrics_mod, "compute_region_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_entry.metrics_mod, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "log_region_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.metrics_mod, "metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(train_entry.metrics_mod, "region_metrics_to_serializable", lambda metrics: {})
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks_by_exposure", lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        train_entry.visualization, "resolve_region_masks", lambda *args, **kwargs: ({}, None, 0, {}),
    )
    monkeypatch.setattr(train_entry.visualization, "plot_band_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_entry.visualization, "plot_true_band_with_fiber_mask", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_entry.visualization, "plot_holdout_region_overlay", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.visualization, "plot_region_cutout_grid", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.visualization, "plot_radial_profiles", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entry.diagnostics, "write_point_sed_diagnostics", _mask_aperture_boom)
    monkeypatch.setattr(
        train_entry.diagnostics, "write_fiber_spectrum_diagnostics", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_entry.diagnostics, "log_radial_diagnostics", lambda *args, **kwargs: None)

    train_entry._run_train_postprocessing(
        trainer=trainer, fit_state=fit_state, bandpasses={}, scene=SimpleNamespace(), data_store=data_store,
        observations={0: {"i": image}}, region_masks_by_exposure={}, primary_region=None,
        holdout_summary=None, run_dir=tmp_path, wandb_run=_RecordingWandb(), wandb_cfg={"log_artifacts": False},
        run_name="test_run", ckpt_path=tmp_path / "model.pt",
    )

    output = capsys.readouterr().out
    assert "Skipping point-SED diagnostics" in output
