from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

import survey_fusion.training.datum_trainer as datum_trainer_module
from support.synthetic_trainer import (
    ConstantKernelFactory, ExposureValueModel, SyntheticStore as MockStore, image_exposure, make_batch,
    make_image_spec_trainer, spec_exposure,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.models.spectral_field import GalaxySpectralField
from survey_fusion.pgm.contracts import DatumKind
from survey_fusion.training.checkpoints import (
    checkpoint_absolute_step,
    model_config_for_checkpoint,
    operator_config_for_trainer,
    validate_checkpoint_model_config,
)
from survey_fusion.training.datum_trainer import DatumTrainer


def _load_entrypoint_module(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_entry = _load_entrypoint_module("train.py", "survey_fusion_train_resume_entry")
eval_entry = train_entry
inspect_entry = _load_entrypoint_module(
    "../scripts/inspect_checkpoint.py", "survey_fusion_inspect_checkpoint_entry"
)


def test_checkpoint_absolute_step_uses_current_fresh_run_field() -> None:
    checkpoint = {
        "resume_meta": {
            "epochs_ran_this_invocation": 3,
            "steps_ran_this_invocation": 250,
            "saved_at_utc": "2026-07-26T00:00:00Z",
        }
    }

    assert checkpoint_absolute_step(
        checkpoint, checkpoint_path=Path("fresh.pt")
    ) == 250


def test_checkpoint_absolute_step_prefers_renamed_field_after_resume() -> None:
    checkpoint = {
        "resume_meta": {
            "global_step_completed": 1250,
            "steps_ran_this_invocation": 250,
        }
    }

    assert checkpoint_absolute_step(
        checkpoint, checkpoint_path=Path("resumed.pt")
    ) == 1250


def test_checkpoint_absolute_step_prefers_absolute_step_over_legacy_field() -> None:
    checkpoint = {
        "resume_meta": {
            "absolute_step": 1250,
            "steps_ran_this_invocation": 250,
        }
    }

    assert checkpoint_absolute_step(
        checkpoint, checkpoint_path=Path("resumed.pt")
    ) == 1250


def test_checkpoint_absolute_step_rejects_missing_step() -> None:
    checkpoint_path = Path("malformed.pt")
    checkpoint = {
        "resume_meta": {
            "epochs_ran_this_invocation": 3,
            "saved_at_utc": "2026-07-26T00:00:00Z",
        }
    }

    with pytest.raises(ValueError) as error:
        checkpoint_absolute_step(checkpoint, checkpoint_path=checkpoint_path)

    message = str(error.value)
    assert str(checkpoint_path) in message
    assert "epochs_ran_this_invocation" in message
    assert "saved_at_utc" in message


def _scheduler_cfg() -> dict[str, object]:
    return {
        "enabled": True, "type": "ReduceLROnPlateau", "monitor": "epoch_mean_total_loss", "mode": "min",
        "factor": 0.5, "patience": 0, "threshold": 0.0, "threshold_mode": "abs", "cooldown": 0,
        "min_lr": 1e-6, "eps": 1e-8,
    }


def _make_trainer(*, epochs: int, max_steps: int, include_scheduler: bool) -> DatumTrainer:
    trainer_cfg: dict[str, Any] = {
        "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": int(epochs), "max_steps": int(max_steps),
        "learning_rate": 1e-2, "beta1": 0.9, "beta2": 0.999, "apply_spatial_psf": False,
        "pixel_batch_size": 1, "spec_batch_size": 1, "dataloader_shuffle": False,
    }
    if include_scheduler:
        trainer_cfg["lr_scheduler"] = _scheduler_cfg()
    return make_image_spec_trainer(
        image_value=2.0, spec_value=3.0, model=ExposureValueModel(), trainer_cfg=trainer_cfg,
    )


def _make_mixed_batch_trainer(
    *, n_image: int, n_spec: int, image_batch_size: int, spec_batch_size: int,
) -> DatumTrainer:
    device = torch.device("cpu")
    total = int(n_image + n_spec)
    exposure_ids = [0] * int(n_image) + [1] * int(n_spec)
    datum_kinds = [int(DatumKind.IMAGE_PIXEL)] * int(n_image) + [int(DatumKind.SPECTRAL_BIN)] * int(n_spec)
    datum_index = list(range(int(n_image))) + list(range(int(n_spec)))
    batch = make_batch(
        y=torch.zeros(total, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(exposure_ids, dtype=torch.long, device=device),
        datum_kind=torch.tensor(datum_kinds, dtype=torch.long, device=device),
        datum_index=torch.tensor(datum_index, dtype=torch.long, device=device),
    )
    store = MockStore(
        exposures={0: image_exposure(2.0), 1: spec_exposure(3.0)}, batch=batch,
        image_band_by_exposure={0: "r"}, kernel_factory=ConstantKernelFactory(batch), n_image=int(n_image),
        n_spec=int(n_spec),
    )
    return DatumTrainer(
        model=ExposureValueModel(),
        grid=build_spatial_grid(2, 2, device),
        data=store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": 1, "max_steps": 0, "learning_rate": 1e-2,
            "beta1": 0.9, "beta2": 0.999, "apply_spatial_psf": False,
            "pixel_batch_size": int(image_batch_size), "spec_batch_size": int(spec_batch_size),
            "dataloader_shuffle": False,
        },
        device=device,
    )


def test_build_training_checkpoint_unwraps_ddp_model(tmp_path: Path) -> None:
    init_path = tmp_path / "ddp_init"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=0,
        world_size=1,
    )
    try:
        base_model = GalaxySpectralField(hidden_dim=8, n_layers=2)
        wrapped_model = torch.nn.parallel.DistributedDataParallel(base_model)
        optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)

        class _FakeTrainer:
            def __init__(self):
                self.model = wrapped_model
                self.optimizer = optimizer
                self.lr_scheduler = None

            def get_calibration_state_dict(self):
                return None

            def optimizer_state_dict(self):
                return self.optimizer.state_dict()

            def refine_state_dict(self):
                return None

            def recipe_record(self):
                return {"schema": 1}

        class _FakeFitState:
            epochs_ran = 1
            steps_ran = 2

        checkpoint = train_entry._build_training_checkpoint(_FakeTrainer(), _FakeFitState())

        expected_state = base_model.state_dict()
        assert checkpoint["model_config"] == {"output_head": "softplus"}
        assert set(checkpoint["model_state"].keys()) == set(expected_state.keys())
        for key, value in expected_state.items():
            assert torch.equal(checkpoint["model_state"][key], value)
    finally:
        dist.destroy_process_group()


def test_checkpoint_model_config_records_output_head() -> None:
    model = GalaxySpectralField(hidden_dim=8, n_layers=2, output_head="zero_centered_softplus")

    assert model_config_for_checkpoint(model) == {"output_head": "zero_centered_softplus"}


def test_checkpoint_model_config_rejects_output_head_mismatch(tmp_path: Path) -> None:
    model = GalaxySpectralField(hidden_dim=8, n_layers=2, output_head="zero_centered_softplus")
    checkpoint = {"model_config": {"output_head": "softplus"}}

    with pytest.raises(ValueError, match="output_head"):
        validate_checkpoint_model_config(
            model, checkpoint, checkpoint_path=tmp_path / "model.pt", context="Eval",
        )


def test_checkpoint_model_config_rejects_missing_model_config(tmp_path: Path) -> None:
    model = GalaxySpectralField(hidden_dim=8, n_layers=2, output_head="zero_centered_softplus")

    with pytest.raises(ValueError, match="model_config"):
        validate_checkpoint_model_config(
            model, {"model_state": model.state_dict()}, checkpoint_path=tmp_path / "legacy_model.pt",
            context="Eval",
        )


def test_resolve_distributed_cuda_device_uses_visible_ordinal_zero_for_single_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_entry.torch.cuda, "device_count", lambda: 1)
    device = train_entry._resolve_distributed_cuda_device(3)
    assert device == torch.device("cuda:0")


def test_resolve_distributed_cuda_device_rejects_out_of_range_local_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_entry.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="out of range"):
        train_entry._resolve_distributed_cuda_device(3)


def test_export_distributed_cuda_env_uses_effective_visible_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    train_entry._export_distributed_cuda_env(torch.device("cuda:0"))
    assert train_entry.os.environ["LOCAL_RANK"] == "0"


def test_wrap_model_for_distributed_disables_buffer_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeDDP:
        def __init__(self, model: torch.nn.Module, **kwargs: object) -> None:
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setattr(train_entry.torch.nn.parallel, "DistributedDataParallel", _FakeDDP)

    model = ExposureValueModel()
    wrapped = train_entry._wrap_model_for_distributed(model, torch.device("cuda:2"))

    assert isinstance(wrapped, _FakeDDP)
    assert captured["model"] is model
    assert captured["device_ids"] == [2]
    assert captured["output_device"] == 2
    assert captured["broadcast_buffers"] is False


def _assert_tensor_mapping_equal(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> None:
    assert lhs.keys() == rhs.keys()
    for key, lhs_tensor in lhs.items():
        rhs_tensor = rhs[key]
        assert torch.allclose(lhs_tensor, rhs_tensor)


def _assert_optimizer_state_equal(lhs: dict[str, object], rhs: dict[str, object]) -> None:
    assert lhs["param_groups"] == rhs["param_groups"]
    lhs_state = lhs["state"]
    rhs_state = rhs["state"]
    assert isinstance(lhs_state, dict)
    assert isinstance(rhs_state, dict)
    assert lhs_state.keys() == rhs_state.keys()
    for key in lhs_state.keys():
        lhs_entry = lhs_state[key]
        rhs_entry = rhs_state[key]
        assert isinstance(lhs_entry, dict)
        assert isinstance(rhs_entry, dict)
        assert lhs_entry.keys() == rhs_entry.keys()
        for field in lhs_entry.keys():
            lhs_value = lhs_entry[field]
            rhs_value = rhs_entry[field]
            if torch.is_tensor(lhs_value):
                assert torch.is_tensor(rhs_value)
                assert torch.allclose(lhs_value, rhs_value)
            else:
                assert rhs_value == lhs_value


def test_resume_checkpoint_roundtrip_loads_optimizer_and_scheduler_state(tmp_path: Path) -> None:
    trainer_a = _make_trainer(epochs=2, max_steps=0, include_scheduler=True)
    fit_state = trainer_a.fit()
    checkpoint = train_entry._build_training_checkpoint(trainer_a, fit_state)
    checkpoint_path = tmp_path / "resume_v3.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    trainer_b = _make_trainer(epochs=0, max_steps=0, include_scheduler=True)
    train_entry._load_resume_state_into_trainer(
        trainer_b, reloaded_checkpoint, checkpoint_path=checkpoint_path,
    )

    _assert_tensor_mapping_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict())
    _assert_optimizer_state_equal(trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict())
    assert trainer_a.lr_scheduler is not None
    assert trainer_b.lr_scheduler is not None
    assert trainer_b.lr_scheduler.state_dict() == trainer_a.lr_scheduler.state_dict()
    assert reloaded_checkpoint["resume_meta"]["global_step_completed"] == fit_state.steps_ran
    assert reloaded_checkpoint["resume_meta"]["steps_ran_this_invocation"] == fit_state.steps_ran


def test_resume_model_only_loads_model_and_skips_optimizer_state(tmp_path: Path) -> None:
    trainer_a = _make_trainer(epochs=2, max_steps=0, include_scheduler=False)
    fit_state = trainer_a.fit()
    checkpoint = train_entry._build_training_checkpoint(trainer_a, fit_state)
    checkpoint["optimizer_state"] = {"format": "hybrid_sparse_v1", "dense": {}, "splat": {}}
    checkpoint_path = tmp_path / "resume_model_only.pt"
    torch.save(checkpoint, checkpoint_path)
    reloaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    trainer_b = _make_trainer(epochs=0, max_steps=0, include_scheduler=False)
    fresh_optimizer_state = trainer_b.optimizer.state_dict()
    train_entry._load_resume_state_into_trainer(
        trainer_b, reloaded_checkpoint, checkpoint_path=checkpoint_path, model_only=True,
    )

    _assert_tensor_mapping_equal(trainer_a.model.state_dict(), trainer_b.model.state_dict())
    assert trainer_b.optimizer.state_dict() == fresh_optimizer_state


def test_load_checkpoint_rejects_missing_checkpoint_version(tmp_path: Path) -> None:
    trainer_a = _make_trainer(epochs=1, max_steps=0, include_scheduler=False)
    trainer_a.fit()
    checkpoint_path = tmp_path / "legacy_model.pt"
    torch.save(
        {
            "model_state": trainer_a.model.state_dict(),
            "model_config": model_config_for_checkpoint(trainer_a.model),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="checkpoint_version"):
        train_entry._load_checkpoint(checkpoint_path, torch.device("cpu"))


def test_resume_requires_optimizer_and_scheduler_state() -> None:
    trainer_a = _make_trainer(epochs=1, max_steps=0, include_scheduler=False)
    trainer_a.fit()
    malformed_checkpoint = {
        "checkpoint_version": 3,
        "model_config": model_config_for_checkpoint(trainer_a.model),
        "operator_config": operator_config_for_trainer(trainer_a),
        "model_state": trainer_a.model.state_dict(),
        "calibration_state": None,
        "optimizer_state": None,
        "lr_scheduler_state": None,
        "resume_meta": {
            "epochs_ran_this_invocation": 1,
            "steps_ran_this_invocation": 1,
            "saved_at_utc": "2026-03-02T00:00:00Z",
        },
    }

    trainer_b = _make_trainer(epochs=1, max_steps=1, include_scheduler=True)
    with pytest.raises(ValueError, match="optimizer_state"):
        train_entry._load_resume_state_into_trainer(
            trainer_b, malformed_checkpoint, checkpoint_path=Path("malformed_model.pt"),
        )

    malformed_checkpoint["optimizer_state"] = trainer_b.optimizer.state_dict()
    with pytest.raises(ValueError, match="lr_scheduler_state"):
        train_entry._load_resume_state_into_trainer(
            trainer_b, malformed_checkpoint, checkpoint_path=Path("malformed_model.pt"),
        )


def test_resume_epoch_budget_is_absolute() -> None:
    trainer_a = _make_trainer(epochs=2, max_steps=2, include_scheduler=False)
    fit_state_a = trainer_a.fit()
    checkpoint = train_entry._build_training_checkpoint(trainer_a, fit_state_a)

    trainer_b = _make_trainer(epochs=4, max_steps=0, include_scheduler=False)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("resume_model.pt"),
    )
    fit_state_b = trainer_b.fit()
    assert fit_state_b.epochs_ran == 2
    assert trainer_b.resume_start_step == 2
    assert fit_state_b.steps_ran == 4
    assert fit_state_b.steps_ran_this_invocation == 2


def test_resume_at_past_absolute_epoch_budget_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    trainer_a = _make_trainer(epochs=2, max_steps=2, include_scheduler=False)
    fit_state_a = trainer_a.fit()
    checkpoint = train_entry._build_training_checkpoint(trainer_a, fit_state_a)

    trainer_b = _make_trainer(epochs=1, max_steps=0, include_scheduler=False)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("resume_model.pt"),
    )
    callback_steps: list[int] = []
    fit_state_b = trainer_b.fit(
        on_step_end=lambda step, _stats: callback_steps.append(step)
    )

    assert fit_state_b.epochs_ran == 0
    assert fit_state_b.steps_ran == 2
    assert fit_state_b.steps_ran_this_invocation == 0
    assert callback_steps == []
    assert (
        "Training budget: target absolute step=1, current step=2, "
        "steps remaining this invocation=0"
    ) in capsys.readouterr().out


def test_legacy_resume_step_key_migrates_to_global_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer_a = _make_trainer(epochs=1, max_steps=0, include_scheduler=False)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100),
    )
    resume_meta = checkpoint["resume_meta"]
    del resume_meta["global_step_completed"]
    resume_meta["steps_ran_this_invocation"] = 123

    trainer_b = _make_trainer(epochs=0, max_steps=0, include_scheduler=False)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("legacy_step_model.pt"),
    )

    assert trainer_b.resume_start_step == 123
    output = capsys.readouterr().out
    assert "migrating legacy resume step field steps_ran_this_invocation" in output


def test_inspect_checkpoint_reports_global_step_and_operator_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "fixture.pt"
    torch.save(
        {
            "checkpoint_version": 4,
            "operator_config": {"psf_model": "gaussian", "pair_n_sigma": 8.0},
            "model_state": {"mu": torch.zeros((4, 2))},
            "resume_meta": {
                "global_step_completed": 17,
                "epochs_ran_this_invocation": 3,
                "saved_at_utc": "2026-08-06T12:00:00Z",
            },
        },
        checkpoint_path,
    )
    monkeypatch.setattr(sys, "argv", ["inspect_checkpoint.py", str(checkpoint_path)])

    inspect_entry.main()

    output = capsys.readouterr().out
    assert "global_step=17" in output
    assert "psf_model=gaussian" in output
    assert "mu_rows=4" in output


def test_iter_epoch_batches_cycles_shorter_loader_when_both_modalities_present() -> None:
    trainer = _make_mixed_batch_trainer(n_image=4, n_spec=6, image_batch_size=2, spec_batch_size=2)

    batches = list(trainer._iter_epoch_batches())

    assert trainer._steps_per_epoch() == 3
    assert len(batches) == 3
    assert all(img is not None for img, _ in batches)
    assert all(spec is not None for _, spec in batches)
    assert [int(img.numel()) for img, _ in batches] == [2, 2, 2]
    assert [int(spec.numel()) for _, spec in batches] == [2, 2, 2]
    assert torch.equal(batches[2][0], batches[0][0])


def test_log_speed_participates_in_distributed_collective_without_local_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _make_trainer(epochs=1, max_steps=1, include_scheduler=False)
    trainer._speed_profile_enabled = True
    trainer.speed_logger = None
    trainer.device = torch.device("cuda", 0)

    original_tensor = torch.tensor
    all_reduce_calls: list[torch.Tensor] = []

    def _cpu_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    def _record_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        del op
        all_reduce_calls.append(tensor.clone())

    monkeypatch.setattr(datum_trainer_module.torch, "tensor", _cpu_tensor)
    monkeypatch.setattr(datum_trainer_module.torch.cuda, "memory_allocated", lambda _device: 1.0)
    monkeypatch.setattr(datum_trainer_module.torch.cuda, "memory_reserved", lambda _device: 2.0)
    monkeypatch.setattr(datum_trainer_module.torch.cuda, "max_memory_allocated", lambda _device: 3.0)
    monkeypatch.setattr(datum_trainer_module.torch.cuda, "max_memory_reserved", lambda _device: 4.0)
    monkeypatch.setattr(datum_trainer_module, "is_distributed_active", lambda _ctx: True)
    monkeypatch.setattr(datum_trainer_module.dist, "all_reduce", _record_all_reduce)

    trainer._log_speed(
        epoch=0, global_step=1, stats={}, data_wait_sec=0.1, transfer_sec=0.2, train_step_sec=0.3,
        total_step_sec=0.6,
    )

    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0].shape == (4,)


def test_unified_eval_can_load_v3_checkpoint_model_state(tmp_path: Path) -> None:
    model = GalaxySpectralField(hidden_dim=16, n_layers=2, softplus_beta=2.0, positional_encoding=None)
    checkpoint_path = tmp_path / "eval_v3_model.pt"
    checkpoint = {
        "checkpoint_version": 3,
        "model_config": model_config_for_checkpoint(model),
        "model_state": model.state_dict(),
        "calibration_state": None,
        "optimizer_state": {},
        "lr_scheduler_state": None,
        "resume_meta": {
            "epochs_ran_this_invocation": 1,
            "steps_ran_this_invocation": 1,
            "saved_at_utc": "2026-03-02T00:00:00Z",
        },
    }
    torch.save(checkpoint, checkpoint_path)

    loaded = eval_entry._load_checkpoint(checkpoint_path, torch.device("cpu"))
    rehydrated = GalaxySpectralField(hidden_dim=16, n_layers=2, softplus_beta=2.0, positional_encoding=None)
    rehydrated.load_state_dict(loaded["model_state"])
    _assert_tensor_mapping_equal(model.state_dict(), rehydrated.state_dict())


def _make_refine_splat_trainer(
    n_splats: int, *, sed_off_grid: str = "clamp", sed_pin_gauge_knots: bool = False,
) -> DatumTrainer:
    from support.synthetic_trainer import make_image_spec_trainer
    from survey_fusion.models.splat_field import SplatField

    torch.manual_seed(3)
    field = SplatField(
        n_splats=n_splats, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4),
        sed_off_grid=sed_off_grid, sed_pin_gauge_knots=sed_pin_gauge_knots,
    )
    return make_image_spec_trainer(
        model=field,
        trainer_cfg={
            "refine": {"enabled": True, "start": 0, "stop": 15_000, "every": 100}
        },
    )


def test_refine_state_persists_across_resume() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    strategy_a = trainer_a.refine_strategy
    assert strategy_a is not None
    strategy_a.grad_accum = torch.linspace(0.1, 0.8, 8)
    strategy_a.count = torch.ones(8)
    trainer_a._post_optimizer_step(100)
    assert strategy_a.threshold is not None
    n_after = int(trainer_a._base_model().mu.shape[0])
    pending_accum = torch.full((n_after,), 0.25)
    strategy_a.grad_accum = pending_accum.clone()
    strategy_a.count = torch.ones(n_after)

    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )
    assert checkpoint["refine_state"]["threshold"] == pytest.approx(
        strategy_a.threshold
    )

    trainer_b = _make_refine_splat_trainer(n_splats=n_after)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("refine_model.pt")
    )
    strategy_b = trainer_b.refine_strategy
    assert strategy_b.threshold == pytest.approx(strategy_a.threshold)
    assert torch.equal(strategy_b.grad_accum, pending_accum)
    assert torch.equal(strategy_b.count, torch.ones(n_after))


def test_resume_without_refine_state_warns_but_loads() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=1)
    )
    del checkpoint["refine_state"]

    trainer_b = _make_refine_splat_trainer(n_splats=8)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("legacy_model.pt")
    )
    assert trainer_b.refine_strategy.threshold is None


def test_refine_state_rejects_splat_count_mismatch() -> None:
    trainer = _make_refine_splat_trainer(n_splats=8)
    with pytest.raises(ValueError, match="8 splats"):
        trainer.load_refine_state_dict(
            {
                "threshold": 0.5,
                "grad_accum": torch.ones(5),
                "count": torch.ones(5),
            }
        )


def test_splat_checkpoint_records_structural_provenance() -> None:
    trainer = _make_refine_splat_trainer(n_splats=8)
    assert model_config_for_checkpoint(trainer.model) == {
        "name": "splat",
        "n_splats": 8,
        "n_templates": 2,
        "n_loglam_grid": 4,
        "sed_off_grid": "clamp",
        "sed_pin_gauge_knots": False,
        "sed_pad_knots_blue": 0,
        "sed_pad_knots_red": 0,
    }


@pytest.mark.parametrize(
    ("saved_off_grid", "saved_pin", "live_off_grid", "live_pin"),
    [("zero", True, "clamp", False), ("clamp", False, "zero", True)],
)
def test_splat_resume_rejects_sed_operator_reinterpretation(
    saved_off_grid: str, saved_pin: bool, live_off_grid: str, live_pin: bool,
) -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(
        n_splats=8, sed_off_grid=saved_off_grid, sed_pin_gauge_knots=saved_pin,
    )
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=1)
    )
    trainer_b = _make_refine_splat_trainer(
        n_splats=8, sed_off_grid=live_off_grid, sed_pin_gauge_knots=live_pin,
    )

    with pytest.raises(ValueError, match="sed_off_grid.*sed_pin_gauge_knots"):
        train_entry._load_resume_state_into_trainer(
            trainer_b, checkpoint, checkpoint_path=Path("sed_mismatch.pt")
        )


def test_legacy_splat_checkpoint_defaults_are_warned_and_nondefault_is_rejected() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=1)
    )
    del checkpoint["model_config"]["sed_off_grid"]
    del checkpoint["model_config"]["sed_pin_gauge_knots"]

    trainer_default = _make_refine_splat_trainer(n_splats=8)
    with pytest.warns(RuntimeWarning, match="sed_off_grid.*sed_pin_gauge_knots"):
        train_entry._load_resume_state_into_trainer(
            trainer_default, checkpoint, checkpoint_path=Path("legacy_sed.pt")
        )

    trainer_nondefault = _make_refine_splat_trainer(
        n_splats=8, sed_off_grid="zero", sed_pin_gauge_knots=True,
    )
    with pytest.raises(ValueError, match="sed_off_grid.*sed_pin_gauge_knots"):
        train_entry._load_resume_state_into_trainer(
            trainer_nondefault, checkpoint, checkpoint_path=Path("legacy_sed.pt")
        )


def test_splat_resume_rejects_structural_mismatch() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=1)
    )
    checkpoint["model_config"]["n_templates"] = 7
    trainer_b = _make_refine_splat_trainer(n_splats=8)
    with pytest.raises(ValueError, match="n_templates"):
        train_entry._load_resume_state_into_trainer(
            trainer_b, checkpoint, checkpoint_path=Path("mismatch_model.pt")
        )


def test_splat_resume_accepts_legacy_checkpoint_without_provenance() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=1)
    )
    checkpoint["model_config"] = {}
    trainer_b = _make_refine_splat_trainer(n_splats=8)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("legacy_model.pt")
    )


def test_v3_checkpoint_migrates_lineage_buffers_on_resume() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )
    assert checkpoint["checkpoint_version"] == train_entry.CURRENT_CHECKPOINT_VERSION
    # forge a pre-lineage v3 checkpoint
    checkpoint["checkpoint_version"] = 3
    for name in train_entry.SPLAT_LINEAGE_BUFFER_DEFAULTS:
        assert name in checkpoint["model_state"]
        del checkpoint["model_state"][name]

    trainer_b = _make_refine_splat_trainer(n_splats=8)
    with torch.no_grad():
        trainer_b._base_model().halving_depth.fill_(7)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("v3_model.pt")
    )
    model = trainer_b._base_model()
    assert torch.equal(model.halving_depth, torch.zeros(8, dtype=torch.uint8))
    assert torch.equal(model.last_halving_event, torch.full((8,), -1, dtype=torch.int32))
    assert torch.equal(model.birth_event, torch.full((8,), -1, dtype=torch.int32))


def test_v4_checkpoint_round_trips_lineage_buffers() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    model_a = trainer_a._base_model()
    with torch.no_grad():
        model_a.halving_depth[2] = 3
        model_a.last_halving_event[2] = 4200
        model_a.birth_event[5] = 1700
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )

    trainer_b = _make_refine_splat_trainer(n_splats=8)
    train_entry._load_resume_state_into_trainer(
        trainer_b, checkpoint, checkpoint_path=Path("v4_model.pt")
    )
    model_b = trainer_b._base_model()
    assert torch.equal(model_b.halving_depth, model_a.halving_depth)
    assert torch.equal(model_b.last_halving_event, model_a.last_halving_event)
    assert torch.equal(model_b.birth_event, model_a.birth_event)


def test_unsupported_checkpoint_version_is_rejected() -> None:
    from types import SimpleNamespace

    trainer = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )
    checkpoint["checkpoint_version"] = 2
    with pytest.raises(ValueError, match="checkpoint_version"):
        train_entry._load_resume_state_into_trainer(
            _make_refine_splat_trainer(n_splats=8),
            checkpoint,
            checkpoint_path=Path("v2_model.pt"),
        )


def test_v4_checkpoint_missing_lineage_buffer_is_rejected() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )
    del checkpoint["model_state"]["halving_depth"]

    with pytest.raises(RuntimeError, match="halving_depth"):
        train_entry._load_resume_state_into_trainer(
            _make_refine_splat_trainer(n_splats=8),
            checkpoint,
            checkpoint_path=Path("corrupt_v4_model.pt"),
        )


def test_resume_rejects_silent_pair_support_change() -> None:
    from types import SimpleNamespace

    trainer_a = _make_refine_splat_trainer(n_splats=8)
    checkpoint = train_entry._build_training_checkpoint(
        trainer_a, SimpleNamespace(epochs_ran=1, steps_ran=100)
    )
    assert checkpoint["operator_config"]["pair_n_sigma"] == 8.0
    assert checkpoint["operator_config"]["psf_model"] == "gaussian"
    assert checkpoint["operator_config"]["psf_mixture_digest"] is None
    # a RECORDED width mismatch is a hard error
    checkpoint["operator_config"]["pair_n_sigma"] = 3.5
    with pytest.raises(ValueError, match="pair_n_sigma"):
        train_entry._load_resume_state_into_trainer(
            _make_refine_splat_trainer(n_splats=8),
            checkpoint,
            checkpoint_path=Path("legacy_model.pt"),
        )
    # the change is legal only when requested explicitly
    train_entry._load_resume_state_into_trainer(
        _make_refine_splat_trainer(n_splats=8),
        checkpoint,
        checkpoint_path=Path("legacy_model.pt"),
        allow_operator_change=True,
    )
    # Absent provenance is UNKNOWN (some runs varied the width through env
    # knobs), so schema v2 fails closed unless the user accepts the unknown
    # operator explicitly.
    del checkpoint["operator_config"]
    with pytest.raises(ValueError, match="incomplete operator identity"):
        train_entry._load_resume_state_into_trainer(
            _make_refine_splat_trainer(n_splats=8),
            checkpoint,
            checkpoint_path=Path("legacy_model.pt"),
        )
    train_entry._load_resume_state_into_trainer(
        _make_refine_splat_trainer(n_splats=8),
        checkpoint,
        checkpoint_path=Path("legacy_model.pt"),
        allow_operator_change=True,
    )


def test_model_config_records_grid_padding() -> None:
    from survey_fusion.models.splat_field import SplatField

    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16),
        sed_pad_knots_blue=5, sed_pad_knots_red=2,
    )
    config = model_config_for_checkpoint(field)
    assert config["n_loglam_grid"] == 16
    assert config["sed_pad_knots_blue"] == 5
    assert config["sed_pad_knots_red"] == 2
    assert field.sed.loglam_grid.numel() == 23


def test_validate_treats_absent_pad_keys_as_unpadded() -> None:
    from survey_fusion.models.splat_field import SplatField

    unpadded = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16),
    )
    saved = dict(model_config_for_checkpoint(unpadded))
    del saved["sed_pad_knots_blue"]
    del saved["sed_pad_knots_red"]
    validate_checkpoint_model_config(
        unpadded, {"model_config": saved},
        checkpoint_path=Path("prepad.pt"), context="test",
    )

    padded = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16),
        sed_pad_knots_blue=5, sed_pad_knots_red=2,
    )
    with pytest.raises(ValueError, match="sed_pad_knots_blue"):
        validate_checkpoint_model_config(
            padded, {"model_config": saved},
            checkpoint_path=Path("prepad.pt"), context="test",
        )


def test_validate_accepts_matching_and_rejects_mismatched_padding() -> None:
    from survey_fusion.models.splat_field import SplatField

    padded = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16),
        sed_pad_knots_blue=5, sed_pad_knots_red=2,
    )
    saved = model_config_for_checkpoint(padded)
    validate_checkpoint_model_config(
        padded, {"model_config": saved},
        checkpoint_path=Path("padded.pt"), context="test",
    )

    other = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16),
        sed_pad_knots_blue=5, sed_pad_knots_red=3,
    )
    with pytest.raises(ValueError, match="sed_pad_knots_red"):
        validate_checkpoint_model_config(
            other, {"model_config": saved},
            checkpoint_path=Path("padded.pt"), context="test",
        )
