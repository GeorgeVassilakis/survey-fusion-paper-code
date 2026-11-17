from __future__ import annotations

import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory, OffsetModel, SyntheticStore as MockStore, image_exposure, make_batch,
    spec_exposure,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import DatumKind
from survey_fusion.training.datum_trainer import DatumTrainer


class RecordingLogger:
    def __init__(self):
        self.records: list[tuple[int, dict[str, float]]] = []

    def log(self, payload, step: int):
        self.records.append((int(step), dict(payload)))


def _make_store(*, include_spec: bool, device: torch.device) -> MockStore:
    exposures = {0: image_exposure()}
    datum_kind = [int(DatumKind.IMAGE_PIXEL)]
    exposure_id = [0]
    datum_index = [0]
    y = [1.0]
    if include_spec:
        exposures[1] = spec_exposure()
        datum_kind.append(int(DatumKind.SPECTRAL_BIN))
        exposure_id.append(1)
        datum_index.append(0)
        y.append(1.0)
    batch = make_batch(
        y=torch.tensor(y, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(exposure_id, dtype=torch.long, device=device),
        datum_kind=torch.tensor(datum_kind, dtype=torch.long, device=device),
        datum_index=torch.tensor(datum_index, dtype=torch.long, device=device),
    )
    return MockStore(
        exposures=exposures, batch=batch, image_band_by_exposure={0: "r"},
        kernel_factory=ConstantKernelFactory(batch), n_image=1, n_spec=int(include_spec),
    )


def _trainer_cfg(
    *, epochs: int, lr: float, include_scheduler: bool, scheduler_enabled: bool,
    calibration_cfg: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg: dict[str, object] = {
        "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": int(epochs), "learning_rate": float(lr),
        "beta1": 0.9, "beta2": 0.999, "apply_spatial_psf": False, "pixel_batch_size": 1, "spec_batch_size": 1,
        "dataloader_shuffle": False,
    }
    if include_scheduler:
        cfg["lr_scheduler"] = {
            "enabled": bool(scheduler_enabled), "type": "ReduceLROnPlateau",
            "monitor": "epoch_mean_total_loss", "mode": "min", "factor": 0.5, "patience": 0, "threshold": 0.0,
            "threshold_mode": "rel", "cooldown": 0, "min_lr": 1e-5, "eps": 1e-8,
        }
    if calibration_cfg is not None:
        cfg["calibration"] = calibration_cfg
    return cfg


def _make_trainer(
    *, include_spec: bool, trainer_cfg: dict[str, object], wandb_logger: RecordingLogger | None = None,
) -> DatumTrainer:
    device = torch.device("cpu")
    store = _make_store(include_spec=include_spec, device=device)
    return DatumTrainer(
        model=OffsetModel(0.0), grid=build_spatial_grid(2, 2, device), data=store, training_bands=["r"],
        held_out_band="i", lambda_step=20.0, trainer_cfg=trainer_cfg, device=device,
        wandb_logger=wandb_logger,
    )


def test_scheduler_enabled_reduces_lr_and_logs_epoch_metric() -> None:
    logger = RecordingLogger()
    trainer = _make_trainer(
        include_spec=False,
        trainer_cfg=_trainer_cfg(epochs=3, lr=1e-1, include_scheduler=True, scheduler_enabled=True),
        wandb_logger=logger,
    )
    trainer.fit()

    final_lr = float(trainer.optimizer.param_groups[0]["lr"])
    assert final_lr == pytest.approx(2.5e-2)
    assert any("train/lr" in payload for _, payload in logger.records)
    assert any("train/epoch_mean_total_loss" in payload for _, payload in logger.records)


@pytest.mark.parametrize(("include_scheduler", "scheduler_enabled"), [(False, False), (True, False)])
def test_scheduler_missing_or_disabled_keeps_lr_constant(
    include_scheduler: bool, scheduler_enabled: bool,
) -> None:
    trainer = _make_trainer(
        include_spec=False,
        trainer_cfg=_trainer_cfg(
            epochs=4, lr=1e-1, include_scheduler=include_scheduler, scheduler_enabled=scheduler_enabled,
        ),
    )
    initial_lr = float(trainer.optimizer.param_groups[0]["lr"])
    trainer.fit()
    final_lr = float(trainer.optimizer.param_groups[0]["lr"])
    assert final_lr == pytest.approx(initial_lr)


@pytest.mark.parametrize(
    ("bad_cfg", "match"),
    [
        (
            {"enabled": True, "type": "reduce_on_plateau", "monitor": "epoch_mean_total_loss"},
            "type must be one of",
        ),
        (
            {"enabled": True, "type": "ReduceLROnPlateau", "monitor": "step_loss"},
            "monitor must be exactly 'epoch_mean_total_loss'",
        ),
        (
            {"enabled": True, "type": "ReduceLROnPlateau", "monitor": "epoch_mean_total_loss", "factor": 1.0},
            r"factor must be in \(0, 1\)",
        ),
    ],
)
def test_invalid_lr_scheduler_config_raises(bad_cfg: dict[str, object], match: str) -> None:
    cfg = _trainer_cfg(epochs=1, lr=1e-3, include_scheduler=False, scheduler_enabled=False)
    cfg["lr_scheduler"] = bad_cfg
    with pytest.raises(ValueError, match=match):
        _make_trainer(include_spec=False, trainer_cfg=cfg)


def test_scheduler_preserves_lr_ratio_across_param_groups() -> None:
    calibration_cfg = {
        "spectral_nuisance": {
            "enabled": True, "anchor": "image", "mapping": "throughput_mix", "prior_sigma_log": 0.15,
            "prior_weight": 0.0, "init_scale": 1.0, "lr_multiplier": 2.0, "apply_to_predictions": True,
        }
    }
    trainer = _make_trainer(
        include_spec=True,
        trainer_cfg=_trainer_cfg(
            epochs=2, lr=1e-1, include_scheduler=True, scheduler_enabled=True,
            calibration_cfg=calibration_cfg,
        ),
    )
    assert len(trainer.optimizer.param_groups) == 2
    initial_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]

    trainer.fit()
    final_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    assert final_lrs[0] == pytest.approx(initial_lrs[0] * 0.5)
    assert final_lrs[1] == pytest.approx(initial_lrs[1] * 0.5)
    assert (final_lrs[1] / final_lrs[0]) == pytest.approx(initial_lrs[1] / initial_lrs[0])


def test_onecycle_lr_steps_each_optimizer_update_and_logs_lr() -> None:
    logger = RecordingLogger()
    cfg = _trainer_cfg(epochs=4, lr=1e-1, include_scheduler=False, scheduler_enabled=False)
    cfg["lr_scheduler"] = {
        "enabled": True, "type": "OneCycleLR", "max_lr": 1e-1, "pct_start": 0.5, "anneal_strategy": "linear",
        "cycle_momentum": False, "div_factor": 10.0, "final_div_factor": 10.0,
    }
    trainer = _make_trainer(include_spec=False, trainer_cfg=cfg, wandb_logger=logger)
    initial_lr = float(trainer.optimizer.param_groups[0]["lr"])

    state = trainer.fit()

    logged_lrs = [float(payload["train/lr"]) for _, payload in logger.records if "train/lr" in payload]
    assert state.steps_ran == 4
    assert initial_lr == pytest.approx(1e-2)
    assert float(trainer.optimizer.param_groups[0]["lr"]) != pytest.approx(initial_lr)
    assert len({round(lr, 12) for lr in logged_lrs}) > 1


def test_onecycle_default_max_lr_preserves_param_group_ratio() -> None:
    calibration_cfg = {
        "spectral_nuisance": {
            "enabled": True, "anchor": "image", "mapping": "throughput_mix", "prior_sigma_log": 0.15,
            "prior_weight": 0.0, "init_scale": 1.0, "lr_multiplier": 2.0, "apply_to_predictions": True,
        }
    }
    cfg = _trainer_cfg(
        epochs=2, lr=1e-1, include_scheduler=False, scheduler_enabled=False, calibration_cfg=calibration_cfg,
    )
    cfg["lr_scheduler"] = {"enabled": True, "type": "OneCycleLR", "cycle_momentum": False}
    trainer = _make_trainer(include_spec=True, trainer_cfg=cfg)

    assert len(trainer.optimizer.param_groups) == 2
    max_lrs = [float(group["max_lr"]) for group in trainer.optimizer.param_groups]
    assert max_lrs[1] / max_lrs[0] == pytest.approx(2.0)


def test_fit_on_step_end_callback_receives_one_based_steps() -> None:
    trainer = _make_trainer(
        include_spec=False,
        trainer_cfg=_trainer_cfg(epochs=4, lr=1e-2, include_scheduler=False, scheduler_enabled=False),
    )
    callback_steps: list[int] = []

    def _on_step_end(step: int, payload: dict[str, float]) -> None:
        assert "total_loss" in payload
        callback_steps.append(int(step))

    state = trainer.fit(on_step_end=_on_step_end)
    assert callback_steps == list(range(1, int(state.steps_ran) + 1))
