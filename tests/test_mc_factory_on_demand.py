from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory as CountingKernelFactory, DummyWCS, ExposureValueModel, NoNormWCS, OffsetModel,
    SyntheticStore as MockStore, image_exposure, make_image_only_trainer,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import DatumBatch
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.distributed import DistributedContext


class StandardForwardModel(OffsetModel):
    def __init__(self) -> None:
        super().__init__(1.0)


def _exposure(value: float, *, wcs=None):
    active_wcs = DummyWCS(value, use_omega_x=True) if wcs is None else wcs
    return image_exposure(value, wcs=active_wcs)


def _make_trainer(
    *,
    kernel_samples_k: int = 4,
    replica_samples_k: int = 1,
    epochs: int = 1,
    pixel_batch_size: int = 1,
    include_kernel_factory: bool = True,
    trainer_overrides: dict | None = None,
    distributed_context: DistributedContext | None = None,
    model: torch.nn.Module | None = None,
):
    trainer_cfg = {
        "kernel_samples_k": int(kernel_samples_k),
        "replica_samples_k": int(replica_samples_k),
        "epochs": int(epochs),
        "learning_rate": 1e-3,
        "pixel_batch_size": int(pixel_batch_size),
        "spec_batch_size": 0,
        "apply_spatial_psf": False,
    }
    if trainer_overrides:
        trainer_cfg.update(trainer_overrides)
    return make_image_only_trainer(
        y_values=[0.0, 0.0, 0.0, 0.0],
        exposure_ids=[0, 1, 0, 1],
        datum_indices=[0, 0, 1, 1],
        exposure_values={0: 3.0, 1: 7.0},
        model=model or ExposureValueModel(zero_bias_grad=True),
        include_kernel_factory=include_kernel_factory,
        record_kernel_calls=True,
        trainer_cfg=trainer_cfg,
        distributed_context=distributed_context,
    )


def test_compute_mu_builds_fresh_kernel_samples_per_call() -> None:
    trainer, factory = _make_trainer()
    assert factory is not None
    idx_a = torch.tensor([0], dtype=torch.long, device=trainer.device)
    idx_b = torch.tensor([1], dtype=torch.long, device=trainer.device)

    trainer.compute_mu(idx_a)
    trainer.compute_mu(idx_b)

    assert factory.calls == 2
    assert factory.call_history == [[0], [1]]


def test_training_fit_smoke_uses_implicit_mc_loss() -> None:
    trainer, factory = _make_trainer(epochs=1, pixel_batch_size=1, trainer_overrides={"max_steps": 1},)
    assert factory is not None

    state = trainer.fit()

    assert state.history["loss"]
    assert np.isfinite(state.history["loss"][0])
    assert factory.calls == 2


def test_evaluate_model_samples_skips_autocast_when_amp_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, _ = _make_trainer()

    def _boom(*args, **kwargs):
        raise AssertionError("autocast should not be called when AMP is disabled")

    monkeypatch.setattr("survey_fusion.training.datum_trainer.torch.autocast", _boom)
    omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=trainer.device)
    loglam = torch.tensor([float(np.log(5500.0))], dtype=torch.float32, device=trainer.device)

    out = trainer._evaluate_model_samples(omega, loglam, exposure_id=0)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_evaluate_model_samples_wraps_model_forward_in_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, _ = _make_trainer()
    trainer.amp_enabled = True
    trainer.amp_device_type = "cuda"
    trainer.amp_dtype = torch.bfloat16
    trainer.amp_dtype_name = "bfloat16"
    calls: list[tuple[str, torch.dtype | None]] = []

    @contextmanager
    def _fake_autocast(*, device_type, dtype):
        calls.append((device_type, dtype))
        yield

    monkeypatch.setattr("survey_fusion.training.datum_trainer.torch.autocast", _fake_autocast)
    omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=trainer.device)
    loglam = torch.tensor([float(np.log(5500.0))], dtype=torch.float32, device=trainer.device)

    out = trainer._evaluate_model_samples(omega, loglam, exposure_id=0)

    assert out.shape == (1,)
    assert torch.isfinite(out).all()
    assert calls == [("cuda", torch.bfloat16)]


def test_replica_samples_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="replica_samples_k must be an integer >= 1"):
        _make_trainer(replica_samples_k=0)


def test_kernel_samples_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="kernel_samples_k must be a positive integer"):
        _make_trainer(kernel_samples_k=0)


def test_kernel_factory_is_required() -> None:
    with pytest.raises(ValueError, match="requires data.kernel_factory"):
        _make_trainer(include_kernel_factory=False)


def test_distributed_trainer_installs_distributed_samplers_and_sets_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _ = _make_trainer(
        epochs=1,
        pixel_batch_size=1,
        trainer_overrides={"max_steps": 1, "dataloader_shuffle": True},
        distributed_context=DistributedContext(
            enabled=True,
            backend="gloo",
            world_size=2,
            rank=0,
            local_rank=0,
            master_addr="127.0.0.1",
            master_port=29500,
        ),
        model=StandardForwardModel(),
    )
    assert trainer.image_sampler is not None
    assert trainer.spec_sampler is None

    calls: list[int] = []

    def _record_set_epoch(epoch: int) -> None:
        calls.append(int(epoch))

    monkeypatch.setattr(trainer.image_sampler, "set_epoch", _record_set_epoch)

    state = trainer.fit()

    assert state.history["loss"]
    assert calls == [0]


def test_distributed_trainer_requires_wcs_norm_support_for_training_exposures() -> None:
    device = torch.device("cpu")
    batch = DatumBatch(
        y=torch.zeros(2, dtype=torch.float32, device=device),
        sigma=torch.ones(2, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 1], dtype=torch.long, device=device),
        datum_kind=torch.tensor([0, 0], dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 1], dtype=torch.long, device=device),
        kernel_samples=None,
    )
    factory = CountingKernelFactory(batch)
    store = MockStore(
        exposures={0: _exposure(3.0), 1: _exposure(7.0, wcs=NoNormWCS(7.0)),},
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=factory,
        n_image=2,
        n_spec=0,
    )
    grid = build_spatial_grid(2, 2, device)
    with pytest.raises(ValueError, match="omega_xyz_to_norm_torch"):
        DatumTrainer(
            model=StandardForwardModel(),
            grid=grid,
            data=store,
            training_bands=["r"],
            held_out_band="i",
            lambda_step=20.0,
            trainer_cfg={
                "kernel_samples_k": 4,
                "replica_samples_k": 1,
                "epochs": 1,
                "learning_rate": 1e-3,
                "pixel_batch_size": 1,
                "spec_batch_size": 0,
                "apply_spatial_psf": False,
            },
            device=device,
            distributed_context=DistributedContext(
                enabled=True,
                backend="gloo",
                world_size=2,
                rank=0,
                local_rank=0,
                master_addr="127.0.0.1",
                master_port=29500,
            ),
        )


def test_distributed_trainer_rejects_non_divisible_dataset_without_drop_last() -> None:
    device = torch.device("cpu")
    batch = DatumBatch(
        y=torch.zeros(5, dtype=torch.float32, device=device),
        sigma=torch.ones(5, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long, device=device),
        datum_kind=torch.tensor([0, 0, 0, 0, 0], dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 0, 1, 1, 2], dtype=torch.long, device=device),
        kernel_samples=None,
    )
    factory = CountingKernelFactory(batch)
    store = MockStore(
        exposures={0: _exposure(3.0), 1: _exposure(7.0)},
        batch=batch,
        image_band_by_exposure={0: "r", 1: "r"},
        kernel_factory=factory,
        n_image=5,
        n_spec=0,
    )
    grid = build_spatial_grid(2, 2, device)
    with pytest.raises(ValueError, match="divisible by world_size"):
        DatumTrainer(
            model=StandardForwardModel(),
            grid=grid,
            data=store,
            training_bands=["r"],
            held_out_band="i",
            lambda_step=20.0,
            trainer_cfg={
                "kernel_samples_k": 4,
                "replica_samples_k": 1,
                "epochs": 1,
                "learning_rate": 1e-3,
                "pixel_batch_size": 2,
                "spec_batch_size": 0,
                "apply_spatial_psf": False,
                "dataloader_drop_last": False,
            },
            device=device,
            distributed_context=DistributedContext(
                enabled=True,
                backend="gloo",
                world_size=2,
                rank=0,
                local_rank=0,
                master_addr="127.0.0.1",
                master_port=29500,
            ),
        )


def test_reduce_step_logging_uses_fixed_timing_schema_when_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _ = _make_trainer(
        distributed_context=DistributedContext(
            enabled=True,
            backend="gloo",
            world_size=2,
            rank=0,
            local_rank=0,
            master_addr="127.0.0.1",
            master_port=29500,
        ),
        model=StandardForwardModel(),
    )
    tensor_lengths: list[int] = []

    def _fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        tensor_lengths.append(int(tensor.numel()))

    monkeypatch.setattr("survey_fusion.training.datum_trainer.is_distributed_active", lambda _ctx: True)
    monkeypatch.setattr("survey_fusion.training.datum_trainer.dist.all_reduce", _fake_all_reduce)

    stats, _ = trainer._reduce_step_logging(
        {
            "local_image_nll": 1.0,
            "local_spec_nll": 0.0,
            "local_total_nll": 1.0,
            "n_img": 1,
            "n_spec": 0,
            "calibration_prior": 0.0,
            "grad_norm_preclip": 0.5,
            "grad_norm_postclip": 0.25,
            "timings": {},
        },
        data_wait_sec=0.1,
        transfer_sec=0.0,
        train_step_sec=0.2,
        total_step_sec=0.3,
    )

    assert "timings" in stats
    assert stats["timings"] == {}
    assert 4 in tensor_lengths


def test_evaluate_model_samples_can_bypass_wrapped_model_for_diagnostics() -> None:
    trainer, _ = _make_trainer(model=StandardForwardModel())

    class _WrappedModel(torch.nn.Module):
        def forward(self, x, y, log_lam):
            raise AssertionError("wrapped model should not be used for diagnostic evaluation")

    base_calls: list[int] = []

    class _BaseModel(torch.nn.Module):
        def forward(self, x, y, log_lam):
            del y, log_lam
            base_calls.append(int(x.shape[0]))
            return x + 2.0

    trainer.model = _WrappedModel()
    trainer._base_model = lambda: _BaseModel()  # type: ignore[method-assign]

    omega = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=trainer.device)
    loglam = torch.tensor([float(np.log(5500.0))], dtype=torch.float32, device=trainer.device)

    out = trainer._evaluate_model_samples(omega, loglam, exposure_id=0, use_ddp=False)

    assert base_calls == [1]
    assert out.shape == (1,)
    assert torch.isfinite(out).all()
