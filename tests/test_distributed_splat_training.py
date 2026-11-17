from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train as train_entry
from support.synthetic_trainer import image_exposure, make_image_only_trainer
from survey_fusion.models.spectral_field import GalaxySpectralField
from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import SplatRefineConfig, SplatRefineStrategy
from survey_fusion.training import datum_trainer as datum_trainer_module
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.distributed import DistributedContext


def _field(n_splats: int = 8) -> SplatField:
    torch.manual_seed(17)
    return SplatField(
        n_splats=n_splats,
        n_templates=2,
        loglam_grid=torch.linspace(-1.0, 1.0, 4),
        init_scale=0.01,
    )


def _distributed_context(rank: int = 0) -> DistributedContext:
    return DistributedContext(
        enabled=True,
        backend="gloo",
        world_size=2,
        rank=rank,
        local_rank=rank,
    )


def _bare_splat_trainer(field: SplatField, *, rank: int = 0) -> DatumTrainer:
    trainer = DatumTrainer.__new__(DatumTrainer)
    trainer.model = field
    trainer.device = torch.device("cpu")
    trainer.distributed_context = _distributed_context(rank)
    trainer.distributed_enabled = True
    trainer.rank = rank
    trainer.world_size = 2
    trainer.spectral_calibrator = None
    trainer.splat_optimizer = None
    trainer.optimizer = torch.optim.Adam(field.param_groups_gsplat())
    trainer.params = list(field.parameters())
    trainer._position_group = next(
        group
        for group in trainer.optimizer.param_groups
        if any(parameter is field.mu for parameter in group["params"])
    )
    return trainer


def _actual_global_batch_trainer(
    distributed_context: DistributedContext | None,
) -> DatumTrainer:
    trainer, _ = make_image_only_trainer(
        y_values=[0.0, 2.0, 0.0, 2.0],
        exposure_ids=[0, 0, 0, 0],
        datum_indices=[0, 0, 0, 0],
        exposure_values={0: 0.0},
        exposure_overrides={
            0: image_exposure(
                0.0, psf_fwhm_arcsec=1.4, height=2, width=2
            )
        },
        model=_field(),
        distributed_context=distributed_context,
        trainer_cfg={
            "analytic_images": True,
            "analytic_spectra": True,
            "apply_spatial_psf": False,
            "dataloader_shuffle": False,
            "epochs": 3,
            "max_steps": 3,
            "pixel_batch_size": 2 if distributed_context is not None else 4,
            "spec_batch_size": 0,
            "grad_clip": 1e9,
            "refine": {
                "enabled": True,
                "grow_grad2d": 0.0,
                "grow_target_frac": 0.25,
                "grow_scale": 1.0,
                "prune_peak_sb_frac": 0.0,
                "start": 0,
                "stop": 10,
                "every": 3,
            },
        },
    )
    return trainer


def _spawn_or_skip(target, args: tuple[object, ...]) -> None:
    try:
        mp.spawn(target, args=args, nprocs=2, join=True)
    except Exception as error:
        message = str(error).lower()
        if "gloo" in message and (
            "cannot resolve" in message
            or "connectfullmesh" in message
            or "nodename nor servname" in message
        ):
            pytest.skip(f"Gloo is unavailable in this sandbox: {error}")
        raise


def _global_batch_worker(rank: int, init_path: str, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        trainer = _actual_global_batch_trainer(_distributed_context(rank))
        assert trainer.refine_strategy is not None
        original_make_plan = trainer.refine_strategy.make_plan
        plan_calls = 0

        def counted_make_plan(model, step):
            nonlocal plan_calls
            plan_calls += 1
            return original_make_plan(model, step)

        trainer.refine_strategy.make_plan = counted_make_plan
        trainer.fit()
        torch.save(
            {
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in trainer._base_model().state_dict().items()
                },
                "threshold": trainer.refine_strategy.threshold,
                "plan_calls": plan_calls,
                "signature": trainer._distributed_state_signature(),
            },
            Path(output_dir) / f"global-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _synchronized_refine_worker(
    rank: int, init_path: str, output_dir: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        field = _field()
        with torch.no_grad():
            field.raw_flux[1] = -100.0
            field.log_scales[0] = math.log(0.001)
            field.log_scales[2] = math.log(0.02)
        trainer = _bare_splat_trainer(field, rank=rank)
        trainer.refine_strategy = SplatRefineStrategy(
            SplatRefineConfig(
                grow_grad2d=0.5,
                grow_scale=0.005,
                prune_peak_sb_frac=0.005,
                refine_start=0,
                refine_stop=10,
                refine_every=1,
            ),
            field_width=100,
            reference_sigma=1.0,
        )
        initial_loss = sum(
            parameter.sum() * float(1 + 2 * rank)
            for parameter in field.parameters()
        )
        initial_loss.backward()
        trainer._sync_model_gradients()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.refine_strategy.grad_accum = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        trainer.refine_strategy.count = torch.ones(8)
        original_make_plan = trainer.refine_strategy.make_plan
        plan_calls = 0

        def counted_make_plan(model, step):
            nonlocal plan_calls
            plan_calls += 1
            return original_make_plan(model, step)

        trainer.refine_strategy.make_plan = counted_make_plan
        report = trainer._distributed_refine(global_step=1)
        continuation_loss = sum(
            parameter.sum() * float(3 - 2 * rank)
            for parameter in field.parameters()
        )
        continuation_loss.backward()
        trainer._sync_model_gradients()
        trainer.optimizer.step()
        signature = trainer._verify_distributed_state_signatures("continuation")
        torch.save(
            {
                "report": report,
                "signature": signature,
                "plan_calls": plan_calls,
            },
            Path(output_dir) / f"refine-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _signature_tripwire_worker(
    rank: int, init_path: str, output_dir: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        field = _field()
        trainer = _bare_splat_trainer(field, rank=rank)
        if rank == 1:
            with torch.no_grad():
                field.mu[0, 0] = torch.nextafter(
                    field.mu[0, 0],
                    torch.tensor(math.inf, dtype=field.mu.dtype),
                )
        error = None
        try:
            trainer._verify_distributed_state_signatures("initialization")
        except RuntimeError as caught:
            error = str(caught)
        torch.save(error, Path(output_dir) / f"tripwire-rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


class _OneParameterCalibrator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1))


def _calibrator_sync_worker(
    rank: int, init_path: str, output_dir: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        trainer = _bare_splat_trainer(_field(), rank=rank)
        trainer.spectral_calibrator = _OneParameterCalibrator()
        parameter = next(trainer.spectral_calibrator.parameters())
        if rank == 0:
            parameter.grad = torch.full_like(parameter, 2.0)
        trainer._sync_spectral_calibrator_grads()
        torch.save(
            parameter.grad.detach().cpu().clone(),
            Path(output_dir) / f"calibrator-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_wrap_policy_excludes_only_splat_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _field()
    spectral = GalaxySpectralField(hidden_dim=8, n_layers=2)

    assert train_entry._should_wrap_model_for_distributed(field) is False
    assert train_entry._should_wrap_model_for_distributed(spectral) is True

    wrapped: list[torch.nn.Module] = []

    class FakeDDP:
        def __init__(self, model: torch.nn.Module, **_kwargs: object) -> None:
            wrapped.append(model)

    monkeypatch.setattr(
        train_entry.torch.nn.parallel, "DistributedDataParallel", FakeDDP
    )
    for model in (field, spectral):
        if train_entry._should_wrap_model_for_distributed(model):
            train_entry._wrap_model_for_distributed(model, torch.device("cpu"))
    assert wrapped == [spectral]


def test_dense_splat_gradient_sync_reduces_every_parameter_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _field()
    trainer = _bare_splat_trainer(field)
    parameters = list(field.parameters())
    parameters[0].grad = torch.full_like(parameters[0], 2.0)
    parameters[1].grad = None
    calls: list[tuple[int, ...]] = []

    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )

    def fake_all_reduce(value: torch.Tensor, *, op=None) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        calls.append(tuple(value.shape))
        value.add_(4.0)

    monkeypatch.setattr(datum_trainer_module.dist, "all_reduce", fake_all_reduce)

    trainer._sync_model_gradients()

    assert calls == [tuple(parameter.shape) for parameter in parameters]
    assert torch.equal(parameters[0].grad, torch.full_like(parameters[0], 3.0))
    assert parameters[1].grad is not None
    assert torch.equal(parameters[1].grad, torch.full_like(parameters[1], 2.0))
    assert all(parameter.grad is not None for parameter in parameters)


def test_distributed_splat_gradient_sync_rejects_sparse_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _field()
    trainer = _bare_splat_trainer(field)
    field.mu.grad = torch.ones_like(field.mu).to_sparse()
    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )

    with pytest.raises(ValueError, match="dense gradients"):
        trainer._sync_model_gradients()


def test_single_process_splat_gradient_sync_never_enters_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _field()
    trainer = _bare_splat_trainer(field)
    trainer.distributed_context = DistributedContext()
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_reduce",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-process training entered a collective")
        ),
    )

    trainer._sync_model_gradients()


def test_distributed_refinement_is_allowed_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )

    trainer, _ = make_image_only_trainer(
        model=_field(),
        distributed_context=_distributed_context(),
        trainer_cfg={
            "analytic_images": True,
            "analytic_spectra": True,
            "refine": {"enabled": True, "start": 0, "stop": 10, "every": 1},
        },
    )

    assert trainer.refine_strategy is not None


def test_distributed_sparse_splat_gradients_fail_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )

    with pytest.raises(
        ValueError, match="distributed splat training requires dense gradients"
    ):
        make_image_only_trainer(
            model=_field(),
            distributed_context=_distributed_context(),
            trainer_cfg={
                "analytic_images": True,
                "analytic_spectra": True,
                "sparse_splat_gradients": True,
            },
        )


def test_state_signature_detects_one_ulp_parameter_change() -> None:
    field = _field()
    trainer = _bare_splat_trainer(field)
    before = trainer._distributed_state_signature()
    with torch.no_grad():
        field.mu[0, 0] = torch.nextafter(
            field.mu[0, 0], torch.tensor(math.inf, dtype=field.mu.dtype)
        )
    after = trainer._distributed_state_signature()

    assert len(before) == 64
    assert before != after


def test_calibrator_sync_materializes_missing_gradient_before_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _bare_splat_trainer(_field())
    trainer.spectral_calibrator = _OneParameterCalibrator()
    parameter = next(trainer.spectral_calibrator.parameters())
    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )

    def fake_all_reduce(value: torch.Tensor, *, op=None) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        value.add_(2.0)

    monkeypatch.setattr(datum_trainer_module.dist, "all_reduce", fake_all_reduce)

    trainer._sync_spectral_calibrator_grads()

    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_distributed_refine_broadcasts_one_plan_and_refreshes_surgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _field()
    with torch.no_grad():
        field.raw_flux[1] = -100.0
    trainer = _bare_splat_trainer(field)
    strategy = SplatRefineStrategy(
        SplatRefineConfig(
            grow_grad2d=0.5,
            grow_scale=1.0,
            prune_peak_sb_frac=0.005,
            refine_start=0,
            refine_stop=10,
            refine_every=1,
        ),
        field_width=100,
        reference_sigma=1.0,
    )
    strategy.grad_accum = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    strategy.count = torch.ones(8)
    trainer.refine_strategy = strategy
    broadcasts: list[object] = []

    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "broadcast_object_list",
        lambda objects, *, src: broadcasts.append(objects[0]),
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_gather_object",
        lambda outputs, value: outputs.__setitem__(slice(None), [value, value]),
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_reduce",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("refinement statistics must not be reduced")
        ),
    )

    report = trainer._distributed_refine(global_step=1)

    assert report == {"dupli": 1, "split": 0, "prune": 1, "n_splats": 8}
    assert len(broadcasts) == 1
    assert {id(parameter) for parameter in trainer.params} == {
        id(parameter) for parameter in field.parameters()
    }
    assert any(
        parameter is field.mu for parameter in trainer._position_group["params"]
    )


def test_nonroot_distributed_refine_never_calls_make_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_field = _field()
    root_strategy = SplatRefineStrategy(
        SplatRefineConfig(
            grow_grad2d=0.5,
            grow_scale=1.0,
            prune_peak_sb_frac=0.0,
            refine_start=0,
            refine_stop=10,
            refine_every=1,
        ),
        field_width=100,
        reference_sigma=1.0,
    )
    root_strategy.grad_accum = torch.ones(8)
    root_strategy.count = torch.ones(8)
    root_plan = root_strategy.make_plan(root_field, 1)
    assert root_plan is not None
    envelope = {"error": None, "plan": root_plan.to_payload()}

    trainer = _bare_splat_trainer(_field(), rank=1)
    trainer.refine_strategy = SplatRefineStrategy(
        root_strategy.config,
        field_width=100,
        reference_sigma=1.0,
    )

    def fail_if_planned(*_args, **_kwargs):
        raise AssertionError("non-root rank called make_plan")

    trainer.refine_strategy.make_plan = fail_if_planned
    monkeypatch.setattr(
        datum_trainer_module, "is_distributed_active", lambda _context: True
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "broadcast_object_list",
        lambda objects, *, src: objects.__setitem__(0, envelope),
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_gather_object",
        lambda outputs, value: outputs.__setitem__(slice(None), [value, value]),
    )

    trainer._distributed_refine(global_step=1)


def test_distributed_refine_off_cadence_uses_no_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _bare_splat_trainer(_field())
    trainer.refine_strategy = SplatRefineStrategy(
        SplatRefineConfig(refine_start=0, refine_stop=10, refine_every=3),
        field_width=100,
        reference_sigma=1.0,
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("off-cadence refinement entered a collective")
        ),
    )
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("off-cadence refinement entered a collective")
        ),
    )

    assert trainer._distributed_refine(global_step=2) == {}


@pytest.fixture(scope="module")
def global_batch_results(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("global-batch")
    _spawn_or_skip(
        _global_batch_worker,
        (str(output_dir / "init"), str(output_dir)),
    )
    rank_zero = torch.load(
        output_dir / "global-rank-0.pt", weights_only=False
    )
    rank_one = torch.load(
        output_dir / "global-rank-1.pt", weights_only=False
    )
    single = _actual_global_batch_trainer(None)
    assert single.refine_strategy is not None
    single.fit()
    single_result = {
        "model": {
            name: value.detach().cpu().clone()
            for name, value in single._base_model().state_dict().items()
        },
        "threshold": single.refine_strategy.threshold,
    }
    return rank_zero, rank_one, single_result


def test_two_rank_global_batch_gradient_updates_match_single_process_exactly(
    global_batch_results,
) -> None:
    rank_zero, rank_one, single = global_batch_results
    assert rank_zero["model"].keys() == rank_one["model"].keys()
    assert rank_zero["model"].keys() == single["model"].keys()
    for name in rank_zero["model"]:
        assert torch.equal(rank_zero["model"][name], rank_one["model"][name])
        assert torch.equal(rank_zero["model"][name], single["model"][name])
    assert rank_zero["threshold"] == rank_one["threshold"]
    assert rank_zero["threshold"] == single["threshold"]


def test_two_rank_refinement_planning_occurs_only_on_rank_zero(
    global_batch_results,
) -> None:
    rank_zero, rank_one, _single = global_batch_results
    assert rank_zero["plan_calls"] == 1
    assert rank_one["plan_calls"] == 0


@pytest.fixture(scope="module")
def synchronized_refine_results(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("synchronized-refine")
    _spawn_or_skip(
        _synchronized_refine_worker,
        (str(output_dir / "init"), str(output_dir)),
    )
    return (
        torch.load(output_dir / "refine-rank-0.pt", weights_only=False),
        torch.load(output_dir / "refine-rank-1.pt", weights_only=False),
    )


def test_two_rank_refinement_and_post_surgery_continuation_stay_synchronized(
    synchronized_refine_results,
) -> None:
    rank_zero, rank_one = synchronized_refine_results
    assert rank_zero["report"] == rank_one["report"]
    assert rank_zero["report"] == {
        "dupli": 1,
        "split": 1,
        "prune": 1,
        "n_splats": 9,
    }
    assert rank_zero["signature"] == rank_one["signature"]


def test_two_rank_forced_refinement_plans_only_on_rank_zero(
    synchronized_refine_results,
) -> None:
    rank_zero, rank_one = synchronized_refine_results
    assert rank_zero["plan_calls"] == 1
    assert rank_one["plan_calls"] == 0


def test_initial_signature_tripwire_raises_on_both_ranks(
    tmp_path: Path,
) -> None:
    _spawn_or_skip(
        _signature_tripwire_worker,
        (str(tmp_path / "init"), str(tmp_path)),
    )
    errors = [
        torch.load(tmp_path / f"tripwire-rank-{rank}.pt", weights_only=False)
        for rank in range(2)
    ]
    assert all(error is not None for error in errors)
    assert all("state signature mismatch" in error for error in errors)


def test_calibrator_missing_gradient_is_synchronized_as_zero(
    tmp_path: Path,
) -> None:
    _spawn_or_skip(
        _calibrator_sync_worker,
        (str(tmp_path / "init"), str(tmp_path)),
    )
    gradients = [
        torch.load(tmp_path / f"calibrator-rank-{rank}.pt", weights_only=False)
        for rank in range(2)
    ]
    assert torch.equal(gradients[0], torch.ones(1))
    assert torch.equal(gradients[1], torch.ones(1))


def test_single_process_splat_fit_never_enters_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _actual_global_batch_trainer(None)
    monkeypatch.setattr(
        datum_trainer_module.dist,
        "all_reduce",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-process fit entered a collective")
        ),
    )

    trainer.fit()


def test_state_signature_changes_when_a_lineage_buffer_changes() -> None:
    trainer = _bare_splat_trainer(_field(), rank=0)
    before = trainer._distributed_state_signature()
    with torch.no_grad():
        trainer._base_model().halving_depth[0] += 1
    after = trainer._distributed_state_signature()
    assert before != after
