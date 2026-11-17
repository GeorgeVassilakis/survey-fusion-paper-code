from __future__ import annotations

import importlib.util
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory, OnesModel, SyntheticStore as MockStore, image_exposure, make_batch,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import DatumKind
from survey_fusion.training.checkpoints import (
    RNG_STATE_KEY, RNG_STATE_SCHEMA_VERSION, capture_rng_state, restore_rng_state,
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


train_entry = _load_entrypoint_module("train.py", "survey_fusion_train_rng_state_entry")

BASE_SEED = 4242


def _make_trainer(
    *, checkpoint_rng_state: bool = True, epochs: int = 1, max_steps: int = 0,
    pixel_batch_size: int = 2, seed: int = BASE_SEED,
) -> DatumTrainer:
    """Importance-sampled image trainer: four pixels, two batches per epoch."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    image_np = np.asarray([10.0, 4.0, 1.0, 0.0], dtype=np.float32).reshape(2, 2)
    batch = make_batch(
        y=torch.tensor(image_np.reshape(-1), dtype=torch.float32, device=device),
        sigma=torch.ones(4, dtype=torch.float32, device=device),
        exposure_id=torch.zeros(4, dtype=torch.long, device=device),
        datum_kind=torch.zeros(4, dtype=torch.long, device=device),
        datum_index=torch.arange(4, dtype=torch.long, device=device),
    )
    store = MockStore(
        exposures={0: image_exposure()}, batch=batch, image_band_by_exposure={0: "r"},
        kernel_factory=ConstantKernelFactory(batch), n_image=4, n_spec=0,
    )
    trainer_cfg: dict[str, Any] = {
        "kernel_samples_k": 4, "replica_samples_k": 1, "epochs": int(epochs),
        "max_steps": int(max_steps), "learning_rate": 1e-3, "apply_spatial_psf": False,
        "pixel_batch_size": int(pixel_batch_size), "spec_batch_size": 0,
        "image_importance_sampling": {"enabled": True, "max_importance_weight": 5.0},
        "checkpoint_rng_state": bool(checkpoint_rng_state),
    }
    return DatumTrainer(
        model=OnesModel(), grid=build_spatial_grid(2, 2, device), data=store,
        training_bands=["r"], held_out_band="i", lambda_step=20.0, trainer_cfg=trainer_cfg,
        device=device,
    )


def _draws(trainer: DatumTrainer, count: int) -> list[list[int]]:
    return [
        trainer._sample_importance_image_batch()[0].tolist() for _ in range(count)
    ]


def _make_spec_trainer(*, num_workers: int = 0, epochs: int = 2) -> DatumTrainer:
    n_spec = 12
    device = torch.device("cpu")
    batch = make_batch(
        y=torch.zeros(n_spec, dtype=torch.float32, device=device),
        sigma=torch.ones(n_spec, dtype=torch.float32, device=device),
        exposure_id=torch.zeros(n_spec, dtype=torch.long, device=device),
        datum_kind=torch.full(
            (n_spec,), int(DatumKind.SPECTRAL_BIN), dtype=torch.long, device=device,
        ),
        datum_index=torch.arange(n_spec, dtype=torch.long, device=device),
    )
    store = MockStore(
        exposures={0: image_exposure()},
        batch=batch, image_band_by_exposure={},
        kernel_factory=ConstantKernelFactory(batch), n_image=0, n_spec=n_spec,
    )
    torch.manual_seed(BASE_SEED)
    return DatumTrainer(
        model=OnesModel(), grid=build_spatial_grid(2, 2, device), data=store,
        training_bands=["r"], held_out_band="i", lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 1, "replica_samples_k": 1, "epochs": int(epochs),
            "learning_rate": 1e-3, "pixel_batch_size": 0, "spec_batch_size": 2,
            "apply_spatial_psf": False, "dataloader_shuffle": True,
            "dataloader_num_workers": int(num_workers), "checkpoint_rng_state": True,
        },
        device=device,
    )


def _set_epoch(trainer: DatumTrainer, epoch: int) -> None:
    for loader in (trainer.image_loader, trainer.spec_loader):
        sampler = getattr(loader, "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(int(epoch))


def _spec_batches(
    trainer: DatumTrainer, *, epoch: int, skip_batches: int = 0, count: int | None = None,
) -> list[list[int]]:
    _set_epoch(trainer, epoch)
    iterator = trainer._iter_epoch_batches(skip_batches=int(skip_batches))
    batches = islice(iterator, count) if count is not None else iterator
    result = []
    try:
        for _image_batch, spec_batch in batches:
            assert spec_batch is not None
            result.append(spec_batch.tolist())
    finally:
        del iterator
    return result


def _shutdown_importance_pool(trainer: DatumTrainer) -> None:
    pool = getattr(trainer, "_importance_sample_pool", None)
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
        trainer._importance_sample_pool = None
        trainer._importance_pending_batch = None


def _importance_prefetch_draws(trainer: DatumTrainer, count: int) -> list[list[int]]:
    iterator = trainer._iter_epoch_batches()
    try:
        return [next(iterator)[0].tolist() for _ in range(count)]
    finally:
        del iterator


def _install_importance_prefetch(
    trainer: DatumTrainer, mode: str,
) -> tuple[threading.Event | None, ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1)
    trainer._importance_sample_pool = pool
    if mode == "consumed":
        trainer._importance_pending_batch = None
        return None, pool
    if mode == "completed":
        future = pool.submit(trainer._sample_importance_image_batch)
        future.result()
        trainer._importance_pending_batch = future
        return None, pool
    if mode != "pending":
        raise AssertionError(f"unknown prefetch mode {mode!r}")
    started = threading.Event()
    release = threading.Event()
    sample = trainer._sample_importance_image_batch

    def blocked_sample():
        started.set()
        result = sample()
        assert release.wait(timeout=5.0)
        return result

    trainer._importance_pending_batch = pool.submit(blocked_sample)
    assert started.wait(timeout=5.0)
    return release, pool


def _checkpoint(trainer: DatumTrainer, *, step: int) -> dict[str, object]:
    return train_entry._build_training_checkpoint(
        trainer, SimpleNamespace(epochs_ran=0, steps_ran=int(step)),
    )


def test_checkpoint_omits_rng_state_when_the_flag_is_off() -> None:
    trainer = _make_trainer(checkpoint_rng_state=False)

    assert RNG_STATE_KEY not in _checkpoint(trainer, step=3)


def test_checkpoint_records_every_sampler_when_the_flag_is_on() -> None:
    trainer = _make_trainer()

    state = _checkpoint(trainer, step=3)[RNG_STATE_KEY]

    assert state["schema"] == RNG_STATE_SCHEMA_VERSION
    assert state["world_size"] == 1
    assert state["python"] is not None
    assert state["numpy"] is not None
    assert torch.is_tensor(state["torch_cpu"])
    assert state["torch_cuda"] is None
    assert torch.is_tensor(state["image_importance_generator"])
    assert state["kernel_factory"] is None or "generators" in state["kernel_factory"]
    assert state["epoch_position"]["steps_per_epoch"] == 2


def test_restore_continues_the_importance_draw_stream(tmp_path: Path) -> None:
    reference = _draws(_make_trainer(), 10)

    interrupted = _make_trainer()
    prefix = _draws(interrupted, 5)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(_checkpoint(interrupted, step=5), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    resumed = _make_trainer()
    assert restore_rng_state(
        resumed, checkpoint, checkpoint_path=checkpoint_path,
    ) is True

    assert prefix == reference[:5]
    assert _draws(resumed, 5) == reference[5:]


def test_mid_epoch_resume_continues_the_shuffled_spectral_stream(
    tmp_path: Path,
) -> None:
    reference = _spec_batches(_make_spec_trainer(num_workers=0), epoch=0)

    interrupted = _make_spec_trainer(num_workers=0)
    _spec_batches(interrupted, epoch=0, count=2)
    interrupted._steps_into_epoch = 2
    checkpoint = _checkpoint(interrupted, step=2)

    resumed = _make_spec_trainer(num_workers=0)
    assert restore_rng_state(
        resumed, checkpoint, checkpoint_path=tmp_path / "spec.pt",
    ) is True
    try:
        tail = _spec_batches(
            resumed, epoch=resumed._resume_epoch_offset,
            skip_batches=resumed._resume_epoch_batch_offset,
        )
    finally:
        for trainer in (interrupted, resumed):
            for loader in (trainer.image_loader, trainer.spec_loader):
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()

    assert tail == reference[2:]


@pytest.mark.parametrize("mode", ["pending", "completed", "consumed"])
def test_importance_prefetch_checkpoint_is_exact_for_every_timing(
    tmp_path: Path, mode: str,
) -> None:
    reference = _draws(_make_trainer(), 10)
    interrupted = _make_trainer()
    prefix = _draws(interrupted, 3)
    release, pool = _install_importance_prefetch(interrupted, mode)
    try:
        if release is None:
            checkpoint = _checkpoint(interrupted, step=3)
        else:
            captured: dict[str, object] = {}
            capture_done = threading.Event()

            def capture() -> None:
                captured["checkpoint"] = _checkpoint(interrupted, step=3)
                capture_done.set()

            capture_thread = threading.Thread(target=capture)
            capture_thread.start()
            assert not capture_done.wait(timeout=0.05)
            release.set()
            capture_thread.join(timeout=5.0)
            assert capture_done.is_set()
            checkpoint = captured["checkpoint"]
    finally:
        if release is not None:
            release.set()
        pool.shutdown(wait=True, cancel_futures=True)
        interrupted._importance_sample_pool = None
        interrupted._importance_pending_batch = None

    resumed = _make_trainer()
    assert restore_rng_state(
        resumed, checkpoint, checkpoint_path=tmp_path / f"importance_{mode}.pt",
    ) is True
    try:
        tail = _importance_prefetch_draws(resumed, 2)
    finally:
        _shutdown_importance_pool(resumed)

    assert prefix == reference[:3]
    assert tail == reference[3:5]


@pytest.mark.parametrize("steps_into_epoch", [0, 6])
def test_epoch_boundary_resume_does_not_skip_or_repeat_a_spectral_epoch(
    tmp_path: Path, steps_into_epoch: int,
) -> None:
    reference = _make_spec_trainer()
    expected_epoch = steps_into_epoch // 6
    expected = _spec_batches(reference, epoch=expected_epoch)

    interrupted = _make_spec_trainer()
    interrupted._epochs_completed = 0
    interrupted._steps_into_epoch = steps_into_epoch
    checkpoint = _checkpoint(interrupted, step=steps_into_epoch)
    resumed = _make_spec_trainer()
    assert restore_rng_state(
        resumed, checkpoint, checkpoint_path=tmp_path / f"boundary_{steps_into_epoch}.pt",
    ) is True

    assert resumed._resume_epoch_offset == expected_epoch
    assert resumed._resume_epoch_batch_offset == 0
    assert _spec_batches(
        resumed, epoch=expected_epoch,
        skip_batches=resumed._resume_epoch_batch_offset,
    ) == expected


def test_flag_off_importance_sequence_remains_the_pre_change_sequence() -> None:
    assert _draws(_make_trainer(checkpoint_rng_state=False), 5) == [
        [1, 1], [0, 0], [0, 0], [0, 1], [1, 0],
    ]


def test_without_restore_a_resumed_trainer_replays_the_prefix(tmp_path: Path) -> None:
    reference = _draws(_make_trainer(), 10)

    interrupted = _make_trainer(checkpoint_rng_state=False)
    _draws(interrupted, 5)
    checkpoint_path = tmp_path / "model.pt"
    checkpoint = _checkpoint(interrupted, step=5)

    replaying = _make_trainer(checkpoint_rng_state=False)
    assert restore_rng_state(
        replaying, checkpoint, checkpoint_path=checkpoint_path,
    ) is False

    replayed = _draws(replaying, 5)
    assert replayed == reference[:5]
    assert replayed != reference[5:]


def test_flag_off_warns_when_the_checkpoint_carries_rng_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _checkpoint(_make_trainer(), step=5)
    trainer = _make_trainer(checkpoint_rng_state=False)

    assert restore_rng_state(
        trainer, checkpoint, checkpoint_path=tmp_path / "model.pt",
    ) is False

    message = capsys.readouterr().out
    assert "WARNING" in message
    assert "checkpoint_rng_state is false" in message
    assert "REPLAYS" in message


def test_legacy_checkpoint_without_rng_state_warns_and_proceeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    legacy = _checkpoint(_make_trainer(checkpoint_rng_state=False), step=5)
    trainer = _make_trainer()

    assert restore_rng_state(
        trainer, legacy, checkpoint_path=tmp_path / "legacy.pt",
    ) is False

    message = capsys.readouterr().out
    assert "WARNING" in message
    assert "predates sampler" in message
    assert "REPLAYS" in message


def test_restore_is_skipped_and_warns_for_distributed_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _checkpoint(_make_trainer(), step=5)
    trainer = _make_trainer()
    trainer.world_size = 2

    assert restore_rng_state(
        trainer, checkpoint, checkpoint_path=tmp_path / "model.pt",
    ) is False

    message = capsys.readouterr().out
    assert "WARNING" in message
    assert "distributed runs are excluded" in message


def test_restore_rejects_an_unknown_payload_schema(tmp_path: Path) -> None:
    checkpoint = _checkpoint(_make_trainer(), step=5)
    checkpoint[RNG_STATE_KEY]["schema"] = RNG_STATE_SCHEMA_VERSION + 1
    trainer = _make_trainer()

    with pytest.raises(ValueError, match="schema"):
        restore_rng_state(trainer, checkpoint, checkpoint_path=tmp_path / "model.pt")


def test_restore_reinstates_the_process_global_streams(tmp_path: Path) -> None:
    trainer = _make_trainer()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    checkpoint = _checkpoint(trainer, step=5)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(trainer, checkpoint, checkpoint_path=tmp_path / "model.pt")

    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected


def test_kernel_factory_jitter_generators_roundtrip() -> None:
    from survey_fusion.pgm.kernel_builders import KernelFactoryFromBuilders

    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    factory = KernelFactoryFromBuilders.__new__(KernelFactoryFromBuilders)
    factory._rng_seed = 7
    factory._generators = {"cpu": generator}
    torch.rand((5,), generator=generator)

    state = factory.rng_state_dict()
    expected = torch.rand((3,), generator=generator)

    restored = KernelFactoryFromBuilders.__new__(KernelFactoryFromBuilders)
    restored._rng_seed = 7
    restored._generators = {}
    assert restored.load_rng_state_dict(state) == []

    assert torch.equal(
        torch.rand((3,), generator=restored._generator_for_device(device)), expected
    )


def test_epoch_position_roundtrip_continues_the_epoch_cycle(tmp_path: Path) -> None:
    interrupted = _make_trainer(epochs=3, max_steps=3)
    fit_state = interrupted.fit()
    assert fit_state.steps_ran == 3
    assert interrupted.epoch_position() == {
        "steps_per_epoch": 2, "epochs_completed": 1, "steps_into_epoch": 1,
    }
    checkpoint = _checkpoint(interrupted, step=3)

    resumed = _make_trainer(epochs=3, max_steps=6)
    resumed.resume_start_step = 3
    assert restore_rng_state(
        resumed, checkpoint, checkpoint_path=tmp_path / "model.pt",
    ) is True
    assert resumed._resume_epoch_offset == 1
    assert resumed._resume_epoch_batch_offset == 1

    resumed_state = resumed.fit()

    assert resumed_state.steps_ran == 6
    assert resumed.epoch_position() == {
        "steps_per_epoch": 2, "epochs_completed": 3, "steps_into_epoch": 0,
    }


def test_epoch_position_restore_does_not_change_the_resumed_step_budget(
    tmp_path: Path,
) -> None:
    interrupted = _make_trainer(epochs=3, max_steps=3)
    interrupted.fit()
    checkpoint = _checkpoint(interrupted, step=3)

    without_restore = _make_trainer(checkpoint_rng_state=False, epochs=3, max_steps=6)
    without_restore.resume_start_step = 3
    restore_rng_state(
        without_restore, checkpoint, checkpoint_path=tmp_path / "model.pt",
    )

    with_restore = _make_trainer(epochs=3, max_steps=6)
    with_restore.resume_start_step = 3
    restore_rng_state(with_restore, checkpoint, checkpoint_path=tmp_path / "model.pt")

    assert with_restore.fit().steps_ran == without_restore.fit().steps_ran


def test_epoch_position_is_refused_when_the_epoch_length_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _checkpoint(_make_trainer(), step=3)
    checkpoint[RNG_STATE_KEY]["epoch_position"] = {
        "steps_per_epoch": 7, "epochs_completed": 2, "steps_into_epoch": 3,
    }
    trainer = _make_trainer()

    assert restore_rng_state(
        trainer, checkpoint, checkpoint_path=tmp_path / "model.pt",
    ) is True

    message = capsys.readouterr().out
    assert "steps_per_epoch=7" in message
    assert trainer._resume_epoch_offset == 0
    assert trainer._resume_epoch_batch_offset == 0


def test_capture_rng_state_reports_the_normalized_epoch_position() -> None:
    trainer = _make_trainer()
    trainer._epochs_completed = 2
    trainer._steps_into_epoch = 2

    state = capture_rng_state(trainer)

    assert state["epoch_position"] == {
        "steps_per_epoch": 2, "epochs_completed": 3, "steps_into_epoch": 0,
    }


def test_rng_state_payload_survives_weights_only_torch_load(
    tmp_path: Path,
) -> None:
    # train.py's _load_checkpoint calls torch.load with the weights_only=True
    # default (torch >= 2.6). A payload smuggling a raw numpy ndarray global
    # fails that load, making every rng_state-bearing checkpoint unresumable.
    trainer = _make_trainer()
    _draws(trainer, 3)
    payload = capture_rng_state(trainer)
    expected = _draws(trainer, 4)

    path = tmp_path / "rng_only.pt"
    torch.save({"rng_state": payload}, path)
    loaded = torch.load(path, map_location="cpu")

    resumed = _make_trainer(seed=BASE_SEED + 999)
    assert restore_rng_state(
        resumed, loaded, checkpoint_path=path,
    ) is True
    assert _draws(resumed, 4) == expected
