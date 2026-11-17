from types import SimpleNamespace

import torch

from survey_fusion.training.datum_trainer import DatumTrainer


def _build(
    exposure_id: list[int],
    datum_kind: list[int],
    datum_index: list[int],
) -> dict[tuple[int, int], torch.Tensor]:
    trainer = DatumTrainer.__new__(DatumTrainer)
    trainer.data = SimpleNamespace(
        batch=SimpleNamespace(
            exposure_id=torch.tensor(exposure_id, dtype=torch.long),
            datum_kind=torch.tensor(datum_kind, dtype=torch.long),
            datum_index=torch.tensor(datum_index, dtype=torch.long),
        )
    )
    return trainer._build_global_index_cache()


def test_global_index_cache_uses_contiguous_runs() -> None:
    cache = _build(
        exposure_id=[0, 0, 0, 1, 1],
        datum_kind=[0, 0, 0, 1, 1],
        datum_index=[0, 1, 2, 0, 1],
    )

    torch.testing.assert_close(cache[(0, 0)], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(cache[(1, 1)], torch.tensor([3, 4]))


def test_global_index_cache_merges_interleaved_runs_in_local_order() -> None:
    cache = _build(
        exposure_id=[4, 7, 4, 4, 7, 4],
        datum_kind=[0, 1, 0, 1, 1, 0],
        datum_index=[3, 1, 0, 8, 0, 2],
    )

    torch.testing.assert_close(cache[(4, 0)], torch.tensor([2, 5, 0]))
    torch.testing.assert_close(cache[(4, 1)], torch.tensor([3]))
    torch.testing.assert_close(cache[(7, 1)], torch.tensor([4, 1]))


def test_global_index_cache_rejects_misaligned_columns() -> None:
    trainer = DatumTrainer.__new__(DatumTrainer)
    trainer.data = SimpleNamespace(
        batch=SimpleNamespace(
            exposure_id=torch.tensor([0, 0]),
            datum_kind=torch.tensor([0]),
            datum_index=torch.tensor([0, 1]),
        )
    )

    try:
        trainer._build_global_index_cache()
    except ValueError as error:
        assert "equal length" in str(error)
    else:
        raise AssertionError("Expected misaligned columns to fail.")
