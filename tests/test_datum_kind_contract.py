from __future__ import annotations

import pytest
import torch

from survey_fusion.pgm.contracts import DatumBatch, DatumKind, validate_batch_datum_kinds


def _batch(kind_values: list[int], local_indices: list[int]) -> DatumBatch:
    n = len(kind_values)
    return DatumBatch(
        y=torch.zeros(n, dtype=torch.float32), sigma=torch.ones(n, dtype=torch.float32),
        exposure_id=torch.zeros(n, dtype=torch.long), datum_kind=torch.tensor(kind_values, dtype=torch.long),
        datum_index=torch.tensor(local_indices, dtype=torch.long), kernel_samples=None,
    )


def test_validate_batch_datum_kinds_accepts_supported_kinds() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)], [0, 0])
    validate_batch_datum_kinds(batch)


def test_validate_batch_datum_kinds_rejects_unknown_kind() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL), 99], [0, 0])
    with pytest.raises(ValueError, match="unsupported datum_kind"):
        validate_batch_datum_kinds(batch)


def test_validate_batch_datum_kinds_rejects_negative_local_index() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL)], [-1])
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_batch_datum_kinds(batch)


def test_validate_batch_datum_kinds_accepts_additive_and_source_gate() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)], [0, 0])
    batch.additive = torch.zeros(2, dtype=torch.float32)
    batch.source_gate = torch.ones(2, dtype=torch.float32)

    validate_batch_datum_kinds(batch)


def test_validate_batch_datum_kinds_rejects_misaligned_additive() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)], [0, 0])
    batch.additive = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="additive must align"):
        validate_batch_datum_kinds(batch)


def test_validate_batch_datum_kinds_rejects_misaligned_source_gate() -> None:
    batch = _batch([int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)], [0, 0])
    batch.source_gate = torch.ones(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="source_gate must align"):
        validate_batch_datum_kinds(batch)
