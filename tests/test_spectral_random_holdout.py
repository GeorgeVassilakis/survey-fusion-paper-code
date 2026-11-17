from __future__ import annotations

import pytest
import torch

from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.training.datum_trainer import DatumTrainer


def test_spectral_random_holdout_is_excluded_from_training() -> None:
    trainer = make_image_spec_trainer(
        trainer_cfg={"random_holdout": {"fraction": 1.0, "seed": 20260723}}
    )

    assert trainer.spec_train_indices.numel() == 0
    assert trainer.spec_holdout_indices.numel() == 1
    assert trainer.spec_holdout_indices.tolist() == [1]


def test_zero_spectral_holdout_fraction_keeps_all_spectral_datums() -> None:
    trainer = make_image_spec_trainer(
        trainer_cfg={"random_holdout": {"fraction": 0.0, "seed": 20260723}}
    )

    assert trainer.spec_train_indices.tolist() == [1]
    assert trainer.spec_holdout_indices.numel() == 0


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_invalid_spectral_holdout_fraction_fails_loudly(fraction: float) -> None:
    with pytest.raises(ValueError, match="random_holdout.fraction"):
        make_image_spec_trainer(
            trainer_cfg={"random_holdout": {"fraction": fraction, "seed": 20260723}}
        )


def test_positive_spectral_holdout_requires_explicit_seed() -> None:
    with pytest.raises(ValueError, match="random_holdout.seed"):
        make_image_spec_trainer(
            trainer_cfg={"random_holdout": {"fraction": 0.1, "seed": 0}}
        )


def test_fixed_residual_summary_uses_supplied_sigmas() -> None:
    summary = DatumTrainer._summarize_fixed_residuals(
        torch.tensor([2.0, 8.0]),
        torch.tensor([1.0, 4.0]),
        torch.tensor([1.0, 2.0]),
    )

    assert summary["n"] == 2
    assert summary["chi2_mean"] == pytest.approx(2.5)
    assert summary["normalized_residual_mean"] == pytest.approx(1.5)
    assert summary["normalized_residual_std"] == pytest.approx(0.5)
    assert summary["mae"] == pytest.approx(2.5)


def test_local_membership_mask_is_linear_lookup_with_stable_order() -> None:
    mask = DatumTrainer._local_membership_mask(
        torch.tensor([4, 0, 2, 1]),
        torch.tensor([1, 4]),
        5,
    )

    assert mask.tolist() == [True, False, False, True]


def test_local_membership_mask_rejects_out_of_bounds_indices() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        DatumTrainer._local_membership_mask(
            torch.tensor([0, 5]),
            torch.tensor([1]),
            5,
        )
