from __future__ import annotations

import math

import torch

from support.synthetic_trainer import make_image_spec_trainer
from survey_fusion.pgm.contracts import DatumKind
from survey_fusion.training.datum_trainer import DatumTrainer


def _make_trainer(
    *, init_scale: float, y: torch.Tensor | None = None, additive: torch.Tensor | None = None,
    source_gate: torch.Tensor | None = None,
) -> DatumTrainer:
    return make_image_spec_trainer(
        calibration_init_scale=init_scale, y=y, additive=additive, source_gate=source_gate,
    )


def test_default_compute_mu_uses_materialized_batch_length() -> None:
    device = torch.device("cpu")
    trainer = _make_trainer(init_scale=1.0)
    idx = torch.arange(int(trainer.data.batch.y.shape[0]), dtype=torch.long, device=device)

    assert int(trainer.n_image_datums) == 1
    assert idx.tolist() == [0, 1]
    assert torch.allclose(trainer.compute_mu(), trainer.compute_mu(idx))
    assert torch.allclose(trainer.compute_source_mu(), trainer.compute_source_mu(idx))


def test_mc_spectral_nuisance_scales_spec_only() -> None:
    device = torch.device("cpu")
    trainer_off = _make_trainer(init_scale=1.0)
    trainer_on = _make_trainer(init_scale=1.5)
    idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    mu_off = trainer_off.compute_mu(idx)
    mu_on = trainer_on.compute_mu(idx)

    assert torch.allclose(mu_on[0], mu_off[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(mu_on[1], 1.5 * mu_off[1], atol=1e-5, rtol=1e-5)


def test_additive_zero_source_gives_gaussian_normalization_nll() -> None:
    device = torch.device("cpu")
    y = torch.tensor([5.0, 7.0], dtype=torch.float32, device=device)
    trainer = _make_trainer(
        init_scale=1.0, y=y, additive=y, source_gate=torch.zeros(2, dtype=torch.float32, device=device),
    )
    terms = trainer._compute_mc_unbiased_gaussian_terms(torch.tensor([0, 1], dtype=torch.long, device=device))
    expected = torch.full_like(terms, 0.5 * math.log(2.0 * math.pi))
    assert torch.allclose(terms, expected, atol=1e-7, rtol=0.0)


def test_zero_additive_unit_gate_matches_source_only_mu() -> None:
    device = torch.device("cpu")
    trainer = _make_trainer(
        init_scale=1.0, additive=torch.zeros(2, dtype=torch.float32, device=device),
        source_gate=torch.ones(2, dtype=torch.float32, device=device),
    )
    idx = torch.tensor([0, 1], dtype=torch.long, device=device)
    assert torch.allclose(trainer.compute_mu(idx), trainer.compute_source_mu(idx))


def test_additive_contract_residual_matches_source_residual() -> None:
    device = torch.device("cpu")
    y_source = torch.tensor([3.0, 11.0], dtype=torch.float32, device=device)
    additive = torch.tensor([5.0, 7.0], dtype=torch.float32, device=device)
    trainer = _make_trainer(init_scale=1.0, y=y_source + additive, additive=additive)
    idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    residual = trainer.compute_mu(idx) - trainer.data.batch.y
    source_residual = trainer.compute_source_mu(idx) - y_source

    assert torch.allclose(residual, source_residual, atol=1e-6, rtol=1e-6)


def test_source_gate_zero_blocks_source_gradient() -> None:
    device = torch.device("cpu")
    trainer = _make_trainer(
        init_scale=1.0, additive=torch.zeros(2, dtype=torch.float32, device=device),
        source_gate=torch.zeros(2, dtype=torch.float32, device=device),
    )
    idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    trainer.compute_mu(idx).sum().backward()

    bias_grad = next(trainer.model.parameters()).grad
    assert bias_grad is not None
    assert torch.allclose(bias_grad, torch.zeros_like(bias_grad), atol=1e-7, rtol=0.0)


def test_source_gated_spectral_rows_are_excluded_from_training_indices() -> None:
    device = torch.device("cpu")
    trainer = _make_trainer(
        init_scale=1.0, source_gate=torch.tensor([1.0, 0.0], dtype=torch.float32, device=device),
    )

    spectral_rows = torch.nonzero(
        trainer.data.batch.datum_kind == int(DatumKind.SPECTRAL_BIN), as_tuple=False,
    ).flatten()

    assert spectral_rows.tolist() == [1]
    assert int(trainer.spec_train_indices.numel()) == 0
    assert trainer.image_train_indices.tolist() == [0]


def test_spectral_calibration_applies_before_additive() -> None:
    device = torch.device("cpu")
    additive = torch.tensor([5.0, 7.0], dtype=torch.float32, device=device)
    trainer = _make_trainer(init_scale=1.5, additive=additive)
    idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    mu = trainer.compute_mu(idx)

    assert torch.allclose(mu[0], torch.tensor(3.0 + 5.0), atol=1e-5, rtol=1e-5)
    assert torch.allclose(mu[1], torch.tensor(1.5 * 11.0 + 7.0), atol=1e-5, rtol=1e-5)
