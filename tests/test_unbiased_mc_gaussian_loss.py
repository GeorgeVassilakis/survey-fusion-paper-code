from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from survey_fusion.pgm.likelihoods import (
    datum_unbiased_gaussian_nll,
    population_weighted_modality_nll,
)


def test_matches_the_exact_gaussian_log_density() -> None:
    """The objective must be a log-likelihood, not a bare chi-square: the same
    number has to be readable as -log p(y | mu, sigma) so that image and
    spectral datums, and held-out scores, are all in one currency."""
    y = torch.tensor([0.4, -1.2, 3.0])
    mu = torch.tensor([0.1, -0.7, 3.5])
    sigma = torch.tensor([0.5, 2.0, 0.25])
    got = datum_unbiased_gaussian_nll(y, mu, mu, sigma)
    expect = -torch.distributions.Normal(mu, sigma).log_prob(y)
    assert torch.allclose(got, expect, atol=1e-6)


def test_perfect_fit_returns_the_entropy_floor_not_zero() -> None:
    sigma = torch.full((4,), 0.3)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    got = datum_unbiased_gaussian_nll(y, y, y, sigma)
    expect = math.log(math.sqrt(2 * math.pi) * 0.3)
    assert torch.allclose(got, torch.full((4,), expect), atol=1e-6)


def test_tiny_sigma_is_used_without_clamping() -> None:
    y = torch.tensor([0.0], dtype=torch.float64)
    mu = torch.tensor([1.0e-8], dtype=torch.float64)
    sigma = torch.tensor([1.0e-8], dtype=torch.float64)
    got = datum_unbiased_gaussian_nll(y, mu, mu, sigma)
    expect = -torch.distributions.Normal(mu, sigma).log_prob(y)
    assert torch.equal(got, expect)


def test_unbiased_gaussian_loss_matches_quadratic_objective_in_expectation() -> None:
    rng = np.random.default_rng(123)
    n_trials = 100_000
    draws_a = rng.choice(np.array([0.0, 2.0], dtype=np.float32), size=n_trials)
    draws_b = rng.choice(np.array([0.0, 2.0], dtype=np.float32), size=n_trials)

    y = torch.full((n_trials,), 0.3, dtype=torch.float32)
    sigma = torch.full((n_trials,), 0.7, dtype=torch.float32)
    mu_a = torch.tensor(draws_a, dtype=torch.float32)
    mu_b = torch.tensor(draws_b, dtype=torch.float32)

    unbiased = datum_unbiased_gaussian_nll(y, mu_a, mu_b, sigma).mean().item()
    naive = (0.5 * (mu_a - y) * (mu_a - y) / (sigma * sigma)).mean().item()

    true_mu = 1.0
    log_norm = math.log(math.sqrt(2 * math.pi) * 0.7)
    exact = ((true_mu - 0.3) ** 2) / (2.0 * (0.7 ** 2)) + log_norm
    naive = naive + log_norm

    assert unbiased == pytest.approx(exact, abs=2.0e-2)
    assert naive > exact + 0.1


def test_population_weighted_modality_nll_is_batch_composition_invariant() -> None:
    image_population = 985_645_126
    spectrum_population = 2_529_181
    image_mean = torch.tensor(2.0)
    spectrum_mean = torch.tensor(50.0)

    first = population_weighted_modality_nll(
        image_nll_sum=image_mean * 200_000,
        image_sample_count=200_000,
        image_population_count=image_population,
        spectrum_nll_sum=spectrum_mean * 1_024,
        spectrum_sample_count=1_024,
        spectrum_population_count=spectrum_population,
    )
    second = population_weighted_modality_nll(
        image_nll_sum=image_mean * 50_000,
        image_sample_count=50_000,
        image_population_count=image_population,
        spectrum_nll_sum=spectrum_mean * 4_096,
        spectrum_sample_count=4_096,
        spectrum_population_count=spectrum_population,
    )
    expected = (
        image_population * image_mean + spectrum_population * spectrum_mean
    ) / (image_population + spectrum_population)

    assert float(first) == pytest.approx(float(expected))
    assert float(second) == pytest.approx(float(expected))
