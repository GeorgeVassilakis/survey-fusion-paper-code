from __future__ import annotations

import math

import torch


def population_weighted_modality_nll(
    *,
    image_nll_sum: torch.Tensor,
    image_sample_count: int | float,
    image_population_count: int,
    spectrum_nll_sum: torch.Tensor,
    spectrum_sample_count: int | float,
    spectrum_population_count: int,
    spectrum_weight: float = 1.0,
) -> torch.Tensor:
    """Estimate the population mean NLL, each spectral term weighted by spectrum_weight.

    The divisor is the unweighted datum count of both populations, so the
    weight changes the spectral share of the objective, not its normalization.
    """
    if image_population_count < 0 or spectrum_population_count < 0:
        raise ValueError("Population counts must be non-negative.")
    if not math.isfinite(spectrum_weight) or spectrum_weight <= 0.0:
        raise ValueError("spectrum_weight must be finite and positive.")
    spectrum_nll_sum = spectrum_nll_sum * spectrum_weight
    population_count = image_population_count + spectrum_population_count
    if population_count <= 0:
        return image_nll_sum * 0.0 + spectrum_nll_sum * 0.0

    loss = image_nll_sum * 0.0 + spectrum_nll_sum * 0.0
    if image_population_count > 0:
        if image_sample_count <= 0:
            raise ValueError("A positive image population requires image samples.")
        loss = loss + (
            float(image_population_count)
            / float(population_count)
            * image_nll_sum
            / float(image_sample_count)
        )
    if spectrum_population_count > 0:
        if spectrum_sample_count <= 0:
            raise ValueError("A positive spectrum population requires spectrum samples.")
        loss = loss + (
            float(spectrum_population_count)
            / float(population_count)
            * spectrum_nll_sum
            / float(spectrum_sample_count)
        )
    return loss


def datum_unbiased_gaussian_nll(
    y: torch.Tensor, mu_a: torch.Tensor, mu_b: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """Per-datum negative log-likelihood of y under Normal(mu, sigma^2).

        -log p(y | mu, sigma) = (mu - y)^2 / (2 sigma^2) + log(sqrt(2 pi) sigma)

    sigma is data (the CCD Poisson + read-noise model for images, the SDSS
    inverse variance for spectra), so the normalisation term is constant in the
    parameters and contributes no gradient. It is included anyway so that the
    reported number is an actual log-likelihood: comparable between runs, across
    different datum counts, and directly usable as the held-out metric. Without
    it a "loss" of 0.4 says nothing about whether the model explains the data.

    mu_a and mu_b are two independent unbiased estimates of mu. Their product is
    an unbiased estimate of the squared residual, where mu_a**2 would not be.
    Pass the same tensor twice when mu is computed exactly rather than by MC.
    """

    if y.shape != mu_a.shape or y.shape != mu_b.shape or y.shape != sigma.shape:
        raise ValueError("y, mu_a, mu_b, and sigma must have the same shape.")
    residual_a = mu_a - y
    residual_b = mu_b - y
    sigma2 = sigma * sigma
    return 0.5 * residual_a * residual_b / sigma2 + 0.5 * torch.log(
        2 * math.pi * sigma2
    )


def current_model_image_sigma(
    y: torch.Tensor, mu: torch.Tensor, sigma_data: torch.Tensor, alpha: torch.Tensor,
    beta: torch.Tensor, outlier_pull: float,
) -> torch.Tensor:
    """CCD sigma of the current model prediction, held fixed within the step.

        sigma^2 = alpha * max(mu, 0) + beta

    with mu (the total prediction, sky included) detached. The gradient of
    datum_unbiased_gaussian_nll in mu is then (mu - y) / (alpha mu + beta), the
    gradient of the shifted-Poisson NLL (mu + c) / alpha - ((y + c) / alpha)
    log(mu + c) with c = beta / alpha, so the fit converges to the Poisson
    maximum-likelihood solution with read noise folded in as an equivalent count.
    The variance of the observed counts biases it low; a gradient through sigma
    would bias it high. Where the model misses y by more than outlier_pull data sigmas
    (artefacts, sources not yet fitted) the data sigma is kept, so a small model
    variance never hands an unmodelled feature a large pull.
    """
    if not (y.shape == mu.shape == sigma_data.shape == alpha.shape == beta.shape):
        raise ValueError("y, mu, sigma_data, alpha, and beta must have the same shape.")
    model = mu.detach()
    model_sigma = torch.sqrt(alpha * torch.clamp(model, min=0.0) + beta)
    return torch.where((y - model).abs() > outlier_pull * sigma_data, sigma_data, model_sigma)
