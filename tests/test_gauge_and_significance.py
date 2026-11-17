from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import (
    SplatRefineConfig,
    SplatRefineStrategy,
    _inv_softplus,
)
from survey_fusion.pgm.splat_spectrum import (
    splat_band_amplitudes,
    splat_rendered_significance,
)

LOGLAM_MIN = math.log(3400.0)
LOGLAM_MAX = math.log(9200.0)
BANDPASSES = {
    "x": (
        torch.tensor([3800.0, 4200.0, 5100.0, 6200.0, 7100.0]),
        torch.tensor([0.0, 0.75, 1.0, 0.4, 0.0]),
    ),
    "y": (
        torch.tensor([6800.0, 7600.0, 8600.0]),
        torch.tensor([0.0, 1.0, 0.0]),
    ),
}
BAND_SIGMA = {"x": 0.05, "y": 0.08}
BAND_PSF_VARIANCE = {"x": 2.5e-6, "y": 4.0e-6}


def _field(n_splats: int = 6) -> SplatField:
    torch.manual_seed(7)
    field = SplatField(
        n_splats=n_splats,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.002,
    )
    field.sed.set_loglam_scale(0.5 * (LOGLAM_MAX - LOGLAM_MIN))
    with torch.no_grad():
        field.raw_flux.uniform_(-2.0, 1.0)
    return field


def _significance(field: SplatField) -> torch.Tensor:
    return splat_rendered_significance(
        field,
        BANDPASSES,
        BAND_SIGMA,
        BAND_PSF_VARIANCE,
        loglam_min=LOGLAM_MIN,
        loglam_max=LOGLAM_MAX,
    )


def _gauge_transform(field: SplatField, factor: float) -> None:
    """Prediction-preserving raw_flux/SED redistribution by `factor`."""
    with torch.no_grad():
        coeffs = field.sed_coeffs.to(torch.float64)
        flux = field.raw_flux.to(torch.float64)
        field.sed_coeffs.copy_(
            _inv_softplus(F.softplus(coeffs) * factor).to(field.sed_coeffs.dtype)
        )
        field.raw_flux.copy_(
            _inv_softplus(F.softplus(flux) / factor).to(field.raw_flux.dtype)
        )


def test_unit_amplitudes_are_full_amplitudes_per_flux() -> None:
    field = _field()
    full = splat_band_amplitudes(
        field, BANDPASSES, loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX
    )
    unit = splat_band_amplitudes(
        field,
        BANDPASSES,
        loglam_min=LOGLAM_MIN,
        loglam_max=LOGLAM_MAX,
        include_flux=False,
    )
    flux = F.softplus(field.raw_flux.detach())
    for band in BANDPASSES:
        torch.testing.assert_close(full[band], unit[band] * flux)


def test_rendered_significance_is_gauge_invariant_below_the_old_clamp() -> None:
    # raw_flux -80 puts softplus(raw_flux) ~ 1.8e-35, far below the retired
    # 1e-30 clamp that broke invariance in the divide-by-flux formulation
    field = _field()
    with torch.no_grad():
        field.raw_flux.fill_(-80.0)
        field.sed_coeffs.add_(40.0)  # keep the rendered product finite
    before = _significance(field)

    _gauge_transform(field, 4.0)
    after = _significance(field)

    assert bool(torch.isfinite(before).all())
    torch.testing.assert_close(after, before, rtol=1e-4, atol=0.0)


def test_rendered_significance_is_psf_limited_for_shrinking_splats() -> None:
    field = _field()
    with torch.no_grad():
        field.log_scales.fill_(math.log(1e-4))
    small = _significance(field)
    with torch.no_grad():
        field.log_scales.fill_(math.log(1e-6))
    tiny = _significance(field)

    # intrinsic peak would grow by (1e-4/1e-6)^2 = 1e4; the rendered peak is
    # pinned by the PSF variance
    torch.testing.assert_close(tiny, small, rtol=1e-2, atol=0.0)


def test_flux_verdicts_survive_the_historical_gauge_flip() -> None:
    field = _field()
    with torch.no_grad():
        field.raw_flux[2] = -30.0  # renders dark however the gauge sits

    def band_snr_fn(model: SplatField) -> torch.Tensor:
        return _significance(model)

    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=10, refine_every=1, prune_band_snr=True,
        prune_band_snr_sigma=1.0,
    )
    strategy = SplatRefineStrategy(
        cfg, field_width=1024, reference_sigma=0.05, band_snr_fn=band_snr_fn
    )
    strategy.grad_accum = torch.zeros(6)
    strategy.count = torch.ones(6)
    verdict_before = strategy.make_plan(field, step=1).prune_after_growth.clone()

    _gauge_transform(field, 16.0)
    verdict_after = strategy.make_plan(field, step=1).prune_after_growth.clone()

    assert bool(verdict_before[2])
    assert torch.equal(verdict_before, verdict_after)


def _project(field: SplatField, optimizers) -> int:
    unit = splat_band_amplitudes(
        field,
        BANDPASSES,
        loglam_min=LOGLAM_MIN,
        loglam_max=LOGLAM_MAX,
        include_flux=False,
    )
    factors = torch.stack([unit[b] for b in sorted(unit)], dim=0).amax(dim=0)
    strategy = SplatRefineStrategy(
        SplatRefineConfig(refine_start=0, refine_stop=10, refine_every=1),
        field_width=1024,
        reference_sigma=0.05,
    )
    return strategy.project_gauge(field, optimizers, factors.detach())


def test_gauge_projection_preserves_predictions_and_canonicalizes() -> None:
    field = _field()
    with torch.no_grad():
        field.sed_coeffs.add_(3.0)  # drive band-max unit amplitude far from 1
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    loss = field.fluxes().sum() + field.sed_coefficients().sum()
    loss.backward()
    optimizer.step()
    before = splat_band_amplitudes(
        field, BANDPASSES, loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX
    )

    projected = _project(field, [optimizer])

    assert projected == field.mu.shape[0]
    after = splat_band_amplitudes(
        field, BANDPASSES, loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX
    )
    for band in BANDPASSES:
        torch.testing.assert_close(after[band], before[band], rtol=5e-5, atol=0.0)
    unit = splat_band_amplitudes(
        field,
        BANDPASSES,
        loglam_min=LOGLAM_MIN,
        loglam_max=LOGLAM_MAX,
        include_flux=False,
    )
    band_max = torch.stack([unit[b] for b in sorted(unit)], dim=0).amax(dim=0)
    torch.testing.assert_close(
        band_max, torch.ones_like(band_max), rtol=5e-5, atol=0.0
    )


def test_gauge_projection_is_idempotent_and_respects_the_drift_band() -> None:
    field = _field()
    with torch.no_grad():
        field.sed_coeffs.add_(3.0)
    optimizer = torch.optim.Adam(field.param_groups_gsplat())

    first = _project(field, [optimizer])
    second = _project(field, [optimizer])

    assert first == field.mu.shape[0]
    assert second == 0  # canonical rows sit at |log a| = 0, inside the band


def test_gauge_projection_resets_moments_only_for_projected_rows() -> None:
    field = _field()
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    loss = (
        field.positions().square().sum()
        + field.fluxes().sum()
        + field.sed_coefficients().sum()
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        mu_moment = optimizer.state[field.mu]["exp_avg"].detach().clone()
        flux_moment = optimizer.state[field.raw_flux]["exp_avg"].detach().clone()
    assert bool(flux_moment.abs().sum() > 0)

    strategy = SplatRefineStrategy(
        SplatRefineConfig(refine_start=0, refine_stop=10, refine_every=1),
        field_width=1024,
        reference_sigma=0.05,
    )
    factors = torch.ones(6)
    factors[1] = 8.0  # only row 1 drifts outside |log a| > log 2
    projected = strategy.project_gauge(field, [optimizer], factors)

    assert projected == 1
    state = optimizer.state[field.raw_flux]
    assert torch.all(state["exp_avg"][1] == 0)
    assert torch.equal(state["exp_avg"][0], flux_moment[0])
    coeff_state = optimizer.state[field.sed_coeffs]
    assert torch.all(coeff_state["exp_avg"][1] == 0)
    # untouched parameters keep their moments wholesale
    assert torch.equal(optimizer.state[field.mu]["exp_avg"], mu_moment)


def test_gauge_projection_validates_factors_and_skips_zero_sed() -> None:
    field = _field()
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    strategy = SplatRefineStrategy(
        SplatRefineConfig(refine_start=0, refine_stop=10, refine_every=1),
        field_width=1024,
        reference_sigma=0.05,
    )
    with pytest.raises(ValueError, match="per-splat"):
        strategy.project_gauge(field, [optimizer], torch.ones(3))
    with pytest.raises(ValueError, match="finite and non-negative"):
        strategy.project_gauge(field, [optimizer], -torch.ones(6))
    factors = torch.zeros(6)  # dead SEDs: unnormalizable, must be left alone
    before = field.raw_flux.detach().clone()
    assert strategy.project_gauge(field, [optimizer], factors) == 0
    assert torch.equal(field.raw_flux, before)
