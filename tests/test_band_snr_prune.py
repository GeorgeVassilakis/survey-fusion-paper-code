from __future__ import annotations

import math

import pytest
import torch

from test_analytic_trainer_dispatch import _bandpasses, _realistic_scene
from test_gauge_and_significance import (
    BANDPASSES,
    BAND_PSF_VARIANCE,
    BAND_SIGMA,
    LOGLAM_MIN,
    LOGLAM_MAX,
    _field,
    _significance,
)
from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import SplatRefineConfig, SplatRefineStrategy
from survey_fusion.training.datum_trainer import DatumTrainer


def _strategy(**overrides) -> SplatRefineStrategy:
    band_snr_fn = overrides.pop("band_snr_fn", None)
    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=0, refine_every=1, **overrides
    )
    return SplatRefineStrategy(
        cfg, field_width=1024, reference_sigma=0.05, band_snr_fn=band_snr_fn,
    )


def _trainer(refine_overrides: dict, *, exclude_band: str | None = None) -> DatumTrainer:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _realistic_scene(), _bandpasses(), psf_fwhm_arcsec=1.0, device=device,
        spectral_response_mode="legacy_imaging_median",
    )
    exclude = [
        int(exp_id)
        for exp_id, band in store.image_band_by_exposure.items()
        if exclude_band is not None and str(band) == exclude_band
    ]
    torch.manual_seed(31)
    model = SplatField(
        n_splats=5, n_templates=3, loglam_grid=torch.linspace(-1, 1, 16),
        init_scale=0.04, init_extent=0.6,
    )
    trainer_cfg = {
        "analytic_images": True, "analytic_spectra": True,
        "kernel_samples_k": 4, "replica_samples_k": 1,
        "epochs": 0, "learning_rate": 1e-3,
        "pixel_batch_size": 8, "spec_batch_size": 2,
        "refine": {"enabled": True, "pair_n_sigma": 8.0, **refine_overrides},
    }
    return DatumTrainer(
        model=model, grid=build_spatial_grid(2, 3, device), data=store,
        training_bands=["g", "r", "i"], held_out_band="i", lambda_step=20.0,
        trainer_cfg=trainer_cfg, device=device,
        exclude_image_exposure_ids=exclude or None,
    )


# --- threshold field ------------------------------------------------------


def test_band_snr_requires_its_own_threshold() -> None:
    with pytest.raises(ValueError, match="prune_band_snr_sigma"):
        _strategy(prune_band_snr=True, band_snr_fn=lambda model: torch.zeros(1))


def test_band_snr_threshold_without_the_flag_is_rejected() -> None:
    with pytest.raises(ValueError, match="silently ignored"):
        _strategy(prune_band_snr_sigma=1e-8)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_band_snr_threshold_must_be_finite_and_non_negative(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _strategy(
            prune_band_snr=True, prune_band_snr_sigma=bad,
            band_snr_fn=lambda model: torch.zeros(1),
        )


def test_band_snr_branch_ignores_prune_peak_sb_frac() -> None:
    # prune_peak_sb_frac would condemn everything if the band-SNR branch still
    # read it; the dedicated threshold is what decides.
    field = _field(n_splats=4)
    factors = torch.tensor([1.0, 1.0, 1e-12, 1e-12])
    strategy = _strategy(
        prune_band_snr=True, prune_band_snr_sigma=1e-8,
        prune_peak_sb_frac=1e30, band_snr_fn=lambda model: factors,
    )
    condemned = strategy.make_plan(field, step=1).prune_after_growth
    assert condemned.tolist() == [False, False, True, True]


# --- the criterion the two paths disagree about ---------------------------


def _gauge_drifted_pair() -> SplatField:
    """Two splats with inflated raw_flux; only the first renders anything.

    The measured canonical-run condition: gauge drift parks the luminosity in
    raw_flux, so the bandless criterion sees both splats as bright while the
    second one's SED integrates to nothing in every band.
    """
    field = _field(n_splats=2)
    with torch.no_grad():
        field.raw_flux.fill_(6.0)
        field.sed_coeffs[1].fill_(-40.0)
    return field


def test_band_snr_kills_the_dark_splat_the_flux_criterion_keeps() -> None:
    field = _gauge_drifted_pair()

    flux_only = _strategy()
    kept = flux_only.make_plan(field, step=1).prune_after_growth
    assert kept.tolist() == [False, False]

    significance = _significance(field)
    assert float(significance[0]) > 1.0
    assert float(significance[1]) < 1e-8

    band_snr = _strategy(
        prune_band_snr=True, prune_band_snr_sigma=1.0,
        band_snr_fn=_significance,
    )
    condemned = band_snr.make_plan(field, step=1).prune_after_growth
    assert condemned.tolist() == [False, True]


def test_flux_criterion_is_unchanged_when_the_flag_is_off() -> None:
    field = _gauge_drifted_pair()
    strategy = _strategy()
    assert strategy.config.prune_band_snr is False
    assert strategy.config.prune_band_snr_sigma is None

    scales = torch.exp(field.log_scales.detach())
    peak_sb = torch.nn.functional.softplus(field.raw_flux.detach()) / (
        2.0 * math.pi * scales[:, 0] * scales[:, 1]
    )
    expected = peak_sb < strategy.config.prune_peak_sb_frac * strategy.reference_sigma
    condemned = strategy.make_plan(field, step=1).prune_after_growth
    assert torch.equal(condemned, expected)


def test_significance_helper_matches_the_strategy_threshold_units() -> None:
    # the band_snr_fn the trainer installs and the threshold the strategy
    # compares against must be the same quantity: peak SNR in sigma units
    field = _gauge_drifted_pair()
    from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes

    unit = splat_band_amplitudes(
        field, BANDPASSES, loglam_min=LOGLAM_MIN, loglam_max=LOGLAM_MAX,
        include_flux=False,
    )
    flux = torch.nn.functional.softplus(field.raw_flux.detach())
    scales_sq = torch.exp(2.0 * field.log_scales.detach())
    expected = None
    for band in BANDPASSES:
        variance = BAND_PSF_VARIANCE[band]
        determinant = (scales_sq[:, 0] + variance) * (scales_sq[:, 1] + variance)
        peak = flux * unit[band] / (2.0 * math.pi * torch.sqrt(determinant))
        snr = peak / BAND_SIGMA[band]
        expected = snr if expected is None else torch.maximum(expected, snr)
    torch.testing.assert_close(_significance(field), expected)


# --- trainer wiring -------------------------------------------------------


def test_trainer_default_refine_leaves_the_threshold_unset() -> None:
    config = _trainer({}).refine_strategy.config
    assert config.prune_band_snr is False
    assert config.prune_band_snr_sigma is None
    assert config.prune_peak_sb_frac == 0.005


def test_trainer_passes_the_threshold_through() -> None:
    config = _trainer(
        {"prune_band_snr": True, "prune_band_snr_sigma": 1e-8}
    ).refine_strategy.config
    assert config.prune_band_snr is True
    assert config.prune_band_snr_sigma == 1e-8


def test_trainer_refuses_band_snr_without_a_threshold() -> None:
    with pytest.raises(ValueError, match="prune_band_snr_sigma"):
        _trainer({"prune_band_snr": True})


def test_band_tables_refuse_a_band_with_no_usable_sigma_rows() -> None:
    trainer = _trainer({}, exclude_band="i")
    with pytest.raises(ValueError, match=r"\['i'\] have image exposures"):
        trainer._prepare_band_tables()


def test_gauge_projection_warns_and_uses_remaining_bands() -> None:
    trainer = _trainer({}, exclude_band="i")
    with pytest.warns(RuntimeWarning, match=r"gauge projection.*\['i'\]"):
        factors = trainer._splat_gauge_factors(trainer._base_model())

    assert factors.shape == (5,)
    assert sorted(trainer._band_table_state["bandpass_tuples"]) == ["g", "r"]


def test_residual_birth_warns_and_uses_remaining_bands() -> None:
    trainer = _trainer({"birth": {"enabled": True}}, exclude_band="i")
    shared_wcs = next(iter(trainer.data.exposures.values())).wcs
    for exposure in trainer.data.exposures.values():
        exposure.wcs = shared_wcs
    with pytest.warns(RuntimeWarning, match=r"residual birth.*\['i'\]"):
        accumulator = trainer._birth_accumulator()

    assert sorted(accumulator.band_index) == ["g", "r"]


def test_band_snr_prune_policy_still_raises_for_a_missing_band() -> None:
    trainer = _trainer({}, exclude_band="i")
    with pytest.raises(ValueError, match=r"\['i'\] have image exposures"):
        trainer._prepare_band_tables(
            missing_band_policy="raise", consumer="band-SNR pruning"
        )


def test_band_tables_cover_every_image_band_when_nothing_is_excluded() -> None:
    tables = _trainer({})._prepare_band_tables()
    assert sorted(tables["bandpass_tuples"]) == ["g", "i", "r"]
    assert sorted(tables["band_sigma"]) == ["g", "i", "r"]
    assert sorted(tables["band_psf_variance"]) == ["g", "i", "r"]


# --- prune observability --------------------------------------------------


def _prune_one_and_capture(trainer: DatumTrainer, capsys) -> tuple[dict, str]:
    report = trainer._post_optimizer_step(1)
    return report, capsys.readouterr().out


def test_band_snr_prune_logs_counts_and_threshold(capsys) -> None:
    trainer = _trainer({
        "start": 0, "every": 1, "stop": 0,
        "prune_band_snr": True, "prune_band_snr_sigma": 1e-8,
    })

    def one_dark_splat(model) -> torch.Tensor:
        factor = torch.ones(int(model.mu.shape[0]))
        factor[0] = 0.0
        return factor

    trainer.refine_strategy.band_snr_fn = one_dark_splat
    report, out = _prune_one_and_capture(trainer, capsys)

    assert report["prune"] == 1
    assert report["n_splats"] == 4
    assert "band_snr 1" in out
    assert "threshold 1e-08 sigma peak SNR" in out
    assert "n_splats=4" in out


def test_flux_prune_log_is_unchanged_when_the_flag_is_off(capsys) -> None:
    trainer = _trainer({"start": 0, "every": 1, "stop": 0})
    with torch.no_grad():
        trainer._base_model().raw_flux[0] = -40.0
    report, out = _prune_one_and_capture(trainer, capsys)

    assert report["prune"] == 1
    assert "flux 1" in out
    assert "band_snr" not in out
    assert "peak SNR" not in out
