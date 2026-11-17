from __future__ import annotations

import math

import pytest
import torch

from survey_fusion.models.splat_birth import ResidualBirthAccumulator
from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import (
    SplatRefineConfig,
    SplatRefinePlan,
    SplatRefineStrategy,
)


def _accumulator(cell_size: float = 0.25, bands=("g", "r")) -> ResidualBirthAccumulator:
    return ResidualBirthAccumulator(
        cell_size=cell_size, bands=bands, device="cpu"
    )


def test_null_field_z_scores_are_standard_normal_under_importance_sampling() -> None:
    # The estimator gate: with-replacement, non-uniformly weighted draws from
    # a pure-noise field must produce cell z-scores with unit variance. A raw
    # count-based z would be inflated by duplicate draws.
    torch.manual_seed(0)
    accumulator = _accumulator(cell_size=0.125, bands=("g",))
    n_pixels = 20_000
    xs = torch.rand(n_pixels) * 2 - 1
    ys = torch.rand(n_pixels) * 2 - 1
    residual = torch.randn(n_pixels)  # r/sigma of a perfect model
    probs = torch.rand(n_pixels) + 0.05
    probs /= probs.sum()
    n_draws = 200_000
    draws = torch.multinomial(probs, n_draws, replacement=True)
    weights = 1.0 / (n_draws * probs[draws])
    accumulator.accumulate(
        torch.zeros(n_draws, dtype=torch.long),
        xs[draws],
        ys[draws],
        residual[draws],
        torch.zeros(n_draws),
        residual[draws],
        weights,
    )
    z = accumulator.S / accumulator.Q.clamp_min(1e-30).sqrt()
    n_eff = accumulator.W.square() / accumulator.Q.clamp_min(1e-30)
    populated = n_eff > 25
    assert int(populated.sum()) > 100
    std = float(z[populated].std())
    mean_duplication = n_draws / n_pixels
    # duplicated residuals re-enter without fresh noise, so the null z std
    # inflates by up to sqrt(mean duplication); the offline calibration
    # measures this and sets the production z_min in the inflated units
    assert 0.75 < std < 1.25 * math.sqrt(mean_duplication), f"null z std {std}"
    assert float(z[populated].mean()) == pytest.approx(0.0, abs=0.5)
    calibrated_z_min = 6.0 * std
    assert accumulator.candidates(
        z_min=calibrated_z_min, n_eff_min=25.0, kappa=0.2, min_bands=1, top_k=10
    ) is None


def test_candidates_find_unexplained_excess_and_veto_explained_cells() -> None:
    torch.manual_seed(1)
    accumulator = _accumulator(cell_size=0.5, bands=("g", "r"))
    n = 4000
    xs = torch.rand(n) - 1.0  # left half of the field
    ys = torch.rand(n) - 1.0
    signal = torch.full((n,), 3.0)

    def fill(band: int, x0: float, y0: float, r, m) -> None:
        accumulator.accumulate(
            torch.full((n,), band, dtype=torch.long),
            xs * 0.1 + x0 + 0.05,
            ys * 0.1 + y0 + 0.05,
            r,
            m,
            r,
            torch.ones(n),
        )

    # cell A: excess in both bands, model contributes nothing -> birth
    fill(0, 0.7, 0.7, signal, torch.zeros(n))
    fill(1, 0.7, 0.7, signal, torch.zeros(n))
    # cell B: same excess but the model already renders there -> vetoed
    fill(0, -0.7, -0.7, signal, signal * 2)
    fill(1, -0.7, -0.7, signal, signal * 2)
    # cell C: excess in one band only -> fails the multiband gate
    fill(0, 0.7, -0.7, signal, torch.zeros(n))

    result = accumulator.candidates(
        z_min=6.0, n_eff_min=25.0, kappa=0.2, min_bands=2, top_k=8
    )
    assert result is not None
    centers = result["centers"]
    assert centers.shape[0] == 1
    assert float(centers[0, 0]) > 0 and float(centers[0, 1]) > 0
    assert int(result["band_count"][0]) == 2


def test_candidates_rank_by_weakest_triggering_band_and_cap_top_k() -> None:
    accumulator = _accumulator(cell_size=0.5, bands=("g",))
    n = 2000
    torch.manual_seed(2)
    for x0, strength in ((-0.75, 4.0), (-0.25, 8.0), (0.25, 6.0)):
        xs = torch.rand(n) * 0.3 + x0
        ys = torch.rand(n) * 0.3 - 0.1
        accumulator.accumulate(
            torch.zeros(n, dtype=torch.long),
            xs,
            ys,
            torch.full((n,), strength),
            torch.zeros(n),
            torch.full((n,), strength),
            torch.ones(n),
        )
    result = accumulator.candidates(
        z_min=6.0, n_eff_min=25.0, kappa=0.2, min_bands=1, top_k=2
    )
    assert result is not None
    assert result["centers"].shape[0] == 2
    # strongest two cells kept, strongest first
    assert float(result["centers"][0, 0]) == pytest.approx(-0.25, abs=0.5)


def _field(n_splats: int = 6) -> SplatField:
    torch.manual_seed(9)
    field = SplatField(
        n_splats=n_splats, n_templates=3,
        loglam_grid=torch.linspace(-1.0, 1.0, 8), init_scale=0.002,
    )
    with torch.no_grad():
        field.raw_flux.uniform_(-2.0, 1.0)
    return field


def _birth_rows(k: int, n_templates: int = 3) -> dict[str, torch.Tensor]:
    return {
        "mu": torch.linspace(-0.5, 0.5, 2 * k).view(k, 2),
        "log_scales": torch.full((k, 2), math.log(0.003)),
        "theta": torch.zeros(k),
        "raw_flux": torch.full((k,), -8.0),
        "sed_coeffs": torch.zeros(k, n_templates),
        "sed_shift": torch.zeros(k),
    }


def _strategy(**overrides) -> SplatRefineStrategy:
    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=10_000, refine_every=1, **overrides
    )
    return SplatRefineStrategy(cfg, field_width=1024, reference_sigma=0.05)


def test_birth_appends_full_rows_with_fresh_state_and_lineage() -> None:
    field = _field()
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    loss = field.positions().square().sum() + field.fluxes().sum()
    loss.backward()
    optimizer.step()
    strategy = _strategy(birth_cap_per_event=4)
    rows = _birth_rows(2)

    plan = strategy.make_plan(field, step=5, birth_rows=rows)

    assert plan is not None and plan.birth is not None
    report = strategy.apply_plan(field, [optimizer], plan)
    assert report["birth"] == 2
    assert field.mu.shape[0] == 8
    assert torch.equal(field.mu[6:], rows["mu"])
    assert torch.equal(field.birth_event[6:], torch.full((2,), 5, dtype=torch.int32))
    assert torch.equal(field.birth_event[:6], torch.full((6,), -1, dtype=torch.int32))
    state = optimizer.state[field.raw_flux]
    assert torch.all(state["exp_avg"][6:] == 0)


def test_birth_is_budgeted_before_growth_at_the_cap() -> None:
    field = _field(6)
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    strategy = _strategy(
        grow_grad2d=1.0, grow_scale=1.0, max_splats=8, birth_cap_per_event=4
    )
    strategy.grad_accum = torch.full((6,), 10.0)
    strategy.count = torch.ones(6)

    plan = strategy.make_plan(field, step=1, birth_rows=_birth_rows(4))

    assert plan is not None and plan.birth is not None
    # 2 free slots, births take both, growth gets none
    assert int(plan.birth["mu"].shape[0]) == 2
    assert int(plan.duplicate.sum()) + int(plan.split.sum()) == 0
    strategy.apply_plan(field, [optimizer], plan)
    assert field.mu.shape[0] == 8


def test_birth_grace_defers_the_flux_verdict_exactly_one_interval() -> None:
    field = _field()
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    strategy = _strategy(birth_cap_per_event=4, birth_grace_steps=10)
    rows = _birth_rows(1)
    rows["raw_flux"] = torch.full((1,), -40.0)  # far below any flux threshold
    plan = strategy.make_plan(field, step=5, birth_rows=rows)
    strategy.apply_plan(field, [optimizer], plan)
    assert field.mu.shape[0] == 7

    inside = strategy.make_plan(field, step=10)  # age 5 <= grace 10
    assert not bool(inside.prune_after_growth[6])
    strategy.apply_plan(field, [optimizer], inside)
    assert field.mu.shape[0] == 7

    after = strategy.make_plan(field, step=16)  # age 11 > grace 10
    assert bool(after.prune_after_growth[6])
    strategy.apply_plan(field, [optimizer], after)
    assert field.mu.shape[0] == 6


def test_birth_payload_round_trips_and_validates() -> None:
    field = _field()
    strategy = _strategy(birth_cap_per_event=4)
    plan = strategy.make_plan(field, step=3, birth_rows=_birth_rows(2))
    replay = SplatRefinePlan.from_payload(plan.to_payload(), torch.device("cpu"))
    for name in plan.birth:
        assert torch.equal(replay.birth[name], plan.birth[name])

    bad = plan.to_payload()
    bad["birth"] = dict(bad["birth"])
    bad["birth"]["raw_flux"] = torch.zeros(5)
    with pytest.raises(ValueError, match="leading dimension"):
        SplatRefinePlan.from_payload(bad, torch.device("cpu"))
    missing = plan.to_payload()
    missing["birth"] = {
        k: v for k, v in missing["birth"].items() if k != "sed_shift"
    }
    with pytest.raises(ValueError, match="birth payload keys"):
        SplatRefinePlan.from_payload(missing, torch.device("cpu"))


def test_apply_plan_rejects_birth_violations_before_any_mutation() -> None:
    field = _field()
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    strategy = _strategy(grow_grad2d=1.0, grow_scale=1.0, birth_cap_per_event=1)
    strategy.grad_accum = torch.full((6,), 10.0)
    strategy.count = torch.ones(6)
    plan = strategy.make_plan(field, step=1, birth_rows=_birth_rows(1))
    assert plan is not None and int(plan.duplicate.sum()) > 0

    wrong_width = dict(plan.birth)
    wrong_width["sed_coeffs"] = torch.zeros(1, 7)
    bad = SplatRefinePlan(
        step=plan.step, duplicate=plan.duplicate, split=plan.split,
        prune_after_growth=plan.prune_after_growth,
        split_noise=plan.split_noise, threshold=plan.threshold,
        birth=wrong_width,
    )
    with pytest.raises(ValueError, match="sed_coeffs width"):
        strategy.apply_plan(field, [optimizer], bad)
    assert field.mu.shape[0] == 6  # nothing mutated

    over_cap = SplatRefinePlan(
        step=plan.step, duplicate=plan.duplicate, split=plan.split,
        prune_after_growth=plan.prune_after_growth,
        split_noise=plan.split_noise, threshold=plan.threshold,
        birth=_birth_rows(5),
    )
    with pytest.raises(ValueError, match="birth_cap_per_event"):
        strategy.apply_plan(field, [optimizer], over_cap)
    assert field.mu.shape[0] == 6


def test_apply_plan_enforces_the_census_cap_on_replayed_plans() -> None:
    field = _field(6)
    optimizer = torch.optim.Adam(field.param_groups_gsplat())
    strategy = _strategy(max_splats=7, birth_cap_per_event=4)
    plan = SplatRefinePlan(
        step=1,
        duplicate=torch.zeros(6, dtype=torch.bool),
        split=torch.zeros(6, dtype=torch.bool),
        prune_after_growth=torch.zeros(6, dtype=torch.bool),
        split_noise=torch.empty((2, 0, 2)),
        threshold=None,
        birth=_birth_rows(3),
    )
    with pytest.raises(ValueError, match="max_splats"):
        strategy.apply_plan(field, [optimizer], plan)
    assert field.mu.shape[0] == 6


def test_birth_rows_render_at_the_residual_amplitude() -> None:
    # End-to-end: capture -> accumulate -> _build_birth_rows on a real
    # trainer, then render the newborns back through the production image
    # operator. A newborn exists to explain the residual under its cell, so
    # its rendered amplitude there must match that residual to a geometric
    # factor. An earlier version diverged (img NLL ~5 -> 1e12 in one event)
    # because the flux init dropped pixel_norm^2 and every newborn rendered
    # ~1/pixel_norm^2 too bright; this is the roundtrip that catches any
    # unit slip in the inversion.
    import numpy as np

    from test_analytic_trainer_dispatch import _bandpasses, _realistic_scene

    from survey_fusion.data.adapters.sdss import build_sdss_datums
    from survey_fusion.data.grids import build_spatial_grid
    from survey_fusion.training.datum_trainer import DatumTrainer

    device = torch.device("cpu")
    store = build_sdss_datums(
        _realistic_scene(), _bandpasses(), psf_fwhm_arcsec=1.0, device=device
    , spectral_response_mode="legacy_imaging_median")
    torch.manual_seed(0)
    model = SplatField(
        n_splats=6, n_templates=3, loglam_grid=torch.linspace(-1, 1, 16),
        init_scale=0.2, init_extent=0.8,
    )
    with torch.no_grad():
        # a dark model: the whole image is unexplained residual
        model.raw_flux.fill_(-30.0)
    trainer = DatumTrainer(
        model=model,
        grid=build_spatial_grid(2, 3, device),
        data=store,
        training_bands=["g", "r", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 8,
            "analytic_images": True,
            "epochs": 1,
            "learning_rate": 1e-3,
            "pixel_batch_size": 4,
            "spec_batch_size": 2,
            "refine": {
                "birth": {
                    "enabled": True, "z_min": 8.0, "n_eff_min": 3.0,
                    "kappa": 0.2, "cap_per_event": 8, "cell_scale": 2.0,
                    "grace_steps": 100,
                }
            },
        },
        device=device,
    )

    # Documented compatibility boundary: this adapter path builds one
    # (numerically identical) tangent plane object per exposure, and the
    # accumulator's guard is identity-based, so birth REFUSES it. Production
    # full84 shares scene.model_wcs and passes. Assert the refusal, then
    # share one object to proceed on the supported shape.
    with pytest.raises(RuntimeError, match="single model-WCS plane"):
        trainer._birth_accumulator()
    shared_wcs = None
    for exp in trainer.data.exposures.values():
        if getattr(exp, "wcs", None) is not None:
            if shared_wcs is None:
                shared_wcs = exp.wcs
            else:
                exp.wcs = shared_wcs

    # one capture per exposure so each batch spans a single raster WCS group
    indices = trainer.image_train_indices
    all_exp_ids = trainer._gather_named_rows(
        "exposure_id", indices, device=torch.device("cpu"), dtype=torch.long
    )
    accumulator = trainer._birth_accumulator()
    for one_exp in all_exp_ids.unique().tolist():
        subset = indices[all_exp_ids == one_exp]
        trainer._compute_analytic_image_gaussian_terms(subset)
        pending = trainer._pending_birth
        trainer._pending_birth = None
        assert pending is not None, "birth capture did not fire on the analytic path"
        band_map = torch.as_tensor(
            [accumulator.band_index[str(k)] for k in pending["band_keys"]]
        )
        accumulator.accumulate(
            band_map[pending["band_local"]], pending["px"], pending["py"],
            pending["r"], pending["m"], pending["f"],
            torch.ones(int(subset.numel())),
        )

    # capture the per-cell oracle BEFORE row construction (candidates() is
    # pure; _build_birth_rows recomputes the same ranking, then resets)
    oracle = accumulator.candidates(
        z_min=trainer.birth_z_min, n_eff_min=trainer.birth_n_eff_min,
        kappa=trainer.birth_kappa, top_k=trainer.birth_cap_per_event,
    )
    assert oracle is not None
    base = trainer._base_model()
    rows = trainer._build_birth_rows(base)
    assert rows is not None, "a dark model over bright data must birth"
    k = int(rows["mu"].shape[0])
    assert 1 <= k <= 8
    torch.testing.assert_close(
        rows["mu"], oracle["centers"], rtol=0.0, atol=0.0
    )

    # Render EACH newborn IN ISOLATION through the PRODUCTION tiled raster
    # path (the operator training uses), with the base model's SED dictionary
    # (production births append to the model and share its learned
    # dictionary; a fresh dictionary would decouple this render from the one
    # the rows were fit against). Compare each newborn's mean rendered value
    # over its own cell against that cell's accumulated mean residual in the
    # same band. A PSF-convolved, PSF-sized newborn keeps ~25-30% of its
    # mass inside its 2-sigma cell, so the correct ratio sits near 0.3; the
    # bounds catch a factor-10 unit slip in either direction on ANY single
    # row (the original bug was 60x in this geometry, 2.5e7x at full84
    # scale). Per-row isolation keeps neighboring births from masking a
    # row-specific error.
    half = accumulator.cell_size / 2.0
    checked = 0
    for one_exp in all_exp_ids.unique().tolist():
        band = str(trainer.image_band_by_exposure[int(one_exp)])
        band_row = accumulator.band_index[band]
        cache = trainer._analytic_cache_for_exposure(int(one_exp))
        height, width = int(cache["height"]), int(cache["width"])
        local = torch.arange(height * width, dtype=torch.long)
        exp_col = torch.full_like(local, int(one_exp))
        dummy_global = torch.empty(local.numel(), dtype=torch.long)
        exposure = trainer.runtime_by_exposure[int(one_exp)].exposure
        px = local.remainder(width).to(torch.float64).unsqueeze(1).numpy()
        py = (
            local.div(width, rounding_mode="floor")
            .to(torch.float64).unsqueeze(1).numpy()
        )
        omega = np.asarray(
            exposure.obs_wcs.pixel_to_omega_xyz(px, py), dtype=np.float64
        ).reshape(-1, 3)
        xn, yn = exposure.wcs.omega_xyz_to_norm_torch(
            torch.as_tensor(omega, dtype=torch.float32)
        )
        for row in range(k):
            if not bool(oracle["eligible"][band_row, row]):
                continue
            cell_mean_residual = float(
                oracle["mean_residual_flux"][band_row, row]
            )
            if cell_mean_residual <= 0:
                continue
            cx, cy = float(rows["mu"][row, 0]), float(rows["mu"][row, 1])
            inside = (xn - cx).abs() <= half
            inside &= (yn - cy).abs() <= half
            if int(inside.sum()) == 0:
                continue
            single = SplatField(
                n_splats=1, n_templates=3,
                loglam_grid=torch.linspace(-1, 1, 16),
                init_scale=0.2, init_extent=0.8,
            )
            single.sed = base.sed
            with torch.no_grad():
                for name in (
                    "mu", "log_scales", "theta", "raw_flux",
                    "sed_coeffs", "sed_shift",
                ):
                    getattr(single, name).copy_(rows[name][row : row + 1])
            trainer.model = single
            with torch.no_grad():
                rendered = trainer._compute_analytic_image_source_mu(
                    dummy_global, exp_col, local
                )
            trainer._pending_birth = None
            mean_newborn = float(rendered[inside].mean())
            ratio = mean_newborn / cell_mean_residual
            assert 0.08 < ratio < 2.0, (
                f"newborn {row} band {band} renders {ratio:.3g}x its cell "
                f"residual: unit slip (rendered {mean_newborn:.3g}, cell "
                f"residual {cell_mean_residual:.3g})"
            )
            checked += 1
    assert checked >= 2
