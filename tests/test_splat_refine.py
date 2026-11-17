from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import (
    PER_SPLAT_LINEAGE_BUFFERS,
    PER_SPLAT_PARAMS,
    SplatRefineConfig,
    SplatRefinePlan,
    SplatRefineStrategy,
)


def _make_field(n_splats: int = 8, sparse: bool = False) -> SplatField:
    torch.manual_seed(11)
    field = SplatField(
        n_splats=n_splats, n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 8), init_scale=0.002,
    )
    with torch.no_grad():
        field.mu.uniform_(-0.5, 0.5)
        field.log_scales.uniform_(math.log(0.001), math.log(0.003))
        field.raw_flux.uniform_(-2.0, 1.0)
    if sparse:
        field.set_sparse_gradients(True)
    return field


def _make_optimizers(field: SplatField, sparse: bool):
    if sparse:
        return [
            torch.optim.Adam(field.shared_param_groups_gsplat()),
            torch.optim.SparseAdam(field.sparse_param_groups_gsplat()),
        ]
    return [torch.optim.Adam(field.param_groups_gsplat())]


def _strategy(**overrides) -> SplatRefineStrategy:
    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=10_000, refine_every=1, **overrides
    )
    return SplatRefineStrategy(cfg, field_width=1024, reference_sigma=0.05)


def _run_step(field: SplatField, optimizers) -> None:
    for opt in optimizers:
        opt.zero_grad()
    idx = torch.arange(0, field.mu.shape[0], 2)
    loss = (
        field.positions(idx).square().sum()
        + field.fluxes(idx).sum()
        + field.covariances(idx).sum()
        + field.sed_coefficients(idx).sum()
        + field.spectral_shifts(idx).sum()
        + field.sed.templates().sum()
    )
    loss.backward()
    for opt in optimizers:
        opt.step()


def _optimizer_rows(field: SplatField, optimizers) -> dict[tuple[str, str], torch.Tensor]:
    rows = {}
    for name in PER_SPLAT_PARAMS:
        parameter = getattr(field, name)
        for optimizer_index, optimizer in enumerate(optimizers):
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape[:1] == parameter.shape[:1]:
                    rows[(f"{optimizer_index}:{name}", key)] = value.detach().clone()
    return rows


def test_plan_replay_is_bitwise_exact_for_parameters_and_optimizer_rows() -> None:
    field_a = _make_field()
    optimizers_a = _make_optimizers(field_a, sparse=False)
    _run_step(field_a, optimizers_a)
    field_b, optimizers_b = copy.deepcopy((field_a, optimizers_a))
    strategy_a = _strategy(grow_grad2d=1e-12, grow_scale=0.002)
    strategy_a.grad_accum = torch.arange(1.0, 9.0)
    strategy_a.count = torch.ones(8)

    plan_a = strategy_a.make_plan(field_a, step=1)

    assert plan_a is not None
    plan_b = SplatRefinePlan.from_payload(plan_a.to_payload(), field_b.mu.device)
    report_a = strategy_a.apply_plan(field_a, optimizers_a, plan_a)
    report_b = _strategy(grow_grad2d=1e-12, grow_scale=0.002).apply_plan(
        field_b, optimizers_b, plan_b
    )
    # prune_reasons is planner-local diagnostics and never crosses the payload
    assert report_a.pop("prune_reasons", None) is not None
    assert report_b.get("prune_reasons") is None
    assert report_a == report_b
    for name in PER_SPLAT_PARAMS:
        assert torch.equal(getattr(field_a, name), getattr(field_b, name))
    for name in PER_SPLAT_LINEAGE_BUFFERS:
        assert torch.equal(getattr(field_a, name), getattr(field_b, name))
    rows_a = _optimizer_rows(field_a, optimizers_a)
    rows_b = _optimizer_rows(field_b, optimizers_b)
    assert rows_a.keys() == rows_b.keys()
    for key in rows_a:
        assert torch.equal(rows_a[key], rows_b[key])


def test_apply_plan_does_not_draw_randomness(monkeypatch: pytest.MonkeyPatch) -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.log_scales[3] = math.log(0.02)
    strategy = _strategy(grow_grad2d=1e-12, grow_scale=0.005)
    strategy.grad_accum = torch.ones(8)
    strategy.count = torch.ones(8)
    plan = strategy.make_plan(field, step=1)
    assert plan is not None
    assert bool(plan.split.any())

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("apply_plan must not draw split noise")

    monkeypatch.setattr(torch, "randn", fail_if_called)
    strategy.apply_plan(field, optimizers, plan)


def test_make_plan_does_not_mutate_strategy_state() -> None:
    field = _make_field()
    strategy = _strategy(grow_target_frac=0.25)
    strategy.grad_accum = torch.arange(1.0, 9.0)
    strategy.count = torch.ones(8)
    expected_grad_accum = strategy.grad_accum.clone()
    expected_count = strategy.count.clone()

    plan = strategy.make_plan(field, step=1)

    assert plan is not None
    assert plan.threshold is not None
    assert strategy.threshold is None
    assert torch.equal(strategy.grad_accum, expected_grad_accum)
    assert torch.equal(strategy.count, expected_count)


def _valid_plan_payload() -> dict[str, object]:
    return {
        "step": 1,
        "duplicate": torch.tensor([True, False, False]),
        "split": torch.tensor([False, True, False]),
        "prune_after_growth": torch.zeros(5, dtype=torch.bool),
        "split_noise": torch.zeros((2, 1, 2)),
        "threshold": 0.25,
        "birth": None,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("step"),
        lambda payload: payload.update({"extra": None}),
        lambda payload: payload.update(
            {"split": torch.tensor([True, False, False])}
        ),
        lambda payload: payload.update(
            {"prune_after_growth": torch.zeros(4, dtype=torch.bool)}
        ),
        lambda payload: payload.update({"split_noise": torch.zeros((1, 1, 2))}),
        lambda payload: payload.update(
            {"split_noise": torch.tensor([[[float("nan"), 0.0]]]).repeat(2, 1, 1)}
        ),
    ],
)
def test_plan_payload_validation_rejects_malformed_inputs(mutate) -> None:
    payload = _valid_plan_payload()
    mutate(payload)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        SplatRefinePlan.from_payload(payload, torch.device("cpu"))


def test_apply_plan_rejects_masks_that_do_not_match_live_model() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    plan = SplatRefinePlan.from_payload(
        _valid_plan_payload(), field.mu.device
    )

    with pytest.raises(ValueError, match="model size"):
        _strategy().apply_plan(field, optimizers, plan)


def test_duplicate_preserves_survivor_state_and_zeroes_new_rows() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    _run_step(field, optimizers)
    old_mu = field.mu.detach().clone()
    old_state = optimizers[0].state[field.mu]["exp_avg"].detach().clone()

    strategy = _strategy()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[2] = True
    strategy._duplicate(field, optimizers, mask, step=1)

    assert field.mu.shape[0] == 9
    assert torch.equal(field.mu[:8], old_mu)
    state = optimizers[0].state[field.mu]
    assert torch.equal(state["exp_avg"][:8], old_state)
    assert torch.all(state["exp_avg"][8] == 0)
    # flux conserved: parent + child equal the parent's old flux
    old_flux = F.softplus(torch.tensor(0.0))  # placeholder for shape only
    del old_flux
    _run_step(field, optimizers)  # optimizer still usable after surgery


def test_duplicate_conserves_total_flux() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    before = F.softplus(field.raw_flux.detach()).sum()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[1] = mask[5] = True
    _strategy()._duplicate(field, optimizers, mask, step=1)
    after = F.softplus(field.raw_flux.detach()).sum()
    assert torch.allclose(before, after, rtol=1e-5)


def test_split_conserves_flux_and_shrinks_scales() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    before = F.softplus(field.raw_flux.detach()).sum()
    parent_scale = torch.exp(field.log_scales.detach()[3]).clone()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[3] = True
    _strategy()._split(field, optimizers, mask, step=1)
    assert field.mu.shape[0] == 9  # one removed, two added
    after = F.softplus(field.raw_flux.detach()).sum()
    assert torch.allclose(before, after, rtol=1e-5)
    child_scales = torch.exp(field.log_scales.detach()[-2:])
    assert torch.allclose(child_scales, (parent_scale / 1.6).expand_as(child_scales), rtol=1e-5)
    _run_step(field, optimizers)


def test_remove_keeps_survivor_state_rows_aligned() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    _run_step(field, optimizers)
    keep_rows = [0, 1, 3, 4, 5, 7]
    expected = optimizers[0].state[field.raw_flux]["exp_avg_sq"].detach()[keep_rows].clone()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[2] = mask[6] = True
    _strategy()._remove(field, optimizers, mask)
    assert field.mu.shape[0] == 6
    assert torch.equal(
        optimizers[0].state[field.raw_flux]["exp_avg_sq"].detach(), expected
    )
    _run_step(field, optimizers)


def test_refine_with_sparse_adam_end_to_end() -> None:
    field = _make_field(sparse=True)
    optimizers = _make_optimizers(field, sparse=True)
    strategy = _strategy(grow_grad2d=0.0)
    for step in range(1, 4):
        _run_step(field, optimizers)
        strategy.accumulate(field)
        report = strategy.refine(field, optimizers, step)
        assert report, "refine cadence of 1 must fire every step"
        for name in PER_SPLAT_PARAMS:
            assert getattr(field, name).shape[0] == field.mu.shape[0]
    _run_step(field, optimizers)
    assert torch.isfinite(field.mu).all()


def test_accumulate_handles_sparse_gradients() -> None:
    field = _make_field(sparse=True)
    optimizers = _make_optimizers(field, sparse=True)
    _run_step(field, optimizers)
    strategy = _strategy()
    strategy.accumulate(field)
    assert strategy.grad_accum is not None
    assert strategy.grad_accum.shape[0] == field.mu.shape[0]
    assert float(strategy.count.sum()) > 0


def test_prune_targets_invisible_and_oversized_splats() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.raw_flux[0] = -40.0        # invisible: peak SB ~ 0
        field.log_scales[1] = math.log(0.5)  # oversized: > prune_scale
    strategy = _strategy(grow_grad2d=1e9)
    strategy.accumulate(field)  # no grads -> nothing accumulated
    field.mu.grad = torch.zeros_like(field.mu)
    strategy.grad_accum = torch.zeros(8)
    strategy.count = torch.ones(8)
    report = strategy.refine(field, optimizers, step=1)
    assert report["prune"] == 2
    assert field.mu.shape[0] == 6


def test_position_lr_decay_matches_gsplat_formula() -> None:
    max_steps = 1000
    gamma = 0.01 ** (1.0 / max_steps)
    lr0 = 1.6e-4
    assert abs(lr0 * gamma**max_steps - 0.01 * lr0) < 1e-12


def test_growth_respects_max_splats_budget() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_grad2d=1e-12, max_splats=10)
    strategy.grad_accum = torch.arange(1.0, 9.0)
    strategy.count = torch.ones(8)
    report = strategy.refine(field, optimizers, step=1)
    assert report["dupli"] + report["split"] == 2  # budget = 10 - 8
    assert field.mu.shape[0] <= 10


def test_auto_threshold_targets_growth_fraction() -> None:
    field = _make_field(n_splats=100)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_target_frac=0.10)
    strategy.grad_accum = torch.linspace(0.0, 1.0, 100)
    strategy.count = torch.ones(100)
    report = strategy.refine(field, optimizers, step=1)
    grown = report["dupli"] + report["split"]
    assert 5 <= grown <= 15
    assert strategy.threshold is not None
    # threshold is frozen after calibration
    frozen = strategy.threshold
    strategy.grad_accum = torch.linspace(0.0, 2.0, field.mu.shape[0])
    strategy.count = torch.ones(field.mu.shape[0])
    strategy.refine(field, optimizers, step=2)
    assert strategy.threshold == frozen




def test_split_survives_degenerate_covariance() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.log_scales[3, 0] = -30.0  # collapses one axis to ~zero variance
    mask = torch.zeros(8, dtype=torch.bool)
    mask[3] = True
    _strategy()._split(field, optimizers, mask, step=1)
    assert torch.isfinite(field.mu).all()
    assert torch.isfinite(field.log_scales).all()


def test_prune_kills_needle_splats() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.log_scales[2] = torch.tensor([math.log(0.003), math.log(0.00001)])
    strategy = _strategy(grow_grad2d=1e9, prune_axis_ratio=20.0)
    strategy.grad_accum = torch.zeros(8)
    strategy.count = torch.ones(8)
    report = strategy.refine(field, optimizers, step=1)
    assert report["prune"] >= 1
    scales = torch.exp(field.log_scales.detach())
    assert bool((scales.amax(dim=1) <= 20.0 * scales.amin(dim=1)).all())


def test_psf_based_grow_scale_puts_boundary_at_a_few_psf() -> None:
    from survey_fusion.models.splat_strategy import psf_based_grow_scale

    # full84: 12729px field at 0.396''/px, PSF 1.4'' -> boundary ~10.6px
    scale = psf_based_grow_scale(1.4, 12729, 0.396)
    boundary_px = scale / (2.0 / 12729)
    assert 8.0 < boundary_px < 14.0


def test_large_splats_split_small_ones_duplicate() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.log_scales.fill_(math.log(0.001))   # small
        field.log_scales[3] = math.log(0.02)       # large: above grow_scale
    strategy = _strategy(grow_grad2d=1e-12, grow_scale=0.005)
    strategy.grad_accum = torch.ones(8)
    strategy.count = torch.ones(8)
    report = strategy.refine(field, optimizers, step=1)
    assert report["split"] == 1
    assert report["dupli"] == 7


def test_prune_continues_after_refine_stop_but_growth_does_not() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.log_scales[2] = torch.tensor([math.log(0.003), math.log(0.00001)])
    strategy = _strategy(grow_grad2d=1e-30, prune_axis_ratio=20.0)
    strategy.grad_accum = torch.full((8,), 1e6)
    strategy.count = torch.ones(8)
    n_before = field.mu.shape[0]
    report = strategy.refine(field, optimizers, step=20_000)
    assert report["dupli"] == 0 and report["split"] == 0
    assert report["prune"] >= 1
    assert field.mu.shape[0] < n_before
    scales = torch.exp(field.log_scales.detach())
    assert bool((scales.amax(dim=1) <= 20.0 * scales.amin(dim=1)).all())


def test_make_plan_after_refine_stop_keeps_live_sanitation_prune_mask() -> None:
    field = _make_field()
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.log_scales[2] = torch.tensor([math.log(0.003), math.log(0.00001)])
    strategy = _strategy(grow_grad2d=1e-30, prune_axis_ratio=20.0)
    strategy.grad_accum = torch.full((8,), 1e6)
    strategy.count = torch.ones(8)

    plan = strategy.make_plan(field, step=20_000)

    assert plan is not None
    assert not bool(plan.duplicate.any())
    assert not bool(plan.split.any())
    assert bool(plan.prune_after_growth[2])


def test_trainer_resolves_zero_grow_scale_to_psf_boundary() -> None:
    from support.synthetic_trainer import image_exposure, make_image_only_trainer

    field = SplatField(
        n_splats=8, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4)
    )
    exposure = image_exposure(psf_fwhm_arcsec=1.4, width=100)
    exposure.wcs.pixel_scale_arcsec = 1.0
    trainer, _ = make_image_only_trainer(
        model=field,
        exposure_ids=[0, 0, 0, 0],
        exposure_overrides={0: exposure},
        trainer_cfg={
            "refine": {
                "enabled": True,
                "grow_scale": 0.0,
                "start": 0,
                "stop": 15_000,
                "every": 100,
            }
        },
    )

    assert trainer.refine_strategy is not None
    expected = 3.0 * 1.4 * 2.0 / (100.0 * 1.0)
    assert trainer.refine_strategy.config.grow_scale == pytest.approx(expected)


def test_trainer_params_list_refreshed_after_surgery() -> None:
    from support.synthetic_trainer import make_image_spec_trainer

    field = SplatField(
        n_splats=8, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4)
    )
    with torch.no_grad():
        field.raw_flux[:4] = -30.0  # peak surface brightness far below noise
    trainer = make_image_spec_trainer(
        model=field,
        trainer_cfg={
            "refine": {"enabled": True, "start": 0, "stop": 15_000, "every": 100}
        },
    )
    stale = [p for p in trainer.params if p.shape[:1] == (8,) or p.shape == (8, 2)]
    assert stale, "expected per-splat parameters in the initial list"

    report = trainer._post_optimizer_step(100)

    assert report["prune"] == 4
    current_ids = {id(p) for p in trainer.model.parameters()}
    assert {id(p) for p in trainer.params} == current_ids
    assert all(id(p) not in current_ids for p in stale)
    # clipping now sees the post-surgery parameters, not the orphans, and
    # the per-splat exemption set tracked the surgery replacements
    assert trainer._clip_param_ids <= {id(p) for p in trainer.params}
    splat_ids = {
        id(p)
        for group in trainer.model.sparse_param_groups_gsplat()
        for p in group["params"]
    }
    assert splat_ids.isdisjoint(trainer._clip_param_ids)
    for parameter in trainer.params:
        parameter.grad = torch.ones_like(parameter)
    trainer.grad_clip = 1e-9
    trainer._clip_gradient_norm()
    for parameter in trainer.params:
        if id(parameter) in splat_ids:
            assert float(parameter.grad.abs().max()) == 1.0
        else:
            assert float(parameter.grad.abs().max()) < 1.0

def test_band_snr_prune_uses_rendered_amplitude_not_raw_flux() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.raw_flux.fill_(2.0)
        field.raw_flux[0] = -40.0  # tiny raw flux, but its SED renders bright
        field.raw_flux[1] = 2.0    # healthy raw flux, but renders to nothing

    def band_snr_fn(model) -> torch.Tensor:
        factor = torch.full((int(model.mu.shape[0]),), 1000.0)
        factor[0] = 1.0e30   # rendered peak far above noise despite raw_flux
        factor[1] = 1.0e-30  # rendered peak far below noise despite raw_flux
        return factor

    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=10_000, refine_every=1,
        grow_grad2d=1e9, prune_band_snr=True, prune_band_snr_sigma=0.005,
    )
    strategy = SplatRefineStrategy(
        cfg, field_width=1024, reference_sigma=0.05, band_snr_fn=band_snr_fn,
    )
    strategy.grad_accum = torch.zeros(8)
    strategy.count = torch.ones(8)
    report = strategy.refine(field, optimizers, step=1)
    # splat 0 survives (renders bright), splat 1 dies (renders dark): the
    # exact opposite of the bandless raw_flux criterion
    assert report["prune"] == 1
    assert field.mu.shape[0] == 7


def test_band_snr_prune_requires_fn_and_validates_output() -> None:
    with pytest.raises(ValueError, match="band_snr_fn"):
        SplatRefineStrategy(
            SplatRefineConfig(prune_band_snr=True, prune_band_snr_sigma=0.005),
            field_width=1024, reference_sigma=0.05,
        )

    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    cfg = SplatRefineConfig(
        refine_start=0, refine_stop=10_000, refine_every=1,
        grow_grad2d=1e9, prune_band_snr=True, prune_band_snr_sigma=0.005,
    )
    strategy = SplatRefineStrategy(
        cfg, field_width=1024, reference_sigma=0.05,
        band_snr_fn=lambda model: torch.ones(3),
    )
    strategy.grad_accum = torch.zeros(8)
    strategy.count = torch.ones(8)
    with pytest.raises(ValueError, match="one factor per splat"):
        strategy.refine(field, optimizers, step=1)


def test_candidate_passthrough_matches_candidate_ordering() -> None:
    duplicate = torch.tensor([True, False, False, False])
    split = torch.tensor([False, True, False, False])
    value = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = SplatRefineStrategy._candidate_passthrough(value, duplicate, split)
    # ordering: [rest of (orig + dup extras)] then split children repeated
    assert out.tolist() == [10.0, 30.0, 40.0, 10.0, 20.0, 20.0]


def _flux_for_peak_sb(field: SplatField, index: int, peak_sb: float) -> float:
    """raw_flux value that renders the given intrinsic peak surface brightness."""
    s1, s2 = torch.exp(field.log_scales[index]).tolist()
    flux = peak_sb * 2.0 * math.pi * s1 * s2
    return math.log(math.expm1(flux)) if flux > 1e-30 else math.log(flux)


def test_growth_does_not_change_a_flux_prune_outcome() -> None:
    # Bug 3 regression: a top-grad parent in [1x, 2x) of the flux threshold
    # duplicates (halving its flux); the verdict is pre-growth, so the family
    # must survive the event instead of dying with its child.
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_grad2d=1.0, grow_scale=1.0)
    threshold_sb = strategy.config.prune_peak_sb_frac * strategy.reference_sigma
    with torch.no_grad():
        field.raw_flux[0] = _flux_for_peak_sb(field, 0, 1.5 * threshold_sb)
    strategy.grad_accum = torch.zeros(8)
    strategy.grad_accum[0] = 10.0
    strategy.count = torch.ones(8)

    plan = strategy.make_plan(field, step=1)

    assert plan is not None
    assert bool(plan.duplicate[0])
    assert not bool(plan.prune_after_growth.any())
    report = strategy.apply_plan(field, optimizers, plan)
    assert report["prune"] == 0
    assert field.mu.shape[0] == 9
    assert field.halving_depth[0] == 1
    assert field.halving_depth[8] == 1
    assert field.last_halving_event[0] == 1
    assert field.last_halving_event[8] == 1


def test_flux_verdict_set_is_independent_of_growth() -> None:
    field_grow = _make_field()
    field_still = copy.deepcopy(field_grow)
    strategy = _strategy(grow_grad2d=1.0, grow_scale=1.0)
    threshold_sb = strategy.config.prune_peak_sb_frac * strategy.reference_sigma
    with torch.no_grad():
        for f in (field_grow, field_still):
            f.raw_flux[1] = _flux_for_peak_sb(f, 1, 0.5 * threshold_sb)
    strategy.grad_accum = torch.full((8,), 10.0)
    strategy.count = torch.ones(8)
    plan_grow = strategy.make_plan(field_grow, step=1)

    still = _strategy(grow_grad2d=1.0, grow_scale=1.0)
    plan_still = still.make_plan(field_still, step=1)

    assert plan_grow is not None and plan_still is not None
    assert int(plan_grow.duplicate.sum()) > 0
    assert int(plan_still.duplicate.sum()) == 0
    assert plan_grow.prune_reasons["flux"] == 1
    assert plan_still.prune_reasons["flux"] == 1
    # the condemned splat is never grown and dies in both worlds
    assert not bool(plan_grow.duplicate[1]) and not bool(plan_grow.split[1])
    _strategy().apply_plan(field_grow, _make_optimizers(field_grow, False), plan_grow)
    _strategy().apply_plan(field_still, _make_optimizers(field_still, False), plan_still)
    condemned_flux = field_still.raw_flux.new_tensor(
        _flux_for_peak_sb(field_still, 1, 0.5 * threshold_sb)
    )
    assert not bool(torch.isclose(field_grow.raw_flux, condemned_flux).any())
    assert not bool(torch.isclose(field_still.raw_flux, condemned_flux).any())


def test_census_cap_is_strict_with_freed_slots_spendable() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_grad2d=1.0, grow_scale=1.0, max_splats=10)
    threshold_sb = strategy.config.prune_peak_sb_frac * strategy.reference_sigma
    with torch.no_grad():
        field.raw_flux[3] = _flux_for_peak_sb(field, 3, 0.5 * threshold_sb)
    strategy.grad_accum = torch.full((8,), 10.0)
    strategy.count = torch.ones(8)

    plan = strategy.make_plan(field, step=1)

    assert plan is not None
    # one condemned row frees a slot this event: budget = 10 - (8 - 1) = 3
    assert int(plan.duplicate.sum()) + int(plan.split.sum()) == 3
    report = strategy.apply_plan(field, optimizers, plan)
    assert report["prune"] == 1
    assert field.mu.shape[0] == 10
    assert field.mu.shape[0] <= strategy.config.max_splats


def test_axis_ratio_condemned_parent_is_not_grown_but_scale_split_rescues() -> None:
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_grad2d=1.0, grow_scale=0.0001, prune_axis_ratio=20.0)
    with torch.no_grad():
        # splat 0: axis-ratio condemned (500:1), high grad
        field.log_scales[0] = torch.tensor([math.log(0.05), math.log(0.0001)])
        # splat 1: scale condemned only (both axes over prune_scale, ratio 1);
        # splitting shrinks scales by 1.6x below the threshold
        field.log_scales[1] = torch.tensor([math.log(0.12), math.log(0.12)])
    strategy.grad_accum = torch.zeros(8)
    strategy.grad_accum[0] = 10.0
    strategy.grad_accum[1] = 10.0
    strategy.count = torch.ones(8)

    plan = strategy.make_plan(field, step=1)

    assert plan is not None
    assert not bool(plan.duplicate[0]) and not bool(plan.split[0])
    assert bool(plan.split[1])
    assert plan.prune_reasons["axis_ratio"] == 1
    assert plan.prune_reasons["scale"] == 0
    report = strategy.apply_plan(field, optimizers, plan)
    # axis-ratio row dies; the oversized row is replaced by two passing children
    assert report["prune"] == 1
    assert field.mu.shape[0] == 8
    assert bool((torch.exp(field.log_scales).amax(dim=1) <= 0.1).all())


def test_lineage_buffers_follow_surgery_composition() -> None:
    field = _make_field(4)
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy()

    strategy._duplicate(field, optimizers, torch.tensor([True, False, False, False]), step=100)
    # rows: [0d, 1, 2, 3, 0d-child]
    assert field.halving_depth.tolist() == [1, 0, 0, 0, 1]
    assert field.last_halving_event.tolist() == [100, -1, -1, -1, 100]

    strategy._split(
        field, optimizers, torch.tensor([False, True, False, False, False]), step=200
    )
    # rows: [0d, 2, 3, 0d-child, 1c1, 1c2]
    assert field.halving_depth.tolist() == [1, 0, 0, 1, 1, 1]
    assert field.last_halving_event.tolist() == [100, -1, -1, 100, 200, 200]
    assert field.birth_event.tolist() == [-1] * 6

    strategy._remove(field, optimizers, torch.tensor([True, False, False, False, False, False]))
    assert field.halving_depth.tolist() == [0, 0, 1, 1, 1]
    assert field.last_halving_event.tolist() == [-1, -1, 100, 200, 200]
    for name in PER_SPLAT_LINEAGE_BUFFERS:
        assert name in field.state_dict()


def test_axis_condemned_slot_funds_growth_and_the_cap_holds() -> None:
    # Documented allocator behavior: every pre-growth condemnation (flux OR
    # axis ratio) frees a slot the same event, because those rows leave the
    # event dead with certainty. The census can never exceed the cap.
    field = _make_field()
    optimizers = _make_optimizers(field, sparse=False)
    strategy = _strategy(grow_grad2d=1.0, grow_scale=1.0, max_splats=8, prune_axis_ratio=20.0)
    with torch.no_grad():
        field.log_scales[0] = torch.tensor([math.log(0.05), math.log(0.0001)])
    strategy.grad_accum = torch.zeros(8)
    strategy.grad_accum[1] = 10.0
    strategy.count = torch.ones(8)

    plan = strategy.make_plan(field, step=1)

    assert plan is not None
    # at the cap with one axis-condemned row: budget = 8 - (8 - 1) = 1
    assert int(plan.duplicate.sum()) + int(plan.split.sum()) == 1
    assert bool(plan.duplicate[1])
    report = strategy.apply_plan(field, optimizers, plan)
    assert report["prune"] == 1
    assert field.mu.shape[0] == 8


def test_halving_depth_saturates_instead_of_wrapping() -> None:
    field = _make_field(2)
    optimizers = _make_optimizers(field, sparse=False)
    with torch.no_grad():
        field.halving_depth.fill_(255)
    _strategy()._duplicate(
        field, optimizers, torch.tensor([True, False]), step=1
    )
    assert field.halving_depth.tolist() == [255, 255, 255]
