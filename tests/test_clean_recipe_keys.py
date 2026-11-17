"""Native config keys that replaced the earlier runtime monkeypatches."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.synthetic_trainer import (  # noqa: E402
    ConstantKernelFactory, SyntheticStore, image_exposure, make_batch, spec_exposure,
)
from survey_fusion.data.grids import build_spatial_grid  # noqa: E402
from survey_fusion.models.splat_field import SplatField  # noqa: E402
from survey_fusion.models.splat_strategy import SplatRefineConfig, SplatRefineStrategy  # noqa: E402
from survey_fusion.pgm.contracts import DatumKind  # noqa: E402
from survey_fusion.pgm.likelihoods import population_weighted_modality_nll  # noqa: E402
from survey_fusion.training.datum_trainer import DatumTrainer, _optional_axis_ratio  # noqa: E402

IMAGE, SPECTRUM = int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN)


def _store() -> SyntheticStore:
    batch = make_batch(
        y=torch.zeros(10),
        exposure_id=torch.tensor([0, 0, 0, 0, 2, 2, 2, 3, 3, 3]),
        datum_kind=torch.tensor([IMAGE] * 4 + [SPECTRUM] * 6),
        datum_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2]),
    )
    lam = np.linspace(5000.0, 5200.0, 3)
    return SyntheticStore(
        exposures={
            0: image_exposure(band="r", psf_fwhm_arcsec=1.0),
            2: spec_exposure(psf_fwhm_arcsec=1.5, lam_spec=lam),
            3: spec_exposure(psf_fwhm_arcsec=1.5, lam_spec=lam),
        },
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=4,
        n_spec=6,
        spec_label_by_exposure={2: "S0001-00001-0001", 3: "S0001-00001-0002"},
    )


def _trainer(**overrides) -> DatumTrainer:
    torch.manual_seed(5)
    field = SplatField(n_splats=4, n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 4))
    trainer_cfg = {
        "kernel_samples_k": 8, "replica_samples_k": 1, "epochs": 0, "learning_rate": 1e-3,
        "apply_spatial_psf": False, **overrides,
    }
    return DatumTrainer(
        model=field, grid=build_spatial_grid(2, 2, torch.device("cpu")), data=_store(),
        training_bands=["r"], held_out_band=None, lambda_step=20.0,
        trainer_cfg=trainer_cfg, device=torch.device("cpu"),
    )


def _nll(spectrum_weight=None):
    kwargs = dict(
        image_nll_sum=torch.tensor(7.0, dtype=torch.float64), image_sample_count=10,
        image_population_count=1000,
        spectrum_nll_sum=torch.tensor(3.0, dtype=torch.float64), spectrum_sample_count=4,
        spectrum_population_count=20,
    )
    if spectrum_weight is not None:
        kwargs["spectrum_weight"] = spectrum_weight
    return population_weighted_modality_nll(**kwargs)


def test_spectrum_weight_scales_only_the_spectral_term() -> None:
    expected = (1000.0 * 7.0 / 10.0 + 5.0 * 20.0 * 3.0 / 4.0) / 1020.0
    assert float(_nll(5.0)) == pytest.approx(expected, rel=1e-15)


def test_unit_spectrum_weight_is_bitwise_the_unweighted_objective() -> None:
    assert torch.equal(_nll(1.0), _nll())


@pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, math.nan])
def test_invalid_spectrum_weight_is_refused(weight: float) -> None:
    with pytest.raises(ValueError, match="spectrum_weight"):
        _nll(weight)


def test_spectral_bin_weight_defaults_to_five() -> None:
    trainer = _trainer()
    assert trainer.spectral_bin_weight == 5.0
    summary = trainer.scientific_modality_summary()
    assert summary["spectral_bin_weight"] == 5.0
    assert summary["spectrum_weight"] == pytest.approx(5.0 * 6 / 10)


@pytest.mark.parametrize("weight", [0.0, -2.0, math.inf])
def test_spectral_bin_weight_must_be_positive_and_finite(weight: float) -> None:
    with pytest.raises(ValueError, match="spectral_bin_weight"):
        _trainer(spectral_bin_weight=weight)


def test_excluded_spectrum_leaves_the_objective_but_stays_loaded() -> None:
    trainer = _trainer(exclude_spectrum_labels=["S0001-00001-0002"])
    assert trainer.spec_exposure_ids == [2]
    assert trainer.excluded_spectrum_ids == (3,)
    trained = torch.index_select(trainer.data.batch.exposure_id, 0, trainer.spec_train_indices)
    assert set(trained.tolist()) == {2}
    assert int(trainer.spec_train_indices.numel()) == 3
    assert 3 in trainer.runtime_by_exposure


def test_no_spectrum_exclusion_keeps_every_spectrum() -> None:
    trainer = _trainer()
    assert trainer.spec_exposure_ids == [2, 3]
    assert trainer.excluded_spectrum_ids == ()
    assert int(trainer.spec_train_indices.numel()) == 6


def test_unknown_spectrum_label_is_refused() -> None:
    with pytest.raises(ValueError, match="not labels of spectra loaded"):
        _trainer(exclude_spectrum_labels=["S9999-99999-0001"])


def test_duplicate_spectrum_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        _trainer(exclude_spectrum_labels=["S0001-00001-0001", "S0001-00001-0001"])


def test_adam_eps_defaults_to_pytorch_value_and_reaches_every_group() -> None:
    default = _trainer()
    assert default.adam_eps == 1e-8
    assert all(group["eps"] == 1e-8 for group in default.optimizer.param_groups)
    tiny = _trainer(adam_eps=1e-15)
    assert all(group["eps"] == 1e-15 for group in tiny.optimizer.param_groups)


@pytest.mark.parametrize("eps", [0.0, -1e-8, math.inf])
def test_adam_eps_must_be_positive_and_finite(eps: float) -> None:
    with pytest.raises(ValueError, match="adam_eps"):
        _trainer(adam_eps=eps)


def _strategy(ratio):
    return SplatRefineStrategy(
        SplatRefineConfig(prune_axis_ratio=ratio), field_width=1024, reference_sigma=0.05
    )


def test_axis_ratio_prune_is_off_by_default() -> None:
    assert SplatRefineConfig().prune_axis_ratio is None
    log_scales = torch.log(torch.tensor([[0.05, 0.0001], [0.01, 0.01]]))
    assert not _strategy(None)._axis_ratio_prune_mask(log_scales).any()


def test_axis_ratio_prune_condemns_needles_when_set() -> None:
    log_scales = torch.log(torch.tensor([[0.05, 0.0001], [0.01, 0.01]]))
    assert _strategy(20.0)._axis_ratio_prune_mask(log_scales).tolist() == [True, False]


def test_trainer_passes_the_axis_ratio_from_config() -> None:
    refine = {"enabled": True, "start": 0, "stop": 100, "every": 10}
    assert _trainer(refine=refine).refine_strategy.config.prune_axis_ratio is None
    assert _trainer(
        refine={**refine, "prune_axis_ratio": 20}
    ).refine_strategy.config.prune_axis_ratio == 20.0


@pytest.mark.parametrize("value", [True, 1.0, 0.5, math.inf, "x"])
def test_invalid_axis_ratio_is_refused(value) -> None:
    with pytest.raises(ValueError):
        _optional_axis_ratio(value)


def test_recipe_record_names_the_objective_and_data_settings() -> None:
    record = _trainer(exclude_spectrum_labels=["S0001-00001-0002"], adam_eps=1e-15).recipe_record()
    assert record["spectral_bin_weight"] == 5.0
    assert record["objective_population"] == {"image": 4, "spectrum": 3}
    assert record["image_variance"] == {"mode": "data", "outlier_pull": None}
    assert record["exclude_spectrum_labels"] == ["S0001-00001-0002"]
    assert record["adam_eps"] == 1e-15
    assert record["prepared_store"] is None
    assert record["execution_flags"]["resolved"]["SPLAT_EIGEN_STRICT_VJP"] is True


def _resume(trainer, saved, *, model_only=False):
    import train

    train._validate_resume_recipe(
        trainer, {"recipe": saved} if saved is not None else {},
        checkpoint_path=Path("ckpt.pt"), model_only=model_only,
    )


def test_full_resume_under_the_same_recipe_passes() -> None:
    trainer = _trainer()
    _resume(trainer, trainer.recipe_record())


def test_full_resume_under_another_recipe_is_refused() -> None:
    saved = _trainer(spectral_bin_weight=5.6211).recipe_record()
    with pytest.raises(RuntimeError, match="spectral_bin_weight"):
        _resume(_trainer(), saved)


def test_model_only_resume_may_change_the_recipe() -> None:
    _resume(_trainer(), _trainer(adam_eps=1e-15).recipe_record(), model_only=True)


def test_resume_of_a_checkpoint_without_a_recipe_warns() -> None:
    with pytest.warns(RuntimeWarning, match="predates recipe provenance"):
        _resume(_trainer(), None)


@pytest.mark.parametrize("name, value", [
    ("SPLAT_EIGEN_FULL_FMA", "0"), ("SPLAT_BAND_CONTRACTION", "sparse"), ("SPLAT_EIGEN_EXP_POLY", "1"),
    ("SPLAT_PAIR_CHUNK", "8000000"), ("SPLAT_AMP_CHUNK", "65536"),
])
def test_full_resume_under_another_execution_path_is_refused(name, value, monkeypatch) -> None:
    trainer = _trainer()
    saved = trainer.recipe_record()
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=f"execution_flags.{name}"):
        _resume(trainer, saved)


@pytest.mark.parametrize("overrides, match", [
    ({"splat_flux_l1": 1e-4}, "splat_flux_l1"),
    ({"objective_denominator": 2000}, "objective_denominator"),
    ({"grad_clip": 0.5}, "trainer_config.grad_clip"),
    ({"calibration": {"spectral_nuisance": {"enabled": False, "prior_sigma_log": 0.3}}}, "trainer_config.calibration"),
])
def test_full_resume_under_another_objective_is_refused(overrides, match) -> None:
    saved = _trainer().recipe_record()
    with pytest.raises(RuntimeError, match=match):
        _resume(_trainer(**overrides), saved)


def test_full_resume_may_change_budget_and_checkpoint_cadence() -> None:
    saved = _trainer().recipe_record()
    _resume(_trainer(max_steps=50, checkpoint_every_n_steps=7, fixed_holdout_eval_every_n_steps=3), saved)


def test_training_step_weights_each_spectral_bin(monkeypatch) -> None:
    """The trainer's own backward, with known per-datum terms, against the objective written out."""

    def gradient(weight):
        trainer = _trainer(spectral_bin_weight=weight)
        terms = torch.tensor([1.0, 2.0, 4.0, 8.0], requires_grad=True)
        monkeypatch.setattr(trainer, "_compute_step_gaussian_terms", lambda images, spectra: terms)
        trainer._train_step(torch.tensor([0, 1]), torch.tensor([4, 5]), compute_band_losses=False)
        return terms.grad

    image_population, spectrum_population, n_img, n_spec = 4, 6, 2, 2
    population = image_population + spectrum_population
    image_coefficient = image_population / population / n_img
    torch.testing.assert_close(
        gradient(3.0),
        torch.tensor([image_coefficient] * 2 + [3.0 * spectrum_population / population / n_spec] * 2),
    )
    torch.testing.assert_close(
        gradient(1.0),
        torch.tensor([image_coefficient] * 2 + [spectrum_population / population / n_spec] * 2),
    )


def test_sparse_optimizers_receive_adam_eps() -> None:
    from test_splat_spectrum import _analytic_spectrum_trainer

    trainer, _ = _analytic_spectrum_trainer(sparse_splat_gradients=True, adam_eps=1e-15)
    assert trainer.splat_optimizer is not None and trainer.splat_optimizer.param_groups
    assert all(g["eps"] == 1e-15 for g in trainer.splat_optimizer.param_groups)
    assert all(g["eps"] == 1e-15 for g in trainer.optimizer.param_groups)


def test_loading_optimizer_state_rebinds_the_position_rate_group() -> None:
    trainer = _trainer()
    state = trainer.optimizer_state_dict()
    old_group = trainer._position_group
    trainer.load_optimizer_state_dict(state)
    assert trainer._position_group is not old_group
    assert any(trainer._position_group is group for group in trainer.optimizer.param_groups)
    assert any(p is trainer._base_model().mu for p in trainer._position_group["params"])


def test_allow_operator_change_releases_only_the_withheld_sets() -> None:
    import train

    saved = _trainer().recipe_record()
    changed = _trainer(exclude_spectrum_labels=["S0001-00001-0002"])
    with pytest.raises(RuntimeError, match="exclude_spectrum_labels"):
        _resume(changed, saved)
    train._validate_resume_recipe(changed, {"recipe": saved}, checkpoint_path=Path("c.pt"), model_only=False,
                                  allow_operator_change=True)
    with pytest.raises(RuntimeError, match="spectral_bin_weight"):
        train._validate_resume_recipe(_trainer(spectral_bin_weight=2.0), {"recipe": saved}, checkpoint_path=Path("c.pt"),
                                      model_only=False, allow_operator_change=True)
