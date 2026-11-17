from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_analytic_trainer_dispatch import _bandpasses, _realistic_scene
from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.contracts import DatumKind, PSFMixtureComponents
from survey_fusion.pgm.splat_raster import (
    PointPSFMixture,
    tiled_mixture_point_prediction,
)
from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes
from survey_fusion.training.checkpoints import (
    operator_config_for_trainer,
    validate_checkpoint_operator_config,
)
from survey_fusion.training.datum_trainer import (
    DatumTrainer,
    transform_psf_mixture_to_normalized,
)
from survey_fusion.utils.speed import SpeedLogger


MIXTURE = PSFMixtureComponents.from_rows([
    [0.7, 1.0, 0.0, 1.0],
    [0.2, 4.0, 0.2, 5.0],
    [0.1, 12.0, -0.4, 15.0],
])


def _trainer(
    psf_model: str | None,
    *,
    apply_spatial_psf: bool = True,
    quadrature_order: int = 1,
    psf_jacobian_rel_tol: float = 1e-2,
    psf_mixture_digest_version: int = 2,
) -> DatumTrainer:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _realistic_scene(), _bandpasses(), psf_fwhm_arcsec=1.0, device=device,
        spectral_response_mode="legacy_imaging_median",
    )
    store.psf_mixture_digest = "d" * 64
    store.psf_mixture_digest_version = psf_mixture_digest_version
    for exp_id in store.image_band_by_exposure:
        exposure = store.exposures[exp_id]
        exposure.psf_mixture_dgpsf = MIXTURE
        exposure.psf_mixture_kl = MIXTURE
        exposure.psf_effective_fwhm_arcsec = 2.0
        exposure.label = f"image-{exp_id}"
    torch.manual_seed(31)
    model = SplatField(
        n_splats=5, n_templates=3, loglam_grid=torch.linspace(-1, 1, 16),
        init_scale=0.04, init_extent=0.6,
    )
    config = {
        "analytic_images": True, "analytic_spectra": True,
        "kernel_samples_k": 4, "replica_samples_k": 1,
        "epochs": 0, "learning_rate": 1e-3,
        "pixel_batch_size": 8, "spec_batch_size": 2,
        "apply_spatial_psf": apply_spatial_psf,
        "analytic_image_quadrature_order": quadrature_order,
        "psf_jacobian_rel_tol": psf_jacobian_rel_tol,
        "refine": {"enabled": True, "pair_n_sigma": 8.0},
    }
    if psf_model is not None:
        config["psf_model"] = psf_model
    return DatumTrainer(
        model=model, grid=build_spatial_grid(2, 3, device), data=store,
        training_bands=["g", "r", "i"], held_out_band="i", lambda_step=20.0,
        trainer_cfg=config, device=device,
    )


def test_non_gaussian_trainer_rejects_legacy_digest_version() -> None:
    with pytest.raises(ValueError, match="digest version 2"):
        _trainer("mog", psf_mixture_digest_version=1)


class _DetectorPlaneWCS:
    def __init__(self, width: int, height: int) -> None:
        self.cx = 0.5 * float(width - 1)
        self.cy = 0.5 * float(height - 1)

    def pixel_to_omega_xyz(self, x, y):
        x_arr, y_arr = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        omega = np.ones(x_arr.shape + (3,), dtype=np.float64)
        omega[..., 0] = x_arr - self.cx
        omega[..., 1] = y_arr - self.cy
        return omega


class _PolynomialModelWCS:
    def __init__(self, quadratic: float) -> None:
        self.quadratic = float(quadratic)

    def omega_xyz_to_norm_torch(self, omega_xyz: torch.Tensor):
        detector_x = omega_xyz[..., 0]
        detector_y = omega_xyz[..., 1]
        x_norm = 0.02 * detector_x + self.quadratic * detector_x.square()
        y_norm = 0.02 * detector_y
        return x_norm, y_norm


class _CancellationModelWCS:
    def omega_xyz_to_norm_torch(self, omega_xyz: torch.Tensor):
        detector_x = omega_xyz[..., 0]
        detector_y = omega_xyz[..., 1]
        return (
            100_000.0 + 0.02 * detector_x,
            -100_000.0 + 0.03 * detector_y,
        )


def _set_polynomial_wcs(trainer: DatumTrainer, exposure_id: int, quadratic: float) -> None:
    exposure = trainer.exposures[exposure_id]
    exposure.obs_wcs = _DetectorPlaneWCS(
        int(exposure.image_width), int(exposure.image_height)
    )
    exposure.wcs = _PolynomialModelWCS(quadratic)
    trainer._analytic_image_cache.clear()


def _jacobian_at(
    trainer: DatumTrainer, exposure_id: int, pixel_x: float, pixel_y: float
) -> torch.Tensor:
    exposure = trainer.exposures[exposure_id]
    probe = np.asarray(
        exposure.obs_wcs.pixel_to_omega_xyz(
            np.array([[pixel_x], [pixel_x + 1.0], [pixel_x]]),
            np.array([[pixel_y], [pixel_y], [pixel_y + 1.0]]),
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    x_norm, y_norm = exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(probe, dtype=torch.float64)
    )
    return torch.stack(
        (x_norm[1:] - x_norm[0], y_norm[1:] - y_norm[0]), dim=0
    ).to(torch.float32)


def _mixture_render_reference(
    trainer: DatumTrainer, exposure_id: int, *, per_point_jacobian: bool
) -> torch.Tensor:
    exposure = trainer.exposures[exposure_id]
    cache = trainer._analytic_cache_for_exposure(exposure_id)
    height = int(exposure.image_height)
    width = int(exposure.image_width)
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    omega = np.asarray(
        exposure.obs_wcs.pixel_to_omega_xyz(cols, rows), dtype=np.float64
    ).reshape(-1, 3)
    x_norm, y_norm = exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(omega, dtype=torch.float32)
    )
    components = exposure.psf_mixture_kl
    if per_point_jacobian:
        mixtures = [
            transform_psf_mixture_to_normalized(
                components,
                _jacobian_at(trainer, exposure_id, float(x), float(y)),
                pixel_variance=1.0 / 12.0,
            )
            for y, x in zip(rows.reshape(-1), cols.reshape(-1))
        ]
        mixture = PointPSFMixture(
            torch.cat([item.weights for item in mixtures], dim=0),
            torch.cat([item.covariances for item in mixtures], dim=0),
        )
    else:
        center = _jacobian_at(
            trainer,
            exposure_id,
            0.5 * float(width - 1),
            0.5 * float(height - 1),
        )
        center_mixture = transform_psf_mixture_to_normalized(
            components, center, pixel_variance=1.0 / 12.0
        )
        mixture = PointPSFMixture(
            center_mixture.weights.expand(x_norm.numel(), -1),
            center_mixture.covariances.expand(
                (x_norm.numel(),) + center_mixture.covariances.shape[1:]
            ),
        )
    band = trainer.image_band_by_exposure[exposure_id]
    amplitudes = splat_band_amplitudes(
        trainer._base_model(),
        {band: (cache["band_lam"], cache["band_throughput"])},
        loglam_min=trainer.loglam_min,
        loglam_max=trainer.loglam_max,
    )[band]
    return tiled_mixture_point_prediction(
        trainer._base_model(), x_norm, y_norm, mixture, amplitudes,
        n_sigma=trainer.pair_n_sigma,
    ).reshape(height, width)


def _image_source_and_grads(trainer: DatumTrainer):
    indices = trainer.image_train_indices[:12]
    exposure_ids = trainer._gather_named_rows(
        "exposure_id", indices, device=torch.device("cpu"), dtype=torch.long,
    )
    local = trainer._gather_named_rows(
        "datum_index", indices, device=torch.device("cpu"), dtype=torch.long,
    )
    values = trainer._compute_analytic_image_source_mu(indices, exposure_ids, local)
    values.sum().backward()
    field = trainer._base_model()
    return values.detach(), tuple(
        parameter.grad.detach().clone()
        for parameter in (field.mu, field.log_scales, field.theta, field.raw_flux)
    )


def test_explicit_and_default_gaussian_are_bit_identical_forward_and_grads() -> None:
    default = _trainer(None)
    explicit = _trainer("gaussian")
    default_values, default_grads = _image_source_and_grads(default)
    explicit_values, explicit_grads = _image_source_and_grads(explicit)
    assert torch.equal(default_values, explicit_values)
    for left, right in zip(default_grads, explicit_grads):
        assert torch.equal(left, right)


def test_mixture_image_forward_is_finite_and_backpropagates() -> None:
    trainer = _trainer("mog")
    values, grads = _image_source_and_grads(trainer)
    assert torch.isfinite(values).all()
    assert all(torch.isfinite(gradient).all() for gradient in grads)


def test_mog_batch_promotes_isotropic_fallback_exposure_to_full_covariance() -> None:
    trainer = _trainer("mog")
    fallback = PSFMixtureComponents.from_rows([
        [0.7, 1.0, 0.0, 1.0], [0.2, 4.0, 0.0, 4.0], [0.1, 9.0, 0.0, 9.0],
    ])
    fallback_id = trainer.image_exposure_ids[0]
    trainer.exposures[fallback_id].psf_mixture_kl = fallback
    indices = torch.cat([
        trainer._global_indices_for(exposure_id, DatumKind.IMAGE_PIXEL)[:1]
        for exposure_id in trainer.image_exposure_ids
    ])
    exposure_ids = trainer._gather_named_rows(
        "exposure_id", indices, device=torch.device("cpu"), dtype=torch.long,
    )
    local = trainer._gather_named_rows(
        "datum_index", indices, device=torch.device("cpu"), dtype=torch.long,
    )
    values = trainer._compute_analytic_image_source_mu(indices, exposure_ids, local)
    assert torch.isfinite(values).all()


def test_lifecycle_and_fiber_tables_remain_on_legacy_scalar() -> None:
    gaussian = _trainer("gaussian")
    mixture = _trainer("dgpsf")
    gaussian_tables = gaussian._prepare_band_tables()
    mixture_tables = mixture._prepare_band_tables()
    assert gaussian_tables["band_psf_variance"] == mixture_tables["band_psf_variance"]
    assert gaussian.refine_strategy.config.grow_scale == mixture.refine_strategy.config.grow_scale
    gaussian_spec = gaussian._spectrum_cache_for_exposure(gaussian.spec_exposure_ids[0])
    mixture_spec = mixture._spectrum_cache_for_exposure(mixture.spec_exposure_ids[0])
    assert torch.equal(gaussian_spec.psf_cov, mixture_spec.psf_cov)
    gaussian_fiber = gaussian._compute_analytic_spectral_source_mu(gaussian.spec_train_indices)
    mixture_fiber = mixture._compute_analytic_spectral_source_mu(mixture.spec_train_indices)
    assert torch.equal(gaussian_fiber, mixture_fiber)
    gaussian_significance = gaussian._build_band_snr_fn(gaussian._base_model())(
        gaussian._base_model()
    )
    mixture_significance = mixture._build_band_snr_fn(mixture._base_model())(
        mixture._base_model()
    )
    assert torch.equal(gaussian_significance, mixture_significance)
    gaussian_birth_cell = gaussian.birth_cell_scale * torch.tensor(
        list(gaussian_tables["band_psf_variance"].values())
    ).sqrt().median()
    mixture_birth_cell = mixture.birth_cell_scale * torch.tensor(
        list(mixture_tables["band_psf_variance"].values())
    ).sqrt().median()
    assert torch.equal(gaussian_birth_cell, mixture_birth_cell)


def test_mixture_mc_image_builder_fails_loudly() -> None:
    trainer = _trainer("dgpsf")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="MC image kernel builders"):
        trainer._build_mc_kernel_samples(trainer.image_train_indices[:1], k=2)


def test_selected_exposure_missing_mixture_names_backfill_script() -> None:
    trainer = _trainer("mog")
    exposure_id = trainer.image_exposure_ids[0]
    trainer.exposures[exposure_id].psf_mixture_kl = None
    with pytest.raises(ValueError, match=r"image-0.*scripts/backfill_psf_mixture.py"):
        trainer._analytic_cache_for_exposure(exposure_id)


def test_mixture_checkpoint_records_model_and_manifest_digest() -> None:
    import train

    trainer = _trainer("mog")
    checkpoint = train._build_training_checkpoint(
        trainer, SimpleNamespace(epochs_ran=1, steps_ran=2),
    )
    assert checkpoint["operator_config"]["psf_model"] == "mog"
    assert checkpoint["operator_config"]["psf_mixture_digest"] == "d" * 64
    assert {
        "operator_schema_version",
        "psf_mixture_digest_version",
        "psf_mixture_provenance_convention",
        "apply_spatial_psf",
        "pair_n_sigma",
        "exact_pair_prune",
        "pixel_response_convention",
        "analytic_image_quadrature_order",
        "analytic_images_enabled",
        "psf_jacobian_rel_tol",
        "psf_jacobian_algorithm",
        "psf_jacobian_max_tiles_per_axis",
        "psf_jacobian_cap_policy",
    } <= set(checkpoint["operator_config"])


def test_mixture_training_step_writes_unambiguous_support_telemetry(
    tmp_path,
) -> None:
    trainer = _trainer("dgpsf")
    trainer.epochs = 1
    trainer.max_steps = 1
    field = trainer._base_model()
    with torch.no_grad():
        field.mu[0].zero_()
        field.mu[1:].fill_(10.0)
        field.log_scales.fill_(math.log(0.001))
    speed_logger = SpeedLogger(
        tmp_path, enabled=True, log_every_n_steps=1, flush_every_records=1
    )
    trainer.speed_logger = speed_logger
    trainer._speed_profile_enabled = True

    state = trainer.fit()
    speed_logger.close()

    assert state.steps_ran == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "speed.log").read_text(encoding="utf-8").splitlines()
    ]
    step = next(record["steps"][0] for record in records if record["type"] == "step_window")
    for key in (
        "model_splats", "active_splats", "candidate_pairs", "retained_pairs",
    ):
        assert key in step
    assert step["model_splats"] != step["active_splats"]


def test_linear_wcs_keeps_byte_identical_single_jacobian_path(capsys) -> None:
    trainer = _trainer("mog")
    exposure_id = trainer.image_exposure_ids[0]
    _set_polynomial_wcs(trainer, exposure_id, quadratic=0.0)

    cache = trainer._analytic_cache_for_exposure(exposure_id)
    exposure = trainer.exposures[exposure_id]
    center_jacobian = _jacobian_at(
        trainer,
        exposure_id,
        0.5 * float(exposure.image_width - 1),
        0.5 * float(exposure.image_height - 1),
    )
    legacy = transform_psf_mixture_to_normalized(
        exposure.psf_mixture_kl, center_jacobian, pixel_variance=1.0 / 12.0
    )

    assert cache["psf_jacobian_path"] == "single"
    assert cache["psf_jacobian_max_rel_variation"] == pytest.approx(0.0, abs=2e-6)
    assert torch.equal(cache["pixel_to_norm_jacobian"], center_jacobian)
    assert torch.equal(cache["psf_mixture"].weights, legacy.weights)
    assert torch.equal(cache["psf_mixture"].covariances, legacy.covariances)
    assert "PSF Jacobian" in capsys.readouterr().out
    actual = torch.as_tensor(
        trainer.render_datum_image_window_deterministic(
            exposure_id, y0=0, y1=32, x0=0, x1=32,
            include_additive=False,
        )
    )
    expected = _mixture_render_reference(
        trainer, exposure_id, per_point_jacobian=False
    )
    assert torch.equal(actual, expected)


def test_nonlinear_wcs_uses_tiles_and_matches_dense_point_jacobians() -> None:
    trainer = _trainer("mog", psf_jacobian_rel_tol=0.08)
    exposure_id = trainer.image_exposure_ids[0]
    _set_polynomial_wcs(trainer, exposure_id, quadratic=4e-4)

    cache = trainer._analytic_cache_for_exposure(exposure_id)
    assert cache["psf_jacobian_path"] == "tiled"
    assert cache["psf_jacobian_max_rel_variation"] > trainer.psf_jacobian_rel_tol
    assert cache["psf_jacobian_tile_grid"] != (1, 1)
    assert (
        cache["psf_jacobian_tile_max_rel_variation"]
        <= trainer.psf_jacobian_rel_tol
    )
    actual = torch.as_tensor(
        trainer.render_datum_image_window_deterministic(
            exposure_id, y0=0, y1=32, x0=0, x1=32,
            include_additive=False,
        )
    )
    dense = _mixture_render_reference(trainer, exposure_id, per_point_jacobian=True)
    single = _mixture_render_reference(trainer, exposure_id, per_point_jacobian=False)
    scale = dense.abs().max().clamp_min(1e-12)
    tiled_error = (actual - dense).abs().max() / scale
    single_error = (single - dense).abs().max() / scale
    assert tiled_error < 0.02
    assert single_error > 0.02


def test_loose_jacobian_tolerance_keeps_nonlinear_wcs_on_single_path() -> None:
    trainer = _trainer("mog", psf_jacobian_rel_tol=10.0)
    exposure_id = trainer.image_exposure_ids[0]
    _set_polynomial_wcs(trainer, exposure_id, quadratic=4e-4)

    cache = trainer._analytic_cache_for_exposure(exposure_id)
    assert 0.0 < cache["psf_jacobian_max_rel_variation"] < 10.0
    assert cache["psf_jacobian_path"] == "single"
    actual = torch.as_tensor(
        trainer.render_datum_image_window_deterministic(
            exposure_id, y0=0, y1=32, x0=0, x1=32,
            include_additive=False,
        )
    )
    single = _mixture_render_reference(trainer, exposure_id, per_point_jacobian=False)
    assert torch.equal(actual, single)


def test_center_jacobian_uses_same_float64_one_pixel_algorithm_as_tiles() -> None:
    trainer = _trainer("mog", psf_jacobian_rel_tol=10.0)
    exposure_id = trainer.image_exposure_ids[0]
    exposure = trainer.exposures[exposure_id]
    exposure.obs_wcs = _DetectorPlaneWCS(
        int(exposure.image_width), int(exposure.image_height)
    )
    exposure.wcs = _CancellationModelWCS()
    trainer._analytic_image_cache.clear()

    cx = 0.5 * float(exposure.image_width - 1)
    cy = 0.5 * float(exposure.image_height - 1)
    expected = trainer._pixel_to_norm_jacobians(
        exposure, np.asarray([cx]), np.asarray([cy])
    )[0]
    cache = trainer._analytic_cache_for_exposure(exposure_id)

    assert torch.equal(cache["pixel_to_norm_jacobian"], expected)


def test_jacobian_tile_cap_fails_closed_when_tolerance_is_unmet() -> None:
    trainer = _trainer("mog", psf_jacobian_rel_tol=1e-8)
    exposure_id = trainer.image_exposure_ids[0]
    _set_polynomial_wcs(trainer, exposure_id, quadratic=4e-4)

    with pytest.raises(RuntimeError, match=r"16x16.*tolerance"):
        trainer._analytic_cache_for_exposure(exposure_id)


@pytest.mark.parametrize("psf_model", ["gaussian", "dgpsf", "mog"])
def test_analytic_no_psf_matches_nonanalytic_no_psf_convention(
    psf_model: str,
) -> None:
    trainer = _trainer(
        psf_model, apply_spatial_psf=False, quadrature_order=4
    )
    exposure_id = trainer.image_exposure_ids[0]
    field = trainer._base_model()
    with torch.no_grad():
        field.log_scales.fill_(math.log(0.12))
    analytic = trainer.render_datum_image_window_deterministic(
        exposure_id, y0=8, y1=24, x0=8, x1=24,
        include_additive=False,
    )
    trainer.analytic_images_enabled = False
    nonanalytic = trainer.render_datum_image_window_deterministic(
        exposure_id, y0=8, y1=24, x0=8, x1=24,
        supersample=16, include_additive=False,
    )
    # Gauss-Legendre order 4 and a uniform 16x supersample are independent
    # numerical integrations of the same unblurred square detector pixel.
    scale = max(float(np.max(np.abs(nonanalytic))), 1e-12)
    assert float(np.max(np.abs(analytic - nonanalytic))) / scale < 0.015


def test_no_psf_analytic_render_is_model_independent_and_resume_identity_is_live() -> None:
    disabled = {
        model: _trainer(model, apply_spatial_psf=False)
        for model in ("gaussian", "dgpsf", "mog")
    }
    renders = {
        model: _image_source_and_grads(trainer)[0]
        for model, trainer in disabled.items()
    }
    assert torch.equal(renders["gaussian"], renders["dgpsf"])
    assert torch.equal(renders["gaussian"], renders["mog"])

    enabled = _trainer("mog", apply_spatial_psf=True)
    assert not torch.equal(renders["mog"], _image_source_and_grads(enabled)[0])
    disabled_config = operator_config_for_trainer(disabled["mog"])
    enabled_config = operator_config_for_trainer(enabled)
    assert disabled_config["apply_spatial_psf"] is False
    assert enabled_config["apply_spatial_psf"] is True
    with pytest.raises(ValueError, match="apply_spatial_psf"):
        validate_checkpoint_operator_config(
            {"operator_config": disabled_config},
            current_operator_config=enabled_config,
            context="Resume",
            resume=True,
        )
