from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support.synthetic_trainer import SyntheticStore, bandpass

from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.contracts import DatumBatch, DatumKind, ExposureState, SkyDiskApertureSpec
from survey_fusion.pgm.kernel_builders import KernelFactoryFromBuilders
from survey_fusion.pgm.splat_raster import tiled_point_prediction
from survey_fusion.utils.wcs import radec_deg_to_omega_xyz
from survey_fusion.pgm.splat_spectrum import (
    _band_weight_tables,
    _hat_band_tables,
    _HatBandWeights,
    _largest_eigenvalue_symmetric_2x2,
    _piecewise_linear_band_weights,
    _TabulatedBandWeights,
    build_spectrum_batch_context,
    build_spectrum_cache,
    paired_positive_hat_bin_average,
    piecewise_linear_band_average,
    piecewise_linear_bin_average,
    positive_hat_bin_average,
    splat_band_amplitudes,
    splat_fiber_spectrum,
)
from survey_fusion.pgm import splat_spectrum as splat_spectrum_module
from survey_fusion.training import datum_trainer as datum_trainer_module
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.wcs import TangentPlaneWCS


def test_closed_form_largest_eigenvalue_matches_symmetric_solver() -> None:
    generator = torch.Generator().manual_seed(19)
    factors = torch.randn((128, 2, 2), generator=generator, dtype=torch.float64)
    covariance = factors @ factors.transpose(-1, -2)

    actual = _largest_eigenvalue_symmetric_2x2(covariance)
    expected = torch.linalg.eigvalsh(covariance)[..., -1]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_closed_form_largest_eigenvalue_has_finite_gradients() -> None:
    matrix = torch.tensor(
        [[[2.0, 0.3], [0.3, 1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    _largest_eigenvalue_symmetric_2x2(matrix).sum().backward()

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def _quadrature_reference(
    knots: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    uniform_x, uniform_w = np.polynomial.legendre.leggauss(160)
    normal_x, normal_w = np.polynomial.hermite.hermgauss(100)
    result = np.empty((lower.size, knots.shape[0]), dtype=np.float64)
    for bin_idx, (lo, hi, sigma_value) in enumerate(zip(lower, upper, sigma)):
        uniform = lo + 0.5 * (uniform_x + 1.0) * (hi - lo)
        if sigma_value > 0.0:
            sample = uniform[:, None] + math.sqrt(2.0) * sigma_value * normal_x[None, :]
            weight = 0.5 * uniform_w[:, None] * normal_w[None, :] / math.sqrt(math.pi)
        else:
            sample = uniform[:, None]
            weight = 0.5 * uniform_w[:, None]
        for spectrum_idx in range(knots.shape[0]):
            evaluated = np.interp(
                sample,
                knots[spectrum_idx],
                values[spectrum_idx],
                left=values[spectrum_idx, 0],
                right=values[spectrum_idx, -1],
            )
            result[bin_idx, spectrum_idx] = np.sum(evaluated * weight)
    return result


def test_piecewise_linear_bin_average_matches_independent_quadrature() -> None:
    knots = np.array(
        [[4800.0, 4950.0, 5100.0, 5350.0], [4700.0, 5000.0, 5200.0, 5500.0]],
        dtype=np.float64,
    )
    values = np.array(
        [[0.8, 1.7, 0.5, 2.1], [1.3, 0.4, 1.9, 1.1]],
        dtype=np.float64,
    )
    lower = np.array([4850.0, 5050.0, 5250.0], dtype=np.float64)
    upper = np.array([4925.0, 5160.0, 5400.0], dtype=np.float64)
    sigma = np.array([0.0, 18.0, 47.0], dtype=np.float64)
    got = piecewise_linear_bin_average(
        torch.tensor(knots),
        torch.tensor(values),
        torch.tensor(lower),
        torch.tensor(upper),
        torch.tensor(sigma),
    ).numpy()
    expected = _quadrature_reference(knots, values, lower, upper, sigma)
    np.testing.assert_allclose(got, expected, atol=1.0e-5, rtol=1.0e-5)


def test_piecewise_linear_bin_average_has_finite_gradients() -> None:
    knots = torch.tensor(
        [[4800.0, 5000.0, 5300.0]], dtype=torch.float64, requires_grad=True
    )
    values = torch.tensor(
        [[0.5, 1.5, 0.8]], dtype=torch.float64, requires_grad=True
    )
    result = piecewise_linear_bin_average(
        knots,
        values,
        torch.tensor([4900.0, 5100.0], dtype=torch.float64),
        torch.tensor([5050.0, 5250.0], dtype=torch.float64),
        torch.tensor([0.0, 25.0], dtype=torch.float64),
    )
    result.sum().backward()
    assert knots.grad is not None and torch.isfinite(knots.grad).all()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_piecewise_linear_bin_average_is_stable_in_float32() -> None:
    torch.manual_seed(7)
    grid = torch.linspace(-1.0, 1.0, 96, dtype=torch.float64)
    log_min = math.log(3000.0)
    log_max = math.log(11000.0)
    shifts = torch.tensor([[-0.05], [0.0], [0.08]], dtype=torch.float64)
    knots = torch.exp(
        0.5 * (log_min + log_max)
        + 0.5 * (log_max - log_min) * (grid[None, :] + shifts)
    )
    values = 2.0 * torch.rand(3, 96, dtype=torch.float64)
    lower = torch.linspace(3600.0, 10350.0, 200, dtype=torch.float64)
    upper = lower + torch.linspace(1.0, 4.0, 200, dtype=torch.float64)
    sigma = torch.linspace(0.0, 4.0, 200, dtype=torch.float64)

    expected = piecewise_linear_bin_average(knots, values, lower, upper, sigma)
    actual = piecewise_linear_bin_average(
        knots.float(),
        values.float(),
        lower.float(),
        upper.float(),
        sigma.float(),
    ).double()
    torch.testing.assert_close(actual, expected, rtol=3.0e-3, atol=5.0e-4)


def test_piecewise_linear_band_average_matches_dense_quadrature() -> None:
    knot_lam = torch.tensor(
        [[3600.0, 4300.0, 5100.0, 6700.0], [3900.0, 4700.0, 5900.0, 7200.0]],
        dtype=torch.float64,
    )
    knot_values = torch.tensor(
        [[0.7, 1.6, 0.4, 2.0], [1.4, 0.5, 1.8, 0.9]],
        dtype=torch.float64,
    )
    response_lam = torch.tensor(
        [3800.0, 4200.0, 5000.0, 6100.0, 7000.0],
        dtype=torch.float64,
    )
    throughput = torch.tensor(
        [0.0, 0.8, 1.0, 0.35, 0.0],
        dtype=torch.float64,
    )
    actual = piecewise_linear_band_average(
        knot_lam, knot_values, response_lam, throughput
    ).numpy()

    sample_lam = np.linspace(3800.0, 7000.0, 400_001)
    response = np.interp(sample_lam, response_lam.numpy(), throughput.numpy())
    denominator = np.trapezoid(response * sample_lam, sample_lam)
    expected = np.array(
        [
            np.trapezoid(
                np.interp(
                    sample_lam,
                    knots,
                    values,
                    left=values[0],
                    right=values[-1],
                )
                * response
                * sample_lam,
                sample_lam,
            )
            / denominator
            for knots, values in zip(knot_lam.numpy(), knot_values.numpy())
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-8, atol=2.0e-8)


def test_piecewise_linear_band_average_has_finite_gradients() -> None:
    knot_lam = torch.tensor(
        [[3900.0, 4700.0, 5800.0, 6900.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    knot_values = torch.tensor(
        [[0.8, 1.4, 0.6, 1.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = piecewise_linear_band_average(
        knot_lam,
        knot_values,
        torch.tensor([3800.0, 4400.0, 5600.0, 7100.0], dtype=torch.float64),
        torch.tensor([0.0, 0.9, 0.5, 0.0], dtype=torch.float64),
    )
    result.sum().backward()
    assert knot_lam.grad is not None and torch.isfinite(knot_lam.grad).all()
    assert knot_values.grad is not None and torch.isfinite(knot_values.grad).all()


def test_splat_band_amplitudes_match_dense_wavelength_sampling() -> None:
    torch.manual_seed(11)
    field = SplatField(
        n_splats=9,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.04,
    ).double()
    loglam_min = math.log(3400.0)
    loglam_max = math.log(9200.0)
    field.sed.set_loglam_scale(0.5 * (loglam_max - loglam_min))
    response_lam = torch.tensor(
        [3800.0, 4200.0, 5100.0, 6200.0, 7100.0],
        dtype=torch.float64,
    )
    throughput = torch.tensor(
        [0.0, 0.75, 1.0, 0.4, 0.0],
        dtype=torch.float64,
    )
    exact = splat_band_amplitudes(
        field,
        {"x": (response_lam, throughput)},
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        splat_chunk=4,
    )["x"]

    lam = torch.linspace(response_lam[0], response_lam[-1], 40_001)
    response = torch.from_numpy(
        np.interp(lam.numpy(), response_lam.numpy(), throughput.numpy())
    )
    trap = torch.ones_like(lam)
    trap[0] = trap[-1] = 0.5
    loglam = 2.0 * (torch.log(lam) - loglam_min) / (
        loglam_max - loglam_min
    ) - 1.0
    sampled = field.fluxes() * (
        field.sed.eval(field.sed_coeffs, loglam, field.sed_shift)
        * (response * lam * trap)[None, :]
    ).sum(dim=1) / (response * lam * trap).sum()

    torch.testing.assert_close(exact, sampled, rtol=2.0e-5, atol=2.0e-6)
    exact.sum().backward()
    assert field.sed.raw_templates.grad is not None
    assert torch.isfinite(field.sed.raw_templates.grad).all()


class _FailingKernelFactory:
    def __init__(self) -> None:
        self.calls = 0

    def build_kernel_sample_plan_for_indices(self, global_indices):
        self.calls += 1
        raise AssertionError("deterministic spectra must bypass kernel sampling")


def _analytic_spectrum_trainer(
    *, failing_factory: bool = True, sparse_splat_gradients: bool = False,
    sed_off_grid: str = "clamp", sed_pin_gauge_knots: bool = False, **trainer_overrides,
) -> tuple[DatumTrainer, object]:
    device = torch.device("cpu")
    width = height = 512
    wcs = TangentPlaneWCS(
        ra0_rad=math.pi,
        dec0_rad=0.0,
        pixel_scale_arcsec=0.4,
        width=width,
        height=height,
    )
    image_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=wcs,
        psf_fwhm_arcsec=1.1,
        lsf_fwhm_angstrom=None,
        bandpasses={"r": bandpass("r")},
        lam_spec=np.array([5000.0], dtype=np.float64),
        fiber_mask=np.ones((1, 1), dtype=np.float32),
        pixel_scale_arcsec=0.4,
        image_height=height,
        image_width=width,
    )
    spec_lam = np.array([4950.0, 5100.0, 5300.0], dtype=np.float64)
    spec_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=None,
        psf_fwhm_arcsec=1.1,
        lsf_fwhm_angstrom=np.array([20.0, 35.0, 50.0], dtype=np.float64),
        bandpasses={},
        lam_spec=spec_lam,
        fiber_mask=np.ones((1, 1), dtype=np.float32),
        pixel_scale_arcsec=0.4,
        lam_edges=np.array(
            [[4875.0, 5025.0], [5025.0, 5200.0], [5200.0, 5400.0]], dtype=np.float64
        ),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.0),
    )
    batch = DatumBatch(
        y=torch.zeros(4),
        sigma=torch.ones(4),
        exposure_id=torch.tensor([0, 1, 1, 1]),
        datum_kind=torch.tensor(
            [int(DatumKind.IMAGE_PIXEL)] + [int(DatumKind.SPECTRAL_BIN)] * 3
        ),
        datum_index=torch.tensor([256 * width + 256, 0, 1, 2]),
        kernel_samples=None,
    )
    exposures = {0: image_exposure, 1: spec_exposure}
    factory = (
        _FailingKernelFactory()
        if failing_factory
        else KernelFactoryFromBuilders(
            batch=batch,
            exposures=exposures,
            image_band_by_exposure={0: "r"},
            rng_seed=19,
        )
    )
    store = SyntheticStore(
        exposures=exposures,
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=factory,
        n_image=1,
        n_spec=3,
    )
    field = SplatField(
        n_splats=12,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 24),
        init_scale=0.01,
        sed_off_grid=sed_off_grid,
        sed_pin_gauge_knots=sed_pin_gauge_knots,
    )
    with torch.no_grad():
        field.mu.normal_(mean=0.0, std=0.01)
    trainer = DatumTrainer(
        model=field,
        grid=build_spatial_grid(height, width, device),
        data=store,
        training_bands=["r"],
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 4,
            "analytic_images": sparse_splat_gradients,
            "analytic_spectra": True,
            "sparse_splat_gradients": sparse_splat_gradients,
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "apply_spatial_psf": True,
            **trainer_overrides,
        },
        device=device,
    )
    return trainer, factory


def _multifiber_analytic_spectrum_trainer(
    *, sed_off_grid: str = "clamp",
) -> DatumTrainer:
    base_trainer, _ = _analytic_spectrum_trainer(
        sparse_splat_gradients=True, sed_off_grid=sed_off_grid,
    )
    image_exposure = base_trainer.runtime_by_exposure[0].exposure
    base_spectrum = base_trainer.runtime_by_exposure[1].exposure
    spectral_exposures = {
        exp_id: replace(
            base_spectrum,
            psf_fwhm_arcsec=0.75 + 0.2 * exp_id,
            aperture=SkyDiskApertureSpec(
                center_radec_deg=(180.0 + 0.0002 * exp_id, 0.0),
                radius_arcsec=0.7 + 0.15 * exp_id,
            ),
        )
        for exp_id in (1, 2, 3)
    }
    exposures = {0: image_exposure, **spectral_exposures}
    batch = DatumBatch(
        y=torch.zeros(10),
        sigma=torch.ones(10),
        exposure_id=torch.tensor([0, 1, 2, 1, 3, 2, 1, 3, 2, 3]),
        datum_kind=torch.tensor(
            [int(DatumKind.IMAGE_PIXEL)] + [int(DatumKind.SPECTRAL_BIN)] * 9
        ),
        datum_index=torch.tensor(
            [256 * 512 + 256, 0, 0, 1, 0, 1, 2, 1, 2, 2]
        ),
        kernel_samples=None,
    )
    factory = _FailingKernelFactory()
    store = SyntheticStore(
        exposures=exposures,
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=factory,
        n_image=1,
        n_spec=9,
    )
    return DatumTrainer(
        model=base_trainer.model,
        grid=base_trainer.grid,
        data=store,
        training_bands=["r"],
        held_out_band=None,
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 4,
            "analytic_images": True,
            "analytic_spectra": True,
            "sparse_splat_gradients": True,
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "apply_spatial_psf": True,
        },
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("bad_bin", [-1, 10_000])
@pytest.mark.parametrize("position", [0, 4, 8])
def test_out_of_range_bin_in_any_exposure_is_rejected(
    bad_bin: int, position: int
) -> None:
    """One batch-wide comparison replaced a check per exposure; it has to
    catch a bad index whichever exposure's block holds it."""
    trainer = _multifiber_analytic_spectrum_trainer()
    indices = torch.arange(1, 10, dtype=torch.long)
    trainer.data.batch.datum_index[int(indices[position])] = bad_bin
    with pytest.raises(ValueError, match="Spectral bin index is out of bounds"):
        trainer._compute_analytic_spectral_source_mu(indices)


def _legacy_analytic_spectral_source_mu(
    trainer: DatumTrainer, global_indices: torch.Tensor
) -> torch.Tensor:
    model = trainer._base_model()
    batch_context = build_spectrum_batch_context(model)
    exp_ids = trainer._gather_named_rows(
        "exposure_id", global_indices, device=torch.device("cpu"), dtype=torch.long
    )
    local = trainer._gather_named_rows(
        "datum_index", global_indices, device=torch.device("cpu"), dtype=torch.long
    )
    source_mu = torch.empty(
        int(global_indices.numel()), device=trainer.device, dtype=torch.float32
    )
    for exp_id in exp_ids.unique().tolist():
        selected = (exp_ids == exp_id).nonzero(as_tuple=True)[0]
        local_cpu = local[selected]
        exposure = trainer.runtime_by_exposure[int(exp_id)].exposure
        cache = trainer._spectrum_cache_for_exposure(int(exp_id))
        n_bins = int(cache.lam_edges.shape[0])
        if bool(torch.any(local_cpu < 0)) or bool(torch.any(local_cpu >= n_bins)):
            raise ValueError("Spectral bin index is out of bounds.")
        local_selected = local_cpu.to(device=trainer.device)
        prediction = splat_fiber_spectrum(
            model,
            exposure,
            local_selected,
            cache=cache,
            batch_context=batch_context,
            loglam_min=trainer.loglam_min,
            loglam_max=trainer.loglam_max,
            apply_spatial_psf=trainer.apply_spatial_psf,
            validate_bins=False,
        )
        prediction = trainer._apply_spectral_nuisance(
            int(exp_id), local_selected, prediction
        )
        source_mu[selected.to(device=trainer.device)] = prediction.to(torch.float32)
    return source_mu


def _batched_candidate_fixture() -> tuple[
    SplatField, list[ExposureState], list[splat_spectrum_module.SplatSpectrumCache]
]:
    torch.manual_seed(43)
    field = SplatField(
        n_splats=317,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 24),
        init_scale=0.01,
    )
    with torch.no_grad():
        field.mu.uniform_(-0.35, 0.35)
        field.log_scales.uniform_(math.log(0.004), math.log(0.025))

    trainer, _ = _analytic_spectrum_trainer()
    base_exposure = trainer.runtime_by_exposure[1].exposure
    exposures = []
    caches = []
    for fiber_index in range(7):
        aperture = SkyDiskApertureSpec(
            center_radec_deg=(180.0 + 0.0001 * fiber_index, 0.0),
            radius_arcsec=0.6 + 0.2 * fiber_index,
        )
        exposure = replace(
            base_exposure,
            psf_fwhm_arcsec=0.7 + 0.15 * fiber_index,
            aperture=aperture,
        )
        exposures.append(exposure)
        caches.append(
            build_spectrum_cache(
                exposure,
                aperture,
                apply_spatial_psf=True,
                radial_nodes=2,
                angular_nodes=4,
                device=field.mu.device,
                dtype=field.mu.dtype,
            )
        )

    caches[0] = replace(caches[0], point_bounds=(-2.0, 2.0, -2.0, 2.0))
    caches[1] = replace(caches[1], point_bounds=(2.0, 2.1, 2.0, 2.1))
    caches[4] = replace(caches[4], point_bounds=None)
    return field, exposures, caches


def _legacy_candidates(
    field: SplatField,
    exposures: list[ExposureState],
    caches: list[splat_spectrum_module.SplatSpectrumCache],
    splat_eigen_max: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        splat_spectrum_module._candidate_spatial_weights(
            field,
            exposure,
            exposure.aperture,
            apply_spatial_psf=True,
            radial_nodes=2,
            angular_nodes=4,
            n_sigma=6.0,
            cache=cache,
            splat_eigen_max=splat_eigen_max,
        )[0]
        for exposure, cache in zip(exposures, caches)
    ]


def test_batched_fiber_candidates_are_bitwise_identical_to_legacy_selection() -> None:
    field, exposures, caches = _batched_candidate_fixture()
    context = build_spectrum_batch_context(field)

    expected = _legacy_candidates(
        field, exposures, caches, context.splat_eigen_max
    )
    actual = splat_spectrum_module.batched_fiber_candidates(
        field,
        caches,
        n_sigma=6.0,
        splat_eigen_max=context.splat_eigen_max,
        fiber_chunk=2,
    )

    assert expected[0].numel() == field.mu.shape[0]
    assert expected[1].numel() == 0
    assert len(actual) == len(expected)
    for actual_fiber, expected_fiber, exposure, cache in zip(
        actual, expected, exposures, caches
    ):
        assert torch.equal(actual_fiber, expected_fiber)
        legacy_candidate, legacy_spatial = (
            splat_spectrum_module._candidate_spatial_weights(
                field,
                exposure,
                exposure.aperture,
                apply_spatial_psf=True,
                radial_nodes=2,
                angular_nodes=4,
                n_sigma=6.0,
                cache=cache,
                splat_eigen_max=context.splat_eigen_max,
            )
        )
        batched_candidate, batched_spatial = (
            splat_spectrum_module._candidate_spatial_weights(
                field,
                exposure,
                exposure.aperture,
                apply_spatial_psf=True,
                radial_nodes=2,
                angular_nodes=4,
                n_sigma=6.0,
                cache=cache,
                splat_eigen_max=context.splat_eigen_max,
                candidate=actual_fiber,
            )
        )
        assert torch.equal(batched_candidate, legacy_candidate)
        assert torch.equal(batched_spatial, legacy_spatial)


def test_batched_fiber_candidate_host_syncs_do_not_grow_with_exposures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field, exposures, caches = _batched_candidate_fixture()
    context = build_spectrum_batch_context(field)
    original_nonzero = torch.nonzero
    calls = 0

    def counted_nonzero(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", counted_nonzero)
    splat_spectrum_module.batched_fiber_candidates(
        field,
        caches,
        n_sigma=6.0,
        splat_eigen_max=context.splat_eigen_max,
        fiber_chunk=len(caches),
    )
    # A fixed number of compaction points per fiber chunk -- the tile index's
    # own prefilter and the exact pair test -- not one per exposure.
    assert calls <= 3

    calls = 0
    _legacy_candidates(field, exposures, caches, context.splat_eigen_max)
    assert calls == len(caches)


def _dense_fiber_candidates(
    mu: torch.Tensor,
    splat_eigen_max: torch.Tensor,
    psf_eigen_max: torch.Tensor,
    bounds: torch.Tensor,
    n_sigma: float,
) -> list[torch.Tensor]:
    """The [n_fibers, n_splats] selection the tile-indexed search replaces."""
    x_min, x_max, y_min, y_max = bounds.unbind(dim=1)
    reach = float(n_sigma) * torch.sqrt(
        splat_eigen_max[None, :] + psf_eigen_max[:, None]
    )
    keep = (
        (mu[None, :, 0] + reach >= x_min[:, None])
        & (mu[None, :, 0] - reach <= x_max[:, None])
        & (mu[None, :, 1] + reach >= y_min[:, None])
        & (mu[None, :, 1] - reach <= y_max[:, None])
    )
    counts = keep.sum(dim=1).tolist()
    splat_indices = torch.nonzero(keep, as_tuple=True)[1]
    return list(torch.split(splat_indices, counts))


def _candidate_probe_cache(
    bounds: tuple[float, float, float, float],
    psf_eigen_max: float,
    *,
    hoist_bounds: bool,
) -> splat_spectrum_module.SplatSpectrumCache:
    x_min, x_max, y_min, y_max = bounds
    points = torch.tensor(
        [[x_min, y_min], [x_max, y_max], [x_min, y_max], [x_max, y_min]],
        dtype=torch.float32,
    )
    return splat_spectrum_module.SplatSpectrumCache(
        exposure_identity=0,
        apply_spatial_psf=True,
        radial_nodes=1,
        angular_nodes=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        lam_edges=torch.tensor([[4000.0, 4001.0]]),
        points=points,
        weights=torch.full((4,), 0.25),
        psf_cov=torch.eye(2) * psf_eigen_max,
        psf_eigen_max=torch.tensor(psf_eigen_max, dtype=torch.float32),
        aperture_area_arcsec2=1.0,
        scalar_lsf_sigma=0.0,
        lsf_sigma_by_bin=None,
        point_bounds=bounds if hoist_bounds else None,
        lsf_mode="zero",
    )


def test_fiber_candidates_drop_only_the_non_finite_splat() -> None:
    """A NaN position must cost its own splat, not every other one.

    The dense test drops a NaN row because every comparison against it is
    false. The tile search sizes its proposal radius from the positions, so a
    reduction over them would turn one NaN into an empty candidate set for the
    whole batch -- which is exactly what a training step with a diverged splat
    would hand it.
    """
    generator = torch.Generator().manual_seed(11)
    mu = (torch.rand(400, 2, generator=generator) * 0.6 - 0.3).float()
    mu[17] = float("nan")
    mu[23, 0] = float("inf")
    splat_eigen_max = torch.full((400,), 4.0e-4)
    caches = [
        _candidate_probe_cache((-0.1, -0.09, 0.02, 0.03), 1.0e-5, hoist_bounds=True),
        _candidate_probe_cache((0.15, 0.16, -0.2, -0.19), 1.0e-5, hoist_bounds=True),
    ]
    bounds = splat_spectrum_module.stacked_fiber_bounds(
        caches, dtype=mu.dtype, device=mu.device
    )
    expected = _dense_fiber_candidates(
        mu,
        splat_eigen_max,
        torch.stack([cache.psf_eigen_max for cache in caches]),
        bounds,
        6.0,
    )
    actual = splat_spectrum_module.batched_fiber_candidates(
        SimpleNamespace(mu=mu),
        caches,
        n_sigma=6.0,
        splat_eigen_max=splat_eigen_max,
    )

    assert sum(int(fiber.numel()) for fiber in expected) > 0
    for actual_fiber, expected_fiber in zip(actual, expected):
        assert torch.equal(actual_fiber, expected_fiber)
        assert 17 not in actual_fiber.tolist()


@pytest.mark.parametrize("case", range(24))
def test_fiber_candidates_match_the_dense_selection_on_random_fields(
    case: int,
) -> None:
    generator = torch.Generator().manual_seed(2026 + case)
    n_splats = int(torch.randint(1, 900, (1,), generator=generator))
    mu = (torch.rand(n_splats, 2, generator=generator) * 2.6 - 1.3).float()
    if case % 6 == 0:
        mu = mu[:1].expand(n_splats, 2).contiguous()
    # Scales from far below a fiber aperture to splats that reach the whole
    # field, so the proposal radius spans several tile widths either way.
    log_scale = (
        math.log(1.0e-4)
        + (math.log(2.0) - math.log(1.0e-4))
        * torch.rand(n_splats, 2, generator=generator)
    )
    splat_eigen_max = torch.exp(2.0 * log_scale.amax(dim=1)).float()

    n_fibers = int(torch.randint(1, 10, (1,), generator=generator))
    caches = []
    for fiber in range(n_fibers):
        centre = (torch.rand(2, generator=generator) * 3.0 - 1.5).tolist()
        half = float(torch.rand((), generator=generator)) * 0.05
        if case % 5 == 0:
            half = 0.0
        caches.append(
            _candidate_probe_cache(
                (
                    centre[0] - half,
                    centre[0] + half,
                    centre[1] - half * 0.5,
                    centre[1] + half * 0.5,
                ),
                float(torch.rand((), generator=generator)) ** 4,
                hoist_bounds=(fiber + case) % 3 != 0,
            )
        )

    field = SimpleNamespace(mu=mu)
    bounds = splat_spectrum_module.stacked_fiber_bounds(
        caches, dtype=mu.dtype, device=mu.device
    )
    expected = _dense_fiber_candidates(
        mu,
        splat_eigen_max,
        torch.stack([cache.psf_eigen_max for cache in caches]),
        bounds,
        6.0,
    )
    actual = splat_spectrum_module.batched_fiber_candidates(
        field,
        caches,
        n_sigma=6.0,
        splat_eigen_max=splat_eigen_max,
        fiber_chunk=1 + case % 4,
    )
    assert len(actual) == len(expected)
    for actual_fiber, expected_fiber in zip(actual, expected):
        assert torch.equal(actual_fiber, expected_fiber)


def test_precomputed_candidate_fiber_spectrum_is_bitwise_identical() -> None:
    field, exposures, caches = _batched_candidate_fixture()
    context = build_spectrum_batch_context(field)
    exposure = exposures[2]
    cache = caches[2]
    local = torch.tensor([0, 1, 2], dtype=torch.long)

    expected = splat_fiber_spectrum(
        field,
        exposure,
        local,
        cache=cache,
        batch_context=context,
        loglam_min=math.log(3400.0),
        loglam_max=math.log(9200.0),
        apply_spatial_psf=True,
        radial_nodes=2,
        angular_nodes=4,
    )
    candidate = splat_spectrum_module.batched_fiber_candidates(
        field,
        [cache],
        n_sigma=6.0,
        splat_eigen_max=context.splat_eigen_max,
    )[0]
    actual = splat_fiber_spectrum(
        field,
        exposure,
        local,
        cache=cache,
        batch_context=context,
        candidate=candidate,
        loglam_min=math.log(3400.0),
        loglam_max=math.log(9200.0),
        apply_spatial_psf=True,
        radial_nodes=2,
        angular_nodes=4,
    )

    assert torch.equal(actual, expected)


def _fiber_spectrum_batch(
    *,
    n_exposures: int,
    bins_per_exposure: tuple[int, ...],
    off_grid: str = "clamp",
    lsf_fwhm: float | np.ndarray | None = 2.5,
    n_splats: int = 317,
    dtype: torch.dtype = torch.float32,
):
    """A field factory plus a fiber batch both spectral paths can be run on."""

    def make_field() -> SplatField:
        torch.manual_seed(43)
        field = SplatField(
            n_splats=n_splats,
            n_templates=4,
            loglam_grid=torch.linspace(-1.0, 1.0, 24),
            init_scale=0.01,
            sed_off_grid=off_grid,
        )
        with torch.no_grad():
            field.mu.uniform_(-0.35, 0.35)
            field.log_scales.uniform_(math.log(0.004), math.log(0.03))
            field.sed_shift.uniform_(-0.25, 0.25)
        return field.to(dtype)

    trainer, _ = _analytic_spectrum_trainer()
    base_exposure = trainer.runtime_by_exposure[1].exposure
    caches = []
    for index in range(n_exposures):
        aperture = SkyDiskApertureSpec(
            center_radec_deg=(180.0 + 0.0003 * index, 0.0002 * (index % 7)),
            radius_arcsec=0.6 + 0.05 * (index % 5),
        )
        exposure = replace(
            base_exposure,
            psf_fwhm_arcsec=0.7 + 0.05 * (index % 4),
            aperture=aperture,
            lsf_fwhm_angstrom=lsf_fwhm,
        )
        caches.append(
            build_spectrum_cache(
                exposure,
                aperture,
                apply_spatial_psf=True,
                radial_nodes=2,
                angular_nodes=4,
                device=torch.device("cpu"),
                dtype=dtype,
            )
        )
    bin_indices = [
        torch.arange(bins_per_exposure[index % len(bins_per_exposure)])
        for index in range(n_exposures)
    ]
    context = build_spectrum_batch_context(make_field())
    candidates = splat_spectrum_module.batched_fiber_candidates(
        make_field(),
        caches,
        n_sigma=6.0,
        splat_eigen_max=context.splat_eigen_max,
    )
    return make_field, caches, candidates, bin_indices


def _fiber_path_prediction(
    field: SplatField, caches, candidates, bin_indices, *, flat: bool
) -> torch.Tensor:
    arguments = dict(
        loglam_min=math.log(3400.0),
        loglam_max=math.log(9200.0),
        batch_context=build_spectrum_batch_context(field),
    )
    if flat:
        return splat_spectrum_module.flat_fiber_spectra(
            field, caches, candidates, bin_indices, **arguments
        )
    return torch.cat(
        splat_spectrum_module.batched_fiber_spectra(
            field, caches, candidates, bin_indices, **arguments
        )
    )


def _max_relative_difference(
    actual: torch.Tensor, expected: torch.Tensor
) -> float:
    deviation = float((actual - expected).detach().abs().max())
    scale = float(expected.detach().abs().max())
    return deviation if scale == 0.0 else deviation / scale


def _dense_gradient(parameter: torch.Tensor) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    if parameter.grad.layout == torch.sparse_coo:
        return parameter.grad.coalesce().to_dense()
    return parameter.grad


@pytest.mark.parametrize(
    "n_exposures, bins_per_exposure",
    [(7, (1,)), (7, (1, 1, 3)), (5, (3,)), (200, (1,))],
)
@pytest.mark.parametrize("off_grid", ["clamp", "zero"])
def test_flat_fiber_spectra_match_the_blocked_path(
    n_exposures: int, bins_per_exposure: tuple[int, ...], off_grid: str
) -> None:
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=n_exposures,
        bins_per_exposure=bins_per_exposure,
        off_grid=off_grid,
    )
    expected = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=False
    )
    actual = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=True
    )

    assert actual.shape == expected.shape
    assert float(expected.detach().abs().max()) > 0.0
    # Same terms, regrouped reductions: fp32 reassociation, not 1e-3 drift.
    assert _max_relative_difference(actual, expected) < 1.0e-5
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=0.0)


@pytest.mark.parametrize("lsf_mode", ["zero", "positive", "mixed"])
def test_flat_fiber_spectra_match_the_blocked_path_for_every_lsf_mode(
    lsf_mode: str,
) -> None:
    if lsf_mode == "mixed":
        lsf_fwhm = np.array([2.5, 0.0, 1.5], dtype=np.float64)
    else:
        lsf_fwhm = {"zero": 0.0, "positive": 2.5}[lsf_mode]
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=6, bins_per_exposure=(1, 2), lsf_fwhm=lsf_fwhm
    )
    assert caches[0].lsf_mode == (None if lsf_mode == "mixed" else lsf_mode)
    expected = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=False
    )
    actual = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=True
    )

    assert _max_relative_difference(actual, expected) < 1.0e-5
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=0.0)


def test_flat_and_blocked_fiber_spectra_are_the_same_terms_in_float64() -> None:
    """The two paths differ only in how their reductions are grouped.

    In float32 the agreement floor is set by the conditioning of the batch, not
    by the regrouping: in the off_grid="zero" corner both paths sit ~1e-3 from a
    float64 evaluation of the same expression. Run the identity where rounding
    cannot hide it.
    """
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=9,
        bins_per_exposure=(1, 1, 3),
        off_grid="zero",
        dtype=torch.float64,
    )
    n_bins = sum(int(index.numel()) for index in bin_indices)
    upstream = torch.linspace(0.5, 1.5, n_bins, dtype=torch.float64)

    results = []
    for flat in (False, True):
        field = make_field()
        with torch.no_grad():
            field.sed_shift.fill_(0.8)
        prediction = _fiber_path_prediction(
            field, caches, candidates, bin_indices, flat=flat
        )
        (prediction * upstream).sum().backward()
        results.append(
            (
                prediction.detach(),
                {name: _dense_gradient(p) for name, p in field.named_parameters()},
            )
        )

    (expected, expected_gradients), (actual, actual_gradients) = results
    assert _max_relative_difference(actual, expected) < 1.0e-12
    for name, expected_gradient in expected_gradients.items():
        if expected_gradient is None:
            continue
        assert _max_relative_difference(
            actual_gradients[name], expected_gradient
        ) < 1.0e-12, name


@pytest.mark.parametrize("sparse_gradients", [False, True])
def test_flat_fiber_spectra_gradients_match_the_blocked_path(
    sparse_gradients: bool,
) -> None:
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=9, bins_per_exposure=(1, 1, 3)
    )
    n_bins = sum(int(index.numel()) for index in bin_indices)
    upstream = torch.linspace(0.5, 1.5, n_bins)

    gradients = []
    for flat in (False, True):
        field = make_field()
        field.set_sparse_gradients(sparse_gradients)
        prediction = _fiber_path_prediction(
            field, caches, candidates, bin_indices, flat=flat
        )
        (prediction * upstream).sum().backward()
        gradients.append(
            {name: _dense_gradient(p) for name, p in field.named_parameters()}
        )

    expected, actual = gradients
    assert set(actual) == set(expected)
    assert any(name.endswith("templates") for name in expected)
    for name, expected_gradient in expected.items():
        if expected_gradient is None:
            assert actual[name] is None
            continue
        scale = float(expected_gradient.abs().max())
        assert scale > 0.0, name
        assert _max_relative_difference(actual[name], expected_gradient) < 1.0e-5, name
        # Elementwise, against a tolerance floor set by the tensor's own scale:
        # entries thousands of times below the largest one are sums that
        # cancel, so their own relative error says nothing about drift.
        assert torch.allclose(
            actual[name], expected_gradient, rtol=1.0e-5, atol=1.0e-6 * scale
        ), name


@pytest.mark.parametrize("budget", ["_FIBER_PAIR_ELEMENTS", "_FIBER_ROW_ELEMENTS"])
def test_flat_fiber_spectra_survive_their_transient_budgets(
    budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunking the pair and row stages must not move a value.

    Both budgets are far above any test shape, so nothing else here ever
    crosses one. Drive them down instead: the pair stage takes its matrix
    inverse over the whole batch and then only reduces over the trailing
    quadrature axis, so its result cannot depend on the split at all.
    """
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=7, bins_per_exposure=(1, 1, 3)
    )
    expected = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=True
    )
    monkeypatch.setattr(splat_spectrum_module, budget, 64)
    actual = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=True
    )

    if budget == "_FIBER_PAIR_ELEMENTS":
        assert torch.equal(actual, expected)
    else:
        assert _max_relative_difference(actual, expected) < 1.0e-5


def test_flat_fiber_spectra_return_zeros_for_exposures_without_candidates() -> None:
    make_field, caches, candidates, bin_indices = _fiber_spectrum_batch(
        n_exposures=4, bins_per_exposure=(2,)
    )
    empty = torch.empty(0, dtype=torch.long)
    candidates = [empty, candidates[1], empty, candidates[3]]
    expected = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=False
    )
    actual = _fiber_path_prediction(
        make_field(), caches, candidates, bin_indices, flat=True
    )

    assert torch.equal(actual[:2], torch.zeros(2))
    assert torch.equal(actual[4:6], torch.zeros(2))
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=0.0)


@pytest.mark.parametrize("sed_off_grid", ["clamp", "zero"])
def test_batched_analytic_spectral_source_and_gradients_are_bitwise_identical(
    sed_off_grid: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPLAT_FIBER_PATH", "blocked")
    legacy_trainer = _multifiber_analytic_spectrum_trainer(
        sed_off_grid=sed_off_grid,
    )
    batched_trainer = _multifiber_analytic_spectrum_trainer(
        sed_off_grid=sed_off_grid,
    )
    batched_trainer.model.load_state_dict(legacy_trainer.model.state_dict())
    if sed_off_grid == "zero":
        with torch.no_grad():
            legacy_trainer.model.sed_shift.fill_(0.8)
            batched_trainer.model.sed_shift.fill_(0.8)
    indices = torch.tensor([9, 1, 5, 3, 8, 2, 7, 4, 6], dtype=torch.long)
    original_batched_candidates = splat_spectrum_module.batched_fiber_candidates
    candidate_calls = 0

    def counted_batched_candidates(*args, **kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        return original_batched_candidates(*args, **kwargs)

    monkeypatch.setattr(
        datum_trainer_module,
        "batched_fiber_candidates",
        counted_batched_candidates,
        raising=False,
    )
    expected = _legacy_analytic_spectral_source_mu(legacy_trainer, indices)
    actual = batched_trainer._compute_analytic_spectral_source_mu(indices)
    assert candidate_calls == 1
    assert torch.equal(actual, expected)

    expected.sum().backward()
    actual.sum().backward()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        legacy_trainer.model.named_parameters(),
        batched_trainer.model.named_parameters(),
    ):
        assert actual_name == expected_name
        if expected_parameter.grad is None or actual_parameter.grad is None:
            assert expected_parameter.grad is None
            assert actual_parameter.grad is None
            continue
        if expected_parameter.grad.layout == torch.sparse_coo:
            expected_gradient = expected_parameter.grad.coalesce()
            actual_gradient = actual_parameter.grad.coalesce()
            assert torch.equal(actual_gradient.indices(), expected_gradient.indices())
            assert torch.equal(actual_gradient.values(), expected_gradient.values())
        else:
            assert torch.equal(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("sed_off_grid", ["clamp", "zero"])
def test_flat_fiber_path_trainer_predictions_match_the_blocked_path(
    sed_off_grid: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_trainer = _multifiber_analytic_spectrum_trainer(
        sed_off_grid=sed_off_grid,
    )
    flat_trainer = _multifiber_analytic_spectrum_trainer(
        sed_off_grid=sed_off_grid,
    )
    flat_trainer.model.load_state_dict(blocked_trainer.model.state_dict())
    if sed_off_grid == "zero":
        with torch.no_grad():
            blocked_trainer.model.sed_shift.fill_(0.8)
            flat_trainer.model.sed_shift.fill_(0.8)
    indices = torch.tensor([9, 1, 5, 3, 8, 2, 7, 4, 6], dtype=torch.long)

    monkeypatch.setenv("SPLAT_FIBER_PATH", "blocked")
    expected = blocked_trainer._compute_analytic_spectral_source_mu(indices)
    monkeypatch.setenv("SPLAT_FIBER_PATH", "flat")
    actual = flat_trainer._compute_analytic_spectral_source_mu(indices)

    assert _max_relative_difference(actual, expected) < 1.0e-5
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=0.0)

    expected.sum().backward()
    actual.sum().backward()
    for (name, expected_parameter), (_, actual_parameter) in zip(
        blocked_trainer.model.named_parameters(),
        flat_trainer.model.named_parameters(),
    ):
        expected_gradient = _dense_gradient(expected_parameter)
        if expected_gradient is None:
            assert _dense_gradient(actual_parameter) is None
            continue
        # off_grid="zero" at a large shift truncates most of the knot
        # support, so the fp32 gradient is a difference of near-equal terms:
        # both paths sit ~1e-3 from a float64 evaluation there, which is what
        # sets this floor. The float64 test above is the identity statement.
        assert _max_relative_difference(
            _dense_gradient(actual_parameter), expected_gradient
        ) < 1.0e-4, name


def test_cached_spectrum_matches_uncached_predictions_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The subject here is the spectrum cache, not the fiber evaluation path, so
    # hold the path that reproduces splat_fiber_spectrum's own grouping.
    monkeypatch.setenv("SPLAT_FIBER_PATH", "blocked")
    cached_trainer, _ = _analytic_spectrum_trainer()
    uncached_trainer, _ = _analytic_spectrum_trainer()
    uncached_trainer.model.load_state_dict(cached_trainer.model.state_dict())
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    local = torch.tensor([0, 1, 2], dtype=torch.long)
    exposure = cached_trainer.runtime_by_exposure[1].exposure

    cached = cached_trainer._compute_analytic_spectral_source_mu(indices)
    uncached = splat_fiber_spectrum(
        uncached_trainer.model,
        exposure,
        local,
        loglam_min=uncached_trainer.loglam_min,
        loglam_max=uncached_trainer.loglam_max,
        apply_spatial_psf=True,
    )
    torch.testing.assert_close(cached, uncached, rtol=1.0e-7, atol=1.0e-8)

    cached.sum().backward()
    uncached.sum().backward()
    for (cached_name, cached_parameter), (uncached_name, uncached_parameter) in zip(
        cached_trainer.model.named_parameters(),
        uncached_trainer.model.named_parameters(),
    ):
        assert cached_name == uncached_name
        if cached_parameter.grad is None or uncached_parameter.grad is None:
            assert cached_parameter.grad is None
            assert uncached_parameter.grad is None
            continue
        torch.testing.assert_close(
            cached_parameter.grad,
            uncached_parameter.grad,
            rtol=1.0e-6,
            atol=2.5e-6,
        )


def test_spectrum_cache_point_bounds_match_separate_reductions() -> None:
    """One host transfer for the four bounds, same numbers as four reads."""
    trainer, _ = _analytic_spectrum_trainer()
    cache = trainer._spectrum_cache_for_exposure(1)
    points = cache.points
    assert cache.point_bounds == (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def test_spectrum_static_cache_reused_but_batch_context_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _ = _analytic_spectrum_trainer()
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    # Every fiber's cache is built before training starts, so no step pays for
    # one at all, and none is rebuilt afterwards.
    assert set(trainer._spectrum_cache_by_exposure) == set(
        trainer.spec_exposure_ids
    )
    calls = {"quadrature": 0, "templates": 0}
    original_quadrature = splat_spectrum_module._disk_quadrature
    original_templates = type(trainer.model.sed).templates

    def counted_quadrature(*args, **kwargs):
        calls["quadrature"] += 1
        return original_quadrature(*args, **kwargs)

    def counted_templates(sed):
        calls["templates"] += 1
        return original_templates(sed)

    monkeypatch.setattr(splat_spectrum_module, "_disk_quadrature", counted_quadrature)
    monkeypatch.setattr(type(trainer.model.sed), "templates", counted_templates)

    first = trainer._compute_analytic_spectral_source_mu(indices)
    second = trainer._compute_analytic_spectral_source_mu(indices)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert calls == {"quadrature": 0, "templates": 2}


def test_spectrum_cache_survives_two_backward_steps_and_splat_count_change() -> None:
    trainer, _ = _analytic_spectrum_trainer()
    indices = torch.tensor([1, 2, 3], dtype=torch.long)

    first = trainer._compute_analytic_spectral_source_mu(indices)
    first.sum().backward()
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            parameter.grad.zero_()
    with torch.no_grad():
        trainer.model.sed_coeffs[0, 0].add_(0.05)
        trainer.model.log_scales[0].add_(0.07)

    second = trainer._compute_analytic_spectral_source_mu(indices)
    assert not torch.allclose(first, second)
    second.sum().backward()
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()

    replacement = SplatField(
        n_splats=13,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 24),
        init_scale=0.01,
    )
    trainer.model = replacement
    refined = trainer._compute_analytic_spectral_source_mu(indices)
    assert refined.shape == indices.shape
    assert torch.isfinite(refined).all()


def test_spectrum_cache_handles_scalar_and_vector_lsf_and_rejects_other_exposure() -> None:
    trainer, _ = _analytic_spectrum_trainer()
    model = trainer.model
    vector_exposure = trainer.runtime_by_exposure[1].exposure
    vector_cache = build_spectrum_cache(
        vector_exposure,
        vector_exposure.aperture,
        apply_spatial_psf=True,
        radial_nodes=8,
        angular_nodes=32,
        device=model.mu.device,
        dtype=model.mu.dtype,
    )
    expected = torch.tensor([20.0, 35.0, 50.0]) / 2.355
    assert vector_cache.lsf_sigma_by_bin is not None
    torch.testing.assert_close(vector_cache.lsf_sigma_by_bin, expected)

    scalar_exposure = replace(vector_exposure, lsf_fwhm_angstrom=33.0)
    scalar_cache = build_spectrum_cache(
        scalar_exposure,
        scalar_exposure.aperture,
        apply_spatial_psf=True,
        radial_nodes=8,
        angular_nodes=32,
        device=model.mu.device,
        dtype=model.mu.dtype,
    )
    assert scalar_cache.scalar_lsf_sigma == 33.0 / 2.355
    assert isinstance(scalar_cache.scalar_lsf_sigma, float)
    assert scalar_cache.lsf_sigma_by_bin is None

    with pytest.raises(ValueError, match="exposure mismatch"):
        splat_fiber_spectrum(
            model,
            scalar_exposure,
            torch.tensor([0]),
            cache=vector_cache,
            batch_context=build_spectrum_batch_context(model),
            loglam_min=trainer.loglam_min,
            loglam_max=trainer.loglam_max,
            apply_spatial_psf=True,
        )


def test_sparse_splat_trainer_runs_joint_image_spectrum_step() -> None:
    trainer, _ = _analytic_spectrum_trainer(sparse_splat_gradients=True)

    stats = trainer._train_step(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1, 2, 3], dtype=torch.long),
    )

    assert math.isfinite(stats["local_total_nll"])
    assert stats["active_splats"] > 0
    assert stats["candidate_pairs"] > 0
    assert stats["retained_pairs"] > 0
    for parameter in trainer.model.per_splat_parameters():
        assert parameter.grad is not None
        assert parameter.grad.layout == torch.sparse_coo
    assert trainer.model.sed.raw_templates.grad is not None
    assert trainer.model.sed.raw_templates.grad.layout == torch.strided


def test_sparse_splat_trainer_step_runs_with_the_sed_basis_fixes_on() -> None:
    """A joint image plus spectrum step must survive both basis flags: the
    band weights no longer sum to one and two knot columns take no gradient."""
    trainer, _ = _analytic_spectrum_trainer(
        sparse_splat_gradients=True, sed_off_grid="zero", sed_pin_gauge_knots=True,
    )

    stats = trainer._train_step(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1, 2, 3], dtype=torch.long),
    )

    assert math.isfinite(stats["local_total_nll"])
    assert stats["active_splats"] > 0
    template_grad = trainer.model.sed.raw_templates.grad
    assert template_grad is not None
    assert torch.isfinite(template_grad).all()
    assert (template_grad[:, 0] == 0.0).all()
    assert (template_grad[:, -1] == 0.0).all()
    assert template_grad[:, 1:-1].abs().sum().item() > 0.0


def test_analytic_spectral_terms_are_repeatable_and_bypass_sampling() -> None:
    trainer, factory = _analytic_spectrum_trainer()
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    first = trainer._compute_analytic_spectral_gaussian_terms(indices)
    second = trainer._compute_analytic_spectral_gaussian_terms(indices)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    first.sum().backward()
    assert trainer.model.sed.raw_templates.grad is not None
    diagnostic_first = trainer.predict_fiber_spectra(include_additive=False)
    diagnostic_second = trainer.predict_fiber_spectra(include_additive=False)
    np.testing.assert_array_equal(diagnostic_first[1][1], diagnostic_second[1][1])
    assert factory.calls == 0


def test_analytic_spectrum_agrees_with_high_sample_mc() -> None:
    trainer, _ = _analytic_spectrum_trainer(failing_factory=False)
    indices = torch.tensor([1, 2, 3], dtype=torch.long)
    local = torch.tensor([0, 1, 2], dtype=torch.long)
    exposure = trainer.runtime_by_exposure[1].exposure
    with torch.no_grad():
        deterministic = splat_fiber_spectrum(
            trainer.model,
            exposure,
            local,
            loglam_min=trainer.loglam_min,
            loglam_max=trainer.loglam_max,
            apply_spatial_psf=True,
        )
        samples = trainer._build_mc_kernel_samples(indices, k=65_536)
        mc = trainer._compute_source_mu_mc_from_samples(
            indices, samples, apply_spectral_nuisance=False
        )
    torch.testing.assert_close(deterministic, mc, rtol=2.0e-2, atol=2.0e-3)


def test_spectral_kernel_normalization_is_aperture_area_in_arcsec2() -> None:
    trainer, _ = _analytic_spectrum_trainer(failing_factory=False)
    indices = torch.tensor([1, 2, 3], dtype=torch.long)

    samples = trainer._build_mc_kernel_samples(indices, k=16)

    expected_area_arcsec2 = math.pi
    torch.testing.assert_close(
        samples.samples.Z,
        torch.full_like(samples.samples.Z, expected_area_arcsec2),
    )


def test_tabulated_band_weights_match_reference_path() -> None:
    torch.manual_seed(3)
    n_knots, n_response, n_splats = 96, 16, 4_000
    log_mid = 0.5 * (math.log(3000.0) + math.log(11_000.0))
    scale = 0.5 * (math.log(11_000.0) - math.log(3000.0))
    grid = torch.linspace(-1.0, 1.0, n_knots, dtype=torch.float64)
    C = torch.exp(log_mid + scale * grid)
    # include extreme shifts so both outermost table buckets are exercised
    shift = torch.cat([
        torch.empty(n_splats - 2, dtype=torch.float64).uniform_(-0.5, 0.5),
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    ])
    t = torch.exp(scale * shift)
    for lo, hi in ((3200.0, 4000.0), (5400.0, 7200.0), (8200.0, 10500.0)):
        lam = torch.linspace(lo, hi, n_response, dtype=torch.float64)
        thr = torch.sin(torch.linspace(0.0, math.pi, n_response, dtype=torch.float64))
        tables = _band_weight_tables(
            lam, thr, C, dtype=torch.float64, device=torch.device("cpu"),
        )
        w = _TabulatedBandWeights.apply(t, C, *tables)
        w_ref = _piecewise_linear_band_weights(C[None, :] * t[:, None], lam, thr)
        assert torch.allclose(w, w_ref, atol=1e-11, rtol=0.0)


def test_tabulated_band_weights_gradient_matches_autograd_reference() -> None:
    torch.manual_seed(4)
    n_knots, n_response, n_splats = 96, 16, 2_000
    log_mid = 0.5 * (math.log(3000.0) + math.log(11_000.0))
    scale = 0.5 * (math.log(11_000.0) - math.log(3000.0))
    grid = torch.linspace(-1.0, 1.0, n_knots, dtype=torch.float64)
    C = torch.exp(log_mid + scale * grid)
    shift = torch.empty(n_splats, dtype=torch.float64).uniform_(-0.3, 0.3)
    lam = torch.linspace(3900.0, 5600.0, n_response, dtype=torch.float64)
    thr = torch.sin(torch.linspace(0.0, math.pi, n_response, dtype=torch.float64))
    tables = _band_weight_tables(
        lam, thr, C, dtype=torch.float64, device=torch.device("cpu"),
    )
    values = torch.randn(n_splats, n_knots, dtype=torch.float64)

    t_new = torch.exp(scale * shift).requires_grad_(True)
    (values * _TabulatedBandWeights.apply(t_new, C, *tables)).sum().backward()
    t_ref = torch.exp(scale * shift).requires_grad_(True)
    (values * _piecewise_linear_band_weights(C[None, :] * t_ref[:, None], lam, thr)).sum().backward()
    assert torch.allclose(t_new.grad, t_ref.grad, atol=1e-9, rtol=0.0)


def test_tabulated_band_weights_are_stable_in_float32() -> None:
    torch.manual_seed(5)
    n_knots, n_response, n_splats = 96, 16, 4_000
    log_mid = 0.5 * (math.log(3000.0) + math.log(11_000.0))
    scale = 0.5 * (math.log(11_000.0) - math.log(3000.0))
    grid64 = torch.linspace(-1.0, 1.0, n_knots, dtype=torch.float64)
    C64 = torch.exp(log_mid + scale * grid64)
    shift = torch.empty(n_splats, dtype=torch.float64).uniform_(-0.15, 0.15)
    t64 = torch.exp(scale * shift)
    lam = torch.linspace(3900.0, 5600.0, n_response, dtype=torch.float64)
    thr = torch.sin(torch.linspace(0.0, math.pi, n_response, dtype=torch.float64))
    w_ref64 = _piecewise_linear_band_weights(C64[None, :] * t64[:, None], lam, thr)

    C32 = C64.to(torch.float32)
    tables = _band_weight_tables(
        lam, thr, C32, dtype=torch.float32, device=torch.device("cpu"),
    )
    w32 = _TabulatedBandWeights.apply(t64.to(torch.float32), C32, *tables)
    assert torch.isfinite(w32).all()
    assert (w32.to(torch.float64) - w_ref64).abs().max().item() < 5.0e-4


def _hat_test_grid(n_knots: int = 96) -> tuple[torch.Tensor, float, float]:
    log_mid = 0.5 * (math.log(3000.0) + math.log(11_000.0))
    scale = 0.5 * (math.log(11_000.0) - math.log(3000.0))
    grid = torch.linspace(-1.0, 1.0, n_knots, dtype=torch.float64)
    return torch.exp(log_mid + scale * grid), log_mid, scale


def _sdss_r_like_band() -> tuple[torch.Tensor, torch.Tensor]:
    """Steep-edged plateau curve shaped like the real SDSS r throughput."""
    lam = torch.arange(5380.0, 7231.0, 25.0, dtype=torch.float64)
    edges = torch.minimum((lam - 5400.0) / 150.0, (7200.0 - lam) / 150.0)
    base = 0.55 * edges.clamp(0.0, 1.0)
    ripple = 1.0 + 0.03 * torch.sin(lam / 40.0)
    return lam, (base * ripple).clamp_min(0.0)


def _hat_weights(
    t: torch.Tensor, C: torch.Tensor, lam: torch.Tensor, thr: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    tables = _hat_band_tables(lam, thr, dtype=dtype, device=torch.device("cpu"))
    return _HatBandWeights.apply(t.to(dtype), C.to(dtype), *tables)


def test_hat_band_weights_match_reference_path() -> None:
    torch.manual_seed(3)
    n_splats = 4_000
    C, _, scale = _hat_test_grid()
    # extreme shifts push the whole grid off band on both sides
    shift = torch.cat([
        torch.empty(n_splats - 2, dtype=torch.float64).uniform_(-0.5, 0.5),
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    ])
    t = torch.exp(scale * shift)
    for lo, hi in ((3200.0, 4000.0), (5400.0, 7200.0), (8200.0, 10500.0)):
        lam = torch.linspace(lo, hi, 16, dtype=torch.float64)
        thr = torch.sin(torch.linspace(0.0, math.pi, 16, dtype=torch.float64))
        w = _hat_weights(t, C, lam, thr, torch.float64)
        w_ref = _piecewise_linear_band_weights(C[None, :] * t[:, None], lam, thr)
        assert torch.allclose(w, w_ref, atol=1e-11, rtol=0.0)


def test_hat_band_weights_are_stable_in_float32_at_high_dynamic_range() -> None:
    """The cumulative-moment tables lose 1e-3..1e-1 of the band amplitude in
    float32 on plateau bandpasses with HDR SEDs; the local hat assembly must
    stay at float32 roundoff scale on the same construction."""
    torch.manual_seed(5)
    n_splats = 20_000
    C, _, scale = _hat_test_grid()
    lam, thr = _sdss_r_like_band()
    bulk = 0.02 * torch.randn(n_splats, dtype=torch.float64)
    tails = (torch.rand(n_splats, dtype=torch.float64) - 0.5) * 0.30
    mix = (torch.rand(n_splats) < 0.10).to(torch.float64)
    t = torch.exp(scale * (bulk * (1.0 - mix) + tails * mix))
    values = torch.exp(3.0 * torch.randn(n_splats, C.numel(), dtype=torch.float64))

    w_ref = _piecewise_linear_band_weights(C[None, :] * t[:, None], lam, thr)
    amp_ref = (values * w_ref).sum(dim=1)

    C32 = C.to(torch.float32)
    legacy_tables = _band_weight_tables(
        lam, thr, C32, dtype=torch.float32, device=torch.device("cpu"),
    )
    w_legacy32 = _TabulatedBandWeights.apply(
        t.to(torch.float32), C32, *legacy_tables
    ).to(torch.float64)
    amp_legacy32 = (values * w_legacy32).sum(dim=1)
    rel_legacy = ((amp_legacy32 - amp_ref).abs() / amp_ref.abs()).max().item()

    w_hat32 = _hat_weights(t, C, lam, thr, torch.float32).to(torch.float64)
    amp_hat32 = (values * w_hat32).sum(dim=1)
    rel_hat = (amp_hat32 - amp_ref).abs() / amp_ref.abs()

    assert rel_legacy > 1.0e-3
    assert rel_hat.max().item() < 1.0e-4
    assert rel_hat.median().item() < 1.0e-5
    assert (w_hat32 >= 0.0).all()
    assert (w_hat32.sum(dim=1) - 1.0).abs().max().item() < 1.0e-5


def test_hat_band_weights_positivity_partition_and_offband_limits() -> None:
    torch.manual_seed(7)
    C, _, scale = _hat_test_grid()
    lam, thr = _sdss_r_like_band()
    # adversarial t: dense sweep, knots pinned exactly onto response points,
    # and whole-grid off-band extremes
    t_pinned = (lam[5] / C[40]) * torch.ones(1, dtype=torch.float64)
    t = torch.cat([
        torch.exp(scale * torch.empty(5_000, dtype=torch.float64).uniform_(-0.6, 0.6)),
        t_pinned,
        torch.tensor([lam[0] / C[0], lam[-1] / C[-1]], dtype=torch.float64),
        torch.tensor([0.2, 5.0], dtype=torch.float64),
    ])
    for dtype in (torch.float64, torch.float32):
        w = _hat_weights(t, C, lam, thr, dtype)
        assert torch.isfinite(w).all()
        assert (w >= 0.0).all()
        assert (w.sum(dim=1) - 1.0).abs().max().item() < 1.0e-5
    # grid entirely above the band: only the first knot's constant
    # extension covers it; entirely below: only the last knot's
    w = _hat_weights(
        torch.tensor([5.0, 0.2], dtype=torch.float64), C, lam, thr, torch.float64
    )
    expected = torch.zeros_like(w)
    expected[0, 0] = 1.0
    expected[1, -1] = 1.0
    assert torch.allclose(w, expected, atol=1e-12, rtol=0.0)


def test_hat_band_weights_window_matches_full_evaluation() -> None:
    torch.manual_seed(9)
    from survey_fusion.pgm.splat_spectrum import _hat_weights_and_grad, _hat_window_width

    n_splats = 3_000
    C, _, scale = _hat_test_grid()
    lam, thr = _sdss_r_like_band()
    shift = torch.empty(n_splats, dtype=torch.float64).uniform_(-0.2, 0.2)
    for dtype in (torch.float64, torch.float32):
        t = torch.exp(scale * shift).to(dtype)
        tables = _hat_band_tables(lam, thr, dtype=dtype, device=torch.device("cpu"))
        width = _hat_window_width(C.to(dtype), tables[-2], tables[-1])
        assert width < 40
        w_full, dw_full = _hat_weights_and_grad(
            t, C.to(dtype), *tables, need_grad=True, window=C.numel(),
        )
        w_win, dw_win = _hat_weights_and_grad(
            t, C.to(dtype), *tables, need_grad=True, window=width,
        )
        assert torch.allclose(w_full, w_win, atol=1e-7, rtol=0.0)
        assert torch.allclose(dw_full, dw_win, atol=1e-4, rtol=1e-6)
        if dtype is torch.float64:
            assert (w_full - w_win).abs().max().item() < 1e-14


def test_hat_band_tables_reject_malformed_bandpasses() -> None:
    kwargs = {"dtype": torch.float64, "device": torch.device("cpu")}
    good = torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive photon-weighted"):
        _hat_band_tables(good, torch.zeros(3, dtype=torch.float64), **kwargs)
    with pytest.raises(ValueError, match="strictly increasing"):
        _hat_band_tables(
            torch.tensor([4000.0, 4000.0, 5000.0], dtype=torch.float64),
            torch.tensor([0.1, 0.5, 0.1], dtype=torch.float64),
            **kwargs,
        )
    with pytest.raises(ValueError, match="finite"):
        _hat_band_tables(
            good,
            torch.tensor([0.1, math.nan, 0.1], dtype=torch.float64),
            **kwargs,
        )
    with pytest.raises(ValueError, match="At least two"):
        _hat_band_tables(
            torch.tensor([4000.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            **kwargs,
        )


def test_hat_band_weights_gradient_matches_autograd_reference() -> None:
    torch.manual_seed(4)
    n_splats = 2_000
    C, _, scale = _hat_test_grid()
    lam = torch.linspace(3900.0, 5600.0, 16, dtype=torch.float64)
    thr = torch.sin(torch.linspace(0.0, math.pi, 16, dtype=torch.float64))
    shift = torch.empty(n_splats, dtype=torch.float64).uniform_(-0.3, 0.3)
    values = torch.randn(n_splats, C.numel(), dtype=torch.float64)

    t_new = torch.exp(scale * shift).requires_grad_(True)
    (values * _hat_weights(t_new, C, lam, thr, torch.float64)).sum().backward()
    t_ref = torch.exp(scale * shift).requires_grad_(True)
    (values * _piecewise_linear_band_weights(
        C[None, :] * t_ref[:, None], lam, thr
    )).sum().backward()
    assert torch.allclose(t_new.grad, t_ref.grad, atol=1e-9, rtol=0.0)

    tables = _hat_band_tables(
        lam, thr, dtype=torch.float64, device=torch.device("cpu"),
    )
    t_small = torch.exp(scale * shift[:24]).clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda tv: _HatBandWeights.apply(tv, C, *tables),
        (t_small,),
        eps=1e-7,
        atol=1e-7,
        rtol=1e-5,
    )


def test_hat_band_weights_float32_gradient_is_bounded_and_off_band_silent() -> None:
    """Off-band splats must get an exactly zero shift gradient (no spurious
    boundary leakage), and on-band float32 gradients must stay within a
    scale-relative bound of the float64 truth."""
    torch.manual_seed(6)
    n_splats = 4_000
    C, _, scale = _hat_test_grid()
    lam, thr = _sdss_r_like_band()
    shift = torch.empty(n_splats, dtype=torch.float64).uniform_(-0.15, 0.15)
    t64 = torch.exp(scale * shift)
    values = torch.exp(3.0 * torch.randn(n_splats, C.numel(), dtype=torch.float64))

    t_ref = t64.clone().requires_grad_(True)
    (values * _piecewise_linear_band_weights(
        C[None, :] * t_ref[:, None], lam, thr
    )).sum().backward()

    t32 = t64.clone().requires_grad_(True)
    (values.to(torch.float32) * _hat_weights(
        t32, C, lam, thr, torch.float32
    )).sum().backward()
    err = (t32.grad.to(torch.float64) - t_ref.grad).abs()
    assert err.max().item() < 1.0e-2 * t_ref.grad.abs().max().item()
    material = t_ref.grad.abs() > 0.01 * t_ref.grad.abs().max()
    rel = err[material] / t_ref.grad.abs()[material]
    assert torch.quantile(rel, 0.99).item() < 5.0e-3

    t_off = torch.tensor([5.0, 0.2], dtype=torch.float32, requires_grad=True)
    w_off = _hat_weights(t_off, C, lam, thr, torch.float32)
    w_off.sum().backward()
    assert (t_off.grad == 0.0).all()


def _red_edge_band() -> tuple[torch.Tensor, torch.Tensor]:
    """A z-like band against the red end of the knot grid.

    The grid spans 3000-11000 A, so a modest blueshifting stretch pulls the
    reddest knot inside this band and leaves part of the response uncovered:
    the geometry the z blowup ran on.
    """
    lam = torch.linspace(8200.0, 10500.0, 40, dtype=torch.float64)
    return lam, torch.sin(torch.linspace(0.0, math.pi, 40, dtype=torch.float64))


def _dense_truncated_hat_weights(
    t: float, C: torch.Tensor, lam: torch.Tensor, thr: torch.Tensor,
    n_samples: int = 200_001,
) -> torch.Tensor:
    """Reference band weights by dense quadrature, hats truncated at the grid.

    Independent of every band helper in the module: the photon response is
    sampled on a fine wavelength grid and integrated against each hat, with
    every hat cut off outside the spectrum's own knot span.
    """
    x = torch.linspace(float(lam[0]), float(lam[-1]), n_samples, dtype=torch.float64)
    response = x * torch.from_numpy(
        np.interp(x.numpy(), lam.numpy(), thr.numpy().clip(0.0))
    )
    knots = t * C
    spacing = knots[1:] - knots[:-1]
    one = torch.ones(1, dtype=torch.float64)
    offset = x[None, :] - knots[:, None]
    rise = (1.0 + offset / torch.cat((one, spacing))[:, None]).clamp(0.0, 1.0)
    fall = (1.0 - offset / torch.cat((spacing, one))[:, None]).clamp(0.0, 1.0)
    on_grid = (x >= knots[0]) & (x <= knots[-1])
    hats = torch.minimum(rise, fall) * on_grid
    return torch.trapezoid(hats * response, x) / torch.trapezoid(response, x)


def test_hat_band_weights_dropping_off_grid_match_dense_integration() -> None:
    """With the off-grid mass dropped the weights are the truncated hats'
    own band integrals, so they sum to the on-grid fraction of the response
    rather than to one. The tolerance here is the reference quadrature's, not
    the operator's."""
    C, _, scale = _hat_test_grid(24)
    lam, thr = _red_edge_band()
    tables = _hat_band_tables(
        lam, thr, dtype=torch.float64, device=torch.device("cpu"),
    )
    # -0.30 leaves the response only partly covered; 0.0 covers it entirely
    shift = torch.tensor([-0.30, -0.12, 0.0], dtype=torch.float64)
    t = torch.exp(scale * shift)

    w = _HatBandWeights.apply(t, C, *tables, None, True)

    for row, t_value in enumerate(t):
        reference = _dense_truncated_hat_weights(float(t_value), C, lam, thr)
        assert torch.allclose(w[row], reference, atol=5.0e-6, rtol=0.0)
    assert w.sum(dim=1)[0].item() < 0.4
    assert abs(w.sum(dim=1)[2].item() - 1.0) < 1.0e-12


def test_hat_band_weights_dropping_off_grid_lose_exactly_the_swept_mass() -> None:
    """The dropped mass must be the response mass outside the knot span
    exactly: the surviving weights are the clamped weights with the two edge
    contributions removed and nothing renormalized."""
    C, _, scale = _hat_test_grid()
    lam, thr = _red_edge_band()
    tables = _hat_band_tables(
        lam, thr, dtype=torch.float64, device=torch.device("cpu"),
    )
    total0 = tables[8]
    # the last four uncover the red end of the band, 1.70 the blue end too
    shift = torch.tensor(
        [0.0, 0.20, -0.10, -0.25, -0.45, 1.70], dtype=torch.float64
    )
    t = torch.exp(scale * shift)

    dropped = _HatBandWeights.apply(t, C, *tables, None, True)
    clamped = _HatBandWeights.apply(t, C, *tables)

    first = (t * C[0]).clamp(float(lam[0]), float(lam[-1]))
    last = (t * C[-1]).clamp(float(lam[0]), float(lam[-1]))
    below, _ = splat_spectrum_module._piecewise_linear_response_moments(first, lam, thr)
    upto, _ = splat_spectrum_module._piecewise_linear_response_moments(last, lam, thr)
    off_grid_fraction = (below + (total0 - upto)) / total0

    assert torch.allclose(
        dropped.sum(dim=1), 1.0 - off_grid_fraction, atol=1.0e-12, rtol=0.0
    )
    assert torch.allclose(dropped[:, 1:-1], clamped[:, 1:-1], atol=1.0e-12, rtol=0.0)
    assert torch.allclose(
        dropped[:, 0], clamped[:, 0] - below / total0, atol=1.0e-12, rtol=0.0
    )
    assert torch.allclose(
        dropped[:, -1],
        clamped[:, -1] - (total0 - upto) / total0,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert (dropped >= 0.0).all()
    # the covered stretches must be untouched, the uncovered ones must lose mass
    assert torch.allclose(
        dropped[:2].sum(dim=1), torch.ones(2, dtype=torch.float64),
        atol=1.0e-12, rtol=0.0,
    )
    assert dropped[4].sum().item() < 1.0e-3
    assert dropped[5].sum().item() < 0.75


def test_hat_band_weights_default_still_dumps_off_grid_mass_on_terminal_knots() -> None:
    """Flag off is the pre-fix operator, edge dump and all: the weights are a
    partition of unity and a blueshifted grid parks the swept response mass on
    the last knot."""
    C, _, scale = _hat_test_grid()
    lam, thr = _red_edge_band()
    shift = torch.tensor(
        [0.0, 0.20, -0.10, -0.25, -0.45, 1.70], dtype=torch.float64
    )
    t = torch.exp(scale * shift)

    w = _hat_weights(t, C, lam, thr, torch.float64)

    reference = _piecewise_linear_band_weights(C[None, :] * t[:, None], lam, thr)
    assert torch.allclose(w, reference, atol=1.0e-11, rtol=0.0)
    assert torch.allclose(
        w.sum(dim=1), torch.ones_like(t), atol=1.0e-12, rtol=0.0
    )
    assert w[4, -1].item() > 0.99
    assert w[3, -1].item() > 0.5
    assert w[5, 0].item() > 0.3


def test_hat_band_weights_drop_off_grid_gradient_matches_numerical() -> None:
    """The analytic derivative branch has to follow the new normalization:
    the sampled stretches straddle the red band edge, so about half the rows
    are losing mass and the dropped-mass derivative is live."""
    C, _, scale = _hat_test_grid()
    lam, thr = _red_edge_band()
    tables = _hat_band_tables(
        lam, thr, dtype=torch.float64, device=torch.device("cpu"),
    )
    torch.manual_seed(13)
    shift = torch.empty(48, dtype=torch.float64).uniform_(-0.35, 0.35)
    values = torch.randn(48, C.numel(), dtype=torch.float64)

    t = torch.exp(scale * shift).requires_grad_(True)
    weights = _HatBandWeights.apply(t, C, *tables, None, True)
    partial = weights.sum(dim=1) < 1.0 - 1.0e-9
    assert 8 < int(partial.sum()) < 40
    (values * weights).sum().backward()

    step = 1.0e-7
    base = torch.exp(scale * shift)
    numerical = torch.zeros_like(base)
    for row in range(base.numel()):
        up, down = base.clone(), base.clone()
        up[row] += step
        down[row] -= step
        numerical[row] = (
            (values * _HatBandWeights.apply(up, C, *tables, None, True)).sum()
            - (values * _HatBandWeights.apply(down, C, *tables, None, True)).sum()
        ) / (2.0 * step)
    scale_ref = numerical.abs().max().clamp_min(1e-12)
    assert ((t.grad - numerical).abs() / scale_ref).max().item() < 1.0e-6

    small = torch.exp(scale * shift[:16]).clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda tv: _HatBandWeights.apply(tv, C, *tables, None, True),
        (small,),
        eps=1e-7,
        atol=1e-7,
        rtol=1e-5,
    )


def test_hat_band_weights_drop_off_grid_are_stable_in_float32() -> None:
    """fp32 is the training dtype, and dropping the edge terms removes the
    renormalization that used to absorb their roundoff, so both the weights
    and the shift gradient are checked against the float64 result."""
    torch.manual_seed(19)
    C, _, scale = _hat_test_grid()
    lam, thr = _red_edge_band()
    shift = torch.empty(4_000, dtype=torch.float64).uniform_(-0.35, 0.35)
    base = torch.exp(scale * shift)
    values = torch.exp(3.0 * torch.randn(4_000, C.numel(), dtype=torch.float64))

    grads = {}
    weights = {}
    for dtype in (torch.float64, torch.float32):
        tables = _hat_band_tables(
            lam, thr, dtype=dtype, device=torch.device("cpu"),
        )
        t = base.to(dtype).clone().requires_grad_(True)
        w = _HatBandWeights.apply(t, C.to(dtype), *tables, None, True)
        weights[dtype] = w.detach().to(torch.float64)
        (values.to(dtype) * w).sum().backward()
        grads[dtype] = t.grad.to(torch.float64)

    assert (weights[torch.float32] >= 0.0).all()
    assert (weights[torch.float32] <= 1.0 + 1.0e-6).all()
    assert (weights[torch.float32] - weights[torch.float64]).abs().max().item() < 1.0e-5
    covered = (
        weights[torch.float32].sum(dim=1) - weights[torch.float64].sum(dim=1)
    ).abs()
    assert covered.max().item() < 1.0e-5
    reference = grads[torch.float64]
    error = (grads[torch.float32] - reference).abs()
    assert error.max().item() < 1.0e-2 * reference.abs().max().item()


def test_hat_band_weights_drop_off_grid_window_matches_full_evaluation() -> None:
    from survey_fusion.pgm.splat_spectrum import _hat_weights_and_grad, _hat_window_width

    torch.manual_seed(17)
    C, _, scale = _hat_test_grid()
    lam, thr = _red_edge_band()
    shift = torch.empty(2_000, dtype=torch.float64).uniform_(-0.3, 0.3)
    t = torch.exp(scale * shift)
    tables = _hat_band_tables(
        lam, thr, dtype=torch.float64, device=torch.device("cpu"),
    )
    width = _hat_window_width(C, tables[-2], tables[-1])

    w_full, dw_full = _hat_weights_and_grad(
        t, C, *tables, need_grad=True, window=C.numel(), drop_off_grid=True,
    )
    w_win, dw_win = _hat_weights_and_grad(
        t, C, *tables, need_grad=True, window=width, drop_off_grid=True,
    )

    assert (w_full - w_win).abs().max().item() < 1e-14
    assert (dw_full - dw_win).abs().max().item() < 1e-12


def test_splat_band_amplitudes_off_grid_zero_drops_unmodelled_red_flux() -> None:
    """End to end: a blueshifting stretch aims a red band at knots the grid no
    longer covers. Under the clamp the terminal knot absorbs the whole band;
    under the fix the band amplitude collapses to what the grid still covers."""
    loglam_min = math.log(3000.0)
    loglam_max = math.log(11_000.0)
    response_lam = torch.linspace(8200.0, 10500.0, 24, dtype=torch.float64)
    throughput = torch.sin(
        torch.linspace(0.0, math.pi, 24, dtype=torch.float64)
    )

    amplitudes = {}
    for mode in ("clamp", "zero"):
        torch.manual_seed(23)
        field = SplatField(
            n_splats=8,
            n_templates=4,
            loglam_grid=torch.linspace(-1.0, 1.0, 96),
            sed_off_grid=mode,
        ).double()
        field.sed.set_loglam_scale(0.5 * (loglam_max - loglam_min))
        with torch.no_grad():
            field.sed_shift[:4] = -0.35
            field.sed_shift[4:] = 0.0
        amplitudes[mode] = splat_band_amplitudes(
            field,
            {"z": (response_lam, throughput)},
            loglam_min=loglam_min,
            loglam_max=loglam_max,
        )["z"]

    shifted = amplitudes["zero"][:4] / amplitudes["clamp"][:4]
    unshifted = amplitudes["zero"][4:] / amplitudes["clamp"][4:]
    assert shifted.max().item() < 0.5
    assert torch.allclose(
        unshifted, torch.ones_like(unshifted), atol=1.0e-12, rtol=0.0
    )


def test_splat_band_amplitudes_leave_pinned_gauge_knots_without_gradient() -> None:
    """The pin has to survive the whole band path, not just templates(): the
    two terminal knots must come out of a band backward with exactly no
    gradient of their own."""
    loglam_min = math.log(3000.0)
    loglam_max = math.log(11_000.0)
    response_lam, throughput = _red_edge_band()
    torch.manual_seed(29)
    field = SplatField(
        n_splats=16,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 96),
        sed_pin_gauge_knots=True,
    ).double()
    field.sed.set_loglam_scale(0.5 * (loglam_max - loglam_min))
    with torch.no_grad():
        field.sed_shift.uniform_(-0.3, 0.3)

    amplitudes = splat_band_amplitudes(
        field,
        {"z": (response_lam, throughput)},
        loglam_min=loglam_min,
        loglam_max=loglam_max,
    )["z"]
    amplitudes.sum().backward()

    grad = field.sed.raw_templates.grad
    assert grad is not None
    assert (grad[:, 0] == 0.0).all()
    assert (grad[:, -1] == 0.0).all()
    assert grad[:, 1:-1].abs().sum().item() > 0.0


def test_splat_band_amplitudes_float32_matches_float64() -> None:
    torch.manual_seed(11)
    field = SplatField(
        n_splats=64,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.04,
    )
    loglam_min = math.log(3400.0)
    loglam_max = math.log(9200.0)
    field.sed.set_loglam_scale(0.5 * (loglam_max - loglam_min))
    with torch.no_grad():
        field.sed_shift.uniform_(-0.6, 0.6)
    response_lam = torch.tensor(
        [3800.0, 4200.0, 5100.0, 6200.0, 7100.0], dtype=torch.float64
    )
    throughput = torch.tensor([0.0, 0.75, 1.0, 0.4, 0.0], dtype=torch.float64)
    amp32 = splat_band_amplitudes(
        field,
        {"x": (response_lam, throughput)},
        loglam_min=loglam_min,
        loglam_max=loglam_max,
    )["x"]
    amp64 = splat_band_amplitudes(
        field.double(),
        {"x": (response_lam, throughput)},
        loglam_min=loglam_min,
        loglam_max=loglam_max,
    )["x"]
    rel = (amp32.to(torch.float64) - amp64).abs() / amp64.abs().clamp_min(1e-30)
    assert rel.max().item() < 1.0e-4


def test_fiber_operator_matches_aperture_integral_of_image_operator() -> None:
    """The fiber prediction must equal the image-operator surface brightness
    integrated over the fiber aperture. A flat SED removes the spectral factor,
    so any disagreement is a units or geometry error between the two paths."""
    torch.manual_seed(7)
    trainer, _ = _analytic_spectrum_trainer(failing_factory=False)
    model = trainer.model
    exposure = trainer.runtime_by_exposure[1].exposure
    aperture = exposure.aperture

    with torch.no_grad():
        # flat unit SED: raw template values chosen so softplus-mixed values are constant
        model.sed.raw_templates.fill_(0.0)
        model.sed_coeffs.fill_(-20.0)
        flat_level = float(
            (torch.nn.functional.softplus(model.sed_coeffs[:1]) @ model.sed.templates())[0, 0]
        )
        model.sed_shift.fill_(0.0)

    local = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        fiber = splat_fiber_spectrum(
            model,
            exposure,
            local,
            loglam_min=trainer.loglam_min,
            loglam_max=trainer.loglam_max,
            apply_spatial_psf=True,
        )

        # dense uniform sampling of the disk in normalized coordinates
        wcs = exposure.wcs
        center_x, center_y = wcs.omega_xyz_to_norm_torch(
            torch.from_numpy(
                radec_deg_to_omega_xyz(
                    np.array([aperture.center_radec_deg[0]]),
                    np.array([aperture.center_radec_deg[1]]),
                )
            ).reshape(-1, 3)
        )
        norm_per_arcsec = 2.0 / (wcs.width * wcs.pixel_scale_arcsec)
        radius_norm = float(aperture.radius_arcsec) * norm_per_arcsec
        n_grid = 401
        gx = torch.linspace(-radius_norm, radius_norm, n_grid) + center_x
        gy = torch.linspace(-radius_norm, radius_norm, n_grid) + center_y
        xx, yy = torch.meshgrid(gx, gy, indexing="ij")
        inside = (
            (xx - center_x).square() + (yy - center_y).square()
        ) <= radius_norm**2
        px = xx[inside].to(model.mu.dtype)
        py = yy[inside].to(model.mu.dtype)
        psf_sigma_norm = (
            float(exposure.psf_fwhm_arcsec) / 2.355 * norm_per_arcsec
        )
        band_amp = model.fluxes() * flat_level
        surface = tiled_point_prediction(
            model, px, py, torch.full_like(px, psf_sigma_norm), band_amp,
        )
        area = math.pi * float(aperture.radius_arcsec) ** 2
        aperture_flux = float(surface.mean()) * area

    # every wide bin of the flat SED should equal the aperture-integrated flux
    for bin_index in range(3):
        ratio = float(fiber[bin_index]) / aperture_flux
        assert abs(ratio - 1.0) < 0.02, (
            f"bin {bin_index}: fiber={float(fiber[bin_index]):.6e} "
            f"aperture integral={aperture_flux:.6e} ratio={ratio:.4f}"
        )


def _high_dynamic_range_spectra(
    log_value_std: float, seed: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.linspace(3600.0, 10000.0, 96, dtype=torch.float64)
    shifts = 20.0 * torch.randn(64, 1, generator=generator, dtype=torch.float64)
    knots = base[None, :] + shifts
    values = torch.exp(
        log_value_std * torch.randn(64, 96, generator=generator, dtype=torch.float64)
    )
    centres = torch.linspace(3650.0, 9950.0, 64, dtype=torch.float64)
    lower, upper = centres - 1.0, centres + 1.0
    sigma = torch.full((64,), 1.2, dtype=torch.float64)
    return knots, values, lower, upper, sigma


def test_positive_hat_matches_independent_quadrature() -> None:
    knots = np.array(
        [[4800.0, 4950.0, 5100.0, 5350.0], [4700.0, 5000.0, 5200.0, 5500.0]],
        dtype=np.float64,
    )
    values = np.array(
        [[0.8, 1.7, 0.5, 2.1], [1.3, 0.4, 1.9, 1.1]],
        dtype=np.float64,
    )
    lower = np.array([4850.0, 5050.0, 5250.0], dtype=np.float64)
    upper = np.array([4925.0, 5160.0, 5400.0], dtype=np.float64)
    sigma = np.array([0.0, 18.0, 47.0], dtype=np.float64)
    got = positive_hat_bin_average(
        torch.tensor(knots),
        torch.tensor(values),
        torch.tensor(lower),
        torch.tensor(upper),
        torch.tensor(sigma),
    ).numpy()
    expected = _quadrature_reference(knots, values, lower, upper, sigma)
    np.testing.assert_allclose(got, expected, atol=1.0e-5, rtol=1.0e-5)


def test_positive_hat_matches_hinge_in_float64_at_high_dynamic_range() -> None:
    for log_value_std in (4.0, 6.0):
        knots, values, lower, upper, sigma = _high_dynamic_range_spectra(
            log_value_std, seed=42
        )
        hinge = piecewise_linear_bin_average(knots, values, lower, upper, sigma)
        hat = positive_hat_bin_average(knots, values, lower, upper, sigma)
        delta = torch.linalg.vector_norm(hat - hinge)
        assert float(delta / torch.linalg.vector_norm(hinge)) < 1.0e-9
        assert float((hat - hinge).abs().max()) < 1.0e-9 * float(hinge.abs().max())


def test_positive_hat_is_nonnegative_where_float32_hinge_is_not() -> None:
    for log_value_std in (4.0, 6.0):
        for seed in (7, 42):
            knots, values, lower, upper, sigma = _high_dynamic_range_spectra(
                log_value_std, seed
            )
            reference = piecewise_linear_bin_average(
                knots, values, lower, upper, sigma
            )
            # Even float64 hinge arithmetic can leave a stray negative at
            # roundoff scale; only the hat factorization guarantees sign.
            assert float((-reference).max()) <= 1.0e-12 * float(reference.abs().max())
            hinge32 = piecewise_linear_bin_average(
                knots.float(), values.float(), lower.float(),
                upper.float(), sigma.float(),
            )
            hat32 = positive_hat_bin_average(
                knots.float(), values.float(), lower.float(),
                upper.float(), sigma.float(),
            )
            assert int((hinge32 < 0).sum()) > 0
            assert int((hat32 < 0).sum()) == 0
            delta = torch.linalg.vector_norm(hat32.double() - reference)
            assert float(delta / torch.linalg.vector_norm(reference)) < 1.0e-4


def test_positive_hat_constant_spectrum_is_exact() -> None:
    knots, _, lower, upper, sigma = _high_dynamic_range_spectra(4.0, seed=7)
    values = torch.full_like(knots, 3.7)
    for dtype, tolerance in ((torch.float64, 1.0e-12), (torch.float32, 1.0e-6)):
        got = positive_hat_bin_average(
            knots.to(dtype), values.to(dtype), lower.to(dtype),
            upper.to(dtype), sigma.to(dtype),
        )
        torch.testing.assert_close(
            got,
            torch.full_like(got, 3.7),
            rtol=tolerance,
            atol=tolerance,
        )


def test_positive_hat_gradients_match_float64_hinge() -> None:
    knots, values, lower, upper, sigma = _high_dynamic_range_spectra(4.0, seed=42)

    def signed_projection_grads(fn, lam, vals):
        lam = lam.detach().clone().requires_grad_(True)
        vals = vals.detach().clone().requires_grad_(True)
        output = fn(lam, vals, lower.to(lam.dtype), upper.to(lam.dtype), sigma.to(lam.dtype))
        projection = torch.linspace(
            -1.0, 1.0, output.numel(), dtype=output.dtype
        ).reshape(output.shape)
        return torch.autograd.grad((output * projection).sum(), (lam, vals))

    ref_lam, ref_vals = signed_projection_grads(
        piecewise_linear_bin_average, knots, values
    )
    hat_lam, hat_vals = signed_projection_grads(
        positive_hat_bin_average, knots, values
    )
    for got, ref in ((hat_lam, ref_lam), (hat_vals, ref_vals)):
        rel = torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)
        assert float(rel) < 1.0e-9

    hat32_lam, hat32_vals = signed_projection_grads(
        positive_hat_bin_average, knots.float(), values.float()
    )
    for got, ref in ((hat32_lam, ref_lam), (hat32_vals, ref_vals)):
        rel = torch.linalg.vector_norm(got.double() - ref) / torch.linalg.vector_norm(ref)
        assert float(rel) < 1.0e-3


def test_positive_hat_lsf_mode_branches_match_generic_path() -> None:
    knots, values, lower, upper, _ = _high_dynamic_range_spectra(4.0, seed=8)
    mixed_sigma = torch.zeros_like(lower)
    mixed_sigma[::2] = 1.2
    generic = positive_hat_bin_average(
        knots, values, lower, upper, mixed_sigma, lsf_mode=None
    )
    zero_rows = positive_hat_bin_average(
        knots, values, lower[1::2], upper[1::2],
        mixed_sigma[1::2], lsf_mode="zero",
    )
    positive_rows = positive_hat_bin_average(
        knots, values, lower[::2], upper[::2],
        mixed_sigma[::2], lsf_mode="positive",
    )
    torch.testing.assert_close(generic[1::2], zero_rows, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(generic[::2], positive_rows, rtol=1e-12, atol=1e-12)


def test_positive_hat_zero_fully_off_span_is_silent_in_values_and_gradients() -> None:
    for dtype in (torch.float32, torch.float64):
        shift = torch.tensor([0.0], dtype=dtype, requires_grad=True)
        knot_lam = (
            torch.tensor([[4000.0, 5000.0, 6000.0]], dtype=dtype)
            + 100.0 * shift[:, None]
        )
        knot_values = torch.tensor(
            [[2.0, 7.0, 11.0]], dtype=dtype, requires_grad=True
        )
        output = positive_hat_bin_average(
            knot_lam,
            knot_values,
            torch.tensor([7000.0], dtype=dtype),
            torch.tensor([7100.0], dtype=dtype),
            torch.tensor([0.0], dtype=dtype),
            off_grid="zero",
        )

        assert torch.equal(output, torch.zeros_like(output))
        output.sum().backward()
        assert torch.equal(knot_values.grad, torch.zeros_like(knot_values))
        assert torch.equal(shift.grad, torch.zeros_like(shift))


def test_positive_hat_zero_straddling_bin_matches_span_truncated_quadrature() -> None:
    knot_lam = torch.tensor([[4000.0, 5000.0, 6000.0]], dtype=torch.float64)
    knot_values = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float64)
    lower = torch.tensor([3500.0, 5500.0], dtype=torch.float64)
    upper = torch.tensor([4500.0, 6500.0], dtype=torch.float64)
    sigma = torch.tensor([0.0, 35.0], dtype=torch.float64)
    got = positive_hat_bin_average(
        knot_lam,
        knot_values,
        lower,
        upper,
        sigma,
        off_grid="zero",
    )[:, 0]

    uniform_x, uniform_w = np.polynomial.legendre.leggauss(240)
    expected = []
    for lo, hi, sigma_value in zip(lower.tolist(), upper.tolist(), sigma.tolist()):
        clip_lo = max(lo, float(knot_lam[0, 0]))
        clip_hi = min(hi, float(knot_lam[0, -1]))
        if clip_hi <= clip_lo:
            expected.append(0.0)
            continue
        if sigma_value > 0.0:
            # Swap the integrations: the normal CDF is smooth on each
            # piecewise-linear source segment, unlike a nested Hermite rule
            # crossing the source knots.
            expected_value = 0.0
            for source_lo, source_hi, source_left, source_right in (
                (4000.0, 5000.0, 1.0, 4.0),
                (5000.0, 6000.0, 4.0, 2.0),
            ):
                x = source_lo + 0.5 * (uniform_x + 1.0) * (source_hi - source_lo)
                source_value = source_left + (
                    source_right - source_left
                ) * (x - source_lo) / (source_hi - source_lo)
                probability = 0.5 * (
                    1.0 + torch.erf(
                        torch.tensor((clip_hi - x) / (math.sqrt(2.0) * sigma_value))
                    )
                ) - 0.5 * (
                    1.0 + torch.erf(
                        torch.tensor((clip_lo - x) / (math.sqrt(2.0) * sigma_value))
                    )
                )
                expected_value += float(
                    np.sum(
                        0.5
                        * uniform_w
                        * (source_hi - source_lo)
                        * source_value
                        * probability.numpy()
                    )
                )
            expected.append(expected_value / (hi - lo))
            continue
        else:
            x = clip_lo + 0.5 * (uniform_x + 1.0) * (clip_hi - clip_lo)
            values = np.interp(
                x,
                knot_lam[0].numpy(),
                knot_values[0].numpy(),
                left=0.0,
                right=0.0,
            )
            expected.append(
                float(
                    np.sum(0.5 * uniform_w * values)
                    * (clip_hi - clip_lo)
                    / (hi - lo)
                )
            )

    np.testing.assert_allclose(got.numpy(), np.asarray(expected), atol=2.0e-6, rtol=0.0)


def test_positive_hat_clamp_default_is_bitwise_explicit() -> None:
    for dtype in (torch.float32, torch.float64):
        knot_lam = torch.tensor(
            [[4000.0, 5000.0, 6000.0], [4050.0, 5050.0, 6050.0]],
            dtype=dtype,
            requires_grad=True,
        )
        knot_values = torch.tensor(
            [[1.0, 4.0, 2.0], [2.0, 3.0, 8.0]],
            dtype=dtype,
            requires_grad=True,
        )
        lower = torch.tensor([3500.0, 5500.0], dtype=dtype)
        upper = torch.tensor([4500.0, 6500.0], dtype=dtype)
        sigma = torch.tensor([0.0, 20.0], dtype=dtype)

        default = positive_hat_bin_average(
            knot_lam, knot_values, lower, upper, sigma, lsf_mode=None,
        )
        default_grad = torch.autograd.grad(
            default.square().sum(), (knot_lam, knot_values), retain_graph=True,
        )
        explicit = positive_hat_bin_average(
            knot_lam, knot_values, lower, upper, sigma,
            lsf_mode=None, off_grid="clamp",
        )
        explicit_grad = torch.autograd.grad(
            explicit.square().sum(), (knot_lam, knot_values),
        )

        assert torch.equal(default, explicit)
        assert torch.equal(default_grad[0], explicit_grad[0])
        assert torch.equal(default_grad[1], explicit_grad[1])


def test_positive_hat_zero_paired_matches_outer_product_to_fp_accumulation() -> None:
    knot_lam = torch.tensor(
        [[4000.0, 5000.0, 6000.0], [4050.0, 5050.0, 6050.0]],
        dtype=torch.float64,
    )
    knot_values = torch.tensor(
        [[1.0, 4.0, 2.0], [2.0, 3.0, 8.0]], dtype=torch.float64,
    )
    lower = torch.tensor([3500.0, 5500.0], dtype=torch.float64)
    upper = torch.tensor([4500.0, 6500.0], dtype=torch.float64)
    sigma = torch.tensor([15.0, 15.0], dtype=torch.float64)
    outer = positive_hat_bin_average(
        knot_lam, knot_values, lower, upper, sigma,
        lsf_mode="positive", off_grid="zero",
    )
    paired = paired_positive_hat_bin_average(
        knot_lam,
        knot_values,
        lower,
        upper,
        sigma,
        lsf_mode="positive",
        off_grid="zero",
        blocks=[(0, 1), (1, 2)],
    )

    torch.testing.assert_close(
        torch.diagonal(outer), paired, rtol=1.0e-12, atol=1.0e-12,
    )


def test_positive_hat_zero_fiber_tiling_matches_imaging_lost_mass() -> None:
    dtype = torch.float64
    response_lam = torch.tensor([3900.0, 5100.0], dtype=dtype)
    throughput = torch.ones(2, dtype=dtype)
    C = torch.tensor([4200.0, 4800.0], dtype=dtype)
    tables = _hat_band_tables(response_lam, throughput, dtype=dtype, device=C.device)
    imaging = _HatBandWeights.apply(
        torch.ones(1, dtype=dtype), C, *tables, None, True,
    )[0].sum()

    edges = torch.arange(3900.0, 5100.0 + 10.0, 10.0, dtype=dtype)
    bins = positive_hat_bin_average(
        C[None, :],
        torch.ones(1, 2, dtype=dtype),
        edges[:-1],
        edges[1:],
        torch.zeros(edges.numel() - 1, dtype=dtype),
        off_grid="zero",
    )[:, 0]
    response_integral, _ = splat_spectrum_module._piecewise_linear_response_moments(
        edges, response_lam, throughput,
    )
    fiber = (bins * response_integral.diff()).sum() / response_integral[-1]

    torch.testing.assert_close(fiber, imaging, rtol=0.0, atol=1.0e-12)
