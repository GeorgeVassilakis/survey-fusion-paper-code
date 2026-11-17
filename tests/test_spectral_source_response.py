from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from support.synthetic_trainer import spec_exposure
from survey_fusion.pgm.contracts import SkyDiskApertureSpec, validate_kernel_samples
from survey_fusion.pgm.kernel_builders import SpectralBinKernelBuilder, TophatBinSampler, GaussianLSF
from survey_fusion.pgm.splat_spectrum import build_spectrum_cache, splat_fiber_spectrum
from test_splat_spectrum import _fiber_spectrum_batch, _fiber_path_prediction, _analytic_spectrum_trainer


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), np.array([1.0, 2.0]), np.array([[1.0]])]
)
def test_invalid_response_fails_closed(value):
    with pytest.raises(ValueError, match="spectral_source_scale"):
        replace(spec_exposure(), spectral_source_scale=value)


@pytest.mark.parametrize("flat", [False, True])
def test_native_bin_response_predictions_and_gradients(flat):
    make_field, caches, candidates, indices = _fiber_spectrum_batch(
        n_exposures=3,
        bins_per_exposure=(1, 3, 2),
        n_splats=17,
        dtype=torch.float64,
    )
    candidates = [torch.arange(17)] * 3
    responses = [None, 2.5, np.array([0.7, 1.3, 2.1])]
    scaled = []
    factors = []
    for cache, value, local in zip(caches, responses, indices):
        e = replace(cache.source_scale_exposure, spectral_source_scale=value)
        scaled.append(
            build_spectrum_cache(
                e,
                e.aperture,
                apply_spatial_psf=True,
                radial_nodes=2,
                angular_nodes=4,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
        )
        factors.append(
            torch.ones(local.numel(), dtype=torch.float64)
            if value is None
            else torch.tensor(np.broadcast_to(value, (3,)).copy(), dtype=torch.float64)[local]
        )
    factor = torch.cat(factors)
    a, b = make_field(), make_field()
    expected = _fiber_path_prediction(a, caches, candidates, indices, flat=flat) * factor
    actual = _fiber_path_prediction(b, scaled, candidates, indices, flat=flat)
    assert torch.equal(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is not None:
            ga = pa.grad.to_dense() if pa.grad.is_sparse else pa.grad
            gb = pb.grad.to_dense() if pb.grad.is_sparse else pb.grad
            torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)
    offset = 0
    for cache, local, candidate in zip(scaled, indices, candidates):
        single = splat_fiber_spectrum(
            b,
            cache.source_scale_exposure,
            local,
            loglam_min=math.log(3400.0),
            loglam_max=math.log(9200.0),
            apply_spatial_psf=True,
            radial_nodes=2,
            angular_nodes=4,
            cache=cache,
            candidate=candidate,
        )
        torch.testing.assert_close(single, actual[offset : offset + local.numel()], rtol=1e-12, atol=1e-12)
        offset += local.numel()


@pytest.mark.parametrize("flat", [False, True])
def test_cache_rejects_mutated_response(flat):
    make_field, caches, candidates, indices = _fiber_spectrum_batch(
        n_exposures=1,
        bins_per_exposure=(3,),
        n_splats=7,
    )
    e = caches[0].source_scale_exposure
    e.spectral_source_scale = np.ones(3)
    with pytest.raises(ValueError, match="stale"):
        _fiber_path_prediction(make_field(), caches, candidates, indices, flat=flat)
    c = build_spectrum_cache(
        e,
        e.aperture,
        apply_spatial_psf=True,
        radial_nodes=2,
        angular_nodes=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    e.spectral_source_scale[0] = 2.0
    with pytest.raises(ValueError, match="stale"):
        _fiber_path_prediction(make_field(), [c], candidates, indices, flat=flat)


def test_mc_response_uses_Z_and_keeps_normalized_weights():
    e = replace(
        spec_exposure(lam_spec=np.array([5000.0, 5100.0])),
        lam_edges=np.array([[4995.0, 5005.0], [5095.0, 5105.0]]),
        aperture=SkyDiskApertureSpec((180.0, 0.0), 0.75),
        psf_fwhm_arcsec=0.0,
    )
    builder = SpectralBinKernelBuilder()

    def run(exposure):
        return builder.build(
            exposure,
            torch.tensor([1, 0, 1]),
            k=16,
            rng=torch.Generator().manual_seed(5),
            device=torch.device("cpu"),
            lam_sampler=TophatBinSampler(exposure.lam_edges),
            lsf=GaussianLSF(0.0),
            aperture_sampler=builder._aperture_sampler(exposure),
        )

    base = run(e)
    scaled = run(replace(e, spectral_source_scale=np.array([2.0, 3.0])))
    validate_kernel_samples(scaled)
    assert torch.equal(base.w, scaled.w)
    assert torch.equal(base.loglam, scaled.loglam)
    assert torch.equal(base.omega_xyz, scaled.omega_xyz)
    assert torch.equal(scaled.Z, base.Z * torch.tensor([3.0, 2.0, 3.0]))


def test_none_response_preserves_source_and_additive_training_prediction():
    trainer, _ = _analytic_spectrum_trainer(sparse_splat_gradients=True)
    indices = torch.arange(1, trainer.data.batch.y.numel())
    before = trainer._compute_analytic_spectral_source_mu(indices)
    for runtime in trainer.runtime_by_exposure.values():
        assert runtime.exposure.spectral_source_scale is None
    assert torch.equal(before, trainer._compute_analytic_spectral_source_mu(indices))


@pytest.mark.parametrize("value", [None, 1.0, np.array([1.0, 1.0])])
def test_calibrated_metadata_cannot_silently_omit_or_change_scale(value):
    e = replace(spec_exposure(), aperture=SkyDiskApertureSpec((180.0, 0.0), 0.75), psf_fwhm_arcsec=1.0)
    factor = 1.0 / -math.expm1(-0.5 * (0.75 / (1.0 / 2.355)) ** 2)
    metadata = {
        "fwhm_arcsec": 1.0,
        "calibration": {"mode": "gaussian_point_source"},
        "resolved_source_scale": factor,
    }
    with pytest.raises(ValueError, match="spectral_source_scale"):
        replace(e, spectral_response_metadata=metadata, spectral_source_scale=value)


@pytest.mark.parametrize("flat", [False, True])
def test_response_precedes_learned_calibration_gate_and_additive(flat, monkeypatch):
    monkeypatch.setenv("SPLAT_FIBER_PATH", "flat" if flat else "blocked")
    trainer, _ = _analytic_spectrum_trainer(sparse_splat_gradients=True)
    indices = torch.arange(1, 4)
    source = trainer._compute_analytic_spectral_source_mu(indices)
    e = trainer.runtime_by_exposure[1].exposure
    e.spectral_source_scale = np.array([2.0, 3.0, 4.0])
    trainer._spectrum_cache_by_exposure.clear()
    scaled = trainer._compute_analytic_spectral_source_mu(indices)
    torch.testing.assert_close(scaled, source * torch.tensor([2.0, 3.0, 4.0]))
    trainer.data.batch.source_gate = torch.tensor([1.0, 1.0, 0.0, 0.5])
    trainer.spec_additive_by_exposure[1] = np.array([11.0, 13.0, 17.0], dtype=np.float32)

    class Calibrator:
        def scale_for_full_exposure(self, exp_id):
            return torch.tensor([1.5, 0.7, 2.0])

    trainer.spectral_calibrator = Calibrator()
    trainer.spectral_calibration_cfg.apply_to_predictions = True
    out = trainer._finalize_fiber_spectrum(1, scaled, include_additive=True)
    expected = scaled * torch.tensor([1.5, 0.7, 2.0]) * torch.tensor([1.0, 0.0, 0.5]) + torch.tensor(
        [11.0, 13.0, 17.0]
    )
    assert torch.equal(out, expected)
    assert out[1] == 13.0


def test_mc_training_prediction_scales_source_only_and_preserves_image():
    trainer, factory = _analytic_spectrum_trainer(failing_factory=False)
    indices = torch.arange(4)
    # Materialize generator before saving its state.
    trainer.compute_source_mu(indices)
    state = factory.rng_state_dict()
    base = trainer.compute_source_mu(indices)
    e = trainer.runtime_by_exposure[1].exposure
    e.spectral_source_scale = np.array([2.0, 3.0, 4.0])
    factory.load_rng_state_dict(state)
    scaled = trainer.compute_source_mu(indices)
    response = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(scaled, base * response, rtol=2e-6, atol=1e-6)
    trainer.data.batch.additive = torch.tensor([7.0, 11.0, 13.0, 17.0])
    trainer.data.batch.source_gate = torch.tensor([1.0, 1.0, 0.0, 0.5])
    factory.load_rng_state_dict(state)
    actual = trainer.compute_mu(indices)
    expected = scaled * trainer.data.batch.source_gate + trainer.data.batch.additive
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
    assert actual[2] == 13.0


def _calibrated_exposure_fixture():
    make_field, caches, candidates, indices = _fiber_spectrum_batch(
        n_exposures=1, bins_per_exposure=(3,), n_splats=7
    )
    e = caches[0].source_scale_exposure
    factor = 1.0 / -math.expm1(-0.5 * (e.aperture.radius_arcsec / (e.psf_fwhm_arcsec / 2.355)) ** 2)
    metadata = {
        "fwhm_arcsec": e.psf_fwhm_arcsec,
        "calibration": {"mode": "gaussian_point_source"},
        "resolved_source_scale": factor,
    }
    e = replace(e, spectral_response_metadata=metadata, spectral_source_scale=factor)
    return make_field, e, candidates, indices


def test_calibrated_response_requires_enabled_spatial_psf():
    make_field, e, candidates, indices = _calibrated_exposure_fixture()
    with pytest.raises(ValueError, match="apply_spatial_psf=True"):
        build_spectrum_cache(
            e,
            e.aperture,
            apply_spatial_psf=False,
            radial_nodes=2,
            angular_nodes=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="apply_spatial_psf=True"):
        splat_fiber_spectrum(
            make_field(),
            e,
            indices[0],
            apply_spatial_psf=False,
            loglam_min=math.log(3400.0),
            loglam_max=math.log(9200.0),
        )


@pytest.mark.parametrize("mutation", ["radius", "seeing", "metadata"])
def test_calibrated_cache_rejects_geometry_or_provenance_changes(mutation):
    make_field, e, candidates, indices = _calibrated_exposure_fixture()
    cache = build_spectrum_cache(
        e,
        e.aperture,
        apply_spatial_psf=True,
        radial_nodes=2,
        angular_nodes=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if mutation == "radius":
        e.aperture = replace(e.aperture, radius_arcsec=e.aperture.radius_arcsec * 1.1)
    elif mutation == "seeing":
        e.psf_fwhm_arcsec *= 1.1
    else:
        e.spectral_response_metadata["new_provenance"] = "new claim"
    with pytest.raises(ValueError):
        _fiber_path_prediction(make_field(), [cache], candidates, indices, flat=True)


@pytest.mark.parametrize("path", ["blocked", "flat"])
def test_corrected_response_runs_three_finite_joint_training_steps(monkeypatch, path):
    """Disposable engineering smoke: unchanged fixture optimizer/trainability."""
    import test_splat_spectrum as fixture_module

    original_exposure = fixture_module.ExposureState
    monkeypatch.setenv("SPLAT_FIBER_PATH", path)

    def exposure_with_declared_response(**kwargs):
        aperture = kwargs.get("aperture")
        if aperture is not None:
            seeing = kwargs["psf_fwhm_arcsec"]
            factor = 1.0 / -math.expm1(-0.5 * (aperture.radius_arcsec / (seeing / 2.355)) ** 2)
            kwargs["spectral_source_scale"] = factor
            kwargs["spectral_response_metadata"] = {
                "fwhm_arcsec": seeing,
                "calibration": {"mode": "gaussian_point_source"},
                "resolved_source_scale": factor,
            }
        return original_exposure(**kwargs)

    monkeypatch.setattr(fixture_module, "ExposureState", exposure_with_declared_response)
    torch.manual_seed(27)
    trainer, _ = fixture_module._analytic_spectrum_trainer(sparse_splat_gradients=True)
    image_indices = torch.tensor([0])
    spec_indices = torch.tensor([1, 2, 3])
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())
    with torch.no_grad():
        image = trainer._compute_analytic_image_source_mu(
            image_indices,
            trainer.data.batch.exposure_id[image_indices],
            trainer.data.batch.datum_index[image_indices],
        )
        source = trainer._compute_analytic_spectral_source_mu(spec_indices)
        teacher = torch.cat([image, source])
        additive = torch.tensor([0.3, 0.7, 0.9, 1.1])
        trainer.data.batch.additive = additive
        # A nearby, source-consistent target gives a genuine nonzero update.
        trainer.data.batch.y.copy_(1.05 * teacher + additive)
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    buffers = {name: buffer.detach().clone() for name, buffer in trainer.model.named_buffers()}
    cache = trainer._spectrum_cache_by_exposure[1]
    optimizer = trainer.optimizer
    splat_optimizer = trainer.splat_optimizer
    for step in range(3):
        stats = trainer._train_step(image_indices, spec_indices, global_step=step)
        assert math.isfinite(stats["local_total_nll"])
        assert math.isfinite(stats["local_spec_nll"])
        assert trainer.optimizer is optimizer
        assert trainer.splat_optimizer is splat_optimizer
        assert trainer._spectrum_cache_by_exposure[1] is cache
        for parameter in trainer.model.parameters():
            assert torch.isfinite(parameter).all()
            if parameter.grad is not None:
                gradient = parameter.grad.to_dense() if parameter.grad.is_sparse else parameter.grad
                assert torch.isfinite(gradient).all()
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters()
    )
    assert trainer.model.sed.raw_templates.grad.abs().sum() > 0
    for name, buffer in trainer.model.named_buffers():
        assert torch.equal(buffer, buffers[name])
