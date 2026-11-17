import numpy as np
import pytest
import torch

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.data.sdss_scene import SdssSceneData, SdssSpectrumData
from survey_fusion.models.splat_field import SplatField
from survey_fusion.training import datum_trainer as datum_trainer_module
from survey_fusion.training.datum_trainer import DatumTrainer

SIZE = 32
PIX_ARCSEC = 0.4


class ConsistentWcs:
    """Linear TAN-like WCS with exactly PIX_ARCSEC per pixel."""

    def all_pix2world(self, x, y, origin):
        del origin
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        step = PIX_ARCSEC / 3600.0
        return 180.0 + x_arr * step, 0.5 + y_arr * step


def _realistic_scene() -> SdssSceneData:
    rng = np.random.default_rng(7)
    imgs = {b: rng.uniform(0.5, 1.5, (SIZE, SIZE)).astype(np.float32) for b in "gri"}
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    sigma = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    wcs = ConsistentWcs()
    spectrum = SdssSpectrumData(
        label="fiber", role="science", path="synthetic.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((SIZE, SIZE), dtype=np.float32),
        spec_ivar=np.ones_like(lam_spec), spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec), sigma_spec_flambda=sigma,
        center_ra_dec=(180.0008, 0.5008), fiber_radius_arcsec=1.5, metadata={},
    )
    ones = np.ones_like(lam_spec)
    return SdssSceneData(
        images={b: imgs[b].copy() for b in imgs},
        images_flambda=imgs,
        spectra=[spectrum],
        lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(),
        flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((SIZE, SIZE), dtype=np.float32),
        fiber_area_arcsec2=7.0,
        pixel_scale_arcsec=PIX_ARCSEC,
        fiber_radius_arcsec=1.5,
        metadata={},
        wcs_per_band={b: wcs for b in imgs},
        pixel_scale_arcsec_per_band={b: PIX_ARCSEC for b in imgs},
        pixel_area_arcsec2={b: PIX_ARCSEC**2 for b in imgs},
        spec_ivar=ones.copy(),
        spec_wdisp=ones.copy(),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=sigma,
        fiber_center_ra_dec=(180.0008, 0.5008),
        image_sigma_flambda={
            b: np.full_like(imgs[b], 0.05, dtype=np.float32) for b in imgs
        },
    )


def _bandpasses():
    from survey_fusion.data.bandpasses import Bandpass

    def bp(band, lo, hi):
        lam = np.linspace(lo, hi, 9)
        thr = np.exp(-0.5 * ((lam - lam.mean()) / ((hi - lo) / 5)) ** 2)
        return Bandpass(band=band, lam=lam, throughput=thr, canonical_id=f"test_{band}")

    return {"g": bp("g", 4200, 5300), "r": bp("r", 5500, 6800), "i": bp("i", 6900, 8100)}


def _make_trainer(
    analytic: bool,
    kernel_samples_k: int,
    *,
    quadrature_order: int = 1,
    amp_band_subsets: bool = True,
) -> DatumTrainer:
    device = torch.device("cpu")
    store = build_sdss_datums(
        _realistic_scene(),
        _bandpasses(),
        psf_fwhm_arcsec=1.0,
        device=device,
        spectral_response_mode="legacy_imaging_median",
    )
    torch.manual_seed(0)
    model = SplatField(
        n_splats=6, n_templates=3, loglam_grid=torch.linspace(-1, 1, 16),
        init_scale=0.2, init_extent=0.8,
    )
    with torch.no_grad():
        model.raw_flux.fill_(0.0)
    trainer = DatumTrainer(
        model=model,
        grid=build_spatial_grid(2, 3, device),
        data=store,
        training_bands=["g", "r", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": kernel_samples_k,
            "replica_samples_k": kernel_samples_k,
            "analytic_images": analytic,
            "analytic_image_quadrature_order": quadrature_order,
            "amp_band_subsets": amp_band_subsets,
            "epochs": 1,
            "learning_rate": 1e-3,
            "pixel_batch_size": 4,
            "spec_batch_size": 2,
        },
        device=device,
    )
    # size splats to ~3 detector pixels using the trainer's own geometry cache
    exp0 = int(
        trainer._gather_named_rows(
            "exposure_id", trainer.image_train_indices[:1],
            device=torch.device("cpu"), dtype=torch.long,
        )[0]
    )
    npp = trainer._analytic_cache_for_exposure(exp0)["norm_per_pixel"]
    with torch.no_grad():
        model.log_scales.fill_(float(np.log(3.0 * npp)))
    return trainer


def test_analytic_source_mu_matches_mc_within_tolerance():
    trainer = _make_trainer(analytic=True, kernel_samples_k=1024)
    image_indices = trainer.image_train_indices
    assert image_indices.numel() > 0
    exp_ids = trainer._gather_named_rows(
        "exposure_id", image_indices, device=torch.device("cpu"), dtype=torch.long
    )
    local = trainer._gather_named_rows(
        "datum_index", image_indices, device=torch.device("cpu"), dtype=torch.long
    )
    with torch.no_grad():
        mu_analytic = trainer._compute_analytic_image_source_mu(
            image_indices, exp_ids, local
        )
        mu_mc = trainer.compute_source_mu(image_indices)
    scale = mu_mc.abs().max().clamp_min(1e-12)
    err = (mu_analytic - mu_mc).abs().max() / scale
    assert err < 0.05, f"analytic vs MC source mu mismatch: {err:.4f}"


def test_analytic_terms_finite_and_backpropagable():
    trainer = _make_trainer(analytic=True, kernel_samples_k=8)
    image_indices = trainer.image_train_indices
    terms = trainer._compute_analytic_image_gaussian_terms(image_indices)
    assert terms.shape == (image_indices.numel(),)
    assert torch.isfinite(terms).all()
    terms.sum().backward()
    model = trainer._base_model()
    assert model.mu.grad is not None and torch.isfinite(model.mu.grad).all()


def test_analytic_pixel_quadrature_is_deterministic_and_backpropagable():
    trainer = _make_trainer(
        analytic=True, kernel_samples_k=8, quadrature_order=2
    )
    image_indices = trainer.image_train_indices
    exp_ids = trainer._gather_named_rows(
        "exposure_id", image_indices, device=torch.device("cpu"), dtype=torch.long
    )
    local = trainer._gather_named_rows(
        "datum_index", image_indices, device=torch.device("cpu"), dtype=torch.long
    )
    first = trainer._compute_analytic_image_source_mu(
        image_indices, exp_ids, local
    )
    second = trainer._compute_analytic_image_source_mu(
        image_indices, exp_ids, local
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    terms = trainer._compute_analytic_image_gaussian_terms(image_indices)
    assert torch.isfinite(terms).all()
    terms.sum().backward()
    model = trainer._base_model()
    assert model.mu.grad is not None and torch.isfinite(model.mu.grad).all()


def test_analytic_image_band_subsets_telemetry_and_opt_out_are_bitwise(
    monkeypatch,
) -> None:
    observed_band_rows = []
    real_splat_band_amplitudes = datum_trainer_module.splat_band_amplitudes

    def record_band_rows(*args, **kwargs):
        observed_band_rows.append(kwargs["band_row_indices"])
        return real_splat_band_amplitudes(*args, **kwargs)

    monkeypatch.setattr(
        datum_trainer_module,
        "splat_band_amplitudes",
        record_band_rows,
    )
    subset_trainer = _make_trainer(
        analytic=True,
        kernel_samples_k=8,
        amp_band_subsets=True,
    )
    full_trainer = _make_trainer(
        analytic=True,
        kernel_samples_k=8,
        amp_band_subsets=False,
    )
    image_indices = subset_trainer.image_train_indices[:12]
    exp_ids = subset_trainer._gather_named_rows(
        "exposure_id", image_indices, device=torch.device("cpu"), dtype=torch.long
    )
    local = subset_trainer._gather_named_rows(
        "datum_index", image_indices, device=torch.device("cpu"), dtype=torch.long
    )

    subset_mu = subset_trainer._compute_analytic_image_source_mu(
        image_indices, exp_ids, local
    )
    subset_rows = observed_band_rows[-1]
    assert subset_rows is not None
    assert subset_trainer._last_packed_image_stats["band_rows_total"] == sum(
        rows.numel() for rows in subset_rows
    )

    full_mu = full_trainer._compute_analytic_image_source_mu(
        image_indices, exp_ids, local
    )
    assert observed_band_rows[-1] is None
    assert "band_rows_total" not in full_trainer._last_packed_image_stats
    assert torch.equal(subset_mu, full_mu)

    subset_mu.square().sum().backward()
    full_mu.square().sum().backward()
    subset_field = subset_trainer._base_model()
    full_field = full_trainer._base_model()
    for name in ("sed_coeffs", "sed_shift", "raw_flux"):
        assert torch.equal(
            getattr(subset_field, name).grad,
            getattr(full_field, name).grad,
        )
    assert torch.equal(
        subset_field.sed.raw_templates.grad,
        full_field.sed.raw_templates.grad,
    )


def test_analytic_pixel_quadrature_order_must_be_positive():
    with pytest.raises(
        ValueError, match="analytic_image_quadrature_order must be positive"
    ):
        _make_trainer(
            analytic=True, kernel_samples_k=8, quadrature_order=0
        )


def test_deterministic_image_diagnostic_uses_analytic_splat_operator(monkeypatch):
    trainer = _make_trainer(
        analytic=True, kernel_samples_k=8, quadrature_order=2
    )
    exp_id = int(trainer.image_exposure_ids[0])
    height, width = 5, 7
    local = (
        torch.arange(height)[:, None] * SIZE + torch.arange(width)[None, :]
    ).reshape(-1)
    exp_ids = torch.full_like(local, exp_id)
    dummy_global = torch.empty(local.numel(), dtype=torch.long)
    with torch.no_grad():
        expected = trainer._compute_analytic_image_source_mu(
            dummy_global, exp_ids, local
        ).reshape(height, width)

    def fail_dense_path(*args, **kwargs):
        raise AssertionError("generic dense diagnostic renderer was used")

    monkeypatch.setattr(trainer, "_render_band_for_omega", fail_dense_path)
    profile = {}
    actual = trainer.render_datum_image_window_deterministic(
        exp_id,
        y0=0,
        y1=height,
        x0=0,
        x1=width,
        include_additive=False,
        profile=profile,
    )

    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), rtol=1e-6, atol=1e-6
    )
    assert profile["actual_mode"] == "analytic_splat"
    assert profile["quadrature_order"] == 2


def test_tiled_source_mu_matches_dense_reference():
    """The trainer's tiled path must reproduce the dense per-exposure
    evaluation it replaces; it exists only to make 10^5-10^7 splats
    affordable."""
    import numpy as np

    from survey_fusion.pgm.analytic_image import analytic_band_prediction

    trainer = _make_trainer(analytic=True, kernel_samples_k=8)
    idx = trainer.image_train_indices
    exp_ids = trainer._gather_named_rows(
        "exposure_id", idx, device=torch.device("cpu"), dtype=torch.long)
    local = trainer._gather_named_rows(
        "datum_index", idx, device=torch.device("cpu"), dtype=torch.long)
    with torch.no_grad():
        mu_tiled = trainer._compute_analytic_image_source_mu(idx, exp_ids, local)

    model = trainer._base_model()
    mu_dense = torch.empty_like(mu_tiled)
    with torch.no_grad():
        for exp_id in exp_ids.unique().tolist():
            sel = (exp_ids == exp_id).nonzero(as_tuple=True)[0]
            cache = trainer._analytic_cache_for_exposure(int(exp_id))
            exposure = trainer.runtime_by_exposure[int(exp_id)].exposure
            li = local[sel]
            px = torch.remainder(li, cache["width"]).to(torch.float64).unsqueeze(1).numpy()
            py = (
                torch.div(li, cache["width"], rounding_mode="floor")
                .to(torch.float64).unsqueeze(1).numpy()
            )
            omega = np.asarray(
                exposure.obs_wcs.pixel_to_omega_xyz(px, py), dtype=np.float64
            ).reshape(-1, 3)
            xn, yn = exposure.wcs.omega_xyz_to_norm_torch(
                torch.as_tensor(omega, dtype=torch.float32)
            )
            mu_dense[sel] = analytic_band_prediction(
                model, xn, yn, cache["sigma"], cache["ll"], cache["w"]
            ).to(torch.float32)

    scale = mu_dense.abs().max().clamp_min(1e-12)
    err = (mu_tiled - mu_dense).abs().max() / scale
    assert err < 0.02, f"tiled vs dense source mu mismatch: {err:.4f}"
