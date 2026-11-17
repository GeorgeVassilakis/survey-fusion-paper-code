"""trainer.image_variance=poisson_current: CCD variance of the current model, gradient stopped."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.data.image_variance import (
    _column_calibration, image_variance_columns, image_variance_outlier_pull, poisson_current_requested,
    sigma_reconstruction_error,
)
from survey_fusion.data.sdss_scene import (
    FNU_PER_NANOMAGGY, SdssSceneData, SdssSpectrumData, _bandpass_fnu_to_flambda_factor, load_sdss_scene,
)
from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.likelihoods import current_model_image_sigma, datum_unbiased_gaussian_nll
from survey_fusion.training.datum_trainer import DatumTrainer

PULL = 10.0


def _shifted_poisson_nll(y, mu, alpha, beta):
    c = beta / alpha
    return (mu + c) / alpha - ((y + c) / alpha) * torch.log(mu + c)


def _random_pixels(n: int, *, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    alpha = torch.empty(n, dtype=torch.float64).uniform_(0.2, 3.0, generator=gen)
    beta = torch.empty(n, dtype=torch.float64).uniform_(0.5, 40.0, generator=gen)
    y = torch.empty(n, dtype=torch.float64).uniform_(20.0, 5000.0, generator=gen)
    sign = torch.where(torch.rand(n, generator=gen, dtype=torch.float64) < 0.5, -1.0, 1.0)
    sigma_data = torch.sqrt(alpha * y + beta)
    # 0.05 to 3 data sigmas from y: no cancellation in either gradient, no guard
    offset = sign * torch.empty(n, dtype=torch.float64).uniform_(0.05, 3.0, generator=gen) * sigma_data
    return y, y + offset, sigma_data, alpha, beta


def test_stopped_gradient_is_the_shifted_poisson_gradient() -> None:
    y, mu0, sigma_data, alpha, beta = _random_pixels(4096)
    mu = mu0.clone().requires_grad_(True)
    sigma = current_model_image_sigma(y, mu, sigma_data, alpha, beta, PULL)
    assert not sigma.requires_grad
    datum_unbiased_gaussian_nll(y, mu, mu, sigma).sum().backward()
    objective_grad = mu.grad.clone()

    mu.grad = None
    _shifted_poisson_nll(y, mu, alpha, beta).sum().backward()
    torch.testing.assert_close(objective_grad, mu.grad, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(objective_grad, (mu0 - y) / (alpha * mu0 + beta), rtol=1e-12, atol=0.0)

    mu.grad = None
    live = torch.sqrt(alpha * mu + beta)
    datum_unbiased_gaussian_nll(y, mu, mu, live).sum().backward()
    assert not torch.allclose(mu.grad, objective_grad, rtol=1e-3, atol=0.0)


def test_guarded_pixels_keep_the_data_sigma_and_its_gradient() -> None:
    y, mu0, sigma_data, alpha, beta = _random_pixels(512, seed=1)
    mu0 = mu0.clone()
    mu0[::2] = y[::2] - 11.0 * sigma_data[::2]
    mu = mu0.clone().requires_grad_(True)
    sigma = current_model_image_sigma(y, mu, sigma_data, alpha, beta, PULL)
    torch.testing.assert_close(sigma[::2], sigma_data[::2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(sigma[1::2], torch.sqrt(alpha * mu0 + beta)[1::2], rtol=0.0, atol=0.0)
    datum_unbiased_gaussian_nll(y, mu, mu, sigma).sum().backward()
    torch.testing.assert_close(mu.grad[::2], ((mu0 - y) / sigma_data**2)[::2], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_guard_switches_exactly_at_the_pull(dtype: torch.dtype) -> None:
    sigma_data = torch.tensor(0.5, dtype=dtype)
    mu = torch.tensor(100.0, dtype=dtype)
    edge_above = torch.tensor(105.0, dtype=dtype)
    edge_below = torch.tensor(95.0, dtype=dtype)
    y = torch.stack([
        edge_above, torch.nextafter(edge_above, torch.tensor(np.inf, dtype=dtype)),
        edge_below, torch.nextafter(edge_below, torch.tensor(-np.inf, dtype=dtype)),
    ])
    ones = torch.ones_like(y)
    sigma = current_model_image_sigma(y, mu.expand(4), sigma_data.expand(4), ones, ones, PULL)
    model_sigma = float(torch.sqrt(torch.tensor(101.0, dtype=dtype)))
    assert sigma.tolist() == pytest.approx([model_sigma, 0.5, model_sigma, 0.5], rel=0, abs=0)


def test_negative_model_takes_the_read_noise_floor() -> None:
    sigma = current_model_image_sigma(
        torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([4.0]), PULL,
    )
    assert sigma.tolist() == [2.0]


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"image_variance": "model"}, "must be one of"),
        ({"image_variance_outlier_pull": 10.0}, "applies only"),
        ({"image_variance": "data", "image_variance_outlier_pull": 10.0}, "applies only"),
        ({"image_variance": "poisson_current", "analytic_images": True}, "requires trainer.image_variance_outlier"),
        ({"image_variance": "poisson_current", "analytic_images": True, "image_variance_outlier_pull": 0.0},
         "finite and positive"),
        ({"image_variance": "poisson_current", "analytic_images": True, "image_variance_outlier_pull": -1.0},
         "finite and positive"),
        ({"image_variance": "poisson_current", "analytic_images": True, "image_variance_outlier_pull": float("nan")},
         "finite and positive"),
        ({"image_variance": "poisson_current", "analytic_images": True, "image_variance_outlier_pull": float("inf")},
         "finite and positive"),
        ({"image_variance": "poisson_current", "analytic_images": True, "image_variance_outlier_pull": True},
         "requires trainer.image_variance_outlier"),
        ({"image_variance": "poisson_current", "image_variance_outlier_pull": 10.0}, "analytic_images=true"),
    ],
)
def test_invalid_image_variance_config_is_refused(cfg: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        image_variance_outlier_pull(cfg)


def test_image_variance_config_accepts_the_two_modes() -> None:
    assert image_variance_outlier_pull({}) is None
    assert image_variance_outlier_pull({"image_variance": "data"}) is None
    cfg = {"image_variance": "poisson_current", "image_variance_outlier_pull": 10, "analytic_images": True}
    assert image_variance_outlier_pull(cfg) == 10.0
    assert poisson_current_requested(cfg, {}) is True
    assert poisson_current_requested({}, {"image_sigma_model_path": "/x.json"}) is False
    with pytest.raises(ValueError, match="image_sigma_model_path"):
        poisson_current_requested(cfg, {"image_sigma_model_path": "/x.json", "image_sigma_model_outlier_pull": 10.0})


# --- loader reproduction on a cutout that is a column sub-array of its frame ---

FRAME_W, FRAME_H, CUT_W, C0 = 10, 6, 6, 3
GAIN, DARK_VARIANCE = 4.5, 1.7
CALIB = np.linspace(0.004, 0.0065, FRAME_W) * (1.0 + 0.03 * np.sin(np.arange(FRAME_W)))


def _wcs_header(crpix1: float, *, rotation_deg: float = 0.0) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 180.0, 0.0
    header["CRPIX1"], header["CRPIX2"] = crpix1, 1.0
    scale = 0.396 / 3600.0
    theta = np.deg2rad(rotation_deg)
    header["CD1_1"], header["CD1_2"] = -scale * np.cos(theta), scale * np.sin(theta)
    header["CD2_1"], header["CD2_2"] = scale * np.sin(theta), scale * np.cos(theta)
    header["BUNIT"] = "nanomaggy"
    return header


def _write_frame(path: Path) -> None:
    allsky = np.linspace(80.0, 120.0, FRAME_W * FRAME_H, dtype=np.float32).reshape(FRAME_H, FRAME_W)
    sky = fits.BinTableHDU.from_columns([
        fits.Column(name="ALLSKY", format=f"{allsky.size}E", dim=f"({FRAME_W},{FRAME_H})", array=[allsky]),
        fits.Column(name="XINTERP", format=f"{FRAME_W}E", array=[np.arange(FRAME_W, dtype=np.float32)]),
        fits.Column(name="YINTERP", format=f"{FRAME_H}E", array=[np.arange(FRAME_H, dtype=np.float32)]),
    ])
    fits.HDUList([
        fits.PrimaryHDU(np.zeros((FRAME_H, FRAME_W), np.float32), header=_wcs_header(1.0 + C0)),
        fits.ImageHDU(CALIB.astype(np.float32)), sky,
    ]).writeto(path, overwrite=True)


def _write_cutout(path: Path, *, rotation_deg: float = 0.0) -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 0.4, (FRAME_H, CUT_W)).astype(np.float32)
    data[2, 3] = 40.0
    data[4, 1] = -5.0
    fits.PrimaryHDU(data, header=_wcs_header(1.0, rotation_deg=rotation_deg)).writeto(path, overwrite=True)


def _write_spectrum(path: Path) -> None:
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loglam = np.log10(np.array([5000.0, 5100.0, 5200.0])).astype(np.float32)
    coadd = fits.BinTableHDU.from_columns([
        fits.Column(name="flux", format="E", array=flux),
        fits.Column(name="loglam", format="E", array=loglam),
        fits.Column(name="ivar", format="E", array=np.ones(3, np.float32)),
        fits.Column(name="and_mask", format="J", array=np.zeros(3, np.int32)),
        fits.Column(name="or_mask", format="J", array=np.zeros(3, np.int32)),
        fits.Column(name="wdisp", format="E", array=np.ones(3, np.float32)),
        fits.Column(name="sky", format="E", array=np.full(3, 0.5, np.float32)),
    ], name="COADD")
    specobj = fits.BinTableHDU.from_columns([
        fits.Column(name="PLUG_RA", format="D", array=np.array([180.0])),
        fits.Column(name="PLUG_DEC", format="D", array=np.array([0.0])),
    ], name="SPECOBJ")
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"BUNIT": "1E-17 erg/cm^2/s/Ang"})), coadd, specobj]).writeto(
        path, overwrite=True
    )


def _bandpasses() -> dict[str, Bandpass]:
    lam = np.array([5000.0, 5500.0, 6000.0])
    return {"r": Bandpass(band="r", lam=lam, throughput=np.ones(3), canonical_id="test_r")}


def _loaded(tmp_path: Path):
    cutout, frame, spectrum = tmp_path / "c.fits", tmp_path / "f.fits", tmp_path / "s.fits"
    _write_cutout(cutout)
    _write_frame(frame)
    _write_spectrum(spectrum)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "image_exposures": [{
            "label": "r-exp1", "band": "r", "cutout_path": str(cutout), "frame_path": str(frame),
            "gain": GAIN, "dark_variance": DARK_VARIANCE, "psf_fwhm_arcsec": 1.2,
        }],
        "spectra": [{"label": "S1", "role": "science", "path": str(spectrum), "ra_deg": 180.0, "dec_deg": 0.0}],
    }))
    cfg = {
        "scene_manifest_path": str(manifest), "training_flux_scale": 1.0e17, "fiber_diameter_arcsec": 3.0,
        "reference_band": "r", "spec_min_ivar": 0.0, "spec_require_clean_mask": True,
    }
    scene = load_sdss_scene(cfg, _bandpasses())
    store = build_sdss_datums(
        scene, _bandpasses(), psf_fwhm_arcsec=1.0, device=torch.device("cpu"),
        spectral_response_mode="legacy_imaging_median",
    )
    return scene, store, cutout, frame


def test_coefficients_reproduce_the_loader_sigma(tmp_path: Path) -> None:
    scene, store, _, _ = _loaded(tmp_path)
    columns = image_variance_columns(scene, store)
    assert list(columns) == [0]
    alpha, beta = columns[0]
    exposure = scene.image_exposures[0]
    u = FNU_PER_NANOMAGGY * _bandpass_fnu_to_flambda_factor(_bandpasses()["r"]) / exposure.pixel_area_arcsec2 * 1e17
    calib = CALIB.astype(np.float32).astype(np.float64)[C0:C0 + CUT_W]
    # calib is interpolated at the WCS round trip of each column, as the loader does
    np.testing.assert_allclose(alpha, u * calib / GAIN, rtol=1e-9)
    np.testing.assert_allclose(beta, (u * calib) ** 2 * DARK_VARIANCE, rtol=1e-9)

    total = np.asarray(exposure.total_flambda, dtype=np.float64)
    assert (total < 0).any()
    rebuilt = np.sqrt(alpha[None, :] * np.maximum(total, 0.0) + beta[None, :])
    np.testing.assert_allclose(rebuilt, exposure.sigma_flambda, rtol=2e-7)

    rows = store.batch.datum_kind == 0
    error = sigma_reconstruction_error(
        store.batch.y[rows].numpy(), store.batch.sigma[rows].numpy(), store.batch.datum_index[rows].numpy(),
        CUT_W, alpha, beta,
    )
    assert error < 2e-7
    shifted = sigma_reconstruction_error(
        store.batch.y[rows].numpy(), store.batch.sigma[rows].numpy(), store.batch.datum_index[rows].numpy() + 1,
        CUT_W, alpha, beta,
    )
    assert shifted > 1e-3


def test_rotated_cutout_is_refused(tmp_path: Path) -> None:
    frame, cutout = tmp_path / "f.fits", tmp_path / "c.fits"
    _write_frame(frame)
    _write_cutout(cutout, rotation_deg=1.0)
    with pytest.raises(ValueError, match="independently of row"):
        _column_calibration(fits.getheader(cutout, 0), str(frame))


# --- trainer integration ---

SIZE = 16
PIX_ARCSEC = 0.4
SKY = 0.8


class _LinearWcs:
    def all_pix2world(self, x, y, origin):
        del origin
        step = PIX_ARCSEC / 3600.0
        return 180.0 + np.asarray(x, np.float64) * step, 0.5 + np.asarray(y, np.float64) * step


def _band_columns(band_index: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(SIZE, dtype=np.float64)
    return 0.002 * (1.0 + 0.1 * band_index) * (1.0 + 0.02 * cols), np.full(SIZE, 0.04**2) * (1.0 + 0.01 * cols)


def _synthetic_store():
    rng = np.random.default_rng(11)
    bands = "gri"
    imgs = {b: (SKY + rng.uniform(0.0, 0.7, (SIZE, SIZE))).astype(np.float32) for b in bands}
    sigma = {}
    for k, b in enumerate(bands):
        alpha, beta = _band_columns(k)
        sigma[b] = np.sqrt(alpha[None, :] * imgs[b].astype(np.float64) + beta[None, :]).astype(np.float32)
    lam_spec = np.array([5000.0, 5100.0, 5200.0])
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    spec_sigma = np.full(3, 0.05, dtype=np.float32)
    spectrum = SdssSpectrumData(
        label="fiber", role="science", path="synthetic.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(), spec_throughput=np.ones(3, np.float32),
        fiber_mask=np.ones((SIZE, SIZE), np.float32), spec_ivar=np.ones(3), spec_wdisp=np.ones(3),
        spec_loglam=np.log10(lam_spec), sigma_spec_flambda=spec_sigma, center_ra_dec=(180.0008, 0.5008),
        fiber_radius_arcsec=1.5, metadata={},
    )
    scene = SdssSceneData(
        images={b: imgs[b].copy() for b in bands}, images_flambda=imgs, spectra=[spectrum], lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(), spec_throughput=np.ones(3, np.float32),
        fiber_mask=np.ones((SIZE, SIZE), np.float32), fiber_area_arcsec2=7.0, pixel_scale_arcsec=PIX_ARCSEC,
        fiber_radius_arcsec=1.5, metadata={}, wcs_per_band={b: _LinearWcs() for b in bands},
        pixel_scale_arcsec_per_band={b: PIX_ARCSEC for b in bands},
        pixel_area_arcsec2={b: PIX_ARCSEC**2 for b in bands}, spec_ivar=np.ones(3), spec_wdisp=np.ones(3),
        spec_loglam=np.log10(lam_spec), sigma_spec_flambda=spec_sigma, fiber_center_ra_dec=(180.0008, 0.5008),
        image_sigma_flambda=sigma,
        image_additive_flambda={b: np.full((SIZE, SIZE), SKY, np.float32) for b in bands},
    )
    def bp(band, lo, hi):
        lam = np.linspace(lo, hi, 9)
        return Bandpass(band=band, lam=lam, throughput=np.exp(-0.5 * ((lam - lam.mean()) / ((hi - lo) / 5)) ** 2),
                        canonical_id=f"test_{band}")
    bandpasses = {"g": bp("g", 4200, 5300), "r": bp("r", 5500, 6800), "i": bp("i", 6900, 8100)}
    store = build_sdss_datums(
        scene, bandpasses, psf_fwhm_arcsec=1.0, device=torch.device("cpu"),
        spectral_response_mode="legacy_imaging_median",
    )
    columns = {
        int(exp_id): _band_columns(bands.index(band)) for exp_id, band in store.image_band_by_exposure.items()
    }
    return store, columns


def _trainer(store, *, columns=None, extra=None) -> DatumTrainer:
    torch.manual_seed(0)
    model = SplatField(n_splats=6, n_templates=3, loglam_grid=torch.linspace(-1, 1, 16), init_scale=0.2,
                       init_extent=0.8)
    cfg = {
        "kernel_samples_k": 4, "replica_samples_k": 4, "analytic_images": True, "epochs": 1,
        "learning_rate": 1e-3, "pixel_batch_size": 4, "spec_batch_size": 2,
    }
    cfg.update(extra or {})
    trainer = DatumTrainer(
        model=model, grid=build_spatial_grid(2, 3, torch.device("cpu")), data=store, training_bands=["g", "r", "i"],
        held_out_band=None, lambda_step=20.0, trainer_cfg=cfg, device=torch.device("cpu"),
        image_variance_columns=columns,
    )
    exp0 = int(store.batch.exposure_id[int(trainer.image_train_indices[0])])
    npp = trainer._analytic_cache_for_exposure(exp0)["norm_per_pixel"]
    with torch.no_grad():
        model.log_scales.fill_(float(np.log(3.0 * npp)))
        model.raw_flux.fill_(-2.0)
    return trainer


POISSON = {"image_variance": "poisson_current", "image_variance_outlier_pull": PULL}


def _rows(trainer: DatumTrainer, columns, *, exposure: int | None = None):
    idx = trainer.image_train_indices
    if exposure is not None:
        idx = idx[trainer.data.batch.exposure_id[idx] == exposure]
    cpu = torch.device("cpu")
    exp_ids = trainer._gather_named_rows("exposure_id", idx, device=cpu, dtype=torch.long)
    local = trainer._gather_named_rows("datum_index", idx, device=cpu, dtype=torch.long)
    y = trainer._gather_named_rows("y", idx, device=cpu, dtype=torch.float32)
    sigma = trainer._gather_named_rows("sigma", idx, device=cpu, dtype=torch.float32)
    alpha = torch.tensor([columns[int(e)][0][int(i) % SIZE] for e, i in zip(exp_ids, local)], dtype=torch.float32)
    beta = torch.tensor([columns[int(e)][1][int(i) % SIZE] for e, i in zip(exp_ids, local)], dtype=torch.float32)
    mu = trainer._apply_observation_contract(idx, trainer._compute_analytic_image_source_mu(idx, exp_ids, local))
    return idx, y, sigma, alpha, beta, mu


def test_default_objective_keeps_the_store_sigma() -> None:
    store, columns = _synthetic_store()
    trainer = _trainer(store)
    assert trainer.image_variance_outlier_pull is None
    idx, y, sigma, _, _, mu = _rows(trainer, columns)
    torch.testing.assert_close(
        trainer._compute_analytic_image_gaussian_terms(idx), datum_unbiased_gaussian_nll(y, mu, mu, sigma),
        rtol=0.0, atol=0.0,
    )


def test_poisson_objective_uses_the_current_model_variance_and_poisson_gradient() -> None:
    store, columns = _synthetic_store()
    trainer = _trainer(store, columns=columns, extra=POISSON)
    idx, y, sigma, alpha, beta, mu = _rows(trainer, columns)
    terms = trainer._compute_analytic_image_gaussian_terms(idx)
    expected_sigma = current_model_image_sigma(y, mu, sigma, alpha, beta, PULL)
    torch.testing.assert_close(terms, datum_unbiased_gaussian_nll(y, mu, mu, expected_sigma), rtol=0.0, atol=0.0)
    guard = (y - mu.detach()).abs() > PULL * sigma
    assert 0 < int(guard.sum()) < int(guard.numel())
    assert not torch.allclose(terms, datum_unbiased_gaussian_nll(y, mu, mu, sigma))

    model = trainer._base_model()
    params = [model.raw_flux, model.mu]
    objective_grads = torch.autograd.grad(terms.sum(), params)
    _, _, _, _, _, mu = _rows(trainer, columns)
    poisson = torch.where(
        guard, 0.5 * (mu - y) ** 2 / sigma**2,
        _shifted_poisson_nll(y.double(), mu.double(), alpha.double(), beta.double()).float(),
    )
    poisson_grads = torch.autograd.grad(poisson.sum(), params)
    _, _, _, _, _, mu = _rows(trainer, columns)
    data_grads = torch.autograd.grad(datum_unbiased_gaussian_nll(y, mu, mu, sigma).sum(), params)
    for got, want, data in zip(objective_grads, poisson_grads, data_grads):
        assert float((got - want).abs().max() / want.abs().max()) < 1e-5
        assert float((data - want).abs().max() / want.abs().max()) > 1e-2


def test_poisson_objective_leaves_birth_residuals_on_the_store_sigma() -> None:
    store, columns = _synthetic_store()
    trainer = _trainer(store, columns=columns, extra={**POISSON, "refine": {"birth": {"enabled": True}}})
    idx, y, sigma, _, _, mu = _rows(trainer, columns, exposure=min(columns))
    trainer._compute_analytic_image_gaussian_terms(idx)
    torch.testing.assert_close(trainer._pending_birth["r"], (y - mu.detach()) / sigma, rtol=0.0, atol=0.0)


def test_poisson_trainer_requires_matching_coefficients() -> None:
    store, columns = _synthetic_store()
    with pytest.raises(ValueError, match="require trainer.image_variance=poisson_current"):
        _trainer(store, columns=columns)
    with pytest.raises(ValueError, match="requires the per-column variance coefficients"):
        _trainer(store, extra=POISSON)
    with pytest.raises(ValueError, match="cover exactly"):
        _trainer(store, columns={k: v for k, v in columns.items() if k != 0}, extra=POISSON)
    with pytest.raises(ValueError, match="columns; its variance coefficients"):
        _trainer(store, columns={k: (a[:-1], b[:-1]) for k, (a, b) in columns.items()}, extra=POISSON)
    with pytest.raises(ValueError, match="finite and positive"):
        _trainer(store, columns={k: (a, 0.0 * b) for k, (a, b) in columns.items()}, extra=POISSON)
    with pytest.raises(ValueError, match="do not reproduce the stored image sigma"):
        _trainer(store, columns={k: (a, 1.001 * b) for k, (a, b) in columns.items()}, extra=POISSON)
    with pytest.raises(ValueError, match="do not reproduce the stored image sigma"):
        _trainer(store, columns={k: (a[::-1].copy(), b[::-1].copy()) for k, (a, b) in columns.items()},
                 extra=POISSON)


def _components_cfg(trainer: dict, data: dict | None = None):
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "data": {"name": "sdss", **(data or {})},
        "trainer": {
            "kernel_samples_k": 4, "replica_samples_k": 4, "analytic_images": True, "epochs": 1,
            "learning_rate": 1e-3, "pixel_batch_size": 4, "spec_batch_size": 2, **trainer,
        },
        "physics": {"bandpasses": {"r": "test_r"}, "held_out_band": None, "training_bands": ["r"],
                    "lambda_step_angstrom": 20.0},
        "calibration": {}, "resume": {"path": None},
        "model": {"name": "splat", "n_splats": 4, "n_templates": 2, "n_loglam_grid": 4, "seed_from_peaks": False},
    })


def test_build_components_wires_coefficients_from_the_prepared_scene(tmp_path: Path, monkeypatch) -> None:
    import train

    scene, store, _, _ = _loaded(tmp_path)
    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: _bandpasses())
    monkeypatch.setattr(train, "prepare_scene_and_datums", lambda *_args, **_kwargs: (scene, store))
    paths = {"registry_path": tmp_path / "registry.csv", "project_root": tmp_path}
    built = train._build_trainer_components(
        _components_cfg(POISSON), paths=paths, device=torch.device("cpu"), precision_policy="portable_fp32",
    )
    trainer = built[4]
    assert trainer.image_variance_outlier_pull == PULL
    alpha, beta = image_variance_columns(scene, store)[0]
    torch.testing.assert_close(trainer._image_variance_alpha, torch.as_tensor(alpha, dtype=torch.float32))
    torch.testing.assert_close(trainer._image_variance_beta, torch.as_tensor(beta, dtype=torch.float32))
    default = train._build_trainer_components(
        _components_cfg({}), paths=paths, device=torch.device("cpu"), precision_policy="portable_fp32",
    )[4]
    assert default.image_variance_outlier_pull is None and not hasattr(default, "_image_variance_alpha")


def test_build_components_refuses_poisson_over_a_fixed_render_store(tmp_path: Path, monkeypatch) -> None:
    import train

    monkeypatch.setattr(train, "load_bandpasses", lambda *_args, **_kwargs: _bandpasses())
    monkeypatch.setattr(
        train, "prepare_scene_and_datums", lambda *_args, **_kwargs: pytest.fail("store loaded before validation"),
    )
    cfg = _components_cfg(POISSON, {"image_sigma_model_path": "/m.json", "image_sigma_model_outlier_pull": 10.0})
    with pytest.raises(ValueError, match="image_sigma_model_path"):
        train._build_trainer_components(
            cfg, paths={"registry_path": tmp_path / "r.csv", "project_root": tmp_path}, device=torch.device("cpu"),
            precision_policy="portable_fp32",
        )
