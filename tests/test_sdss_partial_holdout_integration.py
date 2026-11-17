from __future__ import annotations

import numpy as np
import torch

from survey_fusion.data.adapters.sdss import build_sdss_datums
from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.data.sdss_scene import SdssSceneData, SdssSpectrumData
from survey_fusion.models.spectral_field import GalaxySpectralField
from survey_fusion.training.datum_trainer import DatumTrainer


class DummySdssWcs:
    def all_pix2world(self, x, y, origin):
        del origin
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ra = 180.0 + x_arr * 1.0e-4
        dec = 0.5 + y_arr * 1.0e-4
        return ra, dec


def _bandpasses_with_i() -> dict[str, Bandpass]:
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([0.2, 1.0, 0.2], dtype=np.float64)
    return {
        "g": Bandpass(band="g", lam=lam, throughput=throughput, canonical_id="test_g"),
        "r": Bandpass(band="r", lam=lam + 150.0, throughput=throughput, canonical_id="test_r"),
        "i": Bandpass(band="i", lam=lam + 300.0, throughput=throughput, canonical_id="test_i"),
    }


def _synthetic_sdss_scene() -> SdssSceneData:
    g = np.array([[0.8, 1.2, 1.4], [1.0, 1.3, 1.5]], dtype=np.float32)
    r = np.array([[1.0, 1.4, 1.6], [1.2, 1.5, 1.8]], dtype=np.float32)
    i = np.array([[1.1, 1.5, 1.7], [1.25, 1.6, 1.9]], dtype=np.float32)
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    flux = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    sigma = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    wcs = DummySdssWcs()
    spectrum = SdssSpectrumData(
        label="fiber", role="science", path="synthetic.fits", lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(), flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((2, 3), dtype=np.float32), spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec), spec_loglam=np.log10(lam_spec), sigma_spec_flambda=sigma,
        center_ra_dec=(180.0, 0.5), fiber_radius_arcsec=1.5, metadata={},
    )
    return SdssSceneData(
        images={"g": g.copy(), "r": r.copy(), "i": i.copy()},
        images_flambda={"g": g, "r": r, "i": i},
        spectra=[spectrum],
        lam_spec=lam_spec,
        flux_spec_flambda=flux.copy(),
        flux_spec_flam_lsf=flux.copy(),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32),
        fiber_mask=np.ones((2, 3), dtype=np.float32),
        fiber_area_arcsec2=7.0,
        pixel_scale_arcsec=0.4,
        fiber_radius_arcsec=1.5,
        metadata={},
        wcs_per_band={"g": wcs, "r": wcs, "i": wcs},
        pixel_scale_arcsec_per_band={"g": 0.4, "r": 0.4, "i": 0.4},
        pixel_area_arcsec2={"g": 0.16, "r": 0.16, "i": 0.16},
        spec_ivar=np.ones_like(lam_spec),
        spec_wdisp=np.ones_like(lam_spec),
        spec_loglam=np.log10(lam_spec),
        sigma_spec_flambda=sigma,
        fiber_center_ra_dec=(180.0, 0.5),
        image_sigma_flambda={
            "g": np.full_like(g, 0.05, dtype=np.float32), "r": np.full_like(r, 0.05, dtype=np.float32),
            "i": np.full_like(i, 0.05, dtype=np.float32),
        },
    )


def test_sdss_partial_holdout_mc_fit_smoke() -> None:
    device = torch.device("cpu")
    scene = _synthetic_sdss_scene()
    bandpasses = _bandpasses_with_i()
    store = build_sdss_datums(
        scene,
        bandpasses,
        psf_fwhm_arcsec=1.0,
        device=device,
        holdout_band="i",
        holdout_region_cfg={
            "enabled": True, "center_radec_deg": [180.0002, 0.5000], "radius_arcsec": 0.5,
            "exclude_spectra_in_region": False,
        },
        spectral_response_mode="legacy_imaging_median",
    )
    model = GalaxySpectralField(hidden_dim=16, n_layers=2, softplus_beta=2.0)
    trainer = DatumTrainer(
        model=model,
        grid=build_spatial_grid(2, 3, device),
        data=store,
        training_bands=["g", "r", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 4, "epochs": 1, "max_steps": 1, "learning_rate": 1e-3, "beta1": 0.9,
            "beta2": 0.999, "apply_spatial_psf": False, "pixel_batch_size": 4, "spec_batch_size": 2,
            "dataloader_shuffle": False,
        },
        device=device,
    )
    state = trainer.fit()
    assert state.history["loss"]
    i_exp = next(exp_id for exp_id, band in store.image_band_by_exposure.items() if band == "i")
    excluded = set(store.image_excluded_local_indices_by_exposure[int(i_exp)].tolist())
    selected_i = trainer.image_train_indices[
        trainer.data.batch.exposure_id[trainer.image_train_indices] == int(i_exp)
    ]
    selected_i_local = set(
        torch.index_select(trainer.data.batch.datum_index, 0, selected_i).detach().cpu().tolist()
    )
    assert excluded.isdisjoint(selected_i_local)


def test_sdss_history_cycles_spectral_batches_when_image_loader_is_longer() -> None:
    device = torch.device("cpu")
    scene = _synthetic_sdss_scene()
    bandpasses = _bandpasses_with_i()
    store = build_sdss_datums(
        scene,
        bandpasses,
        psf_fwhm_arcsec=1.0,
        device=device,
        holdout_band="i",
        holdout_region_cfg={
            "enabled": True, "center_radec_deg": [180.0002, 0.5000], "radius_arcsec": 0.5,
            "exclude_spectra_in_region": False,
        },
        spectral_response_mode="legacy_imaging_median",
    )
    model = GalaxySpectralField(hidden_dim=16, n_layers=2, softplus_beta=2.0)
    trainer = DatumTrainer(
        model=model,
        grid=build_spatial_grid(2, 3, device),
        data=store,
        training_bands=["g", "r", "i"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 2, "epochs": 1, "learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999,
            "apply_spatial_psf": False, "pixel_batch_size": 1, "spec_batch_size": 64,
            "dataloader_shuffle": False,
        },
        device=device,
    )
    assert trainer.image_loader is not None
    assert trainer.spec_loader is not None
    assert len(trainer.image_loader) > len(trainer.spec_loader)

    state = trainer.fit()
    history = state.history
    assert "n_spec" in history
    assert len(history["n_spec"]) == len(history["loss"]) == len(history["spec_loss"])

    n_spec = np.asarray(history["n_spec"], dtype=np.int64)
    spec_loss = np.asarray(history["spec_loss"], dtype=np.float64)
    assert np.all(n_spec > 0)
    assert np.all(np.isfinite(spec_loss))
