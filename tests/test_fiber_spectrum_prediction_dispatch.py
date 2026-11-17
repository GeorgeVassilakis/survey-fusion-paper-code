from __future__ import annotations

import numpy as np
import pytest
import torch

from support.synthetic_trainer import ConstantKernelFactory, DummyWCS, SyntheticStore as MockStore, bandpass
from survey_fusion.data.grids import build_fiber_mask, build_spatial_grid
from survey_fusion.pgm.contracts import(DatumBatch, DatumKind, ExposureState, MaskApertureSpec, SkyDiskApertureSpec)
from survey_fusion.pgm.kernel_builders import KernelFactoryFromBuilders
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.training.renderer import SpectralRenderer
from survey_fusion.utils.wcs import TangentPlaneWCS


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        return torch.ones_like(x_norm, dtype=torch.float32, device=self.bias.device) + 0.0 * self.bias


class CurvedNormModel(torch.nn.Module):
    def __init__(self, coeff: float = 3.0):
        super().__init__()
        self.coeff = float(coeff)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del log_lam
        field = 1.0 + self.coeff * (x_norm**2 + 0.5 * y_norm**2)
        return field.to(dtype=torch.float32) + 0.0 * self.bias


class ChromaticCurvedNormModel(torch.nn.Module):
    def __init__(self, spatial_coeff: float = 6.0):
        super().__init__()
        self.spatial_coeff = float(spatial_coeff)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        lam_phase = 0.5 * (log_lam + 1.0)
        spatial = 1.0 + self.spatial_coeff * (x_norm**2 + 0.35 * y_norm**2)
        chromatic = 1.2 + 0.55 * lam_phase + 0.18 * torch.sin(2.0 * torch.pi * lam_phase)
        return (spatial * chromatic).to(dtype=torch.float32) + 0.0 * self.bias


def _make_store(device: torch.device, *, aperture: object) -> MockStore:
    wcs = DummyWCS()
    image_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=wcs,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=None,
        bandpasses={"r": bandpass("r")},
        lam_spec=np.array([5000.0, 5100.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32),
        pixel_scale_arcsec=1.0,
        image_height=2,
        image_width=2,
        lam_edges=None,
        aperture=None,
    )
    spec_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=wcs,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=None,
        bandpasses={},
        lam_spec=np.array([5000.0, 5100.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32),
        pixel_scale_arcsec=1.0,
        image_height=None,
        image_width=None,
        lam_edges=np.array([[4950.0, 5050.0], [5050.0, 5150.0]], dtype=np.float64),
        aperture=aperture,
    )
    batch = DatumBatch(
        y=torch.zeros(6, dtype=torch.float32, device=device),
        sigma=torch.ones(6, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long, device=device),
        datum_kind=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long, device=device),
        datum_index=torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long, device=device),
        kernel_samples=None,
    )
    return MockStore(
        exposures={0: image_exposure, 1: spec_exposure},
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=4,
        n_spec=2,
    )


def _make_trainer(*, aperture: object) -> DatumTrainer:
    device = torch.device("cpu")
    return DatumTrainer(
        model=ConstantModel(),
        grid=build_spatial_grid(2, 2, device),
        data=_make_store(device, aperture=aperture),
        training_bands=["r"],
        held_out_band="r",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 8,
            "replica_samples_k": 1,
            "epochs": 0,
            "learning_rate": 1e-3,
            "apply_spatial_psf": False,
        },
        device=device,
    )


def _mc_mean_spectrum(trainer: DatumTrainer, *, exp_id: int, repeats: int) -> np.ndarray:
    idx = trainer._global_indices_for(exp_id, DatumKind.SPECTRAL_BIN)
    lam = trainer.runtime_by_exposure[int(exp_id)].exposure.lam_spec
    local = torch.index_select(trainer.data.batch.datum_index, 0, idx).to(torch.long)
    if idx.numel() == 0:
        raise ValueError("Expected spectral datums for MC agreement test.")
    samples = []
    with torch.no_grad():
        for _ in range(repeats):
            mu = trainer.compute_mu(
                idx,
                apply_spectral_nuisance=bool(trainer.spectral_calibration_cfg.apply_to_predictions),
            )
            spec = torch.empty((int(lam.size),), dtype=mu.dtype, device=trainer.device)
            spec[local] = mu
            samples.append(spec.detach().cpu().numpy())
    return np.mean(np.stack(samples, axis=0), axis=0)


def _make_sky_disk_mc_trainer() -> DatumTrainer:
    device = torch.device("cpu")
    width = 5
    height = 5
    pixel_scale_arcsec = 1.0
    fiber_radius_arcsec = 0.9
    wcs = TangentPlaneWCS(
        ra0_rad=float(np.pi),
        dec0_rad=0.0,
        pixel_scale_arcsec=pixel_scale_arcsec,
        width=width,
        height=height,
    )
    fiber_mask = build_fiber_mask(
        height,
        width,
        pixel_scale_arcsec,
        2.0 * fiber_radius_arcsec,
    ).to_dense().astype(np.float32)
    image_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=wcs,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=None,
        bandpasses={"r": bandpass("r")},
        lam_spec=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        fiber_mask=fiber_mask,
        pixel_scale_arcsec=pixel_scale_arcsec,
        image_height=height,
        image_width=width,
        lam_edges=None,
        aperture=None,
    )
    spec_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=None,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=None,
        bandpasses={},
        lam_spec=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        fiber_mask=fiber_mask,
        pixel_scale_arcsec=pixel_scale_arcsec,
        image_height=None,
        image_width=None,
        lam_edges=np.array(
            [[4950.0, 5050.0], [5050.0, 5150.0], [5150.0, 5250.0]], dtype=np.float64
        ),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=fiber_radius_arcsec,),
    )
    batch = DatumBatch(
        y=torch.zeros(height * width + 3, dtype=torch.float32, device=device),
        sigma=torch.ones(height * width + 3, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0] * (height * width) + [1, 1, 1], dtype=torch.long, device=device),
        datum_kind=torch.tensor([0] * (height * width) + [1, 1, 1], dtype=torch.long, device=device),
        datum_index=torch.tensor(list(range(height * width)) + [0, 1, 2], dtype=torch.long, device=device),
        kernel_samples=None,
    )
    exposures = {0: image_exposure, 1: spec_exposure}
    store = MockStore(
        exposures=exposures,
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=KernelFactoryFromBuilders(
            batch=batch,
            exposures=exposures,
            image_band_by_exposure={0: "r"},
            rng_seed=9,
        ),
        n_image=height * width,
        n_spec=3,
    )
    return DatumTrainer(
        model=CurvedNormModel(coeff=8.0),
        grid=build_spatial_grid(height, width, device),
        data=store,
        training_bands=["r"],
        held_out_band="r",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 64,
            "replica_samples_k": 1,
            "epochs": 0,
            "learning_rate": 1e-3,
            "apply_spatial_psf": False,
        },
        device=device,
    )


def _make_irregular_sky_disk_mc_trainer() -> DatumTrainer:
    device = torch.device("cpu")
    width = 7
    height = 7
    pixel_scale_arcsec = 0.7
    fiber_radius_arcsec = 1.05
    lam_spec = np.array([5005.0, 5078.0, 5198.0, 5415.0], dtype=np.float64)
    lam_edges = np.array(
        [[4970.0, 5036.0], [5036.0, 5132.0], [5132.0, 5288.0], [5288.0, 5542.0]],
        dtype=np.float64,
    )
    lsf_fwhm = np.array([24.0, 42.0, 67.0, 93.0], dtype=np.float64)
    wcs = TangentPlaneWCS(
        ra0_rad=float(np.pi),
        dec0_rad=0.0,
        pixel_scale_arcsec=pixel_scale_arcsec,
        width=width,
        height=height,
    )
    fiber_mask = build_fiber_mask(
        height,
        width,
        pixel_scale_arcsec,
        2.0 * fiber_radius_arcsec,
    ).to_dense().astype(np.float32)
    image_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=wcs,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=None,
        bandpasses={"r": bandpass("r")},
        lam_spec=lam_spec,
        fiber_mask=fiber_mask,
        pixel_scale_arcsec=pixel_scale_arcsec,
        image_height=height,
        image_width=width,
        lam_edges=None,
        aperture=None,
    )
    spec_exposure = ExposureState(
        wcs=wcs,
        obs_wcs=None,
        psf_fwhm_arcsec=None,
        lsf_fwhm_angstrom=lsf_fwhm,
        bandpasses={},
        lam_spec=lam_spec,
        fiber_mask=fiber_mask,
        pixel_scale_arcsec=pixel_scale_arcsec,
        image_height=None,
        image_width=None,
        lam_edges=lam_edges,
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=fiber_radius_arcsec,),
    )
    batch = DatumBatch(
        y=torch.zeros(height * width + lam_spec.size, dtype=torch.float32, device=device),
        sigma=torch.ones(height * width + lam_spec.size, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(
            [0] * (height * width) + [1] * int(lam_spec.size),
            dtype=torch.long,
            device=device,
        ),
        datum_kind=torch.tensor(
            [0] * (height * width) + [1] * int(lam_spec.size),
            dtype=torch.long,
            device=device,
        ),
        datum_index=torch.tensor(
            list(range(height * width)) + list(range(int(lam_spec.size))),
            dtype=torch.long,
            device=device,
        ),
        kernel_samples=None,
    )
    exposures = {0: image_exposure, 1: spec_exposure}
    store = MockStore(
        exposures=exposures,
        batch=batch,
        image_band_by_exposure={0: "r"},
        kernel_factory=KernelFactoryFromBuilders(
            batch=batch,
            exposures=exposures,
            image_band_by_exposure={0: "r"},
            rng_seed=17,
        ),
        n_image=height * width,
        n_spec=int(lam_spec.size),
    )
    return DatumTrainer(
        model=ChromaticCurvedNormModel(spatial_coeff=7.5),
        grid=build_spatial_grid(height, width, device),
        data=store,
        training_bands=["r"],
        held_out_band="r",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 96,
            "replica_samples_k": 1,
            "epochs": 0,
            "learning_rate": 1e-3,
            "apply_spatial_psf": False,
        },
        device=device,
    )


def test_predict_fiber_spectra_uses_deterministic_aperture_quadrature_for_sky_disk_apertures() -> None:
    trainer = _make_trainer(aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.5))
    calls: list[tuple[str, int]] = []

    trainer._predict_fiber_spectrum_from_field_sum = (  # type: ignore[method-assign]
        lambda exp_id, include_additive=True: (
            calls.append(("field", int(exp_id))),
            (np.array([5000.0, 5100.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float32)),
        )[1]
    )
    trainer._predict_fiber_spectrum_from_aperture_quadrature = (  # type: ignore[method-assign]
        lambda exp_id, include_additive=True: (
            calls.append(("quad", int(exp_id))),
            (np.array([5000.0, 5100.0], dtype=np.float64), np.array([2.0, 2.0], dtype=np.float32)),
        )[1]
    )

    spectra = trainer.predict_fiber_spectra()

    assert calls == [("quad", 1)]
    np.testing.assert_allclose(spectra[1][1], np.array([2.0, 2.0], dtype=np.float32))


def test_predict_fiber_spectra_uses_field_sum_for_mask_apertures() -> None:
    trainer = _make_trainer(aperture=MaskApertureSpec(mask=np.ones((2, 2), dtype=np.float64)))
    calls: list[tuple[str, int]] = []

    trainer._predict_fiber_spectrum_from_field_sum = (  # type: ignore[method-assign]
        lambda exp_id, include_additive=True: (
            calls.append(("field", int(exp_id))),
            (np.array([5000.0, 5100.0], dtype=np.float64), np.array([3.0, 3.0], dtype=np.float32)),
        )[1]
    )
    trainer._predict_fiber_spectrum_from_aperture_quadrature = (  # type: ignore[method-assign]
        lambda exp_id, include_additive=True: (
            calls.append(("quad", int(exp_id))),
            (np.array([5000.0, 5100.0], dtype=np.float64), np.array([4.0, 4.0], dtype=np.float32)),
        )[1]
    )

    spectra = trainer.predict_fiber_spectra()

    assert calls == [("field", 1)]
    np.testing.assert_allclose(spectra[1][1], np.array([3.0, 3.0], dtype=np.float32))


def test_deterministic_sky_disk_spectrum_is_repeatable_and_closer_to_mc_mean_than_mask_shortcut() -> None:
    trainer = _make_sky_disk_mc_trainer()

    det_a = trainer.predict_fiber_spectra()[1][1]
    det_b = trainer.predict_fiber_spectra()[1][1]
    np.testing.assert_allclose(det_a, det_b, atol=1.0e-7, rtol=1.0e-7)

    mc_mean = _mc_mean_spectrum(trainer, exp_id=1, repeats=128)
    mask_shortcut = trainer._predict_fiber_spectrum_from_field_sum(1)[1]

    det_err = float(np.mean(np.abs(det_a - mc_mean)))
    mask_err = float(np.mean(np.abs(mask_shortcut - mc_mean)))

    assert det_err < mask_err * 0.5


def test_spectrum_projector_uses_lam_edges_for_bin_averages() -> None:
    renderer = SpectralRenderer({"r": bandpass("r")}, lambda_step=50.0, device=torch.device("cpu"))
    lam_grid = renderer.grid.lam.detach().cpu().numpy()
    field_sum = torch.tensor(lam_grid / 1000.0, dtype=torch.float32)
    projector = renderer.build_spectrum_projector(
        np.array([5000.0, 5150.0], dtype=np.float64), None, 0.0,
        lam_edges=np.array([[4950.0, 5050.0], [5050.0, 5250.0]], dtype=np.float64),
    )
    assert projector is not None

    projected = renderer.project_field_to_spectrum(
        field_sum, projector, apply_photon_weight=False, apply_throughput=False, apply_lsf=False,
    ).detach().cpu().numpy()

    np.testing.assert_allclose(projected, np.array([5.0, 5.15], dtype=np.float32), atol=1.0e-5)


def test_apply_diagnostic_lsf_accepts_per_bin_fwhm_array() -> None:
    spectrum = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    smoothed = DatumTrainer._apply_diagnostic_lsf(
        spectrum, np.array([5000.0, 5050.0, 5100.0], dtype=np.float64),
        np.array([0.0, 180.0, 0.0], dtype=np.float64),
    )

    assert smoothed.shape == spectrum.shape
    assert float(smoothed[1].item()) < 1.0
    assert float(smoothed[1].item()) > 0.0
    assert float(smoothed[0].item()) == pytest.approx(0.0)
    assert float(smoothed[2].item()) == pytest.approx(0.0)


def test_irregular_sdss_like_deterministic_spectrum_tracks_mc_mean() -> None:
    trainer = _make_irregular_sky_disk_mc_trainer()

    det = trainer.predict_fiber_spectra()[1][1]
    mc_mean = _mc_mean_spectrum(trainer, exp_id=1, repeats=512)
    shortcut = trainer._predict_fiber_spectrum_from_field_sum(1)[1]

    det_mae = float(np.mean(np.abs(det - mc_mean)))
    shortcut_mae = float(np.mean(np.abs(shortcut - mc_mean)))
    mean_signal = float(np.mean(np.abs(mc_mean)))

    assert det_mae / max(mean_signal, 1.0e-6) < 0.08
    assert det_mae < shortcut_mae * 0.85


def test_predict_fiber_spectra_variants_match_two_pass_predictions() -> None:
    trainer = _make_trainer(
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.5)
    )
    trainer.spec_additive_by_exposure = {
        1: np.array([0.25, -0.5], dtype=np.float32)
    }

    source_only, with_additive = trainer.predict_fiber_spectra_variants()
    reference_source = trainer.predict_fiber_spectra(include_additive=False)
    reference_additive = trainer.predict_fiber_spectra(include_additive=True)

    assert set(source_only) == set(reference_source)
    assert set(with_additive) == set(reference_additive)
    for exp_id in reference_source:
        np.testing.assert_array_equal(
            source_only[exp_id][1], reference_source[exp_id][1]
        )
        np.testing.assert_array_equal(
            with_additive[exp_id][1], reference_additive[exp_id][1]
        )
        np.testing.assert_array_equal(source_only[exp_id][0], with_additive[exp_id][0])
