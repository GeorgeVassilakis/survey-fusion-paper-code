from __future__ import annotations

import numpy as np
import torch

from support.synthetic_trainer import ConstantKernelFactory, SyntheticStore as MockStore, bandpass
from support.kernel_factory import PlanPairKernelFactoryMixin
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import DatumBatch, ExposureState, GroupedKernelSamples, KernelSamples
from survey_fusion.training.renderer import convolve_flat_image, gaussian_psf_kernel
from survey_fusion.training.datum_trainer import DatumTrainer


class ConstantWCS:
    def pixel_to_omega_xyz(self, x, y):
        shape = np.broadcast(x, y).shape
        omega = np.zeros(shape + (3,), dtype=np.float64)
        omega[..., 0] = 1.0
        return omega

    def omega_xyz_to_norm_torch(self, omega_xyz: torch.Tensor):
        zeros = torch.zeros(omega_xyz.shape[:-1], dtype=omega_xyz.dtype, device=omega_xyz.device)
        return zeros, zeros


class SmoothUnitWCS:
    def __init__(self, *, width: int, height: int, scale: float = 0.2) -> None:
        self.width = int(width)
        self.height = int(height)
        self.scale = float(scale)

    def pixel_to_omega_xyz(self, x, y):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        x_c = (x_arr - 0.5 * (self.width - 1)) * self.scale
        y_c = (y_arr - 0.5 * (self.height - 1)) * self.scale
        rad2 = x_c * x_c + y_c * y_c
        z = np.sqrt(np.clip(1.0 - rad2, 1.0e-12, None))
        return np.stack([z, x_c, y_c], axis=-1)

    def omega_xyz_to_norm_torch(self, omega_xyz: torch.Tensor):
        return omega_xyz[..., 1], omega_xyz[..., 2]


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        return torch.ones_like(x_norm, dtype=torch.float32, device=self.bias.device) + 0.0 * self.bias


class QuadraticOmegaModel(torch.nn.Module):
    def __init__(self, coeff: float = 10.0):
        super().__init__()
        self.coeff = float(coeff)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_norm, y_norm, log_lam):
        del y_norm, log_lam
        return 1.0 + self.coeff * x_norm**2 + 0.0 * self.bias


class RandomSubpixelKernelFactory(PlanPairKernelFactoryMixin):
    def __init__(self, *, batch: DatumBatch, exposures: dict[int, ExposureState], rng_seed: int = 7) -> None:
        self._batch = batch
        self._exposures = exposures
        self._rng = np.random.default_rng(rng_seed)

    def build_kernel_samples_for_indices(
        self, global_indices: torch.Tensor, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples:
        idx = global_indices.detach().cpu().numpy().astype(np.int64)
        exp_ids = self._batch.exposure_id[global_indices].detach().cpu().numpy().astype(np.int64)
        local = self._batch.datum_index[global_indices].detach().cpu().numpy().astype(np.int64)
        omega_rows = []
        loglam_rows = []
        for exp_id, datum_index in zip(exp_ids, local, strict=True):
            exposure = self._exposures[int(exp_id)]
            width = int(exposure.image_width or exposure.fiber_mask.shape[1])
            x = float(datum_index % width)
            y = float(datum_index // width)
            dx = self._rng.random(int(k)) - 0.5
            dy = self._rng.random(int(k)) - 0.5
            omega = np.asarray(
                (exposure.obs_wcs or exposure.wcs).pixel_to_omega_xyz(x + dx, y + dy), dtype=np.float64,
            )
            omega_rows.append(omega)
            loglam_rows.append(np.full((int(k),), float(np.log(5500.0)), dtype=np.float64))
        omega_t = torch.tensor(np.stack(omega_rows, axis=0), dtype=torch.float32, device=device)
        loglam_t = torch.tensor(np.stack(loglam_rows, axis=0), dtype=torch.float32, device=device)
        w_t = torch.full((idx.size, int(k)), 1.0 / float(k), dtype=torch.float32, device=device)
        z_t = torch.ones((idx.size,), dtype=torch.float32, device=device)
        idx_t = global_indices.to(dtype=torch.long, device=self._batch.exposure_id.device)
        return GroupedKernelSamples(
            samples=KernelSamples(omega_xyz=omega_t, loglam=loglam_t, w=w_t, Z=z_t),
            row_original_index=torch.arange(idx.size, dtype=torch.long, device=device),
            row_exposure_id=torch.index_select(self._batch.exposure_id, 0, idx_t).to(
                dtype=torch.long, device=device
            ),
            row_datum_index=torch.index_select(self._batch.datum_index, 0, idx_t).to(
                dtype=torch.long, device=device
            ),
        )

    def build_kernel_sample_plan_for_indices(self, global_indices: torch.Tensor) -> torch.Tensor:
        return global_indices.clone()

    def build_kernel_samples_from_plan(
        self, plan: torch.Tensor, *, k: int, device: torch.device,
    ) -> GroupedKernelSamples:
        return self.build_kernel_samples_for_indices(plan, k=k, device=device)


def _make_exposure(
    *, band: str, wcs: object, shape: tuple[int, int], psf_fwhm_arcsec: float | None = None,
) -> ExposureState:
    height, width = shape
    return ExposureState(
        wcs=wcs, obs_wcs=wcs, psf_fwhm_arcsec=psf_fwhm_arcsec, lsf_fwhm_angstrom=None,
        bandpasses={band: bandpass(band)}, lam_spec=np.array([5000.0, 5100.0], dtype=np.float64),
        fiber_mask=np.ones(shape, dtype=np.float32), pixel_scale_arcsec=1.0, image_height=height,
        image_width=width,
    )


def _make_batch(device: torch.device, *, exposure_ids: list[int], shape: tuple[int, int]) -> DatumBatch:
    height, width = shape
    n_pix = height * width
    exp = np.repeat(np.asarray(exposure_ids, dtype=np.int64), n_pix)
    local = np.tile(np.arange(n_pix, dtype=np.int64), len(exposure_ids))
    n = exp.size
    return DatumBatch(
        y=torch.zeros(n, dtype=torch.float32, device=device),
        sigma=torch.ones(n, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(exp, dtype=torch.long, device=device),
        datum_kind=torch.zeros(n, dtype=torch.long, device=device),
        datum_index=torch.tensor(local, dtype=torch.long, device=device), kernel_samples=None,
    )


def _make_trainer(
    *, model: torch.nn.Module, exposures: dict[int, ExposureState], image_band_by_exposure: dict[int, str],
    batch: DatumBatch, kernel_factory: object, shape: tuple[int, int], apply_spatial_psf: bool = False,
    image_additive_by_exposure: dict[int, np.ndarray] | None = None,
) -> DatumTrainer:
    device = torch.device("cpu")
    store = MockStore(
        exposures=exposures, batch=batch, image_band_by_exposure=image_band_by_exposure,
        kernel_factory=kernel_factory, n_image=int(batch.y.shape[0]), n_spec=0,
        image_additive_by_exposure=dict(image_additive_by_exposure or {}),
    )
    return DatumTrainer(
        model=model,
        grid=build_spatial_grid(shape[0], shape[1], device),
        data=store,
        training_bands=list(image_band_by_exposure.values()),
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 64, "replica_samples_k": 1, "epochs": 0, "learning_rate": 1e-3,
            "apply_spatial_psf": bool(apply_spatial_psf),
        },
        device=device,
    )


def _single_mc_datum_image_predictions(trainer: DatumTrainer, *, exp_id: int) -> np.ndarray:
    idx = trainer._global_indices_for(exp_id, 0)
    local = torch.index_select(trainer.data.batch.datum_index, 0, idx).to(torch.long)
    runtime = trainer.runtime_by_exposure[int(exp_id)]
    n_pix = int(runtime.height * runtime.width)
    with torch.no_grad():
        mu = trainer.compute_mu(idx)
        flat = torch.empty((n_pix,), dtype=mu.dtype, device=trainer.device)
        flat[local] = mu
    return flat.view(runtime.height, runtime.width).detach().cpu().numpy()


def _mc_mean_datum_image_predictions(trainer: DatumTrainer, *, exp_id: int, repeats: int) -> np.ndarray:
    samples = [_single_mc_datum_image_predictions(trainer, exp_id=exp_id) for _ in range(repeats)]
    return np.mean(np.stack(samples, axis=0), axis=0)


def _full_grid_deterministic_datum_prediction(
    trainer: DatumTrainer, *, exp_id: int, supersample: int, chunk_size: int,
) -> np.ndarray:
    runtime = trainer.runtime_by_exposure[int(exp_id)]
    exposure = runtime.exposure
    height = int(runtime.height)
    width = int(runtime.width)
    band = trainer.image_band_by_exposure[int(exp_id)]
    pixel_wcs = exposure.obs_wcs or exposure.wcs

    kernel_hr = None
    pad = 0
    if trainer.apply_spatial_psf and exposure.psf_fwhm_arcsec is not None:
        kernel_hr = gaussian_psf_kernel(
            float(exposure.psf_fwhm_arcsec), float(exposure.pixel_scale_arcsec) / float(supersample),
            trainer.device,
        )
        pad = int(kernel_hr.shape[-1] // 2)

    x_hr = (np.arange(width * supersample + 2 * pad, dtype=np.float64) - pad + 0.5) / float(supersample) - 0.5
    y_hr = (
        np.arange(height * supersample + 2 * pad, dtype=np.float64) - pad + 0.5
    ) / float(supersample) - 0.5
    yy_hr, xx_hr = np.meshgrid(y_hr, x_hr, indexing="ij")
    omega_hr = np.asarray(pixel_wcs.pixel_to_omega_xyz(xx_hr, yy_hr), dtype=np.float64)
    if omega_hr.shape != (height * supersample + 2 * pad, width * supersample + 2 * pad, 3):
        raise ValueError("Unexpected WCS output shape in full-grid reference renderer.")
    band_hr = trainer._render_band_for_omega(
        int(exp_id), band, omega_hr.reshape(-1, 3), chunk_size=int(chunk_size),
    ).view(height * supersample + 2 * pad, width * supersample + 2 * pad)
    if kernel_hr is not None:
        band_hr = convolve_flat_image(
            band_hr.reshape(-1), height * supersample + 2 * pad, width * supersample + 2 * pad, kernel_hr,
        ).view(height * supersample + 2 * pad, width * supersample + 2 * pad)
        band_hr = band_hr[pad:-pad, pad:-pad]
    image_native = band_hr.view(height, supersample, width, supersample).mean(dim=(1, 3))
    return image_native.detach().cpu().numpy()


def test_deterministic_datum_predictions_repeatable_and_match_shape() -> None:
    device = torch.device("cpu")
    shape = (2, 2)
    exposures = {
        0: _make_exposure(band="g", wcs=ConstantWCS(), shape=shape),
        1: _make_exposure(band="i", wcs=ConstantWCS(), shape=shape),
    }
    batch = _make_batch(device, exposure_ids=[0, 1], shape=shape)
    trainer = _make_trainer(
        model=ConstantModel(), exposures=exposures, image_band_by_exposure={0: "g", 1: "i"}, batch=batch,
        kernel_factory=ConstantKernelFactory(batch), shape=shape,
    )

    pred_a = trainer.compute_datum_image_predictions_deterministic()
    pred_b = trainer.compute_datum_image_predictions_deterministic()

    assert sorted(pred_a.keys()) == [0, 1]
    assert pred_a[0]["g"].shape == shape
    assert pred_a[1]["i"].shape == shape
    np.testing.assert_allclose(pred_a[0]["g"], pred_b[0]["g"])
    np.testing.assert_allclose(pred_a[1]["i"], pred_b[1]["i"])
    assert np.allclose(pred_a[0]["g"], pred_a[0]["g"][0, 0])
    assert np.allclose(pred_a[1]["i"], pred_a[1]["i"][0, 0])


def test_deterministic_datum_predictions_can_return_source_and_total() -> None:
    device = torch.device("cpu")
    shape = (2, 2)
    exposures = {0: _make_exposure(band="r", wcs=ConstantWCS(), shape=shape)}
    batch = _make_batch(device, exposure_ids=[0], shape=shape)
    additive = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    trainer = _make_trainer(
        model=ConstantModel(), exposures=exposures, image_band_by_exposure={0: "r"}, batch=batch,
        kernel_factory=ConstantKernelFactory(batch), shape=shape, image_additive_by_exposure={0: additive},
    )

    source = trainer.compute_datum_image_predictions_deterministic(
        supersample=1, include_additive=False,
    )[0]["r"]
    total = trainer.compute_datum_image_predictions_deterministic(
        supersample=1, include_additive=True,
    )[0]["r"]

    assert np.allclose(source, source[0, 0])
    np.testing.assert_allclose(total, source + additive)


def test_deterministic_datum_matches_full_grid_reference_with_supersample_one_and_no_psf() -> None:
    device = torch.device("cpu")
    shape = (1, 3)
    wcs = SmoothUnitWCS(width=shape[1], height=shape[0], scale=0.2)
    exposures = {0: _make_exposure(band="r", wcs=wcs, shape=shape)}
    batch = _make_batch(device, exposure_ids=[0], shape=shape)
    trainer = _make_trainer(
        model=QuadraticOmegaModel(coeff=10.0), exposures=exposures, image_band_by_exposure={0: "r"},
        batch=batch, kernel_factory=ConstantKernelFactory(batch), shape=shape,
    )

    det = trainer.compute_datum_image_predictions_deterministic(supersample=1)[0]["r"]
    reference = _full_grid_deterministic_datum_prediction(trainer, exp_id=0, supersample=1, chunk_size=2048)

    np.testing.assert_allclose(det, reference, atol=1.0e-6, rtol=1.0e-6)


def test_streamed_deterministic_datum_matches_full_grid_reference() -> None:
    device = torch.device("cpu")
    shape = (5, 7)
    wcs = SmoothUnitWCS(width=shape[1], height=shape[0], scale=0.17)
    exposures = {0: _make_exposure(band="r", wcs=wcs, shape=shape)}
    batch = _make_batch(device, exposure_ids=[0], shape=shape)
    trainer = _make_trainer(
        model=QuadraticOmegaModel(coeff=6.0), exposures=exposures, image_band_by_exposure={0: "r"},
        batch=batch, kernel_factory=ConstantKernelFactory(batch), shape=shape,
    )

    chunk_size = 11
    streamed = trainer.compute_datum_image_predictions_deterministic(
        supersample=4, chunk_size=chunk_size,
    )[0]["r"]
    reference = _full_grid_deterministic_datum_prediction(
        trainer, exp_id=0, supersample=4, chunk_size=chunk_size,
    )

    np.testing.assert_allclose(streamed, reference, atol=1.0e-6, rtol=1.0e-6)


def test_deterministic_datum_agrees_with_mc_mean() -> None:
    device = torch.device("cpu")
    shape = (1, 3)
    wcs = SmoothUnitWCS(width=shape[1], height=shape[0], scale=0.22)
    exposures = {0: _make_exposure(band="r", wcs=wcs, shape=shape)}
    batch = _make_batch(device, exposure_ids=[0], shape=shape)
    factory = RandomSubpixelKernelFactory(batch=batch, exposures=exposures, rng_seed=17)
    trainer = _make_trainer(
        model=QuadraticOmegaModel(coeff=12.0), exposures=exposures, image_band_by_exposure={0: "r"},
        batch=batch, kernel_factory=factory, shape=shape,
    )

    mc_mean = _mc_mean_datum_image_predictions(trainer, exp_id=0, repeats=256)
    det = trainer.compute_datum_image_predictions_deterministic(supersample=4)[0]["r"]

    det_err = float(np.mean(np.abs(det - mc_mean)))

    assert det_err < 0.05


def test_windowed_deterministic_datum_matches_full_grid_reference_with_psf() -> None:
    device = torch.device("cpu")
    shape = (6, 8)
    wcs = SmoothUnitWCS(width=shape[1], height=shape[0], scale=0.15)
    exposures = {0: _make_exposure(band="r", wcs=wcs, shape=shape, psf_fwhm_arcsec=1.3)}
    batch = _make_batch(device, exposure_ids=[0], shape=shape)
    trainer = _make_trainer(
        model=QuadraticOmegaModel(coeff=8.0), exposures=exposures, image_band_by_exposure={0: "r"},
        batch=batch, kernel_factory=ConstantKernelFactory(batch), shape=shape, apply_spatial_psf=True,
    )

    reference = _full_grid_deterministic_datum_prediction(trainer, exp_id=0, supersample=4, chunk_size=13)
    window = trainer.render_datum_image_window_deterministic(
        0, y0=1, y1=5, x0=2, x1=7, supersample=4, chunk_size=13,
    )

    np.testing.assert_allclose(window, reference[1:5, 2:7], atol=1.0e-6, rtol=1.0e-6)
