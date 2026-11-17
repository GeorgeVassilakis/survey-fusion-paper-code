from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.pgm.contracts import (
    DatumBatch, DatumKernelFactory, DatumKind, ExposureState, ExposureTable, MaskApertureSpec,
)
from survey_fusion.pgm.kernel_builders import KernelFactoryFromBuilders
from survey_fusion.models.spectral_field import GalaxySpectralField
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.spectral import per_bin_edges_from_centers
from survey_fusion.utils.wcs import TangentPlaneWCS


@dataclass(slots=True)
class SyntheticDatumStore:
    exposures: ExposureTable
    batch: DatumBatch
    image_band_by_exposure: dict[int, str]
    kernel_factory: DatumKernelFactory
    n_image: int
    n_spec: int


class CountingModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        self.calls = 0
        self.points = 0

    def reset(self) -> None:
        self.calls = 0
        self.points = 0

    def forward(self, x_norm, y_norm, log_lam):
        self.calls += 1
        self.points += int(x_norm.shape[0])
        return self.base(x_norm, y_norm, log_lam)


def _bandpass() -> Bandpass:
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([0.2, 1.0, 0.2], dtype=np.float64)
    return Bandpass(band="r", lam=lam, throughput=throughput, canonical_id="test_r")


def _build_synthetic_store(device: torch.device) -> SyntheticDatumStore:
    del device
    height, width = 2, 2
    lam_spec = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    wcs = TangentPlaneWCS(ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=0.4, width=width, height=height)
    bandpass = _bandpass()
    exposures: ExposureTable = {
        0: ExposureState(
            wcs=wcs, obs_wcs=wcs, psf_fwhm_arcsec=None, lsf_fwhm_angstrom=None, bandpasses={"r": bandpass},
            lam_spec=lam_spec, fiber_mask=np.ones((height, width), dtype=np.float32), pixel_scale_arcsec=0.4,
            image_height=height, image_width=width,
        ),
        1: ExposureState(
            wcs=wcs, obs_wcs=wcs, psf_fwhm_arcsec=None, lsf_fwhm_angstrom=None, bandpasses={},
            lam_spec=lam_spec, fiber_mask=np.ones((height, width), dtype=np.float32), pixel_scale_arcsec=0.4,
            image_height=None, image_width=None, lam_edges=per_bin_edges_from_centers(lam_spec),
            aperture=MaskApertureSpec(mask=np.ones((height, width), dtype=np.float64)),
        ),
    }
    image_y = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    spec_y = np.array([0.8, 0.9, 1.0], dtype=np.float32)
    y = torch.from_numpy(np.concatenate([image_y, spec_y], axis=0))
    sigma = torch.ones_like(y)
    exposure_id = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    datum_kind = torch.tensor(
        [
            int(DatumKind.IMAGE_PIXEL), int(DatumKind.IMAGE_PIXEL), int(DatumKind.IMAGE_PIXEL),
            int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN), int(DatumKind.SPECTRAL_BIN),
            int(DatumKind.SPECTRAL_BIN),
        ],
        dtype=torch.long,
    )
    datum_index = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
    batch = DatumBatch(
        y=y, sigma=sigma, exposure_id=exposure_id, datum_kind=datum_kind, datum_index=datum_index,
        kernel_samples=None,
    )
    image_band_by_exposure = {0: "r"}
    kernel_factory = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure=image_band_by_exposure, rng_seed=123,
    )
    return SyntheticDatumStore(
        exposures=exposures, batch=batch, image_band_by_exposure=image_band_by_exposure,
        kernel_factory=kernel_factory, n_image=int(image_y.size), n_spec=int(spec_y.size),
    )


def test_mc_train_epoch_is_batch_first():
    device = torch.device("cpu")
    grid = build_spatial_grid(2, 2, device)
    base = GalaxySpectralField(hidden_dim=32, n_layers=2, softplus_beta=2.0)
    model = CountingModel(base)

    data_store = _build_synthetic_store(device)
    trainer = DatumTrainer(
        model=model,
        grid=grid,
        data=data_store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "epochs": 1, "max_steps": 1, "learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.999,
            "apply_spatial_psf": False, "kernel_samples_k": 8, "replica_samples_k": 1, "pixel_batch_size": 3,
            "spec_batch_size": 2,
        },
        device=device,
    )
    model.reset()
    trainer.fit()

    expected_points = int(
        (trainer.image_batch_size + trainer.spec_batch_size)
        * trainer.replica_samples_k
        * 2
    )
    assert model.points == expected_points


def test_trainer_ignores_legacy_momentum_alias_for_adam_beta1() -> None:
    device = torch.device("cpu")
    grid = build_spatial_grid(2, 2, device)
    model = GalaxySpectralField(hidden_dim=32, n_layers=2, softplus_beta=2.0)
    data_store = _build_synthetic_store(device)

    trainer = DatumTrainer(
        model=model,
        grid=grid,
        data=data_store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "epochs": 1, "max_steps": 1, "learning_rate": 1e-3, "momentum": 0.123, "beta2": 0.999,
            "apply_spatial_psf": False, "kernel_samples_k": 8, "replica_samples_k": 1, "pixel_batch_size": 3,
            "spec_batch_size": 2,
        },
        device=device,
    )

    assert trainer.optimizer.param_groups[0]["betas"][0] == 0.9
