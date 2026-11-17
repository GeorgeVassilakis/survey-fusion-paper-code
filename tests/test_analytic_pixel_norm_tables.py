"""The tabulated pixel coordinates must equal the per-batch transform.

`_compute_analytic_image_source_mu` used to run the pixel->sky->model-plane map
on the batch's own pixels every step. It now gathers whole-frame tables built
through the same two calls, which is only legitimate because both are
per-point: these tests pin that equality bitwise on a real tangent-plane WCS,
and pin the fallback for the quadrature orders the tables do not cover.
"""

from __future__ import annotations

import numpy as np
import torch

from support.synthetic_trainer import (
    ConstantKernelFactory,
    OnesModel,
    SyntheticStore,
    image_exposure,
    make_batch,
)
from survey_fusion.data.grids import build_spatial_grid
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.utils.wcs import TangentPlaneWCS


EXPOSURE_SHAPES = {0: (5, 7), 1: (4, 3)}
NO_OFFSET = np.zeros((1, 1), dtype=np.float64)


def _tangent_wcs(height: int, width: int, ra0_deg: float) -> TangentPlaneWCS:
    return TangentPlaneWCS(
        ra0_rad=np.deg2rad(ra0_deg),
        dec0_rad=np.deg2rad(2.0),
        pixel_scale_arcsec=0.396,
        width=width,
        height=height,
    )


def _trainer() -> DatumTrainer:
    device = torch.device("cpu")
    exposure_ids: list[int] = []
    datum_index: list[int] = []
    for exp_id, (height, width) in EXPOSURE_SHAPES.items():
        exposure_ids.extend([exp_id] * (height * width))
        datum_index.extend(range(height * width))
    n_image = len(exposure_ids)
    batch = make_batch(
        y=torch.ones(n_image, dtype=torch.float32, device=device),
        exposure_id=torch.tensor(exposure_ids, dtype=torch.long, device=device),
        datum_kind=torch.zeros(n_image, dtype=torch.long, device=device),
        datum_index=torch.tensor(datum_index, dtype=torch.long, device=device),
    )
    store = SyntheticStore(
        exposures={
            exp_id: image_exposure(
                height=height,
                width=width,
                wcs=_tangent_wcs(height, width, 150.0),
                obs_wcs=_tangent_wcs(height, width, 150.0 + 0.01 * exp_id),
            )
            for exp_id, (height, width) in EXPOSURE_SHAPES.items()
        },
        batch=batch,
        image_band_by_exposure={exp_id: "r" for exp_id in EXPOSURE_SHAPES},
        kernel_factory=ConstantKernelFactory(batch),
        n_image=n_image,
        n_spec=0,
    )
    return DatumTrainer(
        model=OnesModel(),
        grid=build_spatial_grid(5, 7, device),
        data=store,
        training_bands=["r"],
        held_out_band="i",
        lambda_step=20.0,
        trainer_cfg={
            "kernel_samples_k": 4,
            "replica_samples_k": 1,
            "epochs": 1,
            "learning_rate": 1e-3,
            "pixel_batch_size": 8,
            "spec_batch_size": 0,
            "apply_spatial_psf": False,
        },
        device=device,
    )


def _per_batch_norm(exposure, local: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """The transform `_compute_analytic_image_source_mu` runs on a batch."""
    width = int(exposure.image_width)
    px = np.mod(local, width).astype(np.float64)[:, None]
    py = np.floor_divide(local, width).astype(np.float64)[:, None]
    omega = np.asarray(
        exposure.obs_wcs.pixel_to_omega_xyz(px + NO_OFFSET, py + NO_OFFSET),
        dtype=np.float64,
    ).reshape(-1, 3)
    return exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(omega, dtype=torch.float32)
    )


def test_tabulated_coordinates_equal_the_per_batch_transform() -> None:
    trainer = _trainer()
    exposure_ids = np.asarray(sorted(EXPOSURE_SHAPES), dtype=np.int64)

    assert trainer._ensure_pixel_norm_tables(exposure_ids, NO_OFFSET, NO_OFFSET, 1)

    for exp_id, (height, width) in EXPOSURE_SHAPES.items():
        exposure = trainer.runtime_by_exposure[exp_id].exposure
        local = np.arange(height * width, dtype=np.int64)
        x_reference, y_reference = _per_batch_norm(exposure, local)
        table_index = torch.as_tensor(
            trainer._pixel_norm_base[exp_id] + local, dtype=torch.long
        )
        assert torch.equal(trainer._pixel_norm_x[table_index], x_reference)
        assert torch.equal(trainer._pixel_norm_y[table_index], y_reference)


def test_tables_are_filled_only_for_the_exposures_a_batch_touches() -> None:
    trainer = _trainer()

    assert trainer._ensure_pixel_norm_tables(
        np.asarray([0], dtype=np.int64), NO_OFFSET, NO_OFFSET, 1
    )
    assert bool(trainer._pixel_norm_filled[0])
    assert not bool(trainer._pixel_norm_filled[1])

    exposure = trainer.runtime_by_exposure[1].exposure
    assert trainer._ensure_pixel_norm_tables(
        np.asarray([0, 1], dtype=np.int64), NO_OFFSET, NO_OFFSET, 1
    )
    local = np.arange(int(exposure.image_height) * int(exposure.image_width))
    x_reference, _ = _per_batch_norm(exposure, local)
    table_index = torch.as_tensor(
        trainer._pixel_norm_base[1] + local, dtype=torch.long
    )
    assert torch.equal(trainer._pixel_norm_x[table_index], x_reference)


def test_multi_node_quadrature_keeps_the_per_batch_transform() -> None:
    trainer = _trainer()

    assert not trainer._ensure_pixel_norm_tables(
        np.asarray([0], dtype=np.int64), NO_OFFSET, NO_OFFSET, 4
    )
    assert trainer._pixel_norm_x is None
