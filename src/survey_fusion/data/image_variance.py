"""Per-column CCD variance coefficients for trainer.image_variance=poisson_current.

The loader's image variance (sdss_scene._compute_poisson_sigma_nanomaggy) is, in
training flux units,

    sigma^2 = alpha * max(total, 0) + beta,
    alpha = u * calib / gain,    beta = (u * calib)^2 * dark_variance,

with u the exposure's nanomaggy-to-training factor and calib the frame calibration
(nanomaggies per DN) interpolated at the pixel's frame column. An SDSS cutout is a
column sub-array of its frame (same projection, integer CRPIX offset), so the
cutout-to-frame column map, calib, alpha and beta depend on the cutout column only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .sdss_scene import (
    FNU_PER_NANOMAGGY, _bandpass_fnu_to_flambda_factor, _interp_1d, _sanitize_header_for_wcs,
)

IMAGE_VARIANCE_MODES = ("data", "poisson_current")
# The loader evaluates calib at a WCS round trip of every pixel; a column-only
# map must agree with it on every sampled row to well below one pixel.
COLUMN_MAP_TOLERANCE_PX = 1e-6
# float32 storage of y and sigma alone gives ~1e-7; one column of misregistration
# gives >2e-4 in every full84 exposure (measured).
SIGMA_RECONSTRUCTION_RTOL = 1e-6


def image_variance_outlier_pull(trainer_cfg: Mapping[str, Any]) -> float | None:
    """The guard pull for poisson_current, or None for the data variance."""
    mode = str(trainer_cfg.get("image_variance", "data"))
    if mode not in IMAGE_VARIANCE_MODES:
        raise ValueError(f"trainer.image_variance must be one of {IMAGE_VARIANCE_MODES}, got {mode!r}.")
    pull = trainer_cfg.get("image_variance_outlier_pull")
    if mode == "data":
        if pull is not None:
            raise ValueError(
                "trainer.image_variance_outlier_pull applies only to trainer.image_variance=poisson_current."
            )
        return None
    if pull is None or isinstance(pull, bool):
        raise ValueError(
            "trainer.image_variance=poisson_current requires trainer.image_variance_outlier_pull."
        )
    pull = float(pull)
    if not math.isfinite(pull) or pull <= 0.0:
        raise ValueError(f"trainer.image_variance_outlier_pull must be finite and positive, got {pull!r}.")
    if not bool(trainer_cfg.get("analytic_images", False)):
        raise ValueError(
            "trainer.image_variance=poisson_current requires trainer.analytic_images=true: the variance is "
            "evaluated at the exact model prediction, which the MC image path does not have."
        )
    return pull


def poisson_current_requested(trainer_cfg: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> bool:
    """Validate the image-variance configuration across sections; True for poisson_current."""
    if image_variance_outlier_pull(trainer_cfg) is None:
        return False
    if data_cfg.get("image_sigma_model_path") is not None:
        raise ValueError(
            "trainer.image_variance=poisson_current takes the variance from the current model; it cannot be "
            "combined with data.image_sigma_model_path, which replaces the store sigma with a fixed render's."
        )
    return True


def _column_calibration(cutout_header: fits.Header, frame_path: str) -> np.ndarray:
    with fits.open(frame_path) as hdul:
        frame_header = hdul[0].header.copy()
        calib = np.asarray(hdul[1].data, dtype=np.float64)
    height, width = int(cutout_header["NAXIS2"]), int(cutout_header["NAXIS1"])
    cutout_wcs = WCS(_sanitize_header_for_wcs(cutout_header))
    frame_wcs = WCS(_sanitize_header_for_wcs(frame_header))
    rows = np.array(sorted({0, height // 2, height - 1}), dtype=np.float64)
    yy, xx = np.meshgrid(rows, np.arange(width, dtype=np.float64), indexing="ij")
    ra_deg, dec_deg = cutout_wcs.all_pix2world(xx, yy, 0)
    frame_x, _ = frame_wcs.all_world2pix(ra_deg, dec_deg, 0)
    row_spread = float(np.max(np.abs(frame_x - frame_x[0])))
    if not row_spread <= COLUMN_MAP_TOLERANCE_PX:
        raise ValueError(
            f"Cutout of {frame_path} does not map onto frame columns independently of row "
            f"(spread {row_spread!r} px); per-column variance coefficients would not be exact."
        )
    calib_col = np.maximum(_interp_1d(calib, frame_x[0]), 1e-30)
    if not np.all(np.isfinite(calib_col)):
        raise ValueError(f"Cutout of {frame_path} has columns outside the frame calibration vector.")
    return calib_col


def image_variance_columns(scene: Any, store: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per image exposure id: float64 (alpha, beta) over the cutout columns.

    Exposure ids are positions in scene.image_exposures, as the adapter assigns
    them; u uses the bandpass the store was built with.
    """
    exposures = getattr(scene, "image_exposures", None)
    if not exposures:
        raise ValueError("Per-column image variance requires a scene of individual image exposures.")
    training_flux_scale = float(scene.metadata.get("training_flux_scale", 1.0))
    columns: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for exp_id, band in sorted(store.image_band_by_exposure.items()):
        exp_id, band = int(exp_id), str(band)
        image_exp = exposures[exp_id]
        if str(image_exp.band) != band:
            raise ValueError(
                f"Image exposure {exp_id} is band {image_exp.band!r} in the scene, {band!r} in the store."
            )
        gain = float(image_exp.metadata["gain"])
        dark_variance = float(image_exp.metadata["dark_variance"])
        if not math.isfinite(gain) or gain <= 0.0 or not math.isfinite(dark_variance) or dark_variance <= 0.0:
            raise ValueError(
                f"Image exposure {image_exp.label!r} needs finite positive gain and dark_variance, "
                f"got {gain!r}, {dark_variance!r}."
            )
        cutout_header = fits.getheader(image_exp.cutout_path, 0)
        width = int(store.exposures[exp_id].image_width)
        if int(cutout_header["NAXIS1"]) != width:
            raise ValueError(
                f"Cutout {image_exp.cutout_path} is {cutout_header['NAXIS1']} wide; the store has {width}."
            )
        conversion = _bandpass_fnu_to_flambda_factor(store.exposures[exp_id].bandpasses[band])
        pixel_area = float(image_exp.pixel_area_arcsec2)
        nmgy_to_training = FNU_PER_NANOMAGGY * conversion / max(pixel_area, 1e-20) * training_flux_scale
        calib = _column_calibration(cutout_header, image_exp.frame_path)
        columns[exp_id] = (
            nmgy_to_training * calib / gain,
            (nmgy_to_training * calib) ** 2 * dark_variance,
        )
    return columns


def sigma_reconstruction_error(
    y: np.ndarray, sigma: np.ndarray, local: np.ndarray, width: int, alpha: np.ndarray, beta: np.ndarray,
) -> float:
    """Max relative error of sqrt(alpha * max(y, 0) + beta) against the stored sigma."""
    column = np.asarray(local, dtype=np.int64) % int(width)
    rebuilt = np.sqrt(alpha[column] * np.maximum(np.asarray(y, dtype=np.float64), 0.0) + beta[column])
    return float(np.max(np.abs(rebuilt / np.asarray(sigma, dtype=np.float64) - 1.0), initial=0.0))
