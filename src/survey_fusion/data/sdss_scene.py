from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..utils.spectral import masked_bin_edges
from ..utils.wcs import TangentPlaneWCS, radec_deg_to_omega_xyz
from .bandpasses import Bandpass
from .flux_systems import (
    IDLSPEC2D_AB_CORRECTION_MAG, SPECTRAL_FLUX_SYSTEMS, pipeline_band_ratios, ratio_knots,
    spectrum_to_image_ratio,
)
from .psf_mixture import psf_mixture_digest as recompute_psf_mixture_digest
from ..pgm.contracts import PSFMixtureComponents


FNU_PER_NANOMAGGY = 3.631e-29  # erg / s / cm^2 / Hz
C_ANG_PER_S = 2.99792458e18


@dataclass(slots=True)
class SdssSceneData:
    images: Dict[str, np.ndarray]
    images_flambda: Dict[str, np.ndarray]
    spectra: list["SdssSpectrumData"]
    lam_spec: np.ndarray
    flux_spec_flambda: np.ndarray
    flux_spec_flam_lsf: np.ndarray
    spec_throughput: np.ndarray
    fiber_mask: np.ndarray
    fiber_area_arcsec2: float
    pixel_scale_arcsec: float
    fiber_radius_arcsec: float
    metadata: Dict[str, Any]
    wcs_per_band: Dict[str, WCS]
    pixel_scale_arcsec_per_band: Dict[str, float]
    pixel_area_arcsec2: Dict[str, float]
    spec_ivar: np.ndarray
    spec_wdisp: np.ndarray
    spec_loglam: np.ndarray
    sigma_spec_flambda: np.ndarray
    fiber_center_ra_dec: Tuple[float, float]
    image_sigma_flambda: Dict[str, np.ndarray] | None = None
    image_source_flambda: Dict[str, np.ndarray] | None = None
    image_additive_flambda: Dict[str, np.ndarray] | None = None
    image_total_flambda: Dict[str, np.ndarray] | None = None
    spec_source_flambda: np.ndarray | None = None
    spec_additive_flambda: np.ndarray | None = None
    spec_total_flambda: np.ndarray | None = None
    image_exposures: list["SdssImageExposureData"] | None = None
    model_wcs: TangentPlaneWCS | None = None


@dataclass(slots=True)
class SdssImageExposureData:
    label: str
    band: str
    cutout_path: str
    frame_path: str
    source_flambda: np.ndarray
    additive_flambda: np.ndarray
    total_flambda: np.ndarray
    target_flambda: np.ndarray
    sigma_flambda: np.ndarray
    wcs: WCS
    pixel_scale_arcsec: float
    pixel_area_arcsec2: float
    psf_fwhm_arcsec: float | None
    metadata: Dict[str, Any]
    psf_mixture_dgpsf: PSFMixtureComponents | None = None
    psf_mixture_kl: PSFMixtureComponents | None = None
    psf_effective_fwhm_arcsec: float | None = None
    psf_effective_fwhm_dg_arcsec: float | None = None
    psf_effective_fwhm_kl_arcsec: float | None = None


@dataclass(slots=True)
class SdssSpectrumData:
    label: str
    role: str
    path: str
    lam_spec: np.ndarray
    flux_spec_flambda: np.ndarray
    flux_spec_flam_lsf: np.ndarray
    spec_throughput: np.ndarray
    fiber_mask: np.ndarray
    spec_ivar: np.ndarray
    spec_wdisp: np.ndarray
    spec_loglam: np.ndarray
    sigma_spec_flambda: np.ndarray
    center_ra_dec: Tuple[float, float]
    fiber_radius_arcsec: float
    metadata: Dict[str, Any]
    spec_source_flambda: np.ndarray | None = None
    spec_additive_flambda: np.ndarray | None = None
    spec_total_flambda: np.ndarray | None = None
    # (n_bins, 2) per-bin [lower, upper] wavelength edges, computed on the
    # full pipeline grid before masking so bins next to masked gaps keep
    # their true widths
    lam_edges: np.ndarray | None = None


class NoValidSpectrumBinsError(ValueError):
    """Raised when a spectrum has no bins surviving the configured validity mask."""


def _as_path(path_str: str | None, base_dir: Path | None = None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    base_dir = base_dir.expanduser()
    if path == base_dir or path.is_relative_to(base_dir):
        return path
    return (base_dir / path).expanduser()


def _resolve_manifest_entry_path(path_str: str | None, *, observation_root: Path | None) -> Path | None:
    path = _as_path(path_str, observation_root)
    if path is None or observation_root is None:
        return path
    if path.exists() or not path.is_absolute():
        return path

    candidates: list[Path] = []
    observation_root = observation_root.expanduser()
    matching_indices = [idx for idx, part in enumerate(path.parts) if part == observation_root.name]
    if matching_indices:
        suffix = path.parts[matching_indices[-1] + 1 :]
        candidates.append(observation_root.joinpath(*suffix) if suffix else observation_root)
    candidates.append(observation_root / path.name)

    ordered_unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_unique_candidates.append(candidate)
        if candidate.exists():
            return candidate

    if ordered_unique_candidates:
        return ordered_unique_candidates[0]
    return path


def _bandpass_fnu_to_flambda_factor(bp: Bandpass) -> float:
    lam = np.asarray(bp.lam, dtype=np.float64)
    throughput = np.asarray(bp.throughput, dtype=np.float64)
    numerator = np.trapezoid(throughput / lam, lam)
    denominator = np.trapezoid(throughput * lam, lam)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError(f"Bandpass {bp.band} has non-positive throughput integral.")
    return C_ANG_PER_S * numerator / denominator


def _sanitize_header_for_wcs(header: fits.Header) -> fits.Header:
    clean = fits.Header()
    for card in header.cards:
        key = card.keyword
        if key in {"COMMENT", "HISTORY"}:
            continue
        value = card.value
        try:
            if isinstance(value, str):
                value.encode("ascii")
            clean[key] = (value, card.comment)
        except Exception:
            continue
    return clean


def _load_wcs_and_scale(header: fits.Header) -> tuple[WCS, float, float]:
    header_clean = _sanitize_header_for_wcs(header)
    wcs = WCS(header_clean)
    scales_deg = proj_plane_pixel_scales(wcs)
    pixel_scale_arcsec = float(np.mean(scales_deg) * 3600.0)
    pixel_area_arcsec2 = float(np.prod(scales_deg) * 3600.0 * 3600.0)
    return wcs, pixel_scale_arcsec, pixel_area_arcsec2


def _image_center_radec(wcs: WCS, shape: tuple[int, int]) -> tuple[float, float]:
    if len(shape) != 2:
        raise ValueError("Image center RA/Dec requires a 2D image shape.")
    y_size, x_size = int(shape[0]), int(shape[1])
    x_ref = 0.5 * float(x_size - 1)
    y_ref = 0.5 * float(y_size - 1)
    ra_deg, dec_deg = wcs.all_pix2world(x_ref, y_ref, 0)
    return float(ra_deg), float(dec_deg)


def _model_wcs_from_manifest(manifest: dict[str, Any]) -> TangentPlaneWCS | None:
    required = {"center_ra_deg", "center_dec_deg", "half_size_x_arcsec", "half_size_y_arcsec"}
    if not required.issubset(manifest):
        return None
    center_ra_deg = float(manifest["center_ra_deg"])
    center_dec_deg = float(manifest["center_dec_deg"])
    half_x_arcsec = float(manifest["half_size_x_arcsec"])
    half_y_arcsec = float(manifest["half_size_y_arcsec"])
    pixel_scale_arcsec = float(manifest.get("model_pixel_scale_arcsec", 0.396))
    if pixel_scale_arcsec <= 0.0 or not np.isfinite(pixel_scale_arcsec):
        raise ValueError("model_pixel_scale_arcsec must be finite and > 0.")
    if half_x_arcsec <= 0.0 or half_y_arcsec <= 0.0:
        raise ValueError("Manifest half-size arcsec values must be > 0 for model WCS.")
    width = int(np.ceil((2.0 * half_x_arcsec) / pixel_scale_arcsec)) + 1
    height = int(np.ceil((2.0 * half_y_arcsec) / pixel_scale_arcsec)) + 1
    return TangentPlaneWCS(
        ra0_rad=float(np.deg2rad(center_ra_deg)), dec0_rad=float(np.deg2rad(center_dec_deg)),
        pixel_scale_arcsec=pixel_scale_arcsec, width=width, height=height,
    )


def _world_to_pixel(wcs_obj: WCS | TangentPlaneWCS, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    if isinstance(wcs_obj, TangentPlaneWCS):
        omega = radec_deg_to_omega_xyz(
            np.asarray(ra_deg, dtype=np.float64), np.asarray(dec_deg, dtype=np.float64),
        )
        x_pix, y_pix = wcs_obj.omega_xyz_to_pixel(omega)
        return float(np.asarray(x_pix, dtype=np.float64)), float(np.asarray(y_pix, dtype=np.float64))
    x_pix, y_pix = wcs_obj.all_world2pix(ra_deg, dec_deg, 0)
    return float(np.asarray(x_pix, dtype=np.float64)), float(np.asarray(y_pix, dtype=np.float64))


def _read_header(path: Path) -> fits.Header:
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def _header_value_as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    return text or None


def _header_flag_is_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _infer_sdss_fiber_diameter_arcsec(
    specobj: fits.FITS_rec | None, *, fallback_diameter_arcsec: float,
) -> tuple[float, str]:
    if specobj is None or len(specobj) == 0 or specobj.columns is None:
        return float(fallback_diameter_arcsec), "config_default"

    row = specobj[0]
    columns = set(specobj.columns.names or [])

    def row_value(name: str) -> Any:
        if name not in columns:
            return None
        return row[name]

    # Legacy SDSS/SEGUE fibers are 3 arcsec. Require an explicit legacy-style marker
    # rather than treating "not BOSS" as legacy.
    if any(
        _header_flag_is_true(row_value(name))
        for name in ("SPECLEGACY", "SPECSEGUE", "SPECSEGUE1", "SPECSEGUE2")
    ):
        return 3.0, "specobj_legacy_flag"

    survey_text = (_header_value_as_text(row_value("SURVEY")) or "").lower()
    instrument_text = (_header_value_as_text(row_value("INSTRUMENT")) or "").lower()
    program_text = (_header_value_as_text(row_value("PROGRAMNAME")) or "").lower()
    if "legacy" in program_text or instrument_text == "sdss" or survey_text == "sdss":
        return 3.0, "specobj_legacy_metadata"

    if _header_flag_is_true(row_value("SPECBOSS")):
        return 2.0, "specobj_boss_flag"
    if "boss" in program_text or instrument_text == "boss" or "boss" in survey_text:
        return 2.0, "specobj_boss_metadata"

    return float(fallback_diameter_arcsec), "config_default"


def _resolve_bunit(cutout_header: fits.Header, frame_header: fits.Header | None) -> str | None:
    bunit = cutout_header.get("BUNIT")
    if bunit is None and frame_header is not None:
        bunit = frame_header.get("BUNIT")
    if isinstance(bunit, bytes):
        bunit = bunit.decode("utf-8")
    return str(bunit).strip().lower() if bunit is not None else None


def _is_nanomaggy_unit(bunit: str | None) -> bool:
    if bunit is None:
        return False
    return "nanomag" in bunit


def _interp_1d(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    values_1d = np.asarray(values, dtype=np.float64)
    if values_1d.ndim != 1 or values_1d.size == 0:
        raise ValueError("Interpolation source must be a non-empty 1D array.")
    x_arr = np.asarray(x, dtype=np.float64)
    x_eval = np.array(x_arr, copy=True)
    base = np.arange(values_1d.size, dtype=np.float64)
    x_lo = float(base[0])
    x_hi = float(base[-1])
    tol = 1.0e-6
    out_of_bounds = (x_arr < (x_lo - tol)) | (x_arr > (x_hi + tol))
    x_eval = np.clip(x_eval, x_lo, x_hi)
    out = np.interp(x_eval, base, values_1d, left=np.nan, right=np.nan)
    out[out_of_bounds] = np.nan
    return out


def _bilinear_sample_2d(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    grid_2d = np.asarray(grid, dtype=np.float64)
    if grid_2d.ndim != 2:
        raise ValueError("Bilinear interpolation grid must be 2D.")
    h, w = grid_2d.shape
    if h == 0 or w == 0:
        raise ValueError("Bilinear interpolation grid must be non-empty.")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("Bilinear interpolation coordinates must have the same shape.")

    x0 = np.floor(x_arr).astype(np.int64)
    y0 = np.floor(y_arr).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    ia = grid_2d[y0c, x0c]
    ib = grid_2d[y1c, x0c]
    ic = grid_2d[y0c, x1c]
    id_ = grid_2d[y1c, x1c]

    wa = (x1 - x_arr) * (y1 - y_arr)
    wb = (x1 - x_arr) * (y_arr - y0)
    wc = (x_arr - x0) * (y1 - y_arr)
    wd = (x_arr - x0) * (y_arr - y0)
    return ia * wa + ib * wb + ic * wc + id_ * wd


def _compute_cutout_sky_nanomaggy(
    cutout_header: fits.Header, cutout_shape: tuple[int, int], frame_path: Path | None,
) -> np.ndarray:
    if frame_path is None or not frame_path.exists():
        raise FileNotFoundError(
            "The additive sky contract requires image_exposures frame_path"
            " entries pointing to existing SDSS frame files."
        )
    if len(cutout_shape) != 2:
        raise ValueError("SDSS cutout must be a 2D image to reconstruct sky background.")
    cutout_h, cutout_w = int(cutout_shape[0]), int(cutout_shape[1])
    if cutout_h <= 0 or cutout_w <= 0:
        raise ValueError("SDSS cutout has invalid shape for sky reconstruction.")

    with fits.open(frame_path) as hdul:
        if len(hdul) < 3:
            raise ValueError(
                f"SDSS frame {frame_path} is missing required HDUs"
                " (need primary, HDU1 calib, HDU2 sky table)."
            )
        frame_data = np.asarray(hdul[0].data, dtype=np.float64)
        if frame_data.ndim != 2:
            raise ValueError(f"SDSS frame {frame_path} primary image is not 2D.")
        frame_header = hdul[0].header.copy()

        calib = np.asarray(hdul[1].data, dtype=np.float64)
        if calib.ndim != 1:
            raise ValueError(f"SDSS frame {frame_path} HDU1 calibration vector is not 1D.")
        if calib.size != frame_data.shape[1]:
            raise ValueError(
                f"SDSS frame {frame_path} HDU1 calibration length {calib.size}"
                f" does not match frame width {frame_data.shape[1]}."
            )

        sky_table = hdul[2].data
        if sky_table is None:
            raise ValueError(f"SDSS frame {frame_path} HDU2 sky table is empty.")
        if not hasattr(sky_table, "columns") or sky_table.columns is None:
            raise ValueError(f"SDSS frame {frame_path} HDU2 does not contain a valid sky table.")
        columns = set(sky_table.columns.names or [])
        required = {"ALLSKY", "XINTERP", "YINTERP"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"SDSS frame {frame_path} HDU2 is missing required sky columns: {missing}.")

        allsky = np.asarray(sky_table["ALLSKY"][0], dtype=np.float64)
        xinterp = np.asarray(sky_table["XINTERP"][0], dtype=np.float64)
        yinterp = np.asarray(sky_table["YINTERP"][0], dtype=np.float64)

    if allsky.ndim != 2:
        raise ValueError(f"SDSS frame {frame_path} ALLSKY must be 2D.")
    if xinterp.ndim != 1 or yinterp.ndim != 1:
        raise ValueError(f"SDSS frame {frame_path} XINTERP/YINTERP must be 1D.")
    if xinterp.size != frame_data.shape[1]:
        raise ValueError(
            f"SDSS frame {frame_path} XINTERP length {xinterp.size}"
            f" does not match frame width {frame_data.shape[1]}."
        )
    if yinterp.size != frame_data.shape[0]:
        raise ValueError(
            f"SDSS frame {frame_path} YINTERP length {yinterp.size}"
            f" does not match frame height {frame_data.shape[0]}."
        )

    cutout_wcs = WCS(_sanitize_header_for_wcs(cutout_header))
    frame_wcs = WCS(_sanitize_header_for_wcs(frame_header))

    yy, xx = np.meshgrid(
        np.arange(cutout_h, dtype=np.float64), np.arange(cutout_w, dtype=np.float64), indexing="ij",
    )
    ra_deg, dec_deg = cutout_wcs.all_pix2world(xx, yy, 0)
    frame_x, frame_y = frame_wcs.all_world2pix(ra_deg, dec_deg, 0)
    finite_frame_xy = np.isfinite(frame_x) & np.isfinite(frame_y)
    if not np.any(finite_frame_xy):
        raise ValueError(f"Sky reconstruction for {frame_path} produced non-finite frame pixel coordinates.")

    allsky_x = _interp_1d(xinterp, frame_x)
    allsky_y = _interp_1d(yinterp, frame_y)
    calib_px = _interp_1d(calib, frame_x)
    finite_sky_xy = finite_frame_xy & np.isfinite(allsky_x) & np.isfinite(allsky_y)
    if not np.any(finite_sky_xy):
        raise ValueError(
            f"Sky reconstruction for {frame_path} could not map cutout pixels onto ALLSKY coordinates."
        )
    if not np.any(np.isfinite(calib_px)):
        raise ValueError(f"Sky reconstruction for {frame_path} produced non-finite calibration values.")

    sky_counts = _bilinear_sample_2d(
        allsky, np.where(np.isfinite(allsky_x), allsky_x, 0.0),
        np.where(np.isfinite(allsky_y), allsky_y, 0.0),
    )
    valid = finite_sky_xy & np.isfinite(calib_px) & np.isfinite(sky_counts)
    if not np.any(valid):
        raise ValueError(f"Sky reconstruction for {frame_path} produced non-finite sky counts.")
    sky_nanomaggy = np.full(cutout_shape, np.nan, dtype=np.float64)
    sky_nanomaggy[valid] = sky_counts[valid] * calib_px[valid]
    return sky_nanomaggy


def _compute_poisson_sigma_nanomaggy(
    total_image_nanomaggy: np.ndarray, cutout_header: fits.Header, frame_path: Path, gain: float,
    dark_variance: float,
) -> np.ndarray:
    """Per-pixel CCD noise sigma in nanomaggies from Poisson + read-noise model.

    var_DN = max(signal_DN, 0) / gain + dark_variance,
    where signal_DN is total signal (source + sky) in DN above bias.
    """
    gain = float(gain)
    dark_variance = float(dark_variance)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("Poisson sigma gain must be finite and > 0.")
    if not np.isfinite(dark_variance) or dark_variance < 0.0:
        raise ValueError("Poisson sigma dark_variance must be finite and >= 0.")
    if frame_path is None or not frame_path.exists():
        raise FileNotFoundError("Poisson sigma computation requires a frame file for the calibration vector.")
    cutout_h, cutout_w = total_image_nanomaggy.shape

    with fits.open(frame_path) as hdul:
        frame_header = hdul[0].header.copy()
        calib = np.asarray(hdul[1].data, dtype=np.float64)

    cutout_wcs = WCS(_sanitize_header_for_wcs(cutout_header))
    frame_wcs = WCS(_sanitize_header_for_wcs(frame_header))

    yy, xx = np.meshgrid(
        np.arange(cutout_h, dtype=np.float64), np.arange(cutout_w, dtype=np.float64), indexing="ij",
    )
    ra_deg, dec_deg = cutout_wcs.all_pix2world(xx, yy, 0)
    frame_x, _ = frame_wcs.all_world2pix(ra_deg, dec_deg, 0)

    calib_px = np.maximum(_interp_1d(calib, frame_x), 1e-30)

    signal_dn = np.asarray(total_image_nanomaggy, dtype=np.float64) / calib_px
    var_dn = np.maximum(signal_dn, 0.0) / gain + dark_variance

    return np.sqrt(var_dn) * calib_px


def _load_scene_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"scene manifest {path} must be a JSON object.")
    return data


def _manifest_image_exposure_specs(
    manifest: dict[str, Any], *, observation_root: Path | None,
) -> list[dict[str, Any]]:
    raw_items = manifest.get("image_exposures")
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise ValueError("scene manifest 'image_exposures' must be a list when provided.")
    specs: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"image_exposures[{idx}] must be a JSON object.")
        band = str(item.get("band", "")).strip()
        if not band:
            raise ValueError(f"image_exposures[{idx}] is missing band.")
        cutout_path = _resolve_manifest_entry_path(item.get("cutout_path"), observation_root=observation_root)
        frame_path = _resolve_manifest_entry_path(item.get("frame_path"), observation_root=observation_root)
        if cutout_path is None:
            raise ValueError(f"image_exposures[{idx}] is missing cutout_path.")
        if frame_path is None:
            raise ValueError(f"image_exposures[{idx}] is missing frame_path.")
        spec = dict(item)
        spec.update(
            {
                "label": str(item.get("label", f"{band}-exp{idx + 1:04d}")),
                "band": band,
                "cutout_path": str(cutout_path),
                "frame_path": str(frame_path),
            }
        )
        specs.append(spec)
    return specs


def _spectrum_specs_from_manifest(
    manifest: dict[str, Any], *, observation_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spectra = manifest.get("spectra")
    if not isinstance(spectra, list):
        raise ValueError("scene manifest must contain a 'spectra' list.")
    specs: list[dict[str, Any]] = []
    auxiliary_specs: list[dict[str, Any]] = []
    for idx, item in enumerate(spectra):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "science")).strip().lower()
        path = _resolve_manifest_entry_path(item.get("path"), observation_root=observation_root)
        if path is None:
            raise ValueError(f"spectrum entry {idx} is missing a valid path.")
        spec = {
            "label": str(item.get("label", f"S{len(specs) + 1}")),
            "role": role,
            "path": str(path),
            "center_ra_dec": (float(item["ra_deg"]), float(item["dec_deg"]))
            if "ra_deg" in item and "dec_deg" in item
            else None,
            "metadata": dict(item),
        }
        specs.append(spec)
        if role != "science":
            auxiliary_specs.append(spec)
    return specs, auxiliary_specs


def _normalize_optional_string_set(values: Any) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    normalized = {str(value).strip().lower() for value in raw_values if str(value).strip()}
    if not normalized or "all" in normalized or "*" in normalized:
        return None
    return normalized


def _spectrum_spec_is_selected(spec: dict[str, Any], selection_cfg: dict[str, Any]) -> bool:
    include_roles = _normalize_optional_string_set(selection_cfg.get("include_roles"))
    include_classes = _normalize_optional_string_set(selection_cfg.get("include_classes"))
    role = str(spec.get("role", "science")).strip().lower()
    metadata = dict(spec.get("metadata") or {})
    spec_class = str(metadata.get("class", metadata.get("CLASS", ""))).strip().lower()
    if include_roles is not None and role not in include_roles:
        return False
    if include_classes is not None and spec_class not in include_classes:
        return False
    return True


def _load_sdss_spectrum(
    spectrum_spec: dict[str, Any], *, min_ivar: float, require_clean_mask: bool, training_flux_scale: float,
    fiber_diameter_arcsec: float, reference_band: str, wcs_per_band: Dict[str, WCS],
    pixel_scale_arcsec_per_band: Dict[str, float], image_shape: tuple[int, int],
    model_wcs: TangentPlaneWCS | None = None, spectral_flux_conversion: dict[str, Any] | None = None,
) -> SdssSpectrumData:
    role = str(spectrum_spec.get("role", "science"))
    is_science_spectrum = role.strip().lower() == "science"
    spectrum_path = Path(str(spectrum_spec["path"]))
    with fits.open(spectrum_path) as hdul:
        spec_header = hdul[0].header
        coadd = hdul["COADD"].data
        specobj = None
        if "SPECOBJ" in hdul:
            specobj = hdul["SPECOBJ"].data
        elif "SPALL" in hdul:
            specobj = hdul["SPALL"].data
        loglam = np.asarray(coadd["loglam"], dtype=np.float64)
        native_spacing_header = spec_header.get("COEFF1")
        if native_spacing_header is not None:
            native_spacing_header = float(native_spacing_header)
            if not np.isfinite(native_spacing_header) or native_spacing_header <= 0:
                raise ValueError("COEFF1 must declare a positive finite log-wavelength spacing.")
            # Stored float32 loglam quantizes each step; tolerate its roundoff,
            # but reject a header whose declared grid disagrees with the data.
            if loglam.size > 1 and not np.allclose(np.diff(loglam), native_spacing_header,
                                                  rtol=0, atol=2e-6):
                raise ValueError("COEFF1 disagrees with the native log-wavelength grid.")
        observed_flux = np.asarray(coadd["flux"], dtype=np.float64)
        columns = set(coadd.columns.names or [])
        response_record = dict(spectrum_spec.get("metadata") or {}).get("spectral_response_metadata", {})
        native_identity = {"plate": spec_header.get("PLATEID"), "mjd": spec_header.get("MJD"),
                           "vers2d": spec_header.get("VERS2D")}
        if specobj is not None and "FIBERID" in (specobj.columns.names or []):
            native_identity["fiberid"] = int(specobj["FIBERID"][0])
        for key, native_value in native_identity.items():
            declared = response_record.get(key)
            if declared is not None and native_value is not None and str(declared).strip() != str(native_value).strip():
                raise ValueError(f"Stale spectral response metadata {key}: {declared!r} != FITS {native_value!r}.")
        if is_science_spectrum:
            if "sky" not in columns:
                raise ValueError(
                    "The additive sky contract requires a COADD 'sky' column in science spectra."
                )
            source_flux = observed_flux.copy()
            additive_flux = np.asarray(coadd["sky"], dtype=np.float64)
        else:
            source_flux = np.zeros_like(observed_flux, dtype=np.float64)
            additive_flux = observed_flux.copy()
        target_flux = source_flux + additive_flux
        total_flux = target_flux.copy()
        ivar = np.asarray(coadd["ivar"], dtype=np.float64)
        wdisp = np.asarray(coadd["wdisp"], dtype=np.float64)
        and_mask = np.asarray(coadd["and_mask"], dtype=np.int64)
        or_mask = np.asarray(coadd["or_mask"], dtype=np.int64)

        center_radec = spectrum_spec.get("center_ra_dec")
        if specobj is not None and "PLUG_RA" in specobj.columns.names:
            fiber_ra = float(specobj["PLUG_RA"][0])
            fiber_dec = float(specobj["PLUG_DEC"][0])
        elif center_radec is not None:
            fiber_ra = float(center_radec[0])
            fiber_dec = float(center_radec[1])
        else:
            fiber_ra = float(spec_header.get("RA"))
            fiber_dec = float(spec_header.get("DEC"))
        fiber_diameter_arcsec, fiber_diameter_source = _infer_sdss_fiber_diameter_arcsec(
            specobj, fallback_diameter_arcsec=fiber_diameter_arcsec,
        )

    native_dlog10_lambda = (native_spacing_header if native_spacing_header is not None else
                           float(np.median(np.diff(loglam))) if loglam.size > 1 else None)
    native_spacing_source = "FITS COEFF1" if native_spacing_header is not None else "inferred unmasked loglam median"
    valid = (
        np.isfinite(loglam)
        & np.isfinite(target_flux)
        & np.isfinite(source_flux)
        & np.isfinite(additive_flux)
        & np.isfinite(total_flux)
        & (ivar > min_ivar)
    )
    if require_clean_mask:
        valid &= (and_mask == 0) & (or_mask == 0)
    lam_edges = masked_bin_edges(10.0 ** loglam, valid)
    loglam = loglam[valid]
    target_flux = target_flux[valid]
    source_flux = source_flux[valid]
    additive_flux = additive_flux[valid]
    total_flux = total_flux[valid]
    ivar = ivar[valid]
    wdisp = wdisp[valid]
    if loglam.size == 0:
        raise NoValidSpectrumBinsError(f"No valid spectral bins after masking for {spectrum_path}.")
    if spectral_flux_conversion is not None:
        ratio = spectrum_to_image_ratio(
            10.0 ** loglam, spectral_flux_conversion["bandpasses"], spectral_flux_conversion["knots"],
        )
        target_flux = target_flux / ratio
        source_flux = source_flux / ratio
        additive_flux = additive_flux / ratio
        total_flux = total_flux / ratio
        ivar = ivar * ratio ** 2

    lam_spec = 10.0 ** loglam
    target_cgs = target_flux * 1e-17
    source_cgs = source_flux * 1e-17
    additive_cgs = additive_flux * 1e-17
    total_cgs = total_flux * 1e-17
    sigma_cgs = np.zeros_like(target_cgs)
    sigma_cgs[ivar > 0] = (1.0 / np.sqrt(ivar[ivar > 0])) * 1e-17

    fiber_radius_arcsec = 0.5 * fiber_diameter_arcsec
    fiber_area_arcsec2 = float(np.pi * fiber_radius_arcsec ** 2)
    flux_aperture = target_cgs * training_flux_scale
    source_aperture = source_cgs * training_flux_scale
    additive_aperture = additive_cgs * training_flux_scale
    total_aperture = total_cgs * training_flux_scale
    sigma_aperture = sigma_cgs * training_flux_scale

    reference_wcs: WCS | TangentPlaneWCS = (
        model_wcs if model_wcs is not None else wcs_per_band[reference_band]
    )
    pixel_scale_arcsec = (
        float(model_wcs.pixel_scale_arcsec)
        if model_wcs is not None
        else float(pixel_scale_arcsec_per_band[reference_band])
    )
    fiber_x_pix, fiber_y_pix = _world_to_pixel(reference_wcs, fiber_ra, fiber_dec)
    if not np.isfinite(fiber_x_pix) or not np.isfinite(fiber_y_pix):
        raise ValueError(
            f"Fiber center projects to non-finite pixel coordinates in"
            f" reference band WCS for {spectrum_path}."
        )

    from .grids import build_fiber_mask

    height, width = int(image_shape[0]), int(image_shape[1])
    fiber_mask = build_fiber_mask(
        height, width, pixel_scale_arcsec, fiber_diameter_arcsec, center_x=fiber_x_pix, center_y=fiber_y_pix,
    )

    metadata = dict(spectrum_spec.get("metadata") or {})
    metadata.update(
        {
            "native_dlog10_lambda": native_dlog10_lambda,
            "native_dlog10_lambda_source": native_spacing_source,
            "native_spectrum_identity": native_identity,
            "fiber_center_pixel_ref": [fiber_x_pix, fiber_y_pix],
            "fiber_center_pixel_ref_band": "model_wcs" if model_wcs is not None else reference_band,
            "fiber_diameter_arcsec": float(fiber_diameter_arcsec), "fiber_area_arcsec2": fiber_area_arcsec2,
            "fiber_diameter_source": str(fiber_diameter_source), "source_path": str(spectrum_path),
        }
    )

    return SdssSpectrumData(
        label=str(spectrum_spec.get("label", "fiber")), role=str(spectrum_spec.get("role", "science")),
        path=str(spectrum_path), lam_spec=lam_spec.astype(np.float64),
        flux_spec_flambda=flux_aperture.astype(np.float32),
        flux_spec_flam_lsf=flux_aperture.astype(np.float32),
        spec_throughput=np.ones_like(lam_spec, dtype=np.float32), fiber_mask=fiber_mask,
        spec_ivar=ivar.astype(np.float64), spec_wdisp=wdisp.astype(np.float64),
        spec_loglam=loglam.astype(np.float64),
        sigma_spec_flambda=sigma_aperture.astype(np.float32),
        center_ra_dec=(float(fiber_ra), float(fiber_dec)), fiber_radius_arcsec=fiber_radius_arcsec,
        metadata=metadata, spec_source_flambda=source_aperture.astype(np.float32),
        spec_additive_flambda=additive_aperture.astype(np.float32),
        spec_total_flambda=total_aperture.astype(np.float32),
        lam_edges=lam_edges,
    )


def _load_sdss_image_exposure(
    spec: dict[str, Any], *, conversion: dict[str, float], training_flux_scale: float,
) -> SdssImageExposureData:
    band = str(spec["band"])
    if band not in conversion:
        raise ValueError(f"Image exposure {spec.get('label', band)!r} uses unknown band {band!r}.")
    cutout_path = Path(str(spec["cutout_path"]))
    frame_path = Path(str(spec["frame_path"]))
    if not cutout_path.exists():
        raise FileNotFoundError(f"Missing SDSS image exposure cutout: {cutout_path}")
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing SDSS image exposure frame: {frame_path}")

    frame_header = _read_header(frame_path)
    with fits.open(cutout_path) as hdul:
        cutout_header = hdul[0].header.copy()
        datum_image_nanomaggy = np.asarray(hdul[0].data, dtype=np.float64)
    bunit = _resolve_bunit(cutout_header, frame_header)
    if not _is_nanomaggy_unit(bunit):
        raise ValueError(f"SDSS cutout {cutout_path} missing nanomaggy BUNIT; got {bunit!r}.")
    wcs, pixel_scale_arcsec, pixel_area = _load_wcs_and_scale(cutout_header)

    known_sky_nanomaggy = _compute_cutout_sky_nanomaggy(
        cutout_header=cutout_header, cutout_shape=datum_image_nanomaggy.shape, frame_path=frame_path,
    )
    source_image_nanomaggy = datum_image_nanomaggy
    additive_image_nanomaggy = known_sky_nanomaggy
    total_image_nanomaggy = source_image_nanomaggy + known_sky_nanomaggy
    target_image_nanomaggy = source_image_nanomaggy + additive_image_nanomaggy

    gain = spec.get("gain")
    dark_variance = spec.get("dark_variance")
    if gain is None or dark_variance is None:
        raise ValueError(
            f"Image exposure {spec.get('label', band)!r} requires real gain and dark_variance metadata."
        )
    nmgy_to_training = (FNU_PER_NANOMAGGY * conversion[band] / max(pixel_area, 1e-20) * training_flux_scale)
    sigma_nmgy = _compute_poisson_sigma_nanomaggy(
        total_image_nanomaggy, cutout_header, frame_path, float(gain), float(dark_variance),
    )
    sigma_model = spec.get("image_sigma_model")
    if sigma_model is not None:
        # Poisson variance of the expected counts: a previous fit's source
        # model plus the known sky, never the pixel's own noisy count. Where
        # that model misses the data by more than outlier_pull data sigmas
        # (trails, ghosts), the model's small variance would hand an unmodelled
        # feature the fit, so those pixels keep the data variance.
        model_nanomaggy = _load_image_sigma_model(sigma_model, datum_image_nanomaggy.shape) / nmgy_to_training
        model_sigma_nmgy = _compute_poisson_sigma_nanomaggy(
            model_nanomaggy + known_sky_nanomaggy, cutout_header, frame_path, float(gain), float(dark_variance),
        )
        outlier = np.abs(source_image_nanomaggy - model_nanomaggy) > float(sigma_model["outlier_pull"]) * sigma_nmgy
        sigma_nmgy = np.where(outlier, sigma_nmgy, model_sigma_nmgy)
    psf_value = spec.get("psf_fwhm_arcsec")
    if psf_value is None:
        raise ValueError(
            f"Image exposure {spec.get('label', band)!r} requires real psf_fwhm_arcsec metadata."
        )
    def mixture(key: str) -> PSFMixtureComponents | None:
        raw = spec.get(key)
        if raw is None:
            return None
        try:
            return PSFMixtureComponents.from_rows(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Image exposure {spec.get('label', band)!r} has invalid {key}: {exc}"
            ) from exc

    def effective_fwhm(key: str) -> float | None:
        value = spec.get(key)
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Image exposure {spec.get('label', band)!r} has invalid "
                f"{key}={value!r}."
            )
        return value

    effective_fwhm_legacy = effective_fwhm("psf_effective_fwhm_arcsec")
    effective_fwhm_dg = effective_fwhm("psf_effective_fwhm_dg_arcsec")
    effective_fwhm_kl = effective_fwhm("psf_effective_fwhm_kl_arcsec")
    return SdssImageExposureData(
        label=str(spec.get("label", f"{band}-exposure")), band=band, cutout_path=str(cutout_path),
        frame_path=str(frame_path),
        source_flambda=(source_image_nanomaggy * nmgy_to_training).astype(np.float32),
        additive_flambda=(additive_image_nanomaggy * nmgy_to_training).astype(np.float32),
        total_flambda=(total_image_nanomaggy * nmgy_to_training).astype(np.float32),
        target_flambda=(target_image_nanomaggy * nmgy_to_training).astype(np.float32),
        sigma_flambda=(sigma_nmgy * nmgy_to_training).astype(np.float32), wcs=wcs,
        pixel_scale_arcsec=float(pixel_scale_arcsec), pixel_area_arcsec2=float(pixel_area),
        psf_fwhm_arcsec=float(psf_value), metadata=dict(spec),
        psf_mixture_dgpsf=mixture("psf_mixture_dgpsf"),
        psf_mixture_kl=mixture("psf_mixture_kl"),
        psf_effective_fwhm_arcsec=effective_fwhm_legacy,
        psf_effective_fwhm_dg_arcsec=effective_fwhm_dg,
        psf_effective_fwhm_kl_arcsec=effective_fwhm_kl,
    )


def _load_image_sigma_model(record: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    path = Path(str(record["path"]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Image sigma model {path} does not match its manifest sha256.")
    model = np.load(path).astype(np.float64)
    if model.shape != tuple(shape) or not np.isfinite(model).all():
        raise ValueError(f"Image sigma model {path} has shape {model.shape} for a {tuple(shape)} cutout or is not finite.")
    return model


def _image_sigma_model_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("units") != "training_source_flambda":
        raise ValueError(f"Image sigma model manifest {manifest_path} must hold training_source_flambda images.")
    return {
        str(label): {"path": str(Path(manifest_path).parent / record["path"]), "sha256": record["sha256"]}
        for label, record in manifest["exposures"].items()
    }


def _default_loader_workers() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(32, available))


def _parallel_map(fn, tasks: list, n_workers: int, chunksize: int) -> list:
    """Order-preserving map over independent file-load tasks.

    Sequential for small task lists: spawn startup costs seconds, which loses
    on the synthetic scenes used in tests and tiny probes.
    """
    if n_workers <= 1 or len(tasks) < 8:
        return [fn(t) for t in tasks]
    # spawn, not fork: the trainer may hold a CUDA context by now. spawn
    # re-imports __main__ in each worker, so every entry point that reaches
    # this pool must be import-safe (body under `if __name__ == "__main__"`).
    with ProcessPoolExecutor(
        max_workers=min(n_workers, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(fn, tasks, chunksize=chunksize))


def _spectrum_load_task(task: tuple) -> tuple:
    """Load one spectrum in a worker; skips must travel back as values, not
    exceptions, so one bad fiber cannot abort the whole pool map."""
    spec, kwargs = task
    try:
        return ("ok", _load_sdss_spectrum(spec, **kwargs))
    except NoValidSpectrumBinsError as exc:
        return (
            "skip",
            {
                "label": str(spec.get("label", "fiber")), "role": str(spec.get("role", "science")),
                "path": str(spec.get("path")), "reason": str(exc),
            },
        )


def _image_load_task(task: tuple) -> "SdssImageExposureData":
    spec, conversion, training_flux_scale = task
    return _load_sdss_image_exposure(
        spec, conversion=conversion, training_flux_scale=training_flux_scale,
    )


def load_sdss_scene(cfg, bandpasses: Dict[str, Bandpass]) -> SdssSceneData:
    observation_root = _as_path(cfg.get("observation_root"))
    scene_manifest_path = _as_path(cfg.get("scene_manifest_path"), observation_root)
    if scene_manifest_path is None:
        raise ValueError("sdss config must include scene_manifest_path.")
    manifest = _load_scene_manifest(scene_manifest_path)
    declared_psf_mixture_digest = manifest.get("psf_mixture_digest")
    digest_version = None
    if declared_psf_mixture_digest is not None:
        declared_psf_mixture_digest = str(declared_psf_mixture_digest).lower()
        if len(declared_psf_mixture_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in declared_psf_mixture_digest
        ):
            raise ValueError(
                "scene manifest psf_mixture_digest must be a 64-character "
                "SHA-256 hex digest."
            )
        digest_version = manifest.get("psf_mixture_digest_version", 1)
        if (
            isinstance(digest_version, bool)
            or not isinstance(digest_version, int)
            or digest_version not in (1, 2)
        ):
            raise ValueError("scene manifest psf_mixture_digest_version must be 1 or 2.")
        if digest_version == 1:
            warnings.warn(
                "scene manifest PSF mixture digest predates QA and pixel-response "
                "convention binding; accepting legacy array-only digest",
                RuntimeWarning,
                stacklevel=2,
            )
        recomputed_digest = recompute_psf_mixture_digest(
            manifest.get("image_exposures") or [], version=digest_version,
        )
        if declared_psf_mixture_digest != recomputed_digest:
            raise ValueError(
                "scene manifest psf_mixture_digest does not match the loaded "
                "mixture arrays and provenance payload: declared "
                f"{declared_psf_mixture_digest}, "
                f"recomputed {recomputed_digest}."
            )
    manifest_image_exposure_specs = _manifest_image_exposure_specs(
        manifest, observation_root=observation_root,
    )
    if not manifest_image_exposure_specs:
        raise ValueError("scene manifest must contain a non-empty 'image_exposures' list.")
    sigma_model_path = _as_path(cfg.get("image_sigma_model_path"))
    if sigma_model_path is not None:
        outlier_pull = cfg.get("image_sigma_model_outlier_pull")
        if outlier_pull is None or not float(outlier_pull) > 0:
            raise ValueError("data.image_sigma_model_path requires a positive data.image_sigma_model_outlier_pull.")
        sigma_models = _image_sigma_model_records(sigma_model_path)
        for spec in manifest_image_exposure_specs:
            label = str(spec.get("label"))
            if label not in sigma_models:
                raise ValueError(f"Image sigma model manifest {sigma_model_path} has no image for {label!r}.")
            spec["image_sigma_model"] = dict(sigma_models[label], outlier_pull=float(outlier_pull))
    model_wcs = _model_wcs_from_manifest(manifest)
    training_flux_scale = float(cfg.get("training_flux_scale", 1.0))
    fiber_diameter_arcsec = float(cfg.get("fiber_diameter_arcsec", 3.0))

    conversion = {band: _bandpass_fnu_to_flambda_factor(bp) for band, bp in bandpasses.items()}
    images_flambda: Dict[str, np.ndarray] = {}
    images: Dict[str, np.ndarray] = {}
    image_source_flambda: Dict[str, np.ndarray] = {}
    image_additive_flambda: Dict[str, np.ndarray] = {}
    image_total_flambda: Dict[str, np.ndarray] = {}
    wcs_per_band: Dict[str, WCS] = {}
    pixel_area_arcsec2: Dict[str, float] = {}
    pixel_scale_arcsec_per_band: Dict[str, float] = {}
    image_exposures: list[SdssImageExposureData] = []

    image_sigma_flambda: Dict[str, np.ndarray] | None = {}

    # FITS parsing is CPU-bound in astropy's column machinery (~0.1 s per
    # table), so 410 exposures + 910 spectra serially is the whole "20 minutes
    # to load". The files are independent; parse them in parallel. spawn, not
    # fork: the trainer may already hold a CUDA context when this runs.
    n_workers = int(cfg.get("loader_workers", 0) or 0) or _default_loader_workers()

    image_tasks = [
        (spec, conversion, training_flux_scale) for spec in manifest_image_exposure_specs
    ]
    for image_exp in _parallel_map(_image_load_task, image_tasks, n_workers, 4):
        image_exposures.append(image_exp)
        band = image_exp.band
        if band not in images_flambda:
            images_flambda[band] = image_exp.target_flambda
            image_source_flambda[band] = image_exp.source_flambda
            image_additive_flambda[band] = image_exp.additive_flambda
            image_total_flambda[band] = image_exp.total_flambda
            images[band] = image_exp.target_flambda
            wcs_per_band[band] = image_exp.wcs
            pixel_scale_arcsec_per_band[band] = image_exp.pixel_scale_arcsec
            pixel_area_arcsec2[band] = image_exp.pixel_area_arcsec2
            image_sigma_flambda[band] = image_exp.sigma_flambda

    min_ivar = float(cfg.get("spec_min_ivar", 0.0))
    require_clean_mask = bool(cfg.get("spec_require_clean_mask", True))
    spectral_flux_system = str(cfg.get("spectral_flux_system") or "pipeline")
    if spectral_flux_system not in SPECTRAL_FLUX_SYSTEMS:
        raise ValueError(
            f"data.spectral_flux_system must be one of {SPECTRAL_FLUX_SYSTEMS}, got {spectral_flux_system!r}."
        )
    psf_mixture_digest = declared_psf_mixture_digest
    metadata = {
        "fiber_diameter_arcsec": fiber_diameter_arcsec, "training_flux_scale": training_flux_scale,
        "spec_min_ivar": min_ivar, "spec_require_clean_mask": require_clean_mask, "auxiliary_spectra": [],
        "psf_mixture_digest": psf_mixture_digest,
        "psf_mixture_digest_version": digest_version,
    }
    if sigma_model_path is not None:
        metadata["image_sigma_source"] = {
            "manifest": str(sigma_model_path), "outlier_pull": float(cfg["image_sigma_model_outlier_pull"]),
            "manifest_sha256": hashlib.sha256(sigma_model_path.read_bytes()).hexdigest(),
        }
    spectral_flux_conversion = None
    if spectral_flux_system == "sdss_imaging":
        spectral_flux_conversion = {"bandpasses": dict(bandpasses), "knots": ratio_knots(bandpasses)}
        metadata["spectral_flux_system"] = {
            "system": spectral_flux_system, "ab_correction_mag": dict(IDLSPEC2D_AB_CORRECTION_MAG),
            "band_ratios": pipeline_band_ratios(), "knots": dict(spectral_flux_conversion["knots"]),
        }

    reference_band = str(cfg.get("reference_band", "r"))
    if reference_band not in pixel_scale_arcsec_per_band:
        reference_band = next(iter(pixel_scale_arcsec_per_band))
    pixel_scale_arcsec = pixel_scale_arcsec_per_band[reference_band]

    first_band = next(iter(images_flambda))
    image_shape = (
        (int(model_wcs.height), int(model_wcs.width))
        if model_wcs is not None
        else images_flambda[first_band].shape
    )
    spectrum_specs, auxiliary_spectrum_specs = _spectrum_specs_from_manifest(
        manifest, observation_root=observation_root,
    )
    response_path = _as_path(cfg.get("spectral_response_metadata_path"), observation_root)
    if response_path is not None:
        rows = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("spectral_response_metadata_path must contain a list keyed by label.")
        response_by_label = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("label"), str):
                raise ValueError("Spectral response metadata rows require a string label.")
            if row["label"] in response_by_label:
                raise ValueError(f"Duplicate spectral response metadata label {row['label']!r}.")
            response_by_label[row["label"]] = row
        for spec in spectrum_specs:
            row = response_by_label.get(spec["label"])
            if row is not None:
                existing = spec["metadata"].get("spatial_response")
                if existing is not None and existing != row.get("spatial_response"):
                    raise ValueError(f"Conflicting spectral response metadata for {spec['label']!r}.")
                spec["metadata"]["spatial_response"] = row.get("spatial_response")
                spec["metadata"]["spectral_response_metadata"] = row
                spec["metadata"]["spectral_response_metadata_path"] = str(response_path)
    selection_cfg = dict(cfg.get("spectra", None) or {})
    if selection_cfg:
        spectrum_specs = [spec for spec in spectrum_specs if _spectrum_spec_is_selected(spec, selection_cfg)]
        auxiliary_spectrum_specs = [
            spec for spec in auxiliary_spectrum_specs if _spectrum_spec_is_selected(spec, selection_cfg)
        ]
    metadata["auxiliary_spectra"] = [
        {
            "label": str(spec.get("label", "fiber")), "role": str(spec.get("role", "science")),
            "path": str(spec.get("path")),
        }
        for spec in auxiliary_spectrum_specs
    ]

    spectrum_kwargs = dict(
        min_ivar=min_ivar, require_clean_mask=require_clean_mask,
        training_flux_scale=training_flux_scale, fiber_diameter_arcsec=fiber_diameter_arcsec,
        reference_band=reference_band, wcs_per_band=wcs_per_band,
        pixel_scale_arcsec_per_band=pixel_scale_arcsec_per_band, image_shape=image_shape,
        model_wcs=model_wcs,
    )
    if spectral_flux_conversion is not None:
        spectrum_kwargs["spectral_flux_conversion"] = spectral_flux_conversion
    spectrum_tasks = [(spec, spectrum_kwargs) for spec in spectrum_specs]
    spectrum_results = _parallel_map(_spectrum_load_task, spectrum_tasks, n_workers, 8)
    spectra: list[SdssSpectrumData] = []
    skipped_spectra: list[dict[str, Any]] = []
    for status, payload in spectrum_results:
        if status == "ok":
            spectra.append(payload)
        else:
            skipped_spectra.append(payload)
    if not spectra:
        if skipped_spectra:
            labels = ", ".join(item["label"] for item in skipped_spectra)
            raise ValueError(
                "No science spectra with valid bins remain after masking. "
                f"Skipped spectra: {labels}."
            )
        if spectrum_specs:
            raise ValueError("No science spectra were loaded for the SDSS scene.")

        fiber_radius_arcsec = 0.5 * fiber_diameter_arcsec
        fiber_area_arcsec2 = float(np.pi * fiber_radius_arcsec * fiber_radius_arcsec)
        empty_flux = np.zeros((0,), dtype=np.float32)
        empty_meta = {
            **metadata, "fiber_center_pixel_ref": None, "fiber_center_pixel_ref_band": reference_band,
            "fiber_area_arcsec2": fiber_area_arcsec2, "spectra_count": 0, "skipped_spectra": skipped_spectra,
            "scene_manifest_path": str(scene_manifest_path) if scene_manifest_path is not None else None,
        }
        fiber_center_ra_dec = _image_center_radec(wcs_per_band[reference_band], image_shape)
        return SdssSceneData(
            images=images, images_flambda=images_flambda, spectra=[],
            lam_spec=np.zeros((0,), dtype=np.float64), flux_spec_flambda=empty_flux,
            flux_spec_flam_lsf=empty_flux, spec_throughput=empty_flux,
            fiber_mask=np.zeros(image_shape, dtype=np.float32), fiber_area_arcsec2=fiber_area_arcsec2,
            pixel_scale_arcsec=pixel_scale_arcsec, fiber_radius_arcsec=fiber_radius_arcsec,
            metadata=empty_meta, wcs_per_band=wcs_per_band,
            pixel_scale_arcsec_per_band=pixel_scale_arcsec_per_band, pixel_area_arcsec2=pixel_area_arcsec2,
            spec_ivar=np.zeros((0,), dtype=np.float64), spec_wdisp=np.zeros((0,), dtype=np.float64),
            spec_loglam=np.zeros((0,), dtype=np.float64), sigma_spec_flambda=empty_flux,
            fiber_center_ra_dec=fiber_center_ra_dec, image_sigma_flambda=image_sigma_flambda,
            image_source_flambda=image_source_flambda, image_additive_flambda=image_additive_flambda,
            image_total_flambda=image_total_flambda, spec_source_flambda=empty_flux,
            spec_additive_flambda=empty_flux, spec_total_flambda=empty_flux, image_exposures=image_exposures,
            model_wcs=model_wcs,
        )
    primary_spectrum = spectra[0]
    # scene.fiber_mask is a whole-grid array used only as a fallback for image
    # exposures; keep the sparse aperture and densify just this one.
    fiber_mask = primary_spectrum.fiber_mask
    if hasattr(fiber_mask, "to_dense"):
        fiber_mask = fiber_mask.to_dense()
    fiber_mask = np.asarray(fiber_mask, dtype=np.float32)
    fiber_ra, fiber_dec = primary_spectrum.center_ra_dec
    fiber_radius_arcsec = float(primary_spectrum.fiber_radius_arcsec)
    fiber_area_arcsec2 = float(primary_spectrum.metadata["fiber_area_arcsec2"])
    metadata["fiber_center_pixel_ref"] = primary_spectrum.metadata["fiber_center_pixel_ref"]
    metadata["fiber_center_pixel_ref_band"] = primary_spectrum.metadata["fiber_center_pixel_ref_band"]
    metadata["fiber_area_arcsec2"] = fiber_area_arcsec2
    metadata["spectra_count"] = int(len(spectra))
    metadata["skipped_spectra"] = skipped_spectra
    metadata["scene_manifest_path"] = str(scene_manifest_path) if scene_manifest_path is not None else None

    return SdssSceneData(
        images=images,
        images_flambda=images_flambda,
        spectra=spectra,
        lam_spec=np.asarray(primary_spectrum.lam_spec, dtype=np.float64),
        flux_spec_flambda=np.asarray(primary_spectrum.flux_spec_flambda, dtype=np.float32),
        flux_spec_flam_lsf=np.asarray(primary_spectrum.flux_spec_flam_lsf, dtype=np.float32),
        spec_throughput=np.asarray(primary_spectrum.spec_throughput, dtype=np.float32),
        fiber_mask=fiber_mask,
        fiber_area_arcsec2=fiber_area_arcsec2,
        pixel_scale_arcsec=pixel_scale_arcsec,
        fiber_radius_arcsec=fiber_radius_arcsec,
        metadata=metadata,
        wcs_per_band=wcs_per_band,
        pixel_scale_arcsec_per_band=pixel_scale_arcsec_per_band,
        pixel_area_arcsec2=pixel_area_arcsec2,
        spec_ivar=np.asarray(primary_spectrum.spec_ivar, dtype=np.float64),
        spec_wdisp=np.asarray(primary_spectrum.spec_wdisp, dtype=np.float64),
        spec_loglam=np.asarray(primary_spectrum.spec_loglam, dtype=np.float64),
        sigma_spec_flambda=np.asarray(primary_spectrum.sigma_spec_flambda, dtype=np.float32),
        fiber_center_ra_dec=(fiber_ra, fiber_dec),
        image_sigma_flambda=image_sigma_flambda,
        image_source_flambda=image_source_flambda,
        image_additive_flambda=image_additive_flambda,
        image_total_flambda=image_total_flambda,
        spec_source_flambda=np.asarray(
            primary_spectrum.spec_source_flambda
            if primary_spectrum.spec_source_flambda is not None
            else primary_spectrum.flux_spec_flambda,
            dtype=np.float32,
        ),
        spec_additive_flambda=np.asarray(
            primary_spectrum.spec_additive_flambda
            if primary_spectrum.spec_additive_flambda is not None
            else np.zeros_like(primary_spectrum.flux_spec_flambda),
            dtype=np.float32,
        ),
        spec_total_flambda=np.asarray(
            primary_spectrum.spec_total_flambda
            if primary_spectrum.spec_total_flambda is not None
            else primary_spectrum.flux_spec_flambda,
            dtype=np.float32,
        ),
        image_exposures=image_exposures,
        model_wcs=model_wcs,
    )
