from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable

import numpy as np

from ..utils.wcs import PixelToOmegaWCS, radec_deg_to_omega_xyz


@dataclass(frozen=True, slots=True)
class SkyCircleRegion:
    name: str
    center_ra_deg: float
    center_dec_deg: float
    radius_arcsec: float


def normalize_pointing_offsets(
    pointing_offsets_arcsec: object, bands: Iterable[str],
) -> Dict[str, tuple[float, float]]:
    band_list = [str(band) for band in bands]
    offsets = {band: (0.0, 0.0) for band in band_list}
    if pointing_offsets_arcsec is None:
        return offsets
    if not isinstance(pointing_offsets_arcsec, dict):
        raise ValueError("pointing_offsets_arcsec must be a mapping {band: [dx, dy]}.")
    for band in band_list:
        value = pointing_offsets_arcsec.get(band, None)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"pointing_offsets_arcsec[{band!r}] must be a length-2 [dx_arcsec, dy_arcsec] sequence."
            )
        offsets[band] = (float(value[0]), float(value[1]))
    return offsets


def _parse_center(center_raw: object, *, key: str) -> tuple[float, float]:
    if not isinstance(center_raw, (list, tuple)) or len(center_raw) != 2:
        raise ValueError(f"{key} must be a length-2 [ra_deg, dec_deg] sequence.")
    ra_deg = float(center_raw[0])
    dec_deg = float(center_raw[1])
    if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        raise ValueError(f"{key} must contain finite numeric coordinates.")
    return ra_deg, dec_deg


def _parse_radius(radius_raw: object, *, key: str) -> float:
    radius_arcsec = float(radius_raw)
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
        raise ValueError(f"{key} must be a positive finite value in arcsec.")
    return radius_arcsec


def parse_holdout_region_config(cfg: object) -> Dict[str, object]:
    cfg_map = dict(cfg or {})
    enabled = bool(cfg_map.get("enabled", False))
    exclude_spectra = bool(cfg_map.get("exclude_spectra_in_region", False))
    if not enabled:
        return {"enabled": False, "region": None, "exclude_spectra_in_region": exclude_spectra}
    ra_deg, dec_deg = _parse_center(cfg_map.get("center_radec_deg"), key="holdout_region.center_radec_deg")
    radius_arcsec = _parse_radius(cfg_map.get("radius_arcsec", 0.0), key="holdout_region.radius_arcsec")
    region = SkyCircleRegion(
        name=str(cfg_map.get("name", "holdout")), center_ra_deg=ra_deg, center_dec_deg=dec_deg,
        radius_arcsec=radius_arcsec,
    )
    return {"enabled": True, "region": region, "exclude_spectra_in_region": exclude_spectra}


def parse_region_metrics_config(cfg: object) -> Dict[str, object]:
    cfg_map = dict(cfg or {})
    enabled = bool(cfg_map.get("enabled", False))
    if not enabled:
        return {"enabled": False, "regions": [], "primary_region": None}
    regions_raw = cfg_map.get("regions", [])
    if not isinstance(regions_raw, list):
        raise ValueError("region_metrics.regions must be a list when enabled=true.")
    regions: list[SkyCircleRegion] = []
    seen_names: set[str] = set()
    for idx, region_raw in enumerate(regions_raw):
        if not isinstance(region_raw, dict):
            raise ValueError(
                f"region_metrics.regions[{idx}] must be a mapping with name/center_radec_deg/radius_arcsec."
            )
        name = str(region_raw.get("name", f"region{idx}"))
        if not name:
            raise ValueError(f"region_metrics.regions[{idx}].name must be non-empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate region_metrics region name {name!r}.")
        seen_names.add(name)
        ra_deg, dec_deg = _parse_center(
            region_raw.get("center_radec_deg"), key=f"region_metrics.regions[{idx}].center_radec_deg",
        )
        radius_arcsec = _parse_radius(
            region_raw.get("radius_arcsec", 0.0), key=f"region_metrics.regions[{idx}].radius_arcsec",
        )
        regions.append(
            SkyCircleRegion(
                name=name, center_ra_deg=ra_deg, center_dec_deg=dec_deg, radius_arcsec=radius_arcsec,
            )
        )
    primary_region_raw = cfg_map.get("primary_region", None)
    primary_region = str(primary_region_raw) if primary_region_raw is not None else None
    if not regions and primary_region is not None:
        raise ValueError(
            "region_metrics.primary_region must be null when no explicit regions are configured."
        )
    if primary_region is not None and primary_region not in seen_names:
        raise ValueError("region_metrics.primary_region must match one of region_metrics.regions[].name.")
    return {"enabled": True, "regions": regions, "primary_region": primary_region}


def sky_circle_mask_from_omega_wcs(
    obs_wcs: PixelToOmegaWCS, image_shape: tuple[int, int], region: SkyCircleRegion,
) -> np.ndarray:
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width).")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive.")
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij",
    )
    omega = np.asarray(obs_wcs.pixel_to_omega_xyz(xx, yy), dtype=np.float64)
    if omega.shape != (height, width, 3):
        raise ValueError("obs_wcs.pixel_to_omega_xyz must return shape [height, width, 3] for 2D inputs.")
    center = radec_deg_to_omega_xyz(
        np.array(region.center_ra_deg, dtype=np.float64), np.array(region.center_dec_deg, dtype=np.float64),
    )
    dot = np.einsum("ijk,k->ij", omega, center)
    dot = np.clip(dot, -1.0, 1.0)
    sep_rad = np.arccos(dot)
    radius_rad = float(region.radius_arcsec) * np.pi / (180.0 * 3600.0)
    return sep_rad <= radius_rad


def local_sky_circle_mask_from_omega_wcs(
    obs_wcs: PixelToOmegaWCS, image_shape: tuple[int, int], region: SkyCircleRegion, *,
    pixel_scale_arcsec: float,
) -> np.ndarray | None:
    raw_wcs = getattr(obs_wcs, "wcs", None)
    if raw_wcs is None or not hasattr(raw_wcs, "all_world2pix"):
        return None
    if not np.isfinite(pixel_scale_arcsec) or float(pixel_scale_arcsec) <= 0.0:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    ra_for_inverse = float(region.center_ra_deg)
    dec_for_inverse = float(region.center_dec_deg)
    dx_arcsec = float(getattr(obs_wcs, "dx_arcsec", 0.0) or 0.0)
    dy_arcsec = float(getattr(obs_wcs, "dy_arcsec", 0.0) or 0.0)
    if dx_arcsec != 0.0 or dy_arcsec != 0.0:
        cos_dec = max(float(np.cos(np.deg2rad(dec_for_inverse))), 1e-12)
        ra_for_inverse -= dx_arcsec / (3600.0 * cos_dec)
        dec_for_inverse -= dy_arcsec / 3600.0
    try:
        x_center, y_center = raw_wcs.all_world2pix(ra_for_inverse, dec_for_inverse, 0)
    except Exception:
        return None
    x_center = float(np.asarray(x_center))
    y_center = float(np.asarray(y_center))
    if not np.isfinite(x_center) or not np.isfinite(y_center):
        return None
    pad = int(math.ceil(float(region.radius_arcsec) / float(pixel_scale_arcsec))) + 3
    x0 = max(int(math.floor(x_center)) - pad, 0)
    x1 = min(int(math.ceil(x_center)) + pad + 1, width)
    y0 = max(int(math.floor(y_center)) - pad, 0)
    y1 = min(int(math.ceil(y_center)) + pad + 1, height)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((height, width), dtype=bool)
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64), np.arange(x0, x1, dtype=np.float64), indexing="ij",
    )
    omega = np.asarray(obs_wcs.pixel_to_omega_xyz(xx, yy), dtype=np.float64)
    if omega.shape != (y1 - y0, x1 - x0, 3):
        return None
    center = radec_deg_to_omega_xyz(
        np.array(region.center_ra_deg, dtype=np.float64), np.array(region.center_dec_deg, dtype=np.float64),
    )
    dot = np.einsum("ijk,k->ij", omega, center)
    dot = np.clip(dot, -1.0, 1.0)
    sep_rad = np.arccos(dot)
    radius_rad = float(region.radius_arcsec) * np.pi / (180.0 * 3600.0)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = sep_rad <= radius_rad
    return mask


def angular_separation_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = np.deg2rad(float(ra1_deg))
    dec1 = np.deg2rad(float(dec1_deg))
    ra2 = np.deg2rad(float(ra2_deg))
    dec2 = np.deg2rad(float(dec2_deg))
    cos_sep = (np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_sep)) * 3600.0)


def aperture_overlaps_region(
    ra_deg: float, dec_deg: float, radius_arcsec: float, region: SkyCircleRegion,
) -> bool:
    aperture_radius_arcsec = float(radius_arcsec)
    if not np.isfinite(aperture_radius_arcsec) or aperture_radius_arcsec < 0.0:
        raise ValueError("radius_arcsec must be a finite non-negative value in arcsec.")
    return (
        angular_separation_arcsec(ra_deg, dec_deg, region.center_ra_deg, region.center_dec_deg)
        <= float(region.radius_arcsec) + aperture_radius_arcsec
    )
