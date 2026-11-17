from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class PixelToOmegaWCS(Protocol):
    def pixel_to_omega_xyz(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Map detector pixel coordinates to unit sky vectors in ICRS."""
        ...


def radec_deg_to_omega_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if ra.shape != dec.shape:
        raise ValueError("ra_deg and dec_deg must have matching shapes.")
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    cos_dec = np.cos(dec_rad)
    omega_x = cos_dec * np.cos(ra_rad)
    omega_y = cos_dec * np.sin(ra_rad)
    omega_z = np.sin(dec_rad)
    return np.stack([omega_x, omega_y, omega_z], axis=-1)


def radec_deg_to_omega_xyz_torch(ra_deg: torch.Tensor, dec_deg: torch.Tensor) -> torch.Tensor:
    if ra_deg.shape != dec_deg.shape:
        raise ValueError("ra_deg and dec_deg must have matching shapes.")
    ra = torch.deg2rad(ra_deg)
    dec = torch.deg2rad(dec_deg)
    cos_dec = torch.cos(dec)
    omega_x = cos_dec * torch.cos(ra)
    omega_y = cos_dec * torch.sin(ra)
    omega_z = torch.sin(dec)
    return torch.stack([omega_x, omega_y, omega_z], dim=-1)


@dataclass(slots=True)
class AstropyWCSAdapter:
    """Adapter exposing pixel->omega conversion through astropy-like WCS objects."""

    wcs: Any
    dx_arcsec: float = 0.0
    dy_arcsec: float = 0.0

    def pixel_to_omega_xyz(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ra_deg, dec_deg = self.wcs.all_pix2world(x, y, 0)
        ra = np.asarray(ra_deg, dtype=np.float64)
        dec = np.asarray(dec_deg, dtype=np.float64)
        if ra.shape != dec.shape:
            raise ValueError("all_pix2world must return RA/Dec arrays with matching shapes.")
        if self.dx_arcsec != 0.0 or self.dy_arcsec != 0.0:
            cos_dec = np.cos(np.deg2rad(dec))
            cos_dec = np.clip(cos_dec, 1e-12, None)
            ra = ra + float(self.dx_arcsec) / (3600.0 * cos_dec)
            dec = dec + float(self.dy_arcsec) / 3600.0
        return radec_deg_to_omega_xyz(ra, dec)


@dataclass(slots=True)
class TangentPlaneWCS:
    """Minimal tangent-plane WCS for small FOV demos.

    Pixel coordinates are zero-indexed with (0,0) at the top-left.
    The reference pixel is at the image center by default.
    """

    ra0_rad: float
    dec0_rad: float
    pixel_scale_arcsec: float
    width: int
    height: int
    ref_x: float | None = None
    ref_y: float | None = None

    def __post_init__(self) -> None:
        if self.ref_x is None:
            self.ref_x = 0.5 * (self.width - 1)
        if self.ref_y is None:
            self.ref_y = 0.5 * (self.height - 1)

    def pixel_to_tangent(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale_rad = (self.pixel_scale_arcsec / 3600.0) * np.pi / 180.0
        dx = (x - self.ref_x) * scale_rad
        dy = (y - self.ref_y) * scale_rad
        return dx, dy

    def tangent_to_pixel(self, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale_inv = (3600.0 / self.pixel_scale_arcsec) * 180.0 / np.pi
        x = dx * scale_inv + self.ref_x
        y = dy * scale_inv + self.ref_y
        return x, y

    def pixel_to_omega_xyz(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx, dy = self.pixel_to_tangent(x, y)
        ra = self.ra0_rad + dx / np.cos(self.dec0_rad)
        dec = self.dec0_rad + dy
        cosd = np.cos(dec)
        omega = np.stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)], axis=-1)
        return omega

    def omega_xyz_to_tangent(self, omega_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = omega_xyz[..., 0]
        y = omega_xyz[..., 1]
        z = omega_xyz[..., 2]
        ra = np.arctan2(y, x)
        dec = np.arcsin(np.clip(z, -1.0, 1.0))
        delta_ra = np.arctan2(np.sin(ra - self.ra0_rad), np.cos(ra - self.ra0_rad))
        dx = delta_ra * np.cos(self.dec0_rad)
        dy = dec - self.dec0_rad
        return dx, dy

    def omega_xyz_to_pixel(self, omega_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = self.omega_xyz_to_tangent(omega_xyz)
        return self.tangent_to_pixel(dx, dy)

    def omega_xyz_to_norm(self, omega_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_pix, y_pix = self.omega_xyz_to_pixel(omega_xyz)
        x_norm = 2.0 * x_pix / (self.width - 1) - 1.0
        y_norm = 2.0 * y_pix / (self.height - 1) - 1.0
        return x_norm, y_norm

    def omega_xyz_to_norm_torch(self, omega_xyz):
        if omega_xyz.shape[-1] != 3:
            raise ValueError("omega_xyz last dimension must be 3.")
        x = omega_xyz[..., 0]
        y = omega_xyz[..., 1]
        z = omega_xyz[..., 2]
        ra = torch.atan2(y, x)
        dec = torch.asin(torch.clamp(z, -1.0, 1.0))
        ra0 = float(self.ra0_rad)
        delta_ra = torch.atan2(torch.sin(ra - ra0), torch.cos(ra - ra0))
        dx = delta_ra * math.cos(float(self.dec0_rad))
        dy = dec - float(self.dec0_rad)
        scale_inv = (3600.0 / float(self.pixel_scale_arcsec)) * 180.0 / math.pi
        x_pix = dx * scale_inv + float(self.ref_x)
        y_pix = dy * scale_inv + float(self.ref_y)
        x_norm = 2.0 * x_pix / (self.width - 1) - 1.0
        y_norm = 2.0 * y_pix / (self.height - 1) - 1.0
        return x_norm, y_norm

    def pixel_to_omega_xyz_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("x and y must have matching shapes.")
        scale_rad = (float(self.pixel_scale_arcsec) / 3600.0) * math.pi / 180.0
        dx = (x - float(self.ref_x)) * scale_rad
        dy = (y - float(self.ref_y)) * scale_rad
        ra = dx / math.cos(float(self.dec0_rad)) + float(self.ra0_rad)
        dec = dy + float(self.dec0_rad)
        cosd = torch.cos(dec)
        return torch.stack([cosd * torch.cos(ra), cosd * torch.sin(ra), torch.sin(dec)], dim=-1)
