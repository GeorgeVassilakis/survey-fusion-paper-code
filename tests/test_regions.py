from __future__ import annotations

import numpy as np
import pytest

from survey_fusion.data.regions import (
    SkyCircleRegion, aperture_overlaps_region, normalize_pointing_offsets, parse_holdout_region_config,
    parse_region_metrics_config, sky_circle_mask_from_omega_wcs,
)
from survey_fusion.utils.wcs import radec_deg_to_omega_xyz


class DummyPixelOmegaWCS:
    def __init__(
        self, *, ra0_deg: float = 180.0, dec0_deg: float = 1.0, scale_arcsec: float = 0.4, ref_x: float = 0.0,
        ref_y: float = 0.0,
    ) -> None:
        self.ra0_deg = float(ra0_deg)
        self.dec0_deg = float(dec0_deg)
        self.scale_deg = float(scale_arcsec) / 3600.0
        self.ref_x = float(ref_x)
        self.ref_y = float(ref_y)

    def pixel_to_omega_xyz(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ra = self.ra0_deg + (x_arr - self.ref_x) * self.scale_deg / np.cos(np.deg2rad(self.dec0_deg))
        dec = self.dec0_deg + (y_arr - self.ref_y) * self.scale_deg
        return radec_deg_to_omega_xyz(ra, dec)


def test_parse_holdout_region_config() -> None:
    cfg = parse_holdout_region_config(
        {
            "enabled": True, "center_radec_deg": [10.0, -2.0], "radius_arcsec": 7.0,
            "exclude_spectra_in_region": True,
        }
    )
    assert bool(cfg["enabled"])
    assert bool(cfg["exclude_spectra_in_region"])
    region = cfg["region"]
    assert isinstance(region, SkyCircleRegion)
    assert region.center_ra_deg == pytest.approx(10.0)
    assert region.center_dec_deg == pytest.approx(-2.0)
    assert region.radius_arcsec == pytest.approx(7.0)

    disabled = parse_holdout_region_config({"enabled": False})
    assert not bool(disabled["enabled"])
    assert disabled["region"] is None

    with pytest.raises(ValueError, match="holdout_region.center_radec_deg"):
        parse_holdout_region_config({"enabled": True, "center_radec_deg": [1.0], "radius_arcsec": 1.0})


def test_parse_region_metrics_config() -> None:
    cfg = parse_region_metrics_config(
        {
            "enabled": True,
            "primary_region": "R2",
            "regions": [
                {"name": "R1", "center_radec_deg": [10.0, 0.0], "radius_arcsec": 4.0},
                {"name": "R2", "center_radec_deg": [11.0, 0.1], "radius_arcsec": 5.0},
            ],
        }
    )
    assert bool(cfg["enabled"])
    assert cfg["primary_region"] == "R2"
    regions = cfg["regions"]
    assert isinstance(regions, list)
    assert [r.name for r in regions] == ["R1", "R2"]

    with pytest.raises(ValueError, match="primary_region"):
        parse_region_metrics_config(
            {
                "enabled": True, "primary_region": "missing",
                "regions": [{"name": "R1", "center_radec_deg": [10.0, 0.0], "radius_arcsec": 4.0}],
            }
        )


def test_normalize_pointing_offsets() -> None:
    bands = ["g", "r", "i"]
    offsets = normalize_pointing_offsets({"i": [0.5, -0.25]}, bands)
    assert offsets["g"] == (0.0, 0.0)
    assert offsets["r"] == (0.0, 0.0)
    assert offsets["i"] == pytest.approx((0.5, -0.25))


def test_sky_circle_mask_from_omega_wcs() -> None:
    wcs = DummyPixelOmegaWCS(ra0_deg=180.0, dec0_deg=1.0, scale_arcsec=0.4, ref_x=5.0, ref_y=5.0)
    region = SkyCircleRegion(name="core", center_ra_deg=180.0, center_dec_deg=1.0, radius_arcsec=0.45)
    mask = sky_circle_mask_from_omega_wcs(wcs, (11, 11), region)
    assert mask.shape == (11, 11)
    assert bool(mask[5, 5])
    assert not bool(mask[0, 0])


def test_aperture_overlaps_region_uses_aperture_radius() -> None:
    region = SkyCircleRegion(name="holdout", center_ra_deg=180.0, center_dec_deg=0.5, radius_arcsec=0.4)
    assert aperture_overlaps_region(180.0, 0.5 + (1.6 / 3600.0), 1.5, region)
    assert not aperture_overlaps_region(180.0, 0.5 + (2.1 / 3600.0), 1.5, region)
