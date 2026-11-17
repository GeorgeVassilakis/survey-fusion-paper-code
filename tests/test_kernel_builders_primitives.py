from __future__ import annotations

import numpy as np
import torch

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.pgm.contracts import MaskApertureSpec, SkyDiskApertureSpec
from survey_fusion.pgm.kernel_builders import (
    BandpassSampler, GaussianLSF, MaskApertureSampler, SkyDiskApertureSampler, TophatBinSampler,
    _LookupPixelToOmegaBackend, _TangentPlaneTorchBackend,
)
from survey_fusion.utils.wcs import AstropyWCSAdapter, TangentPlaneWCS, radec_deg_to_omega_xyz


class DummyLinearWCS:
    def pixel_to_omega_xyz(self, x, y):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_arr = np.zeros_like(x_arr, dtype=np.float64)
        return np.stack([x_arr, y_arr, z_arr], axis=-1)


def test_bandpass_sampler_respects_range_and_weighting() -> None:
    bp = Bandpass(
        band="r", lam=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        throughput=np.array([0.0, 1.0, 0.0], dtype=np.float64), canonical_id="test_r",
    )
    sampler = BandpassSampler(bp)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7)
    samples = sampler.sample(rng, n_rows=1, k=20000)[0].cpu().numpy()
    assert np.all(samples >= 5000.0)
    assert np.all(samples <= 5200.0)
    assert abs(float(np.mean(samples)) - 5100.0) < 5.0


def test_bandpass_sampler_stratifies_the_unit_cdf() -> None:
    bp = Bandpass(
        band="r",
        lam=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        throughput=np.ones(3, dtype=np.float64),
        canonical_id="test_r",
    )
    sampler = BandpassSampler(bp)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(17)
    samples = sampler.sample(rng, n_rows=4, k=32)

    assert torch.all(samples[:, 1:] >= samples[:, :-1])
    assert torch.all(samples >= 5000.0)
    assert torch.all(samples <= 5200.0)


def test_tophat_bin_sampler_rejects_invalid_edges() -> None:
    edges = np.array([[5000.0, 5100.0], [5100.0, 5090.0]], dtype=np.float64)
    try:
        TophatBinSampler(edges)
    except ValueError as exc:
        assert "positive width" in str(exc)
    else:
        raise AssertionError("TophatBinSampler must reject inverted bin edges.")
    legacy_flat = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    try:
        TophatBinSampler(legacy_flat)
    except ValueError as exc:
        assert "(n_bins, 2)" in str(exc)
    else:
        raise AssertionError("TophatBinSampler must reject flat edge arrays.")


def test_gaussian_lsf_supports_per_bin_fwhm() -> None:
    lsf = GaussianLSF(np.array([0.0, 2.355], dtype=np.float64))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(13)
    offsets = lsf.sample_offsets(
        rng, datum_index=torch.tensor([0, 1], dtype=torch.long), k=4096,
    ).cpu().numpy()
    assert offsets.shape == (2, 4096)
    assert np.allclose(offsets[0], 0.0, atol=1e-12)
    std = float(np.std(offsets[1]))
    assert 0.85 < std < 1.15


def test_mask_aperture_sampler_returns_unit_vectors() -> None:
    obs_wcs = TangentPlaneWCS(ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=0.4, width=2, height=2)
    sampler = MaskApertureSampler(MaskApertureSpec(mask=np.ones((2, 2), dtype=np.float64)), obs_wcs)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(5)
    omega = sampler.sample(rng, n_rows=6, k=8).cpu().numpy()
    assert omega.shape == (6, 8, 3)
    assert np.isfinite(omega).all()
    norms = np.linalg.norm(omega, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_mask_aperture_sampler_psf_jitter_increases_spread() -> None:
    aperture = MaskApertureSpec(mask=np.ones((1, 1), dtype=np.float64))
    obs_wcs = DummyLinearWCS()
    sampler_no_psf = MaskApertureSampler(aperture, obs_wcs, psf_sigma_pix=0.0)
    sampler_with_psf = MaskApertureSampler(aperture, obs_wcs, psf_sigma_pix=1.0)
    rng_a = torch.Generator(device="cpu")
    rng_a.manual_seed(22)
    rng_b = torch.Generator(device="cpu")
    rng_b.manual_seed(22)
    omega_no_psf = sampler_no_psf.sample(rng_a, n_rows=64, k=16).cpu().numpy()
    omega_with_psf = sampler_with_psf.sample(rng_b, n_rows=64, k=16).cpu().numpy()
    std_no = float(np.std(omega_no_psf[..., 0]))
    std_yes = float(np.std(omega_with_psf[..., 0]))
    assert std_yes > std_no * 1.5


def test_sky_disk_aperture_sampler_stays_within_radius() -> None:
    spec = SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.0)
    sampler = SkyDiskApertureSampler(spec)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(11)
    omega = sampler.sample(rng, n_rows=8, k=64).cpu().numpy()
    center = radec_deg_to_omega_xyz(
        np.array([spec.center_radec_deg[0]], dtype=np.float64),
        np.array([spec.center_radec_deg[1]], dtype=np.float64),
    )[0]
    dots = np.clip(np.sum(omega * center[None, None, :], axis=-1), -1.0, 1.0)
    angles = np.arccos(dots)
    radius_rad = np.deg2rad(spec.radius_arcsec / 3600.0)
    assert float(np.max(angles)) <= radius_rad * 1.25


def test_tangent_plane_torch_backend_matches_numpy_wcs() -> None:
    wcs = TangentPlaneWCS(
        ra0_rad=np.deg2rad(180.0), dec0_rad=np.deg2rad(1.5), pixel_scale_arcsec=0.4, width=16, height=12,
    )
    backend = _TangentPlaneTorchBackend(wcs)
    x = torch.tensor([[0.1, 3.25, 7.9], [2.2, 8.8, 12.1]], dtype=torch.float32)
    y = torch.tensor([[1.7, 2.4, 6.6], [9.1, 4.0, 10.2]], dtype=torch.float32)
    omega_torch = backend.pixel_to_omega_xyz(x, y).cpu().numpy()
    omega_numpy = wcs.pixel_to_omega_xyz(x.cpu().numpy(), y.cpu().numpy())
    assert np.allclose(omega_torch, omega_numpy, atol=1.0e-6, rtol=0.0)


class DummyAffineAstropyWcs:
    def all_pix2world(self, x, y, origin):
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ra = 180.0 + x_arr * 1.0e-4 + y_arr * 3.0e-7
        dec = 0.25 + y_arr * 1.0e-4 - x_arr * 2.0e-7
        return ra, dec


def test_astropy_lookup_backend_matches_direct_adapter_within_tolerance() -> None:
    adapter = AstropyWCSAdapter(DummyAffineAstropyWcs(), dx_arcsec=0.05, dy_arcsec=-0.03)
    backend = _LookupPixelToOmegaBackend(adapter, width=32, height=24, pad_pixels=4)
    rng = np.random.default_rng(19)
    x = rng.uniform(-0.4, 31.4, size=(64, 8)).astype(np.float32)
    y = rng.uniform(-0.4, 23.4, size=(64, 8)).astype(np.float32)
    direct = adapter.pixel_to_omega_xyz(x, y)
    approx = backend.pixel_to_omega_xyz(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),
    ).cpu().numpy()
    chord = np.linalg.norm(direct - approx, axis=-1)
    angle_arcsec = np.rad2deg(2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))) * 3600.0
    pixel_scale_arcsec = 0.4
    assert float(np.median(angle_arcsec)) <= 0.05 * pixel_scale_arcsec
    assert float(np.quantile(angle_arcsec, 0.99)) <= 0.25 * pixel_scale_arcsec


def test_astropy_lookup_backend_handles_exact_grid_boundary_coordinates() -> None:
    adapter = AstropyWCSAdapter(DummyAffineAstropyWcs(), dx_arcsec=0.0, dy_arcsec=0.0)
    backend = _LookupPixelToOmegaBackend(adapter, width=4, height=3, pad_pixels=2)
    x = torch.tensor([[-2.5, 4.5], [0.0, 3.9999]], dtype=torch.float32)
    y = torch.tensor([[-2.5, 3.5], [0.0, 2.9999]], dtype=torch.float32)
    omega = backend.pixel_to_omega_xyz(x, y).cpu().numpy()
    assert omega.shape == (2, 2, 3)
    assert np.isfinite(omega).all()
    norms = np.linalg.norm(omega, axis=-1)
    assert np.allclose(norms, 1.0, atol=1.0e-4)
