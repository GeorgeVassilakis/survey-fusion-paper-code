"""Independent physical references for the spectral observation response."""

import math
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import ncx2
from survey_fusion.pgm.splat_spectrum import candidate_aperture_weights, positive_hat_bin_average
from survey_fusion.data.adapters.sdss import _compute_lsf_fwhm_angstrom_by_bin


@pytest.mark.parametrize("loglam", [np.array([3.6]), np.array([3.6, 3.6003, 3.6006])])
def test_lsf_uses_native_pixel_spacing_even_after_sparse_mask(loglam):
    lam = 10**loglam
    wdisp = np.linspace(0.7, 1.3, len(lam))
    got = _compute_lsf_fwhm_angstrom_by_bin(loglam, wdisp, lam, dlog10_lambda=1e-4)
    np.testing.assert_allclose(got, 2.355 * np.log(10.0) * lam * wdisp * 1e-4, rtol=1e-13)


def test_explicit_legacy_lsf_spacing_reproduces_masked_pixel_convention():
    loglam = np.array([3.6, 3.6003, 3.6006])
    lam = 10**loglam
    got = _compute_lsf_fwhm_angstrom_by_bin(loglam, np.ones(3), lam)
    np.testing.assert_allclose(got, 2.355 * np.log(10.0) * lam * 3e-4, rtol=1e-10)


class GaussianField:
    def __init__(self, width, offset):
        self.width, self.offset = width, offset

    def shape_rows(self, *args, **kwargs):
        return torch.full((1, 2), math.log(self.width), dtype=torch.float64), torch.zeros(
            1, dtype=torch.float64
        )

    def positions(self, *args, **kwargs):
        return torch.tensor([[self.offset, 0.0]], dtype=torch.float64)


@pytest.mark.parametrize("radius", [1.0, 1.5])
@pytest.mark.parametrize("width", [0.0001, 0.4, 1.5])
@pytest.mark.parametrize("offset", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("seeing", [1.0, 1.4065684080123901, 2.0])
def test_disk_gaussian_throughput_matches_independent_cdf(radius, width, offset, seeing):
    x, w = np.polynomial.legendre.leggauss(8)
    r = radius * np.sqrt((x + 1) / 2)
    theta = (np.arange(32) + 0.5) * 2 * np.pi / 32
    points = torch.tensor(
        np.stack([np.outer(r, np.cos(theta)).ravel(), np.outer(r, np.sin(theta)).ravel()], axis=1)
    )
    weights = torch.tensor(np.repeat(w / 64, 32))
    sigma = seeing / 2.355
    got = candidate_aperture_weights(
        GaussianField(width, offset),
        torch.tensor([0]),
        points,
        weights,
        torch.eye(2, dtype=torch.float64) * sigma**2,
        math.pi * radius**2,
    ).item()
    total_sigma = math.hypot(width, sigma)
    expected = ncx2.cdf((radius / total_sigma) ** 2, 2, (offset / total_sigma) ** 2)
    assert got == pytest.approx(expected, abs=5e-10)


@pytest.mark.parametrize("sigma", [0.0, 0.5, 2.0, 5.0])
def test_lsf_single_convolution_and_bin_average_preserve_padded_line_flux(sigma):
    knots = np.array([4900.0, 4990.0, 5000.0, 5010.0, 5100.0])
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    edges = np.arange(4940.0, 5061.0, 1.0)
    lo = edges[:-1]
    hi = edges[1:]
    got = positive_hat_bin_average(
        torch.tensor(knots[None, :]),
        torch.tensor(values[None, :]),
        torch.tensor(lo),
        torch.tensor(hi),
        torch.full((lo.size,), sigma, dtype=torch.float64),
        off_grid="zero",
    ).numpy()[:, 0]
    if sigma:
        reference = np.array(
            [
                quad(
                    lambda t: (
                        np.interp(t, knots, values)
                        * (ndtr((b - t) / sigma) - ndtr((a - t) / sigma))
                        / (b - a)
                    ),
                    4990,
                    5010,
                    points=[5000],
                    epsabs=1e-11,
                )[0]
                for a, b in zip(lo, hi)
            ]
        )
    else:
        reference = np.array(
            [
                quad(lambda t: np.interp(t, knots, values), a, b, epsabs=1e-11)[0] / (b - a)
                for a, b in zip(lo, hi)
            ]
        )
    np.testing.assert_allclose(got, reference, atol=5e-10, rtol=1e-8)
    assert np.sum(got * (hi - lo)) == pytest.approx(10.0, abs=1e-8)


def test_disposable_corrected_synthetic_field_takes_three_finite_steps():
    """Engineering smoke test; no scientific experiment or saved field is fit."""
    from dataclasses import replace
    from test_splat_spectrum import _fiber_spectrum_batch, _fiber_path_prediction
    from survey_fusion.pgm.splat_spectrum import build_spectrum_cache

    make_field, caches, _, indices = _fiber_spectrum_batch(
        n_exposures=1,
        bins_per_exposure=(3,),
        n_splats=3,
        dtype=torch.float64,
    )
    exposure = replace(caches[0].source_scale_exposure, spectral_source_scale=1.3)
    cache = build_spectrum_cache(
        exposure,
        exposure.aperture,
        apply_spatial_psf=True,
        radial_nodes=2,
        angular_nodes=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    field = make_field()
    with torch.no_grad():
        field.mu.zero_()
    candidates = [torch.arange(3)]

    def predict():
        return _fiber_path_prediction(field, [cache], candidates, indices, flat=True)

    initial = predict().detach()
    assert torch.isfinite(initial).all() and bool(torch.any(initial > 0))
    target = initial * 1.01
    scale = initial.abs().clamp_min(1e-8)
    optimizer = torch.optim.SGD(field.parameters(), lr=1e-8)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (((predict() - target) / scale) ** 2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in field.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                assert torch.isfinite(gradient).all()
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in field.parameters())
        losses.append(float(loss.detach()))
    assert losses[-1] <= losses[0]
