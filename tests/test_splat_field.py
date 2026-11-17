import math

import torch

from survey_fusion.models.splat_field import SplatField


def make_field(n_splats=32):
    return SplatField(
        n_splats=n_splats,
        n_templates=8,
        loglam_grid=torch.linspace(-1, 1, 64),
        init_extent=1.0,
    )


def test_forward_shape_and_finite():
    f = make_field()
    n = 100
    x = torch.rand(n) * 2 - 1
    y = torch.rand(n) * 2 - 1
    ll = torch.rand(n) * 2 - 1
    out = f(x, y, ll)
    assert out.shape == (n,)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_matches_bruteforce_gaussian_sum():
    torch.manual_seed(0)
    f = make_field(n_splats=4)
    x, y, ll = torch.tensor([0.1]), torch.tensor([-0.2]), torch.tensor([0.0])
    out = f(x, y, ll)
    with torch.no_grad():
        f.sed_shift.copy_(torch.tensor([0.0, 0.1, -0.2, 0.05]))
    out = f(x, y, ll)
    mu, cov, w = f.positions(), f.covariances(), f.fluxes()
    sed = f.sed.eval(f.sed_coeffs, ll, f.sed_shift)
    expect = torch.zeros(1)
    for j in range(4):
        d = torch.stack([x - mu[j, 0], y - mu[j, 1]], -1)
        prec = torch.linalg.inv(cov[j])
        q = (d @ prec * d).sum(-1)
        norm = 1.0 / (2 * math.pi * torch.sqrt(torch.linalg.det(cov[j])))
        expect += w[j] * norm * torch.exp(-0.5 * q) * sed[j, 0]
    assert torch.allclose(out, expect, rtol=1e-4)


def test_covariances_are_spd():
    f = make_field()
    cov = f.covariances()
    eig = torch.linalg.eigvalsh(cov)
    assert (eig > 0).all()


def test_gradients_flow_to_all_parameters():
    f = make_field(n_splats=8)
    n = 64
    x = torch.rand(n) * 2 - 1
    y = torch.rand(n) * 2 - 1
    ll = torch.rand(n) * 2 - 1
    f(x, y, ll).sum().backward()
    for name, p in f.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"nan grad for {name}"


def test_field_carries_no_free_sky():
    """The sky is data, not a parameter: the trainer adds the SDSS-measured sky
    as a known additive offset. A free sky here is redundant, and being
    softplus-positive it planted a fixed off-centre blob at init that swamped
    the sources."""
    f = make_field(n_splats=4)
    names = dict(f.named_parameters())
    assert "raw_sky" not in names and "uniform_raw" not in names
    with torch.no_grad():
        f.raw_flux.fill_(-30.0)
    out = f(torch.tensor([-0.9, 0.0, 0.9]), torch.tensor([0.8, 0.0, -0.8]),
            torch.zeros(3))
    assert torch.allclose(out, torch.zeros(3), atol=1e-9)


def test_forward_chunks_large_query_batches():
    """A caller asking for many points at once must not allocate n*J: the MC
    path routinely passes 640k points, which at 12k splats is ~94 GiB."""
    f = SplatField(n_splats=500, n_templates=4,
                   loglam_grid=torch.linspace(-1, 1, 16))
    n = 20000
    x = torch.rand(n) * 2 - 1
    y = torch.rand(n) * 2 - 1
    ll = torch.rand(n) * 2 - 1
    with torch.no_grad():
        whole = f(x, y, ll, max_elements=10**9)          # single shot
        chunked = f(x, y, ll, max_elements=100_000)      # forced into chunks
    assert chunked.shape == (n,)
    assert torch.allclose(whole, chunked, rtol=1e-5, atol=1e-6)


def test_chunked_forward_gradients_match_unchunked():
    f = SplatField(n_splats=200, n_templates=4,
                   loglam_grid=torch.linspace(-1, 1, 16))
    n = 5000
    x = torch.rand(n) * 2 - 1
    y = torch.rand(n) * 2 - 1
    ll = torch.rand(n) * 2 - 1
    f(x, y, ll, max_elements=10**9).sum().backward()
    g_whole = f.mu.grad.clone()
    f.zero_grad(set_to_none=True)
    f(x, y, ll, max_elements=50_000).sum().backward()
    # Each gradient entry is a 5000-term float32 reduction, and chunking changes
    # the summation order. Compare against the gradient's own scale: an
    # element-wise rtol instead tests the components that happen to sit near
    # zero, where reduction noise is unbounded in relative terms and says
    # nothing about whether chunking is correct.
    scale = g_whole.abs().max()
    err = (g_whole - f.mu.grad).abs().max() / scale
    assert err < 1e-5, f"chunking changed gradients by {err:.2e} of full scale"
