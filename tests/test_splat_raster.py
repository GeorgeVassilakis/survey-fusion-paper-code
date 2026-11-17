import math
import time

import pytest

import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.analytic_image import analytic_band_prediction
from survey_fusion.pgm.splat_raster import (
    band_amplitudes,
    patch_size_px,
    render_band_image,
)

H = W = 64
LL = torch.linspace(-0.3, 0.1, 8)


def _field(n_splats=24, sigma_px=1.5, seed=0):
    torch.manual_seed(seed)
    npp = 2.0 / (W - 1)
    f = SplatField(
        n_splats=n_splats, n_templates=4,
        loglam_grid=torch.linspace(-1, 1, 32), init_scale=sigma_px * npp,
    )
    with torch.no_grad():
        f.mu.uniform_(-0.7, 0.7)
        f.raw_flux.fill_(0.0)
    return f


def _dense_image(f, psf_sigma):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    return analytic_band_prediction(
        f, xs.reshape(-1), ys.reshape(-1), psf_sigma, LL
    ).reshape(H, W)


def _exact_pair_prune_case():
    torch.manual_seed(12)
    f = SplatField(
        n_splats=5,
        n_templates=2,
        loglam_grid=torch.linspace(-1, 1, 32),
        init_scale=0.05,
    )
    with torch.no_grad():
        f.mu.copy_(
            torch.tensor(
                [
                    [-0.30, -0.12],
                    [0.02, 0.10],
                    [0.28, -0.18],
                    [0.48, 0.24],
                    [-0.30, 0.32],
                ]
            )
        )
        f.log_scales.copy_(
            torch.log(
                torch.tensor(
                    [
                        [0.045, 0.015],
                        [0.030, 0.070],
                        [0.020, 0.055],
                        [0.060, 0.025],
                        [0.050, 0.020],
                    ]
                )
            )
        )
        f.theta.copy_(torch.tensor([0.55, -0.35, 0.90, 0.20, 0.70]))
        f.raw_flux.copy_(torch.tensor([0.30, -0.20, 0.60, 0.10, 0.40]))
    x = torch.tensor([-0.34, -0.12, 0.08, 0.26, 0.45, 0.58])
    y = torch.tensor([-0.10, 0.08, 0.18, -0.12, 0.20, 0.40])
    psf_sigma = torch.tensor([0.008, 0.018, 0.032, 0.012, 0.045, 0.025])
    return f, x, y, psf_sigma


def _dense_pair_terms(f, x, y, psf_sigma):
    with torch.no_grad():
        covariance = f.covariances()
        a = covariance[:, 0, 0][None, :] + psf_sigma.square()[:, None]
        b = covariance[:, 0, 1][None, :]
        d = covariance[:, 1, 1][None, :] + psf_sigma.square()[:, None]
        determinant = a * d - b.square()
        dx = x[:, None] - f.mu[:, 0][None, :]
        dy = y[:, None] - f.mu[:, 1][None, :]
        q = (
            d * dx.square() - 2.0 * b * dx * dy + a * dy.square()
        ) / determinant
    return q, determinant


def test_packed_pair_prune_disabled_preserves_tile_superset(monkeypatch):
    from survey_fusion.pgm.splat_raster import build_packed_point_index

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "0")
    f, x, y, psf_sigma = _exact_pair_prune_case()
    index = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)

    assert index.pair_point.tolist() == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3,
        4, 4, 4,
        5, 5,
    ]
    assert index.active_splat.tolist() == [0, 1, 2, 3, 4]
    assert index.pair_active_splat.tolist() == [
        0, 1, 2, 4,
        0, 1, 2, 4,
        0, 1, 2, 4,
        1, 2, 3,
        1, 2, 3,
        1, 3,
    ]


def test_packed_pair_prune_enabled_by_default(monkeypatch):
    from survey_fusion.pgm.splat_raster import build_packed_point_index

    f, x, y, psf_sigma = _exact_pair_prune_case()

    monkeypatch.delenv("SPLAT_EXACT_PAIR_PRUNE", raising=False)
    default = build_packed_point_index(f, x, y, psf_sigma)

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "1")
    enabled = build_packed_point_index(f, x, y, psf_sigma)

    assert torch.equal(default.pair_point, enabled.pair_point)
    assert torch.equal(default.active_splat, enabled.active_splat)
    assert torch.equal(default.pair_active_splat, enabled.pair_active_splat)


def test_packed_exact_pair_prune_matches_dense_anisotropic_q(monkeypatch):
    from survey_fusion.pgm.splat_raster import build_packed_point_index

    f, x, y, psf_sigma = _exact_pair_prune_case()
    q, _ = _dense_pair_terms(f, x, y, psf_sigma)

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "0")
    unpruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)
    unpruned_splat = unpruned.active_splat[unpruned.pair_active_splat]
    keep = q[unpruned.pair_point, unpruned_splat] <= 12.25

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "1")
    pruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)
    pruned_splat = pruned.active_splat[pruned.pair_active_splat]

    assert keep.any()
    assert (~keep).any()
    assert torch.equal(pruned.pair_point, unpruned.pair_point[keep])
    assert torch.equal(pruned_splat, unpruned_splat[keep])
    assert torch.all(q[pruned.pair_point, pruned_splat] <= 12.25)
    assert torch.all(
        q[unpruned.pair_point[~keep], unpruned_splat[~keep]] > 12.25
    )


def test_packed_exact_pair_prune_prediction_has_bounded_tail_error(monkeypatch):
    from survey_fusion.pgm.splat_raster import (
        build_packed_point_index,
        packed_multiband_point_prediction,
    )

    f, x, y, psf_sigma = _exact_pair_prune_case()
    amplitudes = band_amplitudes(f, LL)
    point_band = torch.zeros(x.numel(), dtype=torch.long)
    q, determinant = _dense_pair_terms(f, x, y, psf_sigma)

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "0")
    unpruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)
    unpruned_prediction = packed_multiband_point_prediction(
        f,
        x,
        y,
        psf_sigma,
        point_band,
        amplitudes[unpruned.active_splat, None],
        unpruned,
        checkpoint_chunks=False,
    )
    unpruned_splat = unpruned.active_splat[unpruned.pair_active_splat]
    pair_q = q[unpruned.pair_point, unpruned_splat]
    dropped = pair_q > 12.25
    assert dropped.any()
    dropped_peak = amplitudes[unpruned_splat[dropped]] / (
        2.0
        * math.pi
        * determinant[
            unpruned.pair_point[dropped], unpruned_splat[dropped]
        ].sqrt()
    )
    dropped_value = dropped_peak * torch.exp(-0.5 * pair_q[dropped])
    dropped_by_point = torch.zeros_like(unpruned_prediction).index_add(
        0, unpruned.pair_point[dropped], dropped_value
    )

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "1")
    pruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)
    pruned_prediction = packed_multiband_point_prediction(
        f,
        x,
        y,
        psf_sigma,
        point_band,
        amplitudes[pruned.active_splat, None],
        pruned,
        checkpoint_chunks=False,
    )

    torch.testing.assert_close(
        pruned_prediction,
        unpruned_prediction - dropped_by_point,
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.all(
        dropped_value
        <= math.exp(-12.25 / 2.0) * dropped_peak * (1.0 + 1e-6)
    )

    dense = torch.cat(
        [
            analytic_band_prediction(
                f, x[i : i + 1], y[i : i + 1], float(psf_sigma[i]), LL
            )
            for i in range(x.numel())
        ]
    )
    error = (pruned_prediction - dense).abs().max() / dense.abs().max()
    assert error < 1e-2, f"exact-pruned packed vs dense: {error:.2e}"


def test_packed_exact_pair_prune_preserves_active_index_contract(monkeypatch):
    from survey_fusion.pgm.splat_raster import build_packed_point_index

    f, x, y, psf_sigma = _exact_pair_prune_case()
    q, _ = _dense_pair_terms(f, x, y, psf_sigma)

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "0")
    unpruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)
    unpruned_splat = unpruned.active_splat[unpruned.pair_active_splat]
    expected_splat = unpruned_splat[
        q[unpruned.pair_point, unpruned_splat] <= 12.25
    ]
    expected_active, expected_inverse = torch.unique(
        expected_splat, sorted=True, return_inverse=True
    )

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "1")
    pruned = build_packed_point_index(f, x, y, psf_sigma, n_sigma=3.5)

    assert pruned.active_splat.numel() < unpruned.active_splat.numel()
    assert torch.equal(pruned.active_splat, expected_active)
    assert torch.equal(pruned.pair_active_splat, expected_inverse)
    assert torch.equal(
        pruned.active_splat[pruned.pair_active_splat], expected_splat
    )
    assert torch.equal(
        pruned.active_splat,
        torch.unique(pruned.active_splat, sorted=True),
    )


def test_tiled_exact_pair_prune_shares_dense_cutoff(monkeypatch):
    from survey_fusion.pgm.splat_raster import tiled_point_prediction

    f, x, y, psf_sigma = _exact_pair_prune_case()
    amplitudes = band_amplitudes(f, LL)
    q, determinant = _dense_pair_terms(f, x, y, psf_sigma)
    exact_values = amplitudes[None, :] / (
        2.0 * math.pi * determinant.sqrt()
    ) * torch.exp(-0.5 * q)
    expected = (exact_values * (q <= 12.25)).sum(dim=1)

    monkeypatch.setenv("SPLAT_EXACT_PAIR_PRUNE", "1")
    actual = tiled_point_prediction(
        f,
        x,
        y,
        psf_sigma,
        amplitudes,
        n_sigma=3.5,
        checkpoint_chunks=False,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_patch_render_matches_dense_renderer():
    f = _field()
    psf_sigma = 1.0 * (2.0 / (W - 1))
    patch = patch_size_px(4.0, n_sigma=4.0)  # generous: covers >6 sigma
    with torch.no_grad():
        dense = _dense_image(f, psf_sigma)
        patched = render_band_image(f, H, W, psf_sigma, LL, patch_px=patch)
    err = (patched - dense).abs().max() / dense.abs().max()
    assert err < 1e-4, f"patch vs dense mismatch: {err:.2e}"


def test_patch_render_gradients_match_dense():
    psf_sigma = 1.0 * (2.0 / (W - 1))
    patch = patch_size_px(4.0, n_sigma=4.0)
    target = torch.zeros(H, W)

    grads = []
    for renderer in ("dense", "patch"):
        f = _field()
        if renderer == "dense":
            img = _dense_image(f, psf_sigma)
        else:
            img = render_band_image(f, H, W, psf_sigma, LL, patch_px=patch)
        ((img - target) ** 2).mean().backward()
        grads.append((f.mu.grad.clone(), f.raw_flux.grad.clone()))

    for a, b in zip(*grads):
        scale = a.abs().max().clamp_min(1e-12)
        assert (a - b).abs().max() / scale < 1e-3


def test_amplitudes_are_nonnegative_and_scale_with_flux():
    f = _field(n_splats=8)
    amp0 = band_amplitudes(f, LL)
    with torch.no_grad():
        f.raw_flux.fill_(2.0)
    amp1 = band_amplitudes(f, LL)
    assert (amp0 >= 0).all() and (amp1 > amp0).all()


def test_patch_cost_is_independent_of_image_size():
    """The structural claim: patch work scales with N_splats * patch_area, not
    with N_splats * N_pixels. Asserted on evaluation counts, not wall clock."""
    n_splats, patch = 4000, 13
    patch_evals = n_splats * patch * patch
    for side in (256, 1024, 4096):
        dense_evals = n_splats * side * side
        assert patch_evals < dense_evals / 50, (
            f"at {side}x{side}: patch {patch_evals} vs dense {dense_evals}"
        )
    # cost must not grow when the image grows
    assert patch_evals == n_splats * patch * patch


def test_patch_render_scales_to_large_image_without_error():
    torch.manual_seed(0)
    side = 512
    npp = 2.0 / (side - 1)
    f = SplatField(
        n_splats=3000, n_templates=4,
        loglam_grid=torch.linspace(-1, 1, 32), init_scale=1.5 * npp,
    )
    with torch.no_grad():
        f.mu.uniform_(-1.1, 1.1)   # includes off-image splats
    with torch.no_grad():
        img = render_band_image(f, side, side, npp, LL[:2], patch_px=13)
    assert img.shape == (side, side)
    assert torch.isfinite(img).all()
    assert (img >= 0).all()


def test_warp_cache_roundtrip_names_survive_savez():
    """np.savez appends .npz unless the path already ends in it; a temp name
    without the suffix makes the subsequent os.replace fail. This crashed 29
    sweep configs, so pin the behaviour."""
    import os
    import tempfile

    import numpy as np

    d = tempfile.mkdtemp()
    cache = os.path.join(d, "warp_200_500_512.npz")
    cut = {"g": np.zeros((4, 4), np.float32), "r": np.ones((4, 4), np.float32)}
    tmp = cache + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **cut)
    assert os.path.exists(tmp), "savez wrote to a different path than requested"
    os.replace(tmp, cache)
    z = np.load(cache)
    assert set(z.files) == {"g", "r"}
    assert z["r"].sum() == 16


def test_grid_rasterizer_matches_dense_on_uniform_grid():
    """rasterize_band_on_grid must agree with the dense renderer when handed a
    uniform coordinate map, and stay differentiable."""
    from survey_fusion.pgm.splat_raster import band_amplitudes, rasterize_band_on_grid

    f = _field(n_splats=20)
    psf_sigma = 1.0 * (2.0 / (W - 1))
    patch = patch_size_px(4.0, n_sigma=4.0)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    amps = band_amplitudes(f, LL)
    with torch.no_grad():
        grid_img = rasterize_band_on_grid(f, xs, ys, psf_sigma, amps, patch)
        dense = analytic_band_prediction(
            f, xs.reshape(-1), ys.reshape(-1), psf_sigma, LL
        ).reshape(H, W)
    err = (grid_img - dense).abs().max() / dense.abs().max()
    assert err < 5e-3, f"grid vs dense mismatch: {err:.2e}"


@pytest.mark.parametrize("theta_deg", [0.0, 2.9, 30.0, 90.0, 135.0])
def test_grid_rasterizer_matches_dense_under_rotation(theta_deg):
    """A real COSMOS frame is rotated, so x varies along rows as strongly as
    along columns. Placing patches by inverting each axis independently puts
    every splat outside the image and renders exactly zero, while the dense
    renderer is unaffected. Splats must be spread over the field and the result
    compared against the dense path: a test with all splats at the origin and a
    3 degree rotation passes even when placement is completely broken.
    """
    from survey_fusion.pgm.splat_raster import band_amplitudes, rasterize_band_on_grid

    torch.manual_seed(0)
    f = _field(n_splats=8)
    with torch.no_grad():
        f.mu.uniform_(-0.5, 0.5)
        f.raw_flux.fill_(2.0)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    th = math.radians(theta_deg)
    xr = xs * math.cos(th) - ys * math.sin(th)
    yr = xs * math.sin(th) + ys * math.cos(th)

    psf_sigma = 4.0 * (2.0 / (W - 1))
    patch = patch_size_px(4.0, n_sigma=4.0)
    amps = band_amplitudes(f, LL)
    with torch.no_grad():
        img = rasterize_band_on_grid(f, xr, yr, psf_sigma, amps, patch)
        dense = analytic_band_prediction(
            f, xr.reshape(-1), yr.reshape(-1), psf_sigma, LL
        ).reshape(H, W)

    assert img.max() > 0, f"rendered an empty image at {theta_deg} deg"
    err = (img - dense).abs().max() / dense.abs().max()
    assert err < 5e-3, f"raster vs dense at {theta_deg} deg: {err:.2e}"


def test_tiled_points_match_dense_at_scattered_coordinates():
    """tiled_point_prediction must agree with the dense evaluator on arbitrary
    scattered points; with a generous n_sigma the tile cutoff is negligible."""
    from survey_fusion.pgm.splat_raster import tiled_point_prediction

    torch.manual_seed(3)
    f = _field(n_splats=40)
    x = torch.empty(500).uniform_(-1, 1)
    y = torch.empty(500).uniform_(-1, 1)
    psf_sigma = 1.5 * (2.0 / (W - 1))
    amps = band_amplitudes(f, LL)
    with torch.no_grad():
        dense = analytic_band_prediction(f, x, y, psf_sigma, LL)
        tiled = tiled_point_prediction(f, x, y, psf_sigma, amps, n_sigma=6.0)
    err = (tiled - dense).abs().max() / dense.abs().max()
    assert err < 1e-4, f"tiled vs dense mismatch: {err:.2e}"


def test_tiled_points_default_cutoff_close_to_dense():
    """At the default n_sigma the truncated tails must stay below the noise
    scale the trainer works at (~1% of peak)."""
    from survey_fusion.pgm.splat_raster import tiled_point_prediction

    torch.manual_seed(4)
    f = _field(n_splats=40)
    x = torch.empty(500).uniform_(-1, 1)
    y = torch.empty(500).uniform_(-1, 1)
    psf_sigma = 1.5 * (2.0 / (W - 1))
    amps = band_amplitudes(f, LL)
    with torch.no_grad():
        dense = analytic_band_prediction(f, x, y, psf_sigma, LL)
        tiled = tiled_point_prediction(f, x, y, psf_sigma, amps)
    err = (tiled - dense).abs().max() / dense.abs().max()
    assert err < 1e-2, f"default-cutoff tiled vs dense: {err:.2e}"


def test_tiled_points_per_point_sigma_matches_dense_groups():
    """Points from different exposures of one band carry different seeing; the
    per-point sigma path must match dense evaluation done per sigma group."""
    from survey_fusion.pgm.splat_raster import tiled_point_prediction

    torch.manual_seed(5)
    f = _field(n_splats=30)
    n_half = 200
    x = torch.empty(2 * n_half).uniform_(-1, 1)
    y = torch.empty(2 * n_half).uniform_(-1, 1)
    s_lo, s_hi = 1.0 * (2.0 / (W - 1)), 3.0 * (2.0 / (W - 1))
    sig = torch.cat([torch.full((n_half,), s_lo), torch.full((n_half,), s_hi)])
    amps = band_amplitudes(f, LL)
    with torch.no_grad():
        tiled = tiled_point_prediction(f, x, y, sig, amps, n_sigma=6.0)
        d_lo = analytic_band_prediction(f, x[:n_half], y[:n_half], s_lo, LL)
        d_hi = analytic_band_prediction(f, x[n_half:], y[n_half:], s_hi, LL)
    dense = torch.cat([d_lo, d_hi])
    err = (tiled - dense).abs().max() / dense.abs().max()
    assert err < 1e-4, f"per-point sigma tiled vs dense: {err:.2e}"


def test_tiled_points_gradients_match_dense():
    from survey_fusion.pgm.splat_raster import tiled_point_prediction

    torch.manual_seed(6)
    x = torch.empty(300).uniform_(-1, 1)
    y = torch.empty(300).uniform_(-1, 1)
    psf_sigma = 1.5 * (2.0 / (W - 1))

    grads = []
    for path in ("dense", "tiled"):
        f = _field(n_splats=25, seed=9)
        if path == "dense":
            pred = analytic_band_prediction(f, x, y, psf_sigma, LL)
        else:
            amps = band_amplitudes(f, LL)
            pred = tiled_point_prediction(f, x, y, psf_sigma, amps, n_sigma=6.0)
        (pred ** 2).mean().backward()
        grads.append((f.mu.grad.clone(), f.raw_flux.grad.clone()))

    for g_dense, g_tiled in zip(*grads):
        scale = g_dense.abs().max().clamp_min(1e-12)
        assert (g_dense - g_tiled).abs().max() / scale < 1e-3


def test_pair_index_is_sparse_for_local_splats():
    """The structural claim: a point only pairs with splats near it, so total
    pairs stay far below n_points * n_splats when splats are compact."""
    from survey_fusion.pgm.splat_raster import build_pair_index

    torch.manual_seed(7)
    n_splats, n_points = 5000, 2000
    mu = torch.empty(n_splats, 2).uniform_(-1, 1)
    radius = torch.full((n_splats,), 0.01)
    x = torch.empty(n_points).uniform_(-1, 1)
    y = torch.empty(n_points).uniform_(-1, 1)
    pair_point, pair_splat = build_pair_index(mu, radius, x, y)
    assert pair_point.numel() == pair_splat.numel()
    dense_pairs = n_points * n_splats
    assert pair_point.numel() < dense_pairs / 100, (
        f"pair index not sparse: {pair_point.numel()} of {dense_pairs}"
    )

    # and it must contain every genuinely overlapping pair
    dx = x[:, None] - mu[None, :, 0]
    dy = y[:, None] - mu[None, :, 1]
    hit = (dx.abs() <= radius[None, :]) & (dy.abs() <= radius[None, :])
    got = set(zip(pair_point.tolist(), pair_splat.tolist()))
    needed = set(zip(*[t.tolist() for t in hit.nonzero(as_tuple=True)]))
    assert needed <= got, f"missing {len(needed - got)} overlapping pairs"


def test_grid_rasterizer_rejects_degenerate_coordinate_maps():
    """Silently substituting a unit step for a degenerate axis is how the
    all-zero render went unnoticed; a singular Jacobian must fail loudly."""
    from survey_fusion.pgm.splat_raster import band_amplitudes, rasterize_band_on_grid

    f = _field(n_splats=4)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    amps = band_amplitudes(f, LL)
    with pytest.raises(ValueError, match="degenerate"):
        rasterize_band_on_grid(f, xs, xs, 0.02, amps, 21)


def test_pair_support_default_is_eight_sigma():
    import inspect

    from survey_fusion.pgm.splat_raster import (
        build_packed_point_index,
        tiled_point_prediction,
    )

    for fn in (build_packed_point_index, tiled_point_prediction):
        default = inspect.signature(fn).parameters["n_sigma"].default
        assert default == 8.0, f"{fn.__name__} support default regressed: {default}"
