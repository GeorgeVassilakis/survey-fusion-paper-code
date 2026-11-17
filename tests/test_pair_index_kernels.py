import pytest
import torch

from survey_fusion.pgm.splat_raster import _build_pair_index_eager, _prune_exact_pairs, build_pair_index

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("case", ["ordinary", "coarsened", "zero_radius", "single_point", "outside", "empty"])
def test_fused_pair_index_preserves_census_order_and_edges(monkeypatch, case):
    generator = torch.Generator(device="cuda").manual_seed(177)
    mu = torch.rand((10_000, 2), generator=generator, device="cuda") * 2 - 1
    radius = torch.rand((10_000,), generator=generator, device="cuda") * 0.04
    radius_y = radius * 0.4
    x = torch.tensor([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], device="cuda")
    y = x.flip(0)
    budget = 32_000_000
    if case == "coarsened":
        budget = 1000
    elif case == "zero_radius":
        radius.zero_()
        radius_y.zero_()
        mu[: x.numel()] = torch.stack((x, y), dim=1)
    elif case == "single_point":
        x, y = x[:1], y[:1]
    elif case == "outside":
        mu.add_(20)
    elif case == "empty":
        x, y = x[:0], y[:0]
    expected = _build_pair_index_eager(mu, radius, x, y, budget, radius_y=radius_y)
    monkeypatch.setenv("SPLAT_PAIR_INDEX", "fused")
    actual = build_pair_index(mu, radius, x, y, budget, radius_y=radius_y)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype == torch.int64
        assert torch.equal(a, b)


@pytest.mark.parametrize("per_point", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_fused_support_includes_exact_threshold(monkeypatch, per_point, asymmetric):
    x = torch.tensor([8.0, 0.0, -8.0, 0.0, 8.0], device="cuda")
    x[-1] = torch.nextafter(x[-1], torch.tensor(float("inf"), device="cuda"))
    y = torch.tensor([0.0, 8.0, 0.0, -8.0, 0.0], device="cuda")
    mu = torch.zeros((1, 2), device="cuda")
    aa = torch.ones(1, device="cuda")
    bb = torch.zeros(1, device="cuda")
    dd = torch.ones(1, device="cuda")
    cc = bb.clone() if asymmetric else None
    variance = torch.zeros(5, device="cuda") if per_point else 0.0
    pp = torch.arange(5, device="cuda")
    ps = torch.zeros(5, dtype=torch.int64, device="cuda")
    monkeypatch.setenv("SPLAT_PAIR_INDEX", "eager")
    expected = _prune_exact_pairs(pp, ps, x, y, mu, aa, bb, dd, variance, 8.0, cc)
    monkeypatch.setenv("SPLAT_PAIR_INDEX", "fused")
    actual = _prune_exact_pairs(pp, ps, x, y, mu, aa, bb, dd, variance, 8.0, cc)
    assert torch.equal(actual[0], torch.arange(4, device="cuda"))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
