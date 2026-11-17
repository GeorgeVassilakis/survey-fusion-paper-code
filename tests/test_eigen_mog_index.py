from types import SimpleNamespace
import os

import pytest
import torch

from survey_fusion.training.eigen_mog_renderer import EigenMogRenderer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("radius", [0.0, 16.0, 18.0])
def test_fused_index_preserves_exact_pairs(radius):
    torch.manual_seed(651)
    dev = "cuda"
    mu = torch.rand(2048, 2, device=dev) * 0.02
    logs = torch.empty(2048, 2, device=dev).uniform_(-15, -5)
    theta = torch.rand(2048, device=dev) * 6.28

    def cov():
        c, s = theta.cos(), theta.sin()
        a, b = logs.exp().square().unbind(1)
        return c * c * a + s * s * b, c * s * (a - b), s * s * a + c * c * b

    model = SimpleNamespace(mu=mu, log_scales=logs, theta=theta, covariance_components=cov)
    x, y = (torch.rand(512, device=dev) * 0.02 for _ in range(2))
    j = torch.tensor([[2.05e-4, 1.8e-6], [-1.2e-6, 2.02e-4]], device=dev, dtype=torch.float64)
    forward = j[None].repeat(512, 1, 1)
    forward[::3, 0, 1] *= 2
    context = dict(inverse=torch.linalg.inv(forward), forward=forward, wing=radius * forward.abs().sum(-1))
    renderer = SimpleNamespace(support_radius=radius)
    previous = os.environ.get("SPLAT_EIGEN_FUSED_INDEX")
    try:
        os.environ["SPLAT_EIGEN_FUSED_INDEX"] = "0"
        a = EigenMogRenderer.index(renderer, model, x, y, context, 8, chunk=17777)
        os.environ["SPLAT_EIGEN_FUSED_INDEX"] = "1"
        b = EigenMogRenderer.index(renderer, model, x, y, context, 8, chunk=17777)
    finally:
        if previous is None:
            os.environ.pop("SPLAT_EIGEN_FUSED_INDEX", None)
        else:
            os.environ["SPLAT_EIGEN_FUSED_INDEX"] = previous
    assert a.candidate_pairs == b.candidate_pairs
    for name in ["pair_point", "active_splat", "pair_active_splat"]:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
