import pytest
import torch

from survey_fusion.pgm.splat_spectrum import _paired_gaussian_quadrature_eager, paired_gaussian_quadrature

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("rows,nodes", [(31, 1), (1000, 32), (40_001, 256)])
def test_fused_quadrature_preserves_all_production_gradients(monkeypatch, rows, nodes):
    generator = torch.Generator(device="cuda").manual_seed(92)
    points = [torch.randn((13, nodes), generator=generator, device="cuda") * 0.01 for _ in range(2)]
    weights = torch.rand((13, nodes), generator=generator, device="cuda")
    fibers = torch.randint(13, (rows,), generator=generator, device="cuda")
    mu = torch.randn((rows, 2), generator=generator, device="cuda") * 0.01
    precision = (
        torch.ones(rows, device="cuda") * 300,
        torch.zeros(rows, device="cuda"),
        torch.zeros(rows, device="cuda"),
        torch.ones(rows, device="cuda") * 400,
    )
    norm = torch.rand(rows, generator=generator, device="cuda")
    go = torch.randn(rows, generator=generator, device="cuda")
    go[::17] = -0.0

    def evaluate(fused):
        monkeypatch.setenv("SPLAT_FUSED_QUADRATURE", str(int(fused)))
        tensors = tuple(t.detach().requires_grad_() for t in (mu, *precision, norm))
        result = paired_gaussian_quadrature(*points, weights, fibers, tensors[0], tensors[1:5], tensors[5])
        return result, *torch.autograd.grad(result, tensors, go)

    reference, actual = evaluate(False), evaluate(True)
    for a, b in zip(actual, reference):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_differentiable_quadrature_data_keeps_eager_fallback(monkeypatch):
    monkeypatch.setenv("SPLAT_FUSED_QUADRATURE", "1")
    px = torch.ones((1, 4), device="cuda", requires_grad=True)
    py = torch.zeros_like(px)
    weights = torch.ones_like(px)
    fibers = torch.zeros(3, dtype=torch.long, device="cuda")
    mu = torch.zeros((3, 2), device="cuda")
    precision = (torch.ones(3, device="cuda"),) * 4
    norm = torch.ones(3, device="cuda")
    args = (px, py, weights, fibers, mu, precision, norm)
    ref = _paired_gaussian_quadrature_eager(*args)
    actual = paired_gaussian_quadrature(*args)
    assert torch.equal(ref.view(torch.int32), actual.view(torch.int32))
    assert torch.equal(torch.autograd.grad(ref.sum(), px)[0], torch.autograd.grad(actual.sum(), px)[0])
