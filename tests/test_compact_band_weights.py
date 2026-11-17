import pytest
import torch

from survey_fusion.pgm.splat_spectrum import _hat_band_tables, _hat_window_width, _HatBandWeights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("rows", [31, 1000, 4096, 200_001])
def test_compact_hat_weights_preserve_dense_values_and_shift_gradients(rows):
    from survey_fusion.pgm.compact_band_kernels import CompactHatWeights

    generator = torch.Generator(device="cuda").manual_seed(815)
    C = torch.logspace(3.47712125472, 4.04139268516, 96, device="cuda")
    lam = torch.linspace(4000, 7400, 61, device="cuda")
    throughput = torch.sin(torch.linspace(0, torch.pi, 61, device="cuda")).clamp_min(0)
    tables = _hat_band_tables(lam, throughput, dtype=torch.float32, device=torch.device("cuda"))
    window = _hat_window_width(C, tables[-2], tables[-1])
    scale = torch.exp(torch.linspace(-0.8, 0.8, rows, device="cuda"))
    go = torch.randn((rows, 96), generator=generator, device="cuda")
    go[::97] = -0.0
    t = scale.detach().requires_grad_()
    dense = _HatBandWeights.apply(t, C, *tables, window, True)
    reference_grad = torch.autograd.grad(dense, t, go)[0]
    t = scale.detach().requires_grad_()
    compact, starts = CompactHatWeights.apply(t, C, tables, window)
    indices = starts[:, None] + torch.arange(window, device="cuda")[None, :]
    compact_grad = torch.autograd.grad(compact, t, go.gather(1, indices))[0]
    restored = torch.zeros_like(dense).scatter(1, indices, compact)
    assert torch.equal(restored.view(torch.int32), dense.view(torch.int32))
    assert torch.equal(compact_grad.view(torch.int32), reference_grad.view(torch.int32))
