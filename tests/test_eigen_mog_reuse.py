from types import SimpleNamespace

import numpy as np
import pytest
import torch

from survey_fusion.pgm.splat_raster import PackedPointIndex
from survey_fusion.training.eigen_mog_renderer import EigenMogRenderer, eigen_mog_identity
from test_eigen_mog import write_sidecar


@pytest.mark.parametrize("execution", ["cached", "grouped"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk", [7, 1000])
def test_reuse_full_renderer_predictions_and_parameter_gradients(tmp_path, execution, device, chunk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(189)
    sidecar = write_sidecar(tmp_path / "psf.npz", k=10)
    with np.load(sidecar) as data:
        values = dict(data)
    rng = np.random.default_rng(29)
    values["means"] = rng.normal(0, .2, values["means"].shape)
    values["weights"][..., 0] += .3
    values["weights"][..., 1] -= .3  # Signed components must not be clipped.
    values["kl_coefficients"][:, 1, 1, 0] = .8
    values["kl_coefficients"][:, 2, 0, 1] = .4
    np.savez(sidecar, **values)
    _, identity = eigen_mog_identity(dict(eigen_mog_sidecar=str(sidecar), eigen_mog_support_radius=18))
    trainer = SimpleNamespace(eigen_mog_sidecar_path=str(sidecar), eigen_mog_config=identity,
                             eigen_mog_pair_chunk=chunk, device=torch.device(device),
                             exposures={i: SimpleNamespace(label=f"e{i:04d}", image_width=8) for i in range(2)})
    x, y = [torch.randn(12, dtype=torch.float64, device=device) for _ in range(2)]
    exposures = np.arange(12) % 2
    local = np.arange(12)
    jac = torch.tensor([[[1.0, .1], [-.1, .8]], [[.9, -.2], [.05, 1.2]]],
                       dtype=torch.float64, device=device)[torch.arange(12, device=device) % 2]
    pair_point = torch.arange(12, device=device).repeat_interleave(5)
    pair_active = torch.arange(5, device=device).repeat(12)
    index = PackedPointIndex(pair_point, torch.arange(5, device=device), pair_active, 60, 60)
    base_pos = torch.randn(5, 2, dtype=torch.float64, device=device)
    base_cov = torch.tensor([[.7, .03, .8]], dtype=torch.float64, device=device).repeat(5, 1)
    base_amp = torch.rand(5, 2, dtype=torch.float64, device=device)
    cotangent = torch.randn(12, dtype=torch.float64, device=device)
    results = []
    for mode in ("reference", execution):
        trainer.eigen_mog_execution = mode
        renderer = EigenMogRenderer(trainer)
        context = renderer.prepare(exposures, local, jac)
        pos, cov, amp = [t.clone().requires_grad_() for t in (base_pos, base_cov, base_amp)]
        model = SimpleNamespace(positions=lambda ids, **kw: pos[ids],
                                covariance_components=lambda ids, **kw: tuple(cov[ids].unbind(1)))
        out = renderer.predict(model, x, y, torch.arange(12, device=device) % 2, amp, index, context)
        grads = torch.autograd.grad((out * cotangent).sum(), (pos, cov, amp))
        results.append((out, *grads))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


def test_static_tables_reused_and_pixel_weights_exact(tmp_path):
    from survey_fusion.pgm.eigen_mog import EigenMixture
    sidecar = write_sidecar(tmp_path / "psf.npz", k=10)
    rep = EigenMixture.load(sidecar)
    e = torch.tensor([0, 0, 1, 17])
    r = torch.tensor([100., 300., 512., 1050.], dtype=torch.float64)
    c = r.flip(0)
    cached = rep.cached_pixel_tables(e, r, c, torch.float64)
    reference = rep.pixel_tables(e, r, c)
    torch.testing.assert_close(cached[0], reference[0], rtol=0, atol=0)
    for a, b in zip(cached[1:], reference[1:]):
        torch.testing.assert_close(a[e], b, rtol=0, atol=0)
    assert rep.device_tables("cpu", torch.float64) is rep.device_tables("cpu", torch.float64)
