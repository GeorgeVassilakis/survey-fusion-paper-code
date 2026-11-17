import pytest
import torch

from survey_fusion.models.splat_field import SplatField

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("usage", ["direct", "fiber_matrix", "raster_rows", "one_output"])
def test_fused_covariance_preserves_values_and_gradients(monkeypatch, usage):
    generator = torch.Generator(device="cuda").manual_seed(91)
    logs = torch.randn((4096, 2), generator=generator, device="cuda") - 4
    theta = torch.randn((4096,), generator=generator, device="cuda") * 3
    theta[:2] = torch.tensor([0.0, -0.0], device="cuda")
    width = {"direct": 3, "fiber_matrix": 4, "raster_rows": 4, "one_output": 1}[usage]
    grad = torch.randn((4096, width), generator=generator, device="cuda")
    grad[::73] = -0.0

    def evaluate(fused):
        monkeypatch.setenv("SPLAT_FUSED_COVARIANCE", str(int(fused)))
        ll = logs.detach().requires_grad_()
        tt = theta.detach().requires_grad_()
        a, b, d = SplatField.covariance_components_from_rows(ll, tt)
        if usage == "fiber_matrix":
            output = torch.stack((a, b, b, d), dim=1)
        elif usage == "raster_rows":
            output = torch.stack((a, d, 2.0 * b, b.square()), dim=1)
        elif usage == "one_output":
            output = a[:, None]
        else:
            output = torch.stack((a, b, d), dim=1)
        grads = torch.autograd.grad(output, (ll, tt), grad)
        return output, *grads

    expected = evaluate(False)
    actual = evaluate(True)
    for a, b in zip(actual, expected):
        mismatch = int((a.view(torch.int32) != b.view(torch.int32)).sum())
        assert mismatch == 0, f"{mismatch} raw-bit mismatches for {usage}"
