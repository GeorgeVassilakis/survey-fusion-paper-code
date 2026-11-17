import pytest
import torch

from survey_fusion.training.row_sparse_adam import RowSparseAdam

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("deterministic", [False, True])
def test_fused_adam_matches_eager_subnormal_parameter_and_moments(monkeypatch, deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        values = torch.logspace(-45, -10, 2000, device="cuda").reshape(1000, 2)
        values[::2] *= -1
        rows = torch.arange(1000, device="cuda")
        results = []
        for mode in ("eager", "fused"):
            monkeypatch.setenv("SPLAT_ADAM_PATH", mode)
            parameter = torch.nn.Parameter(torch.zeros_like(values))
            optimizer = RowSparseAdam([parameter], lr=0.001)
            for _ in range(3):
                parameter.grad = torch.sparse_coo_tensor(
                    rows[None], values, parameter.shape, is_coalesced=True
                )
                optimizer.step()
            results.append(
                (
                    parameter.detach().clone(),
                    optimizer.state[parameter]["exp_avg"].clone(),
                    optimizer.state[parameter]["exp_avg_sq"].clone(),
                )
            )
        for actual, expected in zip(results[1], results[0]):
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    finally:
        torch.use_deterministic_algorithms(previous)
