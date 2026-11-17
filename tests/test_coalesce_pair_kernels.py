import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", ["overlap", "prefix", "equal", "disjoint", "unequal_prefix"])
def test_direct_two_block_coalescence(case):
    from survey_fusion.pgm.coalesce_pair_kernels import coalesce_pair

    cases = dict(
        overlap=([0, 2, 5, 9], [0, 1, 5, 7]),
        prefix=([0, 2, 5, 9], [0, 2]),
        equal=([0, 2, 5], [0, 2, 5]),
        disjoint=([0, 2, 4], [5, 8, 9]),
        unequal_prefix=([0, 2], [0, 2, 5, 9]),
    )
    a, b = (torch.tensor(x, device="cuda") for x in cases[case])
    parameters = []
    expected = []
    for width, reverse in [(None, False), (1, False), (2, True), (32, False), (7, True)]:
        shape = (10,) if width is None else (10, width)
        p = torch.nn.Parameter(torch.randn(shape, device="cuda"))
        indices = torch.cat((b, a) if reverse else (a, b))
        values_shape = (len(indices),) if width is None else (len(indices), width)
        values = torch.randn(values_shape, device="cuda")
        p.grad = torch.sparse_coo_tensor(indices[None], values, p.shape, check_invariants=False)
        expected.append(p.grad.coalesce())
        parameters.append(p)
    assert coalesce_pair([a, b], parameters)
    for p, wanted in zip(parameters, expected):
        assert torch.equal(p.grad.indices(), wanted.indices())
        torch.testing.assert_close(p.grad.values(), wanted.values(), rtol=0, atol=0)
