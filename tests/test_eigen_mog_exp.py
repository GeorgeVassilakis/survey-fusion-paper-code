import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp64_exp_range_reduction():
    import triton
    import triton.language as tl
    from survey_fusion.pgm.eigen_mog_full import _exp_polynomial

    @triton.jit
    def evaluate(x, y, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(x + i, i < n, 0.0)
        tl.store(y + i, _exp_polynomial(value), i < n)

    x = torch.linspace(-700.0, 0.0, 200001, device="cuda", dtype=torch.float64)
    edges = (torch.arange(1000, device="cuda", dtype=torch.float64) + 0.5) * (
        -torch.log(torch.tensor(2.0, device="cuda", dtype=torch.float64))
    )
    x = torch.cat(
        (
            x,
            edges,
            torch.tensor(
                [-float("inf"), -745.0, -744.0, -710.0, -709.0, 0.0, float("nan")],
                device="cuda",
                dtype=torch.float64,
            ),
        )
    )
    y = torch.empty_like(x)
    evaluate[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    ref = torch.exp(x)
    regular = (ref > torch.finfo(torch.float64).tiny) & torch.isfinite(ref)
    assert float(((y[regular] - ref[regular]).abs() / ref[regular]).max()) < 2e-14
    torch.testing.assert_close(
        y[~regular], ref[~regular], rtol=0, atol=torch.finfo(torch.float64).tiny * 2e-14, equal_nan=True
    )
