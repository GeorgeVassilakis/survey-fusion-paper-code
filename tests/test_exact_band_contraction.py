import pytest
import torch

from survey_fusion.pgm.splat_spectrum import _DenseBandContraction, _ExactBandContraction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("rows", [4096, 201_001])
@pytest.mark.parametrize("reverse", [False, True])
def test_cublas_order_contraction_matches_all_raw_bits(rows, reverse):
    from survey_fusion.pgm.exact_band_contraction_kernels import applies

    generator = torch.Generator(device="cuda").manual_seed(8123)
    values = torch.randn((rows, 96), generator=generator, device="cuda") * 0.03
    values[0, 0] = -0.0
    selected = tuple(torch.arange(band, rows, band + 2, device="cuda") for band in range(5))
    if reverse:
        selected = tuple(r.flip(0) for r in selected)
    placements = tuple(enumerate(selected))
    blocks = tuple(torch.randn((r.numel(), 96), generator=generator, device="cuda") * 0.02 for r in selected)
    assert applies(values, 5, placements, blocks)
    grad = torch.randn((rows, 5), generator=generator, device="cuda")
    grad[::53] = -0.0
    blocks[0][0, 0] = -0.0

    def evaluate(cls):
        v = values.detach().requires_grad_()
        bs = tuple(b.detach().requires_grad_() for b in blocks)
        out = cls.apply(v, 5, placements, *bs)
        return out, *torch.autograd.grad(out, (v, *bs), grad)

    reference = evaluate(_DenseBandContraction)
    actual = evaluate(_ExactBandContraction)
    for a, b in zip(actual, reference):
        assert bool(torch.isfinite(a).all())
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_unverified_cublas_shapes_are_ineligible():
    from survey_fusion.pgm.exact_band_contraction_kernels import applies

    values = torch.ones((13, 96), device="cuda")
    selected = torch.arange(13, device="cuda")
    assert not applies(values, 5, [(0, selected)], [values])
    large = torch.ones((4096, 96), device="cuda")
    selected = torch.arange(4096, device="cuda")
    assert not applies(large, 3, [(0, selected)], [large])
    assert not applies(large[:, ::2], 5, [(0, selected)], [large[:, ::2]])


@pytest.mark.parametrize("rows", [65536, 65548, 65599, 65664, 983092, 983045])
def test_short_cublas_batch_tails_use_the_original_path(rows):
    from survey_fusion.utils.perf_env import exact_band_reduction_rows_supported

    assert not exact_band_reduction_rows_supported(rows)
