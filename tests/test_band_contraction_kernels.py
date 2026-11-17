from __future__ import annotations

import pytest
import torch

from survey_fusion.pgm.splat_spectrum import _DenseBandContraction, _SparseBandContraction
from survey_fusion.utils.perf_env import sparse_band_contraction_enabled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_sparse_band_contraction_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SPLAT_BAND_CONTRACTION", raising=False)
    assert not sparse_band_contraction_enabled()
    monkeypatch.setenv("SPLAT_BAND_CONTRACTION", "sparse")
    assert sparse_band_contraction_enabled()
    monkeypatch.setenv("SPLAT_BAND_CONTRACTION", "other")
    with pytest.raises(ValueError, match="must be 'dense' or 'sparse'"):
        sparse_band_contraction_enabled()


def _dense_weights(values, n_bands, placements, blocks):
    weights = values.new_zeros(values.shape[0], n_bands, values.shape[1])
    for (band, rows), block in zip(placements, blocks):
        weights[:, band].index_copy_(0, rows, block)
    return weights


def _run(contraction, values, n_bands, placements, blocks, cotangent):
    values = values.detach().requires_grad_(True)
    blocks = tuple(block.detach().requires_grad_(True) for block in blocks)
    output = contraction.apply(values, n_bands, placements, *blocks)
    gradients = torch.autograd.grad((output * cotangent).sum(), (values, *blocks))
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _cases(device):
    generator = torch.Generator(device=device).manual_seed(1701)
    base_values = torch.randn((13, 192), generator=generator, device=device)
    values = base_values[:, ::2]
    values[0, 0] = -0.0
    row_sets = (
        torch.tensor([10, 2, 5, 0], device=device),
        torch.tensor([2, 10, 6], device=device),
        torch.tensor([10, 1, 4], device=device),
        torch.tensor([10, 11], device=device),
        torch.tensor([10], device=device),
    )
    placements = tuple((band, rows) for band, rows in enumerate(row_sets))
    blocks = []
    for band, rows in placements:
        backing = torch.randn((rows.numel(), 192), generator=generator, device=device)
        block = backing[:, ::2]
        if band == 0:
            block[0, 0] = -0.0
        blocks.append(block)
    yield values, 5, placements, tuple(blocks)

    single = torch.randn((1, 96), generator=generator, device=device)
    single[0, 0] = -0.0
    rows = torch.tensor([0], device=device)
    block = torch.randn((1, 96), generator=generator, device=device)
    yield single, 3, ((1, rows),), (block,)


@pytest.mark.parametrize("case_index", [0, 1])
def test_sparse_band_contraction_matches_dense_and_float64(case_index: int) -> None:
    device = torch.device("cuda")
    values, n_bands, placements, blocks = list(_cases(device))[case_index]
    generator = torch.Generator(device=device).manual_seed(908 + case_index)
    cotangent = torch.randn(
        (values.shape[0], n_bands), generator=generator, device=device
    )

    dense_output, dense_grads = _run(
        _DenseBandContraction, values, n_bands, placements, blocks, cotangent
    )
    sparse_output, sparse_grads = _run(
        _SparseBandContraction, values, n_bands, placements, blocks, cotangent
    )

    torch.testing.assert_close(sparse_output, dense_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sparse_grads[0], dense_grads[0], rtol=1e-5, atol=1e-6)
    for sparse_grad, dense_grad in zip(sparse_grads[1:], dense_grads[1:]):
        assert sparse_grad.view(torch.int32).equal(dense_grad.view(torch.int32))

    weights64 = _dense_weights(values, n_bands, placements, blocks).to(torch.float64)
    values64 = values.to(torch.float64)
    cotangent64 = cotangent.to(torch.float64)
    control_output = torch.einsum("jg,jbg->jb", values64, weights64)
    control_grad_values = torch.einsum("jb,jbg->jg", cotangent64, weights64)
    torch.testing.assert_close(dense_output.double(), control_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sparse_output.double(), control_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dense_grads[0].double(), control_grad_values, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(sparse_grads[0].double(), control_grad_values, rtol=1e-6, atol=1e-6)
    dense_distance = (dense_output.double() - control_output).abs()
    sparse_distance = (sparse_output.double() - control_output).abs()
    assert sparse_distance.max() <= dense_distance.max()
    assert sparse_distance.square().mean() <= dense_distance.square().mean()

    assert torch.isfinite(sparse_output).all()
    assert torch.isfinite(sparse_grads[0]).all()
    zero_rows = torch.tensor([3, 7, 8, 9, 12], device=device)
    if case_index == 0:
        assert torch.equal(sparse_output[zero_rows], torch.zeros_like(sparse_output[zero_rows]))
        assert not torch.signbit(sparse_output[zero_rows]).any()


def test_sparse_band_contraction_production_scale() -> None:
    device = torch.device("cuda")
    rows = 1_000_000
    knots = 96
    generator = torch.Generator(device=device).manual_seed(314159)
    values = torch.randn((rows, knots), generator=generator, device=device) * 0.03
    values[0, 0] = -0.0
    band_rows = (
        torch.arange(0, rows, 2, device=device),
        torch.arange(rows - 1, -1, -3, device=device),
        torch.arange(1, rows, 4, device=device),
        torch.arange(2, rows, 5, device=device),
        torch.arange(3, rows, 6, device=device),
    )
    placements = tuple((band, selected) for band, selected in enumerate(band_rows))
    blocks = tuple(
        torch.rand((selected.numel(), knots), generator=generator, device=device) * 0.02
        for selected in band_rows
    )
    blocks[0][0, 0] = -0.0
    cotangent = torch.randn((rows, 5), generator=generator, device=device) * 0.01

    dense_output, dense_grads = _run(
        _DenseBandContraction, values, 5, placements, blocks, cotangent
    )
    sparse_output, sparse_grads = _run(
        _SparseBandContraction, values, 5, placements, blocks, cotangent
    )

    torch.testing.assert_close(sparse_output, dense_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sparse_grads[0], dense_grads[0], rtol=1e-5, atol=1e-6)
    for sparse_grad, dense_grad in zip(sparse_grads[1:], dense_grads[1:]):
        assert sparse_grad.view(torch.int32).equal(dense_grad.view(torch.int32))
    assert torch.isfinite(sparse_output).all()
    assert all(torch.isfinite(gradient).all() for gradient in sparse_grads)
