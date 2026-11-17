from __future__ import annotations

import numpy as np
import pytest
import torch

from survey_fusion.utils.torch_utils import (
    as_tensor_on_device,
    copy_tensor_to_device,
    tensor_on_device,
)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.int64, [3, -1, 0, 8]),
        (torch.float32, [0.0, -0.0, 1.25, -3.5]),
        (torch.float64, [0.0, -0.0, 1.25, -3.5]),
    ],
)
def test_host_transfer_helpers_match_plain_cpu_path(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    values: list[float | int],
) -> None:
    source = np.asarray(values, dtype=torch.empty((), dtype=dtype).numpy().dtype)
    expected = torch.as_tensor(source, dtype=dtype, device="cpu")

    monkeypatch.setenv("SPLAT_PINNED_H2D", "1")
    created = as_tensor_on_device(source, dtype=dtype, device="cpu")
    constructed = tensor_on_device(source, dtype=dtype, device="cpu")
    copied = copy_tensor_to_device(torch.from_numpy(source), device="cpu")

    for actual in (created, constructed, copied):
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
        if dtype.is_floating_point:
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_pinned_h2d_switch_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPLAT_PINNED_H2D", "yes")
    with pytest.raises(ValueError, match="SPLAT_PINNED_H2D"):
        as_tensor_on_device([1.0], dtype=torch.float32, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("sorted_indices", [True, False])
def test_pinned_h2d_large_cuda_copy_is_bitwise_exact(
    monkeypatch: pytest.MonkeyPatch, sorted_indices: bool,
) -> None:
    n_rows = 1_000_003
    indices = torch.arange(n_rows, dtype=torch.int64)
    if not sorted_indices:
        indices = indices.roll(n_rows // 3)
    values = torch.linspace(-4.0, 7.0, n_rows, dtype=torch.float32)
    values[0] = -0.0
    values[1] = 0.0
    state = torch.linspace(-2.0, 3.0, n_rows, dtype=torch.float64)
    state[0] = -0.0

    monkeypatch.setenv("SPLAT_PINNED_H2D", "0")
    expected = tuple(
        as_tensor_on_device(value, device="cuda")
        for value in (indices, values, state)
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("SPLAT_PINNED_H2D", "1")
    actual = tuple(
        as_tensor_on_device(value, device="cuda")
        for value in (indices, values, state)
    )
    torch.cuda.synchronize()

    for reference, candidate in zip(expected, actual):
        assert reference.dtype == candidate.dtype
        assert reference.shape == candidate.shape
        assert torch.equal(reference, candidate)
        if reference.is_floating_point():
            int_dtype = torch.int32 if reference.dtype == torch.float32 else torch.int64
            assert torch.equal(reference.view(int_dtype), candidate.view(int_dtype))
            assert torch.equal(torch.signbit(reference), torch.signbit(candidate))

    reference_parameter = expected[1].detach().requires_grad_()
    candidate_parameter = actual[1].detach().requires_grad_()
    reference_output = reference_parameter[expected[0]]
    candidate_output = candidate_parameter[actual[0]]
    assert torch.equal(reference_output.view(torch.int32), candidate_output.view(torch.int32))

    reference_loss = reference_output.square().sum()
    candidate_loss = candidate_output.square().sum()
    reference_loss.backward()
    candidate_loss.backward()
    assert torch.equal(reference_loss.view(torch.int32), candidate_loss.view(torch.int32))
    assert torch.equal(
        reference_parameter.grad.view(torch.int32),
        candidate_parameter.grad.view(torch.int32),
    )
    assert torch.equal(
        torch.signbit(reference_parameter.grad),
        torch.signbit(candidate_parameter.grad),
    )

    reference_optimizer = torch.optim.Adam(
        [reference_parameter], lr=1.0e-3, foreach=False
    )
    candidate_optimizer = torch.optim.Adam(
        [candidate_parameter], lr=1.0e-3, foreach=False
    )
    reference_optimizer.step()
    candidate_optimizer.step()
    assert torch.equal(
        reference_parameter.view(torch.int32), candidate_parameter.view(torch.int32)
    )
    reference_state = reference_optimizer.state[reference_parameter]
    candidate_state = candidate_optimizer.state[candidate_parameter]
    assert reference_state.keys() == candidate_state.keys()
    for key in reference_state:
        assert torch.equal(reference_state[key], candidate_state[key])
        if reference_state[key].dtype == torch.float32:
            assert torch.equal(
                reference_state[key].view(torch.int32),
                candidate_state[key].view(torch.int32),
            )
