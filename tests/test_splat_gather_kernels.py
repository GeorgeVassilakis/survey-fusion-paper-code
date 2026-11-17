"""Bitwise gates for the fused six-parameter splat row gather."""

import copy

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.training.row_sparse_adam import RowSparseAdam


def _small_field() -> SplatField:
    field = SplatField(
        n_splats=16,
        n_templates=3,
        loglam_grid=torch.linspace(-1.0, 1.0, 7),
    ).cuda()
    field.set_sparse_gradients(True)
    with torch.no_grad():
        for parameter in field.per_splat_parameters():
            parameter.copy_(
                torch.arange(parameter.numel(), device="cuda").reshape(parameter.shape)
            )
            parameter.reshape(-1)[0] = -0.0
    return field


def _assert_float_bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def _legacy_gather(
    field: SplatField, indices: torch.Tensor, *, sorted_unique: bool,
) -> tuple[torch.Tensor, ...]:
    return (
        field.positions(indices, sorted_unique=sorted_unique),
        field._gather_parameter(
            field.log_scales, indices, sorted_unique=sorted_unique
        ),
        field._gather_parameter(field.theta, indices, sorted_unique=sorted_unique),
        field.raw_flux_values(indices, sorted_unique=sorted_unique),
        field.sed_coefficients(indices, sorted_unique=sorted_unique),
        field.spectral_shifts(indices, sorted_unique=sorted_unique),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("sorted_unique", [True, False])
@pytest.mark.parametrize("with_grad", [True, False])
def test_fused_gather_preserves_strided_indices(sorted_unique, with_grad) -> None:
    reference = _small_field()
    candidate = copy.deepcopy(reference)
    storage = torch.arange(16, device="cuda")
    if not sorted_unique:
        storage[3] = 1  # Duplicate logical index in the strided view.
    indices = storage[1::2]
    assert not indices.is_contiguous() and indices.storage_offset() != 0
    with torch.set_grad_enabled(with_grad):
        expected = _legacy_gather(reference, indices, sorted_unique=sorted_unique)
        actual = candidate.gather_splat_parameters(indices, sorted_unique=sorted_unique)
        for a, b in zip(actual, expected):
            _assert_float_bits_equal(a, b)
        if with_grad:
            for outputs in (expected, actual):
                torch.autograd.backward(outputs, tuple(torch.ones_like(t) for t in outputs))
            for a, b in zip(candidate.per_splat_parameters(), reference.per_splat_parameters()):
                assert torch.equal(a.grad.coalesce().indices(), b.grad.coalesce().indices())
                _assert_float_bits_equal(a.grad.coalesce().values(), b.grad.coalesce().values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("used_output", [0, 3])
def test_unused_gather_outputs_leave_adam_parameters_and_state_untouched(used_output) -> None:
    reference = _small_field()
    candidate = copy.deepcopy(reference)
    fields = (reference, candidate)
    optimizers = tuple(RowSparseAdam(f.per_splat_parameters(), lr=0.01) for f in fields)
    indices = torch.arange(8, device="cuda")
    for field, optimizer in zip(fields, optimizers):
        # First give every parameter nonzero Adam momentum.
        outputs = _legacy_gather(field, indices, sorted_unique=True)
        torch.autograd.backward(outputs, tuple(torch.ones_like(t) for t in outputs))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    before = tuple(p.detach().clone() for p in reference.per_splat_parameters())
    expected = _legacy_gather(reference, indices, sorted_unique=True)
    actual = candidate.gather_splat_parameters(indices, sorted_unique=True)
    expected[used_output].sum().backward()
    actual[used_output].sum().backward()
    for i, (a, b) in enumerate(zip(candidate.per_splat_parameters(), reference.per_splat_parameters())):
        assert (a.grad is None) == (b.grad is None) == (i != used_output)
    for optimizer in optimizers:
        optimizer.step()
    for i, (a, b) in enumerate(zip(candidate.per_splat_parameters(), reference.per_splat_parameters())):
        _assert_float_bits_equal(a, b)
        actual_state, expected_state = optimizers[1].state[a], optimizers[0].state[b]
        assert actual_state["step"] == expected_state["step"] == (2 if i == used_output else 1)
        for name in ("exp_avg", "exp_avg_sq"):
            _assert_float_bits_equal(actual_state[name], expected_state[name])
        if i != used_output:
            _assert_float_bits_equal(a, before[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case", ["float64", "strided_parameters"])
@pytest.mark.parametrize("with_grad", [True, False])
def test_unsupported_gather_storage_uses_reference_path(monkeypatch, case, with_grad) -> None:
    from survey_fusion.pgm import splat_gather_kernels

    field = _small_field()
    if case == "float64":
        field = field.double()
    else:
        for name in ("mu", "log_scales", "theta", "raw_flux", "sed_coeffs", "sed_shift"):
            parameter = getattr(field, name)
            storage = torch.empty((32, *parameter.shape[1:]), device="cuda")
            storage[::2].copy_(parameter.detach())
            setattr(field, name, torch.nn.Parameter(storage[::2]))
    reference = copy.deepcopy(field)
    indices = torch.arange(0, 16, 2, device="cuda")

    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Unsupported storage reached the fused kernel")

    monkeypatch.setattr(splat_gather_kernels, "gather_splat_rows", unexpected_kernel)
    with torch.set_grad_enabled(with_grad):
        actual = field.gather_splat_parameters(indices, sorted_unique=True)
        expected = _legacy_gather(reference, indices, sorted_unique=True)
        bits = torch.int64 if case == "float64" else torch.int32
        for a, b in zip(actual, expected):
            assert torch.equal(a.view(bits), b.view(bits))
        if with_grad:
            for outputs in (actual, expected):
                torch.autograd.backward(outputs, tuple(torch.ones_like(t) for t in outputs))
            for a, b in zip(field.per_splat_parameters(), reference.per_splat_parameters()):
                assert torch.equal(a.grad.coalesce().indices(), b.grad.coalesce().indices())
                assert torch.equal(a.grad.coalesce().values().view(bits), b.grad.coalesce().values().view(bits))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("sorted_unique", [True, False])
def test_fused_gather_is_bitwise_at_production_like_shape(sorted_unique) -> None:
    device = torch.device("cuda")
    n_splats = 1_050_017
    n_rows = 1_000_003
    torch.manual_seed(2401)
    reference = SplatField(
        n_splats=n_splats,
        n_templates=32,
        loglam_grid=torch.linspace(-1.0, 1.0, 64),
    ).to(device)
    candidate = SplatField(
        n_splats=n_splats,
        n_templates=32,
        loglam_grid=torch.linspace(-1.0, 1.0, 64),
    ).to(device)
    candidate.load_state_dict(reference.state_dict())
    reference.set_sparse_gradients(True)
    candidate.set_sparse_gradients(True)
    with torch.no_grad():
        for ref_parameter, cand_parameter in zip(
            reference.per_splat_parameters(), candidate.per_splat_parameters()
        ):
            ref_parameter.reshape(-1)[0] = -0.0
            cand_parameter.reshape(-1)[0] = -0.0

    indices = torch.arange(n_rows, device=device)
    if not sorted_unique:
        indices = indices.roll(17)
        indices[-1] = indices[0]
    old_outputs = _legacy_gather(
        reference, indices, sorted_unique=sorted_unique
    )
    new_outputs = candidate.gather_splat_parameters(
        indices, sorted_unique=sorted_unique
    )
    for actual, expected in zip(new_outputs, old_outputs):
        _assert_float_bits_equal(actual, expected)

    generator = torch.Generator(device=device).manual_seed(9307)
    output_gradients = [
        torch.randn(output.shape, generator=generator, device=device)
        for output in old_outputs
    ]
    for gradient in output_gradients:
        gradient.reshape(-1)[0] = -0.0
    torch.autograd.backward(old_outputs, output_gradients)
    torch.autograd.backward(new_outputs, output_gradients)

    for actual, expected in zip(
        candidate.per_splat_parameters(), reference.per_splat_parameters()
    ):
        actual_gradient = actual.grad.coalesce()
        expected_gradient = expected.grad.coalesce()
        assert torch.equal(actual_gradient.indices(), expected_gradient.indices())
        _assert_float_bits_equal(
            actual_gradient.values(), expected_gradient.values()
        )

    reference_optimizer = RowSparseAdam(reference.per_splat_parameters(), lr=1e-3)
    candidate_optimizer = RowSparseAdam(candidate.per_splat_parameters(), lr=1e-3)
    reference_optimizer.step()
    candidate_optimizer.step()
    for actual, expected in zip(
        candidate.per_splat_parameters(), reference.per_splat_parameters()
    ):
        _assert_float_bits_equal(actual, expected)
        actual_state = candidate_optimizer.state[actual]
        expected_state = reference_optimizer.state[expected]
        assert actual_state["step"] == expected_state["step"]
        _assert_float_bits_equal(actual_state["exp_avg"], expected_state["exp_avg"])
        _assert_float_bits_equal(
            actual_state["exp_avg_sq"], expected_state["exp_avg_sq"]
        )
