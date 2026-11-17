"""RowSparseAdam must be a bitwise drop-in for torch.optim.SparseAdam."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_strategy import SplatRefineStrategy
from survey_fusion.training.row_sparse_adam import RowSparseAdam

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _field(*, n_splats: int = 96, seed: int = 7, device: str = "cpu") -> SplatField:
    torch.manual_seed(seed)
    field = SplatField(
        n_splats=n_splats,
        n_templates=5,
        loglam_grid=torch.linspace(-1.0, 1.0, 24),
        init_scale=0.015,
    ).to(device)
    with torch.no_grad():
        field.mu.uniform_(-0.8, 0.8)
        field.log_scales.uniform_(math.log(0.008), math.log(0.025))
        field.theta.uniform_(0.0, math.pi)
        field.raw_flux.normal_(-3.0, 0.5)
        field.sed_coeffs.normal_(0.0, 1.0)
        field.sed_shift.normal_(0.0, 0.02)
    field.set_sparse_gradients(True)
    return field


def _accumulate_sparse_gradients(field: SplatField, rows: torch.Tensor) -> None:
    loss = (
        field.positions(rows, sorted_unique=True).square().sum()
        + field.covariances(rows, sorted_unique=True)[:, 0, 0].sum()
        + field.fluxes(rows, sorted_unique=True).square().sum()
        + field.sed_coefficients(rows, sorted_unique=True).square().sum()
        + field.spectral_shifts(rows, sorted_unique=True).square().sum()
    )
    loss.backward()


def _copy_gradients(source: SplatField, target: SplatField) -> None:
    for from_parameter, to_parameter in zip(
        source.per_splat_parameters(), target.per_splat_parameters()
    ):
        gradient = from_parameter.grad
        to_parameter.grad = None if gradient is None else gradient.clone()


def _assert_same_state(
    reference: torch.optim.Optimizer,
    candidate: torch.optim.Optimizer,
    reference_field: SplatField,
    candidate_field: SplatField,
    label: str,
) -> None:
    for reference_parameter, candidate_parameter in zip(
        reference_field.per_splat_parameters(),
        candidate_field.per_splat_parameters(),
    ):
        assert torch.equal(reference_parameter, candidate_parameter), label
        reference_state = reference.state[reference_parameter]
        candidate_state = candidate.state[candidate_parameter]
        assert reference_state["step"] == candidate_state["step"], label
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                reference_state[key], candidate_state[key]
            ), f"{label}: {key}"


def _assert_same_raw_bits(left: torch.Tensor, right: torch.Tensor) -> None:
    assert left.dtype is torch.float32
    assert right.dtype is torch.float32
    assert left.view(torch.int32).equal(right.view(torch.int32))
    assert torch.signbit(left).equal(torch.signbit(right))


def _assign_sparse_gradients(
    parameters: list[torch.nn.Parameter],
    rows: torch.Tensor,
    values: list[torch.Tensor],
    *,
    is_coalesced: bool = False,
) -> None:
    indices = rows.unsqueeze(0)
    for parameter, gradient_values in zip(parameters, values):
        parameter.grad = torch.sparse_coo_tensor(
            indices,
            gradient_values,
            size=parameter.shape,
            device=parameter.device,
            is_coalesced=is_coalesced,
        )


def _assert_optimizer_raw_bits(
    eager: RowSparseAdam,
    fused: RowSparseAdam,
    eager_parameters: list[torch.nn.Parameter],
    fused_parameters: list[torch.nn.Parameter],
) -> None:
    for eager_parameter, fused_parameter in zip(
        eager_parameters, fused_parameters
    ):
        _assert_same_raw_bits(eager_parameter, fused_parameter)
        assert eager_parameter.grad is not None
        assert fused_parameter.grad is not None
        assert eager_parameter.grad.is_sparse
        assert fused_parameter.grad.is_sparse
        assert eager_parameter.grad._indices().equal(
            fused_parameter.grad._indices()
        )
        _assert_same_raw_bits(
            eager_parameter.grad._values(), fused_parameter.grad._values()
        )
        eager_state = eager.state[eager_parameter]
        fused_state = fused.state[fused_parameter]
        assert eager_state["step"] == fused_state["step"]
        _assert_same_raw_bits(eager_state["exp_avg"], fused_state["exp_avg"])
        _assert_same_raw_bits(
            eager_state["exp_avg_sq"], fused_state["exp_avg_sq"]
        )


@pytest.mark.parametrize("device", DEVICES)
def test_matches_sparse_adam_bitwise_over_many_steps(device: str) -> None:
    reference_field = _field(device=device)
    candidate_field = _field(device=device)
    reference = torch.optim.SparseAdam(
        reference_field.sparse_param_groups_gsplat(), betas=(0.9, 0.999)
    )
    candidate = RowSparseAdam(
        candidate_field.sparse_param_groups_gsplat(), betas=(0.9, 0.999)
    )

    generator = torch.Generator().manual_seed(11)
    for step in range(25):
        count = int(torch.randint(1, 40, (1,), generator=generator).item())
        rows = torch.unique(
            torch.randint(0, 96, (count,), generator=generator)
        ).to(device)
        reference.zero_grad()
        candidate.zero_grad()
        _accumulate_sparse_gradients(reference_field, rows)
        # A second backward on an overlapping row set exercises the
        # accumulated, non-coalesced gradient the trainer actually hands over.
        _accumulate_sparse_gradients(reference_field, rows[: max(1, count // 2)])
        _copy_gradients(reference_field, candidate_field)
        reference.step()
        candidate.step()
        _assert_same_state(
            reference, candidate, reference_field, candidate_field, f"step {step}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_matches_sparse_adam_across_a_refine_event(device: str) -> None:
    reference_field = _field(device=device)
    candidate_field = _field(device=device)
    reference = torch.optim.SparseAdam(reference_field.sparse_param_groups_gsplat())
    candidate = RowSparseAdam(candidate_field.sparse_param_groups_gsplat())

    rows = torch.arange(0, 96, 3, device=device)
    for _ in range(3):
        reference.zero_grad()
        candidate.zero_grad()
        _accumulate_sparse_gradients(reference_field, rows)
        _copy_gradients(reference_field, candidate_field)
        reference.step()
        candidate.step()

    duplicate = torch.zeros(96, dtype=torch.bool, device=device)
    duplicate[torch.arange(0, 96, 8, device=device)] = True
    for field, optimizer in (
        (reference_field, reference),
        (candidate_field, candidate),
    ):
        SplatRefineStrategy._duplicate(
            SplatRefineStrategy, field, [optimizer], duplicate, step=3
        )
    assert reference_field.mu.shape[0] == 108
    assert candidate_field.mu.shape[0] == 108

    grown = torch.arange(0, 108, 2, device=device)
    for step in range(5):
        reference.zero_grad()
        candidate.zero_grad()
        _accumulate_sparse_gradients(reference_field, grown)
        _copy_gradients(reference_field, candidate_field)
        reference.step()
        candidate.step()
        _assert_same_state(
            reference,
            candidate,
            reference_field,
            candidate_field,
            f"post-refine step {step}",
        )


def test_state_dict_round_trips_with_sparse_adam() -> None:
    reference_field = _field()
    candidate_field = _field()
    reference = torch.optim.SparseAdam(reference_field.sparse_param_groups_gsplat())
    candidate = RowSparseAdam(candidate_field.sparse_param_groups_gsplat())

    rows = torch.arange(0, 96, 5)
    for _ in range(4):
        reference.zero_grad()
        _accumulate_sparse_gradients(reference_field, rows)
        reference.step()

    # A SparseAdam checkpoint has to resume into RowSparseAdam unchanged.
    with torch.no_grad():
        for source, target in zip(
            reference_field.per_splat_parameters(),
            candidate_field.per_splat_parameters(),
        ):
            target.copy_(source)
    candidate.load_state_dict(copy.deepcopy(reference.state_dict()))

    for _ in range(3):
        reference.zero_grad()
        candidate.zero_grad()
        _accumulate_sparse_gradients(reference_field, rows)
        _copy_gradients(reference_field, candidate_field)
        reference.step()
        candidate.step()
    _assert_same_state(
        reference, candidate, reference_field, candidate_field, "after resume"
    )

    # And the reverse: RowSparseAdam state has to load into SparseAdam.
    resumed_field = _field()
    resumed = torch.optim.SparseAdam(resumed_field.sparse_param_groups_gsplat())
    resumed.load_state_dict(copy.deepcopy(candidate.state_dict()))
    for parameter in resumed_field.per_splat_parameters():
        state = resumed.state[parameter]
        assert state["step"] == 7
        assert state["exp_avg"].shape == parameter.shape


def test_rejects_dense_gradients() -> None:
    field = _field()
    optimizer = RowSparseAdam(field.sparse_param_groups_gsplat())
    field.raw_flux.grad = torch.zeros_like(field.raw_flux)
    with pytest.raises(RuntimeError, match="does not support dense gradients"):
        optimizer.step()


def test_counts_a_step_for_an_empty_gradient() -> None:
    """SparseAdam bumps the bias correction even when no row moves."""
    field = _field()
    optimizer = RowSparseAdam(field.sparse_param_groups_gsplat())
    empty = torch.sparse_coo_tensor(
        torch.zeros((1, 0), dtype=torch.long),
        torch.zeros((0,)),
        size=field.raw_flux.shape,
    )
    field.raw_flux.grad = empty
    optimizer.step()
    assert optimizer.state[field.raw_flux]["step"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_path_matches_eager_raw_bits_for_unsorted_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager_parameter = torch.nn.Parameter(torch.randn(16, 2, device="cuda"))
    fused_parameter = torch.nn.Parameter(eager_parameter.detach().clone())
    eager = RowSparseAdam([eager_parameter])
    fused = RowSparseAdam([fused_parameter])
    rows = torch.tensor([12, 1, 9, 3], device="cuda")
    values = torch.tensor(
        [[-0.0, 0.0], [0.25, -0.5], [1.0, 2.0], [-3.0, 4.0]],
        device="cuda",
    )
    _assign_sparse_gradients([eager_parameter], rows, [values])
    _assign_sparse_gradients([fused_parameter], rows, [values.clone()])
    monkeypatch.setenv("SPLAT_ADAM_PATH", "eager")
    eager.step()
    monkeypatch.setenv("SPLAT_ADAM_PATH", "fused")
    fused.step()
    _assert_optimizer_raw_bits(
        eager, fused, [eager_parameter], [fused_parameter]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_path_matches_eager_at_production_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_rows = 10_000_000
    n_touched = 1_800_000
    widths = (2, 2, 1, 1, 32, 1)
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    eager_parameters = [
        torch.nn.Parameter(
            torch.randn(
                (n_rows,) if width == 1 else (n_rows, width),
                generator=generator,
                device="cuda",
            )
        )
        for width in widths
    ]
    fused_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in eager_parameters
    ]
    eager = RowSparseAdam(eager_parameters, lr=3e-4, betas=(0.9, 0.999))
    fused = RowSparseAdam(fused_parameters, lr=3e-4, betas=(0.9, 0.999))

    for _ in range(3):
        rows = torch.randperm(
            n_rows, generator=generator, device="cuda"
        )[:n_touched].sort().values
        values = [
            torch.randn(
                (n_touched,) if width == 1 else (n_touched, width),
                generator=generator,
                device="cuda",
            )
            for width in widths
        ]
        for gradient_values in values:
            flat = gradient_values.view(-1)
            flat[0] = -0.0
            flat[1] = 0.0
        _assign_sparse_gradients(
            eager_parameters, rows, values, is_coalesced=True
        )
        _assign_sparse_gradients(
            fused_parameters,
            rows,
            [value.clone() for value in values],
            is_coalesced=True,
        )
        monkeypatch.setenv("SPLAT_ADAM_PATH", "eager")
        eager.step()
        monkeypatch.setenv("SPLAT_ADAM_PATH", "fused")
        fused.step()
        _assert_optimizer_raw_bits(
            eager, fused, eager_parameters, fused_parameters
        )
