"""Bitwise gate for overlapping the image and fiber forward branches."""

import pytest
import torch

from survey_fusion.models.splat_field import _SparseRowGather
from survey_fusion.training.datum_trainer import DatumTrainer
from survey_fusion.training.row_sparse_adam import RowSparseAdam
from survey_fusion.utils.perf_env import fiber_side_stream_enabled


class _StreamShell:
    _compute_step_gaussian_terms = DatumTrainer._compute_step_gaussian_terms
    _join_fiber_side_stream = DatumTrainer._join_fiber_side_stream

    def __init__(self, rows: torch.Tensor, shared: torch.Tensor, *, sorted_fiber: bool):
        self.device = rows.device
        self.rows = torch.nn.Parameter(rows)
        self.shared = torch.nn.Parameter(shared)
        self.sorted_fiber = bool(sorted_fiber)
        self.analytic_images_enabled = True
        self.analytic_spectra_enabled = True
        self._fiber_side_stream = (
            torch.cuda.Stream(device=self.device)
            if fiber_side_stream_enabled()
            else None
        )
        self._fiber_side_stream_used = False

    def _compute_analytic_image_gaussian_terms(
        self, indices: torch.Tensor,
    ) -> torch.Tensor:
        gathered = _SparseRowGather.apply(self.rows, indices, True)
        return gathered[:, 0] * self.shared[0]

    def _compute_analytic_spectral_gaussian_terms(
        self, indices: torch.Tensor,
    ) -> torch.Tensor:
        gathered = _SparseRowGather.apply(
            self.rows, indices, self.sorted_fiber
        )
        return gathered[:, 1] * self.shared[0]


def _initial_rows(n_rows: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(n_rows, dtype=torch.float32)
    values = torch.stack(
        ((row.remainder(257.0) - 128.0) / 257.0,
         (row.remainder(509.0) - 254.0) / 509.0),
        dim=1,
    )
    values[0, 0] = -0.0
    values[1, 1] = -0.0
    return values.to(device=device)


def _snapshot_optimizer(
    optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter,
) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for key, value in optimizer.state[parameter].items():
        snapshot[key] = value.detach().clone() if torch.is_tensor(value) else value
    return snapshot


def _run_arm(
    monkeypatch: pytest.MonkeyPatch, stream_value: str, *, sorted_fiber: bool,
) -> dict[str, object]:
    monkeypatch.setenv("SPLAT_FIBER_STREAM", stream_value)
    device = torch.device("cuda")
    n_rows = 1_000_003
    shell = _StreamShell(
        _initial_rows(n_rows, device),
        torch.tensor([1.0, -0.0], dtype=torch.float32, device=device),
        sorted_fiber=sorted_fiber,
    )
    image_indices = torch.arange(0, n_rows, 2, device=device)
    if sorted_fiber:
        fiber_indices = torch.arange(1, n_rows, 3, device=device)
    else:
        fiber_indices = torch.arange(n_rows - 1, 0, -3, device=device)
    row_optimizer = RowSparseAdam([shell.rows], lr=3.0e-4)
    shared_optimizer = torch.optim.Adam([shell.shared], lr=3.0e-4)
    outputs = []
    gradients = []

    for _ in range(2):
        row_optimizer.zero_grad()
        shared_optimizer.zero_grad()
        terms = shell._compute_step_gaussian_terms(
            image_indices, fiber_indices
        )
        assert terms is not None
        image_terms = terms[:image_indices.numel()]
        fiber_terms = terms[image_indices.numel():]
        image_loss = image_terms.sum()
        fiber_loss = fiber_terms.sum()
        loss = image_loss * 0.75 + fiber_loss * 1.25
        loss.backward()
        shell._join_fiber_side_stream()
        shell.rows.grad = shell.rows.grad.coalesce()
        outputs.append(
            tuple(
                value.detach().clone()
                for value in (terms, image_terms, fiber_terms, image_loss, fiber_loss, loss)
            )
        )
        gradients.append(
            (
                shell.rows.grad.indices().detach().clone(),
                shell.rows.grad.values().detach().clone(),
                shell.shared.grad.detach().clone(),
            )
        )
        row_optimizer.step()
        shared_optimizer.step()
        del terms, image_terms, fiber_terms, image_loss, fiber_loss, loss

    return {
        "outputs": outputs,
        "gradients": gradients,
        "rows": shell.rows.detach().clone(),
        "shared": shell.shared.detach().clone(),
        "row_state": _snapshot_optimizer(row_optimizer, shell.rows),
        "shared_state": _snapshot_optimizer(shared_optimizer, shell.shared),
    }


def _assert_bitwise_equal(actual: object, expected: object) -> None:
    if torch.is_tensor(actual):
        assert torch.is_tensor(expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
        if actual.is_floating_point():
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))
            if actual.dtype == torch.float32:
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_bitwise_equal(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_bitwise_equal(actual_item, expected_item)
        return
    assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("sorted_fiber", [True, False])
def test_fiber_side_stream_is_bitwise_for_outputs_gradients_and_optimizer_state(
    monkeypatch: pytest.MonkeyPatch, sorted_fiber: bool,
) -> None:
    old = _run_arm(monkeypatch, "default", sorted_fiber=sorted_fiber)
    new = _run_arm(monkeypatch, "side", sorted_fiber=sorted_fiber)

    assert torch.signbit(old["outputs"][0][0]).any()
    _assert_bitwise_equal(new, old)


def test_fiber_stream_switch_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLAT_FIBER_STREAM", "unknown")
    with pytest.raises(ValueError, match="SPLAT_FIBER_STREAM"):
        fiber_side_stream_enabled()
