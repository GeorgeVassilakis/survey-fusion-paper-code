"""The sort-free coalesce of the per-splat gradients reproduces coalesce().

Every per-splat parameter is gathered once by the image path and once by the
fiber path, so its gradient is a concatenation of coalesced COO blocks. The
trainer rebuilds the coalesced tensor from the union of the gathered index
sets instead of sorting the concatenation; these tests pin both halves of that
claim: the gather log records what was gathered, and the assembled gradient is
bitwise what `coalesce()` returns.
"""
import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.training.datum_trainer import DatumTrainer


class _Shell:
    _base_model = DatumTrainer._base_model
    _assemble_coalesced_rows = staticmethod(DatumTrainer._assemble_coalesced_rows)
    _coalesce_sparse_gradients = DatumTrainer._coalesce_sparse_gradients


def _field():
    return SplatField(
        n_splats=12,
        n_templates=3,
        loglam_grid=torch.linspace(-1.0, 1.0, 8),
        init_scale=0.05,
    )


def _shell(field):
    shell = _Shell()
    shell.model = field
    shell.params = list(field.parameters())
    return shell


def _accumulate(parameter, index_blocks, generator):
    """The gradient autograd hands back: one COO block per gather, concatenated.

    Built directly rather than by adding the blocks, because torch's sparse
    add coalesces eagerly on the host: accumulating here would hand the
    trainer a tensor it never sees on the device, and every assertion below
    would pass without the assembly running at all.
    """
    values = torch.cat(
        [
            torch.randn(
                (indices.numel(),) + tuple(parameter.shape[1:]),
                generator=generator,
            )
            for indices in index_blocks
        ]
    )
    parameter.grad = torch.sparse_coo_tensor(
        torch.cat(index_blocks).reshape(1, -1),
        values,
        size=parameter.shape,
        check_invariants=False,
    )


def test_gather_log_records_every_gather_and_stops_when_taken():
    field = _field()
    field.set_sparse_gradients(True)
    image_rows = torch.tensor([0, 3, 7])
    fiber_rows = torch.tensor([3, 9])
    field.start_sparse_gather_log()
    field.positions(image_rows, sorted_unique=True)
    field.positions(fiber_rows, sorted_unique=True)
    field.fluxes(image_rows, sorted_unique=True)

    log = field.take_sparse_gather_log()

    assert set(log) == {id(field.mu), id(field.raw_flux)}
    assert [t.tolist() for t in log[id(field.mu)]] == [[0, 3, 7], [3, 9]]
    assert [t.tolist() for t in log[id(field.raw_flux)]] == [[0, 3, 7]]
    assert field.take_sparse_gather_log() is None
    # Recording is off again, so a later gather does not grow a stale log.
    field.positions(fiber_rows, sorted_unique=True)
    assert field.take_sparse_gather_log() is None


def test_no_log_is_kept_without_sparse_gradients():
    field = _field()
    field.start_sparse_gather_log()
    field.positions(torch.tensor([1, 2]), sorted_unique=True)
    assert field.take_sparse_gather_log() == {}


def test_unverified_gather_disables_assembly_for_that_parameter():
    field = _field()
    field.set_sparse_gradients(True)
    field.start_sparse_gather_log()
    field.positions(torch.tensor([0, 3]), sorted_unique=True)
    field.positions(torch.tensor([3, 0, 3]))
    # A later trusted gather must not make the earlier one eligible again.
    field.positions(torch.tensor([1, 4]), sorted_unique=True)
    field.fluxes(torch.tensor([1, 4]), sorted_unique=True)

    log = field.take_sparse_gather_log()
    assert log[id(field.mu)] is None
    assert len(log[id(field.raw_flux)]) == 1


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_three_gather_blocks_use_reference_coalescence(device, monkeypatch):
    """Three contributions can cancel differently when their sum is reordered."""
    field = _field().to(device)
    field.set_sparse_gradients(True)
    blocks = [torch.arange(12, device=device) for _ in range(3)]
    values = torch.tensor([1e20, -1e20, 1.0], device=device)
    values = values.repeat_interleave(12).unsqueeze(1).expand(-1, 2).clone()
    field.mu.grad = torch.sparse_coo_tensor(
        torch.cat(blocks).unsqueeze(0), values, size=field.mu.shape,
        check_invariants=False,
    )
    reference = field.mu.grad.coalesce()
    field.start_sparse_gather_log()
    field._sparse_gather_log[id(field.mu)] = blocks
    shell = _shell(field)

    def refuse_assembly(*args):
        pytest.fail("Three gather blocks must use ordinary coalesce().")

    monkeypatch.setattr(shell, "_assemble_coalesced_rows", refuse_assembly)
    shell._coalesce_sparse_gradients()

    assert torch.equal(field.mu.grad.indices(), reference.indices())
    assert torch.equal(field.mu.grad.values(), reference.values())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_assembled_gradients_are_bitwise_coalesce(seed):
    generator = torch.Generator().manual_seed(seed)
    field = _field()
    field.set_sparse_gradients(True)
    # Two ascending duplicate-free blocks that overlap, as the image and fiber
    # gathers do: rows 4 and 8 carry two values, the rest carry one.
    blocks = [torch.tensor([0, 2, 4, 8, 11]), torch.tensor([1, 4, 8])]
    parameters = [field.mu, field.log_scales, field.theta, field.sed_coeffs]
    for parameter in parameters:
        _accumulate(parameter, blocks, generator)
        # Otherwise the trainer would skip it and prove nothing.
        assert not parameter.grad.is_coalesced()
    expected = [parameter.grad.coalesce() for parameter in parameters]

    field.start_sparse_gather_log()
    for parameter in parameters:
        field._sparse_gather_log[id(parameter)] = list(blocks)
    _shell(field)._coalesce_sparse_gradients()

    for parameter, reference in zip(parameters, expected):
        assert parameter.grad.is_coalesced()
        assert torch.equal(parameter.grad._indices(), reference._indices())
        assert torch.equal(parameter.grad._values(), reference._values())


def test_disjoint_and_identical_blocks_both_reproduce_coalesce():
    generator = torch.Generator().manual_seed(7)
    for blocks in (
        [torch.tensor([0, 1]), torch.tensor([5, 9])],
        [torch.tensor([2, 6]), torch.tensor([2, 6])],
        [torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])],
    ):
        field = _field()
        field.set_sparse_gradients(True)
        _accumulate(field.mu, blocks, generator)
        assert not field.mu.grad.is_coalesced()
        reference = field.mu.grad.coalesce()
        field.start_sparse_gather_log()
        field._sparse_gather_log[id(field.mu)] = list(blocks)
        _shell(field)._coalesce_sparse_gradients()
        assert torch.equal(field.mu.grad._indices(), reference._indices())
        assert torch.equal(field.mu.grad._values(), reference._values())


def test_gradient_unaccounted_by_the_log_falls_back_to_coalesce():
    """A logged gather that never reached the loss must not shrink the rows."""
    generator = torch.Generator().manual_seed(3)
    field = _field()
    field.set_sparse_gradients(True)
    _accumulate(field.mu, [torch.tensor([0, 5])], generator)
    reference = field.mu.grad.coalesce()

    field.start_sparse_gather_log()
    # The log claims a second gather the gradient does not carry.
    field._sparse_gather_log[id(field.mu)] = [
        torch.tensor([0, 5]), torch.tensor([7])
    ]
    _shell(field)._coalesce_sparse_gradients()

    assert torch.equal(field.mu.grad._indices(), reference._indices())
    assert torch.equal(field.mu.grad._values(), reference._values())


def test_already_coalesced_gradients_are_left_alone():
    field = _field()
    field.set_sparse_gradients(True)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    field.mu.grad = torch.sparse_coo_tensor(
        torch.tensor([[1, 6]]), values, size=field.mu.shape,
        check_invariants=False, is_coalesced=True,
    )
    before = field.mu.grad
    field.start_sparse_gather_log()
    _shell(field)._coalesce_sparse_gradients()
    assert field.mu.grad is before


def test_a_group_spanning_two_row_counts_is_refused():
    generator = torch.Generator().manual_seed(5)
    field = _field()
    field.set_sparse_gradients(True)
    short = torch.nn.Parameter(torch.zeros(4))
    blocks = [torch.tensor([0, 2]), torch.tensor([1, 2])]
    _accumulate(field.mu, blocks, generator)
    _accumulate(short, blocks, generator)
    shell = _shell(field)
    shell.params = [field.mu, short]
    field.start_sparse_gather_log()
    field._sparse_gather_log[id(field.mu)] = list(blocks)
    field._sparse_gather_log[id(short)] = list(blocks)

    with pytest.raises(RuntimeError, match="share a row count"):
        shell._coalesce_sparse_gradients()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_accumulation_leaves_the_concatenation_the_assembly_expects():
    """The premise of the whole scheme, on the device it runs on.

    Two gathers hand autograd one coalesced COO block each; the accumulator
    adds them, and on CUDA that add concatenates rather than sorting. That is
    what leaves a row carrying at most one value per block, and it is why the
    trainer can rebuild the coalesced tensor from the gathered rows.
    """
    device = torch.device("cuda")
    field = _field().to(device)
    field.set_sparse_gradients(True)
    image_rows = torch.arange(0, 10, 2, device=device)
    fiber_rows = torch.tensor([4, 8, 9], device=device)

    field.start_sparse_gather_log()
    total = (
        field.positions(image_rows, sorted_unique=True).sum()
        + field.positions(fiber_rows, sorted_unique=True).sum()
    )
    total.backward()

    assert not field.mu.grad.is_coalesced()
    assert int(field.mu.grad._nnz()) == image_rows.numel() + fiber_rows.numel()
    reference = field.mu.grad.coalesce()
    _shell(field)._coalesce_sparse_gradients()
    assert torch.equal(field.mu.grad._indices(), reference._indices())
    assert torch.equal(field.mu.grad._values(), reference._values())


def test_dense_gradients_are_untouched():
    field = _field()
    field.mu.grad = torch.ones_like(field.mu)
    _shell(field)._coalesce_sparse_gradients()
    assert torch.equal(field.mu.grad, torch.ones_like(field.mu))
