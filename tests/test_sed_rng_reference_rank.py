import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.models.splat_init import seed_calibrated


def _field(n_templates, reference, n_splats=40):
    return SplatField(
        n_splats, n_templates, torch.linspace(-1.0, 1.0, 96),
        sed_off_grid="zero", sed_pin_gauge_knots=True,
        sed_rng_reference_templates=reference,
    )


def _state(model):
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def test_reference_rank_is_a_no_op_at_the_reference_rank():
    torch.manual_seed(42)
    original = _state(_field(32, None))
    after_original = torch.get_rng_state()
    torch.manual_seed(42)
    referenced = _state(_field(32, 32))
    assert torch.equal(torch.get_rng_state(), after_original)
    assert original.keys() == referenced.keys()
    for name in original:
        assert torch.equal(original[name], referenced[name]), name


@pytest.mark.parametrize("rank", [16, 8])
def test_smaller_rank_shares_the_reference_stream(rank):
    torch.manual_seed(7)
    reference = _field(32, None)
    after_reference = torch.get_rng_state()
    torch.manual_seed(7)
    small = _field(rank, 32)
    assert torch.equal(torch.get_rng_state(), after_reference)
    assert small.sed_coeffs.shape == (40, rank) and small.sed.raw_templates.shape == (rank, 96)
    assert small.sed_coeffs.is_contiguous() and small.sed.raw_templates.is_contiguous()
    assert torch.equal(small.sed_coeffs, reference.sed_coeffs[:, :rank])
    for name in ("mu", "log_scales", "theta", "raw_flux", "sed_shift"):
        assert torch.equal(getattr(small, name), getattr(reference, name)), name


@pytest.mark.parametrize("rank", [16, 8])
def test_smaller_rank_keeps_spatial_seeding_identical(rank):
    peaks = torch.tensor([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]])
    values = torch.tensor([3.0, 2.0, 1.0])
    seeded = {}
    for k, reference in ((32, None), (rank, 32)):
        torch.manual_seed(123)
        model = _field(k, reference)
        seed_calibrated(model, peaks, values, n_scales=4)
        seeded[k] = model
    for name in ("mu", "log_scales", "theta", "sed_shift"):
        assert torch.equal(getattr(seeded[rank], name), getattr(seeded[32], name)), name


def test_unreferenced_smaller_rank_would_shift_spatial_seeding():
    # Negative control: without the reference rank the jitter stream moves with K.
    peaks = torch.tensor([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]])
    values = torch.tensor([3.0, 2.0, 1.0])
    seeded = {}
    for k in (32, 16):
        torch.manual_seed(123)
        model = _field(k, None)
        seed_calibrated(model, peaks, values, n_scales=4)
        seeded[k] = model
    assert not torch.equal(seeded[16].mu, seeded[32].mu)


@pytest.mark.parametrize("reference", [4, 16.5, True])
def test_reference_rank_must_cover_the_dictionary(reference):
    with pytest.raises(ValueError, match="rng_reference_templates"):
        _field(8, reference)
