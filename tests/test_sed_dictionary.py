import math

import pytest
import torch

from survey_fusion.models.sed_dictionary import SEDDictionary


def make_dict(K=8, n_grid=64):
    return SEDDictionary(n_templates=K, loglam_grid=torch.linspace(-1.0, 1.0, n_grid))


def test_templates_nonnegative_and_shaped():
    d = make_dict()
    T = d.templates()
    assert T.shape == (8, 64)
    assert (T >= 0).all()


def test_eval_interpolates_on_and_off_grid():
    d = make_dict()
    c = torch.randn(5, 8)
    on = d.eval(c, d.loglam_grid[:3])
    assert on.shape == (5, 3)
    off = d.eval(c, torch.tensor([-0.95, 0.13]))
    assert off.shape == (5, 2)
    assert (on >= 0).all() and (off >= 0).all()


def test_eval_matches_grid_values_exactly_on_nodes():
    d = make_dict(K=3, n_grid=16)
    c = torch.randn(2, 3)
    T = d.templates()
    expected = torch.nn.functional.softplus(c) @ T
    got = d.eval(c, d.loglam_grid)
    assert torch.allclose(got, expected, atol=1e-6)


def test_interpolation_is_linear_in_wavelength() -> None:
    d = SEDDictionary(n_templates=1, loglam_grid=torch.tensor([-1.0, 1.0]))
    d.set_loglam_scale(math.log(2.0))
    with torch.no_grad():
        values = torch.tensor([[1.0, 3.0]])
        d.raw_templates.copy_(torch.log(torch.expm1(values)))
    query = torch.tensor([math.log(1.25) / math.log(2.0)])
    got = d.templates_at(query)
    assert torch.allclose(got, torch.tensor([[2.0]]), atol=1.0e-6)


def test_clamp_mode_reads_the_edge_value_off_grid() -> None:
    """The default is the pre-fix behaviour: an off-grid query returns the
    terminal knot value and routes gradient to that knot."""
    d = make_dict(K=3, n_grid=16)
    grid = d.loglam_grid
    off = torch.tensor([grid[0] - 0.7, grid[-1] + 0.7])

    got = d.templates_at(off)

    T = d.templates()
    assert torch.allclose(got[:, 0], T[:, 0], atol=0.0, rtol=0.0)
    assert torch.allclose(got[:, 1], T[:, -1], atol=0.0, rtol=0.0)

    got.sum().backward()
    grad = d.raw_templates.grad
    assert grad is not None
    assert (grad[:, 0] != 0).all()
    assert (grad[:, -1] != 0).all()


def test_zero_mode_returns_zero_off_grid_with_no_gradient_to_any_knot() -> None:
    d = SEDDictionary(
        n_templates=3, loglam_grid=torch.linspace(-1.0, 1.0, 16), off_grid="zero"
    )
    grid = d.loglam_grid
    off = torch.tensor([grid[0] - 1e-4, grid[0] - 0.7, grid[-1] + 1e-4, grid[-1] + 2.0])

    got = d.templates_at(off)

    assert (got == 0.0).all()

    got.sum().backward()
    assert d.raw_templates.grad is not None
    assert (d.raw_templates.grad == 0.0).all()


def test_zero_mode_is_unchanged_on_grid_and_at_the_grid_edges() -> None:
    d = SEDDictionary(
        n_templates=4, loglam_grid=torch.linspace(-1.0, 1.0, 24), off_grid="zero"
    )
    reference = make_dict(K=4, n_grid=24)
    with torch.no_grad():
        reference.raw_templates.copy_(d.raw_templates)
    query = torch.cat([d.loglam_grid, torch.linspace(-0.87, 0.61, 11)])

    assert torch.equal(d.templates_at(query), reference.templates_at(query))


def test_zero_mode_drops_the_off_grid_part_of_a_shifted_evaluation() -> None:
    d = SEDDictionary(
        n_templates=5, loglam_grid=torch.linspace(-1.0, 1.0, 32), off_grid="zero"
    )
    coeffs = torch.randn(3, 5)
    query = torch.linspace(-0.6, 0.9, 9)
    shift = torch.tensor([0.0, -0.4, 0.4])

    got = d.eval(coeffs, query, shift)

    # splat 1 reads the grid at query + 0.4, so its reddest queries fall off
    assert (got[1, query + 0.4 > 1.0] == 0.0).all()
    assert (got[1, query + 0.4 <= 1.0] > 0.0).all()
    assert (got[0] > 0.0).all()


def test_pinned_gauge_knots_track_their_interior_neighbours() -> None:
    d = SEDDictionary(
        n_templates=6,
        loglam_grid=torch.linspace(-1.0, 1.0, 20),
        pin_gauge_knots=True,
    )

    T = d.templates()

    assert torch.equal(T[:, 0], T[:, 1])
    assert torch.equal(T[:, -1], T[:, -2])
    free = torch.nn.functional.softplus(d.raw_templates)
    assert torch.equal(T[:, 1:-1], free[:, 1:-1])
    # the pin must not move the interior, only the two terminal columns
    assert not torch.equal(T[:, 0], free[:, 0])


def test_pinned_gauge_knots_receive_no_gradient() -> None:
    d = SEDDictionary(
        n_templates=6,
        loglam_grid=torch.linspace(-1.0, 1.0, 20),
        pin_gauge_knots=True,
    )
    coeffs = torch.randn(4, 6)

    d.eval(coeffs, d.loglam_grid).pow(2).sum().backward()

    grad = d.raw_templates.grad
    assert grad is not None
    assert (grad[:, 0] == 0.0).all()
    assert (grad[:, -1] == 0.0).all()
    assert (grad[:, 1] != 0.0).all()
    assert (grad[:, -2] != 0.0).all()


def test_flags_leave_the_state_dict_shape_untouched() -> None:
    plain = make_dict(K=6, n_grid=20)
    fixed = SEDDictionary(
        n_templates=6,
        loglam_grid=torch.linspace(-1.0, 1.0, 20),
        off_grid="zero",
        pin_gauge_knots=True,
    )

    assert list(plain.state_dict()) == list(fixed.state_dict())
    for key, tensor in plain.state_dict().items():
        assert tensor.shape == fixed.state_dict()[key].shape
    fixed.load_state_dict(plain.state_dict())


def test_invalid_flag_values_raise() -> None:
    with pytest.raises(ValueError, match="off_grid must be one of"):
        SEDDictionary(
            n_templates=2, loglam_grid=torch.linspace(-1.0, 1.0, 8), off_grid="edge"
        )
    with pytest.raises(ValueError, match="pin_gauge_knots needs >= 3"):
        SEDDictionary(
            n_templates=2,
            loglam_grid=torch.tensor([-1.0, 1.0]),
            pin_gauge_knots=True,
        )


def test_loglam_scale_round_trips_in_standalone_state_dict():
    source = make_dict()
    source.set_loglam_scale(2.5)
    state_dict = source.state_dict()
    assert "loglam_scale" in state_dict

    target = make_dict()
    target.load_state_dict(state_dict)
    assert target.loglam_scale.item() == pytest.approx(2.5)


def test_legacy_state_dict_preserves_initialized_loglam_scale_and_warns():
    source = make_dict()
    state_dict = source.state_dict()
    state_dict.pop("loglam_scale")

    target = make_dict()
    target.set_loglam_scale(2.5)
    with pytest.warns(
        UserWarning,
        match="old SEDDictionary checkpoint without 'loglam_scale'",
    ) as caught:
        result = target.load_state_dict(state_dict)

    assert len(caught) == 1
    assert result.missing_keys == []
    assert target.loglam_scale.item() == pytest.approx(2.5)


def test_shift_translates_template_evaluation():
    d = make_dict(K=4, n_grid=128)
    c = torch.randn(3, 4)
    ll = torch.linspace(-0.5, 0.5, 7)
    delta = 0.25
    shifted = d.eval(c, ll, shift=torch.full((3,), delta))
    reference = d.eval(c, ll - delta)
    assert torch.allclose(shifted, reference, atol=1e-6)


def test_templates_are_distinct_at_init():
    """Identical templates receive identical gradients and never separate,
    collapsing the dictionary to rank 1."""
    d = make_dict(K=16, n_grid=64)
    T = d.templates()
    spread = T.std(dim=0).mean() / T.mean()
    assert spread > 0.05, f"templates too similar at init: relative spread {spread:.4f}"
    # and they must span a range of spectral slopes, not just be noisy copies
    blue_to_red = (T[:, -1] / T[:, 0].clamp_min(1e-9))
    assert blue_to_red.max() / blue_to_red.min() > 10


def test_distinct_templates_receive_distinct_gradients():
    d = make_dict(K=8, n_grid=32)
    coeffs = torch.zeros(4, 8, requires_grad=True)
    ll = torch.linspace(-0.9, 0.9, 16)
    d.eval(coeffs, ll).pow(2).sum().backward()
    g = d.raw_templates.grad
    assert g is not None
    per_template = g.abs().mean(dim=1)
    assert per_template.std() / per_template.mean() > 0.01, "gradients are symmetric"


def test_shifted_eval_bounds_its_own_template_intermediate():
    """eval(shift=...) allocates (K, n_splats, n_query) but returns something K
    times smaller, so no caller can budget for it from the result shape. Every
    caller sized chunks as if this cost (n_splats, n_query), and at 32 templates
    that turned a 200M-element cap into a 24 GiB allocation."""
    d = make_dict(K=32, n_grid=64)
    J, n = 500, 40
    coeffs = torch.randn(J, 32)
    ll = torch.linspace(-0.5, 0.5, n)
    shift = 0.05 * torch.randn(J)
    budget = 200_000

    peak = 0
    real = d.templates_at

    def spy(loglam):
        nonlocal peak
        out = real(loglam)
        peak = max(peak, out.numel())
        return out

    d.templates_at = spy
    got = d.eval(coeffs, ll, shift, max_elements=budget)
    assert peak <= budget, f"allocated {peak} elements against a {budget} budget"

    d.templates_at = real
    whole = d.eval(coeffs, ll, shift, max_elements=10**9)
    assert got.shape == (J, n)
    assert torch.allclose(whole, got, rtol=1e-5, atol=1e-7)


def test_chunked_shifted_eval_checkpoints_and_matches_gradients():
    """Chunking bounds the transient allocation but not the graph: without
    checkpointing every chunk's (K, chunk, n_query) activations are retained and
    the total is unchanged, which is how a 200k-splat step died holding 94 GiB
    while asking for 192 MiB."""
    J, n = 400, 24
    ll = torch.linspace(-0.4, 0.4, n)

    grads = []
    for budget in (10**9, 20_000):
        # reseed per iteration: the dictionary and the inputs are all random,
        # so seeding once outside compares two different problems
        torch.manual_seed(0)
        d = make_dict(K=16, n_grid=48)
        coeffs = torch.randn(J, 16, requires_grad=True)
        shift = (0.05 * torch.randn(J)).requires_grad_(True)
        d.eval(coeffs, ll, shift, max_elements=budget).pow(2).sum().backward()
        grads.append((coeffs.grad.clone(), d.raw_templates.grad.clone()))

    for a, b in zip(*grads):
        scale = a.abs().max().clamp_min(1e-12)
        assert (a - b).abs().max() / scale < 1e-4


def test_pad_zero_is_bitwise_inert() -> None:
    grid = torch.linspace(-1.0, 1.0, 32)
    torch.manual_seed(11)
    plain = SEDDictionary(n_templates=5, loglam_grid=grid)
    torch.manual_seed(11)
    padded = SEDDictionary(
        n_templates=5, loglam_grid=grid, pad_knots_blue=0, pad_knots_red=0
    )
    assert torch.equal(plain.loglam_grid, padded.loglam_grid)
    assert torch.equal(plain.raw_templates, padded.raw_templates)


def test_padded_grid_keeps_the_data_span_and_spacing() -> None:
    grid = torch.linspace(-1.0, 1.0, 16)
    d = SEDDictionary(
        n_templates=3, loglam_grid=grid, pad_knots_blue=3, pad_knots_red=5
    )
    assert d.loglam_grid.numel() == 24
    assert torch.equal(d.loglam_grid[3:19], grid)
    steps = d.loglam_grid[1:] - d.loglam_grid[:-1]
    spacing = (grid[-1] - grid[0]) / (grid.numel() - 1)
    assert (steps - spacing).abs().max() < 1e-4 * spacing
    assert d.raw_templates.shape == (3, 24)


def test_padded_init_is_anchored_to_the_data_span() -> None:
    grid = torch.linspace(-1.0, 1.0, 16)
    K, blue, red = 4, 3, 5
    torch.manual_seed(23)
    d = SEDDictionary(
        n_templates=K, loglam_grid=grid, pad_knots_blue=blue, pad_knots_red=red
    )
    # Replay the init with the slope coordinate anchored to the data span:
    # padding must extrapolate the same exponentials, not rescale them.
    lam = d.loglam_grid
    lam_n = 2 * (lam - grid[0]) / (grid[-1] - grid[0]) - 1.0
    slopes = torch.linspace(-2.0, 2.0, K)
    shape = torch.exp(slopes[:, None] * lam_n[None, :])
    shape = shape / shape.mean(dim=1, keepdim=True)
    torch.manual_seed(23)
    expected = torch.log(torch.expm1(shape.clamp_min(1e-3))) + 0.01 * torch.randn(
        K, lam.numel()
    )
    assert torch.equal(d.raw_templates.detach(), expected)


def test_padding_moves_the_zero_boundary_to_the_pad_edge() -> None:
    grid = torch.linspace(-1.0, 1.0, 16)
    d = SEDDictionary(
        n_templates=3, loglam_grid=grid, off_grid="zero",
        pad_knots_blue=4, pad_knots_red=4,
    )
    spacing = float(grid[1] - grid[0])
    inside_pad = torch.tensor([-1.0 - 2 * spacing, 1.0 + 2 * spacing])
    beyond_pad = torch.tensor([-1.0 - 5 * spacing, 1.0 + 5 * spacing])

    got_pad = d.templates_at(inside_pad)
    assert (got_pad > 0).all()
    got_pad.sum().backward()
    g = d.raw_templates.grad
    assert g is not None and g.abs().sum() > 0
    # gradient lands only on padding knots (interior data knots untouched)
    assert g[:, 4:20].abs().sum() == 0

    d.raw_templates.grad = None
    got_beyond = d.templates_at(beyond_pad)
    assert (got_beyond == 0).all()
    got_beyond.sum().backward()
    assert d.raw_templates.grad.abs().sum() == 0


def test_padded_pin_ties_the_new_terminal_knots() -> None:
    grid = torch.linspace(-1.0, 1.0, 16)
    d = SEDDictionary(
        n_templates=3, loglam_grid=grid, pin_gauge_knots=True,
        pad_knots_blue=2, pad_knots_red=2,
    )
    T = d.templates()
    assert torch.equal(T[:, 0], T[:, 1])
    assert torch.equal(T[:, -1], T[:, -2])


def test_invalid_pad_values_raise() -> None:
    grid = torch.linspace(-1.0, 1.0, 16)
    with pytest.raises(ValueError, match="pad_knots_blue"):
        SEDDictionary(n_templates=2, loglam_grid=grid, pad_knots_blue=-1)
    with pytest.raises(ValueError, match="pad_knots_red"):
        SEDDictionary(n_templates=2, loglam_grid=grid, pad_knots_red=True)
    ragged = torch.tensor([-1.0, -0.2, 0.3, 1.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        SEDDictionary(n_templates=2, loglam_grid=ragged, pad_knots_red=2)
