from __future__ import annotations

import copy
import math

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.splat_raster import (
    PackedPointIndex,
    build_packed_point_index,
    packed_multiband_point_prediction,
    tiled_point_prediction,
)
from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes
from survey_fusion.pgm import splat_spectrum as splat_spectrum_module


def _field(*, n_splats: int = 64, seed: int = 19) -> SplatField:
    torch.manual_seed(seed)
    field = SplatField(
        n_splats=n_splats,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 24),
        init_scale=0.015,
    )
    with torch.no_grad():
        field.mu.uniform_(-0.8, 0.8)
        field.log_scales.uniform_(math.log(0.008), math.log(0.025))
        field.theta.uniform_(-0.4, 0.4)
        field.raw_flux.uniform_(-1.0, 1.0)
        field.sed_shift.uniform_(-0.1, 0.1)
    return field


def _bandpasses() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "g": (
            torch.tensor([3900.0, 4300.0, 5100.0, 5600.0]),
            torch.tensor([0.0, 0.8, 1.0, 0.0]),
        ),
        "r": (
            torch.tensor([5400.0, 5800.0, 6800.0, 7200.0]),
            torch.tensor([0.0, 1.0, 0.7, 0.0]),
        ),
    }


def _five_bandpasses() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    limits = {
        "u": (3500.0, 4400.0),
        "g": (4000.0, 5500.0),
        "r": (5200.0, 6900.0),
        "i": (6600.0, 8200.0),
        "z": (7900.0, 9100.0),
    }
    return {
        key: (
            torch.linspace(lower, upper, 7),
            torch.tensor([0.0, 0.4, 0.9, 1.0, 0.7, 0.2, 0.0]),
        )
        for key, (lower, upper) in limits.items()
    }


def _configure_wavelength_grid(field: SplatField) -> tuple[float, float]:
    loglam_min = math.log(3400.0)
    loglam_max = math.log(9200.0)
    field.sed.set_loglam_scale(0.5 * (loglam_max - loglam_min))
    return loglam_min, loglam_max


def _membership_rows(
    n_active: int,
    n_bands: int,
    pair_active_splat: torch.Tensor,
    pair_point: torch.Tensor,
    point_band: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    member = torch.zeros(n_active, n_bands, dtype=torch.bool)
    member[pair_active_splat, point_band[pair_point]] = True
    rows = [
        member[:, band_index].nonzero(as_tuple=True)[0]
        for band_index in range(n_bands)
    ]
    return member, rows


def _assert_parameter_grad_equal(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    if expected.layout == torch.sparse_coo:
        assert actual.layout == torch.sparse_coo
        expected = expected.coalesce()
        actual = actual.coalesce()
        assert torch.equal(expected.indices(), actual.indices())
        assert torch.equal(expected.values(), actual.values())
    else:
        assert torch.equal(expected, actual)


def test_band_amplitude_subset_weights_match_full_stack_bitwise(monkeypatch) -> None:
    field = _field(n_splats=11, seed=20260805)
    loglam_min, loglam_max = _configure_wavelength_grid(field)
    bandpasses = _five_bandpasses()
    band_rows = [
        torch.tensor([0, 3, 4, 8, 10]),
        torch.empty(0, dtype=torch.long),
        torch.tensor([1, 2, 5, 9]),
        torch.tensor([0, 6, 7, 10]),
        torch.tensor([2, 3, 8]),
    ]
    member = torch.zeros(11, len(bandpasses), dtype=torch.bool)
    for band_index, rows in enumerate(band_rows):
        member[rows, band_index] = True

    original_einsum = torch.einsum
    captured_weights: list[torch.Tensor] = []

    def capture_einsum(equation, values, weights):
        assert equation == "jg,jbg->jb"
        captured_weights.append(weights.detach().clone())
        return original_einsum(equation, values, weights)

    monkeypatch.setattr(splat_spectrum_module.torch, "einsum", capture_einsum)
    with torch.no_grad():
        full = splat_band_amplitudes(
            field,
            bandpasses,
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            splat_chunk=4,
        )
        full_weights = torch.cat(captured_weights)
        captured_weights.clear()
        subset = splat_band_amplitudes(
            field,
            bandpasses,
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            splat_chunk=4,
            band_row_indices=band_rows,
        )
        subset_weights = torch.cat(captured_weights)

    full_matrix = torch.stack(list(full.values()), dim=1)
    subset_matrix = torch.stack(list(subset.values()), dim=1)
    assert torch.equal(subset_weights[member], full_weights[member])
    assert torch.count_nonzero(subset_weights[~member]) == 0
    assert torch.equal(subset_matrix[member], full_matrix[member])
    assert torch.count_nonzero(subset_matrix[~member]) == 0


@pytest.mark.parametrize("disable_checkpoint", [False, True])
@pytest.mark.parametrize("sparse_gradients", [False, True])
def test_band_amplitude_subsets_preserve_raster_and_gradients_bitwise(
    monkeypatch,
    disable_checkpoint: bool,
    sparse_gradients: bool,
) -> None:
    if disable_checkpoint:
        monkeypatch.setenv("SPLAT_NO_CHECKPOINT", "1")
    else:
        monkeypatch.delenv("SPLAT_NO_CHECKPOINT", raising=False)

    full_field = _field(n_splats=13, seed=20260805)
    subset_field = copy.deepcopy(full_field)
    full_field.set_sparse_gradients(sparse_gradients)
    subset_field.set_sparse_gradients(sparse_gradients)
    loglam_min, loglam_max = _configure_wavelength_grid(full_field)
    _configure_wavelength_grid(subset_field)
    bandpasses = _five_bandpasses()
    band_keys = list(bandpasses)

    active_splat = torch.tensor([0, 2, 4, 6, 8, 10, 12])
    point_band = torch.tensor([0, 1, 2, 3, 4, 0, 2, 4, 1, 3])
    pair_point = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 2])
    pair_active_splat = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6])
    packed_index = PackedPointIndex(
        pair_point=pair_point,
        active_splat=active_splat,
        pair_active_splat=pair_active_splat,
    )
    member, band_rows = _membership_rows(
        active_splat.numel(),
        len(bandpasses),
        pair_active_splat,
        pair_point,
        point_band,
    )
    assert all(0 in rows for rows in band_rows)

    full = splat_band_amplitudes(
        full_field,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        splat_chunk=3,
        indices=active_splat,
        indices_sorted_unique=True,
    )
    subset = splat_band_amplitudes(
        subset_field,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        splat_chunk=3,
        indices=active_splat,
        indices_sorted_unique=True,
        band_row_indices=band_rows,
    )
    full_matrix = torch.stack([full[key] for key in band_keys], dim=1)
    subset_matrix = torch.stack([subset[key] for key in band_keys], dim=1)
    assert torch.equal(subset_matrix[member], full_matrix[member])
    assert torch.count_nonzero(subset_matrix[~member]) == 0

    generator = torch.Generator().manual_seed(73)
    x = torch.rand(point_band.numel(), generator=generator) * 1.6 - 0.8
    y = torch.rand(point_band.numel(), generator=generator) * 1.6 - 0.8
    psf_sigma = torch.rand(point_band.numel(), generator=generator) * 0.01 + 0.01
    full_prediction = packed_multiband_point_prediction(
        full_field,
        x,
        y,
        psf_sigma,
        point_band,
        full_matrix,
        packed_index,
    )
    subset_prediction = packed_multiband_point_prediction(
        subset_field,
        x,
        y,
        psf_sigma,
        point_band,
        subset_matrix,
        packed_index,
    )
    assert torch.equal(subset_prediction, full_prediction)

    full_prediction.square().sum().backward()
    subset_prediction.square().sum().backward()
    for name in ("sed_coeffs", "sed_shift", "raw_flux"):
        full_grad = getattr(full_field, name).grad
        subset_grad = getattr(subset_field, name).grad
        assert full_grad is not None and subset_grad is not None
        _assert_parameter_grad_equal(full_grad, subset_grad)
    assert full_field.sed.raw_templates.grad is not None
    assert subset_field.sed.raw_templates.grad is not None
    assert torch.equal(
        full_field.sed.raw_templates.grad,
        subset_field.sed.raw_templates.grad,
    )


def test_band_amplitude_subsets_handle_empty_pairs_bitwise() -> None:
    full_field = _field(n_splats=5, seed=20260805)
    subset_field = copy.deepcopy(full_field)
    loglam_min, loglam_max = _configure_wavelength_grid(full_field)
    _configure_wavelength_grid(subset_field)
    bandpasses = _five_bandpasses()
    empty = torch.empty(0, dtype=torch.long)
    band_rows = [empty for _ in bandpasses]

    full = splat_band_amplitudes(
        full_field,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        indices=empty,
    )
    subset = splat_band_amplitudes(
        subset_field,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        indices=empty,
        band_row_indices=band_rows,
    )
    full_matrix = torch.stack(list(full.values()), dim=1)
    subset_matrix = torch.stack(list(subset.values()), dim=1)
    assert torch.equal(subset_matrix, full_matrix)

    index = PackedPointIndex(empty, empty, empty)
    x = torch.tensor([0.1, -0.2])
    y = torch.tensor([0.3, 0.4])
    point_band = torch.tensor([0, 4])
    full_prediction = packed_multiband_point_prediction(
        full_field, x, y, 0.01, point_band, full_matrix, index
    )
    subset_prediction = packed_multiband_point_prediction(
        subset_field, x, y, 0.01, point_band, subset_matrix, index
    )
    assert torch.equal(subset_prediction, full_prediction)
    assert torch.count_nonzero(subset_prediction) == 0


def test_band_amplitude_subset_indices_are_validated() -> None:
    field = _field(n_splats=3)
    loglam_min, loglam_max = _configure_wavelength_grid(field)
    bandpasses = _bandpasses()
    with pytest.raises(ValueError, match="one tensor per bandpass"):
        splat_band_amplitudes(
            field,
            {},
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            band_row_indices=[torch.tensor([0])],
        )
    with pytest.raises(ValueError, match="one tensor per bandpass"):
        splat_band_amplitudes(
            field,
            bandpasses,
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            band_row_indices=[torch.tensor([0])],
        )
    with pytest.raises(ValueError, match="long-dtype tensors"):
        splat_band_amplitudes(
            field,
            bandpasses,
            loglam_min=loglam_min,
            loglam_max=loglam_max,
            band_row_indices=[torch.tensor([0], dtype=torch.int32)] * 2,
        )


def test_sparse_parameter_gathers_match_dense_gradients() -> None:
    active = torch.tensor([2, 7, 7, 18, 41], dtype=torch.long)
    dense = _field()
    sparse = copy.deepcopy(dense)
    sparse.set_sparse_gradients(True)

    def objective(field: SplatField) -> torch.Tensor:
        covariance = field.covariances(active)
        position = field.positions(active)
        flux = field.fluxes(active)
        coefficients = field.sed_coefficients(active)
        shift = field.spectral_shifts(active)
        return (
            covariance.square().sum()
            + position.square().sum()
            + flux.square().sum()
            + coefficients.square().sum()
            + shift.square().sum()
        )

    objective(dense).backward()
    objective(sparse).backward()

    expected_active = set(active.tolist())
    for dense_parameter, sparse_parameter in zip(
        dense.per_splat_parameters(), sparse.per_splat_parameters()
    ):
        assert dense_parameter.grad is not None
        assert sparse_parameter.grad is not None
        assert sparse_parameter.grad.layout == torch.sparse_coo
        torch.testing.assert_close(
            sparse_parameter.grad.to_dense(),
            dense_parameter.grad,
            rtol=1e-6,
            atol=1e-7,
        )
        assert set(sparse_parameter.grad.coalesce().indices()[0].tolist()) == expected_active


def test_sorted_unique_covariance_gather_matches_coalesced_sparse_gradient() -> None:
    indices = torch.tensor([1, 6, 14, 37, 52], dtype=torch.long)
    coalesced = _field()
    sorted_unique = copy.deepcopy(coalesced)
    coalesced.set_sparse_gradients(True)
    sorted_unique.set_sparse_gradients(True)
    weights = torch.arange(1, 21, dtype=torch.float32).reshape(5, 2, 2)

    (coalesced.covariances(indices) * weights).sum().backward()
    (
        sorted_unique.covariances(indices, sorted_unique=True) * weights
    ).sum().backward()

    for coalesced_parameter, sorted_unique_parameter in (
        (coalesced.log_scales, sorted_unique.log_scales),
        (coalesced.theta, sorted_unique.theta),
    ):
        assert coalesced_parameter.grad is not None
        assert sorted_unique_parameter.grad is not None
        expected = coalesced_parameter.grad.coalesce()
        actual = sorted_unique_parameter.grad.coalesce()
        assert torch.equal(actual.indices(), expected.indices())
        assert torch.equal(actual.values(), expected.values())
        assert torch.equal(actual.to_dense(), expected.to_dense())


def test_sparse_adam_changes_only_active_rows_and_moments() -> None:
    field = _field(n_splats=32)
    field.set_sparse_gradients(True)
    active = torch.tensor([1, 5, 11, 23], dtype=torch.long)
    optimizer = torch.optim.SparseAdam(field.sparse_param_groups_gsplat())

    loss = (
        field.positions(active).square().sum()
        + field.covariances(active)[:, 0, 0].sum()
        + field.fluxes(active).square().sum()
        + field.sed_coefficients(active).square().sum()
        + field.spectral_shifts(active).square().sum()
    )
    loss.backward()
    before = [parameter.detach().clone() for parameter in field.per_splat_parameters()]
    optimizer.step()

    inactive = torch.ones(32, dtype=torch.bool)
    inactive[active] = False
    for parameter, original in zip(field.per_splat_parameters(), before):
        torch.testing.assert_close(parameter[inactive], original[inactive], rtol=0.0, atol=0.0)
        assert torch.any(parameter[active] != original[active])
        state = optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][inactive]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][inactive]) == 0


def test_packed_multiband_prediction_matches_existing_tiled_path_and_gradients() -> None:
    torch.manual_seed(31)
    dense = _field(n_splats=96)
    sparse = copy.deepcopy(dense)
    sparse.set_sparse_gradients(True)
    loglam_min, loglam_max = _configure_wavelength_grid(dense)
    _configure_wavelength_grid(sparse)
    bandpasses = _bandpasses()

    n_points = 300
    x = torch.empty(n_points).uniform_(-0.9, 0.9)
    y = torch.empty(n_points).uniform_(-0.9, 0.9)
    psf_sigma = torch.empty(n_points).uniform_(0.006, 0.02)
    point_band = torch.randint(0, 2, (n_points,))
    band_keys = list(bandpasses)

    dense_amplitudes = splat_band_amplitudes(
        dense,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
    )
    dense_prediction = torch.empty(n_points)
    for band_id, band_key in enumerate(band_keys):
        selected = torch.nonzero(point_band == band_id, as_tuple=True)[0]
        dense_prediction[selected] = tiled_point_prediction(
            dense,
            x[selected],
            y[selected],
            psf_sigma[selected],
            dense_amplitudes[band_key],
            n_sigma=6.0,
        )

    packed_index = build_packed_point_index(
        sparse, x, y, psf_sigma, n_sigma=6.0
    )
    sparse_amplitudes = splat_band_amplitudes(
        sparse,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        indices=packed_index.active_splat,
    )
    amplitude_matrix = torch.stack(
        [sparse_amplitudes[key] for key in band_keys], dim=1
    )
    sparse_prediction = packed_multiband_point_prediction(
        sparse,
        x,
        y,
        psf_sigma,
        point_band,
        amplitude_matrix,
        packed_index,
    )

    torch.testing.assert_close(
        sparse_prediction,
        dense_prediction,
        rtol=2e-5,
        atol=2e-7,
    )

    dense_packed_amplitudes = splat_band_amplitudes(
        dense,
        bandpasses,
        loglam_min=loglam_min,
        loglam_max=loglam_max,
        indices=packed_index.active_splat,
    )
    dense_packed_prediction = packed_multiband_point_prediction(
        dense,
        x,
        y,
        psf_sigma,
        point_band,
        torch.stack(
            [dense_packed_amplitudes[key] for key in band_keys], dim=1
        ),
        packed_index,
    )
    dense_packed_prediction.square().mean().backward()
    sparse_prediction.square().mean().backward()
    for dense_parameter, sparse_parameter in zip(
        dense.per_splat_parameters(), sparse.per_splat_parameters()
    ):
        assert sparse_parameter.grad is not None
        assert sparse_parameter.grad.layout == torch.sparse_coo
        torch.testing.assert_close(
            sparse_parameter.grad.to_dense(),
            dense_parameter.grad,
            rtol=3e-4,
            atol=2e-7,
        )
    torch.testing.assert_close(
        sparse.sed.raw_templates.grad,
        dense.sed.raw_templates.grad,
        rtol=3e-4,
        atol=2e-7,
    )


def test_packed_index_excludes_spatially_inactive_splats() -> None:
    field = _field(n_splats=1000)
    with torch.no_grad():
        field.mu[:20].uniform_(-0.02, 0.02)
        field.mu[20:, 0].uniform_(0.5, 0.9)
        field.mu[20:, 1].uniform_(0.5, 0.9)
        field.log_scales.fill_(math.log(0.002))

    x = torch.empty(200).uniform_(-0.01, 0.01)
    y = torch.empty(200).uniform_(-0.01, 0.01)
    index = build_packed_point_index(field, x, y, 0.002, n_sigma=4.0)

    assert index.pair_point.numel() == index.pair_active_splat.numel()
    assert index.active_splat.numel() <= 20
    assert set(index.active_splat.tolist()) <= set(range(20))
