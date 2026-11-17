"""The point-SED diagnostic sums the operator's support, not the whole census.

`_evaluate_spectra_for_omega` is the only diagnostic that evaluated the field
densely against every splat. Restricting it to the truncated support has to
reproduce the dense spectra to float32 rounding, and has to stand aside for
models that are not splat fields.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.training.datum_trainer import DatumTrainer

LOGLAM_MIN = math.log(3560.0)
LOGLAM_MAX = math.log(10400.0)


def _field(*, n_splats: int = 400, seed: int = 3) -> SplatField:
    torch.manual_seed(seed)
    field = SplatField(
        n_splats=n_splats,
        n_templates=4,
        loglam_grid=torch.linspace(-1.0, 1.0, 32),
        init_scale=0.01,
    )
    with torch.no_grad():
        field.mu.uniform_(-0.9, 0.9)
        field.log_scales.uniform_(math.log(0.004), math.log(0.012))
        field.theta.uniform_(0.0, math.pi)
        field.raw_flux.normal_(-2.0, 0.5)
        field.sed_coeffs.normal_(0.0, 1.0)
        field.sed_shift.normal_(0.0, 0.01)
    return field


def _shell(model, *, pair_n_sigma: float = 8.0) -> SimpleNamespace:
    wcs = SimpleNamespace(
        omega_xyz_to_norm_torch=lambda omega: (omega[:, 0], omega[:, 1])
    )
    return SimpleNamespace(
        _base_model=lambda: model,
        runtime_by_exposure={0: SimpleNamespace(exposure=SimpleNamespace(wcs=wcs))},
        loglam_min=LOGLAM_MIN,
        loglam_max=LOGLAM_MAX,
        device=torch.device("cpu"),
        pair_n_sigma=pair_n_sigma,
    )


def _dense_spectra(field: SplatField, x, y, loglam) -> torch.Tensor:
    n_points = int(x.numel())
    n_lambda = int(loglam.numel())
    loglam_norm = 2.0 * (loglam - LOGLAM_MIN) / (LOGLAM_MAX - LOGLAM_MIN) - 1.0
    flux = field(
        x.repeat_interleave(n_lambda),
        y.repeat_interleave(n_lambda),
        loglam_norm.repeat(n_points),
    )
    return flux.view(n_points, n_lambda)


def test_support_spectra_match_the_dense_field() -> None:
    field = _field()
    # Sit two points on splats and two in near-empty sky.
    x = torch.cat([field.mu[:2, 0], torch.tensor([0.95, -0.93])])
    y = torch.cat([field.mu[:2, 1], torch.tensor([-0.94, 0.92])])
    omega = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    loglam = torch.linspace(LOGLAM_MIN, LOGLAM_MAX, 48)

    actual = DatumTrainer._spectra_over_splat_support(_shell(field), 0, omega, loglam)
    with torch.no_grad():
        expected = _dense_spectra(field, x, y, loglam)

    assert actual.shape == expected.shape
    scale = expected.abs().max()
    assert scale > 0
    assert float((actual - expected).abs().max() / scale) < 1e-6


def test_support_is_a_strict_subset_of_the_census() -> None:
    """The whole point is not to touch every splat."""
    field = _field(n_splats=2000)
    x = field.mu[:3, 0].clone()
    y = field.mu[:3, 1].clone()
    omega = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    loglam = torch.linspace(LOGLAM_MIN, LOGLAM_MAX, 16)

    from survey_fusion.pgm.splat_raster import build_packed_point_index

    with torch.no_grad():
        index = build_packed_point_index(field, x, y, 0.0, n_sigma=8.0)
    assert 0 < index.active_splat.numel() < 2000

    actual = DatumTrainer._spectra_over_splat_support(_shell(field), 0, omega, loglam)
    with torch.no_grad():
        expected = _dense_spectra(field, x, y, loglam)
    assert float((actual - expected).abs().max() / expected.abs().max()) < 1e-6


def test_a_point_with_no_splat_in_support_gives_zero() -> None:
    field = _field(n_splats=32)
    with torch.no_grad():
        field.mu.uniform_(0.5, 0.9)
    x = torch.tensor([-0.95])
    y = torch.tensor([-0.95])
    omega = torch.stack([x, y, torch.zeros_like(x)], dim=-1)
    loglam = torch.linspace(LOGLAM_MIN, LOGLAM_MAX, 8)

    actual = DatumTrainer._spectra_over_splat_support(_shell(field), 0, omega, loglam)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_non_splat_model_falls_back_to_the_dense_path() -> None:
    model = torch.nn.Linear(3, 1)
    omega = torch.zeros((2, 3))
    loglam = torch.linspace(LOGLAM_MIN, LOGLAM_MAX, 4)
    assert (
        DatumTrainer._spectra_over_splat_support(_shell(model), 0, omega, loglam)
        is None
    )
