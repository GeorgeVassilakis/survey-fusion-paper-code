"""Numerical stability of the fiber-aperture spatial weight for elongated splats.

The captured failure: fiber S3154-54821-0498 (exposure 993) of an
epoch-13 checkpoint predicted 3378 times its observed flux because the
quadratic form of a 1.34 x 194 arcsec splat was contracted through its
precision matrix in a TF32 GEMM. The fixture under tests/fixtures holds the
real fp32 inputs of that fiber's spatial stage together with the TF32 output
the old contraction produced and an fp64 reference on the same geometry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm import splat_spectrum as splat_spectrum_module
from survey_fusion.pgm.splat_spectrum import (
    SplatSpectrumCache,
    _largest_eigenvalue_symmetric_2x2,
    candidate_aperture_weights,
    convolved_gaussian_coefficients,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "spectral_precision"
COMPACT_FIXTURE = FIXTURE_DIR / "needle_quadratic_form.npz"
ARCSEC_PER_NORM = 2520.144
NEEDLE_MINOR_ARCSEC = 1.3414
PSF_COV = torch.tensor(
    [[5.985031492627968e-08, 0.0], [0.0, 5.611328646182301e-08]],
    dtype=torch.float64,
)
AREA_ARCSEC2 = 7.06858347


def _legacy_quadratic_form(delta: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    """The CUDA contraction this fix replaces, verbatim."""
    delta_jq = delta.permute(1, 0, 2)
    return (
        torch.bmm(delta_jq, precision)
        .mul(delta_jq)
        .sum(dim=-1)
        .permute(1, 0)
        .contiguous()
    )


def _reference_weights(
    mu: torch.Tensor,
    log_scales: torch.Tensor,
    theta: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    psf_cov: torch.Tensor,
    area_arcsec2: float,
) -> torch.Tensor:
    """Independent fp64 evaluation: explicit rotation, inverse and einsum."""
    mu, log_scales, theta = mu.double(), log_scales.double(), theta.double()
    points, weights, psf_cov = points.double(), weights.double(), psf_cov.double()
    cos, sin = torch.cos(theta), torch.sin(theta)
    rotation = torch.stack(
        [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
    )
    covariance = (
        rotation @ torch.diag_embed(torch.exp(2.0 * log_scales)) @ rotation.transpose(-1, -2)
        + psf_cov
    )
    precision = torch.linalg.inv(covariance)
    delta = points[:, None, :] - mu[None, :, :]
    mahalanobis = torch.einsum("qji,jik,qjk->qj", delta, precision, delta)
    norm = 1.0 / (2.0 * math.pi * torch.sqrt(torch.linalg.det(covariance)))
    density = norm[None, :] * torch.exp(-0.5 * mahalanobis)
    return torch.sum(density * weights[:, None], dim=0) * area_arcsec2


def _field(mu: torch.Tensor, log_scales: torch.Tensor, theta: torch.Tensor, dtype: torch.dtype) -> SplatField:
    field = SplatField(
        n_splats=int(mu.shape[0]),
        n_templates=2,
        loglam_grid=torch.linspace(-1.0, 1.0, 4),
    ).to(dtype)
    with torch.no_grad():
        field.mu.copy_(mu)
        field.log_scales.copy_(log_scales)
        field.theta.copy_(theta)
    return field


def _cache(
    exposure: object,
    points: torch.Tensor,
    weights: torch.Tensor,
    psf_cov: torch.Tensor,
    area_arcsec2: float,
    apply_spatial_psf: bool = True,
) -> SplatSpectrumCache:
    return SplatSpectrumCache(
        exposure_identity=id(exposure),
        apply_spatial_psf=apply_spatial_psf,
        radial_nodes=8,
        angular_nodes=32,
        dtype=points.dtype,
        device=points.device,
        lam_edges=torch.tensor([[5000.0, 5010.0]], dtype=points.dtype, device=points.device),
        points=points,
        weights=weights,
        psf_cov=psf_cov,
        psf_eigen_max=_largest_eigenvalue_symmetric_2x2(psf_cov),
        aperture_area_arcsec2=float(area_arcsec2),
        scalar_lsf_sigma=0.0,
        lsf_sigma_by_bin=None,
        lsf_mode="zero",
    )


def _spatial_weights(field: SplatField, cache: SplatSpectrumCache, exposure: object, candidate=None):
    return splat_spectrum_module._candidate_spatial_weights(
        field,
        exposure,
        None,
        apply_spatial_psf=cache.apply_spatial_psf,
        radial_nodes=8,
        angular_nodes=32,
        n_sigma=6.0,
        cache=cache,
        candidate=candidate,
    )


def _load_capture(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: torch.from_numpy(np.ascontiguousarray(payload[key])) for key in payload.files}


def _shapes_from_capture(capture: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mu, log_scales, theta) in fp32 that regenerate the captured covariances.

    The capture holds the fp32 total covariance; the splat part is recovered
    in fp64 and re-expressed through the field's own parameterization.
    """
    splat_cov = capture["covariance"].double() - capture["psf_cov"].double()
    eigenvalues, eigenvectors = torch.linalg.eigh(splat_cov)
    theta = torch.atan2(eigenvectors[:, 1, 0], eigenvectors[:, 0, 0])
    log_scales = 0.5 * torch.log(eigenvalues)
    mu = capture["points"][0].double() - capture["delta"][0].double()
    return mu.float(), log_scales.float(), theta.float()


def _full_shape_capture(compact: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Restore the captured 459-column batch shape around the retained columns.

    cuBLAS picks its strided-batched GEMM kernel from the batch shape, and only
    the kernel it picked for the real fiber rounds to TF32: at 17 columns the
    legacy contraction stays positive on the GPU. Each retained column goes
    back to its source position; the others are filled by cycling the retained
    ordinary columns, which changes nothing per column since every batch
    element of the GEMM is independent.
    """
    n_columns = int(compact["source_candidate_count"])
    source = compact["source_column"]
    filler = torch.arange(n_columns) % int(source.numel())
    column_of = filler.clone()
    column_of[source] = torch.arange(int(source.numel()))
    full = {
        key: compact[key] for key in ("points", "weights", "psf_cov", "area_arcsec2", "offending_row")
    }
    full["delta"] = compact["delta"][:, column_of, :].contiguous()
    for key in ("covariance", "precision"):
        full[key] = compact[key][column_of].contiguous()
    for key in ("q_tf32", "q_strict", "q_reference"):
        full[key] = compact[key][:, column_of].contiguous()
    full["offending_candidate_index"] = source[int(compact["offending_candidate_index"])]
    full["retained_columns"] = source
    return full


def _synthetic_needles(axis_ratio: float, n_splats: int = 400, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    minor = torch.full((n_splats,), math.log(NEEDLE_MINOR_ARCSEC / ARCSEC_PER_NORM), dtype=torch.float64)
    log_scales = torch.stack([minor, minor + math.log(axis_ratio)], dim=1)
    swap = torch.rand(n_splats, generator=generator, dtype=torch.float64) < 0.5
    log_scales = torch.where(swap[:, None], log_scales.flip(1), log_scales)
    theta = torch.rand(n_splats, generator=generator, dtype=torch.float64) * math.pi
    # Centres spread from inside the aperture to a few arcminutes away, so the
    # sample covers the peak, the flanks and the long-axis tail.
    mu = torch.randn(n_splats, 2, generator=generator, dtype=torch.float64) * 0.02
    mu[: n_splats // 4] *= 0.02
    radius = torch.sqrt(torch.rand(96, generator=generator, dtype=torch.float64)) * 1.5 / ARCSEC_PER_NORM
    angle = torch.rand(96, generator=generator, dtype=torch.float64) * 2.0 * math.pi
    points = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    weights = torch.full((96,), 1.0 / 96.0, dtype=torch.float64)
    return mu, log_scales, theta, points, weights


def _rotated_psf(sigma_major_arcsec: float, axis_ratio: float, angle: float) -> torch.Tensor:
    """A symmetric positive-definite PSF covariance with a nonzero off-diagonal."""
    major = (sigma_major_arcsec / ARCSEC_PER_NORM) ** 2
    minor = major / axis_ratio**2
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float64)
    return rotation @ torch.diag(torch.tensor([major, minor], dtype=torch.float64)) @ rotation.T


def _relative_to_peak(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double(), expected.detach().double()
    return float((actual - expected).abs().max() / expected.abs().max())


def _fp64_tolerance(axis_ratio: float) -> float:
    """The fp64 reference inverts the total covariance, so its own error grows
    with the condition number; the tolerance follows that, not the fixed path."""
    return 1.0e-12 * max(1.0, (axis_ratio / 10.0) ** 2)


@pytest.fixture(scope="module")
def compact_capture() -> dict[str, torch.Tensor]:
    return _load_capture(COMPACT_FIXTURE)


def test_compact_fixture_provenance_matches_the_capture() -> None:
    provenance = json.loads((FIXTURE_DIR / "needle_quadratic_form.json").read_text())
    assert provenance["fiber"] == "S3154-54821-0498"
    assert provenance["exposure_id"] == 993
    assert provenance["offending_row"] == 967906
    capture = _load_capture(COMPACT_FIXTURE)
    assert capture["delta"].shape == (256, len(provenance["kept_source_columns"]), 2)
    assert int(capture["offending_row"]) == 967906
    assert set(provenance["affected_source_columns"]) <= set(provenance["kept_source_columns"])


def test_full_shape_reconstruction_keeps_retained_columns_in_place(compact_capture) -> None:
    full = _full_shape_capture(compact_capture)
    assert full["delta"].shape == (256, 459, 2)
    assert full["precision"].shape == (459, 2, 2)
    assert full["delta"].is_contiguous() and full["precision"].is_contiguous()
    assert int(full["offending_candidate_index"]) == 335
    retained = full["retained_columns"]
    assert torch.equal(full["delta"][:, retained, :], compact_capture["delta"])
    assert torch.equal(full["q_tf32"][:, retained], compact_capture["q_tf32"])
    assert float(full["q_tf32"][:, 335].min()) < -20.0


def test_capture_records_the_tf32_failure_and_a_sound_strict_result(compact_capture) -> None:
    q_tf32 = compact_capture["q_tf32"].double()
    q_strict = compact_capture["q_strict"].double()
    q_reference = compact_capture["q_reference"]
    offending = int(compact_capture["offending_candidate_index"])

    # The old contraction under TF32 made the quadratic form negative, which
    # exponentiates into a density orders of magnitude above the peak.
    assert q_tf32[:, offending].min() < -20.0
    assert torch.exp(-0.5 * q_tf32).max() > 1.0e5
    assert (q_reference >= 0.0).all()
    # Identical inputs without TF32 rounding were sound: the failure is the
    # contraction, not the captured covariance or geometry.
    assert (q_strict >= 0.0).all()
    assert float(((q_strict - q_reference).abs() / (q_reference.abs() + 1.0e-2)).max()) < 5.0e-4


def test_fixture_quadratic_form_is_nonnegative_and_matches_the_fp64_reference(compact_capture) -> None:
    mu, log_scales, theta = _shapes_from_capture(compact_capture)
    psf_cov = compact_capture["psf_cov"]
    inv_a, shear, coef, norm = convolved_gaussian_coefficients(log_scales, theta, psf_cov)
    delta_x = compact_capture["delta"][..., 0]
    delta_y = compact_capture["delta"][..., 1]
    sheared = delta_y - shear[None, :] * delta_x
    mahalanobis = delta_x * delta_x * inv_a[None, :] + sheared * sheared * coef[None, :]

    assert torch.isfinite(mahalanobis).all()
    assert (mahalanobis >= 0.0).all()
    density_error = (torch.exp(-0.5 * mahalanobis.double()) - torch.exp(-0.5 * compact_capture["q_reference"])).abs()
    assert float(density_error.max()) < 2.0e-5

    # The captured fp32 covariance entries have already lost the minor axis
    # of the needles, so the normalization is checked against the exact
    # determinant of the parameters that regenerate them.
    a, b, d = SplatField.covariance_components_from_rows(log_scales.double(), theta.double())
    psf = psf_cov.double()
    reference_det = (
        torch.exp(2.0 * log_scales.double().sum(dim=1))
        + psf[0, 0] * psf[1, 1]
        + a * psf[1, 1]
        + d * psf[0, 0]
    )
    reference_norm = 1.0 / (2.0 * math.pi * torch.sqrt(reference_det))
    assert float(((norm.double() - reference_norm) / reference_norm).abs().max()) < 2.0e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixture_spatial_weights_match_the_fp64_reference_through_the_spectral_path(
    compact_capture, dtype: torch.dtype
) -> None:
    mu, log_scales, theta = _shapes_from_capture(compact_capture)
    field = _field(mu, log_scales, theta, dtype)
    points = compact_capture["points"].to(dtype)
    weights = compact_capture["weights"].to(dtype)
    psf_cov = compact_capture["psf_cov"].to(dtype)
    area = float(compact_capture["area_arcsec2"])
    exposure = object()
    cache = _cache(exposure, points, weights, psf_cov, area)

    candidate, spatial = _spatial_weights(field, cache, exposure)
    # Every captured candidate passed the production reach test on the GPU;
    # the fixed-geometry reselection must keep all of them.
    assert torch.equal(candidate, torch.arange(mu.shape[0]))
    expected = _reference_weights(mu, log_scales, theta, points, weights, psf_cov, area)
    assert torch.isfinite(spatial).all()
    assert (spatial >= 0.0).all()
    tolerance = 2.0e-6 if dtype == torch.float32 else 1.0e-12
    assert _relative_to_peak(spatial, expected) < tolerance
    offending = int(compact_capture["offending_candidate_index"])
    assert float(spatial.detach()[offending]) == pytest.approx(float(expected[offending]), rel=1.0e-4)


@pytest.mark.parametrize("axis_ratio", [1.0, 10.0, 145.0, 1000.0, 5000.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_synthetic_rotated_needles_match_the_fp64_reference(axis_ratio: float, dtype: torch.dtype) -> None:
    mu, log_scales, theta, points, weights = _synthetic_needles(axis_ratio)
    field = _field(mu, log_scales, theta, dtype)
    exposure = object()
    cache = _cache(exposure, points.to(dtype), weights.to(dtype), PSF_COV.to(dtype), AREA_ARCSEC2)
    candidate = torch.arange(mu.shape[0])

    _, spatial = _spatial_weights(field, cache, exposure, candidate=candidate)
    expected = _reference_weights(
        field.mu.detach(), field.log_scales.detach(), field.theta.detach(),
        cache.points, cache.weights, cache.psf_cov, AREA_ARCSEC2,
    )
    assert torch.isfinite(spatial).all()
    assert (spatial >= 0.0).all()
    tolerance = 5.0e-6 if dtype == torch.float32 else _fp64_tolerance(axis_ratio)
    assert _relative_to_peak(spatial, expected) < tolerance
    # Entries carrying at least a millionth of the peak agree to a relative
    # tolerance of their own, so the tail is checked as well as the peak.
    significant = expected > 1.0e-6 * expected.max()
    relative = ((spatial.detach().double() - expected) / expected)[significant].abs().max()
    assert float(relative) < (5.0e-5 if dtype == torch.float32 else 10.0 * _fp64_tolerance(axis_ratio))


@pytest.mark.parametrize("psf_axis_ratio, angle", [(1.3, 0.4), (2.0, 1.1), (4.0, 2.6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotated_anisotropic_psf_with_off_diagonal_matches_the_reference(
    psf_axis_ratio: float, angle: float, dtype: torch.dtype
) -> None:
    psf_cov = _rotated_psf(0.6, psf_axis_ratio, angle)
    assert float(psf_cov[0, 1]) != 0.0
    mu, log_scales, theta, points, weights = _synthetic_needles(145.0, n_splats=300)
    field = _field(mu, log_scales, theta, dtype)
    exposure = object()
    cache = _cache(exposure, points.to(dtype), weights.to(dtype), psf_cov.to(dtype), AREA_ARCSEC2)

    _, spatial = _spatial_weights(field, cache, exposure, candidate=torch.arange(mu.shape[0]))
    expected = _reference_weights(
        field.mu.detach(), field.log_scales.detach(), field.theta.detach(),
        cache.points, cache.weights, cache.psf_cov, AREA_ARCSEC2,
    )
    assert torch.isfinite(spatial).all() and (spatial >= 0.0).all()
    assert _relative_to_peak(spatial, expected) < (5.0e-6 if dtype == torch.float32 else 1.0e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_psf_mode_uses_the_exact_splat_determinant(dtype: torch.dtype) -> None:
    mu, log_scales, theta, points, weights = _synthetic_needles(1000.0, n_splats=300)
    field = _field(mu, log_scales, theta, dtype)
    zero_psf = torch.zeros(2, 2, dtype=dtype)
    exposure = object()
    cache = _cache(exposure, points.to(dtype), weights.to(dtype), zero_psf, AREA_ARCSEC2, apply_spatial_psf=False)

    _, spatial = _spatial_weights(field, cache, exposure, candidate=torch.arange(mu.shape[0]))
    inv_a, shear, coef, norm = convolved_gaussian_coefficients(field.log_scales.detach(), field.theta.detach(), zero_psf)
    exact_norm = 1.0 / (2.0 * math.pi * torch.exp(field.log_scales.detach().sum(dim=1)))
    assert torch.allclose(norm, exact_norm, rtol=(4.0e-7 if dtype == torch.float32 else 1.0e-14), atol=0.0)
    expected = _reference_weights(
        field.mu.detach(), field.log_scales.detach(), field.theta.detach(),
        cache.points, cache.weights, zero_psf, AREA_ARCSEC2,
    )
    assert torch.isfinite(spatial).all() and (spatial >= 0.0).all()
    assert _relative_to_peak(spatial, expected) < (5.0e-6 if dtype == torch.float32 else _fp64_tolerance(1000.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ill_conditioned_psf_is_bounded_in_float32_and_exact_in_float64(dtype: torch.dtype) -> None:
    """The documented residual: an fp32 PSF with cond 1e6 aligned with a needle.

    The trace term tr(adj(Sigma) Psi) is a sum of products that cancels by
    1 / cond(Psi) once the splat's minor variance drops below the PSF's minor
    variance (here 0.6 arcsec by 0.0006 arcsec against a 194 by 0.0001 arcsec
    splat). Such a PSF is far outside the near-isotropic WCS-derived ones (cond
    about 1.07) and its minor eigenvalue is not representable in its fp32
    entries, so the float32 normalization is allowed to be percent-level off
    (measured 2.2%); it must still be finite, non-negative and bounded, and
    exact once the PSF entries are fp64.
    """
    angle = 0.9
    psf_cov = _rotated_psf(0.6, 1000.0, angle)
    n_splats = 64
    minor = torch.full((n_splats,), math.log(1.0e-4 / ARCSEC_PER_NORM), dtype=torch.float64)
    log_scales = torch.stack([minor + math.log(194.47 / 1.0e-4), minor], dim=1)
    theta = torch.full((n_splats,), angle, dtype=torch.float64)
    mu = torch.randn(n_splats, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 4.0e-4
    _, _, _, points, weights = _synthetic_needles(1.0, n_splats=8)
    field = _field(mu, log_scales, theta, dtype)
    exposure = object()
    cache = _cache(exposure, points.to(dtype), weights.to(dtype), psf_cov.to(dtype), AREA_ARCSEC2)

    _, spatial = _spatial_weights(field, cache, exposure, candidate=torch.arange(n_splats))
    expected = _reference_weights(
        field.mu.detach(), field.log_scales.detach(), field.theta.detach(),
        cache.points, cache.weights, cache.psf_cov, AREA_ARCSEC2,
    )
    assert torch.isfinite(spatial).all() and (spatial >= 0.0).all()
    if dtype == torch.float32:
        assert _relative_to_peak(spatial, expected) < 5.0e-2
        return
    # With the splat aligned to the PSF the determinant is analytic; the fp64
    # reference above inverts a matrix of condition 1e11 and is itself only
    # good to about 1e-6 here, so the normalization is checked directly.
    _, _, _, norm = convolved_gaussian_coefficients(field.log_scales.detach(), field.theta.detach(), cache.psf_cov)
    psf_eigen = torch.linalg.eigvalsh(psf_cov)
    splat_eigen = torch.exp(2.0 * field.log_scales.detach())
    exact_det = (
        splat_eigen[:, 0] * splat_eigen[:, 1]
        + psf_eigen[0] * psf_eigen[1]
        + splat_eigen[:, 0] * psf_eigen[0]
        + splat_eigen[:, 1] * psf_eigen[1]
    )
    exact_norm = 1.0 / (2.0 * math.pi * torch.sqrt(exact_det))
    assert float(((norm - exact_norm) / exact_norm).abs().max()) < 1.0e-9


def test_cpu_autocast_leaves_the_spatial_weights_in_float32(compact_capture) -> None:
    """Autocast only rewrites matmul-class kernels; the spatial stage has none."""
    mu, log_scales, theta = _shapes_from_capture(compact_capture)
    field = _field(mu, log_scales, theta, torch.float32)
    exposure = object()
    cache = _cache(
        exposure, compact_capture["points"], compact_capture["weights"], compact_capture["psf_cov"],
        float(compact_capture["area_arcsec2"]),
    )
    candidate = torch.arange(mu.shape[0])
    _, plain = _spatial_weights(field, cache, exposure, candidate=candidate)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, under_autocast = _spatial_weights(field, cache, exposure, candidate=candidate)
    assert under_autocast.dtype == torch.float32
    assert torch.equal(under_autocast, plain)


def test_whitened_form_bounds_the_density_where_the_covariance_entries_cannot() -> None:
    """Axis ratio 5000 with a PSF floor: fp32 entries alone cannot express det."""
    mu, log_scales, theta, points, weights = _synthetic_needles(5000.0)
    inv_a, shear, coef, norm = convolved_gaussian_coefficients(
        log_scales.float(), theta.float(), PSF_COV.float()
    )
    assert torch.isfinite(norm).all() and (norm > 0.0).all()
    assert (coef > 0.0).all() and (inv_a > 0.0).all()
    delta_x = points.float()[:, 0, None] - mu.float()[None, :, 0]
    delta_y = points.float()[:, 1, None] - mu.float()[None, :, 1]
    sheared = delta_y - shear[None, :] * delta_x
    mahalanobis = delta_x * delta_x * inv_a[None, :] + sheared * sheared * coef[None, :]
    assert (mahalanobis >= 0.0).all()
    # The old adjugate determinant from the same fp32 entries is not positive.
    a, b, d = SplatField.covariance_components_from_rows(log_scales.float(), theta.float())
    total_a, total_b, total_d = a + PSF_COV[0, 0].float(), b, d + PSF_COV[1, 1].float()
    assert (total_a * total_d - total_b * total_b <= 0.0).any()


def test_gradients_are_finite_and_match_the_fp64_reference() -> None:
    mu, log_scales, theta, points, weights = _synthetic_needles(145.0, n_splats=120)
    upstream = torch.linspace(0.5, 1.5, mu.shape[0], dtype=torch.float64)

    field = _field(mu, log_scales, theta, torch.float32)
    exposure = object()
    cache = _cache(exposure, points.float(), weights.float(), PSF_COV.float(), AREA_ARCSEC2)
    _, spatial = _spatial_weights(field, cache, exposure, candidate=torch.arange(mu.shape[0]))
    (spatial * upstream.float()).sum().backward()

    leaves = tuple(
        parameter.detach().double().clone().requires_grad_()
        for parameter in (field.mu, field.log_scales, field.theta)
    )
    reference = _reference_weights(*leaves, points, weights, PSF_COV, AREA_ARCSEC2)
    expected = torch.autograd.grad((reference * upstream).sum(), leaves)
    for name, actual, wanted in zip(("mu", "log_scales", "theta"), (field.mu.grad, field.log_scales.grad, field.theta.grad), expected):
        assert actual is not None, name
        assert torch.isfinite(actual).all(), name
        assert _relative_to_peak(actual, wanted) < 1.0e-4, name




def test_empty_candidates_yield_empty_weights() -> None:
    mu, log_scales, theta, points, weights = _synthetic_needles(10.0, n_splats=8)
    field = _field(mu, log_scales, theta, torch.float32)
    exposure = object()
    cache = _cache(exposure, points.float(), weights.float(), PSF_COV.float(), AREA_ARCSEC2)
    empty = torch.empty(0, dtype=torch.long)

    candidate, spatial = _spatial_weights(field, cache, exposure, candidate=empty)
    assert candidate.numel() == 0
    assert spatial.shape == (0,)
    assert spatial.dtype == torch.float32
    assert candidate_aperture_weights(field, empty, cache.points, cache.weights, cache.psf_cov, AREA_ARCSEC2).shape == (0,)


def test_noncontiguous_points_and_candidate_views_match_contiguous_inputs() -> None:
    mu, log_scales, theta, points, weights = _synthetic_needles(145.0, n_splats=64)
    field = _field(mu, log_scales, theta, torch.float32)
    contiguous_points = points.float()
    strided_points = torch.stack([contiguous_points[:, 0], contiguous_points[:, 1]], dim=0).T
    assert not strided_points.is_contiguous()
    candidate = torch.arange(0, 64, 2)
    assert candidate.is_contiguous()
    strided_candidate = torch.arange(64)[::2]
    assert not strided_candidate.is_contiguous()

    expected = candidate_aperture_weights(field, candidate, contiguous_points, weights.float(), PSF_COV.float(), AREA_ARCSEC2)
    actual = candidate_aperture_weights(field, strided_candidate, strided_points, weights.float(), PSF_COV.float(), AREA_ARCSEC2)
    assert torch.equal(actual, expected)




@pytest.fixture
def tf32_flags():
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous[0]
    torch.backends.cudnn.allow_tf32 = previous[1]
    torch.set_float32_matmul_precision(previous[2])


def _set_tf32(flag: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag


def _cuda_supports_tf32() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and not _cuda_supports_tf32(),
    reason="TF32 needs compute capability 8 or above; the legacy failure cannot occur here",
)
def test_cuda_legacy_contraction_under_tf32_reproduces_the_captured_failure(compact_capture, tf32_flags) -> None:
    """The captured inputs, at their captured batch shape, must still break the replaced contraction."""
    capture = _full_shape_capture(compact_capture)
    _set_tf32(True)
    delta = capture["delta"].cuda()
    precision = capture["precision"].cuda()
    legacy = _legacy_quadratic_form(delta, precision)
    offending = int(capture["offending_candidate_index"])
    assert float(legacy[:, offending].min()) < 0.0
    covariance = capture["covariance"].cuda()
    norm = 1.0 / (2.0 * math.pi * torch.sqrt(torch.linalg.det(covariance)))
    legacy_weights = torch.sum(
        norm[None, :] * torch.exp(-0.5 * legacy) * capture["weights"].cuda()[:, None], dim=0
    ) * float(capture["area_arcsec2"])
    mu, log_scales, theta = _shapes_from_capture(capture)
    expected = _reference_weights(
        mu, log_scales, theta, capture["points"], capture["weights"], capture["psf_cov"], float(capture["area_arcsec2"])
    )
    assert float(legacy_weights[offending]) > 1.0e3 * float(expected[offending])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("allow_tf32", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_spatial_weights_match_the_reference_with_and_without_tf32(
    compact_capture, tf32_flags, allow_tf32: bool, dtype: torch.dtype
) -> None:
    capture = _full_shape_capture(compact_capture)
    _set_tf32(allow_tf32)
    mu, log_scales, theta = _shapes_from_capture(capture)
    field = _field(mu, log_scales, theta, dtype).cuda()
    points = capture["points"].to(device="cuda", dtype=dtype)
    weights = capture["weights"].to(device="cuda", dtype=dtype)
    psf_cov = capture["psf_cov"].to(device="cuda", dtype=dtype)
    area = float(capture["area_arcsec2"])
    exposure = object()
    cache = _cache(exposure, points, weights, psf_cov, area)
    candidate, spatial = _spatial_weights(field, cache, exposure)
    expected = _reference_weights(mu, log_scales, theta, capture["points"], capture["weights"], capture["psf_cov"], area)
    assert torch.equal(candidate.cpu(), torch.arange(mu.shape[0]))
    assert torch.isfinite(spatial).all()
    assert (spatial >= 0.0).all()
    assert _relative_to_peak(spatial.cpu(), expected) < (2.0e-6 if dtype == torch.float32 else 1.0e-12)
    offending = int(capture["offending_candidate_index"])
    assert float(spatial.detach()[offending]) == pytest.approx(float(expected[offending]), rel=1.0e-4)
    _set_tf32(not allow_tf32)
    _, other = _spatial_weights(field, cache, exposure, candidate=candidate)
    # No matmul-class kernel remains on the path, so the flag cannot move a bit.
    assert torch.equal(other, spatial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("allow_tf32", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parameter_gradients_match_the_fp64_reference_on_the_captured_needle(
    compact_capture, tf32_flags, allow_tf32: bool, dtype: torch.dtype
) -> None:
    capture = _full_shape_capture(compact_capture)
    _set_tf32(allow_tf32)
    mu, log_scales, theta = _shapes_from_capture(capture)
    upstream = torch.linspace(0.5, 1.5, mu.shape[0], dtype=torch.float64)
    field = _field(mu, log_scales, theta, dtype).cuda()
    exposure = object()
    area = float(capture["area_arcsec2"])
    cache = _cache(
        exposure,
        capture["points"].to(device="cuda", dtype=dtype),
        capture["weights"].to(device="cuda", dtype=dtype),
        capture["psf_cov"].to(device="cuda", dtype=dtype),
        area,
    )
    _, spatial = _spatial_weights(field, cache, exposure, candidate=torch.arange(mu.shape[0], device="cuda"))
    (spatial * upstream.to(device="cuda", dtype=dtype)).sum().backward()

    # The reference leaves are float64 copies of the exact parameter values the
    # field ran with; a float32 leaf would hand back float32-rounded gradients.
    leaves = tuple(
        parameter.detach().cpu().double().clone().requires_grad_()
        for parameter in (field.mu, field.log_scales, field.theta)
    )
    reference = _reference_weights(*leaves, capture["points"], capture["weights"], capture["psf_cov"], area)
    expected = torch.autograd.grad((reference * upstream).sum(), leaves)
    tolerance = 1.0e-4 if dtype == torch.float32 else 1.0e-9
    for name, actual, wanted in zip(
        ("mu", "log_scales", "theta"), (field.mu.grad, field.log_scales.grad, field.theta.grad), expected
    ):
        assert actual is not None, name
        assert torch.isfinite(actual).all(), name
        assert _relative_to_peak(actual.cpu(), wanted) < tolerance, name
        offending = int(capture["offending_candidate_index"])
        assert float(actual[offending].abs().max()) > 0.0, name
        assert _relative_to_peak(actual[offending].cpu(), wanted[offending]) < (
            1.0e-3 if dtype == torch.float32 else 1.0e-8
        ), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_cuda_autocast_leaves_the_spatial_weights_in_float32(compact_capture, tf32_flags, amp_dtype: torch.dtype) -> None:
    capture = _full_shape_capture(compact_capture)
    _set_tf32(True)
    mu, log_scales, theta = _shapes_from_capture(capture)
    field = _field(mu, log_scales, theta, torch.float32).cuda()
    exposure = object()
    cache = _cache(
        exposure, capture["points"].cuda(), capture["weights"].cuda(), capture["psf_cov"].cuda(), float(capture["area_arcsec2"])
    )
    candidate = torch.arange(mu.shape[0], device="cuda")
    _, plain = _spatial_weights(field, cache, exposure, candidate=candidate)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        _, under_autocast = _spatial_weights(field, cache, exposure, candidate=candidate)
    assert under_autocast.dtype == torch.float32
    assert torch.equal(under_autocast, plain)
