from __future__ import annotations

import functools
import math

import pytest
import torch
import torch.nn.functional as F

from survey_fusion.pgm.contracts import PSFMixtureComponents
from survey_fusion.training.datum_trainer import transform_psf_mixture_to_normalized


MIXTURE = PSFMixtureComponents.from_rows([
    [0.6, 1.0, 0.2, 2.0],
    [0.3, 3.0, -0.4, 5.0],
    [0.1, 8.0, 0.5, 10.0],
])


@functools.lru_cache(maxsize=1)
def _dense_pixel_integrated_native_moment() -> torch.Tensor:
    covariance = torch.tensor([[0.7, 0.15], [0.15, 1.2]], dtype=torch.float64)
    coordinate = torch.linspace(-7.0, 7.0, 561, dtype=torch.float64)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    positions = torch.stack((xx, yy), dim=-1)
    precision = torch.linalg.inv(covariance)
    gaussian = torch.exp(
        -0.5 * torch.einsum("...i,ij,...j->...", positions, precision, positions)
    )
    # Fine-grid convolution with a unit square is a supersampled numerical
    # raster of the detector-pixel-integrated Gaussian.
    samples_per_pixel = 41
    pixel_weights = torch.ones(samples_per_pixel, dtype=torch.float64)
    pixel_weights[[0, -1]] = 0.5
    pixel_weights /= pixel_weights.sum()
    kernel_x = pixel_weights.reshape(1, 1, 1, -1)
    kernel_y = kernel_x.transpose(-1, -2)
    integrated = F.conv2d(
        F.conv2d(gaussian[None, None], kernel_x, padding="same"),
        kernel_y, padding="same",
    )[0, 0]
    weights = integrated / integrated.sum()
    mean = (weights[..., None] * positions).sum(dim=(0, 1))
    centered = positions - mean
    return torch.einsum("...,...i,...j->ij", weights, centered, centered)


@pytest.mark.parametrize("jacobian", [
    torch.eye(2, dtype=torch.float64),
    torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
    torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    torch.tensor([[2.0, 0.0], [0.0, 0.5]], dtype=torch.float64),
])
def test_full_jacobian_covariance_transform_oracle(jacobian: torch.Tensor) -> None:
    result = transform_psf_mixture_to_normalized(MIXTURE, jacobian)
    assert not result.isotropic
    for index, (_, vxx, vxy, vyy) in enumerate(MIXTURE.components):
        native = torch.tensor([[vxx, vxy], [vxy, vyy]], dtype=torch.float64)
        expected = jacobian @ native @ jacobian.T
        actual = result.covariances[0, index]
        torch.testing.assert_close(
            actual, torch.tensor([expected[0, 0], expected[0, 1], expected[1, 1]]),
            rtol=0.0, atol=1e-14,
        )


def test_isotropic_reduction_and_pixel_fold() -> None:
    isotropic = PSFMixtureComponents.from_rows([
        [0.5, 1.0, 0.0, 1.0], [0.3, 4.0, 0.0, 4.0], [0.2, 9.0, 0.0, 9.0],
    ])
    scale = 0.02
    angle = 0.37
    jacobian = scale * torch.tensor([
        [math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)],
    ], dtype=torch.float64)
    result = transform_psf_mixture_to_normalized(
        isotropic, jacobian, pixel_variance=1.0 / 12.0,
    )
    assert result.isotropic
    expected = scale**2 * torch.tensor([[1.0, 4.0, 9.0]], dtype=torch.float64) + scale**2 / 12.0
    torch.testing.assert_close(result.covariances, expected, rtol=1e-13, atol=1e-16)


def test_detector_pixel_covariance_is_folded_before_anisotropic_jacobian() -> None:
    isotropic = PSFMixtureComponents.from_rows([[1.0, 1.0, 0.0, 1.0]])
    jacobian = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    result = transform_psf_mixture_to_normalized(
        isotropic, jacobian, pixel_variance=1.0 / 12.0,
    )

    assert not result.isotropic
    torch.testing.assert_close(
        result.covariances[0, 0],
        torch.tensor(
            [4.0 + 1.0 / 3.0, 0.0, 1.0 + 1.0 / 12.0],
            dtype=torch.float64,
        ),
        rtol=0.0, atol=1e-14,
    )


@pytest.mark.parametrize("jacobian", [
    torch.tensor([[1.0, 0.45], [0.0, 1.0]], dtype=torch.float64),
    torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
    torch.tensor([[2.0, 0.0], [0.0, 0.5]], dtype=torch.float64),
])
def test_pixel_fold_matches_dense_supersampled_second_moments(
    jacobian: torch.Tensor,
) -> None:
    mixture = PSFMixtureComponents.from_rows([[1.0, 0.7, 0.15, 1.2]])
    result = transform_psf_mixture_to_normalized(
        mixture, jacobian, pixel_variance=1.0 / 12.0,
    )
    dense_native = _dense_pixel_integrated_native_moment()
    expected = jacobian @ dense_native @ jacobian.T
    actual = result.covariances[0, 0]

    torch.testing.assert_close(
        actual,
        torch.stack((expected[0, 0], expected[0, 1], expected[1, 1])),
        rtol=2e-4, atol=2e-4,
    )


def test_unequal_axis_scales_prevent_isotropic_reduction() -> None:
    isotropic = PSFMixtureComponents.from_rows([[1.0, 1.0, 0.0, 1.0]])
    result = transform_psf_mixture_to_normalized(
        isotropic, torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    assert not result.isotropic
