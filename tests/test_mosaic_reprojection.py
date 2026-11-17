from __future__ import annotations

import numpy as np

from survey_fusion.evaluation.mosaic import accumulate_bilinear_samples, finalize_mosaic


def _empty(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    return tuple(np.zeros(shape, dtype=np.float64) for _ in range(5))


def test_integer_aligned_samples_are_preserved_exactly() -> None:
    data_num, model_num, weight_sum, variance_num, coverage = _empty((2, 2))

    accumulate_bilinear_samples(
        data_num=data_num,
        model_num=model_num,
        weight_sum=weight_sum,
        variance_num=variance_num,
        coverage=coverage,
        x=np.array([0.0, 1.0, 0.0, 1.0]),
        y=np.array([0.0, 0.0, 1.0, 1.0]),
        data=np.array([1.0, 2.0, 3.0, 4.0]),
        model=np.array([1.5, 2.5, 3.5, 4.5]),
        sigma=np.full(4, 2.0),
    )

    np.testing.assert_allclose(data_num / weight_sum, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(model_num / weight_sum, [[1.5, 2.5], [3.5, 4.5]])
    np.testing.assert_allclose(np.sqrt(variance_num) / weight_sum, 2.0)
    np.testing.assert_allclose(coverage, 1.0)


def test_half_pixel_sample_has_no_flux_or_uncertainty_seam() -> None:
    data_num, model_num, weight_sum, variance_num, coverage = _empty((2, 2))

    accumulate_bilinear_samples(
        data_num=data_num,
        model_num=model_num,
        weight_sum=weight_sum,
        variance_num=variance_num,
        coverage=coverage,
        x=np.array([0.5]),
        y=np.array([0.5]),
        data=np.array([7.0]),
        model=np.array([6.0]),
        sigma=np.array([3.0]),
    )

    np.testing.assert_allclose(data_num / weight_sum, 7.0)
    np.testing.assert_allclose(model_num / weight_sum, 6.0)
    np.testing.assert_allclose(np.sqrt(variance_num) / weight_sum, 3.0)
    np.testing.assert_allclose(coverage, 0.25)


def test_overlapping_samples_propagate_inverse_variance() -> None:
    data_num, model_num, weight_sum, variance_num, coverage = _empty((1, 1))

    accumulate_bilinear_samples(
        data_num=data_num,
        model_num=model_num,
        weight_sum=weight_sum,
        variance_num=variance_num,
        coverage=coverage,
        x=np.zeros(2),
        y=np.zeros(2),
        data=np.array([2.0, 4.0]),
        model=np.array([1.0, 5.0]),
        sigma=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(data_num / weight_sum, [[2.4]])
    np.testing.assert_allclose(model_num / weight_sum, [[1.8]])
    np.testing.assert_allclose(np.sqrt(variance_num) / weight_sum, [[np.sqrt(0.8)]])
    np.testing.assert_allclose(coverage, 2.0)


def test_adjacent_offset_exposures_do_not_create_a_constant_field_seam() -> None:
    data_num, model_num, weight_sum, variance_num, coverage = _empty((2, 5))
    x = np.concatenate([np.arange(3), np.arange(3) + 1.5])
    y = np.concatenate([np.zeros(3), np.ones(3)])

    accumulate_bilinear_samples(
        data_num=data_num,
        model_num=model_num,
        weight_sum=weight_sum,
        variance_num=variance_num,
        coverage=coverage,
        x=x,
        y=y,
        data=np.full(6, 11.0),
        model=np.full(6, 10.0),
        sigma=np.ones(6),
    )

    valid = weight_sum > 0.0
    np.testing.assert_allclose((data_num / np.where(valid, weight_sum, 1.0))[valid], 11.0)
    np.testing.assert_allclose((model_num / np.where(valid, weight_sum, 1.0))[valid], 10.0)
    assert np.all(coverage[valid] > 0.0)


def test_samples_with_no_kernel_overlap_are_dropped() -> None:
    arrays = _empty((2, 2))

    accepted = accumulate_bilinear_samples(
        data_num=arrays[0],
        model_num=arrays[1],
        weight_sum=arrays[2],
        variance_num=arrays[3],
        coverage=arrays[4],
        x=np.array([-2.0, 3.0]),
        y=np.array([0.0, 1.0]),
        data=np.ones(2),
        model=np.ones(2),
        sigma=np.ones(2),
    )

    assert accepted == 0
    for array in arrays:
        np.testing.assert_array_equal(array, 0.0)


def test_finalize_mosaic_preserves_mask_and_residual_identity() -> None:
    result = finalize_mosaic(
        data_num=np.array([[4.0, 0.0]]),
        model_num=np.array([[2.0, 0.0]]),
        weight_sum=np.array([[2.0, 0.0]]),
        variance_num=np.array([[1.0, 0.0]]),
    )

    np.testing.assert_allclose(result["data"][0, 0], 2.0)
    np.testing.assert_allclose(result["model"][0, 0], 1.0)
    np.testing.assert_allclose(result["residual"][0, 0], 1.0)
    np.testing.assert_allclose(result["uncertainty"][0, 0], 0.5)
    assert np.isnan(result["data"][0, 1])
    assert not result["valid"][0, 1]
