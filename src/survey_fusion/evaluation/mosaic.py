from __future__ import annotations

import numpy as np


def finalize_mosaic(
    *,
    data_num: np.ndarray,
    model_num: np.ndarray,
    weight_sum: np.ndarray,
    variance_num: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalize mosaic accumulators into source, residual, and uncertainty maps."""
    arrays = tuple(
        np.asarray(array, dtype=np.float64)
        for array in (data_num, model_num, weight_sum, variance_num)
    )
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("All mosaic accumulators must have the same shape.")
    valid = (
        np.isfinite(arrays[2])
        & np.isfinite(arrays[3])
        & (arrays[2] > 0.0)
        & (arrays[3] >= 0.0)
    )
    data = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    model = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    uncertainty = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    data[valid] = (arrays[0][valid] / arrays[2][valid]).astype(np.float32)
    model[valid] = (arrays[1][valid] / arrays[2][valid]).astype(np.float32)
    uncertainty[valid] = (
        np.sqrt(arrays[3][valid]) / arrays[2][valid]
    ).astype(np.float32)
    return {
        "data": data,
        "model": model,
        "residual": data - model,
        "uncertainty": uncertainty,
        "valid": valid,
    }


def accumulate_bilinear_samples(
    *,
    data_num: np.ndarray,
    model_num: np.ndarray,
    weight_sum: np.ndarray,
    variance_num: np.ndarray,
    coverage: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
) -> int:
    """Accumulate irregular image samples on a common grid.

    The data and exposure-matched model use identical inverse-variance and
    bilinear weights. ``variance_num`` retains the squared interpolation
    weights required for the uncertainty of the normalized mosaic.
    """
    arrays = (data_num, model_num, weight_sum, variance_num, coverage)
    if any(array.shape != data_num.shape for array in arrays[1:]):
        raise ValueError("All mosaic accumulators must have the same shape.")
    if data_num.ndim != 2:
        raise ValueError("Mosaic accumulators must be two-dimensional.")

    values = tuple(
        np.asarray(array, dtype=np.float64).reshape(-1)
        for array in (x, y, data, model, sigma)
    )
    n = values[0].size
    if any(array.size != n for array in values[1:]):
        raise ValueError("Sample coordinate, value, model, and sigma arrays must match.")
    x_values, y_values, data_values, model_values, sigma_values = values
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.isfinite(data_values)
        & np.isfinite(model_values)
        & np.isfinite(sigma_values)
        & (sigma_values > 0.0)
    )
    if not np.any(valid):
        return 0

    x_values = x_values[valid]
    y_values = y_values[valid]
    data_values = data_values[valid]
    model_values = model_values[valid]
    sigma_values = sigma_values[valid]
    x0 = np.floor(x_values).astype(np.int64)
    y0 = np.floor(y_values).astype(np.int64)
    dx = x_values - x0
    dy = y_values - y0
    inverse_variance = 1.0 / np.square(sigma_values)
    height, width = data_num.shape
    accepted = 0

    for x_offset, y_offset, kernel_weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + x_offset
        yi = y0 + y_offset
        keep = (
            (kernel_weight > 0.0)
            & (xi >= 0)
            & (xi < width)
            & (yi >= 0)
            & (yi < height)
        )
        if not np.any(keep):
            continue
        flat = yi[keep] * width + xi[keep]
        kernel = kernel_weight[keep]
        weighted = kernel * inverse_variance[keep]
        np.add.at(data_num.reshape(-1), flat, weighted * data_values[keep])
        np.add.at(model_num.reshape(-1), flat, weighted * model_values[keep])
        np.add.at(weight_sum.reshape(-1), flat, weighted)
        np.add.at(
            variance_num.reshape(-1),
            flat,
            np.square(kernel) * inverse_variance[keep],
        )
        np.add.at(coverage.reshape(-1), flat, kernel)
        accepted += int(np.count_nonzero(keep))
    return accepted
