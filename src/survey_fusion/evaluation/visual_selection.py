from __future__ import annotations

import math

import numpy as np


def _local_maxima(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    peaks = valid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(values, dy, axis=0), dx, axis=1)
            peaks &= values > shifted
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    return peaks


def _window_counts(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    values = np.asarray(mask, dtype=np.int64)
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y = np.arange(values.shape[0])
    x = np.arange(values.shape[1])
    y0 = np.maximum(0, y - radius)
    y1 = np.minimum(values.shape[0], y + radius + 1)
    x0 = np.maximum(0, x - radius)
    x1 = np.minimum(values.shape[1], x + radius + 1)
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def select_visual_panel_centres(
    reference: np.ndarray,
    *,
    native_shape: tuple[int, int],
    downsample: int,
    native_margin: int = 0,
) -> list[dict[str, int | str]]:
    """Select fixed panel centres using the data reference alone."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("reference must be a two-dimensional overview.")
    if downsample <= 0:
        raise ValueError("downsample must be positive.")
    if native_margin < 0:
        raise ValueError("native_margin must be non-negative.")
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("reference has no finite pixels.")

    native_height, native_width = (int(native_shape[0]), int(native_shape[1]))
    anchors = (
        ("center", 0.50, 0.50),
        ("southwest_quadrant", 0.25, 0.25),
        ("southeast_quadrant", 0.75, 0.25),
        ("northwest_quadrant", 0.25, 0.75),
        ("northeast_quadrant", 0.75, 0.75),
    )
    selected = [
        {
            "category": category,
            "x": int(round(x_fraction * (native_width - 1))),
            "y": int(round(y_fraction * (native_height - 1))),
        }
        for category, x_fraction, y_fraction in anchors
    ]

    finite_values = values[valid]
    background = float(np.median(finite_values))
    noise = float(np.median(np.abs(finite_values - background)) * 1.4826)
    noise = max(noise, np.finfo(np.float64).eps)
    peaks = _local_maxima(values, valid & (values > background + 3.0 * noise))
    overview_margin = int(math.ceil(native_margin / downsample))
    eligible = np.ones_like(peaks)
    if overview_margin:
        eligible[:] = False
        eligible[
            overview_margin : values.shape[0] - overview_margin,
            overview_margin : values.shape[1] - overview_margin,
        ] = True
        peaks &= eligible
    peak_y, peak_x = np.nonzero(peaks)
    if peak_x.size == 0:
        raise ValueError("reference has no data-selected peaks.")
    peak_flux = values[peak_y, peak_x]

    order = np.argsort(peak_flux)[::-1]
    bright_x = int(peak_x[order[0]])
    bright_y = int(peak_y[order[0]])
    isolation_radius = max(2, int(round(128 / downsample)))
    for candidate in order:
        dx = peak_x - int(peak_x[candidate])
        dy = peak_y - int(peak_y[candidate])
        neighbours = np.count_nonzero(
            (dx * dx + dy * dy <= isolation_radius * isolation_radius)
            & ((dx != 0) | (dy != 0))
        )
        if neighbours == 0:
            bright_x = int(peak_x[candidate])
            bright_y = int(peak_y[candidate])
            break
    selected.append(
        {
            "category": "bright_isolated",
            "x": int(bright_x * downsample),
            "y": int(bright_y * downsample),
        }
    )

    count_radius = max(2, int(round(256 / downsample)))
    crowded_counts = _window_counts(peaks, count_radius)
    crowded_counts[~(valid & eligible)] = -1
    crowded_y, crowded_x = np.unravel_index(
        int(np.argmax(crowded_counts)),
        crowded_counts.shape,
    )
    selected.append(
        {
            "category": "crowded_blend",
            "x": int(crowded_x * downsample),
            "y": int(crowded_y * downsample),
        }
    )

    low, high = np.quantile(peak_flux, [0.25, 0.65])
    faint_peaks = peaks & (values >= low) & (values <= high)
    faint_counts = _window_counts(faint_peaks, count_radius)
    faint_counts[~(valid & eligible)] = -1
    faint_y, faint_x = np.unravel_index(
        int(np.argmax(faint_counts)),
        faint_counts.shape,
    )
    selected.append(
        {
            "category": "faint_sources",
            "x": int(faint_x * downsample),
            "y": int(faint_y * downsample),
        }
    )
    return selected
