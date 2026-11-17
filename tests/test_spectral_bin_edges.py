from __future__ import annotations

import numpy as np
import pytest

from survey_fusion.utils.spectral import (
    lam_edges_from_centers,
    masked_bin_edges,
    per_bin_edges_from_centers,
)


def test_per_bin_edges_pair_contiguous_edges() -> None:
    centers = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    pairs = per_bin_edges_from_centers(centers)
    flat = lam_edges_from_centers(centers)
    assert pairs.shape == (3, 2)
    np.testing.assert_allclose(pairs[:, 0], flat[:-1])
    np.testing.assert_allclose(pairs[:, 1], flat[1:])


def test_masked_bin_edges_keep_true_widths_across_gaps() -> None:
    # uniform 10-bin grid with bins 4-6 masked out
    centers = 5000.0 + 10.0 * np.arange(10, dtype=np.float64)
    valid = np.ones(10, dtype=bool)
    valid[4:7] = False

    pairs = masked_bin_edges(centers, valid)

    assert pairs.shape == (7, 2)
    # every surviving bin keeps its full-grid width of 10 A; edges derived
    # from surviving centers would give bin 3 an upper edge at the midpoint
    # of centers 3 and 7 (5065), i.e. a 20 A bin spanning the masked gap
    np.testing.assert_allclose(pairs[:, 1] - pairs[:, 0], 10.0)
    np.testing.assert_allclose(pairs[3], [5025.0, 5035.0])
    np.testing.assert_allclose(pairs[4], [5065.0, 5075.0])


def test_masked_bin_edges_ignore_nonfinite_masked_centers() -> None:
    centers = np.array([5000.0, 5010.0, np.nan, 5030.0], dtype=np.float64)
    valid = np.array([True, True, False, True])
    pairs = masked_bin_edges(centers, valid)
    assert pairs.shape == (3, 2)
    assert bool(np.isfinite(pairs).all())
    # the finite grid is irregular across the dropped center; the surviving
    # neighbours still meet at the midpoints of the finite grid
    np.testing.assert_allclose(pairs[1], [5005.0, 5020.0])
    np.testing.assert_allclose(pairs[2], [5020.0, 5040.0])


def test_masked_bin_edges_reject_valid_nonfinite_centers() -> None:
    centers = np.array([5000.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        masked_bin_edges(centers, np.array([True, True]))
