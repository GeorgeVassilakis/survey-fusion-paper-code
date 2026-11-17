import numpy as np

from survey_fusion.data.grids import FiberMask, build_fiber_mask


def test_matches_dense_reference():
    """The sparse mask must select exactly the pixels the dense formula would."""
    h, w, ps, diam = 64, 80, 0.396, 3.0
    cx, cy = 33.2, 21.7
    m = build_fiber_mask(h, w, ps, diam, center_x=cx, center_y=cy)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r_pix = (diam / 2.0) / ps
    dense = (((y - cy) ** 2 + (x - cx) ** 2) <= r_pix**2).astype(np.float32)
    assert np.array_equal(m.to_dense(), dense)


def test_cost_is_independent_of_grid_size():
    """A 3 arcsec fiber covers ~50 pixels. On an 84 arcmin tile the model grid is
    12727^2, so a dense mask is 648 MB of mostly zeros and 910 of them is ~590 GB.
    Storage must track the aperture, not the tile."""
    small = build_fiber_mask(256, 256, 0.396, 3.0, center_x=128.0, center_y=128.0)
    big = build_fiber_mask(12727, 12727, 0.396, 3.0, center_x=6000.0, center_y=7000.0)
    assert big.indices.size == small.indices.size
    assert big.indices.nbytes < 2000
    assert big.shape == (12727, 12727)


def test_rowcol_roundtrip():
    m = build_fiber_mask(50, 60, 0.396, 3.0, center_x=25.0, center_y=20.0)
    r, c = m.rowcol()
    assert np.array_equal(r * 60 + c, m.indices)
    assert r.max() < 50 and c.max() < 60


def test_clipped_at_edges():
    m = build_fiber_mask(40, 40, 0.396, 3.0, center_x=1.0, center_y=1.0)
    r, c = m.rowcol()
    assert m.indices.size > 0
    assert r.min() >= 0 and c.min() >= 0


def test_entirely_offgrid_is_empty():
    m = build_fiber_mask(40, 40, 0.396, 3.0, center_x=-500.0, center_y=-500.0)
    assert m.indices.size == 0
    assert m.to_dense().sum() == 0.0
