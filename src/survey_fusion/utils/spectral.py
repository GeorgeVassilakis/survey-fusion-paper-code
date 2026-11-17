"""Spectral grid and uncertainty helper utilities."""

from __future__ import annotations

import numpy as np

__all__ = (
    "lam_edges_from_centers",
    "per_bin_edges_from_centers",
    "masked_bin_edges",
)


def lam_edges_from_centers(lam_centers: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam_centers, dtype=np.float64)
    if lam.ndim != 1 or lam.size == 0:
        raise ValueError("lam_spec must be a non-empty 1D array.")
    if np.any(np.diff(lam) <= 0):
        raise ValueError("lam_spec must be strictly increasing.")
    if lam.size == 1:
        center = float(lam[0])
        return np.array([center - 0.5, center + 0.5], dtype=np.float64)
    mids = 0.5 * (lam[:-1] + lam[1:])
    left = lam[0] - (mids[0] - lam[0])
    right = lam[-1] + (lam[-1] - mids[-1])
    edges = np.concatenate(([left], mids, [right]))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Computed lam_edges are not strictly increasing.")
    return edges


def per_bin_edges_from_centers(lam_centers: np.ndarray) -> np.ndarray:
    """Per-bin [lower, upper] wavelength edges, shape (n_bins, 2)."""
    edges = lam_edges_from_centers(lam_centers)
    return np.stack([edges[:-1], edges[1:]], axis=1)


def masked_bin_edges(lam_centers: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-bin edges for the valid subset of a wavelength grid.

    Edges are computed on the full grid first, then restricted, so bins
    bordering masked regions keep their true widths. Deriving edges from the
    surviving centers instead would place the shared edge at the midpoint
    across the gap, silently widening every bin next to a mask.
    """
    lam = np.asarray(lam_centers, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool)
    if lam.ndim != 1 or keep.shape != lam.shape:
        raise ValueError("lam_centers must be 1D and valid must match its shape.")
    finite = np.isfinite(lam)
    if np.any(keep & ~finite):
        raise ValueError("valid bins must have finite wavelength centers.")
    pairs = np.full((lam.size, 2), np.nan, dtype=np.float64)
    if np.any(finite):
        pairs[finite] = per_bin_edges_from_centers(lam[finite])
    return pairs[keep]
