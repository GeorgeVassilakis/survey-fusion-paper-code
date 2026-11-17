"""Frozen random (pixel, band) datum holdout.

Held-out datums are excluded from every training gradient step via
image_excluded_local_indices_by_exposure and evaluated separately.
"""

from __future__ import annotations

import numpy as np
import torch


def random_datum_holdout_mask(
    n_datums: int, fraction: float, seed: int
) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(int(n_datums), generator=g) < float(fraction)


def random_holdout_local_indices(
    n_local: int, fraction: float, seed: int, salt: int
) -> np.ndarray:
    """Deterministic held-out local (flattened pixel) indices for one exposure.

    salt disambiguates exposures under one seed; (seed, salt) fully
    determines the mask.
    """
    g = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(salt))
    draws = torch.rand(int(n_local), generator=g)
    mask = draws < float(fraction)
    if fraction > 0.0 and n_local > 0 and not mask.any():
        mask[int(draws.argmin())] = True
    return np.flatnonzero(mask.numpy()).astype(np.int64)
