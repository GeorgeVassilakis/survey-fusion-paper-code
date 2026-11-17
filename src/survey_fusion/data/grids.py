from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

    TorchTensor = torch.Tensor
else: # pragma: no cover
    TorchTensor = Any


@dataclass(slots=True)
class SpatialGrid:
    height: int
    width: int
    x_norm: TorchTensor
    y_norm: TorchTensor
    x_np: np.ndarray
    y_np: np.ndarray


def build_spatial_grid(height: int, width: int, device: torch.device) -> SpatialGrid:
    import torch

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x_norm = 2.0 * x_coords / (width - 1) - 1.0
    y_norm = 2.0 * y_coords / (height - 1) - 1.0

    x_norm_t = torch.tensor(x_norm.flatten(), dtype=torch.float32, device=device)
    y_norm_t = torch.tensor(y_norm.flatten(), dtype=torch.float32, device=device)

    return SpatialGrid(height, width, x_norm_t, y_norm_t, x_coords, y_coords)


@dataclass(slots=True)
class FiberMask:
    """A fiber aperture stored by the pixels it actually covers.

    A dense mask over the model grid is what made full-field COSMOS impossible:
    a 3 arcsec fiber covers about 50 pixels, but on an 84 arcmin tile the grid is
    12727 x 12727, so each mask was a 648 MB float32 array of almost entirely
    zeros. With 910 spectra that is ~590 GB in fiber masks alone.

    Every consumer immediately reduced the dense array to
    `nonzero(mask.flatten() > 0)`, so the indices are the real interface. Keep
    those, and materialise a dense array only where one is genuinely wanted
    (plots of a single exposure).
    """

    shape: tuple[int, int]
    indices: np.ndarray            # flat indices into a (height*width) image
    weights: np.ndarray            # weight per index, same length

    @property
    def height(self) -> int:
        return int(self.shape[0])

    @property
    def width(self) -> int:
        return int(self.shape[1])

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(int(self.shape[0]) * int(self.shape[1]), dtype=np.float32)
        dense[self.indices] = self.weights
        return dense.reshape(self.shape)

    def rowcol(self) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of each covered pixel."""
        return (
            (self.indices // int(self.shape[1])).astype(np.int64),
            (self.indices % int(self.shape[1])).astype(np.int64),
        )


def build_fiber_mask(
    height: int, width: int, pixel_scale_arcsec: float, fiber_diameter_arcsec: float, *,
    center_x: float | None = None, center_y: float | None = None,
) -> FiberMask:
    """Circular fiber aperture, evaluated only over its bounding box.

    Cost is O(r^2) in the fiber radius rather than O(height*width), so this stays
    constant as the model tile grows.
    """
    cy = float(center_y) if center_y is not None else (height - 1) / 2.0
    cx = float(center_x) if center_x is not None else (width - 1) / 2.0
    r_pix = (float(fiber_diameter_arcsec) / 2.0) / float(pixel_scale_arcsec)

    r_ceil = int(np.ceil(r_pix)) + 1
    r0 = max(0, int(np.floor(cy)) - r_ceil)
    r1 = min(int(height) - 1, int(np.ceil(cy)) + r_ceil)
    c0 = max(0, int(np.floor(cx)) - r_ceil)
    c1 = min(int(width) - 1, int(np.ceil(cx)) + r_ceil)
    if r1 < r0 or c1 < c0:
        empty_i = np.zeros((0,), dtype=np.int64)
        return FiberMask((int(height), int(width)), empty_i, np.zeros((0,), dtype=np.float32))

    rows = np.arange(r0, r1 + 1, dtype=np.int64)
    cols = np.arange(c0, c1 + 1, dtype=np.int64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r_pix**2
    sel_r = yy[inside]
    sel_c = xx[inside]
    flat = (sel_r * int(width) + sel_c).astype(np.int64)
    return FiberMask(
        (int(height), int(width)), flat, np.ones(flat.size, dtype=np.float32)
    )
