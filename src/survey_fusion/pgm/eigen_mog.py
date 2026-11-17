"""Per-exposure signed Gaussian mixtures of native KL PSF eigenimages."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..utils.perf_env import eigen_async_validate_enabled


ALGORITHM = "eigen_mog_signed_per_eigenimage_landing_pixel_v1"
SCHEMA = 1
INV_TWO_PI = 1.0 / (2.0 * math.pi)


class EigenMixture:
    """Validated fitted eigenimage tables and KL calibration polynomials."""

    def __init__(self, means, covariances, weights, eigen_sums, kl_coefficients, records, metadata):
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.eigen_sums = np.asarray(eigen_sums, dtype=np.float64)
        self.kl_coefficients = np.asarray(kl_coefficients, dtype=np.float64)
        self.records = list(records)
        self.metadata = dict(metadata)
        if self.means.ndim != 4 or self.means.shape[-1] != 2:
            raise ValueError("means must have shape (E, L, K, 2)")
        exposure_count, eigen_count, component_count = self.means.shape[:3]
        if self.covariances.shape != (exposure_count, eigen_count, component_count, 3):
            raise ValueError("covariances do not match means")
        if self.weights.shape != (exposure_count, eigen_count, component_count):
            raise ValueError("weights do not match means")
        if self.eigen_sums.shape != (exposure_count, eigen_count):
            raise ValueError("eigen_sums do not match mixture rows")
        if self.kl_coefficients.shape[:2] != (exposure_count, eigen_count):
            raise ValueError("kl_coefficients do not match mixture rows")
        if len(self.records) != exposure_count or exposure_count != 410:
            raise ValueError("Exactly 410 exposure records are required")
        arrays = (self.means, self.covariances, self.weights, self.eigen_sums, self.kl_coefficients)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Sidecar contains nonfinite values")
        sxx, sxy, syy = np.moveaxis(self.covariances, -1, 0)
        if (sxx <= 0).any() or (syy <= 0).any() or (sxx * syy - sxy * sxy <= 0).any():
            raise ValueError("Every component covariance must be positive definite")
        scale = max(float(np.abs(self.eigen_sums).max()), 1e-300)
        if not np.allclose(self.weights.sum(-1), self.eigen_sums, rtol=1e-8, atol=1e-8 * scale):
            raise ValueError("Weight rows do not integrate to eigen_sums")
        zero = self.eigen_sums == 0
        if not np.all(self.weights[zero] == 0):
            raise ValueError("Padded zero eigenimages must have zero weights")
        self.by_label = {str(record["label"]): index for index, record in enumerate(self.records)}
        if len(self.by_label) != exposure_count:
            raise ValueError("Exposure labels must be unique")
        self.k = component_count
        self.n_eigen = eigen_count
        self._device_tables = {}

    def device_tables(self, device, dtype):
        """Cache immutable exposure tables, without a per-pixel geometry copy."""
        key = (torch.device(device), dtype)
        if key not in self._device_tables:
            self._device_tables[key] = {
                name: torch.as_tensor(value, device=device, dtype=dtype).contiguous()
                for name, value in (
                    ("weights", self.weights),
                    ("means", self.means),
                    ("covariances", self.covariances),
                    ("sums", self.eigen_sums),
                    ("coefficients", self.kl_coefficients),
                )
            }
            tables = self._device_tables[key]
            means = tables["means"].flatten(1, 2)
            cov = tables["covariances"].flatten(1, 2)
            tables["geometry"] = tuple(
                t.contiguous() for t in (means[..., 0], means[..., 1], cov[..., 0], cov[..., 1], cov[..., 2])
            )
        return self._device_tables[key]

    def cached_pixel_tables(self, exposure_index, row, col, compute_dtype):
        """Keep polynomial arithmetic in float64, as in the reference path."""
        fixed = self.device_tables(row.device, row.dtype)
        coefficients = fixed["coefficients"][exposure_index]
        sums = fixed["sums"][exposure_index]
        row_powers = (0.001 * row[:, None]) ** torch.arange(
            coefficients.shape[-2], device=row.device, dtype=row.dtype
        )
        col_powers = (0.001 * col[:, None]) ** torch.arange(
            coefficients.shape[-1], device=row.device, dtype=row.dtype
        )
        alpha = torch.einsum("pi,pj,plij->pl", row_powers, col_powers, coefficients)
        total = (alpha * sums).sum(1)
        asynchronous = total.is_cuda and eigen_async_validate_enabled()
        if asynchronous:
            torch._assert_async(
                torch.isfinite(total).all() & (total > 0).all(),
                "Native PSF normalization is nonpositive or nonfinite",
            )
        elif not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
            raise ValueError("Native PSF normalization is nonpositive or nonfinite")
        weights = ((alpha / total[:, None])[:, :, None] * fixed["weights"][exposure_index]).flatten(1)
        if asynchronous:
            torch._assert_async(
                torch.isclose(weights.sum(1), torch.ones_like(row), rtol=1e-10, atol=1e-12).all(),
                "Per-pixel eigen_mog weights do not sum to one",
            )
        elif not torch.allclose(weights.sum(1), torch.ones_like(row), rtol=1e-10, atol=1e-12):
            raise ValueError("Per-pixel eigen_mog weights do not sum to one")
        geometry = self.device_tables(row.device, compute_dtype)["geometry"]
        return (weights.to(compute_dtype), *geometry)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema") != SCHEMA:
                raise ValueError("Unsupported eigen_mog sidecar schema")
            if metadata.get("algorithm") != ALGORITHM:
                raise ValueError("Unexpected eigen_mog algorithm")
            if metadata.get("pixel_response") != "already_in_native_stamp":
                raise ValueError("Sidecar must include the native stamp pixel response")
            records = metadata.pop("records")
            return cls(
                data["means"],
                data["covariances"],
                data["weights"],
                data["eigen_sums"],
                data["kl_coefficients"],
                records,
                metadata,
            )

    @property
    def n_components(self) -> int:
        return self.n_eigen * self.k

    def normalized_kl_weights(self, exposure_index, row, col):
        """Return alpha / total at frame coordinates for aligned pixel rows."""
        coefficients = torch.as_tensor(self.kl_coefficients, device=row.device, dtype=row.dtype)[
            exposure_index
        ]
        sums = torch.as_tensor(self.eigen_sums, device=row.device, dtype=row.dtype)[exposure_index]
        row_powers = (0.001 * row[:, None]) ** torch.arange(
            coefficients.shape[-2], device=row.device, dtype=row.dtype
        )
        col_powers = (0.001 * col[:, None]) ** torch.arange(
            coefficients.shape[-1], device=row.device, dtype=row.dtype
        )
        alpha = torch.einsum("pi,pj,plij->pl", row_powers, col_powers, coefficients)
        total = (alpha * sums).sum(1)
        if not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
            raise ValueError("Native PSF normalization is nonpositive or nonfinite")
        return alpha / total[:, None]

    def pixel_tables(self, exposure_index, row, col):
        """Return w, mx, my, sxx, sxy, syy tables, each shaped (P, 4K)."""
        alpha = self.normalized_kl_weights(exposure_index, row, col)
        device, dtype = row.device, row.dtype
        weights = torch.as_tensor(self.weights, device=device, dtype=dtype)[exposure_index]
        means = torch.as_tensor(self.means, device=device, dtype=dtype)[exposure_index]
        covariances = torch.as_tensor(self.covariances, device=device, dtype=dtype)[exposure_index]
        weighted = (alpha[:, :, None] * weights).reshape(len(row), -1)
        means = means.reshape(len(row), -1, 2)
        covariances = covariances.reshape(len(row), -1, 3)
        if not torch.allclose(weighted.sum(1), torch.ones_like(row), rtol=1e-10, atol=1e-12):
            raise ValueError("Per-pixel eigen_mog weights do not sum to one")
        return (
            weighted,
            means[..., 0],
            means[..., 1],
            covariances[..., 0],
            covariances[..., 1],
            covariances[..., 2],
        )

    def identity(self):
        return {
            "algorithm": ALGORITHM,
            "atlas_sha256": self.metadata.get("atlas_sha256"),
            "k": self.k,
            "n_components": self.n_components,
            "fit": self.metadata.get("fit"),
        }


def eigen_mog_pair_values(
    dnx,
    dny,
    cxx,
    cxy,
    cyy,
    w_rows,
    mx_rows,
    my_rows,
    sxx_rows,
    sxy_rows,
    syy_rows,
):
    """Differentiable reference value for already-gathered per-pair tables."""
    tables = (w_rows, mx_rows, my_rows, sxx_rows, sxy_rows, syy_rows)
    if any(table.ndim != 2 or table.shape[0] != dnx.numel() for table in tables):
        raise ValueError("Every table must have shape (pairs, components)")
    if any(table.shape != w_rows.shape for table in tables):
        raise ValueError("All per-pair tables must have the same shape")
    ex = dnx[:, None] - mx_rows
    ey = dny[:, None] - my_rows
    a = cxx[:, None] + sxx_rows
    b = cxy[:, None] + sxy_rows
    d = cyy[:, None] + syy_rows
    determinant = a * d - b * b
    quadratic = (d * ex * ex - 2 * b * ex * ey + a * ey * ey) / determinant
    density = torch.exp(-0.5 * quadratic) * (INV_TWO_PI / determinant.sqrt())
    return (w_rows * density).sum(1)
