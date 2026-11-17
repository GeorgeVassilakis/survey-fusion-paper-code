from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class Bandpass:
    band: str
    lam: np.ndarray
    throughput: np.ndarray
    canonical_id: str

    def as_torch(self, device): # pragma: no cover
        import torch

        lam_t = torch.tensor(self.lam, dtype=torch.float32, device=device)
        T_t = torch.tensor(self.throughput, dtype=torch.float32, device=device)
        return lam_t, T_t


def load_bandpass(registry_path: Path, canonical_id: str, band: str | None = None) -> Bandpass:
    registry = pd.read_csv(registry_path)
    row = registry.loc[registry["canonical_id"] == canonical_id]
    if row.empty:
        raise ValueError(f"Bandpass {canonical_id} not found in {registry_path}")
    curve_path_raw = Path(str(row.iloc[0]["curve_path"])).expanduser()
    candidates: list[Path] = []
    if curve_path_raw.is_absolute():
        candidates.append(curve_path_raw)
    else:
        candidates.append(curve_path_raw)
        candidates.append((registry_path.parent / curve_path_raw).resolve())
        candidates.append((PROJECT_ROOT / curve_path_raw).resolve())
    # Backward compatibility for absolute paths from a different machine.
    candidates.append(PROJECT_ROOT / "data" / "filters" / curve_path_raw.name)

    curve_path = next((p for p in candidates if p.exists()), None)
    if curve_path is None:
        raise FileNotFoundError(
            f"Bandpass curve file not found for {canonical_id}. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )
    curve_df = pd.read_csv(curve_path)
    lam = curve_df["lambda_A"].to_numpy(dtype=float)
    T = curve_df["T"].to_numpy(dtype=float)
    idx = np.argsort(lam)
    lam_sorted = lam[idx]
    T_sorted = T[idx]
    return Bandpass(band or canonical_id, lam_sorted, T_sorted, canonical_id)


def load_bandpasses(registry_path: Path, id_map: Dict[str, str]) -> Dict[str, Bandpass]:
    return {band: load_bandpass(registry_path, cid, band) for band, cid in id_map.items()}


def build_throughput_mix_weights(
    lam_spec: np.ndarray, bandpasses: Dict[str, Bandpass], band_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(lam_spec, dtype=np.float64)
    if lam.ndim != 1:
        raise ValueError("lam_spec must be a 1D array.")
    if lam.size == 0:
        return np.zeros((0, len(band_order)), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    q = np.zeros((lam.size, len(band_order)), dtype=np.float64)
    for idx, band in enumerate(band_order):
        bp = bandpasses.get(band)
        if bp is None:
            continue
        throughput = np.interp(
            lam, np.asarray(bp.lam, dtype=np.float64), np.asarray(bp.throughput, dtype=np.float64), left=0.0,
            right=0.0,
        )
        q[:, idx] = np.clip(throughput, 0.0, None) * lam
    q_sum = q.sum(axis=1, keepdims=True)
    covered = (q_sum[:, 0] > 0).astype(np.float64)
    q_sum_safe = np.where(q_sum > 0, q_sum, 1.0)
    weights = q / q_sum_safe
    return weights, covered
