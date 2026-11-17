"""Place SDSS spectra on the imaging flux system.

The images are native SDSS nanomaggies. The spectroscopic pipeline fluxes each
plate to its standards' CALIBFLUX after multiplying it by 10**(-c/2.5):
idlspec2d readplugmap.pro, "putative AB corrections ... the same numbers as in
SDSSFLUX2AB in the photoop product", hard-wired identically in v5_3_12,
v5_3_23 and v5_13_2. Dividing a spectrum by the resulting per-band ratio puts
both modalities on one flux system.
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .bandpasses import Bandpass, build_throughput_mix_weights

IDLSPEC2D_AB_CORRECTION_MAG = {"u": -0.042, "g": 0.036, "r": 0.015, "i": 0.013, "z": -0.002}
SPECTRAL_FLUX_SYSTEMS = ("pipeline", "sdss_imaging")


def pipeline_band_ratios() -> Dict[str, float]:
    """Spectrum/image flux ratio per band implied by the pipeline's anchors."""
    return {band: 10.0 ** (-mag / 2.5) for band, mag in IDLSPEC2D_AB_CORRECTION_MAG.items()}


def _band_response(bandpass: Bandpass, lam: np.ndarray) -> np.ndarray:
    throughput = np.interp(
        lam, np.asarray(bandpass.lam, dtype=np.float64),
        np.asarray(bandpass.throughput, dtype=np.float64), left=0.0, right=0.0,
    )
    return np.clip(throughput, 0.0, None) * lam


def ratio_knots(bandpasses: Mapping[str, Bandpass]) -> Dict[str, float]:
    """Knots s_k of the ratio curve sum_k w_k(lam) s_k.

    w are the spectral calibrator's throughput-mix weights, so the curve has
    the calibrator's shape. s solves A s = ratio exactly, where A[b, k] is the
    photon-weighted (lam R_b) mean of w_k over band b: every band average of
    the curve equals the pipeline ratio for that band.
    """
    ratios = pipeline_band_ratios()
    bands = sorted(bandpasses)
    if bands != sorted(ratios):
        raise ValueError(f"Spectral flux conversion needs bands {sorted(ratios)}, got {bands}.")
    tabulated = np.concatenate([np.asarray(bandpasses[b].lam, dtype=np.float64) for b in bands])
    lam = np.arange(tabulated.min(), tabulated.max() + 0.5, 0.5)
    weights, _ = build_throughput_mix_weights(lam, dict(bandpasses), bands)
    design = np.empty((len(bands), len(bands)), dtype=np.float64)
    for row, band in enumerate(bands):
        response = _band_response(bandpasses[band], lam)
        design[row] = np.trapezoid(response[:, None] * weights, lam, axis=0) / np.trapezoid(response, lam)
    target = np.array([ratios[b] for b in bands], dtype=np.float64)
    knots = np.linalg.solve(design, target)
    if not np.allclose(design @ knots, target, rtol=0.0, atol=1e-12) or np.any(knots <= 0.0):
        raise ValueError(f"Spectral flux conversion knots are not a valid solution: {knots}.")
    return {band: float(value) for band, value in zip(bands, knots)}


def spectrum_to_image_ratio(
    lam: np.ndarray, bandpasses: Mapping[str, Bandpass], knots: Mapping[str, float],
) -> np.ndarray:
    """Spectrum/image flux ratio at each wavelength; divide a spectrum by it."""
    bands = sorted(knots)
    weights, covered = build_throughput_mix_weights(np.asarray(lam, dtype=np.float64), dict(bandpasses), bands)
    if not bool(np.all(covered > 0)):
        raise ValueError("Spectral flux conversion is undefined outside the bandpass coverage.")
    return weights @ np.array([knots[b] for b in bands], dtype=np.float64)
