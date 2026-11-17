"""Backfill SDSS manifests with three-component DGPSF and KL PSF mixtures.

The input manifest is never modified.  Each component is stored as
``[weight, vxx, vxy, vyy]`` in native detector-pixel units, with ``x=column``
and ``y=row``.  DGPSF fits target the normalized profile truncated at 15
arcsec.  KL fits are evaluated at the true cutout center in native-frame
coordinates and deconvolved from the stamp's detector-pixel response before
storage; a failed KL fit falls back loudly to that exposure's DGPSF fit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import least_squares
from scipy.special import i0e
from scipy.stats import multivariate_normal

from survey_fusion.data.psf_mixture import (
    PSF_MIXTURE_DIGEST_PAYLOAD_KEY,
    PSF_MIXTURE_DIGEST_VERSION,
    psf_mixture_digest,
)


BAND_INDEX = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
DGPSF_COLUMNS = (
    "psf_sigma1", "psf_sigma2", "psf_b", "psf_p0", "psf_beta", "psf_sigmap",
)
EE_RADII_ARCSEC = (0.75, 1.0, 1.5, 3.0)
DGPSF_TRUNCATE_ARCSEC = 15.0
KL_TV_THRESHOLD = 0.08
PIXEL_VARIANCE_NATIVE = 1.0 / 12.0
KL_COVARIANCE_EIGENVALUE_FLOOR_NATIVE = 1e-5
CANONICAL_DECIMALS = 5
RC_SCALE = 0.001
K = 3

# KL eigenimage stamps are pixel-integrated.  Fits therefore subtract the
# native 1/12 pixel variance before storage, and runtime QA adds it back once.
KL_STAMP_PIXEL_RESPONSE_EVIDENCE = "pixel_integrated_decided_upstream"
KL_PIXEL_RESPONSE_CONVENTION = "deconvolved_1_12_native"
DGPSF_PIXEL_RESPONSE_CONVENTION = "analytic_unfolded_native"
FINAL_QA_CONVENTION = "runtime_pixel_folded_covariance_1_12_native"


class KLFitGateError(RuntimeError):
    """A converged KL fit that failed final-operator QA."""

    def __init__(self, message: str, qa: dict[str, object]):
        super().__init__(message)
        self.qa = qa


def _pixel_scale_arcsec(path: Path) -> float:
    with fits.open(path) as hdul:
        wcs = WCS(hdul[0].header).celestial
    scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid pixel scale {scale!r} from {path}.")
    return scale


def _read_dgpsf_row(path: Path, band: str) -> dict[str, float]:
    index = BAND_INDEX[band]
    with fits.open(path) as hdul:
        table = hdul[6].data
        values = {}
        for column in DGPSF_COLUMNS:
            data = table[column]
            values[column] = (
                float(data[0][index])
                if getattr(data, "ndim", 1) > 1
                else float(data[index])
            )
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{path.name} {band}: non-finite DGPSF parameters.")
    if values["psf_sigma1"] <= 0 or values["psf_sigma2"] <= 0 or values["psf_sigmap"] <= 0:
        raise ValueError(f"{path.name} {band}: DGPSF widths must be positive.")
    if values["psf_b"] < 0 or values["psf_p0"] < 0 or values["psf_beta"] <= 2:
        raise ValueError(f"{path.name} {band}: invalid DGPSF wing parameters.")
    return values


def _read_kl_table(path: Path, band: str) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path) as hdul:
        data = hdul[BAND_INDEX[band] + 1].data
        eigs: list[np.ndarray] = []
        coeffs: list[np.ndarray] = []
        polynomial_shape = None
        for record in data:
            nrow_b, ncol_b = int(record["nrow_b"]), int(record["ncol_b"])
            rnrow, rncol = int(record["RNROW"]), int(record["RNCOL"])
            if rnrow != rncol or rnrow % 2 != 1:
                raise ValueError(
                    f"{path.name} {band}: KL eigenimage must be square and odd, got {rnrow}x{rncol}."
                )
            coeff = np.asarray(record["c"], dtype=np.float64)
            if coeff.ndim == 0:
                coeff = coeff.reshape(1, 1)
            elif coeff.ndim == 1:
                side = int(round(math.sqrt(coeff.size)))
                if side * side != coeff.size:
                    raise ValueError(f"{path.name} {band}: cannot reshape KL coefficient array.")
                coeff = coeff.reshape(side, side)
            if nrow_b > coeff.shape[0] or ncol_b > coeff.shape[1]:
                raise ValueError(f"{path.name} {band}: KL polynomial order exceeds coefficient shape.")
            if polynomial_shape is None:
                polynomial_shape = (nrow_b, ncol_b)
            elif polynomial_shape != (nrow_b, ncol_b):
                raise ValueError(f"{path.name} {band}: mixed KL polynomial orders.")
            coeffs.append(coeff[:nrow_b, :ncol_b])
            eigs.append(
                np.asarray(record["rrows"], dtype=np.float64)[: rnrow * rncol]
                .reshape(rnrow, rncol)
            )
    eig = np.stack(eigs)
    coeff = np.stack(coeffs)
    if not np.isfinite(eig).all() or not np.isfinite(coeff).all():
        raise ValueError(f"{path.name} {band}: non-finite KL basis values.")
    return eig, coeff


def _kl_stamp(eig: np.ndarray, coeff: np.ndarray, row: float, col: float) -> np.ndarray:
    row_powers = (RC_SCALE * row) ** np.arange(coeff.shape[1])
    col_powers = (RC_SCALE * col) ** np.arange(coeff.shape[2])
    weights = np.einsum("kij,i,j->k", coeff, row_powers, col_powers)
    stamp = np.einsum("k,kij->ij", weights, eig)
    total = float(stamp.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"KL stamp at row={row}, col={col} has non-positive sum {total!r}.")
    return stamp / total


def _cutout_frame_offsets(cutout_path: Path, frame_path: Path) -> tuple[float, float]:
    with fits.open(cutout_path) as hdul:
        cutout_wcs = WCS(hdul[0].header).celestial
    with fits.open(frame_path) as hdul:
        frame_wcs = WCS(hdul[0].header).celestial
    probe_x = np.array([0.0, 1.0, 0.0])
    probe_y = np.array([0.0, 0.0, 1.0])
    world = cutout_wcs.pixel_to_world(probe_x, probe_y)
    frame_x, frame_y = frame_wcs.world_to_pixel(world)
    dx = np.array([frame_x[1] - frame_x[0], frame_y[1] - frame_y[0]])
    dy = np.array([frame_x[2] - frame_x[0], frame_y[2] - frame_y[0]])
    if (
        abs(dx[0] - 1.0) > 1e-3 or abs(dx[1]) > 1e-3
        or abs(dy[0]) > 1e-3 or abs(dy[1] - 1.0) > 1e-3
    ):
        raise ValueError(
            f"Cutout {cutout_path.name} axes are not aligned with its frame "
            f"(dx={dx.tolist()}, dy={dy.tolist()}); cannot map native row/col."
        )
    return float(frame_y[0]), float(frame_x[0])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    result = np.exp(shifted)
    return result / result.sum()


def _dgpsf_density(radius_px: np.ndarray, params: dict[str, float]) -> np.ndarray:
    core = np.exp(-0.5 * (radius_px / params["psf_sigma1"]) ** 2)
    outer = params["psf_b"] * np.exp(-0.5 * (radius_px / params["psf_sigma2"]) ** 2)
    wing = params["psf_p0"] * (
        1.0 + (radius_px / params["psf_sigmap"]) ** 2 / params["psf_beta"]
    ) ** (-0.5 * params["psf_beta"])
    return core + outer + wing


def _radial_mixture_density(radius_px: np.ndarray, weights: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    return np.sum(
        weights[:, None]
        * np.exp(-0.5 * (radius_px[None, :] / sigmas[:, None]) ** 2)
        / (2.0 * np.pi * sigmas[:, None] ** 2),
        axis=0,
    )


def _canonicalize_components(components: list[list[float]]) -> list[list[float]]:
    """Return deterministic, quantized rows independent of optimizer labels."""
    rows = np.asarray(components, dtype=np.float64)
    if rows.shape != (K, 4) or not np.isfinite(rows).all():
        raise RuntimeError(f"KL mixture has invalid component rows with shape {rows.shape}.")
    rows[:, 0] /= rows[:, 0].sum()
    order = sorted(
        range(K),
        key=lambda index: (
            round(math.sqrt(max(np.linalg.det([
                [rows[index, 1], rows[index, 2]],
                [rows[index, 2], rows[index, 3]],
            ]), 0.0)), 12),
            *(round(float(value), 12) for value in rows[index, 1:]),
            round(float(rows[index, 0]), 12),
        ),
    )
    rows = np.round(rows[order], decimals=CANONICAL_DECIMALS)
    rows[-1, 0] = round(1.0 - float(rows[:-1, 0].sum()), CANONICAL_DECIMALS)
    rows[rows == 0.0] = 0.0  # canonicalize the IEEE -0.0 sign bit for digests
    return rows.tolist()


def _render_mixture_stamp(
    shape: tuple[int, int], components: list[list[float]], *, pixel_folded: bool,
) -> np.ndarray:
    """Sample a native-frame mixture at detector-pixel centres."""
    height, width = shape
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    offsets = np.stack((xx - cx, yy - cy), axis=-1)
    model = np.zeros(shape, dtype=np.float64)
    for weight, vxx, vxy, vyy in components:
        covariance = np.array([[vxx, vxy], [vxy, vyy]], dtype=np.float64)
        if pixel_folded:
            covariance += PIXEL_VARIANCE_NATIVE * np.eye(2)
        model += float(weight) * multivariate_normal.pdf(
            offsets, mean=np.zeros(2), cov=covariance,
        )
    return model


def _mixture_outside_mass(
    shape: tuple[int, int], components: list[list[float]], *, pixel_folded: bool,
) -> float:
    """Continuous mixture mass outside the rectangular stamp footprint."""
    height, width = shape
    lower = np.array([-width / 2.0, -height / 2.0], dtype=np.float64)
    upper = -lower
    inside = 0.0
    for index, (weight, vxx, vxy, vyy) in enumerate(components):
        covariance = np.array([[vxx, vxy], [vxy, vyy]], dtype=np.float64)
        if pixel_folded:
            covariance += PIXEL_VARIANCE_NATIVE * np.eye(2)
        component_inside = multivariate_normal.cdf(
            upper, mean=np.zeros(2), cov=covariance, lower_limit=lower,
            rng=np.random.default_rng(index),
        )
        inside += float(weight) * float(component_inside)
    return float(np.clip(1.0 - inside, 0.0, 1.0))


def _score_mixture_stamp(
    target: np.ndarray, components: list[list[float]], pixel_scale_arcsec: float,
    *, pixel_folded: bool,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()
    model = _render_mixture_stamp(target.shape, components, pixel_folded=pixel_folded)
    model = model / model.sum()
    outside_mass = _mixture_outside_mass(
        target.shape, components, pixel_folded=pixel_folded,
    )
    tv_inside = float(0.5 * np.abs(model - target).sum())
    result = {
        "tv_inside": tv_inside,
        "outside_mass": outside_mass,
        "tv": float(tv_inside + 0.5 * outside_mass),
        "central_value": float(model[target.shape[0] // 2, target.shape[1] // 2]),
    }
    cy, cx = (np.asarray(target.shape, dtype=np.float64) - 1.0) / 2.0
    yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]].astype(np.float64)
    radius_arcsec = np.hypot(xx - cx, yy - cy) * pixel_scale_arcsec
    for value in EE_RADII_ARCSEC:
        mask = radius_arcsec <= value
        result[f"ee_error_{value:g}_arcsec"] = float(model[mask].sum() - target[mask].sum())
    return result


def _effective_fwhm_arcsec(
    components: list[list[float]], pixel_scale_arcsec: float, *, truncate_px: float | None = None,
) -> float:
    rows = np.asarray(components, dtype=np.float64)
    if truncate_px is None:
        mean_variance = float(np.sum(rows[:, 0] * (rows[:, 1] + rows[:, 3])) / 2.0)
    else:
        if not np.allclose(rows[:, 2], 0.0) or not np.allclose(rows[:, 1], rows[:, 3]):
            raise ValueError("Truncated effective FWHM requires isotropic components.")
        variances = rows[:, 1]
        scaled_radius = truncate_px**2 / (2.0 * variances)
        component_mass = 1.0 - np.exp(-scaled_radius)
        radial_second_numerator = 2.0 * variances * (
            1.0 - np.exp(-scaled_radius) * (1.0 + scaled_radius)
        )
        mean_variance = float(
            np.sum(rows[:, 0] * radial_second_numerator)
            / np.sum(rows[:, 0] * component_mass)
            / 2.0
        )
    return 2.355 * math.sqrt(mean_variance) * pixel_scale_arcsec


def _fold_radial_density(
    radius: np.ndarray, density: np.ndarray, pixel_variance: float,
) -> np.ndarray:
    """Convolve an isotropic radial density with an isotropic Gaussian.

    Exact for radial functions: with sigma^2 = ``pixel_variance``,
    folded(r) = (1/sigma^2) * int exp(-(r-r')^2 / (2 sigma^2))
                * i0e(r r' / sigma^2) * density(r') * r' dr'
    using I0(x) = i0e(x) * exp(x) so the exponent never overflows.  For a
    Gaussian density of variance v this returns the Gaussian of variance
    v + pixel_variance, which is what the runtime pixel fold renders.
    """
    r = radius[:, None]
    rp = radius[None, :]
    kernel = (
        np.exp(-0.5 * (r - rp) ** 2 / pixel_variance)
        * i0e(r * rp / pixel_variance)
        / pixel_variance
    )
    return np.trapezoid(kernel * (density[None, :] * rp), radius, axis=1)


def fit_dgpsf_mixture(
    params: dict[str, float], pixel_scale_arcsec: float,
) -> tuple[list[list[float]], dict[str, object]]:
    rmax_px = DGPSF_TRUNCATE_ARCSEC / pixel_scale_arcsec
    radius = np.linspace(0.0, rmax_px, 6001, dtype=np.float64)
    target = _dgpsf_density(radius, params)
    target /= np.trapezoid(target * 2.0 * np.pi * radius, radius)
    radial_weight = np.sqrt(np.maximum(2.0 * np.pi * radius, 1e-12))

    starts = (
        np.array([0.0, -2.0, -4.0, math.log(params["psf_sigma1"]),
                  math.log(params["psf_sigma2"]), math.log(max(2.5 * params["psf_sigmap"], 0.4))]),
        np.array([0.0, -1.5, -3.5, math.log(0.8 * params["psf_sigma1"]),
                  math.log(1.5 * params["psf_sigma2"]), math.log(max(4.0 * params["psf_sigmap"], 0.6))]),
    )
    best = None
    upper_sigma = max(8.0, rmax_px)
    bounds = (
        np.array([-20.0] * K + [math.log(0.3)] * K),
        np.array([20.0] * K + [math.log(upper_sigma)] * K),
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        weights = _softmax(vector[:K])
        sigmas = np.exp(vector[K:])
        model = _radial_mixture_density(radius, weights, sigmas)
        return (model - target) * radial_weight

    for start in starts:
        solution = least_squares(
            residual, np.clip(start, bounds[0] + 1e-9, bounds[1] - 1e-9),
            bounds=bounds, method="trf", max_nfev=6000,
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        score = float(np.dot(solution.fun, solution.fun))
        if best is None or score < best[0]:
            best = (score, solution)
    assert best is not None
    solution = best[1]
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"DGPSF optimizer failed: {solution.message}")
    weights = _softmax(solution.x[:K])
    sigmas = np.exp(solution.x[K:])
    order = np.argsort(sigmas, kind="stable")
    weights, sigmas = weights[order], sigmas[order]
    mixture = _canonicalize_components([
        [float(weight), float(sigma * sigma), 0.0, float(sigma * sigma)]
        for weight, sigma in zip(weights, sigmas)
    ])
    canonical = np.asarray(mixture, dtype=np.float64)
    weights = canonical[:, 0]
    sigmas = np.sqrt(canonical[:, 1])
    model = _radial_mixture_density(radius, weights, sigmas)
    tv_inside = 0.5 * np.trapezoid(np.abs(model - target) * 2.0 * np.pi * radius, radius)
    tail = float(np.sum(weights * np.exp(-0.5 * (rmax_px / sigmas) ** 2)))
    tv = float(tv_inside + 0.5 * tail)
    # The psField DG profile is continuous (pre-pixel): raw KL stamps carry
    # ~1/12 px^2 more second moment than this profile (measured across the
    # full84 psFields).  The runtime therefore folds the stored mixture once,
    # and a folded-convention QA must fold BOTH sides; comparing the folded
    # model against the continuous target just measures the fold cost.
    folded_sigmas = np.sqrt(sigmas**2 + PIXEL_VARIANCE_NATIVE)
    folded_model = _radial_mixture_density(radius, weights, folded_sigmas)
    folded_target = _fold_radial_density(radius, target, PIXEL_VARIANCE_NATIVE)
    folded_tv_inside = 0.5 * np.trapezoid(
        np.abs(folded_model - folded_target) * 2.0 * np.pi * radius, radius,
    )
    folded_tail = float(
        np.sum(weights * np.exp(-0.5 * (rmax_px / folded_sigmas) ** 2))
    )
    qa: dict[str, object] = {
        "tv": tv,
        "tv_unfolded": tv,
        "tv_folded": float(folded_tv_inside + 0.5 * folded_tail),
        "outside_mass_unfolded": tail,
        "outside_mass_folded": folded_tail,
        "truncate_arcsec": DGPSF_TRUNCATE_ARCSEC,
        "pixel_response": DGPSF_PIXEL_RESPONSE_CONVENTION,
    }
    target_integrand = target * 2.0 * np.pi * radius
    model_integrand = model * 2.0 * np.pi * radius
    folded_target_integrand = folded_target * 2.0 * np.pi * radius
    for value in EE_RADII_ARCSEC:
        stop = int(np.searchsorted(radius, value / pixel_scale_arcsec, side="right"))
        target_ee = float(np.trapezoid(target_integrand[:stop], radius[:stop]))
        model_ee = float(np.trapezoid(model_integrand[:stop], radius[:stop]))
        qa[f"ee_error_{value:g}_arcsec"] = model_ee - target_ee
        folded_model_ee = float(np.trapezoid(
            folded_model[:stop] * 2.0 * np.pi * radius[:stop], radius[:stop],
        ))
        folded_target_ee = float(np.trapezoid(
            folded_target_integrand[:stop], radius[:stop],
        ))
        qa[f"ee_error_{value:g}_arcsec_unfolded"] = model_ee - target_ee
        qa[f"ee_error_{value:g}_arcsec_folded"] = folded_model_ee - folded_target_ee
    if qa["tv_folded"] >= 0.01:
        raise RuntimeError(
            f"DGPSF mixture TV {qa['tv_folded']:.6f} fails < 0.01 gate."
        )
    bad_ee = {
        key: value for key, value in qa.items()
        if key.startswith("ee_error_")
        and key.endswith("_folded")
        and abs(value) > 0.005
    }
    if bad_ee:
        raise RuntimeError(f"DGPSF mixture EE gates failed: {bad_ee}")
    return mixture, qa


def _mog_stamp(shape: tuple[int, int], vector: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    height, width = shape
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    weights = _softmax(vector[:K])
    components: list[list[float]] = []
    model = np.zeros(shape, dtype=np.float64)
    for index in range(K):
        sigma_a = math.exp(float(vector[K + 3 * index]))
        sigma_b = math.exp(float(vector[K + 3 * index + 1]))
        theta = float(vector[K + 3 * index + 2])
        ct, st = math.cos(theta), math.sin(theta)
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        model += (
            weights[index]
            * np.exp(-0.5 * ((u / sigma_a) ** 2 + (v / sigma_b) ** 2))
            / (2.0 * np.pi * sigma_a * sigma_b)
        )
        vxx = ct * ct * sigma_a**2 + st * st * sigma_b**2
        vyy = st * st * sigma_a**2 + ct * ct * sigma_b**2
        vxy = ct * st * (sigma_a**2 - sigma_b**2)
        components.append([float(weights[index]), vxx, vxy, vyy])
    return model, components


def _deconvolve_pixel_response(
    components: list[list[float]],
) -> tuple[list[list[float]], int]:
    deconvolved = []
    n_floor_components = 0
    for weight, vxx, vxy, vyy in components:
        covariance = np.array([[vxx, vxy], [vxy, vyy]], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues -= PIXEL_VARIANCE_NATIVE
        if np.any(eigenvalues < KL_COVARIANCE_EIGENVALUE_FLOOR_NATIVE):
            n_floor_components += 1
            eigenvalues = np.maximum(
                eigenvalues, KL_COVARIANCE_EIGENVALUE_FLOOR_NATIVE,
            )
        covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        deconvolved.append([
            float(weight),
            float(covariance[0, 0]),
            float(covariance[0, 1]),
            float(covariance[1, 1]),
        ])
    return _canonicalize_components(deconvolved), n_floor_components


def _fit_kl_components(
    stamp: np.ndarray,
) -> tuple[list[list[float]], int, int]:
    target = np.asarray(stamp, dtype=np.float64)
    target /= target.sum()
    positive = np.clip(target, 0.0, None)
    positive /= positive.sum()
    height, width = target.shape
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    sigma0 = max(
        math.sqrt(float((positive * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum()) / 2.0),
        0.8,
    )
    lower = np.array([-20.0] * K + [value for _ in range(K) for value in (math.log(0.3), math.log(0.3), -math.pi)])
    upper = np.array([20.0] * K + [value for _ in range(K) for value in (math.log(8.0), math.log(8.0), math.pi)])
    starts = []
    for angle_offset in (0.0, math.pi / 6.0, -math.pi / 6.0):
        vector = [math.log(1.0 / K)] * K
        for index, multiplier in enumerate((0.55, 1.1, 2.2)):
            scale = min(max(sigma0 * multiplier, 0.31), 7.9)
            vector.extend((math.log(scale), math.log(scale * (0.9 if index == 1 else 1.0)), angle_offset))
        starts.append(np.asarray(vector, dtype=np.float64))

    def residual(vector: np.ndarray) -> np.ndarray:
        model, _ = _mog_stamp(target.shape, vector)
        return (model / model.sum() - target).ravel()

    best = None
    for start in starts:
        solution = least_squares(
            residual, np.clip(start, lower + 1e-9, upper - 1e-9), bounds=(lower, upper),
            method="trf", max_nfev=5000, xtol=1e-11, ftol=1e-11, gtol=1e-11,
        )
        score = float(np.dot(solution.fun, solution.fun))
        if best is None or score < best[0]:
            best = (score, solution)
    assert best is not None
    solution = best[1]
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"KL optimizer failed: {solution.message}")
    model, components = _mog_stamp(target.shape, solution.x)
    weights = np.asarray([component[0] for component in components])
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise RuntimeError("KL mixture has non-positive or non-finite weights.")
    weights /= weights.sum()
    for component, weight in zip(components, weights):
        component[0] = float(weight)
        covariance = np.array([[component[1], component[2]], [component[2], component[3]]])
        eigenvalues = np.linalg.eigvalsh(covariance)
        sigmas = np.sqrt(eigenvalues)
        if not np.isfinite(sigmas).all() or np.any(sigmas < 0.3 - 1e-8) or np.any(sigmas > 8.0 + 1e-8):
            raise RuntimeError(f"KL mixture covariance sigma bounds failed: {sigmas.tolist()}")
    stored, n_floor_components = _deconvolve_pixel_response(components)
    return stored, int(solution.nfev), n_floor_components


def fit_kl_mixture(
    stamp: np.ndarray, *, pixel_scale_arcsec: float = 1.0,
    tv_threshold: float = KL_TV_THRESHOLD, qa_padded: bool = False,
) -> tuple[list[list[float]], dict[str, object]]:
    if not math.isfinite(tv_threshold) or not 0.0 < tv_threshold <= 1.0:
        raise ValueError(f"KL TV threshold must be in (0, 1], got {tv_threshold!r}.")
    target = np.asarray(stamp, dtype=np.float64)
    components, optimizer_nfev, n_floor_components = _fit_kl_components(target)
    unfolded = _score_mixture_stamp(
        target, components, pixel_scale_arcsec, pixel_folded=False,
    )
    folded = _score_mixture_stamp(
        target, components, pixel_scale_arcsec, pixel_folded=True,
    )
    qa: dict[str, object] = {
        "tv": folded["tv"],
        "tv_unfolded": unfolded["tv"],
        "tv_folded": folded["tv"],
        "tv_inside_unfolded": unfolded["tv_inside"],
        "tv_inside_folded": folded["tv_inside"],
        "outside_mass": folded["outside_mass"],
        "outside_mass_unfolded": unfolded["outside_mass"],
        "outside_mass_folded": folded["outside_mass"],
        "optimizer_nfev": optimizer_nfev,
        "n_covariance_floor_components": n_floor_components,
        "pixel_response": KL_PIXEL_RESPONSE_CONVENTION,
        "scored_convention": FINAL_QA_CONVENTION,
    }
    for value in EE_RADII_ARCSEC:
        key = f"ee_error_{value:g}_arcsec"
        qa[f"{key}_unfolded"] = unfolded[key]
        qa[f"{key}_folded"] = folded[key]
    unfolded_peak = unfolded["central_value"]
    qa["central_peak_fractional_change_folded_vs_unfolded"] = (
        float(folded["central_value"] / unfolded_peak - 1.0)
        if unfolded_peak != 0.0 else None
    )
    if qa_padded:
        pad = max(2, min(target.shape) // 4)
        padded_target = np.pad(target, pad)
        padded_components, padded_nfev, padded_n_floor = _fit_kl_components(
            padded_target
        )
        padded_folded = _score_mixture_stamp(
            padded_target, padded_components, pixel_scale_arcsec, pixel_folded=True,
        )
        drift = np.asarray(padded_components) - np.asarray(components)
        qa["padded"] = {
            "pad_pixels": pad,
            "optimizer_nfev": padded_nfev,
            "n_covariance_floor_components": padded_n_floor,
            "component_max_abs_drift": float(np.max(np.abs(drift))),
            "component_rms_drift": float(np.sqrt(np.mean(drift**2))),
            "outside_mass_change": float(
                padded_folded["outside_mass"] - folded["outside_mass"]
            ),
            "tv_folded": padded_folded["tv"],
        }
    if float(qa["tv_folded"]) >= tv_threshold:
        raise KLFitGateError(
            f"KL final-operator TV {qa['tv_folded']:.6f} fails < {tv_threshold:.6f} gate",
            qa,
        )
    return components, qa


def backfill(
    manifest_path: Path, observation_root: Path, psfield_dir: Path, out_path: Path,
    *, kl_tv_threshold: float = KL_TV_THRESHOLD, qa_padded: bool = False,
) -> dict:
    if manifest_path.resolve() == out_path.resolve():
        raise ValueError("Refusing to write PSF mixtures in place; --out must name a new manifest file.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exposures = manifest.get("image_exposures")
    if not isinstance(exposures, list) or not exposures:
        raise ValueError("Manifest must contain a non-empty image_exposures list.")
    qa_rows = []
    fallback_count = 0
    fallback_labels = []
    kl_cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray]] = {}
    for exposure in exposures:
        label = str(exposure.get("label", "<unlabelled>"))
        band = str(exposure["band"])
        run, camcol, field = int(exposure["run"]), int(exposure["camcol"]), int(exposure["field"])
        psfield_path = psfield_dir / f"psField-{run:06d}-{camcol}-{field:04d}.fit"
        if not psfield_path.exists():
            raise FileNotFoundError(f"Missing {psfield_path} for exposure {label!r}.")
        cutout_path = observation_root / str(exposure["cutout_path"])
        frame_path = observation_root / str(exposure["frame_path"])
        pixel_scale = _pixel_scale_arcsec(cutout_path)
        dgpsf, dgpsf_qa = fit_dgpsf_mixture(_read_dgpsf_row(psfield_path, band), pixel_scale)
        exposure["psf_mixture_dgpsf"] = dgpsf
        key = (psfield_path, band)
        if key not in kl_cache:
            kl_cache[key] = _read_kl_table(psfield_path, band)
        kl_fit_position = None
        kl_offcenter_positions = []
        kl_offcenter_tv_folded = []
        kl_offcenter_skipped = []
        kl_offcenter_candidates = []
        try:
            row0, col0 = _cutout_frame_offsets(cutout_path, frame_path)
            with fits.open(cutout_path) as hdul:
                height, width = hdul[0].data.shape
            with fits.open(frame_path) as hdul:
                frame_height, frame_width = hdul[0].data.shape
            center_row = row0 + (height - 1) / 2.0
            center_col = col0 + (width - 1) / 2.0
            kl_fit_position = [center_row, center_col]
            row_offset = (height - 1) / 4.0
            col_offset = (width - 1) / 4.0
            for delta_row, delta_col in (
                (-row_offset, -col_offset),
                (-row_offset, col_offset),
                (row_offset, -col_offset),
                (row_offset, col_offset),
            ):
                position = [center_row + delta_row, center_col + delta_col]
                offset = [delta_row, delta_col]
                in_support = (
                    0.0 <= position[0] <= frame_height - 1
                    and 0.0 <= position[1] <= frame_width - 1
                )
                kl_offcenter_candidates.append((offset, position, in_support))
                if not in_support:
                    kl_offcenter_skipped.append({
                        "offset": offset,
                        "position": position,
                        "reason": "outside_frame_support",
                    })
            kl_stamp = _kl_stamp(*kl_cache[key], center_row, center_col)
            kl, kl_qa = fit_kl_mixture(
                kl_stamp, pixel_scale_arcsec=pixel_scale,
                tv_threshold=kl_tv_threshold, qa_padded=qa_padded,
            )
            for offset, position, in_support in kl_offcenter_candidates:
                if not in_support:
                    continue
                try:
                    offcenter_stamp = _kl_stamp(
                        *kl_cache[key], position[0], position[1]
                    )
                    offcenter_score = _score_mixture_stamp(
                        offcenter_stamp, kl, pixel_scale, pixel_folded=True,
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    kl_offcenter_skipped.append({
                        "offset": offset,
                        "position": position,
                        "reason": "diagnostic_error",
                        "error": str(exc),
                    })
                    continue
                kl_offcenter_positions.append(position)
                kl_offcenter_tv_folded.append(offcenter_score["tv"])
            fallback = False
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            kl = [component.copy() for component in dgpsf]
            kl_qa = dict(getattr(exc, "qa", {}))
            kl_qa["gate_error"] = str(exc)
            fallback = True
            fallback_count += 1
            fallback_labels.append(label)
            for offset, position, in_support in kl_offcenter_candidates:
                if in_support:
                    kl_offcenter_skipped.append({
                        "offset": offset,
                        "position": position,
                        "reason": "center_fit_failed",
                    })
            print(f"WARNING: KL->DGPSF FALLBACK for exposure {label!r}: {exc}", flush=True)
        exposure["psf_mixture_kl"] = kl
        dg_effective_fwhm = _effective_fwhm_arcsec(
            dgpsf, pixel_scale,
            truncate_px=DGPSF_TRUNCATE_ARCSEC / pixel_scale,
        )
        kl_effective_fwhm = _effective_fwhm_arcsec(kl, pixel_scale)
        exposure["psf_effective_fwhm_dg_arcsec"] = dg_effective_fwhm
        exposure["psf_effective_fwhm_kl_arcsec"] = kl_effective_fwhm
        # Compatibility alias for schema-8 manifests and existing consumers.
        exposure["psf_effective_fwhm_arcsec"] = kl_effective_fwhm
        exposure[PSF_MIXTURE_DIGEST_PAYLOAD_KEY] = {
            "dgpsf": {
                "operator_kind": "dgpsf",
                "qa_pass": True,
                "fallback": False,
                "pixel_response": DGPSF_PIXEL_RESPONSE_CONVENTION,
            },
            "kl": {
                "operator_kind": "kl",
                "qa_pass": not fallback,
                "fallback": fallback,
                "pixel_response": (
                    DGPSF_PIXEL_RESPONSE_CONVENTION
                    if fallback else KL_PIXEL_RESPONSE_CONVENTION
                ),
            },
        }
        qa_rows.append({
            "label": label,
            "band": band,
            "scored_convention": FINAL_QA_CONVENTION,
            "dgpsf": dgpsf_qa,
            "kl": kl_qa,
            "fallback": fallback,
            "kl_fit_position": kl_fit_position,
            "kl_offcenter_positions": kl_offcenter_positions,
            "kl_offcenter_tv_folded": kl_offcenter_tv_folded,
            "kl_offcenter_skipped": kl_offcenter_skipped,
            "pixel_response": (
                DGPSF_PIXEL_RESPONSE_CONVENTION
                if fallback else KL_PIXEL_RESPONSE_CONVENTION
            ),
            "psf_effective_fwhm_dg_arcsec": dg_effective_fwhm,
            "psf_effective_fwhm_kl_arcsec": kl_effective_fwhm,
        })
    manifest["psf_mixture_digest_version"] = PSF_MIXTURE_DIGEST_VERSION
    manifest["psf_mixture_digest"] = psf_mixture_digest(
        exposures, version=PSF_MIXTURE_DIGEST_VERSION,
    )
    manifest["psf_mixture_qa"] = {
        "schema_version": 1,
        "scored_convention": FINAL_QA_CONVENTION,
        "pixel_variance_native": PIXEL_VARIANCE_NATIVE,
        "pixel_response": KL_PIXEL_RESPONSE_CONVENTION,
        "dgpsf_pixel_response": DGPSF_PIXEL_RESPONSE_CONVENTION,
        "kl_stamp_pixel_response_evidence": KL_STAMP_PIXEL_RESPONSE_EVIDENCE,
        "pixel_response_treatment": (
            "KL stamps are pixel-integrated; stored KL covariances subtract "
            "1/12 native pixel variance before runtime folds it once"
        ),
        "kl_tv_threshold": kl_tv_threshold,
        "qa_padded": qa_padded,
        "n_kl_fallback_dg": fallback_count,
        "kl_fallback_dg_labels": fallback_labels,
        "exposures": qa_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Backfilled {len(exposures)} image exposures -> {out_path}")
    for band in "ugriz":
        rows = [row for row in qa_rows if row["band"] == band]
        if rows:
            kl_unfolded_values = [
                float(row["kl"]["tv_unfolded"])
                for row in rows if "tv_unfolded" in row["kl"]
            ]
            kl_folded_values = [
                float(row["kl"]["tv_folded"])
                for row in rows if "tv_folded" in row["kl"]
            ]
            kl_unfolded_median = (
                float(np.median(kl_unfolded_values)) if kl_unfolded_values else float("nan")
            )
            kl_folded_median = (
                float(np.median(kl_folded_values)) if kl_folded_values else float("nan")
            )
            print(
                f"band {band}: median TV dgpsf "
                f"unfolded={np.median([row['dgpsf']['tv_unfolded'] for row in rows]):.6f} "
                f"folded={np.median([row['dgpsf']['tv_folded'] for row in rows]):.6f}; "
                f"KL unfolded={kl_unfolded_median:.6f} folded={kl_folded_median:.6f}"
            )
    effective = np.asarray([row["psf_effective_fwhm_kl_arcsec"] for row in qa_rows])
    print(f"KL->DGPSF fallbacks: {fallback_count}")
    print(f"KL effective FWHM range: {effective.min():.6f} .. {effective.max():.6f} arcsec")
    print(
        f"QA convention: {FINAL_QA_CONVENTION}; KL stamp pixel-response evidence: "
        f"{KL_STAMP_PIXEL_RESPONSE_EVIDENCE}"
    )
    print(f"psf_mixture_digest: {manifest['psf_mixture_digest']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--psfield-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kl-tv-threshold", type=float, default=KL_TV_THRESHOLD,
        help="Maximum runtime-folded KL total variation before loud DG fallback "
        f"(default: {KL_TV_THRESHOLD:g}).",
    )
    parser.add_argument(
        "--qa-padded", action="store_true",
        help="Refit each KL target with zero padding and record stability diagnostics.",
    )
    args = parser.parse_args()
    backfill(
        args.manifest,
        args.observation_root,
        args.psfield_dir or args.observation_root / "psfields",
        args.out,
        kl_tv_threshold=args.kl_tv_threshold,
        qa_padded=args.qa_padded,
    )


if __name__ == "__main__":
    main()
