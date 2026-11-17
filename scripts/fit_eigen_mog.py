"""Fit bounded signed Gaussian expansions of every SDSS KL eigenimage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


ALGORITHM = "eigen_mog_signed_per_eigenimage_landing_pixel_v1"
SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def covariance(logs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    variances = torch.exp(2 * logs)
    cosine, sine = theta.cos(), theta.sin()
    return torch.stack(
        (
            cosine * cosine * variances[..., 0] + sine * sine * variances[..., 1],
            cosine * sine * (variances[..., 0] - variances[..., 1]),
            sine * sine * variances[..., 0] + cosine * cosine * variances[..., 1],
        ),
        -1,
    )


def design(xy: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor) -> torch.Tensor:
    delta = xy[None, :, None, :] - means[:, None, :, :]
    sxx, sxy, syy = covariances.unbind(-1)
    determinant = sxx * syy - sxy * sxy
    quadratic = (
        syy[:, None] * delta[..., 0].square()
        - 2 * sxy[:, None] * delta[..., 0] * delta[..., 1]
        + sxx[:, None] * delta[..., 1].square()
    ) / determinant[:, None]
    return torch.exp(-0.5 * quadratic) / (2 * math.pi * torch.sqrt(determinant)[:, None])


def constrained_weights(
    matrix: torch.Tensor,
    target: torch.Tensor,
    integral: torch.Tensor,
    ridge: float = 1e-7,
) -> torch.Tensor:
    gram = matrix.transpose(1, 2) @ matrix
    right = (matrix.transpose(1, 2) @ target[..., None])[..., 0]
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).detach()
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    gram = gram + (ridge * scale)[:, None, None] * eye
    both = torch.linalg.solve(gram, torch.stack((right, torch.ones_like(right)), -1))
    unconstrained, constraint = both.unbind(-1)
    return unconstrained + constraint * (
        (integral - unconstrained.sum(-1)) / constraint.sum(-1)
    )[:, None]


def fit_gaussians(
    targets: torch.Tensor,
    xy: torch.Tensor,
    integrals: torch.Tensor,
    *,
    k: int,
    seed: int,
    steps: int,
    max_offset: float,
    min_sigma: float,
    max_sigma: float,
) -> dict[str, object]:
    torch.manual_seed(seed)
    dtype, device = targets.dtype, targets.device
    row_count = len(targets)
    probabilities = targets.detach().abs() + 1e-8
    selected = torch.multinomial(probabilities, k, replacement=False)
    means = xy[selected].clone()
    means[:, : min(3, k)] = 0
    means += 0.15 * torch.randn_like(means)
    radii = torch.logspace(
        math.log10(0.6), math.log10(4.5), k, device=device, dtype=dtype
    )
    logs = radii.log()[None, :, None].expand(row_count, k, 2).clone()
    theta = torch.rand(row_count, k, dtype=dtype, device=device) * math.pi
    means.requires_grad_()
    logs.requires_grad_()
    theta.requires_grad_()
    parameters = [means, logs, theta]
    optimizer = torch.optim.Adam(parameters, lr=0.025)
    best_loss = torch.full((row_count,), float("inf"), device=device, dtype=dtype)
    best = [
        torch.zeros_like(means),
        torch.zeros_like(logs),
        torch.zeros_like(theta),
        torch.zeros(row_count, k, device=device, dtype=dtype),
    ]
    trace: list[dict[str, float | int]] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        covariances = covariance(logs, theta)
        matrix = design(xy, means, covariances)
        weights = constrained_weights(matrix, targets, integrals)
        prediction = (matrix * weights[:, None]).sum(-1)
        loss = prediction.sub(targets).square().sum(-1) / targets.square().sum(-1).clamp_min(1e-20)
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError("Nonfinite eigenimage fit")
        with torch.no_grad():
            better = loss < best_loss
            best_loss[better] = loss[better]
            for destination, source in zip(best, (means, logs, theta, weights)):
                destination[better] = source[better]
        if step % 100 == 0 or step == steps:
            record = {
                "step": step,
                "median_relative_l2": float(best_loss.sqrt().median()),
                "max_relative_l2": float(best_loss.sqrt().max()),
            }
            trace.append(record)
            print(json.dumps({"seed": seed, **record}), flush=True)
        if step == steps:
            break
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        with torch.no_grad():
            means.clamp_(-max_offset, max_offset)
            logs.clamp_(math.log(min_sigma), math.log(max_sigma))
        for group in optimizer.param_groups:
            group["lr"] = 0.025 * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
            )
    fitted_means, fitted_logs, fitted_theta, fitted_weights = best
    return {
        "means": fitted_means.detach().cpu().numpy(),
        "covariances": covariance(fitted_logs, fitted_theta).detach().cpu().numpy(),
        "weights": fitted_weights.detach().cpu().numpy(),
        "relative_l2": best_loss.sqrt().cpu().numpy(),
        "trace": trace,
    }


def evaluate_numpy(
    xy: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    delta = xy[:, None, :] - means[None, :, :]
    sxx, sxy, syy = covariances.T
    determinant = sxx * syy - sxy * sxy
    quadratic = (
        syy * delta[..., 0] ** 2
        - 2 * sxy * delta[..., 0] * delta[..., 1]
        + sxx * delta[..., 1] ** 2
    ) / determinant
    return (np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(determinant)) * weights).sum(-1)


def sinc_image(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    native = np.arange(image.shape[-1]) - (image.shape[-1] - 1) / 2
    matrix = np.sinc(np.asarray(coords)[:, None] - native[None])
    return np.einsum("ai,...ij,bj->...ab", matrix, image, matrix)


def kl_alpha(coefficients: np.ndarray, row: float, col: float) -> np.ndarray:
    return np.einsum(
        "kij,i,j->k",
        coefficients,
        (0.001 * row) ** np.arange(coefficients.shape[-2]),
        (0.001 * col) ** np.arange(coefficients.shape[-1]),
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def fit_all(
    targets: np.ndarray,
    scales: np.ndarray,
    xy: np.ndarray,
    integrals: np.ndarray,
    nonzero: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], np.ndarray]:
    device = torch.device(args.device)
    grid = torch.tensor(xy, device=device, dtype=torch.float64)
    rows = np.flatnonzero(nonzero)
    chunk_size = len(rows) if args.chunk == 0 else args.chunk
    per_seed: list[dict[str, object]] = []
    fitted_by_seed: list[dict[str, np.ndarray]] = []
    for seed in args.seeds:
        seed_started = time.monotonic()
        assembled = {
            "means": np.zeros((len(rows), args.k, 2), dtype=np.float64),
            "covariances": np.zeros((len(rows), args.k, 3), dtype=np.float64),
            "weights": np.zeros((len(rows), args.k), dtype=np.float64),
            "relative_l2": np.zeros(len(rows), dtype=np.float64),
        }
        chunk_reports = []
        for chunk_index, start in enumerate(range(0, len(rows), chunk_size)):
            stop = min(start + chunk_size, len(rows))
            row_indices = rows[start:stop]
            chunk_seed = seed if args.chunk == 0 else seed * 1000 + chunk_index
            result = fit_gaussians(
                torch.tensor(targets[row_indices], device=device, dtype=torch.float64),
                grid,
                torch.tensor(integrals[row_indices], device=device, dtype=torch.float64),
                k=args.k,
                seed=chunk_seed,
                steps=args.steps,
                max_offset=args.max_offset,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
            )
            trace = result.pop("trace")
            for key in assembled:
                assembled[key][start:stop] = result[key]
            chunk_reports.append({
                "chunk_index": chunk_index,
                "start": start,
                "stop": stop,
                "seed": chunk_seed,
                "trace": trace,
            })
        assembled["weights"] *= scales[rows, None]
        fitted_by_seed.append(assembled)
        per_seed.append({
            "seed": seed,
            "seconds": time.monotonic() - seed_started,
            "chunks": chunk_reports,
            "median_objective_relative_l2": float(np.median(assembled["relative_l2"])),
            "max_objective_relative_l2": float(np.max(assembled["relative_l2"])),
        })
    objective = np.stack([fit["relative_l2"] for fit in fitted_by_seed])
    selected_seed = np.argmin(objective, axis=0)
    merged = {
        key: np.stack([fitted_by_seed[choice][key][index] for index, choice in enumerate(selected_seed)])
        for key in ("means", "covariances", "weights", "relative_l2")
    }
    return merged, per_seed, selected_seed


def validate(
    eigenimages: np.ndarray,
    coefficients: np.ndarray,
    records: list[dict[str, object]],
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    validation_rows = []
    grid = np.arange(-25.0, 25.0) + 0.25
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), -1)
    radius = np.sqrt(xx * xx + yy * yy)
    eigen_sums = eigenimages.sum((-1, -2))
    positions = ((256.0, 256.0), (768.0, 1024.0), (1280.0, 1792.0))
    for exposure_index, record in enumerate(records):
        for row, col in positions:
            alpha = kl_alpha(coefficients[exposure_index], row, col)
            total = float(np.sum(alpha * eigen_sums[exposure_index]))
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Invalid KL normalization for {record['label']} at {row},{col}: {total}")
            native = np.einsum("k,kij->ij", alpha, eigenimages[exposure_index]) / total
            prediction = evaluate_numpy(
                points,
                means[exposure_index].reshape(-1, 2),
                covariances[exposure_index].reshape(-1, 3),
                (weights[exposure_index] * alpha[:, None] / total).ravel(),
            ).reshape(len(grid), len(grid))
            truth = sinc_image(native, grid)
            residual = prediction - truth
            validation_rows.append({
                "label": record["label"],
                "row": row,
                "col": col,
                "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(truth)),
                "max_relative_peak": float(np.max(np.abs(residual)) / np.max(truth)),
                "continuous_integral": float(np.sum(weights[exposure_index] * alpha[:, None]) / total),
                "negative_mass_discrete": float(-prediction[prediction < 0].sum()),
                "encircled_error": {
                    str(aperture): float(residual[radius < aperture].sum())
                    for aperture in (2, 4, 8, 16, 24)
                },
            })
    relative_l2 = np.array([row["relative_l2"] for row in validation_rows])
    return validation_rows, {
        "median_relative_l2": float(np.median(relative_l2)),
        "max_relative_l2": float(np.max(relative_l2)),
    }


def truncation_audit(
    coefficients: np.ndarray,
    eigen_sums: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[dict[str, float | int]], int]:
    radii = (12, 14, 16, 18, 20, 22, 24)
    maximum = {radius: 0.0 for radius in radii}
    grid = np.arange(-40.0, 40.0001, 0.25)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), -1)
    positions = ((256.0, 256.0), (768.0, 1024.0), (1280.0, 1792.0))
    for exposure_index in range(len(coefficients)):
        flat_means = means[exposure_index].reshape(-1, 2)
        flat_covariances = covariances[exposure_index].reshape(-1, 3)
        for row, col in positions:
            alpha = kl_alpha(coefficients[exposure_index], row, col)
            total = float(np.sum(alpha * eigen_sums[exposure_index]))
            mixture = evaluate_numpy(
                points,
                flat_means,
                flat_covariances,
                (weights[exposure_index] * alpha[:, None] / total).ravel(),
            ).reshape(len(grid), len(grid))
            absolute = np.abs(mixture)
            denominator = float(absolute.sum())
            for radius_value in radii:
                outside = (np.abs(xx) > radius_value) | (np.abs(yy) > radius_value)
                fraction = float(absolute[outside].sum() / denominator)
                maximum[radius_value] = max(maximum[radius_value], fraction)
    table = [
        {"radius": radius, "maximum_truncated_absolute_mass_fraction": maximum[radius]}
        for radius in radii
    ]
    passing = [radius for radius in radii if radius <= 20 and maximum[radius] <= 1e-3]
    return table, min(passing) if passing else 20


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    atlas_path = Path(args.atlas).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atlas = np.load(atlas_path, allow_pickle=False)
    eigenimages = np.asarray(atlas["eigenimages"], dtype=np.float64)
    coefficients = np.asarray(atlas["coefficients"], dtype=np.float64)
    atlas_metadata = json.loads(str(atlas["metadata_json"]))
    records = [
        {key: row[key] for key in ("label", "row0", "col0")}
        for row in atlas_metadata["records"]
    ]
    if eigenimages.shape != (410, 4, 51, 51) or coefficients.shape != (410, 4, 5, 5):
        raise ValueError((eigenimages.shape, coefficients.shape))
    if len(records) != 410 or len({record["label"] for record in records}) != 410:
        raise ValueError("Atlas records are not 410 unique labels")

    coords = np.arange(-25.0, 25.01, 0.5)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    xy = np.stack((xx.ravel(), yy.ravel()), -1)
    sampled = sinc_image(eigenimages, coords).reshape(-1, len(xy))
    scales = np.linalg.norm(sampled, axis=-1)
    nonzero = scales > 0
    normalized = np.zeros_like(sampled)
    normalized[nonzero] = sampled[nonzero] / scales[nonzero, None]
    eigen_sums = eigenimages.sum((-1, -2))
    integrals = np.zeros_like(scales)
    integrals[nonzero] = eigen_sums.ravel()[nonzero] / scales[nonzero]

    merged, fit_reports, selected_seed = fit_all(
        normalized, scales, xy, integrals, nonzero, args
    )
    full_means = np.zeros((410 * 4, args.k, 2), dtype=np.float64)
    full_covariances = np.zeros((410 * 4, args.k, 3), dtype=np.float64)
    full_covariances[..., 0] = 1.0
    full_covariances[..., 2] = 1.0
    full_weights = np.zeros((410 * 4, args.k), dtype=np.float64)
    rows = np.flatnonzero(nonzero)
    full_means[rows] = merged["means"]
    full_covariances[rows] = merged["covariances"]
    full_weights[rows] = merged["weights"]
    full_means = full_means.reshape(410, 4, args.k, 2)
    full_covariances = full_covariances.reshape(410, 4, args.k, 3)
    full_weights = full_weights.reshape(410, 4, args.k)
    if not np.allclose(full_weights.sum(-1), eigen_sums, rtol=1e-8, atol=1e-10):
        raise RuntimeError("Fitted weights do not reproduce eigenimage sums")

    validation_rows, validation_summary = validate(
        eigenimages, coefficients, records, full_means, full_covariances, full_weights
    )
    truncation_table, support_radius = truncation_audit(
        coefficients, eigen_sums, full_means, full_covariances, full_weights
    )
    metadata = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "pixel_response": "already_in_native_stamp",
        "atlas_sha256": sha256(atlas_path),
        "records": records,
        "fit": {
            "k": args.k,
            "seeds": args.seeds,
            "steps": args.steps,
            "max_offset": args.max_offset,
            "max_sigma": args.max_sigma,
            "min_sigma": args.min_sigma,
            "grid": "half-pixel arange(-25,25.01,0.5)",
            "chunking": {"chunk": args.chunk, "seed_rule": "seed if unchunked else seed*1000+chunk_index"},
        },
        "validation": validation_summary,
        "support": {"table": truncation_table, "radius": support_radius},
    }
    npz_path = output / "eigen_mog_k10_bounded.npz"
    np.savez_compressed(
        npz_path,
        means=full_means,
        covariances=full_covariances,
        weights=full_weights,
        eigen_sums=eigen_sums,
        kl_coefficients=coefficients,
        metadata_json=np.array(json.dumps(metadata)),
    )
    report = {
        "status": "complete",
        "arguments": vars(args),
        "atlas_sha256": sha256(atlas_path),
        "git_commit": git_commit(),
        "nonzero_eigenimages": int(nonzero.sum()),
        "fit_reports": fit_reports,
        "selected_seed_index": selected_seed.tolist(),
        "per_row_objective_relative_l2": merged["relative_l2"].tolist(),
        "validation_rows": validation_rows,
        "validation_summary": validation_summary,
        "truncation_table": truncation_table,
        "support_radius": support_radius,
        "seconds": time.monotonic() - started,
        "npz_sha256": sha256(npz_path),
    }
    report_path = output / "eigen_mog_k10_bounded.report.json"
    save_json(report_path, report)
    gate_passed = (
        validation_summary["median_relative_l2"] <= 0.015
        and validation_summary["max_relative_l2"] <= 0.06
    )
    completion = {
        "status": "complete",
        "gate_passed": gate_passed,
        "npz": str(npz_path),
        "npz_sha256": report["npz_sha256"],
        "report": str(report_path),
        "validation_summary": validation_summary,
        "support_radius": support_radius,
    }
    save_json(output / ("FIT_COMPLETE.json" if gate_passed else "FIT_GATE_FAILED.json"), completion)
    if not gate_passed:
        raise RuntimeError(
            "Fit gate failed: median relative L2 must be <= 0.015 and max <= 0.06"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 83, 127])
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--max-offset", type=float, default=10.0)
    parser.add_argument("--max-sigma", type=float, default=5.0)
    parser.add_argument("--min-sigma", type=float, default=0.25)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.k != 10:
        raise ValueError("This PSF sidecar schema requires k=10")
    if args.chunk < 0 or args.steps <= 0:
        raise ValueError("chunk must be nonnegative and steps positive")
    if not (0 < args.min_sigma <= args.max_sigma and args.max_offset > 0):
        raise ValueError("Invalid fit bounds")
    return args


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(parse_args())
