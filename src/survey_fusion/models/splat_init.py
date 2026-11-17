"""Seed splat positions from local maxima of a band image."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from ..utils.torch_utils import startup_worker_count


def _normalized_xy(exposure, x_pixel, y_pixel) -> torch.Tensor:
    import numpy as np

    omega = np.asarray(
        exposure.obs_wcs.pixel_to_omega_xyz(x_pixel, y_pixel),
        dtype=np.float64,
    ).reshape(-1, 3)
    xn, yn = exposure.wcs.omega_xyz_to_norm_torch(
        torch.as_tensor(omega, dtype=torch.float32)
    )
    return torch.stack([xn, yn], dim=-1)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    """log(exp(x) - 1), written so it stays finite at both ends. The direct
    log(expm1(x)) overflows for large x and loses all precision for small x,
    and the seeded fluxes span both regimes."""
    return x + torch.log(-torch.expm1(-x))


def peaks_from_image(
    img: torch.Tensor, n_peaks: int, min_sep: int = 3
) -> torch.Tensor:
    """Return up to (n_peaks, 2) [row, col] of the brightest local maxima,
    greedily suppressing a (2*min_sep+1) square around each accepted peak."""
    pooled = F.max_pool2d(
        img[None, None], kernel_size=2 * min_sep + 1, stride=1, padding=min_sep
    )[0, 0]
    is_peak = img == pooled
    neg_inf = torch.tensor(float("-inf"), dtype=img.dtype)
    flat = torch.where(is_peak, img, neg_inf).flatten()
    order = torch.argsort(flat, descending=True)
    out: list[tuple[int, int]] = []
    taken = torch.zeros_like(img, dtype=torch.bool)
    n_cols = img.shape[1]
    for idx in order.tolist():
        if len(out) == n_peaks or flat[idx] == float("-inf"):
            break
        r, c = divmod(idx, n_cols)
        r0, c0 = max(0, r - min_sep), max(0, c - min_sep)
        if taken[r0 : r + min_sep + 1, c0 : c + min_sep + 1].any():
            continue
        taken[r, c] = True
        out.append((r, c))
    return torch.tensor(out, dtype=torch.float32)


def seed_positions(
    model, norm_xy: torch.Tensor, floor_frac: float = 0.2
) -> None:
    """Copy peak positions (normalized tangent-plane coords, (P, 2)) into the
    model's splat means. The last floor_frac of splats stay uniform-random as a
    low-flux floor for undetected sources."""
    with torch.no_grad():
        n = model.mu.shape[0]
        n_floor = int(n * floor_frac)
        n_seed = n - n_floor
        src = norm_xy[torch.randint(0, norm_xy.shape[0], (n_seed,))]
        jitter = 0.002 * torch.randn_like(src)
        model.mu[:n_seed].copy_(src + jitter)


def seed_calibrated(
    model, norm_xy: torch.Tensor, peak_values: torch.Tensor, loglam_norm: float = 0.0,
    n_scales: int = 4, base_scales: torch.Tensor | None = None,
) -> int:
    """Seed every splat on a detected peak, with flux calibrated so the model
    reproduces the measured peak amplitude at init.

    Splats are assigned round-robin to peaks, so when they outnumber the peaks
    each source gets several splats scattered by about one splat width and
    sharing its flux. That is the intended regime: a galaxy is not one Gaussian.

    Every splat must land on a peak. Parking the surplus at raw_flux = -30
    instead looks harmless but is fatal: softplus'(-30) is 1e-13, so those
    splats receive no gradient and, with no densification pass to revive them,
    stay dead for the whole run. Returns the number of distinct peaks used."""
    import math

    with torch.no_grad():
        dev = model.mu.device
        norm_xy = norm_xy.to(dev)
        peak_values = peak_values.to(dev)
        n = int(model.mu.shape[0])
        n_pk = int(norm_xy.shape[0])
        if n_pk == 0:
            raise ValueError("seed_calibrated needs at least one detected peak.")
        assign = torch.arange(n, device=dev) % n_pk
        model.mu.copy_(norm_xy[assign])

        if base_scales is not None:
            scale = torch.as_tensor(
                base_scales, dtype=model.log_scales.dtype, device=dev
            )
            if scale.ndim == 1:
                scale = scale[:, None].expand(-1, 2)
            if scale.shape != (n_pk, 2):
                raise ValueError("base_scales must have shape (n_peaks,) or (n_peaks, 2).")
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError("base_scales must be finite and positive.")
            model.log_scales.copy_(torch.log(scale[assign]))

        # Multi-scale: the r-th splat on a peak is 2^r times wider than the
        # first. Seeding every splat at one narrow width reproduces each source's
        # peak pixel but not its wings, which shows up twice: bright sources keep
        # a positive halo in the image residual, and a 3 arcsec fibre integrating
        # the too-compact model collects several times too little flux. One
        # defect, two symptoms.
        occurrence = torch.arange(n, device=dev) // n_pk
        widen = torch.log(
            2.0 ** (occurrence % n_scales).to(model.log_scales.dtype)
        )
        model.log_scales += widen[:, None]

        spare = torch.arange(n, device=dev) >= n_pk
        if bool(spare.any()):
            # jitter by each splat's own width, not a single global one
            sigma_j = torch.exp(model.log_scales).mean(dim=1)
            model.mu[spare] += (
                sigma_j[spare][:, None] * torch.randn_like(model.mu[spare])
            )

        # Calibrate against the geometry as placed, not against an assumed one
        # splat per peak: a jittered splat contributes less than its full
        # amplitude at the peak, so dividing the flux evenly would leave every
        # seeded source systematically faint.
        cov = model.covariances()
        d = norm_xy[assign] - model.mu
        q = torch.einsum("nk,nkl,nl->n", d, torch.linalg.inv(cov), d)
        ll = torch.tensor([float(loglam_norm)], device=dev)
        S0 = model.sed.eval(model.sed_coeffs, ll, model.sed_shift)[:, 0]
        unit = (
            torch.exp(-0.5 * q) * S0 / (2 * math.pi * torch.linalg.det(cov).sqrt())
        )
        total = torch.zeros(n_pk, dtype=unit.dtype, device=dev).scatter_add_(
            0, assign, unit
        )
        want = (peak_values.to(unit.dtype) / total.clamp_min(1e-30))[assign]
        model.raw_flux.copy_(_inv_softplus(want.clamp_min(1e-9)))
    return n_pk


def _dedupe_multiband_seeds(
    pos: torch.Tensor, vals: torch.Tensor, scales: torch.Tensor,
    radius_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    if not math.isfinite(radius_pixels) or radius_pixels <= 0.0:
        raise ValueError("dedupe_radius_pixels must be finite and positive.")
    radii = scales * float(radius_pixels)
    cell = float(radii.max().item())
    order = torch.argsort(vals, descending=True)
    occupied: dict[tuple[int, int], list[int]] = {}
    keep_idx = []
    for idx in order.tolist():
        x_cell = int(torch.floor(pos[idx, 0] / cell).item())
        y_cell = int(torch.floor(pos[idx, 1] / cell).item())
        matched = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for kept_idx in occupied.get((x_cell + dx, y_cell + dy), ()):
                    distance = torch.linalg.vector_norm(pos[idx] - pos[kept_idx])
                    match_radius = torch.maximum(radii[idx], radii[kept_idx])
                    if bool(distance <= match_radius):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if not matched:
            keep_idx.append(idx)
            occupied.setdefault((x_cell, y_cell), []).append(idx)
    keep = torch.tensor(keep_idx, dtype=torch.long)
    return pos[keep], vals[keep], scales[keep]


def seed_from_store(
    model, store, prefer_band: str | None = None, n_scales: int = 4,
    exclude_exposure_ids=None,
    dedupe_radius_pixels: float = 3.0,
) -> int:
    """Seed splat positions from source-image peaks across ALL exposures of the
    preferred band (every band when prefer_band is None), mapping pixel coords
    through each exposure's WCS into the model's normalized tangent-plane frame.

    Seeding from a single exposure only covers that exposure's footprint. On a
    single-field scene that is the whole tile, but on the 84-arcmin COSMOS tile
    one frame is ~1.5% of the field: every splat lands in one corner and
    training can never migrate them tens of arcminutes to the data they should
    explain. Returns the number of exposures used.
    """
    excluded = {int(e) for e in (exclude_exposure_ids or ())}
    if prefer_band is None:
        exp_ids = [
            int(e) for e in store.image_band_by_exposure if int(e) not in excluded
        ]
    else:
        exp_ids = [
            int(e)
            for e, b in store.image_band_by_exposure.items()
            if str(b) == prefer_band and int(e) not in excluded
        ] or [int(e) for e in store.image_band_by_exposure if int(e) not in excluded]
    if not exp_ids:
        raise ValueError(
            "seed_from_store: every image exposure is excluded; cannot seed splats."
        )
    n_splats = int(model.mu.shape[0])
    per_exp = max(32, -(-n_splats // len(exp_ids)))

    def _seed_one_exposure(exp_id: int):
        img = torch.as_tensor(store.image_source_by_exposure[exp_id]).float()
        pk = peaks_from_image(img, n_peaks=per_exp, min_sep=3)
        if pk.shape[0] == 0:
            return None
        exposure = store.exposures[exp_id]
        px = pk[:, 1].to(torch.float64).unsqueeze(1).numpy()
        py = pk[:, 0].to(torch.float64).unsqueeze(1).numpy()

        pos = _normalized_xy(exposure, px, py)
        x_next = _normalized_xy(exposure, px + 1.0, py)
        y_next = _normalized_xy(exposure, px, py + 1.0)
        pixel_scale = torch.sqrt(
            torch.linalg.vector_norm(x_next - pos, dim=1)
            * torch.linalg.vector_norm(y_next - pos, dim=1)
        )
        vals = img[pk[:, 0].long(), pk[:, 1].long()]
        return pos, vals, pixel_scale

    # Each exposure detects its own peaks from its own image and consumes no
    # RNG, so the tasks are independent and `map` returns them in exposure
    # order: the concatenation below is the same one the serial loop built.
    with ThreadPoolExecutor(max_workers=startup_worker_count(len(exp_ids))) as pool:
        seeds = [seed for seed in pool.map(_seed_one_exposure, exp_ids) if seed is not None]

    if not seeds:
        raise ValueError("No image peaks found in any exposure; cannot seed splats.")
    pos = torch.cat([s[0] for s in seeds])
    vals = torch.cat([s[1] for s in seeds])
    scales = torch.cat([s[2] for s in seeds])
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("Could not determine a finite positive pixel scale from the image WCS.")

    if prefer_band is None:
        pos, vals, scales = _dedupe_multiband_seeds(
            pos, vals, scales, dedupe_radius_pixels
        )
    else:
        # Overlapping exposures find the same source twice; seeding both copies at
        # full peak value doubles its flux and the brightest objects then dominate
        # the loss for hundreds of steps. Keep the brightest seed per ~pixel cell.
        cell = float(torch.median(scales).item())
        key = (
            (pos[:, 0] / cell).round().long() * 2_000_003
            + (pos[:, 1] / cell).round().long()
        )
        order = torch.argsort(vals, descending=True)
        key_sorted = key[order]
        seen: dict = {}
        keep_idx = []
        for i, k in zip(order.tolist(), key_sorted.tolist()):
            if k not in seen:
                seen[k] = True
                keep_idx.append(i)
        keep = torch.tensor(keep_idx, dtype=torch.long)
        pos, vals = pos[keep], vals[keep]
        scales = scales[keep]

    if pos.shape[0] > n_splats:
        top = torch.topk(vals, n_splats).indices
        pos, vals = pos[top], vals[top]
        scales = scales[top]
    seed_calibrated(model, pos, vals, base_scales=scales, n_scales=n_scales)
    return len(exp_ids)


def _catalog_pixel_xy(exposure, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    import math
    import numpy as np

    obs_wcs = exposure.obs_wcs
    raw_wcs = getattr(obs_wcs, "wcs", None)
    if raw_wcs is None or not callable(getattr(raw_wcs, "all_world2pix", None)):
        raise ValueError("Catalog seeding requires an invertible observation WCS.")
    ra_for_inverse = float(ra_deg)
    dec_for_inverse = float(dec_deg)
    dx_arcsec = float(getattr(obs_wcs, "dx_arcsec", 0.0) or 0.0)
    dy_arcsec = float(getattr(obs_wcs, "dy_arcsec", 0.0) or 0.0)
    if dx_arcsec != 0.0 or dy_arcsec != 0.0:
        cos_dec = max(math.cos(math.radians(dec_for_inverse)), 1.0e-12)
        ra_for_inverse -= dx_arcsec / (3600.0 * cos_dec)
        dec_for_inverse -= dy_arcsec / 3600.0
    x_pixel, y_pixel = raw_wcs.all_world2pix(
        np.asarray([ra_for_inverse]), np.asarray([dec_for_inverse]), 0
    )
    return float(np.asarray(x_pixel)[0]), float(np.asarray(y_pixel)[0])


def seed_from_catalog(
    model, store, catalog_path, exclude_exposure_ids=None,
) -> int:
    """Seed splats from catalog RA/Dec positions covered by image exposures."""
    import math
    import numpy as np

    path = Path(catalog_path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or ())
        missing = {"ra", "dec"} - fieldnames
        if missing:
            raise ValueError(
                f"Seed catalog {path} is missing required columns: {sorted(missing)}."
            )
        has_flux = "flux" in fieldnames
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                ra_deg = float(row["ra"])
                dec_deg = float(row["dec"])
                flux = float(row["flux"]) if has_flux else None
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Seed catalog {path} has invalid numeric data at CSV row {line_number}."
                ) from exc
            values = (ra_deg, dec_deg) if flux is None else (ra_deg, dec_deg, flux)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Seed catalog {path} has non-finite data at CSV row {line_number}."
                )
            rows.append((line_number, ra_deg, dec_deg, flux))
    if not rows:
        raise ValueError(f"Seed catalog {path} contains no data rows.")

    excluded = {int(e) for e in (exclude_exposure_ids or ())}
    exp_ids = [
        int(e) for e in store.image_band_by_exposure if int(e) not in excluded
    ]
    if not exp_ids:
        raise ValueError(
            "seed_from_catalog: every image exposure is excluded; cannot seed splats."
        )

    positions = []
    peak_values = []
    pixel_scales = []
    outside_rows = []
    for line_number, ra_deg, dec_deg, flux in rows:
        covering = []
        for exp_id in exp_ids:
            image = np.asarray(store.image_source_by_exposure[exp_id])
            if image.ndim != 2:
                raise ValueError(f"Image exposure {exp_id} source must be 2D.")
            x_pixel, y_pixel = _catalog_pixel_xy(
                store.exposures[exp_id], ra_deg, dec_deg
            )
            height, width = image.shape
            if (
                math.isfinite(x_pixel) and math.isfinite(y_pixel)
                and -0.5 <= x_pixel < width - 0.5
                and -0.5 <= y_pixel < height - 0.5
            ):
                covering.append((exp_id, x_pixel, y_pixel, image))
        if not covering:
            outside_rows.append(line_number)
            continue

        first_exp_id, x_pixel, y_pixel, _ = covering[0]
        exposure = store.exposures[first_exp_id]
        px = np.asarray([[x_pixel]], dtype=np.float64)
        py = np.asarray([[y_pixel]], dtype=np.float64)
        pos = _normalized_xy(exposure, px, py)
        x_next = _normalized_xy(exposure, px + 1.0, py)
        y_next = _normalized_xy(exposure, px, py + 1.0)
        pixel_scale = torch.sqrt(
            torch.linalg.vector_norm(x_next - pos, dim=1)
            * torch.linalg.vector_norm(y_next - pos, dim=1)
        )
        positions.append(pos)
        pixel_scales.append(pixel_scale)
        if flux is None:
            samples = []
            for _, sample_x, sample_y, image in covering:
                col = int(math.floor(sample_x + 0.5))
                row = int(math.floor(sample_y + 0.5))
                samples.append(float(image[row, col]))
            peak_values.append(float(np.median(np.asarray(samples, dtype=np.float64))))
        else:
            peak_values.append(flux)

    if outside_rows:
        raise ValueError(
            f"Seed catalog positions outside every non-excluded image footprint at "
            f"CSV rows {outside_rows}."
        )

    pos = torch.cat(positions)
    vals = torch.tensor(peak_values, dtype=torch.float32)
    scales = torch.cat(pixel_scales)
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("Could not determine a finite positive pixel scale from the image WCS.")
    if not bool(torch.isfinite(vals).all()):
        raise ValueError("Catalog seed fluxes must be finite.")

    n_splats = int(model.mu.shape[0])
    if pos.shape[0] > n_splats:
        top = torch.topk(vals, n_splats).indices
        pos, vals = pos[top], vals[top]
        scales = scales[top]
    seed_calibrated(model, pos, vals, base_scales=scales)
    return len(rows)
