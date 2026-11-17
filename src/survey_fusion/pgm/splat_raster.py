"""Patch-local rasterization of a SplatField onto a pixel grid.

Dense evaluation costs O(N_pixels * N_splats), which forces training on a small
random subset of pixels: with 80k splats on a 1024^2 field only ~1% of pixels
can be visited per step, so each pixel receives a handful of gradient updates
over an entire run and the fit stays mushy.

A Gaussian is numerically zero beyond ~3 sigma, so each splat only needs the
pixels inside a small patch around its centre. Cost becomes
O(N_splats * patch_area), independent of image size, which makes it affordable
to render - and differentiate - the *whole* image at every step. This is the
locality trick behind 3DGS/gsplat tile rasterization, written in portable
PyTorch so it runs on CUDA, MPS and CPU.

Patch placement uses the rounded splat centre and is therefore piecewise
constant in `mu`; the rendered values remain differentiable with respect to
mu, covariance and flux, exactly as in gsplat's tile assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from survey_fusion.utils.perf_env import (
    PAIR_CHUNK,
    activation_checkpointing_disabled,
    device_index_fill_enabled,
    env_chunk,
    exact_pair_pruning_enabled,
    fused_pair_index_enabled,
    fused_raster_pairs_enabled,
)


def patch_size_px(max_sigma_px: float, n_sigma: float = 3.0) -> int:
    """Odd patch width covering +-n_sigma of the widest splat."""
    return int(2 * math.ceil(n_sigma * float(max_sigma_px)) + 1)


def band_amplitudes(
    field,
    bandpass_ll: torch.Tensor,
    bandpass_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-splat band amplitude: total flux times band-averaged SED. (J,)

    SEDDictionary.eval bounds its own (n_templates, n_splats, n_bandpass)
    intermediate internally, so no chunking is needed here.
    """
    if bandpass_weights is None:
        bandpass_weights = torch.ones_like(bandpass_ll)
    w = bandpass_weights / bandpass_weights.sum()
    sed = field.sed.eval(field.sed_coeffs, bandpass_ll, field.sed_shift)
    return field.fluxes() * (sed * w[None, :]).sum(-1)


def rasterize_band(
    field,
    height: int,
    width: int,
    psf_sigma: float,
    amps: torch.Tensor,
    patch_px: int,
    splat_chunk: int = 16384,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Render the splat component of one band onto a (height, width) grid.

    Pixel (row, col) sits at normalized coordinates
    x = 2*col/(width-1) - 1, y = 2*row/(height-1) - 1, matching the convention
    used by the dense analytic renderer.
    """
    device, dtype = field.mu.device, field.mu.dtype
    cov = field.covariances() + (float(psf_sigma) ** 2) * torch.eye(
        2, device=device, dtype=dtype
    )
    a, b = cov[:, 0, 0], cov[:, 0, 1]
    c, d = cov[:, 1, 0], cov[:, 1, 1]
    det = (a * d - b * c).clamp_min(1e-30)
    norm = 1.0 / (2 * math.pi * det.sqrt())

    P = int(patch_px)
    radius = P // 2
    offs = torch.arange(P, device=device)
    sx, sy = 2.0 / (width - 1), 2.0 / (height - 1)
    col_c = (field.mu[:, 0].detach() + 1.0) / sx
    row_c = (field.mu[:, 1].detach() + 1.0) / sy

    img = torch.zeros(height * width, device=device, dtype=dtype)
    n_splats = field.mu.shape[0]

    def _chunk(mu_c, a_c, b_c, c_c, d_c, det_c, norm_c, amp_c, cols, rows, mask):
        x_pix = cols.to(dtype) * sx - 1.0
        y_pix = rows.to(dtype) * sy - 1.0
        dx = (x_pix - mu_c[:, 0:1])[:, None, :]
        dy = (y_pix - mu_c[:, 1:2])[:, :, None]
        q = (d_c[:, None, None] * dx * dx
             - (b_c + c_c)[:, None, None] * dx * dy
             + a_c[:, None, None] * dy * dy) / det_c[:, None, None]
        val = (norm_c * amp_c)[:, None, None] * torch.exp(-0.5 * q)
        return val * mask

    use_ckpt = torch.is_grad_enabled() and checkpoint_chunks
    for start in range(0, n_splats, splat_chunk):
        end = min(start + splat_chunk, n_splats)
        cols = col_c[start:end].round().long()[:, None] - radius + offs[None, :]
        rows = row_c[start:end].round().long()[:, None] - radius + offs[None, :]
        mask = ((rows >= 0) & (rows < height))[:, :, None] & (
            (cols >= 0) & (cols < width))[:, None, :]
        args = (field.mu[start:end], a[start:end], b[start:end], c[start:end],
                d[start:end], det[start:end], norm[start:end], amps[start:end],
                cols, rows, mask)
        # Recomputing the chunk in backward costs one extra forward (which is
        # ~20x cheaper than the stored-activation backward) and drops peak
        # memory from O(all splats x patch^2) to O(one chunk).
        val = checkpoint(_chunk, *args, use_reentrant=False) if use_ckpt else _chunk(*args)
        idx = rows.clamp(0, height - 1)[:, :, None] * width + cols.clamp(0, width - 1)[:, None, :]
        img = img.index_add(0, idx.reshape(-1), val.reshape(-1))
    return img.reshape(height, width)


def render_band_image(
    field,
    height: int,
    width: int,
    psf_sigma: float,
    bandpass_ll: torch.Tensor,
    bandpass_weights: torch.Tensor | None = None,
    patch_px: int = 21,
    splat_chunk: int = 16384,
) -> torch.Tensor:
    """Full band image: patch-rasterized splats plus the smooth background."""
    amps = band_amplitudes(field, bandpass_ll, bandpass_weights)
    img = rasterize_band(
        field, height, width, psf_sigma, amps, patch_px, splat_chunk
    )
    device, dtype = field.mu.device, field.mu.dtype
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device, dtype=dtype),
        torch.linspace(-1, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return img


# Pair-support truncation width, in sigma of the PSF-convolved splat.
# 8.0 discards pair weight exp(-32) ~ 1.3e-14, below fp32 resolution.
DEFAULT_PAIR_N_SIGMA = 8.0

def _build_pair_index_eager(
    mu: torch.Tensor,
    radius: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    duplication_budget: int = 32_000_000,
    max_tiles_per_axis: int = 2048,
    radius_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair every query point with the splats whose +-radius box can reach it.

    Splats are binned into square tiles (each splat is listed in every tile its
    bounding box overlaps, as in 3DGS tile assignment); each point then reads
    only its own tile's list. Returns (pair_point, pair_splat) index tensors of
    equal length M, where M is O(n_points * splats_per_tile) instead of the
    dense n_points * n_splats.

    ``radius`` bounds the reach along x and ``radius_y`` along y, defaulting to
    ``radius`` for a square box. A rectangular bound matters because what the
    caller keeps is an anisotropic ellipse, whose bounding box is generally much
    smaller than the square built from its larger semi-axis.

    All inputs are treated as constants: this decides WHICH pairs are evaluated,
    not their values, exactly like patch placement in the rasterizers above.

    The tile width starts at twice the median radius and doubles until the
    duplicated splat list fits `duplication_budget`; wide splats land in many
    tiles, which costs index memory, not evaluation work.
    """
    device = mu.device
    n = int(x.numel())
    if n == 0 or mu.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    mu = mu.detach()
    radius = radius.detach()
    radius_y = radius if radius_y is None else radius_y.detach()

    # One host transfer for all five scalars rather than five separate .item()
    # stalls. The reduction kernels are unchanged, so each value is bitwise what
    # the separate calls produced.
    xmin, xmax, ymin, ymax, radius_median = torch.stack(
        (
            x.min(),
            x.max(),
            y.min(),
            y.max(),
            radius.median().to(dtype=x.dtype),
        )
    ).tolist()
    extent = max(xmax - xmin, ymax - ymin, 1e-12)

    keep = (
        (mu[:, 0] + radius >= xmin) & (mu[:, 0] - radius <= xmax)
        & (mu[:, 1] + radius_y >= ymin) & (mu[:, 1] - radius_y <= ymax)
    )
    sidx = keep.nonzero(as_tuple=True)[0]
    n_kept = int(sidx.numel())
    if n_kept == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    mk = mu[sidx]
    rk = radius[sidx]
    rky = radius_y[sidx]

    tile = max(2.0 * radius_median, extent / max_tiles_per_axis, 1e-9)
    while True:
        ntx = int((xmax - xmin) / tile) + 1
        nty = int((ymax - ymin) / tile) + 1
        tx0 = ((mk[:, 0] - rk - xmin) / tile).floor().clamp(0, ntx - 1).long()
        tx1 = ((mk[:, 0] + rk - xmin) / tile).floor().clamp(0, ntx - 1).long()
        ty0 = ((mk[:, 1] - rky - ymin) / tile).floor().clamp(0, nty - 1).long()
        ty1 = ((mk[:, 1] + rky - ymin) / tile).floor().clamp(0, nty - 1).long()
        cx = tx1 - tx0 + 1
        cy = ty1 - ty0 + 1
        counts = cx * cy
        dup = int(counts.sum())
        if dup <= duplication_budget or ntx * nty <= 4:
            break
        tile *= 2.0

    # A splat spans ~2.5 tiles at production shape, and repeat_interleave gives
    # one thread per *input* row, so 10M threads each write 2.5 slots while the
    # machine idles. Resolving the owning splat of every slot by binary search
    # over the run ends instead gives one thread per slot. Integer comparisons
    # on integer offsets, so every slot resolves to the same splat.
    ends = counts.cumsum(0)
    slot = torch.arange(dup, device=device)
    rep = torch.searchsorted(ends, slot, right=True)
    within = (slot - (ends - counts)[rep]).to(torch.int32)
    # Fold the row term into a per-splat origin before gathering, so the tile id
    # needs two gathers over `dup` rather than three, and hold both tables in
    # int32 so each gather moves half the bytes. Every quantity here is bounded
    # by ntx*nty <= (max_tiles_per_axis + 1)^2, far inside int32, and the
    # arithmetic is exact integer arithmetic, so the ids are unchanged.
    tile_origin = (ty0 * ntx + tx0).to(torch.int32)
    cx = cx.to(torch.int32)
    # tid < ntx*nty <= max_tiles_per_axis^2, so int32 holds every tile id. The
    # sort moves half as many key bytes and returns the same permutation.
    #
    # stable=True is not a preference, it is the property the operator rests on.
    # Entries of one tile are emitted in ascending splat order and carry
    # distinct splat ids, so a stable sort leaves every tile's list ascending;
    # since a query point reads exactly one tile and the exact prune compacts
    # with nonzero, the retained pairs come out ordered (point, splat) whatever
    # tile width or support radius produced them. That is what makes the pair
    # ordering a function of the retained SET alone, and therefore what makes
    # tightening the candidate box bitwise-neutral downstream. Measured free
    # here (0.962 ms vs 0.961 ms unstable at dup = 2.6e7).
    cx_rep = cx[rep]
    tid = tile_origin[rep] + (within // cx_rep) * ntx + within % cx_rep
    # A radix sort costs one pass per byte of key, so narrow the key to the
    # width the tile grid actually needs. Same integer values, same stable
    # order, half the passes whenever the grid fits 15 bits.
    key_dtype = torch.int16 if ntx * nty <= 32767 else torch.int32
    tid_sorted, order = torch.sort(tid.to(key_dtype), stable=True)
    splat_sorted = sidx[rep[order]]
    # The sort already produced the ordered tile ids, so the per-tile run
    # boundaries are a binary search rather than a 28M-element atomic histogram
    # followed by a prefix sum. Integer counts either way, so identical.
    tile_offsets = torch.searchsorted(
        tid_sorted,
        torch.arange(ntx * nty + 1, device=device, dtype=tid_sorted.dtype),
    )
    tile_counts = tile_offsets[1:] - tile_offsets[:-1]
    tile_offsets = tile_offsets[:-1]

    ptid = (
        ((y - ymin) / tile).floor().clamp(0, nty - 1).long() * ntx
        + ((x - xmin) / tile).floor().clamp(0, ntx - 1).long()
    )
    cnt_p = tile_counts[ptid]
    # The per-point run ends are needed anyway for `base` below, and reading
    # the last one gives the pair census. repeat_interleave would resolve that
    # same length on the host and then stall a second time inside itself, so
    # resolve the owning point of every slot the way the splat side does:
    # by binary search over the run ends. Integer comparisons on integer
    # offsets, so slot s lands on the point whose run contains it, which is
    # exactly what repeat_interleave emits.
    point_ends = cnt_p.cumsum(0)
    total = int(point_ends[-1])
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    slot = torch.arange(total, device=device)
    pair_point = torch.searchsorted(point_ends, slot, right=True)
    # Each point's slice of `splat_sorted` starts at its tile offset minus the
    # point's own base in the pair array, so folding the two per-point terms
    # together leaves one gather over the pair census instead of three.
    base = tile_offsets[ptid] - (point_ends - cnt_p)
    pair_splat = splat_sorted[base[pair_point] + slot]
    return pair_point, pair_splat


def build_pair_index(
    mu: torch.Tensor,
    radius: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    duplication_budget: int = 32_000_000,
    max_tiles_per_axis: int = 2048,
    radius_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the reference ordered tile pairs, optionally fusing exact stages."""
    tensors = (mu, radius, x, y) + (() if radius_y is None else (radius_y,))
    if (
        fused_pair_index_enabled()
        and mu.is_cuda
        and mu.shape[0] < 2**29
        and max_tiles_per_axis <= 2048
        and duplication_budget < 2**31
        and all(t.device == mu.device and t.dtype == torch.float32 for t in tensors)
    ):
        from .pair_index_kernels import build_pair_index as fused_build

        return fused_build(mu, radius, x, y, duplication_budget, max_tiles_per_axis, radius_y)
    return _build_pair_index_eager(mu, radius, x, y, duplication_budget, max_tiles_per_axis, radius_y)


@dataclass(frozen=True)
class PackedPointIndex:
    pair_point: torch.Tensor
    active_splat: torch.Tensor
    pair_active_splat: torch.Tensor
    candidate_pairs: int | None = None
    retained_pairs: int | None = None


@dataclass(frozen=True)
class PointPSFMixture:
    """Validated per-query Gaussian mixture in model-normalized coordinates.

    ``covariances`` is either ``(n_points, K)`` for isotropic variances or
    ``(n_points, K, 3)`` containing ``[vxx, vxy, vyy]``.
    """

    weights: torch.Tensor
    covariances: torch.Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 2:
            raise ValueError("PSF mixture weights must have shape (n_points, K).")
        n_points, components = self.weights.shape
        if components < 1:
            raise ValueError("PSF mixture must contain at least one component.")
        if self.covariances.shape not in (
            (n_points, components), (n_points, components, 3),
        ):
            raise ValueError(
                "PSF mixture covariances must have shape (n_points, K) or "
                "(n_points, K, 3)."
            )
        if self.weights.device != self.covariances.device:
            raise ValueError("PSF mixture weights and covariances must share a device.")
        if self.weights.dtype != self.covariances.dtype:
            raise ValueError("PSF mixture weights and covariances must share a dtype.")
        if not bool(torch.isfinite(self.weights).all()) or not bool(torch.isfinite(self.covariances).all()):
            raise ValueError("PSF mixture contains non-finite values.")
        if bool(torch.any(self.weights <= 0.0)):
            raise ValueError("PSF mixture weights must be positive.")
        if not bool(torch.allclose(
            self.weights.sum(dim=1), torch.ones(n_points, device=self.weights.device, dtype=self.weights.dtype),
            rtol=0.0, atol=1e-5,
        )):
            raise ValueError("PSF mixture weights must sum to one per point.")
        if self.isotropic:
            if bool(torch.any(self.covariances <= 0.0)):
                raise ValueError("Isotropic PSF mixture variances must be positive.")
        else:
            vxx, vxy, vyy = self.covariances.unbind(dim=-1)
            if bool(torch.any(vxx <= 0.0)) or bool(torch.any(vyy <= 0.0)):
                raise ValueError("PSF mixture covariance diagonals must be positive.")
            if bool(torch.any(vxx * vyy - vxy.square() <= 0.0)):
                raise ValueError("PSF mixture covariances must be SPD.")

    @property
    def isotropic(self) -> bool:
        return self.covariances.ndim == 2

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[1])


def _mixture_max_eigenvalue(mixture: PointPSFMixture) -> torch.Tensor:
    if mixture.isotropic:
        return mixture.covariances.amax(dim=1)
    vxx, vxy, vyy = mixture.covariances.unbind(dim=-1)
    eigen_max = 0.5 * (vxx + vyy) + torch.sqrt(
        (0.25 * (vxx - vyy).square() + vxy.square()).clamp_min(0.0)
    )
    return eigen_max.amax(dim=1)


def build_mixture_packed_point_index(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    n_sigma: float = DEFAULT_PAIR_N_SIGMA,
    point_group: torch.Tensor | None = None,
) -> PackedPointIndex:
    """Build one per-group conservative index shared by mixture components."""
    device = field.mu.device
    n = int(x.numel())
    if mixture.weights.shape[0] != n:
        raise ValueError("PSF mixture must contain one row per query point.")
    if point_group is None:
        point_group = torch.zeros(n, dtype=torch.long, device=device)
    else:
        point_group = point_group.to(device=device, dtype=torch.long)
        if point_group.shape != (n,):
            raise ValueError("point_group must contain one group index per query point.")
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return PackedPointIndex(empty, empty, empty, 0, 0)
    with torch.no_grad():
        point_variance_max = _mixture_max_eigenvalue(mixture)
        splat_variance_max = torch.exp(2.0 * field.log_scales.detach().amax(dim=1))
        pair_point_parts = []
        pair_splat_parts = []
        for group in torch.unique(point_group, sorted=True):
            point_index = torch.nonzero(point_group == group, as_tuple=True)[0]
            psf_variance_max = float(point_variance_max[point_index].max())
            radius = float(n_sigma) * torch.sqrt(
                splat_variance_max + psf_variance_max
            )
            group_pair_point, group_pair_splat = build_pair_index(
                field.mu, radius, x[point_index], y[point_index]
            )
            if group_pair_splat.numel() == 0:
                continue
            pair_point_parts.append(point_index[group_pair_point])
            pair_splat_parts.append(group_pair_splat)
        if not pair_splat_parts:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return PackedPointIndex(empty, empty, empty, 0, 0)
        pair_point = torch.cat(pair_point_parts)
        pair_splat = torch.cat(pair_splat_parts)
        if pair_splat.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return PackedPointIndex(empty, empty, empty, 0, 0)
        candidate_pairs = int(pair_splat.numel())
        membership = torch.zeros(int(field.mu.shape[0]), dtype=torch.bool, device=device)
        if device_index_fill_enabled() and membership.is_cuda:
            membership.index_fill_(0, pair_splat, True)
        else:
            membership[pair_splat] = True
        active_splat = membership.nonzero(as_tuple=True)[0]
        inverse = torch.zeros(int(field.mu.shape[0]), dtype=torch.long, device=device)
        inverse[active_splat] = torch.arange(active_splat.numel(), device=device)
        pair_active_splat = inverse[pair_splat]
    return PackedPointIndex(
        pair_point, active_splat, pair_active_splat, candidate_pairs, None
    )


@torch.no_grad()
def _prune_exact_pairs(
    pair_point: torch.Tensor,
    pair_splat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    point_variance,
    n_sigma: float,
    c: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fused_pair_index_enabled() and mu.is_cuda:
        from . import pair_index_kernels

        if pair_index_kernels.applies(pair_point, pair_splat, x, y, mu, a, b, d, point_variance, c):
            return pair_index_kernels.prune_pairs(
                pair_point, pair_splat, x, y, mu, a, b, d, point_variance, n_sigma, c
            )
    variance = (
        point_variance[pair_point]
        if torch.is_tensor(point_variance)
        else point_variance
    )
    a_pair = a[pair_splat] + variance
    b_pair = b[pair_splat]
    d_pair = d[pair_splat] + variance
    if c is None:
        covariance_product = b_pair.square()
        cross_covariance = 2.0 * b_pair
    else:
        c_pair = c[pair_splat]
        covariance_product = b_pair * c_pair
        cross_covariance = b_pair + c_pair
    determinant = (a_pair * d_pair - covariance_product).clamp_min(1e-30)
    # mu is (J, 2), so gathering one column reads every cache line it touches
    # and uses half of it; taking the row once and slicing its columns moves
    # the same values with half the fetches.
    centre = mu.index_select(0, pair_splat)
    dx = x[pair_point] - centre[:, 0]
    dy = y[pair_point] - centre[:, 1]
    q = (
        d_pair * dx.square()
        - cross_covariance * dx * dy
        + a_pair * dy.square()
    ) / determinant
    keep_pair = torch.nonzero(
        q <= float(n_sigma) ** 2, as_tuple=True
    )[0]
    return (
        pair_point.index_select(0, keep_pair),
        pair_splat.index_select(0, keep_pair),
    )


def build_packed_point_index(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    psf_sigma,
    n_sigma: float = DEFAULT_PAIR_N_SIGMA,
) -> PackedPointIndex:
    """Build one shared point-splat index for a mixed-band image batch."""
    device = field.mu.device
    if x.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return PackedPointIndex(empty, empty, empty, 0, 0)
    if torch.is_tensor(psf_sigma) and psf_sigma.numel() > 1:
        # Left on device: its only consumer is the radius below, where a 0-dim
        # tensor broadcasts to the identical float32 sum without a host stall.
        psf_variance_max = psf_sigma.detach().to(device=device).square().max()
    else:
        psf_variance_max = float(psf_sigma) ** 2
    exact_prune = exact_pair_pruning_enabled()
    with torch.no_grad():
        if exact_prune:
            # The pairs this index may drop are exactly the ones the Mahalanobis
            # prune below rejects, i.e. those outside the ellipse q <= n_sigma^2
            # of M = Sigma + s^2 I. That ellipse's bounding box has half-widths
            # n_sigma*sqrt(M_xx) = n_sigma*sqrt(a + s^2) along x and
            # n_sigma*sqrt(M_yy) = n_sigma*sqrt(d + s^2) along y. Neither
            # exceeds the isotropic n_sigma*sqrt(lambda_max(Sigma) + s^2), since
            # a and d are diagonal entries and so never exceed the larger
            # eigenvalue: the box still contains every pair the prune keeps, and
            # for an elongated splat it is far smaller. (a, b, d) is then handed
            # to the prune rather than recomputed over the census a second time.
            #
            # This tightening is only sound BECAUSE the exact prune follows it.
            # With SPLAT_EXACT_PAIR_PRUNE=0 the candidate set is itself the
            # operator's support, so that legacy mode keeps the isotropic box
            # unchanged rather than silently rendering a narrower one.
            a, b, d = field.covariance_components()
            radius = float(n_sigma) * torch.sqrt(a + psf_variance_max)
            radius_y = float(n_sigma) * torch.sqrt(d + psf_variance_max)
        else:
            a = b = d = None
            radius = float(n_sigma) * torch.sqrt(
                torch.exp(2.0 * field.log_scales.detach().amax(dim=1))
                + psf_variance_max
            )
            radius_y = None
        pair_point, pair_splat = build_pair_index(
            field.mu, radius, x, y, radius_y=radius_y
        )
        if pair_splat.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return PackedPointIndex(empty, empty, empty, 0, 0)
        candidate_pairs = int(pair_splat.numel())
        if exact_prune:
            if torch.is_tensor(psf_sigma) and psf_sigma.numel() > 1:
                point_variance = (
                    psf_sigma.detach()
                    .to(device=device, dtype=field.mu.dtype)
                    .reshape(-1)
                    .square()
                )
            else:
                point_variance = float(psf_sigma) ** 2
            pair_point, pair_splat = _prune_exact_pairs(
                pair_point,
                pair_splat,
                x,
                y,
                field.mu,
                a,
                b,
                d,
                point_variance,
                n_sigma,
            )
            if pair_splat.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                return PackedPointIndex(empty, empty, empty, candidate_pairs, 0)
        # Direct-address membership map: same (sorted-unique active set,
        # inverse mapping) as torch.unique(sorted=True, return_inverse=True)
        # without its O(P log P) sort — cost is O(P + J) scatter/gather over
        # a J-sized table instead.
        n_splats = int(field.mu.shape[0])
        membership = torch.zeros(n_splats, dtype=torch.bool, device=device)
        if device_index_fill_enabled() and membership.is_cuda:
            membership.index_fill_(0, pair_splat, True)
        else:
            membership[pair_splat] = True
        active_splat = membership.nonzero(as_tuple=True)[0]
        inverse = torch.zeros(n_splats, dtype=torch.long, device=device)
        inverse[active_splat] = torch.arange(
            active_splat.numel(), device=device
        )
        pair_active_splat = inverse[pair_splat]
    return PackedPointIndex(
        pair_point, active_splat, pair_active_splat,
        candidate_pairs, int(pair_splat.numel()),
    )


def packed_multiband_point_prediction(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    psf_sigma,
    point_band: torch.Tensor,
    amplitudes: torch.Tensor,
    index: PackedPointIndex,
    pair_chunk: int = PAIR_CHUNK,
    checkpoint_chunks: bool = True,
    validate_bands: bool = True,
    gathered_parameters: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Evaluate one mixed-band point batch using only its active splats.

    amplitudes has shape (n_active_splats, n_bands), and point_band maps every
    query point to its amplitude column.

    validate_bands=False skips the band-index bounds check (two host syncs);
    pass it only where the caller has already range-checked the band ids on
    the host, as `_compute_analytic_image_source_mu` does.
    """
    device, dtype = field.mu.device, field.mu.dtype
    n = int(x.numel())
    point_band = point_band.to(device=device, dtype=torch.long)
    if point_band.shape != (n,):
        raise ValueError("point_band must contain one band index per query point.")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != index.active_splat.numel():
        raise ValueError(
            "amplitudes must have shape (n_active_splats, n_bands)."
        )
    if validate_bands and point_band.numel() and (
        bool(torch.any(point_band < 0))
        or bool(torch.any(point_band >= amplitudes.shape[1]))
    ):
        raise ValueError("point_band contains an out-of-range band index.")

    out = torch.zeros(n, device=device, dtype=dtype)
    if index.pair_point.numel() == 0:
        return out + field.sed.templates().sum() * 0.0

    per_point_sigma = torch.is_tensor(psf_sigma) and psf_sigma.numel() > 1
    if per_point_sigma:
        point_variance = (
            psf_sigma.to(device=device, dtype=dtype).reshape(-1).square()
        )
        if point_variance.shape != (n,):
            raise ValueError("psf_sigma must be scalar or contain one value per point.")
    else:
        point_variance = float(psf_sigma) ** 2

    # active_splat comes from torch.unique(sorted=True): ascending unique,
    # so the sparse-gather backward can skip its coalesce sort.
    if gathered_parameters is None:
        a, b, d = field.covariance_components(
            index.active_splat, sorted_unique=True
        )
        position = field.positions(index.active_splat, sorted_unique=True)
    else:
        position, log_scales, theta, _, _, _ = gathered_parameters
        a, b, d = field.covariance_components_from_rows(log_scales, theta)
    # One contiguous row of per-splat scalars, so each pair chunk performs a
    # single row gather (and its backward a single row-wise scatter) instead
    # of six random-access gathers with atomic-scatter backwards. The cross
    # term b+c and product b*c are precomputed per splat: elementwise-then-
    # gather is bitwise identical to gather-then-elementwise.
    splat_rows = torch.stack(
        (a, d, 2.0 * b, b * b, position[:, 0], position[:, 1]), dim=1
    )

    def _pair_values(pp, active, rows, amp):
        r = rows[active]
        variance = point_variance[pp] if per_point_sigma else point_variance
        a_pair = r[:, 0] + variance
        d_pair = r[:, 1] + variance
        determinant = (a_pair * d_pair - r[:, 3]).clamp_min(1e-30)
        dx = x[pp] - r[:, 4]
        dy = y[pp] - r[:, 5]
        exponent = (
            d_pair * dx.square()
            - r[:, 2] * dx * dy
            + a_pair * dy.square()
        ) / determinant
        pair_amplitude = amp[active, point_band[pp]]
        return (
            pair_amplitude
            / (2.0 * math.pi * determinant.sqrt())
            * torch.exp(-0.5 * exponent)
        )

    use_checkpoint = (
        torch.is_grad_enabled()
        and checkpoint_chunks
        and not activation_checkpointing_disabled()
    )
    pair_chunk = env_chunk("SPLAT_PAIR_CHUNK", pair_chunk)
    use_fused_pairs = fused_raster_pairs_enabled() and not use_checkpoint
    for start in range(0, int(index.pair_point.numel()), int(pair_chunk)):
        pp = index.pair_point[start : start + int(pair_chunk)]
        active = index.pair_active_splat[start : start + int(pair_chunk)]
        args = (pp, active, splat_rows, amplitudes)
        if use_checkpoint:
            values = checkpoint(_pair_values, *args, use_reentrant=False)
        elif use_fused_pairs and splat_rows.is_cuda:
            from survey_fusion.pgm.raster_pair_kernels import applies, pair_values

            values = (
                pair_values(
                    pp, active, splat_rows, amplitudes, x, y, point_band,
                    point_variance,
                )
                if applies(
                    pp, active, splat_rows, amplitudes, x, y, point_band,
                    point_variance,
                )
                else _pair_values(*args)
            )
        else:
            values = _pair_values(*args)
        out = out.index_add(0, pp, values)
    return out


def packed_mixture_multiband_point_prediction(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    point_band: torch.Tensor,
    amplitudes: torch.Tensor,
    index: PackedPointIndex,
    n_sigma: float = DEFAULT_PAIR_N_SIGMA,
    pair_chunk: int = PAIR_CHUNK,
    checkpoint_chunks: bool = True,
    support_stats: dict[str, int] | None = None,
    validate_bands: bool = True,
    gathered_parameters: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Evaluate a packed image batch with a co-centered Gaussian PSF mixture.

    ``support_stats`` receives the number of component-specific pairs retained
    by the exact Mahalanobis prune.
    """
    device, dtype = field.mu.device, field.mu.dtype
    n = int(x.numel())
    point_band = point_band.to(device=device, dtype=torch.long)
    if mixture.weights.shape[0] != n:
        raise ValueError("PSF mixture must contain one row per query point.")
    if point_band.shape != (n,):
        raise ValueError("point_band must contain one band index per query point.")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != index.active_splat.numel():
        raise ValueError("amplitudes must have shape (n_active_splats, n_bands).")
    if validate_bands and point_band.numel() and (
        bool(torch.any(point_band < 0)) or bool(torch.any(point_band >= amplitudes.shape[1]))
    ):
        raise ValueError("point_band contains an out-of-range band index.")
    out = torch.zeros(n, device=device, dtype=dtype)
    if index.pair_point.numel() == 0:
        if support_stats is not None:
            support_stats["retained_pairs"] = 0
        return out + field.sed.templates().sum() * 0.0
    weights = mixture.weights.to(device=device, dtype=dtype)
    covariances = mixture.covariances.to(device=device, dtype=dtype)
    if gathered_parameters is None:
        a, b, d = field.covariance_components(
            index.active_splat, sorted_unique=True
        )
        position = field.positions(index.active_splat, sorted_unique=True)
    else:
        position, log_scales, theta, _, _, _ = gathered_parameters
        a, b, d = field.covariance_components_from_rows(log_scales, theta)
    splat_rows = torch.stack((a, d, 2.0 * b, b * b, position[:, 0], position[:, 1]), dim=1)
    threshold = float(n_sigma) ** 2

    @torch.no_grad()
    def _component_keep(pp, active, rows, component: int):
        row = rows.detach()[active]
        if mixture.isotropic:
            variance = covariances.detach()[pp, component]
            a_pair = row[:, 0] + variance
            d_pair = row[:, 1] + variance
            cross = row[:, 2]
            covariance_product = row[:, 3]
        else:
            component_covariance = covariances.detach()[pp, component]
            a_pair = row[:, 0] + component_covariance[:, 0]
            d_pair = row[:, 1] + component_covariance[:, 2]
            cross = row[:, 2] + 2.0 * component_covariance[:, 1]
            covariance_product = (0.5 * row[:, 2] + component_covariance[:, 1]).square()
        determinant = (a_pair * d_pair - covariance_product).clamp_min(1e-30)
        dx = x.detach()[pp] - row[:, 4]
        dy = y.detach()[pp] - row[:, 5]
        q = (d_pair * dx.square() - cross * dx * dy + a_pair * dy.square()) / determinant
        return torch.nonzero(q <= threshold, as_tuple=True)[0]

    def _pair_values(pp, active, rows, amp, component: int):
        row = rows[active]
        if mixture.isotropic:
            variance = covariances[pp, component]
            a_pair = row[:, 0] + variance
            d_pair = row[:, 1] + variance
            cross = row[:, 2]
            covariance_product = row[:, 3]
        else:
            component_covariance = covariances[pp, component]
            a_pair = row[:, 0] + component_covariance[:, 0]
            d_pair = row[:, 1] + component_covariance[:, 2]
            cross = row[:, 2] + 2.0 * component_covariance[:, 1]
            covariance_product = (0.5 * row[:, 2] + component_covariance[:, 1]).square()
        determinant = (a_pair * d_pair - covariance_product).clamp_min(1e-30)
        dx = x[pp] - row[:, 4]
        dy = y[pp] - row[:, 5]
        q = (d_pair * dx.square() - cross * dx * dy + a_pair * dy.square()) / determinant
        pair_amplitude = amp[active, point_band[pp]] * weights[pp, component]
        return pair_amplitude / (2.0 * math.pi * determinant.sqrt()) * torch.exp(-0.5 * q)

    use_checkpoint = (
        torch.is_grad_enabled() and checkpoint_chunks and not activation_checkpointing_disabled()
    )
    pair_chunk = env_chunk("SPLAT_PAIR_CHUNK", pair_chunk)
    retained_pairs = 0
    # Components are completed sequentially.  Each checkpoint encloses one
    # component/chunk pair evaluation; its values are immediately index-added
    # before the next component, so SPLAT_NO_CHECKPOINT retains no K-wide pair tensor.
    for component in range(mixture.n_components):
        for start in range(0, int(index.pair_point.numel()), int(pair_chunk)):
            pp = index.pair_point[start : start + int(pair_chunk)]
            active = index.pair_active_splat[start : start + int(pair_chunk)]
            keep = _component_keep(pp, active, splat_rows, component)
            retained_pairs += int(keep.numel())
            if keep.numel() == 0:
                continue
            pp = pp.index_select(0, keep)
            active = active.index_select(0, keep)
            args = (pp, active, splat_rows, amplitudes, component)
            values = (
                checkpoint(_pair_values, *args, use_reentrant=False)
                if use_checkpoint else _pair_values(*args)
            )
            out = out.index_add(0, pp, values)
    if support_stats is not None:
        support_stats["retained_pairs"] = retained_pairs
    return out


def tiled_point_prediction(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    psf_sigma,
    amps: torch.Tensor,
    n_sigma: float = DEFAULT_PAIR_N_SIGMA,
    pair_chunk: int = 8_000_000,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Band flux at scattered points, evaluating only splats within n_sigma.

    The dense path costs n_points * n_splats and is what made training a large
    field impossible; the whole-image rasterizers render every pixel of an
    exposure to use a few thousand of them. This evaluates exactly the sampled
    points against exactly the splats that can reach them.

    x, y: (n,) normalized tangent-plane coordinates. psf_sigma: scalar or (n,)
    per-point PSF stddev in the same coordinates (points from different
    exposures of one band carry their own seeing). amps: (J,) band amplitudes.
    Differentiable in mu, covariance and flux; the pair selection itself is
    piecewise constant, as tile assignment is in gsplat.
    """
    device, dtype = field.mu.device, field.mu.dtype
    n = int(x.numel())
    if n == 0:
        return torch.zeros(0, device=device, dtype=dtype)
    per_point_sigma = torch.is_tensor(psf_sigma) and psf_sigma.numel() > 1
    if per_point_sigma:
        s2 = (psf_sigma.to(device=device, dtype=dtype) ** 2).reshape(-1)
        s2max = float(s2.max())
    else:
        s2 = float(psf_sigma) ** 2
        s2max = s2

    cov = field.covariances()
    a, b = cov[:, 0, 0], cov[:, 0, 1]
    c, d = cov[:, 1, 0], cov[:, 1, 1]

    with torch.no_grad():
        lam_max = 0.5 * (a + d) + torch.sqrt(
            (0.25 * (a - d) ** 2 + b * c).clamp_min(0.0)
        )
        r = n_sigma * torch.sqrt(lam_max + s2max)
        pair_point, pair_splat = build_pair_index(field.mu, r, x, y)
        if exact_pair_pruning_enabled():
            pair_point, pair_splat = _prune_exact_pairs(
                pair_point,
                pair_splat,
                x,
                y,
                field.mu,
                a,
                b,
                d,
                s2,
                n_sigma,
                c=c,
            )

    out = torch.zeros(n, device=device, dtype=dtype)
    if pair_point.numel() == 0:
        return out

    def _pair_values(pp, ps):
        s2_p = s2[pp] if per_point_sigma else s2
        a_p = a[ps] + s2_p
        d_p = d[ps] + s2_p
        b_p, c_p = b[ps], c[ps]
        det = (a_p * d_p - b_p * c_p).clamp_min(1e-30)
        dxv = x[pp] - field.mu[ps, 0]
        dyv = y[pp] - field.mu[ps, 1]
        q = (d_p * dxv * dxv - (b_p + c_p) * dxv * dyv + a_p * dyv * dyv) / det
        return amps[ps] / (2 * math.pi * det.sqrt()) * torch.exp(-0.5 * q)

    use_ckpt = torch.is_grad_enabled() and checkpoint_chunks
    for start in range(0, int(pair_point.numel()), pair_chunk):
        pp = pair_point[start : start + pair_chunk]
        ps = pair_splat[start : start + pair_chunk]
        if use_ckpt:
            vals = checkpoint(_pair_values, pp, ps, use_reentrant=False)
        else:
            vals = _pair_values(pp, ps)
        out = out.index_add(0, pp, vals)
    return out


def tiled_mixture_point_prediction(
    field,
    x: torch.Tensor,
    y: torch.Tensor,
    mixture: PointPSFMixture,
    amps: torch.Tensor,
    n_sigma: float = DEFAULT_PAIR_N_SIGMA,
    pair_chunk: int = PAIR_CHUNK,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Single-exposure tiled prediction using one shared conservative index."""
    device, dtype = field.mu.device, field.mu.dtype
    n = int(x.numel())
    if mixture.weights.shape[0] != n:
        raise ValueError("PSF mixture must contain one row per query point.")
    if n == 0:
        return torch.zeros(0, device=device, dtype=dtype)
    cov = field.covariances()
    a, b = cov[:, 0, 0], cov[:, 0, 1]
    c, d = cov[:, 1, 0], cov[:, 1, 1]
    with torch.no_grad():
        splat_max = 0.5 * (a + d) + torch.sqrt(
            (0.25 * (a - d).square() + b * c).clamp_min(0.0)
        )
        psf_max = float(_mixture_max_eigenvalue(mixture).max())
        radius = float(n_sigma) * torch.sqrt(splat_max + psf_max)
        pair_point, pair_splat = build_pair_index(field.mu, radius, x, y)
    out = torch.zeros(n, device=device, dtype=dtype)
    if pair_point.numel() == 0:
        return out
    weights = mixture.weights.to(device=device, dtype=dtype)
    psf_cov = mixture.covariances.to(device=device, dtype=dtype)
    threshold = float(n_sigma) ** 2

    @torch.no_grad()
    def _component_keep(pp, ps, component: int):
        if mixture.isotropic:
            variance = psf_cov.detach()[pp, component]
            a_pair = a.detach()[ps] + variance
            d_pair = d.detach()[ps] + variance
            cross = b.detach()[ps] + c.detach()[ps]
            product = b.detach()[ps] * c.detach()[ps]
        else:
            component_covariance = psf_cov.detach()[pp, component]
            a_pair = a.detach()[ps] + component_covariance[:, 0]
            d_pair = d.detach()[ps] + component_covariance[:, 2]
            b_pair = b.detach()[ps] + component_covariance[:, 1]
            c_pair = c.detach()[ps] + component_covariance[:, 1]
            cross, product = b_pair + c_pair, b_pair * c_pair
        determinant = (a_pair * d_pair - product).clamp_min(1e-30)
        dx = x.detach()[pp] - field.mu.detach()[ps, 0]
        dy = y.detach()[pp] - field.mu.detach()[ps, 1]
        q = (d_pair * dx.square() - cross * dx * dy + a_pair * dy.square()) / determinant
        return torch.nonzero(q <= threshold, as_tuple=True)[0]

    def _pair_values(pp, ps, component: int):
        if mixture.isotropic:
            variance = psf_cov[pp, component]
            a_pair, d_pair = a[ps] + variance, d[ps] + variance
            cross = b[ps] + c[ps]
            product = b[ps] * c[ps]
        else:
            component_covariance = psf_cov[pp, component]
            a_pair = a[ps] + component_covariance[:, 0]
            d_pair = d[ps] + component_covariance[:, 2]
            b_pair = b[ps] + component_covariance[:, 1]
            c_pair = c[ps] + component_covariance[:, 1]
            cross, product = b_pair + c_pair, b_pair * c_pair
        determinant = (a_pair * d_pair - product).clamp_min(1e-30)
        dx = x[pp] - field.mu[ps, 0]
        dy = y[pp] - field.mu[ps, 1]
        q = (d_pair * dx.square() - cross * dx * dy + a_pair * dy.square()) / determinant
        return (
            amps[ps] * weights[pp, component]
            / (2.0 * math.pi * determinant.sqrt()) * torch.exp(-0.5 * q)
        )

    use_checkpoint = (
        torch.is_grad_enabled() and checkpoint_chunks and not activation_checkpointing_disabled()
    )
    pair_chunk = env_chunk("SPLAT_PAIR_CHUNK", pair_chunk)
    for component in range(mixture.n_components):
        for start in range(0, int(pair_point.numel()), int(pair_chunk)):
            pp = pair_point[start : start + int(pair_chunk)]
            ps = pair_splat[start : start + int(pair_chunk)]
            keep = _component_keep(pp, ps, component)
            if keep.numel() == 0:
                continue
            pp = pp.index_select(0, keep)
            ps = ps.index_select(0, keep)
            args = (pp, ps, component)
            values = checkpoint(_pair_values, *args, use_reentrant=False) if use_checkpoint else _pair_values(*args)
            out = out.index_add(0, pp, values)
    return out


def rasterize_band_on_grid(
    field,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    psf_sigma: float,
    amps: torch.Tensor,
    patch_px: int,
    splat_chunk: int = 16384,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Rasterize onto an arbitrary coordinate grid.

    x_norm, y_norm are (H, W) maps giving each pixel's normalized tangent-plane
    coordinate, as produced by the exposure WCS. Unlike `rasterize_band`, which
    assumes pixels lie on a uniform grid, this evaluates the Gaussian at the
    true coordinate of every pixel, so it is exact for any projection.

    Patch *placement* still uses a linear estimate of the pixel index from the
    coordinate maps; that only decides which pixels are touched and is not
    differentiated, exactly as tile assignment works in gsplat.
    """
    device, dtype = field.mu.device, field.mu.dtype
    height, width = x_norm.shape
    cov = field.covariances() + (float(psf_sigma) ** 2) * torch.eye(
        2, device=device, dtype=dtype
    )
    a, b = cov[:, 0, 0], cov[:, 0, 1]
    c, d = cov[:, 1, 0], cov[:, 1, 1]
    det = (a * d - b * c).clamp_min(1e-30)
    norm = 1.0 / (2 * math.pi * det.sqrt())

    # Affine inverse of the coordinate maps, for patch placement only.
    #
    # The map is a rotated tangent-plane projection, so x depends on BOTH row
    # and column. Inverting each axis independently (col from dx/dcol, row from
    # dy/drow) silently fails whenever the frame is rotated: on a COSMOS frame
    # rotated ~90 degrees, x varies 40x more per row than per column, every
    # patch lands outside the image, and the rendered band is exactly zero while
    # the dense renderer is fine. Solve the 2x2 system instead.
    # Named jx*/jy* rather than a/b/c/d: those are already the covariance
    # components above, and shadowing them replaces per-splat tensors with
    # scalars.
    x0 = x_norm[0, 0]
    y0 = y_norm[0, 0]
    jx_col = (x_norm[0, -1] - x0) / max(width - 1, 1)
    jx_row = (x_norm[-1, 0] - x0) / max(height - 1, 1)
    jy_col = (y_norm[0, -1] - y0) / max(width - 1, 1)
    jy_row = (y_norm[-1, 0] - y0) / max(height - 1, 1)
    jdet = jx_col * jy_row - jx_row * jy_col
    if bool(jdet.abs() < 1e-24):
        raise ValueError(
            "Coordinate maps are degenerate: the pixel-to-sky Jacobian is "
            f"singular (det={float(jdet):.3e}). Patch placement would be "
            "arbitrary and the rendered image silently empty."
        )

    P = int(patch_px)
    radius = P // 2
    offs = torch.arange(P, device=device)
    ux = field.mu[:, 0].detach() - x0
    uy = field.mu[:, 1].detach() - y0
    col_c = (jy_row * ux - jx_row * uy) / jdet
    row_c = (-jy_col * ux + jx_col * uy) / jdet

    def _patch_values(mu_c, amp_c, a_c, b_c, c_c, d_c, det_c, norm_c, rr, cc, keep):
        # true coordinates of the patch pixels: (n, P, P)
        gx = x_norm[rr[:, :, None], cc[:, None, :]]
        gy = y_norm[rr[:, :, None], cc[:, None, :]]
        dxp = gx - mu_c[:, 0][:, None, None]
        dyp = gy - mu_c[:, 1][:, None, None]
        q = (d_c[:, None, None] * dxp * dxp
             - (b_c + c_c)[:, None, None] * dxp * dyp
             + a_c[:, None, None] * dyp * dyp) / det_c[:, None, None]
        val = (norm_c * amp_c)[:, None, None] * torch.exp(-0.5 * q)
        return val * keep

    img = torch.zeros(height * width, device=device, dtype=dtype)
    for start in range(0, field.mu.shape[0], splat_chunk):
        end = min(start + splat_chunk, field.mu.shape[0])
        cols = col_c[start:end].round().long()[:, None] - radius + offs[None, :]
        rows = row_c[start:end].round().long()[:, None] - radius + offs[None, :]
        ok_c = (cols >= 0) & (cols < width)
        ok_r = (rows >= 0) & (rows < height)
        cc = cols.clamp(0, width - 1)
        rr = rows.clamp(0, height - 1)
        keep = ok_r[:, :, None] & ok_c[:, None, :]

        args = (field.mu[start:end], amps[start:end], a[start:end], b[start:end],
                c[start:end], d[start:end], det[start:end], norm[start:end],
                rr, cc, keep)
        # Recompute the patch instead of storing it. Each chunk holds six
        # (n, P, P) intermediates; at 200k splats over ten exposures those
        # retained activations alone run to tens of GiB, which is what filled
        # the GPU before the first optimizer step. rasterize_band already does
        # this; this path was simply missing it.
        if torch.is_grad_enabled() and checkpoint_chunks:
            val = checkpoint(_patch_values, *args, use_reentrant=False)
        else:
            val = _patch_values(*args)

        idx = rr[:, :, None] * width + cc[:, None, :]
        img = img.index_add(0, idx.reshape(-1), val.reshape(-1))
    return img.reshape(height, width)
