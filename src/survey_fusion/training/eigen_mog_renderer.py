"""Analytic image renderer for per-eigenimage signed Gaussian mixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..pgm.eigen_mog import EigenMixture
from ..pgm.eigen_mog_triton import eigen_mog_pairs
from ..pgm.eigen_mog_reuse import reuse_pairs
from ..pgm.splat_raster import PackedPointIndex, build_pair_index
from ..utils.perf_env import (
    PAIR_CHUNK, eigen_fast_context_enabled, eigen_fused_index_enabled, eigen_fused_predict_enabled,
    eigen_reuse_gather_enabled, eigen_shape_cache_enabled, env_chunk,
)


COMPUTE_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def eigen_mog_identity(config):
    path = Path(str(config.get("eigen_mog_sidecar", "")))
    if not path.is_file():
        raise ValueError("eigen_mog requires trainer.eigen_mog_sidecar")
    representation = EigenMixture.load(path)
    support = config.get("eigen_mog_support_radius")
    if support is None:
        raise ValueError("eigen_mog requires trainer.eigen_mog_support_radius")
    support = float(support)
    if not np.isfinite(support) or support <= 0:
        raise ValueError("eigen_mog_support_radius must be positive and finite")
    compute_dtype = str(config.get("eigen_mog_compute_dtype", "float64"))
    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError("trainer.eigen_mog_compute_dtype must be float32 or float64")
    identity = {
        **representation.identity(),
        "sidecar_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "support_radius": support,
        "compute_dtype": compute_dtype,
        "coefficient_position": "landing_pixel",
        "pixel_response": "included_in_native_samples",
    }
    return str(path), identity


class EigenMogRenderer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.identity = trainer.eigen_mog_config
        self.representation = EigenMixture.load(trainer.eigen_mog_sidecar_path)
        self.compute_dtype = COMPUTE_DTYPES[self.identity["compute_dtype"]]
        self.support_radius = float(self.identity["support_radius"])
        self.pair_chunk = int(getattr(trainer, "eigen_mog_pair_chunk", PAIR_CHUNK))
        if self.pair_chunk < 1:
            raise ValueError("eigen_mog_pair_chunk must be positive")
        self._exposure_to_row = {}
        self.last_stats = {}
        self.execution = getattr(trainer, "eigen_mog_execution", "reference")
        if self.execution not in {"reference", "cached", "grouped"}:
            raise ValueError("eigen_mog_execution must be reference, cached, or grouped")
        self._host_tables = None
        self._wcs_tables = None

    def _ensure_host_tables(self):
        if self._host_tables is not None:
            return self._host_tables
        size = max(self.trainer.exposures) + 1
        rows = np.full(size, -1, dtype=np.int64)
        widths = np.ones(size, dtype=np.int64)
        row0 = np.zeros(size, dtype=np.float64)
        col0 = np.zeros(size, dtype=np.float64)
        for eid, exposure in self.trainer.exposures.items():
            if exposure.label not in self.representation.by_label:
                continue
            row = self.sidecar_row(eid)
            rows[eid] = row
            widths[eid] = int(exposure.image_width)
            record = self.representation.records[row]
            row0[eid], col0[eid] = record["row0"], record["col0"]
        self._host_tables = dict(rows=rows, widths=widths, row0=row0, col0=col0)
        return self._host_tables

    def batch_metadata(self, exposure_ids):
        """Retain the original first-occurrence band ordering without a pixel loop."""
        if not hasattr(self, "_band_metadata"):
            codes = np.full(max(self.trainer.exposures) + 1, -1, dtype=np.int64)
            keys = []
            for eid, key in sorted(self.trainer.image_band_by_exposure.items()):
                if key not in keys:
                    keys.append(key)
                codes[eid] = keys.index(key)
            self._band_metadata = (codes, keys)
        codes, keys = self._band_metadata
        values = codes[np.asarray(exposure_ids)]
        if (values < 0).any():
            raise ValueError("Missing image band metadata")
        unique, first = np.unique(values, return_index=True)
        order = np.argsort(first)
        unique, first = unique[order], first[order]
        mapping = np.zeros(len(keys), dtype=np.int64)
        mapping[unique] = np.arange(len(unique))
        bands = [keys[i] for i in unique]
        throughput = {}
        for band, idx in zip(bands, first):
            cache = self.trainer._analytic_cache_for_exposure(int(exposure_ids[idx]))
            throughput[band] = (cache["band_lam"], cache["band_throughput"])
        return bands, throughput, mapping[values]

    def _lookup_wcs(self, exposure_ids, local_indices):
        host = self._ensure_host_tables()
        if self._wcs_tables is None:
            grids = np.ones((len(host["rows"]), 2), dtype=np.int64)
            bases = np.zeros(len(host["rows"]), dtype=np.int64)
            pieces = []
            offset = 0
            for eid in np.flatnonzero(host["rows"] >= 0):
                cache = self.trainer._analytic_cache_for_exposure(int(eid))
                if cache["psf_jacobian_path"] == "tiled":
                    grid = cache["psf_jacobian_tile_grid"]
                    matrices = cache["pixel_to_norm_jacobian_tiles"].to(torch.float64)
                else:
                    grid = (1, 1)
                    matrices = cache["pixel_to_norm_jacobian"].to(torch.float64)[None]
                grids[eid] = grid
                bases[eid] = offset
                pieces.append(matrices)
                offset += len(matrices)
            jac = torch.cat(pieces)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            if not bool(torch.isfinite(det).all()) or bool((det.abs() <= 1e-300).any()):
                raise ValueError("Singular pixel-to-normalized Jacobian")
            inverse = (
                torch.stack([jac[:, 1, 1], -jac[:, 0, 1], -jac[:, 1, 0], jac[:, 0, 0]], -1).reshape(-1, 2, 2)
                / det[:, None, None]
            )
            self._wcs_tables = dict(
                grids=grids,
                bases=bases,
                forward=jac,
                inverse=inverse,
                det=det.abs(),
                wing=self.support_radius * jac.abs().sum(-1),
            )
        table = self._wcs_tables
        widths = host["widths"][exposure_ids]
        # Heights are immutable too; cache their exposure-indexed array once.
        if "heights" not in host:
            host["heights"] = np.ones(len(host["rows"]), dtype=np.int64)
            for eid in np.flatnonzero(host["rows"] >= 0):
                host["heights"][eid] = self.trainer.exposures[int(eid)].image_height
        grid = table["grids"][exposure_ids]
        tx = np.floor((local_indices % widths + 0.5) * grid[:, 1] / widths).astype(np.int64)
        ty = np.floor((local_indices // widths + 0.5) * grid[:, 0] / host["heights"][exposure_ids]).astype(
            np.int64
        )
        tx = np.clip(tx, 0, grid[:, 1] - 1)
        ty = np.clip(ty, 0, grid[:, 0] - 1)
        index = torch.as_tensor(
            table["bases"][exposure_ids] + ty * grid[:, 1] + tx, device=self.trainer.device
        )
        return {name: table[name][index] for name in ["forward", "inverse", "det", "wing"]}

    def sidecar_row(self, exposure_id: int) -> int:
        row = self._exposure_to_row.get(int(exposure_id))
        if row is None:
            label = self.trainer.exposures[int(exposure_id)].label
            if label not in self.representation.by_label:
                raise ValueError(f"eigen_mog sidecar lacks exposure {label}")
            row = self.representation.by_label[label]
            self._exposure_to_row[int(exposure_id)] = row
        return row

    def prepare(self, exposure_ids, local_indices, jacobians):
        exposure_ids = np.asarray(exposure_ids, dtype=np.int64)
        local_indices = np.asarray(local_indices, dtype=np.int64)
        if exposure_ids.shape != local_indices.shape or (
            jacobians is not None and jacobians.shape != (len(exposure_ids), 2, 2)
        ):
            raise ValueError("Context inputs must align one row per pixel")
        fast = eigen_fast_context_enabled()
        if fast:
            host = self._ensure_host_tables()
            rows = host["rows"][exposure_ids]
            if (rows < 0).any():
                raise ValueError("Missing PSF exposure metadata")
            frame_row = host["row0"][exposure_ids] + local_indices // host["widths"][exposure_ids]
            frame_col = host["col0"][exposure_ids] + local_indices % host["widths"][exposure_ids]
        else:
            rows = np.empty(len(exposure_ids), dtype=np.int64)
            frame_row = np.empty(len(exposure_ids), dtype=np.float64)
            frame_col = np.empty(len(exposure_ids), dtype=np.float64)
        for exposure_id in [] if fast else np.unique(exposure_ids):
            where = np.flatnonzero(exposure_ids == exposure_id)
            exposure = self.trainer.exposures[int(exposure_id)]
            sidecar_row = self.sidecar_row(int(exposure_id))
            record = self.representation.records[sidecar_row]
            width = int(exposure.image_width)
            rows[where] = sidecar_row
            frame_col[where] = record["col0"] + local_indices[where] % width
            frame_row[where] = record["row0"] + local_indices[where] // width
        device = self.trainer.device
        exposure_row = torch.as_tensor(rows, device=device)
        table_inputs = (
            exposure_row,
            torch.as_tensor(frame_row, device=device, dtype=torch.float64),
            torch.as_tensor(frame_col, device=device, dtype=torch.float64),
        )
        if self.execution == "reference":
            tables = self.representation.pixel_tables(*table_inputs)
        else:
            tables = self.representation.cached_pixel_tables(*table_inputs, self.compute_dtype)
        static_wcs = self._lookup_wcs(exposure_ids, local_indices) if jacobians is None else None
        jacobian = (
            static_wcs["forward"]
            if static_wcs is not None
            else jacobians.to(device=device, dtype=torch.float64)
        )
        if static_wcs is not None:
            determinant, inverse = static_wcs["det"], static_wcs["inverse"]
        else:
            determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
            if not bool(torch.isfinite(determinant).all()) or bool((determinant.abs() <= 1e-300).any()):
                raise ValueError("Singular pixel-to-normalized Jacobian")
            inverse = (
                torch.stack(
                    [jacobian[:, 1, 1], -jacobian[:, 0, 1], -jacobian[:, 1, 0], jacobian[:, 0, 0]], -1
                ).reshape(-1, 2, 2)
                / determinant[:, None, None]
            )
        names = ("weights", "mx", "my", "sxx", "sxy", "syy")
        context = {name: value.to(self.compute_dtype).contiguous() for name, value in zip(names, tables)}
        context.update(
            exposure_row=exposure_row,
            inverse=inverse,
            forward=jacobian,
            det=determinant.abs(),
            wing=static_wcs["wing"]
            if static_wcs is not None
            else self.support_radius * jacobian.abs().sum(-1),
        )
        if static_wcs is not None:
            context.update(static_wcs)
        if self.execution == "grouped":
            # Group only exactly identical exposure/Jacobian tuples. No spatial averaging.
            group_rows = torch.cat((exposure_row[:, None], inverse.flatten(1)), 1)
            groups, point_group = torch.unique(group_rows, dim=0, return_inverse=True)
            context.update(point_group=point_group, n_groups=len(groups))
        return context

    def index(self, model, x, y, context, n_sigma, *, chunk=4_194_304):
        with torch.no_grad():
            covariance_xx, covariance_xy, covariance_yy = model.covariance_components()
            wing = context["wing"].amax(0)
            point_pair, splat_pair = build_pair_index(
                model.mu,
                n_sigma * covariance_xx.sqrt() + wing[0],
                x,
                y,
                radius_y=n_sigma * covariance_yy.sqrt() + wing[1],
            )
            candidates = int(len(point_pair))
            if candidates == 0:
                empty = torch.empty(0, dtype=torch.long, device=x.device)
                return PackedPointIndex(empty, empty, empty, 0, 0)
            radius = self.support_radius
            threshold = float(n_sigma) ** 2 * (1 + 1e-7)
            keep_parts = []
            basis = None
            if x.is_cuda and eigen_shape_cache_enabled():
                from ..pgm.eigen_mog_index import shape_basis

                basis = shape_basis(model)
            for start in range(0, candidates, chunk):
                point = point_pair[start : start + chunk]
                source = splat_pair[start : start + chunk]
                if x.is_cuda and eigen_fused_index_enabled():
                    from ..pgm.eigen_mog_index import ellipse_box_keep

                    keep_parts.append(
                        ellipse_box_keep(model, x, y, point, source, context, radius, threshold, basis=basis)
                    )
                    continue
                inverse = context["inverse"][point]
                sigma = model.log_scales[source].double().exp()
                angle = model.theta[source].double()
                cosine, sine = angle.cos(), angle.sin()
                fx0 = (inverse[:, 0, 0] * cosine + inverse[:, 0, 1] * sine) * sigma[:, 0]
                fy0 = (inverse[:, 1, 0] * cosine + inverse[:, 1, 1] * sine) * sigma[:, 0]
                fx1 = (-inverse[:, 0, 0] * sine + inverse[:, 0, 1] * cosine) * sigma[:, 1]
                fy1 = (-inverse[:, 1, 0] * sine + inverse[:, 1, 1] * cosine) * sigma[:, 1]
                cxx = fx0 * fx0 + fx1 * fx1
                cxy = fx0 * fy0 + fx1 * fy1
                cyy = fy0 * fy0 + fy1 * fy1
                forward = context["forward"][point]
                w00 = (cosine * forward[:, 0, 0] + sine * forward[:, 1, 0]) / sigma[:, 0]
                w01 = (cosine * forward[:, 0, 1] + sine * forward[:, 1, 1]) / sigma[:, 0]
                w10 = (-sine * forward[:, 0, 0] + cosine * forward[:, 1, 0]) / sigma[:, 1]
                w11 = (-sine * forward[:, 0, 1] + cosine * forward[:, 1, 1]) / sigma[:, 1]
                delta = torch.stack([x[point], y[point]], -1).double() - model.mu[source].double()
                dx = inverse[:, 0, 0] * delta[:, 0] + inverse[:, 0, 1] * delta[:, 1]
                dy = inverse[:, 1, 0] * delta[:, 0] + inverse[:, 1, 1] * delta[:, 1]
                inside = torch.maximum(dx.abs(), dy.abs()) <= radius
                distance = torch.full_like(dx, torch.inf)

                def squared(ux, uy):
                    return (w00 * ux + w01 * uy).square() + (w10 * ux + w11 * uy).square()

                for sign in (-1.0, 1.0):
                    ux = dx - sign * radius
                    uy = dy - (dy - cxy / cxx * ux).clamp(-radius, radius)
                    distance = torch.minimum(distance, squared(ux, uy))
                    uy = dy - sign * radius
                    ux = dx - (dx - cxy / cyy * uy).clamp(-radius, radius)
                    distance = torch.minimum(distance, squared(ux, uy))
                keep_parts.append(inside | (distance <= threshold) | ~torch.isfinite(distance))
            keep = torch.cat(keep_parts)
            point_pair, splat_pair = point_pair[keep], splat_pair[keep]
            membership = torch.zeros(len(model.mu), dtype=torch.bool, device=x.device)
            if eigen_fused_index_enabled():
                membership.index_fill_(0, splat_pair, True)
            else:
                membership[splat_pair] = True
            active = membership.nonzero(as_tuple=True)[0]
            inverse_map = torch.zeros(len(model.mu), dtype=torch.long, device=x.device)
            inverse_map.index_copy_(0, active, torch.arange(active.numel(), device=x.device))
        return PackedPointIndex(point_pair, active, inverse_map[splat_pair], candidates, int(len(point_pair)))

    def predict(
        self,
        model,
        x,
        y,
        bands,
        amplitudes,
        index,
        context,
        *,
        implementation="auto",
        gathered_parameters=None,
    ):
        if gathered_parameters is not None and eigen_reuse_gather_enabled():
            positions = gathered_parameters[0].double()
            cxx, cxy, cyy = model.covariance_components_from_rows(
                gathered_parameters[1], gathered_parameters[2]
            )
        else:
            positions = model.positions(index.active_splat, sorted_unique=True).double()
            cxx, cxy, cyy = model.covariance_components(index.active_splat, sorted_unique=True)
        cxx, cxy, cyy = cxx.double(), cxy.double(), cyy.double()
        output = x.new_zeros(len(x))
        self.last_stats = {"pairs": int(index.pair_point.numel())}
        if not index.pair_point.numel():
            return output + model.sed.templates().sum() * 0.0
        pair_chunk = env_chunk("SPLAT_PAIR_CHUNK", self.pair_chunk)
        for start in range(0, int(index.pair_point.numel()), int(pair_chunk)):
            point = index.pair_point[start : start + int(pair_chunk)]
            active = index.pair_active_splat[start : start + int(pair_chunk)]
            if (
                self.execution == "cached"
                and x.is_cuda
                and eigen_fused_predict_enabled()
            ):
                from ..pgm.eigen_mog_full import full_pairs

                value = full_pairs(positions, cxx, cxy, cyy, amplitudes, x, y, bands, point, active, context)
                output = output.index_add(0, point, value.to(output.dtype))
                continue
            inverse = context["inverse"][point]
            i00, i01 = inverse[:, 0, 0], inverse[:, 0, 1]
            i10, i11 = inverse[:, 1, 0], inverse[:, 1, 1]
            delta = torch.stack([x[point], y[point]], -1).double() - positions[active]
            dnx = i00 * delta[:, 0] + i01 * delta[:, 1]
            dny = i10 * delta[:, 0] + i11 * delta[:, 1]
            a, b, d = cxx[active], cxy[active], cyy[active]
            native_xx = i00 * i00 * a + 2 * i00 * i01 * b + i01 * i01 * d
            native_xy = i00 * i10 * a + (i00 * i11 + i01 * i10) * b + i01 * i11 * d
            native_yy = i10 * i10 * a + 2 * i10 * i11 * b + i11 * i11 * d
            geometry = [v.to(self.compute_dtype) for v in (dnx, dny, native_xx, native_xy, native_yy)]
            tables = [context[k] for k in ("weights", "mx", "my", "sxx", "sxy", "syy")]
            if self.execution == "reference":
                value = eigen_mog_pairs(*geometry, point, *tables, implementation=implementation)
            else:
                exposure = context["exposure_row"][point]
                group = None
                if self.execution == "grouped":
                    key = active * context["n_groups"] + context["point_group"][point]
                    unique, group = torch.unique(key, return_inverse=True)
                    representative = torch.full_like(unique, len(point))
                    representative.scatter_reduce_(
                        0, group, torch.arange(len(point), device=x.device), reduce="amin", include_self=True
                    )
                    exposure = exposure[representative]
                    geometry[2:] = [v[representative] for v in geometry[2:]]
                value = reuse_pairs(
                    *geometry, point, exposure, tables, group=group, implementation=implementation
                )
            value = value.double() * amplitudes[active, bands[point]].double() / context["det"][point]
            output = output.index_add(0, point, value.to(output.dtype))
        return output
