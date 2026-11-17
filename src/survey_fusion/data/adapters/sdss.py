from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, cast

import numpy as np
import torch

from ..bandpasses import Bandpass
from ..regions import (
    SkyCircleRegion, aperture_overlaps_region, angular_separation_arcsec,
    local_sky_circle_mask_from_omega_wcs, normalize_pointing_offsets, parse_holdout_region_config,
    parse_region_metrics_config, sky_circle_mask_from_omega_wcs,
)
from ..random_holdout import random_holdout_local_indices
from ..sdss_scene import SdssSceneData, _bandpass_fnu_to_flambda_factor
from ...evaluation.diagnostics import SpectrumDiagnosticTarget
from ...pgm.contracts import (
    DatumBatch, DatumKernelFactory, DatumKind, ExposureState, ExposureTable, SkyDiskApertureSpec,
    validate_batch_datum_kinds, validate_batch_exposure_links,
)
from ...pgm.kernel_builders import KernelFactoryFromBuilders
from ...utils.spectral import per_bin_edges_from_centers
from ...utils.wcs import AstropyWCSAdapter, TangentPlaneWCS


@dataclass(slots=True)
class SdssDatumStore:
    exposures: ExposureTable
    batch: DatumBatch
    image_band_by_exposure: Dict[int, str]
    spec_diagnostics_by_exposure: Dict[int, SpectrumDiagnosticTarget]
    spec_label_by_exposure: Dict[int, str]
    spec_role_by_exposure: Dict[int, str]
    kernel_factory: DatumKernelFactory
    n_image: int
    n_spec: int
    image_excluded_local_indices_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    image_random_holdout_indices_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    region_masks_by_exposure: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    holdout_region_summary: Dict[str, Any] = field(default_factory=dict)
    region_metrics_enabled: bool = False
    partial_holdout_enabled: bool = False
    image_source_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    image_additive_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    image_total_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    spec_source_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    spec_additive_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    spec_total_by_exposure: Dict[int, np.ndarray] = field(default_factory=dict)
    psf_mixture_digest: str | None = None
    psf_mixture_digest_version: int | None = None

    def gather_rows(self, field: str, global_indices: torch.Tensor) -> torch.Tensor:
        key = str(field)
        values = getattr(self.batch, key)
        if values is None:
            raise KeyError(f"DatumBatch field {field!r} is not present.")
        idx_cpu = global_indices.detach().to(dtype=torch.long, device="cpu")
        return torch.index_select(values, 0, idx_cpu)

    def materialize_image_by_exposure(self, exposure_id: int, kind: str) -> np.ndarray:
        exp_id = int(exposure_id)
        if exp_id not in self.image_band_by_exposure:
            raise KeyError(f"Exposure {exp_id} is not an image exposure.")
        exposure = self.exposures[int(exp_id)]
        height = exposure.image_height
        width = exposure.image_width
        if height is None or width is None:
            raise ValueError(f"Exposure {exp_id} does not have image dimensions.")
        key = str(kind).strip().lower()
        if key in {"target", "total", "y"} and exp_id in self.image_total_by_exposure:
            return np.asarray(self.image_total_by_exposure[exp_id], dtype=np.float32)
        if key == "source" and exp_id in self.image_source_by_exposure:
            return np.asarray(self.image_source_by_exposure[exp_id], dtype=np.float32)
        if key == "additive" and exp_id in self.image_additive_by_exposure:
            return np.asarray(self.image_additive_by_exposure[exp_id], dtype=np.float32)
        image_kind = int(DatumKind.IMAGE_PIXEL)
        mask = (self.batch.exposure_id == int(exp_id)) & (self.batch.datum_kind == image_kind)
        rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if int(rows.numel()) == 0:
            raise ValueError(f"Exposure {exp_id} has no image datums to materialize.")
        local = torch.index_select(self.batch.datum_index, 0, rows).detach().cpu().numpy()
        y = torch.index_select(self.batch.y, 0, rows).detach().cpu().numpy().astype(np.float32)
        if self.batch.additive is None:
            additive = np.zeros_like(y, dtype=np.float32)
        else:
            additive = (
                torch.index_select(self.batch.additive, 0, rows)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        if int(np.asarray(local).size) == 0:
            raise ValueError(f"Exposure {exp_id} has no image datums to materialize.")
        if key in {"target", "total", "y"}:
            values = y
        elif key == "additive":
            values = additive
        elif key == "source":
            values = y - additive
        else:
            raise ValueError(
                f"Unsupported image materialization kind {kind!r}; "
                "expected 'total', 'source', or 'additive'."
            )
        image = np.full((int(height), int(width)), np.nan, dtype=np.float32)
        image.reshape(-1)[local.astype(np.int64, copy=False)] = values
        return image

    def materialize_image_total(self, exposure_id: int) -> np.ndarray:
        return self.materialize_image_by_exposure(int(exposure_id), "total")

    def materialize_image_source(self, exposure_id: int) -> np.ndarray:
        return self.materialize_image_by_exposure(int(exposure_id), "source")

    def materialize_image_additive(self, exposure_id: int) -> np.ndarray:
        return self.materialize_image_by_exposure(int(exposure_id), "additive")


def _normalize_psf_map(
    psf_fwhm_arcsec: float | Dict[str, float] | None, bands: list[str]
) -> tuple[Dict[str, float], float]:
    if psf_fwhm_arcsec is None:
        raise ValueError("psf_fwhm_arcsec must be provided for SDSS data.")
    if isinstance(psf_fwhm_arcsec, dict):
        psf_map = {band: float(psf_fwhm_arcsec.get(band, 0.0)) for band in bands}
    else:
        psf_map = {band: float(psf_fwhm_arcsec) for band in bands}
    values = [v for v in psf_map.values() if v > 0]
    default = float(np.median(values)) if values else 0.0
    return psf_map, default


def _resolve_spectral_spatial_response(spectrum: object, *, mode: str, legacy_fwhm: float) -> dict[str, Any]:
    """Validate declared geometry; seeing alone never supplies a flux correction."""
    if mode == "legacy_imaging_median":
        return {
            "schema_version": 1, "model": "gaussian", "fwhm_arcsec": float(legacy_fwhm),
            "flux_convention": "legacy_finite_aperture_assumption",
            "provenance": {"source": "image exposures", "field": "psf_fwhm_arcsec",
                           "method": "legacy imaging median inheritance"},
        }
    if mode != "metadata":
        raise ValueError("spectral_response_mode must be metadata or legacy_imaging_median.")
    raw = dict(getattr(spectrum, "metadata", {}) or {}).get("spatial_response")
    label = getattr(spectrum, "label", "spectrum")
    if not isinstance(raw, dict):
        raise ValueError(f"Spectrum {label!r} requires explicit spatial_response metadata.")
    response = dict(raw)
    if response.get("schema_version") != 1 or response.get("model") != "gaussian":
        raise ValueError(f"Spectrum {label!r} requires spatial_response schema_version=1, model=gaussian.")
    value = response.get("fwhm_arcsec")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Spectrum {label!r} spatial_response fwhm_arcsec must be positive and finite.")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(key), str) or not provenance[key].strip()
        for key in ("source", "field", "method")
    ):
        raise ValueError(f"Spectrum {label!r} spatial_response requires source, field, method provenance.")
    if response.get("flux_convention") not in ("published_point_source_calibrated", "finite_aperture"):
        raise ValueError(f"Spectrum {label!r} spatial_response requires an identified flux_convention.")
    calibration = response.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"Spectrum {label!r} requires explicit calibration metadata.")
    calibration_mode = calibration.get("mode")
    if calibration_mode == "finite_aperture":
        if response["flux_convention"] != "finite_aperture":
            raise ValueError("finite_aperture calibration requires finite_aperture flux_convention.")
    elif calibration_mode == "gaussian_point_source":
        if response["flux_convention"] != "published_point_source_calibrated":
            raise ValueError("gaussian_point_source calibration requires published_point_source_calibrated flux_convention.")
        if calibration.get("approximation") != "centered_gaussian_standard_same_seeing":
            raise ValueError("gaussian_point_source calibration requires an explicit centered Gaussian approximation.")
        provenance = calibration.get("provenance")
        if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(key), str) or not provenance[key].strip()
            for key in ("source", "field", "method")
        ):
            raise ValueError("gaussian_point_source calibration requires source, field, method provenance.")
    else:
        raise ValueError("calibration mode must be finite_aperture or gaussian_point_source.")
    response["fwhm_arcsec"] = float(value)
    return response


def _compute_lsf_fwhm_angstrom_by_bin(
    spec_loglam: np.ndarray, spec_wdisp: np.ndarray, lam_spec: np.ndarray,
    *, dlog10_lambda: float | None = None,
) -> np.ndarray:
    lam = np.asarray(lam_spec, dtype=np.float64)
    if lam.ndim != 1:
        raise ValueError("lam_spec must be 1D.")
    if lam.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if dlog10_lambda is None:
        if np.asarray(spec_loglam).size < 2:
            return np.zeros_like(lam, dtype=np.float64)
        dlog = float(np.median(np.diff(np.asarray(spec_loglam, dtype=np.float64))))
    else:
        dlog = float(dlog10_lambda)
        if not math.isfinite(dlog) or dlog <= 0:
            raise ValueError("dlog10_lambda must be positive and finite.")
    widths = np.asarray(spec_wdisp, dtype=np.float64)
    if dlog10_lambda is not None and (np.any(~np.isfinite(widths)) or np.any(widths <= 0)):
        raise ValueError("Native spec_wdisp widths must be positive and finite.")
    sigma_log10 = widths * dlog
    if sigma_log10.shape != lam.shape:
        raise ValueError("spec_wdisp must align with lam_spec for per-bin LSF conversion.")
    sigma_lam = sigma_log10 * np.log(10.0) * lam
    fwhm = 2.355 * sigma_lam
    fwhm = np.where(np.isfinite(fwhm), np.clip(fwhm, 0.0, None), 0.0)
    return fwhm.astype(np.float64)


def _validate_scene_geometry_consistency(
    scene: SdssSceneData, *, bands: list[str], reference_band: str, reference_shape: tuple[int, int],
    pixel_scale_rtol: float = 1.0e-3, pixel_scale_atol: float = 1.0e-6,
) -> None:
    reference_scale = float(scene.pixel_scale_arcsec_per_band[reference_band])
    mismatched_shapes: list[str] = []
    mismatched_scales: list[str] = []
    for band in bands:
        image = np.asarray(scene.images_flambda[band])
        if image.ndim != 2:
            raise ValueError(f"SDSS image band {band!r} must be 2D, got shape {image.shape}.")
        if tuple(int(dim) for dim in image.shape) != tuple(int(dim) for dim in reference_shape):
            mismatched_shapes.append(f"{band}:{tuple(int(dim) for dim in image.shape)}")
        pixel_scale = float(scene.pixel_scale_arcsec_per_band[band])
        if not np.isclose(
            pixel_scale, reference_scale, rtol=float(pixel_scale_rtol), atol=float(pixel_scale_atol),
        ):
            mismatched_scales.append(f"{band}:{pixel_scale:.8g}")
    if mismatched_shapes:
        raise ValueError(
            "SDSS image bands must share one scene geometry for manifest training. "
            f"Reference band {reference_band!r} has shape {reference_shape}, but got "
            + ", ".join(mismatched_shapes)
            + "."
        )
    if mismatched_scales:
        raise ValueError(
            "SDSS image bands must share one scene pixel scale for manifest training. "
            f"Reference band {reference_band!r} has pixel_scale_arcsec={reference_scale:.8g}, but got "
            + ", ".join(mismatched_scales)
            + "."
        )


def _image_fiber_mask_for_exposure(
    scene: SdssSceneData, spectra: list[object], *, obs_wcs: object, image_shape: tuple[int, int],
    pixel_scale_arcsec: float,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    science_mask = np.zeros((height, width), dtype=bool)
    for spectrum in spectra:
        fiber_ra, fiber_dec = tuple(
            float(value) for value in getattr(spectrum, "center_ra_dec", (np.nan, np.nan))
        )
        radius_arcsec = float(getattr(spectrum, "fiber_radius_arcsec", 0.0) or 0.0)
        if not np.isfinite(fiber_ra) or not np.isfinite(fiber_dec):
            continue
        if not np.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
            continue
        region = SkyCircleRegion(
            name=str(getattr(spectrum, "label", "fiber")), center_ra_deg=fiber_ra, center_dec_deg=fiber_dec,
            radius_arcsec=radius_arcsec,
        )
        local_mask = local_sky_circle_mask_from_omega_wcs(
            obs_wcs, image_shape, region, pixel_scale_arcsec=float(pixel_scale_arcsec),
        )
        if local_mask is None:
            local_mask = np.asarray(sky_circle_mask_from_omega_wcs(obs_wcs, image_shape, region), dtype=bool)
        science_mask |= local_mask
    if not science_mask.any():
        scene_mask = np.asarray(scene.fiber_mask, dtype=np.float32)
        if scene_mask.ndim != 2:
            raise ValueError("scene.fiber_mask must be a 2D array.")
        if scene_mask.shape != tuple(int(v) for v in image_shape):
            return np.zeros(image_shape, dtype=np.float32)
        return scene_mask.astype(np.float32)
    return science_mask.astype(np.float32)


def _assemble_column(parts: list, dtype, name: str, spill_dir: str | None):
    """Join per-exposure pieces into one column.

    np.concatenate needs the parts and the result live at once, so peak memory is
    twice the column. At COSMOS scale the seven columns total ~50 GB, so that
    spike alone is ~50 GB of avoidable headroom. When spill_dir is set, write
    straight into a memmap and release each part as it is consumed: the array
    stays paged by the OS and resident memory tracks the working set rather than
    the whole table.
    """
    total = int(sum(int(p.size) for p in parts))
    if not spill_dir:
        out = np.concatenate(parts, axis=0).astype(dtype, copy=False)
        parts.clear()
        return out
    os.makedirs(spill_dir, exist_ok=True)
    path = os.path.join(spill_dir, f"datum_{name}.npy")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(total,))
    off = 0
    while parts:
        p = parts.pop(0)
        arr[off : off + p.size] = p
        off += int(p.size)
        del p
    arr.flush()
    return arr


def build_sdss_datums(
    scene: SdssSceneData, bandpasses: Dict[str, Bandpass], *,
    psf_fwhm_arcsec: float | Dict[str, float] | None, pointing_offsets_arcsec: object = None,
    rng_seed: int | None = None, device: torch.device, holdout_band: str | None = None,
    holdout_region_cfg: dict | None = None, region_metrics_cfg: dict | None = None,
    validate_kernel_samples: bool = True, random_holdout_cfg: dict | None = None,
    spill_dir: str | None = None,
    spectral_response_mode: str = "metadata",
    spectral_lsf_sampling: str = "native",
    spectral_response_missing_policy: str = "error",
) -> SdssDatumStore:
    if spectral_response_mode not in ("metadata", "legacy_imaging_median"):
        raise ValueError("spectral_response_mode must be metadata or legacy_imaging_median.")
    if spectral_lsf_sampling not in ("native", "legacy_masked_spacing"):
        raise ValueError("spectral_lsf_sampling must be native or legacy_masked_spacing.")
    if spectral_response_missing_policy not in ("error", "exclude"):
        raise ValueError("spectral_response_missing_policy must be error or exclude.")
    bands = list(bandpasses.keys())
    reference_band = bands[0] if bands else None
    if reference_band is None:
        raise ValueError("No bandpasses provided for SDSS datums.")
    image_exposures = list(getattr(scene, "image_exposures", None) or [])
    scene_metadata = dict(getattr(scene, "metadata", {}) or {})
    has_image_exposures = bool(image_exposures)
    has_individual_image_exposures = has_image_exposures
    if reference_band not in scene.images_flambda:
        reference_band = next(iter(scene.images_flambda))

    ref_image = scene.images_flambda[reference_band]
    if has_individual_image_exposures and getattr(scene, "model_wcs", None) is not None:
        height = int(scene.model_wcs.height)
        width = int(scene.model_wcs.width)
    else:
        height, width = ref_image.shape
    if not has_individual_image_exposures:
        _validate_scene_geometry_consistency(
            scene, bands=bands, reference_band=reference_band, reference_shape=(height, width),
        )
    if has_individual_image_exposures:
        exposure_psfs = [
            float(getattr(image_exp, "psf_fwhm_arcsec"))
            for image_exp in image_exposures
            if getattr(image_exp, "psf_fwhm_arcsec", None) is not None
        ]
        if not exposure_psfs:
            raise ValueError("Image-exposure SDSS datums require per-exposure psf_fwhm_arcsec.")
        psf_map = {}
        psf_default = float(np.median(exposure_psfs))
    else:
        psf_map, psf_default = _normalize_psf_map(psf_fwhm_arcsec, bands)
    pointing_offsets = normalize_pointing_offsets(pointing_offsets_arcsec, bands)
    holdout_cfg = parse_holdout_region_config(holdout_region_cfg)
    holdout_enabled = bool(holdout_cfg.get("enabled", False))
    holdout_region = cast(SkyCircleRegion | None, holdout_cfg.get("region", None))
    exclude_spectra_in_region = bool(holdout_cfg.get("exclude_spectra_in_region", False))
    holdout_band_name = str(holdout_band) if holdout_band is not None else None
    if holdout_enabled and holdout_region is None:
        raise ValueError("holdout_region.enabled=true requires valid holdout_region config.")
    if holdout_enabled and holdout_band_name is None:
        raise ValueError("holdout_region.enabled=true requires holdout_band to be provided.")
    if holdout_enabled and holdout_band_name not in bands:
        raise ValueError(f"holdout_band {holdout_band_name!r} not present in SDSS bandpasses {bands!r}.")
    region_metrics = parse_region_metrics_config(region_metrics_cfg)
    region_metrics_enabled = bool(region_metrics.get("enabled", False))
    regions = region_metrics.get("regions", [])
    primary_region = region_metrics.get("primary_region", None)
    spectra = list(getattr(scene, "spectra", []))
    science_spectra = [
        spectrum
        for spectrum in spectra
        if str(getattr(spectrum, "role", "science")).strip().lower() == "science"
    ]
    if science_spectra:
        primary_fiber_ra, primary_fiber_dec = science_spectra[0].center_ra_dec
    elif spectra:
        primary_fiber_ra, primary_fiber_dec = spectra[0].center_ra_dec
    else:
        primary_fiber_ra, primary_fiber_dec = tuple(
            float(value) for value in getattr(scene, "fiber_center_ra_dec", (0.0, 0.0))
        )
    exposure_ids_by_band = {band: idx for idx, band in enumerate(bands)}
    scene_ra0_rad = math.radians(primary_fiber_ra)
    scene_dec0_rad = math.radians(primary_fiber_dec)
    exposures: ExposureTable = {}
    image_band_by_exposure: Dict[int, str] = {}
    spec_diagnostics_by_exposure: Dict[int, SpectrumDiagnosticTarget] = {}
    spec_label_by_exposure: Dict[int, str] = {}
    spec_role_by_exposure: Dict[int, str] = {}
    image_excluded_local_indices_by_exposure: Dict[int, np.ndarray] = {}
    image_random_holdout_indices_by_exposure: Dict[int, np.ndarray] = {}
    random_holdout_fraction = float((random_holdout_cfg or {}).get("fraction", 0.0) or 0.0)
    random_holdout_seed = int((random_holdout_cfg or {}).get("seed", 0) or 0)
    if random_holdout_fraction > 0.0 and not (random_holdout_cfg or {}).get("seed"):
        raise ValueError("random_holdout.fraction > 0 requires an explicit random_holdout.seed.")
    region_masks_by_exposure: Dict[int, Dict[str, np.ndarray]] = {}
    spectra_summary: list[Dict[str, Any]] = []
    y_parts = []
    sigma_parts = []
    additive_parts = []
    source_gate_parts = []
    datum_kind_parts = []
    datum_index_parts = []
    exposure_id_parts = []
    holdout_summary: Dict[str, Any] = {
        "enabled": bool(holdout_enabled),
        "band": holdout_band_name,
        "center_radec_deg": (
            [float(holdout_region.center_ra_deg), float(holdout_region.center_dec_deg)]
            if holdout_region is not None
            else None
        ),
        "radius_arcsec": (float(holdout_region.radius_arcsec) if holdout_region is not None else None),
        "exclude_spectra_in_region": bool(exclude_spectra_in_region),
        "excluded_pixels_by_exposure": {},
        "primary_region": str(primary_region) if primary_region is not None else None,
        "region_names": [str(region.name) for region in regions],
        "spectra": spectra_summary,
        "spectrum_center_radec_deg": None,
        "spectrum_center_sep_arcsec": None,
        "spectrum_excluded": False,
    }
    n_image = 0
    image_source_by_exposure: Dict[int, np.ndarray] = {}
    image_additive_by_exposure: Dict[int, np.ndarray] = {}
    image_total_by_exposure: Dict[int, np.ndarray] = {}
    spec_source_by_exposure: Dict[int, np.ndarray] = {}
    spec_additive_by_exposure: Dict[int, np.ndarray] = {}
    spec_total_by_exposure: Dict[int, np.ndarray] = {}
    image_conversion = {band: _bandpass_fnu_to_flambda_factor(bp) for band, bp in bandpasses.items()}
    training_flux_scale = float(scene_metadata.get("training_flux_scale", 1.0))

    def _iter_image_items():
        if has_image_exposures:
            for exp_id, image_exp in enumerate(image_exposures):
                band = str(getattr(image_exp, "band"))
                if band not in bandpasses:
                    continue
                yield (
                    int(exp_id), band, str(getattr(image_exp, "label", f"{band}-e{exp_id}")),
                    np.asarray(getattr(image_exp, "target_flambda"), dtype=np.float32),
                    np.asarray(getattr(image_exp, "additive_flambda"), dtype=np.float32),
                    np.asarray(getattr(image_exp, "source_flambda"), dtype=np.float32),
                    np.asarray(getattr(image_exp, "total_flambda"), dtype=np.float32),
                    np.asarray(getattr(image_exp, "sigma_flambda"), dtype=np.float32),
                    getattr(image_exp, "wcs"), float(getattr(image_exp, "pixel_scale_arcsec")),
                    float(getattr(image_exp, "psf_fwhm_arcsec")),
                    getattr(image_exp, "psf_mixture_dgpsf", None),
                    getattr(image_exp, "psf_mixture_kl", None),
                    getattr(image_exp, "psf_effective_fwhm_arcsec", None),
                    getattr(image_exp, "psf_effective_fwhm_dg_arcsec", None),
                    getattr(image_exp, "psf_effective_fwhm_kl_arcsec", None),
                )
        else:
            for band in bands:
                image = scene.images_flambda[band].astype(np.float32)
                image_additive = np.asarray(
                    scene.image_additive_flambda[band]
                    if getattr(scene, "image_additive_flambda", None) is not None
                    and band in cast(dict[str, np.ndarray], scene.image_additive_flambda)
                    else np.zeros_like(image),
                    dtype=np.float32,
                )
                image_source = np.asarray(
                    scene.image_source_flambda[band]
                    if getattr(scene, "image_source_flambda", None) is not None
                    and band in cast(dict[str, np.ndarray], scene.image_source_flambda)
                    else image - image_additive,
                    dtype=np.float32,
                )
                image_total = np.asarray(
                    scene.image_total_flambda[band]
                    if getattr(scene, "image_total_flambda", None) is not None
                    and band in cast(dict[str, np.ndarray], scene.image_total_flambda)
                    else image,
                    dtype=np.float32,
                )
                if scene.image_sigma_flambda is None or band not in scene.image_sigma_flambda:
                    raise ValueError(
                        "SDSS image datums require per-pixel variance maps for Poisson image sigma, "
                        f"but none were found for band {band!r}. "
                        "Add gain/dark_variance to manifest.json bands."
                    )
                yield (
                    int(exposure_ids_by_band[band]), band, band, image, image_additive, image_source,
                    image_total, np.asarray(scene.image_sigma_flambda[band], dtype=np.float32),
                    scene.wcs_per_band[band], float(scene.pixel_scale_arcsec_per_band[band]),
                    float(psf_map.get(band, psf_default)), None, None, None, None, None,
                )

    for (
        exp_id, band, image_label, image, image_additive, image_source, image_total, image_sigma, image_wcs,
        image_pixel_scale_arcsec, image_psf_fwhm_arcsec, image_psf_mixture_dgpsf,
        image_psf_mixture_kl, image_psf_effective_fwhm_arcsec,
        image_psf_effective_fwhm_dg_arcsec, image_psf_effective_fwhm_kl_arcsec,
    ) in _iter_image_items():
        if image_additive.shape != image.shape:
            raise ValueError(
                f"Image additive shape for exposure {image_label!r} does not match "
                "target image."
            )
        flat = image.flatten()
        sigma = image_sigma.flatten().astype(np.float32)
        additive_flat = image_additive.flatten()
        valid_flat = (
            np.isfinite(flat)
            & np.isfinite(sigma)
            & (sigma > 0.0)
            & np.isfinite(additive_flat)
            & np.isfinite(image_source.flatten())
            & np.isfinite(image_total.flatten())
        )
        if not np.any(valid_flat):
            raise ValueError(f"Image exposure {image_label!r} has no finite training pixels.")
        valid_indices = np.flatnonzero(valid_flat).astype(np.int64)
        y_parts.append(flat[valid_flat])
        sigma_parts.append(sigma[valid_flat])
        additive_parts.append(additive_flat[valid_flat])
        source_gate_parts.append(np.ones(valid_indices.size, dtype=np.float32))
        datum_kind_parts.append(np.full(valid_indices.size, int(DatumKind.IMAGE_PIXEL), dtype=np.int64))
        datum_index_parts.append(valid_indices)
        exposure_id_parts.append(np.full(valid_indices.size, exp_id, dtype=np.int64))
        n_image += int(valid_indices.size)
        image_band_by_exposure[exp_id] = band
        image_source_by_exposure[exp_id] = image_source
        image_additive_by_exposure[exp_id] = image_additive
        image_total_by_exposure[exp_id] = image_total
        wcs = (
            scene.model_wcs
            if has_individual_image_exposures and getattr(scene, "model_wcs", None) is not None
            else TangentPlaneWCS(
                ra0_rad=scene_ra0_rad, dec0_rad=scene_dec0_rad,
                pixel_scale_arcsec=float(image_pixel_scale_arcsec), width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
        )
        dx_arcsec, dy_arcsec = pointing_offsets.get(band, (0.0, 0.0))
        obs_wcs = AstropyWCSAdapter(image_wcs, dx_arcsec=dx_arcsec, dy_arcsec=dy_arcsec)
        image_fiber_mask = _image_fiber_mask_for_exposure(
            scene, science_spectra, obs_wcs=obs_wcs, image_shape=(int(image.shape[0]), int(image.shape[1])),
            pixel_scale_arcsec=float(image_pixel_scale_arcsec),
        )
        if holdout_enabled and band == holdout_band_name:
            holdout_mask = sky_circle_mask_from_omega_wcs(obs_wcs, image.shape, holdout_region)
            excluded_local = np.flatnonzero(holdout_mask.reshape(-1)).astype(np.int64)
            image_excluded_local_indices_by_exposure[exp_id] = excluded_local
            holdout_summary["excluded_pixels_by_exposure"][str(exp_id)] = int(excluded_local.size)
        if random_holdout_fraction > 0.0:
            rh_local = random_holdout_local_indices(
                int(image.size), random_holdout_fraction, random_holdout_seed, salt=int(exp_id)
            )
            image_random_holdout_indices_by_exposure[exp_id] = rh_local
            prior = image_excluded_local_indices_by_exposure.get(exp_id)
            union = rh_local if prior is None else np.union1d(prior, rh_local)
            image_excluded_local_indices_by_exposure[exp_id] = union.astype(np.int64)
            holdout_summary.setdefault("random_holdout_pixels_by_exposure", {})[str(exp_id)] = int(
                rh_local.size
            )
        if region_metrics_enabled:
            masks: Dict[str, np.ndarray] = {}
            for region in regions:
                masks[str(region.name)] = sky_circle_mask_from_omega_wcs(obs_wcs, image.shape, region)
            region_masks_by_exposure[exp_id] = masks
        exposures[exp_id] = ExposureState(
            wcs=wcs, obs_wcs=obs_wcs, psf_fwhm_arcsec=float(image_psf_fwhm_arcsec), lsf_fwhm_angstrom=None,
            bandpasses={band:bandpasses[band]}, lam_spec=np.asarray(scene.lam_spec, dtype=np.float64),
            fiber_mask=image_fiber_mask.copy(), pixel_scale_arcsec=float(image_pixel_scale_arcsec),
            image_height=int(image.shape[0]), image_width=int(image.shape[1]), lam_edges=None, aperture=None,
            psf_mixture_dgpsf=image_psf_mixture_dgpsf,
            psf_mixture_kl=image_psf_mixture_kl,
            psf_effective_fwhm_arcsec=image_psf_effective_fwhm_arcsec,
            psf_effective_fwhm_dg_arcsec=image_psf_effective_fwhm_dg_arcsec,
            psf_effective_fwhm_kl_arcsec=image_psf_effective_fwhm_kl_arcsec,
            label=str(image_label),
        )

    # A fiber outside every image exposure's footprint has no photometric
    # constraint: no peak seeds it and no image gradient can build it, so its
    # bins are unfittable and only bias the spectral loss. Coverage requires an
    # invertible WCS (real survey WCS objects; synthetic forward-only test WCS
    # objects skip this check).
    coverage_footprints = []
    if has_image_exposures:
        for image_exp in image_exposures:
            if str(getattr(image_exp, "band")) not in bandpasses:
                continue
            image_wcs = getattr(image_exp, "wcs", None)
            if not callable(getattr(image_wcs, "wcs_world2pix", None)):
                continue
            frame_height, frame_width = image_exp.source_flambda.shape
            coverage_footprints.append((image_wcs, float(frame_height), float(frame_width)))

    n_spec = 0
    n_spec_no_coverage = 0
    first_spec_exposure_id = (max(image_band_by_exposure.keys()) + 1 if image_band_by_exposure else 0)
    primary_summary: Dict[str, Any] | None = None
    for spec_offset, spectrum in enumerate(spectra):
        spec_targets = np.asarray(spectrum.flux_spec_flambda, dtype=np.float32)
        spec_additive = np.asarray(
            spectrum.spec_additive_flambda
            if getattr(spectrum, "spec_additive_flambda", None) is not None
            else np.zeros_like(spec_targets),
            dtype=np.float32,
        )
        spec_source = np.asarray(
            spectrum.spec_source_flambda
            if getattr(spectrum, "spec_source_flambda", None) is not None
            else spec_targets - spec_additive,
            dtype=np.float32,
        )
        spec_total = np.asarray(
            spectrum.spec_total_flambda
            if getattr(spectrum, "spec_total_flambda", None) is not None
            else spec_targets,
            dtype=np.float32,
        )
        if spec_additive.shape != spec_targets.shape:
            raise ValueError(
                f"Spectrum additive shape for {getattr(spectrum, 'label', spec_offset)!r} "
                "does not match target spectrum."
            )
        sigma_spec = np.asarray(spectrum.sigma_spec_flambda, dtype=np.float32)
        n_spec_raw = int(spec_targets.size)
        include_spec = n_spec_raw > 0
        fiber_ra, fiber_dec = spectrum.center_ra_dec
        spec_summary: Dict[str, Any] = {
            "label": str(getattr(spectrum, "label", f"S{spec_offset + 1}")),
            "role": str(getattr(spectrum, "role", "science")),
            "center_radec_deg": [float(fiber_ra), float(fiber_dec)], "center_sep_arcsec": None,
            "excluded": False, "exposure_id": None, "n_bins": int(n_spec_raw),
        }
        if holdout_enabled and holdout_region is not None:
            spec_summary["center_sep_arcsec"] = float(
                angular_separation_arcsec(
                    float(fiber_ra), float(fiber_dec), float(holdout_region.center_ra_deg),
                    float(holdout_region.center_dec_deg),
                )
            )
        if (
            include_spec
            and holdout_enabled
            and holdout_region is not None
            and exclude_spectra_in_region
            and aperture_overlaps_region(
                float(fiber_ra), float(fiber_dec), float(spectrum.fiber_radius_arcsec), holdout_region,
            )
        ):
            include_spec = False
            spec_summary["excluded"] = True

        if include_spec and coverage_footprints:
            fiber_covered = False
            for image_wcs, frame_height, frame_width in coverage_footprints:
                fx, fy = image_wcs.wcs_world2pix(
                    np.array([float(fiber_ra)]), np.array([float(fiber_dec)]), 0
                )
                if (
                    -0.5 <= float(fx[0]) < frame_width - 0.5
                    and -0.5 <= float(fy[0]) < frame_height - 0.5
                ):
                    fiber_covered = True
                    break
            if not fiber_covered:
                include_spec = False
                spec_summary["excluded"] = True
                spec_summary["excluded_reason"] = "no_image_coverage"
                n_spec_no_coverage += 1

        if (include_spec and spectral_response_mode == "metadata"
                and dict(getattr(spectrum, "metadata", {}) or {}).get("spatial_response") is None
                and spectral_response_missing_policy == "exclude"):
            include_spec = False
            spec_summary["excluded"] = True
            spec_summary["excluded_reason"] = "missing_spectral_response"
        if include_spec and n_spec_raw > 0:
            spectral_response = _resolve_spectral_spatial_response(
                spectrum, mode=spectral_response_mode, legacy_fwhm=psf_default,
            )
            spectral_source_scale = None
            if spectral_response.get("calibration", {}).get("mode") == "gaussian_point_source":
                radius = float(spectrum.fiber_radius_arcsec)
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError("Gaussian calibration requires a positive finite fiber radius.")
                sigma = spectral_response["fwhm_arcsec"] / 2.355
                acceptance = -math.expm1(-0.5 * (radius / sigma) ** 2)
                spectral_source_scale = 1.0 / acceptance
                if not math.isfinite(spectral_source_scale) or spectral_source_scale <= 0:
                    raise ValueError("Gaussian calibration source scale must be positive and finite.")
                spectral_response["resolved_source_scale"] = spectral_source_scale
            spec_summary["spatial_response"] = spectral_response
            spec_summary["lsf_sampling"] = spectral_lsf_sampling
            spec_exposure_id = first_spec_exposure_id + len(spec_label_by_exposure)
            lam_spec = np.asarray(spectrum.lam_spec, dtype=np.float64)
            scene_lam_edges = getattr(spectrum, "lam_edges", None)
            if scene_lam_edges is not None:
                lam_edges = np.asarray(scene_lam_edges, dtype=np.float64)
                if lam_edges.shape != (lam_spec.size, 2):
                    raise ValueError(
                        "Scene lam_edges must have shape (n_bins, 2) matching lam_spec."
                    )
            else:
                # synthetic scenes without a pipeline grid: no masked gaps, so
                # center-derived edges are exact
                lam_edges = per_bin_edges_from_centers(lam_spec)
            lsf_fwhm_angstrom = _compute_lsf_fwhm_angstrom_by_bin(
                np.asarray(spectrum.spec_loglam, dtype=np.float64),
                np.asarray(spectrum.spec_wdisp, dtype=np.float64), lam_spec,
                dlog10_lambda=(dict(getattr(spectrum, "metadata", {}) or {}).get("native_dlog10_lambda")
                              if spectral_lsf_sampling == "native" else None),
            )
            y_parts.append(spec_targets)
            sigma_parts.append(sigma_spec)
            additive_parts.append(spec_additive)
            spec_role = str(getattr(spectrum, "role", "science")).strip().lower()
            source_gate_value = 0.0 if spec_role != "science" else 1.0
            source_gate_parts.append(np.full(n_spec_raw, source_gate_value, dtype=np.float32))
            datum_kind_parts.append(np.full(n_spec_raw, int(DatumKind.SPECTRAL_BIN), dtype=np.int64))
            datum_index_parts.append(np.arange(n_spec_raw, dtype=np.int64))
            exposure_id_parts.append(np.full(n_spec_raw, int(spec_exposure_id), dtype=np.int64))
            spec_wcs = (
                scene.model_wcs
                if has_individual_image_exposures and getattr(scene, "model_wcs", None) is not None
                else TangentPlaneWCS(
                    ra0_rad=scene_ra0_rad, dec0_rad=scene_dec0_rad,
                    pixel_scale_arcsec=float(scene.pixel_scale_arcsec), width=width, height=height,
                )
            )
            exposures[int(spec_exposure_id)] = ExposureState(
                wcs=spec_wcs,
                obs_wcs=None,
                psf_fwhm_arcsec=spectral_response["fwhm_arcsec"],
                spectral_response_metadata=spectral_response,
                spectral_source_scale=spectral_source_scale,
                lsf_fwhm_angstrom=lsf_fwhm_angstrom,
                bandpasses={},
                lam_spec=lam_spec,
                fiber_mask=spectrum.fiber_mask,
                pixel_scale_arcsec=float(scene.pixel_scale_arcsec),
                image_height=None,
                image_width=None,
                lam_edges=lam_edges,
                aperture=SkyDiskApertureSpec(
                    center_radec_deg=(float(fiber_ra), float(fiber_dec)),
                    radius_arcsec=float(spectrum.fiber_radius_arcsec),
                ),
            )
            spec_diagnostics_by_exposure[int(spec_exposure_id)] = SpectrumDiagnosticTarget(
                label=str(spec_summary["label"]), role=str(spec_summary["role"]),
                lam=np.asarray(lam_spec, dtype=np.float64),
                truth_flux_density=np.asarray(spectrum.flux_spec_flam_lsf, dtype=np.float64),
                truth_counts=None, throughput=np.asarray(spectrum.spec_throughput, dtype=np.float64),
            )
            spec_label_by_exposure[int(spec_exposure_id)] = str(spec_summary["label"])
            spec_role_by_exposure[int(spec_exposure_id)] = str(spec_summary["role"])
            spec_source_by_exposure[int(spec_exposure_id)] = spec_source
            spec_additive_by_exposure[int(spec_exposure_id)] = spec_additive
            spec_total_by_exposure[int(spec_exposure_id)] = spec_total
            spec_summary["exposure_id"] = int(spec_exposure_id)
            n_spec += int(n_spec_raw)
        spectra_summary.append(spec_summary)
        if str(spec_summary["role"]).strip().lower() == "science" and primary_summary is None:
            primary_summary = spec_summary

    missing_response_labels = [row["label"] for row in spectra_summary
                               if row.get("excluded_reason") == "missing_spectral_response"]
    holdout_summary["spectral_response_mode"] = spectral_response_mode
    holdout_summary["spectral_lsf_sampling"] = spectral_lsf_sampling
    holdout_summary["spectral_response_missing_policy"] = spectral_response_missing_policy
    holdout_summary["missing_spectral_response_labels"] = missing_response_labels
    holdout_summary["missing_spectral_response_count"] = len(missing_response_labels)
    if n_spec_no_coverage:
        print(
            f"[sdss] excluded {n_spec_no_coverage} spectra with no image coverage "
            f"(excluded_reason=no_image_coverage in the spectra summary)",
            flush=True,
        )

    if primary_summary is None and spectra_summary:
        primary_summary = spectra_summary[0]

    if primary_summary is not None:
        holdout_summary["spectrum_center_radec_deg"] = list(primary_summary["center_radec_deg"])
        holdout_summary["spectrum_center_sep_arcsec"] = primary_summary["center_sep_arcsec"]
        holdout_summary["spectrum_excluded"] = bool(primary_summary["excluded"])

    if y_parts:
        y = _assemble_column(y_parts, np.float32, "y", spill_dir)
        sigma = _assemble_column(sigma_parts, np.float32, "sigma", spill_dir)
        additive = _assemble_column(additive_parts, np.float32, "additive", spill_dir)
        source_gate = _assemble_column(source_gate_parts, np.float32, "source_gate", spill_dir)
        datum_kind = _assemble_column(datum_kind_parts, np.int64, "datum_kind", spill_dir)
        datum_index = _assemble_column(datum_index_parts, np.int64, "datum_index", spill_dir)
        exposure_id = _assemble_column(exposure_id_parts, np.int64, "exposure_id", spill_dir)
    else:
        y = np.empty((0,), dtype=np.float32)
        sigma = np.empty((0,), dtype=np.float32)
        additive = np.empty((0,), dtype=np.float32)
        source_gate = np.empty((0,), dtype=np.float32)
        datum_kind = np.empty((0,), dtype=np.int64)
        datum_index = np.empty((0,), dtype=np.int64)
        exposure_id = np.empty((0,), dtype=np.int64)
    batch = DatumBatch(
        y=torch.from_numpy(y), sigma=torch.from_numpy(sigma), exposure_id=torch.from_numpy(exposure_id),
        datum_kind=torch.from_numpy(datum_kind), datum_index=torch.from_numpy(datum_index),
        kernel_samples=None, additive=torch.from_numpy(additive), source_gate=torch.from_numpy(source_gate),
    )
    validate_batch_exposure_links(batch, exposures)
    validate_batch_datum_kinds(batch)
    kernel_factory = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure=image_band_by_exposure, rng_seed=rng_seed,
        validate=bool(validate_kernel_samples),
    )
    return SdssDatumStore(
        exposures=exposures, batch=batch, image_band_by_exposure=image_band_by_exposure,
        spec_diagnostics_by_exposure=spec_diagnostics_by_exposure,
        spec_label_by_exposure=spec_label_by_exposure, spec_role_by_exposure=spec_role_by_exposure,
        kernel_factory=kernel_factory, n_image=n_image, n_spec=n_spec,
        image_excluded_local_indices_by_exposure=image_excluded_local_indices_by_exposure,
        image_random_holdout_indices_by_exposure=image_random_holdout_indices_by_exposure,
        region_masks_by_exposure=region_masks_by_exposure, holdout_region_summary=holdout_summary,
        region_metrics_enabled=bool(region_metrics_enabled), partial_holdout_enabled=bool(holdout_enabled),
        image_source_by_exposure=image_source_by_exposure,
        image_additive_by_exposure=image_additive_by_exposure,
        image_total_by_exposure=image_total_by_exposure, spec_source_by_exposure=spec_source_by_exposure,
        spec_additive_by_exposure=spec_additive_by_exposure, spec_total_by_exposure=spec_total_by_exposure,
        psf_mixture_digest=scene_metadata.get("psf_mixture_digest"),
        psf_mixture_digest_version=scene_metadata.get("psf_mixture_digest_version"),
    )
