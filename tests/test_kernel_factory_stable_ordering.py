from __future__ import annotations

import numpy as np
import pytest
import torch

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.pgm.contracts import (
    DatumBatch, DatumKind, ExposureState, GroupedKernelSamples, KernelSamples, SkyDiskApertureSpec,
)
from survey_fusion.pgm.kernel_builders import KernelFactoryFromBuilders
from survey_fusion.utils.wcs import TangentPlaneWCS, radec_deg_to_omega_xyz


def _bandpass() -> Bandpass:
    return Bandpass(
        band="r", lam=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        throughput=np.array([0.2, 1.0, 0.2], dtype=np.float64), canonical_id="test_r",
    )


def _make_exposures() -> dict[int, ExposureState]:
    wcs = TangentPlaneWCS(ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=0.4, width=2, height=2)
    img = ExposureState(
        wcs=wcs, obs_wcs=wcs, psf_fwhm_arcsec=1.1, lsf_fwhm_angstrom=None, bandpasses={"r": _bandpass()},
        lam_spec=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32), pixel_scale_arcsec=0.4, image_height=2, image_width=2,
        lam_edges=None, aperture=None,
    )
    spec = ExposureState(
        wcs=wcs, obs_wcs=None, psf_fwhm_arcsec=1.1,
        lsf_fwhm_angstrom=np.array([2.0, 2.0, 2.0], dtype=np.float64), bandpasses={},
        lam_spec=np.array([5000.0, 5100.0, 5200.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32), pixel_scale_arcsec=0.4, image_height=None,
        image_width=None,
        lam_edges=np.array(
            [[4950.0, 5050.0], [5050.0, 5150.0], [5150.0, 5250.0]], dtype=np.float64
        ),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.5),
    )
    return {0: img, 1: spec}


def _make_batch(device: torch.device) -> DatumBatch:
    # Datums: image(3), image(0), spec(1), spec(2), image(1)
    return DatumBatch(
        y=torch.zeros(5, dtype=torch.float32, device=device),
        sigma=torch.ones(5, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0, 0, 1, 1, 0], dtype=torch.long, device=device),
        datum_kind=torch.tensor(
            [
                int(DatumKind.IMAGE_PIXEL), int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN),
                int(DatumKind.SPECTRAL_BIN), int(DatumKind.IMAGE_PIXEL),
            ],
            dtype=torch.long,
            device=device,
        ),
        datum_index=torch.tensor([3, 0, 1, 2, 1], dtype=torch.long, device=device),
        kernel_samples=None,
    )


def _reorder_to_ids(tensor: torch.Tensor, ids_in_call: list[int], ids_target: list[int]) -> torch.Tensor:
    positions = {datum_id: i for i, datum_id in enumerate(ids_in_call)}
    order = [positions[datum_id] for datum_id in ids_target]
    return tensor[torch.tensor(order, dtype=torch.long, device=tensor.device)]


def _restore_call_order(samples: GroupedKernelSamples) -> KernelSamples:
    restored = samples.samples
    order = samples.row_original_index
    omega = torch.empty_like(restored.omega_xyz)
    loglam = torch.empty_like(restored.loglam)
    w = torch.empty_like(restored.w)
    z = torch.empty_like(restored.Z)
    omega[order] = restored.omega_xyz
    loglam[order] = restored.loglam
    w[order] = restored.w
    z[order] = restored.Z
    return KernelSamples(omega_xyz=omega, loglam=loglam, w=w, Z=z)


def _assert_grouped_samples_equal(lhs: GroupedKernelSamples, rhs: GroupedKernelSamples) -> None:
    assert lhs.exposure_blocks == rhs.exposure_blocks
    assert torch.equal(lhs.row_original_index, rhs.row_original_index)
    assert torch.equal(lhs.row_exposure_id, rhs.row_exposure_id)
    assert torch.equal(lhs.row_datum_index, rhs.row_datum_index)
    assert torch.allclose(lhs.samples.omega_xyz, rhs.samples.omega_xyz, atol=1e-6, rtol=0.0)
    assert torch.allclose(lhs.samples.loglam, rhs.samples.loglam, atol=1e-6, rtol=0.0)
    assert torch.allclose(lhs.samples.w, rhs.samples.w, atol=1e-8, rtol=0.0)
    assert torch.allclose(lhs.samples.Z, rhs.samples.Z, atol=1e-8, rtol=0.0)


def test_kernel_factory_is_invariant_to_input_index_order() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = _make_batch(device)
    ids_a = [0, 1, 2, 3, 4]
    ids_b = [4, 2, 0, 3, 1]
    idx_a = torch.tensor(ids_a, dtype=torch.long, device=device)
    idx_b = torch.tensor(ids_b, dtype=torch.long, device=device)
    factory_a = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=123,
    )
    factory_b = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=123,
    )
    samples_a = factory_a.build_kernel_samples_for_indices(idx_a, k=8, device=device)
    samples_b = factory_b.build_kernel_samples_for_indices(idx_b, k=8, device=device)
    raw_a = _restore_call_order(samples_a)
    raw_b = _restore_call_order(samples_b)

    omega_b = _reorder_to_ids(raw_b.omega_xyz, ids_b, ids_a)
    loglam_b = _reorder_to_ids(raw_b.loglam, ids_b, ids_a)
    w_b = _reorder_to_ids(raw_b.w, ids_b, ids_a)
    z_b = _reorder_to_ids(raw_b.Z, ids_b, ids_a)
    assert torch.allclose(raw_a.omega_xyz, omega_b, atol=1e-6, rtol=0.0)
    assert torch.allclose(raw_a.loglam, loglam_b, atol=1e-6, rtol=0.0)
    assert torch.allclose(raw_a.w, w_b, atol=1e-8, rtol=0.0)
    assert torch.allclose(raw_a.Z, z_b, atol=1e-8, rtol=0.0)


def test_kernel_factory_plan_reuse_matches_direct_build_sequence() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = _make_batch(device)
    idx = torch.tensor([4, 2, 0, 3, 1], dtype=torch.long, device=device)
    factory_direct = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=321,
    )
    factory_plan = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=321,
    )

    direct_a = factory_direct.build_kernel_samples_for_indices(idx, k=8, device=device)
    direct_b = factory_direct.build_kernel_samples_for_indices(idx, k=8, device=device)

    plan = factory_plan.build_kernel_sample_plan_for_indices(idx)
    plan_a = factory_plan.build_kernel_samples_from_plan(plan, k=8, device=device)
    plan_b = factory_plan.build_kernel_samples_from_plan(plan, k=8, device=device)

    _assert_grouped_samples_equal(plan_a, direct_a)
    _assert_grouped_samples_equal(plan_b, direct_b)


def test_kernel_factory_plan_sorts_by_exposure_kind_index_and_position() -> None:
    device = torch.device("cpu")
    batch = DatumBatch(
        y=torch.zeros(6, dtype=torch.float32, device=device),
        sigma=torch.ones(6, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.long, device=device),
        datum_kind=torch.tensor(
            [
                int(DatumKind.SPECTRAL_BIN), int(DatumKind.SPECTRAL_BIN), int(DatumKind.IMAGE_PIXEL),
                int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN), int(DatumKind.IMAGE_PIXEL),
            ],
            dtype=torch.long,
            device=device,
        ),
        datum_index=torch.tensor([2, 1, 0, 5, 2, 1], dtype=torch.long, device=device),
        kernel_samples=None,
    )
    factory = KernelFactoryFromBuilders(
        batch=batch, exposures=_make_exposures(), image_band_by_exposure={0: "r"}, rng_seed=123,
    )

    plan = factory.build_kernel_sample_plan_for_indices(torch.arange(6, dtype=torch.long, device=device))

    np.testing.assert_array_equal(plan.order, np.array([5, 3, 1, 2, 0, 4]))
    np.testing.assert_array_equal(plan.sorted_exp, np.array([0, 0, 0, 1, 1, 1]))
    np.testing.assert_array_equal(
        plan.sorted_kind,
        np.array(
            [
                int(DatumKind.IMAGE_PIXEL), int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN),
                int(DatumKind.IMAGE_PIXEL), int(DatumKind.SPECTRAL_BIN), int(DatumKind.SPECTRAL_BIN),
            ]
        ),
    )
    np.testing.assert_array_equal(plan.sorted_idx, np.array([1, 5, 1, 0, 2, 2]))
    assert plan.exposure_blocks == ((0, 0, 3), (1, 3, 6))
    assert plan.sample_groups == (
        (0, int(DatumKind.IMAGE_PIXEL), 0, 2), (0, int(DatumKind.SPECTRAL_BIN), 2, 3),
        (1, int(DatumKind.IMAGE_PIXEL), 3, 4), (1, int(DatumKind.SPECTRAL_BIN), 4, 6),
    )


def test_kernel_factory_enforces_image_index_bounds() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = DatumBatch(
        y=torch.zeros(1, dtype=torch.float32, device=device),
        sigma=torch.ones(1, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0], dtype=torch.long, device=device),
        datum_kind=torch.tensor([int(DatumKind.IMAGE_PIXEL)], dtype=torch.long, device=device),
        datum_index=torch.tensor([99], dtype=torch.long, device=device), kernel_samples=None,
    )
    factory = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=9,
    )
    with pytest.raises(ValueError, match="out of bounds"):
        factory.build_kernel_samples_for_indices(
            torch.tensor([0], dtype=torch.long, device=device), k=4, device=device,
        )


def test_kernel_factory_spectral_psf_broadens_sky_disk_sampling() -> None:
    device = torch.device("cpu")
    wcs = TangentPlaneWCS(ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=0.4, width=2, height=2)
    exposure_no_psf = ExposureState(
        wcs=wcs, obs_wcs=None, psf_fwhm_arcsec=0.0, lsf_fwhm_angstrom=np.array([2.0], dtype=np.float64),
        bandpasses={}, lam_spec=np.array([5000.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32), pixel_scale_arcsec=0.4, image_height=None,
        image_width=None, lam_edges=np.array([[4950.0, 5050.0]], dtype=np.float64),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=300.0),
    )
    exposure_with_psf = ExposureState(
        wcs=wcs, obs_wcs=None, psf_fwhm_arcsec=150.0, lsf_fwhm_angstrom=np.array([2.0], dtype=np.float64),
        bandpasses={}, lam_spec=np.array([5000.0], dtype=np.float64),
        fiber_mask=np.ones((2, 2), dtype=np.float32), pixel_scale_arcsec=0.4, image_height=None,
        image_width=None, lam_edges=np.array([[4950.0, 5050.0]], dtype=np.float64),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=300.0),
    )
    batch = DatumBatch(
        y=torch.zeros(1, dtype=torch.float32, device=device),
        sigma=torch.ones(1, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0], dtype=torch.long, device=device),
        datum_kind=torch.tensor([int(DatumKind.SPECTRAL_BIN)], dtype=torch.long, device=device),
        datum_index=torch.tensor([0], dtype=torch.long, device=device), kernel_samples=None,
    )
    idx_many = torch.zeros(256, dtype=torch.long, device=device)
    center = radec_deg_to_omega_xyz(np.array([180.0], dtype=np.float64), np.array([0.0], dtype=np.float64))[0]

    fac_no = KernelFactoryFromBuilders(
        batch=batch, exposures={0: exposure_no_psf}, image_band_by_exposure={}, rng_seed=17,
    )
    fac_yes = KernelFactoryFromBuilders(
        batch=batch, exposures={0: exposure_with_psf}, image_band_by_exposure={}, rng_seed=17,
    )
    sam_no = fac_no.build_kernel_samples_for_indices(idx_many, k=8, device=device)
    sam_yes = fac_yes.build_kernel_samples_for_indices(idx_many, k=8, device=device)

    dots_no = np.clip(
        np.sum(sam_no.samples.omega_xyz.detach().cpu().numpy() * center[None, None,: ], axis=-1), -1.0, 1.0,
    )
    dots_yes = np.clip(
        np.sum(sam_yes.samples.omega_xyz.detach().cpu().numpy() * center[None, None,: ], axis=-1), -1.0, 1.0,
    )
    mean_no = float(np.mean(np.arccos(dots_no)))
    mean_yes = float(np.mean(np.arccos(dots_yes)))
    assert mean_yes > mean_no


def test_kernel_factory_caches_spectral_samplers_per_exposure() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = _make_batch(device)
    factory = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=33,
    )
    idx = torch.tensor([2, 3], dtype=torch.long, device=device)
    factory.build_kernel_samples_for_indices(idx, k=4, device=device)
    lam_sampler_first = factory._spectral_lam_sampler_by_exposure[1]
    lsf_first = factory._spectral_lsf_by_exposure[1]
    aperture_first = factory._spectral_aperture_sampler_by_exposure[1]
    factory.build_kernel_samples_for_indices(idx, k=4, device=device)
    assert lam_sampler_first is factory._spectral_lam_sampler_by_exposure[1]
    assert lsf_first is factory._spectral_lsf_by_exposure[1]
    assert aperture_first is factory._spectral_aperture_sampler_by_exposure[1]


def test_kernel_factory_builds_fused_independent_replica_pair() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = _make_batch(device)
    idx = torch.tensor([4, 0, 2, 1, 3], dtype=torch.long, device=device)
    factory_pair = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=55,
    )
    plan = factory_pair.build_kernel_sample_plan_for_indices(idx)
    samples_a, samples_b = factory_pair.build_kernel_sample_pair_from_plan(plan, k=4, device=device)
    assert samples_a.samples.omega_xyz.shape == (5, 4, 3)
    assert samples_b.samples.omega_xyz.shape == (5, 4, 3)
    assert torch.allclose(samples_a.samples.w.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose(samples_b.samples.w.sum(dim=1), torch.ones(5), atol=1e-6)
    assert not torch.allclose(samples_a.samples.omega_xyz, samples_b.samples.omega_xyz)

    factory_fused = KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=55,
    )
    fused = factory_fused.build_kernel_samples_from_plan(plan, k=8, device=device)
    expected_a, expected_b = factory_fused._split_kernel_sample_replicas(fused.samples, 4)
    assert torch.allclose(samples_a.samples.omega_xyz, expected_a.omega_xyz, atol=1e-6)
    assert torch.allclose(samples_b.samples.omega_xyz, expected_b.omega_xyz, atol=1e-6)
    assert torch.allclose(samples_a.samples.loglam, expected_a.loglam, atol=1e-6)
    assert torch.allclose(samples_b.samples.loglam, expected_b.loglam, atol=1e-6)
    assert torch.allclose(samples_a.samples.w, expected_a.w, atol=1e-8)
    assert torch.allclose(samples_b.samples.w, expected_b.w, atol=1e-8)


class BadImageBuilder:
    def build(
        self, exposure, datum_index, *, k, rng, bandpass_sampler, device, wcs_backend, psf, validate_bounds,
    ):
        n = int(np.asarray(datum_index).size)
        omega = torch.zeros((n, k, 3), dtype=torch.float32, device=device)
        loglam = torch.zeros((n, k), dtype=torch.float32, device=device)
        w = torch.full((n, k), 1.0 / float(k), dtype=torch.float32, device=device)
        z = torch.ones((n,), dtype=torch.float32, device=device)
        return KernelSamples(omega_xyz=omega, loglam=loglam, w=w, Z=z)


def test_kernel_factory_validate_flag_can_be_disabled() -> None:
    device = torch.device("cpu")
    exposures = _make_exposures()
    batch = DatumBatch(
        y=torch.zeros(1, dtype=torch.float32, device=device),
        sigma=torch.ones(1, dtype=torch.float32, device=device),
        exposure_id=torch.tensor([0], dtype=torch.long, device=device),
        datum_kind=torch.tensor([int(DatumKind.IMAGE_PIXEL)], dtype=torch.long, device=device),
        datum_index=torch.tensor([0], dtype=torch.long, device=device), kernel_samples=None,
    )
    idx = torch.tensor([0], dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="unit-normalized"):
        KernelFactoryFromBuilders(
            batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=1,
            image_builder=BadImageBuilder(),
        ).build_kernel_samples_for_indices(idx, k=4, device=device)

    KernelFactoryFromBuilders(
        batch=batch, exposures=exposures, image_band_by_exposure={0: "r"}, rng_seed=1,
        image_builder=BadImageBuilder(), validate=False,
    ).build_kernel_samples_for_indices(idx, k=4, device=device)
