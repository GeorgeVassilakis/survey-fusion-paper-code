from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_holdout_exposures.py"
    spec = importlib.util.spec_from_file_location("survey_fusion_eval_holdout_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verdict_assembly_always_records_operator_provenance() -> None:
    evaluation = {
        "psf_model": "gaussian",
        "apply_spatial_psf": True,
        "pair_n_sigma": 8.0,
    }

    report = _load_eval_module().assemble_verdict(
        labels=["r-held-out"],
        evaluating=evaluation,
        checkpoint={},
        source_commit="synthetic-evaluator-commit",
    )

    provenance = report["operator_provenance"]
    assert provenance is not None
    assert set(provenance) == {
        "evaluating", "trained", "evaluator_source_commit",
    }
    assert provenance["evaluating"] == evaluation
    assert provenance["trained"] == "unrecorded_by_checkpoint_format"
    assert provenance["evaluator_source_commit"] == (
        "synthetic-evaluator-commit"
    )


def test_score_exposure_reports_flux_ratio_numerator_and_denominator() -> None:
    module = _load_eval_module()
    model_source = np.array([[2.0, 4.0], [8.0, 16.0]], dtype=np.float64)
    obs_source = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float64)
    trainer = SimpleNamespace(
        render_datum_image_window_deterministic=lambda *args, **kwargs: model_source,
        _global_indices_for=lambda *args: torch.arange(4),
    )
    store = SimpleNamespace(
        exposures={
            7: SimpleNamespace(
                image_height=2, image_width=2, psf_fwhm_arcsec=1.3,
            )
        },
        image_source_by_exposure={7: obs_source},
        image_band_by_exposure={7: "r"},
        batch=SimpleNamespace(
            datum_index=torch.arange(4), sigma=torch.ones(4),
        ),
    )

    row = module.score_exposure(trainer, store, 7, chunk_size=16)

    preexisting_keys = {
        "exposure_id", "band", "psf_fwhm_arcsec", "n_valid_pixels",
        "mean_square_pull", "median_abs_pull", "p95_abs_pull",
        "flux_ratio_model_over_obs", "_arrays",
    }
    assert preexisting_keys <= row.keys()
    assert row["model_total_flux"] == float(model_source.sum())
    assert row["obs_total_flux"] == float(obs_source.sum())
    assert row["flux_ratio_model_over_obs"] == pytest.approx(
        row["model_total_flux"] / row["obs_total_flux"]
    )


def test_operator_identity_resolves_labels_from_manifest(tmp_path) -> None:
    import json

    from omegaconf import OmegaConf

    module = _load_eval_module()
    manifest = {
        "image_exposures": [
            {"label": "g-000001-1-0001", "band": "g"},
            {"label": "r-000001-1-0001", "band": "r"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    cfg = OmegaConf.create({
        "physics": {"bandpasses": {"r": "sdss_r"}},
        "data": {"scene_manifest_path": str(manifest_path)},
    })
    store = SimpleNamespace(
        exposures={
            1: SimpleNamespace(psf_fwhm_arcsec=1.4, pixel_scale_arcsec=0.396),
        },
        image_band_by_exposure={1: "r"},
    )
    trainer = SimpleNamespace(
        apply_spatial_psf=False,
        analytic_images_enabled=True,
        analytic_image_quadrature_order=3,
        pair_n_sigma=8.0,
    )
    identity = module.operator_identity_for_evaluation(cfg, trainer, store, [1])
    assert identity["psf_parameters_by_exposure"] == [{
        "exposure_id": 1,
        "label": "r-000001-1-0001",
        "psf_fwhm_arcsec": 1.4,
        "pixel_scale_arcsec": 0.396,
    }]

    store.image_band_by_exposure[1] = "g"
    with pytest.raises(ValueError, match="band mismatch"):
        module.operator_identity_for_evaluation(cfg, trainer, store, [1])


FRAME_HEIGHT = 10
FRAME_WIDTH = 20
PIXEL_SCALE_ARCSEC = 0.396


def _tan_wcs(*, x_shift_pixels: float = 0.0, y_shift_pixels: float = 0.0,
             ra_deg: float = 150.0, dec_deg: float = 2.0):
    """TAN WCS on a shared tangent point: shifting crpix by N pixels shifts the
    pixel grid by exactly N pixels, so footprint overlaps are analytic."""
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600.0, PIXEL_SCALE_ARCSEC / 3600.0]
    wcs.wcs.crpix = [
        0.5 * FRAME_WIDTH + 0.5 - x_shift_pixels,
        0.5 * FRAME_HEIGHT + 0.5 - y_shift_pixels,
    ]
    return wcs


def _tan_sip_wcs():
    from astropy.wcs import Sip, WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600.0, PIXEL_SCALE_ARCSEC / 3600.0]
    wcs.wcs.crpix = [0.5 * FRAME_WIDTH + 0.5, 0.5 * FRAME_HEIGHT + 0.5]
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3, 3), dtype=np.float64)
    ap = np.zeros((3, 3), dtype=np.float64)
    bp = np.zeros((3, 3), dtype=np.float64)
    a[2, 0] = 0.02
    b[0, 2] = -0.013
    wcs.sip = Sip(a, b, ap, bp, np.asarray(wcs.wcs.crpix))
    return wcs


def _exposure(wcs, *, dx_arcsec: float = 0.0, dy_arcsec: float = 0.0):
    from survey_fusion.utils.wcs import AstropyWCSAdapter

    return SimpleNamespace(
        image_height=FRAME_HEIGHT,
        image_width=FRAME_WIDTH,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        psf_fwhm_arcsec=1.3,
        obs_wcs=AstropyWCSAdapter(wcs, dx_arcsec=dx_arcsec, dy_arcsec=dy_arcsec),
    )


def _coverage_store():
    """Target exposure 0 plus three same-band exposures: one covering columns
    x >= 8, one covering rows y >= 4, one a degree away, and one in another
    band that overlaps the target exactly."""
    exposures = {
        0: _exposure(_tan_wcs()),
        1: _exposure(_tan_wcs(x_shift_pixels=8.0)),
        2: _exposure(_tan_wcs(y_shift_pixels=4.0)),
        3: _exposure(_tan_wcs(ra_deg=151.0)),
        4: _exposure(_tan_wcs()),
    }
    bands = {0: "z", 1: "z", 2: "z", 3: "z", 4: "r"}
    return SimpleNamespace(exposures=exposures, image_band_by_exposure=bands)


def test_coverage_depth_counts_analytic_footprint_overlaps() -> None:
    module = _load_eval_module()
    store = _coverage_store()

    depth = module.build_coverage_depth_map(store, 0, [1, 2, 3])

    assert depth.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert np.all(depth[:4, :8] == 0)
    assert np.all(depth[:4, 8:] == 1)
    assert np.all(depth[4:, :8] == 1)
    assert np.all(depth[4:, 8:] == 2)
    assert int((depth == 0).sum()) == 32


def test_coverage_depth_is_zero_without_sources() -> None:
    module = _load_eval_module()
    store = _coverage_store()

    depth = module.build_coverage_depth_map(store, 0, [])

    assert depth.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert not np.any(depth)


def test_coverage_depth_refuses_self_coverage() -> None:
    module = _load_eval_module()
    store = _coverage_store()

    with pytest.raises(ValueError, match="its own coverage source set"):
        module.build_coverage_depth_map(store, 0, [0, 1])


def test_coverage_sources_are_same_band_trained_and_leave_one_out() -> None:
    module = _load_eval_module()
    store = _coverage_store()

    sources = module.coverage_source_exposure_ids(store, 0, {0, 1, 3, 4})

    assert sources == [1, 3]


def test_sky_to_pixel_inverts_the_pointing_offset() -> None:
    module = _load_eval_module()
    exposure = _exposure(_tan_wcs(), dx_arcsec=1.5, dy_arcsec=-0.7)
    x_pixel = np.array([0.0, 5.0, 19.0])
    y_pixel = np.array([0.0, 3.0, 9.0])
    omega = np.asarray(exposure.obs_wcs.pixel_to_omega_xyz(x_pixel, y_pixel))
    ra, dec = module._omega_to_radec_deg(omega)

    x_back, y_back = module.sky_to_pixel(exposure, 0, ra, dec)

    assert np.allclose(x_back, x_pixel, atol=1e-6)
    assert np.allclose(y_back, y_pixel, atol=1e-6)


def test_sip_inverse_roundtrip_and_coverage_membership() -> None:
    module = _load_eval_module()
    target = _exposure(_tan_sip_wcs())
    source = _exposure(_tan_sip_wcs())
    x_pixel = np.array([19.0])
    y_pixel = np.array([9.0])
    omega = np.asarray(target.obs_wcs.pixel_to_omega_xyz(x_pixel, y_pixel))
    ra, dec = module._omega_to_radec_deg(omega)

    x_back, y_back = module.sky_to_pixel(source, 1, ra, dec)

    assert float(np.hypot(x_back[0] - x_pixel[0], y_back[0] - y_pixel[0])) < 0.01
    store = SimpleNamespace(
        exposures={0: target, 1: source},
        image_band_by_exposure={0: "z", 1: "z"},
    )
    depth = module.build_coverage_depth_map(store, 0, [1])
    assert depth[-1, -1] == 1


def test_distortion_free_tan_coverage_mask_is_unchanged_bitwise() -> None:
    module = _load_eval_module()
    store = _coverage_store()

    depth = module.build_coverage_depth_map(store, 0, [1, 2, 3])
    expected = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int32)
    expected[:4, 8:] += 1
    expected[4:, :8] += 1
    expected[4:, 8:] += 2

    assert np.array_equal(depth, expected)


def test_sip_inverse_divergence_names_exposure() -> None:
    module = _load_eval_module()

    class DivergingWCS:
        sip = object()

        def all_world2pix(self, *args, **kwargs):
            assert kwargs == {
                "tolerance": 1.0e-10,
                "maxiter": 100,
                "detect_divergence": True,
                "quiet": False,
            }
            raise RuntimeError("iteration diverged")

        def wcs_world2pix(self, *args, **kwargs):
            raise AssertionError("distorted WCS must not fall back to core inverse")

    exposure = SimpleNamespace(
        obs_wcs=SimpleNamespace(wcs=DivergingWCS(), dx_arcsec=0.0, dy_arcsec=0.0),
    )
    with pytest.raises(RuntimeError, match=r"Exposure 23.*all_world2pix"):
        module.sky_to_pixel(exposure, 23, np.array([150.0]), np.array([2.0]))


def test_coverage_depth_refuses_wcs_without_an_inverse() -> None:
    module = _load_eval_module()
    store = _coverage_store()
    store.exposures[1].obs_wcs = SimpleNamespace(
        wcs=SimpleNamespace(),
        pixel_to_omega_xyz=store.exposures[1].obs_wcs.pixel_to_omega_xyz,
    )

    with pytest.raises(RuntimeError, match="--skip-coverage-strata"):
        module.build_coverage_depth_map(store, 0, [1])


def _stratification_arrays():
    """Left half of the frame is the hole, where the model renders 3x the
    observation; the covered half renders it exactly."""
    obs = np.full((4, 4), 2.0)
    sigma = np.ones((4, 4))
    depth = np.zeros((4, 4), dtype=np.int32)
    depth[:, 2:] = 1
    model = obs.copy()
    model[:, :2] *= 3.0
    return model, obs, sigma, depth


def test_stratified_flux_metrics_splits_a_known_hole() -> None:
    module = _load_eval_module()
    model, obs, sigma, depth = _stratification_arrays()

    metrics = module.stratified_flux_metrics(model, obs, sigma, depth, snr_threshold=1.0)

    assert metrics["n_valid_pixels"] == 16
    assert metrics["n_hole_pixels"] == 8
    assert metrics["n_covered_pixels"] == 8
    assert metrics["hole_fraction"] == pytest.approx(0.5)
    assert metrics["max_depth"] == 1
    assert metrics["depth_histogram"] == {"0": 8, "1": 8}
    assert metrics["strata"]["hole"]["flux_ratio_model_over_obs"] == pytest.approx(3.0)
    assert metrics["strata"]["covered"]["flux_ratio_model_over_obs"] == pytest.approx(1.0)
    assert metrics["strata"]["all"]["flux_ratio_model_over_obs"] == pytest.approx(2.0)
    assert metrics["strata"]["hole"]["median_pixel_flux_ratio"] == pytest.approx(3.0)
    assert metrics["strata"]["covered"]["median_pixel_flux_ratio"] == pytest.approx(1.0)
    assert metrics["strata"]["hole"]["mean_square_pull"] == pytest.approx(16.0)
    assert metrics["strata"]["covered"]["mean_square_pull"] == pytest.approx(0.0)
    # numerator over every valid pixel, denominator over the covered ones only
    assert metrics["flux_ratio_masked_denominator"] == pytest.approx(64.0 / 16.0)


def test_stratified_all_stratum_reproduces_the_whole_frame_ratio() -> None:
    module = _load_eval_module()
    model, obs, sigma, depth = _stratification_arrays()
    sigma[0, 0] = np.nan

    metrics = module.stratified_flux_metrics(model, obs, sigma, depth)
    valid = np.isfinite(sigma)

    assert metrics["n_valid_pixels"] == 15
    assert metrics["strata"]["all"]["flux_ratio_model_over_obs"] == pytest.approx(
        float(model[valid].sum() / obs[valid].sum())
    )
    assert metrics["strata"]["all"]["median_abs_pull"] == pytest.approx(
        float(np.median(np.abs((model[valid] - obs[valid]) / sigma[valid])))
    )


def test_stratified_flux_metrics_reports_empty_strata_without_inventing_numbers() -> None:
    module = _load_eval_module()
    model, obs, sigma, depth = _stratification_arrays()
    depth[:] = 1

    metrics = module.stratified_flux_metrics(model, obs, sigma, depth)

    assert metrics["n_hole_pixels"] == 0
    assert metrics["hole_fraction"] == 0.0
    hole = metrics["strata"]["hole"]
    assert hole["n_pixels"] == 0
    assert hole["flux_ratio_model_over_obs"] is None
    assert hole["flux_ratio_model_over_obs_undefined_reason"] == "empty_stratum"
    assert hole["median_pixel_flux_ratio"] is None
    assert hole["median_pixel_flux_ratio_undefined_reason"] == "empty_stratum"
    assert hole["mean_square_pull"] is None
    assert hole["mean_square_pull_undefined_reason"] == "empty_stratum"
    assert hole["median_abs_pull"] is None
    assert hole["median_abs_pull_undefined_reason"] == "empty_stratum"


def test_empty_hole_report_is_strict_json_with_null_reason(tmp_path) -> None:
    module = _load_eval_module()
    model, obs, sigma, depth = _stratification_arrays()
    depth[:] = 1
    report = {"coverage": module.stratified_flux_metrics(model, obs, sigma, depth)}
    path = tmp_path / "report.json"

    module.write_report(report, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    hole = loaded["coverage"]["strata"]["hole"]
    assert hole["flux_ratio_model_over_obs"] is None
    assert hole["flux_ratio_model_over_obs_undefined_reason"] == "empty_stratum"
    assert "NaN" not in path.read_text(encoding="utf-8")
    summary = module.format_coverage_summary(
        "holdout",
        {"exposure_id": 7, "coverage": {
            **report["coverage"], "n_source_exposures": 1,
        }},
    )
    assert "n/a (empty_stratum)" in summary
    with pytest.raises(ValueError, match="Out of range float values"):
        module.write_report({"invalid": float("nan")}, path)


def test_stratified_undefined_reasons_cover_zero_denominator_and_no_significant_pixels() -> None:
    module = _load_eval_module()
    model = np.ones((2, 2), dtype=np.float64)
    obs = np.zeros((2, 2), dtype=np.float64)
    sigma = np.ones((2, 2), dtype=np.float64)
    depth = np.ones((2, 2), dtype=np.int32)

    metrics = module.stratified_flux_metrics(model, obs, sigma, depth)

    covered = metrics["strata"]["covered"]
    assert covered["flux_ratio_model_over_obs"] is None
    assert covered["flux_ratio_model_over_obs_undefined_reason"] == "zero_denominator"
    assert covered["median_pixel_flux_ratio"] is None
    assert covered["median_pixel_flux_ratio_undefined_reason"] == "no_significant_pixels"
    assert metrics["flux_ratio_masked_denominator"] is None
    assert metrics["flux_ratio_masked_denominator_undefined_reason"] == "zero_denominator"


def test_stratified_flux_metrics_rejects_bad_inputs() -> None:
    module = _load_eval_module()
    model, obs, sigma, depth = _stratification_arrays()

    with pytest.raises(ValueError, match="matching shapes"):
        module.stratified_flux_metrics(model, obs, sigma, depth[:, :2])
    with pytest.raises(ValueError, match="snr_threshold"):
        module.stratified_flux_metrics(model, obs, sigma, depth, snr_threshold=0.0)


def _stratified_trainer_and_store():
    store = _coverage_store()
    obs = np.full((FRAME_HEIGHT, FRAME_WIDTH), 2.0)
    model = obs.copy()
    model[:4, :8] *= 5.0
    store.image_source_by_exposure = {0: obs}
    store.batch = SimpleNamespace(
        datum_index=torch.arange(FRAME_HEIGHT * FRAME_WIDTH),
        sigma=torch.ones(FRAME_HEIGHT * FRAME_WIDTH),
    )
    trainer = SimpleNamespace(
        render_datum_image_window_deterministic=lambda *args, **kwargs: model,
        _global_indices_for=lambda *args: torch.arange(FRAME_HEIGHT * FRAME_WIDTH),
    )
    return trainer, store


def test_evaluate_exposure_stratified_adds_coverage_and_keeps_existing_keys() -> None:
    module = _load_eval_module()
    trainer, store = _stratified_trainer_and_store()

    row = module.evaluate_exposure_stratified(
        trainer, store, 0, trained_exposure_ids={1, 2, 3, 4}, chunk_size=16,
    )

    assert "_arrays" not in row
    assert row["flux_ratio_model_over_obs"] == pytest.approx(
        row["coverage"]["strata"]["all"]["flux_ratio_model_over_obs"]
    )
    coverage = row["coverage"]
    assert coverage["source_exposure_ids"] == [1, 2, 3]
    assert coverage["n_source_exposures"] == 3
    assert coverage["method"] == module.COVERAGE_METHOD
    assert coverage["n_hole_pixels"] == 32
    assert coverage["strata"]["hole"]["flux_ratio_model_over_obs"] == pytest.approx(5.0)
    assert coverage["strata"]["covered"]["flux_ratio_model_over_obs"] == pytest.approx(1.0)
    summary = module.format_coverage_summary("holdout", row)
    assert "hole 5.0000" in summary


def test_evaluate_exposure_stratified_can_keep_arrays_for_saving() -> None:
    module = _load_eval_module()
    trainer, store = _stratified_trainer_and_store()

    row = module.evaluate_exposure_stratified(
        trainer, store, 0, trained_exposure_ids={1, 2}, chunk_size=16, keep_arrays=True,
    )

    arrays = row["_arrays"]
    assert set(arrays) == {"model_source", "obs_source", "sigma", "coverage_depth"}
    assert arrays["coverage_depth"].shape == (FRAME_HEIGHT, FRAME_WIDTH)


def test_labels_by_exposure_id_keys_rows_independently_of_config_order(tmp_path) -> None:
    """The report's config-order `labels` list is not a positional key.

    Rows are emitted in sorted-exposure-id order, so a band holding more than
    one frame gets its results misattributed by anyone zipping the two. This
    pins the per-id resolution that makes zipping unnecessary.
    """
    import json

    module = _load_eval_module()
    manifest = {
        "image_exposures": [
            {"label": "z-000001-1-0001", "band": "z"},
            {"label": "u-000002-2-0002", "band": "u"},
            {"label": "u-000003-3-0003", "band": "u"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    data_cfg = {"scene_manifest_path": str(manifest_path)}
    store = SimpleNamespace(image_band_by_exposure={0: "z", 1: "u", 2: "u"})

    # config order deliberately unlike sorted-id order
    resolved = module.labels_by_exposure_id(data_cfg, [2, 0, 1], store)
    assert resolved == {
        0: "z-000001-1-0001",
        1: "u-000002-2-0002",
        2: "u-000003-3-0003",
    }
    for exp_id, label in resolved.items():
        assert label.split("-")[0] == store.image_band_by_exposure[exp_id]


def test_labels_by_exposure_id_rejects_manifest_store_band_mismatch(tmp_path) -> None:
    import json

    module = _load_eval_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(
        {"image_exposures": [{"label": "z-000001-1-0001", "band": "z"}]}
    ))
    store = SimpleNamespace(image_band_by_exposure={0: "u"})
    with pytest.raises(ValueError, match="band mismatch"):
        module.labels_by_exposure_id(
            {"scene_manifest_path": str(manifest_path)}, [0], store
        )


def test_labels_by_exposure_id_refuses_out_of_range_and_missing_manifest(tmp_path) -> None:
    import json

    module = _load_eval_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(
        {"image_exposures": [{"label": "z-000001-1-0001", "band": "z"}]}
    ))
    store = SimpleNamespace(image_band_by_exposure={0: "z", 5: "z"})
    with pytest.raises(ValueError, match="outside manifest range"):
        module.labels_by_exposure_id(
            {"scene_manifest_path": str(manifest_path)}, [5], store
        )
    with pytest.raises(ValueError, match="scene_manifest_path is required"):
        module.labels_by_exposure_id({}, [0], store)
