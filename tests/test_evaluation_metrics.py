from __future__ import annotations

import json
import numpy as np
import pytest

from survey_fusion.evaluation.metrics import (
    BandMetrics, _bgsub_positive_flux, _flux_error_from_sums, _metric_key, _region_metric_key,
    _sort_metric_keys, _sort_region_metric_keys, compute_metrics, compute_region_metrics, log_metrics,
    log_region_metrics, metrics_to_serializable, region_metrics_to_serializable,
)


def _sample_metric(**overrides: float | int | str) -> BandMetrics:
    values: dict[str, float | int | str] = {
        "mse": 1.0, "mae": 1.0, "mean_relative_error": 1.0, "mae_signal": 1.0, "relative_mae_signal_pct": 1.0,
        "rmse": 1.0, "nrmse_pct": 1.0, "peak_error_pct": 1.0, "flux_error_pct": 1.0,
        "fiber_flux_error_pct": float("nan"), "bgsub_flux_error_pct": 1.0, "total_abs_flux_error_pct": 1.0,
        "max_abs_residual": 1.0, "max_abs_residual_pct": 1.0, "signal_fraction_pct": 50.0, "signal_pixels": 2,
        "total_pixels": 4, "observed_min": 0.0, "observed_max": 10.0, "predicted_min": 0.0,
        "predicted_max": 10.0, "status": "training",
    }
    values.update(overrides)
    return BandMetrics(**values)


def test_compute_metrics_computes_expected_values_and_skips_missing_entries() -> None:
    predictions = {
        1: {
            "g": np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64),
            "u": np.array([[99.0]], dtype=np.float64),
        },
        2: {"g": np.array([[9.0]], dtype=np.float64)},
    }
    observations = {1: {"g": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)},}
    fiber_mask = np.array([[True, False], [True, False]], dtype=bool)

    metrics = compute_metrics(
        predictions, observations, held_out_band="g", signal_threshold=0.5, fiber_mask=fiber_mask,
    )

    assert set(metrics.keys()) == {"e1:g"}
    m = metrics["e1:g"]
    assert m.status == "HELD-OUT"
    assert m.mse == pytest.approx(3.5)
    assert m.mae == pytest.approx(1.5)
    assert m.mean_relative_error == pytest.approx((0.0 + 0.5 + (2.0 / 3.0) + 0.75) / 4.0)
    assert m.rmse == pytest.approx(np.sqrt(3.5))
    assert m.nrmse_pct == pytest.approx(100.0 * np.sqrt(3.5) / 3.0)
    assert m.mae_signal == pytest.approx(2.0)
    assert m.relative_mae_signal_pct == pytest.approx(100.0 * 2.0 / 3.0)
    assert m.signal_pixels == 3
    assert m.total_pixels == 4
    assert m.signal_fraction_pct == pytest.approx(75.0)
    assert m.peak_error_pct == pytest.approx(75.0)
    assert m.max_abs_residual == pytest.approx(3.0)
    assert m.max_abs_residual_pct == pytest.approx(75.0)
    assert m.flux_error_pct == pytest.approx(60.0)
    assert m.fiber_flux_error_pct == pytest.approx(50.0)
    assert m.bgsub_flux_error_pct == pytest.approx(100.0)
    assert m.total_abs_flux_error_pct == pytest.approx(60.0)
    assert m.observed_min == pytest.approx(1.0)
    assert m.observed_max == pytest.approx(4.0)
    assert m.predicted_min == pytest.approx(1.0)
    assert m.predicted_max == pytest.approx(7.0)


def test_compute_metrics_returns_nan_signal_metrics_when_no_signal_pixels() -> None:
    predictions = {1: {"r": np.full((2, 2), 2.0, dtype=np.float64)}}
    observations = {1: {"r": np.full((2, 2), 1.0, dtype=np.float64)}}
    fiber_mask = np.zeros((2, 2), dtype=bool)

    metrics = compute_metrics(predictions, observations, signal_threshold=2.0, fiber_mask=fiber_mask)

    m = metrics["e1:r"]
    assert m.status == "training"
    assert m.signal_pixels == 0
    assert np.isnan(m.mae_signal)
    assert np.isnan(m.relative_mae_signal_pct)
    assert np.isnan(m.fiber_flux_error_pct)


def test_compute_metrics_partial_holdout_changes_band_status_label() -> None:
    predictions = {1: {"i": np.full((2, 2), 2.0, dtype=np.float64)}}
    observations = {1: {"i": np.full((2, 2), 1.0, dtype=np.float64)}}

    metrics = compute_metrics(predictions, observations, held_out_band="i", partial_holdout_enabled=True)

    assert metrics["e1:i"].status == "PARTIAL-HOLDOUT-TRAINING-BAND"


def test_compute_metrics_background_subtracted_signal_threshold_avoids_background_only_signal_mask() -> None:
    observations = {1: {"i": np.array([[1.0, 1.0], [1.0, 20.0]], dtype=np.float64)},}
    predictions = {1: {"i": np.array([[1.0, 1.0], [1.0, 18.0]], dtype=np.float64)},}

    metrics = compute_metrics(
        predictions, observations, signal_threshold=0.05, background_subtracted_signal=True,
    )

    assert metrics["e1:i"].signal_pixels < metrics["e1:i"].total_pixels
    assert metrics["e1:i"].signal_fraction_pct < 100.0


def test_compute_metrics_rejects_invalid_fiber_masks() -> None:
    predictions = {1: {"g": np.ones((2, 2), dtype=np.float64)}}
    observations = {1: {"g": np.ones((2, 2), dtype=np.float64)}}

    with pytest.raises(ValueError, match="fiber_mask must be a 2D array"):
        compute_metrics(predictions, observations, fiber_mask=np.array([True, False]))

    with pytest.raises(ValueError, match="does not match observation shape"):
        compute_metrics(predictions, observations, fiber_mask=np.ones((3, 3), dtype=bool))


def test_compute_metrics_rejects_prediction_observation_shape_mismatch() -> None:
    predictions = {1: {"g": np.ones((1, 2), dtype=np.float64)}}
    observations = {1: {"g": np.ones((2, 1), dtype=np.float64)}}
    with pytest.raises(ValueError, match="prediction shape .* does not match observation shape"):
        compute_metrics(predictions, observations)


def test_compute_metrics_supports_per_exposure_fiber_mask_dict() -> None:
    predictions = {
        1: {"g": np.array([[3.0, 1.0]], dtype=np.float64)},
        2: {"g": np.array([[2.0, 2.0]], dtype=np.float64)},
    }
    observations = {
        1: {"g": np.array([[1.0, 1.0]], dtype=np.float64)},
        2: {"g": np.array([[1.0, 1.0]], dtype=np.float64)},
    }
    fiber_masks = {1: np.array([[True, False]], dtype=bool)}

    metrics = compute_metrics(predictions, observations, fiber_mask=fiber_masks)

    assert metrics["e1:g"].fiber_flux_error_pct == pytest.approx(200.0)
    assert np.isnan(metrics["e2:g"].fiber_flux_error_pct)


def test_compute_metrics_ignores_non_finite_pixels() -> None:
    predictions = {1: {"r": np.array([[2.0, np.nan], [4.0, 8.0]], dtype=np.float64)}}
    observations = {1: {"r": np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float64)}}

    metrics = compute_metrics(predictions, observations)

    m = metrics["e1:r"]
    assert m.total_pixels == 2
    assert m.mse == pytest.approx(((2.0 - 1.0) ** 2 + (8.0 - 4.0) ** 2) / 2.0)
    assert m.observed_min == pytest.approx(1.0)
    assert m.observed_max == pytest.approx(4.0)


def test_helper_functions_and_key_sorting() -> None:
    assert _metric_key(3, "r") == "e3:r"
    assert _flux_error_from_sums(12.0, 10.0) == pytest.approx(20.0)

    mixed = np.array([np.nan, -1.0, 3.0, np.inf], dtype=np.float64)
    assert _bgsub_positive_flux(mixed) == pytest.approx(2.0)
    assert _bgsub_positive_flux(np.array([np.nan, np.inf, -np.inf], dtype=np.float64)) == 0.0

    keys = ["e2:z", "e1:r", "x:q", "e1:g", "e1:x"]
    assert _sort_metric_keys(keys) == ["x:q", "e1:g", "e1:r", "e1:x", "e2:z"]


def test_metrics_to_serializable_and_log_metrics_output(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = {
        "e1:r": _sample_metric(status="training", fiber_flux_error_pct=float("nan")),
        "e1:g": _sample_metric(status="HELD-OUT", fiber_flux_error_pct=12.0),
    }

    serialized = metrics_to_serializable(metrics)
    assert serialized["e1:g"]["status"] == "HELD-OUT"
    assert serialized["e1:r"]["signal_pixels"] == 2
    assert serialized["e1:r"]["fiber_flux_error_pct"] is None
    json.dumps(serialized, allow_nan=False)

    log_metrics(metrics, held_out_band="g")
    out = capsys.readouterr().out
    assert "QUANTITATIVE ERROR ANALYSIS" in out
    assert "Exposure e1 Band g (HELD-OUT):" in out
    assert "Exposure e1 Band r (training):" in out
    assert out.index("Exposure e1 Band g") < out.index("Exposure e1 Band r")
    assert "*** HELD-OUT ***" in out
    assert "FLUX-WEIGHTED ERRORS:" in out
    assert "(HELD-OUT)" in out
    assert "(training)" in out
    assert "fiber:" in out


def test_log_metrics_uses_metric_status_for_partial_holdout(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = {"e1:i": _sample_metric(status="PARTIAL-HOLDOUT-TRAINING-BAND")}

    log_metrics(metrics, held_out_band="i")
    out = capsys.readouterr().out
    assert "Exposure e1 Band i (PARTIAL-HOLDOUT-TRAINING-BAND):" in out
    assert "*** PARTIAL-HOLDOUT-TRAINING-BAND ***" in out
    assert "(PARTIAL-HOLDOUT-TRAINING-BAND)" in out


def test_log_metrics_noop_on_empty(capsys: pytest.CaptureFixture[str]) -> None:
    log_metrics({})
    assert capsys.readouterr().out == ""


def test_compute_region_metrics_and_serialization() -> None:
    predictions = {
        1: {
            "i": np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float64),
            "g": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        }
    }
    observations = {
        1: {
            "i": np.array([[1.0, 5.0], [1.0, 2.0]], dtype=np.float64),
            "g": np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float64),
        }
    }
    region_masks = {1: {"R2": np.array([[True, False], [False, True]], dtype=bool)}}
    metrics = compute_region_metrics(
        predictions, observations, region_masks, held_out_band="i", primary_region="R2",
    )
    assert set(metrics.keys()) == {"region=R2|e1:i", "region=R2|e1:g"}
    assert metrics["region=R2|e1:i"].status == "PRIMARY_HELD-OUT"
    assert metrics["region=R2|e1:g"].status == "PRIMARY"
    assert metrics["region=R2|e1:i"].mse == pytest.approx(1.0)
    serial = region_metrics_to_serializable(metrics)
    assert serial["region=R2|e1:i"]["status"] == "PRIMARY_HELD-OUT"
    json.dumps(serial, allow_nan=False)


def test_compute_region_metrics_ignores_non_finite_pixels() -> None:
    predictions = {1: {"r": np.array([[2.0, np.nan], [4.0, 8.0]], dtype=np.float64)}}
    observations = {1: {"r": np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float64)}}
    region_masks = {1: {"R": np.ones((2, 2), dtype=bool)}}

    metrics = compute_region_metrics(predictions, observations, region_masks)

    m = metrics["region=R|e1:r"]
    assert m.total_pixels == 2
    assert m.mse == pytest.approx(((2.0 - 1.0) ** 2 + (8.0 - 4.0) ** 2) / 2.0)


def test_compute_region_metrics_partial_holdout_distinguishes_overlap() -> None:
    predictions = {1: {"i": np.array([[2.0, 4.0], [1.0, 3.0]], dtype=np.float64), }}
    observations = {1: {"i": np.array([[1.0, 5.0], [1.0, 2.0]], dtype=np.float64), }}
    region_masks = {
        1: {
            "R1": np.array([[True, False], [False, False]], dtype=bool),
            "R2": np.array([[True, True], [False, False]], dtype=bool),
            "R3": np.array([[False, False], [True, True]], dtype=bool),
        }
    }
    holdout_masks = {1: np.array([[True, False], [False, False]], dtype=bool),}

    metrics = compute_region_metrics(
        predictions, observations, region_masks, held_out_band="i", primary_region="R2",
        partial_holdout_enabled=True, holdout_mask_by_exposure=holdout_masks,
    )

    assert metrics["region=R1|e1:i"].status == "HOLDOUT-REGION"
    assert metrics["region=R2|e1:i"].status == "PRIMARY_MIXED-HOLDOUT-REGION"
    assert metrics["region=R3|e1:i"].status == "PARTIAL-HOLDOUT-TRAINING-BAND"


def test_region_metric_helpers_and_logging(capsys: pytest.CaptureFixture[str]) -> None:
    assert _region_metric_key("R2", 3, "i") == "region=R2|e3:i"
    sorted_keys = _sort_region_metric_keys(
        ["region=R1|e1:r", "region=R2|e1:i", "region=R2|e1:g"],
        primary_region="R2",
        held_out_band="i",
    )
    assert sorted_keys[0] == "region=R2|e1:i"
    metrics = {
        "region=R2|e1:i": _sample_metric(status="PRIMARY_HELD-OUT"),
        "region=R1|e1:i": _sample_metric(status="HELD-OUT"),
    }
    log_region_metrics(metrics, held_out_band="i", primary_region="R2")
    out = capsys.readouterr().out
    assert "REGION-LEVEL ERROR ANALYSIS" in out
    assert "Region R2 [PRIMARY]" in out


def test_bgsub_flux_error_uses_joint_valid_population() -> None:
    observations = {1: {"r": np.array([[0.0, 10.0], [np.nan, 0.0]], dtype=np.float64)}}
    predictions = {1: {"r": np.array([[0.0, 10.0], [100.0, 0.0]], dtype=np.float64)}}

    metrics = compute_metrics(predictions, observations)

    assert metrics["e1:r"].bgsub_flux_error_pct == pytest.approx(0.0)


def test_relative_mae_signal_uses_background_subtracted_denominator() -> None:
    observation = np.full((5, 5), 100.0, dtype=np.float64)
    observation[2, 2] = 120.0
    prediction = observation.copy()
    prediction[2, 2] = 110.0

    metrics = compute_metrics(
        {1: {"r": prediction}}, {1: {"r": observation}},
        signal_threshold=0.5, background_subtracted_signal=True,
    )

    # source peak 20 above the 100 background, residual 10: 50% source-relative
    # error, not 10/120 = 8.3% against the background-bearing total
    assert metrics["e1:r"].relative_mae_signal_pct == pytest.approx(50.0)
