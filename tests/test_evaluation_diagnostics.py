from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

import survey_fusion.evaluation.diagnostics as diagnostics_mod


def _radial_step_image(center: float, mid: float, outer: float, size: int = 5) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    cy = cx = (size - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    image = np.full((size, size), outer, dtype=np.float64)
    image[rr < 1.0] = center
    image[(rr >= 1.0) & (rr < 2.0)] = mid
    return image


def test_log_radial_diagnostics_returns_early_for_empty_inputs(capsys: pytest.CaptureFixture[str],) -> None:
    diagnostics_mod.log_radial_diagnostics({}, {})
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("radial_bin_width", [0.0, -0.5])
def test_log_radial_diagnostics_rejects_nonpositive_bin_width(radial_bin_width: float) -> None:
    obs = {"g": np.ones((2, 2), dtype=np.float64)}
    pred = {"g": np.ones((2, 2), dtype=np.float64)}
    with pytest.raises(ValueError, match="radial_bin_width must be > 0"):
        diagnostics_mod.log_radial_diagnostics(obs, pred, radial_bin_width=radial_bin_width)


def test_log_radial_diagnostics_returns_early_when_no_radial_bins(
    capsys: pytest.CaptureFixture[str],
) -> None:
    obs = {"g": np.array([[1.0]], dtype=np.float64)}
    pred = {"g": np.array([[1.0]], dtype=np.float64)}
    diagnostics_mod.log_radial_diagnostics(obs, pred)
    assert capsys.readouterr().out == ""


def test_log_radial_diagnostics_handles_single_radial_bin(capsys: pytest.CaptureFixture[str],) -> None:
    obs = {"g": np.ones((2, 2), dtype=np.float64)}
    pred = {"g": np.ones((2, 2), dtype=np.float64)}
    diagnostics_mod.log_radial_diagnostics(obs, pred, radial_bin_width=2.0)
    out = capsys.readouterr().out
    assert "Band g:" in out
    assert "matched widths" in out


def test_log_radial_diagnostics_reports_sorted_bands_and_diagnostic_hints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    obs = {
        "x": _radial_step_image(8.0, 2.0, 0.0), "r": _radial_step_image(5.0, 3.0, 0.0),
        "u": _radial_step_image(2.0, 1.0, 0.0), "g": _radial_step_image(10.0, 1.0, 0.0),
        "i": _radial_step_image(8.0, 2.0, 0.0), "z": np.zeros((5, 5), dtype=np.float64),
    }
    pred = {
        "x": _radial_step_image(8.0, 2.0, 0.0), "r": _radial_step_image(9.0, 1.0, 0.0),
        "g": _radial_step_image(6.0, 4.0, 1.0), "i": _radial_step_image(8.0, 2.0, 0.0),
        "z": np.zeros((5, 5), dtype=np.float64),
    }

    diagnostics_mod.log_radial_diagnostics(obs, pred, held_out_band="r", radial_bin_width=1.0)
    out = capsys.readouterr().out

    assert "RADIAL PROFILE TEST (ring diagnostics) [obs-peak-centered]" in out
    assert "pred broader" in out
    assert "pred sharper" in out
    assert "matched widths" in out
    assert "insufficient signal for FWHM" in out
    assert "Peak Δ" in out
    assert "*** HELD-OUT BAND ***" in out
    assert "Band u:" not in out

    band_positions = [
        out.index("Band g:"),
        out.index("Band r:"),
        out.index("Band i:"),
        out.index("Band z:"),
        out.index("Band x:"),
    ]
    assert band_positions == sorted(band_positions)


def test_log_radial_diagnostics_reports_peak_offset_against_observed_center(
    capsys: pytest.CaptureFixture[str],
) -> None:
    obs = np.zeros((5, 5), dtype=np.float64)
    pred = np.zeros((5, 5), dtype=np.float64)
    obs[2, 2] = 10.0
    pred[2, 3] = 10.0

    diagnostics_mod.log_radial_diagnostics({"i": obs}, {"i": pred}, radial_bin_width=1.0)
    out = capsys.readouterr().out

    assert "Peak(obs)  = (2.0, 2.0) pix" in out
    assert "Peak(pred) = (2.0, 3.0) pix" in out
    assert "Peak Δ     = 1.000 pix" in out


def test_log_radial_diagnostics_partial_holdout_uses_partial_banner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = _radial_step_image(5.0, 3.0, 0.0)
    diagnostics_mod.log_radial_diagnostics(
        {"i": image}, {"i": image}, held_out_band="i", partial_holdout_enabled=True, radial_bin_width=1.0,
    )
    out = capsys.readouterr().out
    assert "*** PARTIAL-HOLDOUT TRAINING BAND ***" in out
    assert "*** HELD-OUT BAND ***" not in out


def test_log_radial_diagnostics_omits_holdout_banner_when_no_holdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = _radial_step_image(5.0, 3.0, 0.0)
    diagnostics_mod.log_radial_diagnostics(
        {"i": image}, {"i": image}, held_out_band=None, radial_bin_width=1.0,
    )
    out = capsys.readouterr().out
    assert "*** HELD-OUT BAND ***" not in out
    assert "*** PARTIAL-HOLDOUT TRAINING BAND ***" not in out


def test_log_radial_diagnostics_handles_nonfinite_profiles(capsys: pytest.CaptureFixture[str],) -> None:
    obs = {"g": np.full((5, 5), np.nan, dtype=np.float64), "r": _radial_step_image(np.inf, 1.0, 0.0)}
    pred = {"g": np.full((5, 5), np.nan, dtype=np.float64), "r": _radial_step_image(np.inf, 1.0, 0.0)}

    diagnostics_mod.log_radial_diagnostics(obs, pred, radial_bin_width=1.0)
    out = capsys.readouterr().out
    assert out.count("insufficient signal for FWHM") == 2


def test_log_radial_diagnostics_handles_missing_halflevel_crossings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original_where = diagnostics_mod.np.where

    def fake_where(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].dtype == bool:
            return (np.array([], dtype=np.int64),)
        return original_where(*args, **kwargs)

    monkeypatch.setattr(diagnostics_mod.np, "where", fake_where)

    image = _radial_step_image(10.0, 1.0, 0.0)
    diagnostics_mod.log_radial_diagnostics({"g": image}, {"g": image}, radial_bin_width=1.0)
    out = capsys.readouterr().out
    assert "insufficient signal for FWHM" in out


def test_log_radial_diagnostics_handles_interp_with_no_finite_samples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original_isfinite = diagnostics_mod.np.isfinite

    def fake_isfinite(value):
        if isinstance(value, np.ndarray):
            for frame in inspect.stack()[1:6]:
                if frame.function == "_interp_cross":
                    return np.zeros_like(value, dtype=bool)
        return original_isfinite(value)

    monkeypatch.setattr(diagnostics_mod.np, "isfinite", fake_isfinite)

    image = _radial_step_image(10.0, 1.0, 0.0)
    diagnostics_mod.log_radial_diagnostics({"g": image}, {"g": image}, radial_bin_width=1.0)
    out = capsys.readouterr().out
    assert "insufficient signal for FWHM" in out


def test_log_radial_diagnostics_handles_flat_interp_segment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(diagnostics_mod.np, "isclose", lambda *args, **kwargs: True)

    image = _radial_step_image(0.0, 2.0, 0.0)
    diagnostics_mod.log_radial_diagnostics({"g": image}, {"g": image}, radial_bin_width=1.0)
    out = capsys.readouterr().out
    assert "Band g:" in out


def _spectrum_target(
    *, label: str, truth_flux_density: np.ndarray, truth_counts: np.ndarray | None = None,
    throughput: np.ndarray | None = None,
) -> diagnostics_mod.SpectrumDiagnosticTarget:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    return diagnostics_mod.SpectrumDiagnosticTarget(
        label=label,
        role="science",
        lam=lam,
        truth_flux_density=np.asarray(truth_flux_density, dtype=np.float64),
        truth_counts=(None if truth_counts is None else np.asarray(truth_counts, dtype=np.float64)),
        throughput=(None if throughput is None else np.asarray(throughput, dtype=np.float64)),
    )


def test_spectrum_error_summary_preserves_legacy_sums_and_adds_integrals() -> None:
    lam = np.geomspace(1.0, 8.0, 4)
    truth = np.ones(4, dtype=np.float64)
    leading = diagnostics_mod._spectrum_error_summary(
        np.array([2.0, 0.0, 0.0, 0.0]), truth, lam,
    )
    trailing = diagnostics_mod._spectrum_error_summary(
        np.array([0.0, 0.0, 0.0, 0.5]), truth, lam,
    )

    assert leading["predicted_flux_sum"] != trailing["predicted_flux_sum"]
    assert leading["predicted_flux_integral"] == pytest.approx(
        trailing["predicted_flux_integral"], rel=0.0, abs=1e-12,
    )
    assert leading["truth_flux_integral"] == trailing["truth_flux_integral"]
    assert leading["flux_error_integrated_pct"] == pytest.approx(
        trailing["flux_error_integrated_pct"], rel=0.0, abs=1e-12,
    )

    fixture_pred = np.array([1.25, 2.5, 0.75], dtype=np.float64)
    fixture_truth = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    fixture = diagnostics_mod._spectrum_error_summary(
        fixture_pred, fixture_truth, np.array([5000.0, 5100.0, 5200.0]),
    )
    assert fixture["predicted_flux_sum"] == float(fixture_pred.sum())
    assert fixture["truth_flux_sum"] == float(fixture_truth.sum())
    assert fixture["flux_error_pct"] == float(
        100.0 * (fixture_pred.sum() - fixture_truth.sum()) / (fixture_truth.sum() + 1e-12)
    )


def test_write_fiber_spectrum_diagnostics_single_spectrum_writes_primary_alias(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra={3: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64))},
        spec_diagnostics_by_exposure={
            3: _spectrum_target(
                label="fiber", truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64),
                truth_counts=np.array([9.0, 10.0, 11.0], dtype=np.float64),
                throughput=np.array([1.0, 0.9, 0.8], dtype=np.float64),
            )
        },
        use_flux_density=True,
        run_dir=tmp_path,
    )

    assert (tmp_path / "fiber_spectrum.png").exists()
    assert (tmp_path / "fiber_spectrum_e3_fiber.png").exists()
    assert not (tmp_path / "fiber_spectra_overlay.png").exists()
    assert not (tmp_path / "fiber_spectra.npz").exists()

    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert list(metrics.keys()) == ["3"]
    assert metrics["3"]["exposure_id"] == 3
    assert metrics["3"]["label"] == "fiber"
    assert metrics["3"]["role"] == "science"


def test_write_fiber_spectrum_diagnostics_opt_in_npz_has_aligned_ragged_spectra(
    tmp_path: Path,
) -> None:
    lam3 = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    lam5 = np.array([6000.0, 6100.0], dtype=np.float64)
    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra={
            3: (lam3, np.array([1.0, 1.1, 1.2], dtype=np.float64)),
            5: (lam5, np.array([2.0, 2.1], dtype=np.float64)),
        },
        spec_diagnostics_by_exposure={
            3: _spectrum_target(label="G1", truth_flux_density=np.array([0.9, 1.0, 1.1])),
            5: diagnostics_mod.SpectrumDiagnosticTarget(
                label="G2", role="science", lam=lam5,
                truth_flux_density=np.array([1.8, 1.9], dtype=np.float64),
                truth_counts=None, throughput=None,
            ),
        },
        use_flux_density=True,
        run_dir=tmp_path,
        max_individual_spectra=0,
        save_fiber_spectra=True,
        spec_sigma_by_exposure={
            3: np.array([0.1, 0.2, 0.3], dtype=np.float64),
            5: np.array([0.4, 0.5], dtype=np.float64),
        },
    )

    with np.load(tmp_path / "fiber_spectra.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["exposure_ids"], np.array([3, 5]))
        for exp_id, expected_n in ((3, 3), (5, 2)):
            prefix = f"e{exp_id}"
            assert payload[f"{prefix}_lam"].shape == (expected_n,)
            assert payload[f"{prefix}_predicted"].shape == (expected_n,)
            assert payload[f"{prefix}_truth"].shape == (expected_n,)
            assert payload[f"{prefix}_sigma"].shape == (expected_n,)
            assert payload[f"{prefix}_role"].item() == "science"
            assert payload[f"{prefix}_truth_includes_additive"].item() is True
        assert payload["e3_label"].item() == "G1"
        np.testing.assert_allclose(payload["e5_sigma"], np.array([0.4, 0.5]))


def test_write_fiber_spectrum_diagnostics_source_only_scores_against_source_truth(
    tmp_path: Path,
) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra={3: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64))},
        spec_diagnostics_by_exposure={
            3: _spectrum_target(label="fiber", truth_flux_density=np.array([0.9, 1.0, 1.1]))
        },
        use_flux_density=True,
        run_dir=tmp_path,
        include_additive=False,
        spec_additive_by_exposure={3: np.array([0.4, 0.5, 0.6], dtype=np.float64)},
        save_fiber_spectra=True,
        spec_sigma_by_exposure={3: np.array([0.1, 0.1, 0.1], dtype=np.float64)},
    )

    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert metrics["3"]["truth_includes_additive"] is False
    assert metrics["3"]["flux_error_pct"] == pytest.approx(100.0 * (3.3 - 1.5) / 1.5, rel=1e-9)
    with np.load(tmp_path / "fiber_spectra.npz", allow_pickle=False) as payload:
        np.testing.assert_allclose(payload["e3_truth"], np.array([0.5, 0.5, 0.5]))
        assert payload["e3_truth_includes_additive"].item() is False


def test_write_fiber_spectrum_diagnostics_source_only_requires_additive_spectra(
    tmp_path: Path,
) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    fiber_spectra = {3: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64))}
    targets = {3: _spectrum_target(label="fiber", truth_flux_density=np.array([0.9, 1.0, 1.1]))}

    with pytest.raises(ValueError, match="spec_additive_by_exposure"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets,
            use_flux_density=True, run_dir=tmp_path, include_additive=False,
        )
    with pytest.raises(ValueError, match="exposure 3"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets,
            use_flux_density=True, run_dir=tmp_path, include_additive=False,
            spec_additive_by_exposure={4: np.array([0.4, 0.5, 0.6], dtype=np.float64)},
        )
    with pytest.raises(ValueError, match="does not match truth"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets,
            use_flux_density=True, run_dir=tmp_path, include_additive=False,
            spec_additive_by_exposure={3: np.array([0.4, 0.5], dtype=np.float64)},
        )


def test_write_fiber_spectrum_diagnostics_multispec_writes_overlay_and_unique_outputs(
    tmp_path: Path,
) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra={
            3: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64)),
            5: (lam, np.array([1.2, 1.25, 1.3], dtype=np.float64)),
        },
        spec_diagnostics_by_exposure={
            3: _spectrum_target(label="G1", truth_flux_density=np.array([0.9, 1.0, 1.1])),
            5: _spectrum_target(label="G2", truth_flux_density=np.array([1.1, 1.15, 1.2])),
        },
        use_flux_density=True,
        run_dir=tmp_path,
    )

    assert (tmp_path / "fiber_spectrum.png").exists()
    assert (tmp_path / "fiber_spectrum_e3_G1.png").exists()
    assert (tmp_path / "fiber_spectrum_e5_G2.png").exists()
    assert (tmp_path / "fiber_spectra_overlay.png").exists()

    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"3", "5"}


def test_write_fiber_spectrum_diagnostics_passes_overlay_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def _capture_overlay(items, run_dir, **kwargs):
        seen["n_items"] = len(items)
        seen["max_spectra"] = kwargs.get("max_spectra")

    import survey_fusion.evaluation.visualization as visualization_mod

    monkeypatch.setattr(visualization_mod, "plot_fiber_spectra_overlay", _capture_overlay)
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    fiber_spectra = {idx: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64)) for idx in range(5)}
    targets = {
        idx: _spectrum_target(label=f"S{idx}", truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64))
        for idx in range(5)
    }

    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets, use_flux_density=True,
        run_dir=tmp_path, max_overlay_spectra=2,
    )

    assert seen == {"n_items": 5, "max_spectra": 2}


def test_write_fiber_spectrum_diagnostics_caps_individual_plots_but_keeps_metrics(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    fiber_spectra = {idx: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64)) for idx in range(5)}
    targets = {
        idx: _spectrum_target(label=f"S{idx}", truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64))
        for idx in range(5)
    }

    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets, use_flux_density=True,
        run_dir=tmp_path, max_individual_spectra=2,
    )

    assert (tmp_path / "fiber_spectrum.png").exists()
    assert (tmp_path / "fiber_spectrum_e0_S0.png").exists()
    assert (tmp_path / "fiber_spectrum_e1_S1.png").exists()
    assert not (tmp_path / "fiber_spectrum_e2_S2.png").exists()
    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"0", "1", "2", "3", "4"}


def test_write_fiber_spectrum_diagnostics_can_disable_individual_plots(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    fiber_spectra = {idx: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64)) for idx in range(2)}
    targets = {
        idx: _spectrum_target(label=f"S{idx}", truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64))
        for idx in range(2)
    }

    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra=fiber_spectra, spec_diagnostics_by_exposure=targets, use_flux_density=True,
        run_dir=tmp_path, max_individual_spectra=0,
    )

    assert not (tmp_path / "fiber_spectrum.png").exists()
    assert not (tmp_path / "fiber_spectrum_e0_S0.png").exists()
    assert (tmp_path / "fiber_spectra_overlay.png").exists()
    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"0", "1"}


def test_write_fiber_spectrum_diagnostics_duplicate_labels_do_not_collide(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    diagnostics_mod.write_fiber_spectrum_diagnostics(
        fiber_spectra={
            7: (lam, np.array([1.0, 1.0, 1.0], dtype=np.float64)),
            9: (lam, np.array([1.1, 1.1, 1.1], dtype=np.float64)),
        },
        spec_diagnostics_by_exposure={
            7: _spectrum_target(label="dup", truth_flux_density=np.array([0.9, 0.9, 0.9])),
            9: _spectrum_target(label="dup", truth_flux_density=np.array([1.0, 1.0, 1.0])),
        },
        use_flux_density=True,
        run_dir=tmp_path,
    )

    assert (tmp_path / "fiber_spectrum_e7_dup.png").exists()
    assert (tmp_path / "fiber_spectrum_e9_dup.png").exists()
    metrics = json.loads((tmp_path / "fiber_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"7", "9"}


def test_write_fiber_spectrum_diagnostics_rejects_missing_target(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Missing spec_diagnostics_by_exposure entry"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={4: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64))},
            spec_diagnostics_by_exposure={}, use_flux_density=True, run_dir=tmp_path,
        )


def test_write_fiber_spectrum_diagnostics_requires_counts_truth_in_counts_mode(tmp_path: Path,) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    with pytest.raises(ValueError, match="counts truth"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={4: (lam, np.array([1.0, 1.1, 1.2], dtype=np.float64))},
            spec_diagnostics_by_exposure={
                4: _spectrum_target(
                    label="fiber", truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64),
                    truth_counts=None,
                )
            },
            use_flux_density=False,
            run_dir=tmp_path,
        )


def test_write_fiber_spectrum_diagnostics_rejects_wavelength_grid_mismatch(tmp_path: Path,) -> None:
    lam_pred = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    with pytest.raises(ValueError, match="wavelength grid does not match"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={4: (lam_pred, np.array([1.0, 1.1, 1.2], dtype=np.float64))},
            spec_diagnostics_by_exposure={
                4: diagnostics_mod.SpectrumDiagnosticTarget(
                    label="fiber", role="science", lam=np.array([5000.0, 5100.0, 5300.0], dtype=np.float64),
                    truth_flux_density=np.array([0.9, 1.0, 1.1], dtype=np.float64),
                    truth_counts=np.array([9.0, 10.0, 11.0], dtype=np.float64),
                    throughput=np.array([1.0, 0.9, 0.8], dtype=np.float64),
                )
            },
            use_flux_density=True,
            run_dir=tmp_path,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_write_fiber_spectrum_diagnostics_rejects_nonfinite_predictions_before_export(
    tmp_path: Path, bad_value: float,
) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    good = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    bad = np.array([1.0, bad_value, 1.2], dtype=np.float64)
    targets = {
        3: _spectrum_target(label="G1", truth_flux_density=np.array([0.9, 1.0, 1.1])),
        7: _spectrum_target(label="S3257-54888-0286", truth_flux_density=np.array([0.9, 1.0, 1.1])),
    }

    with pytest.raises(ValueError, match=r"exposure 7 / label 'S3257-54888-0286' has 1 non-finite of 3 bins"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={3: (lam, good), 7: (lam, bad)},
            spec_diagnostics_by_exposure=targets,
            use_flux_density=True,
            run_dir=tmp_path,
            max_individual_spectra=0,
            save_fiber_spectra=True,
            spec_sigma_by_exposure={3: np.full(3, 0.1), 7: np.full(3, 0.1)},
        )

    # The invalid exposure stops the export before anything is written: no
    # metrics, no spectra archive, and no plot for the earlier valid fiber.
    assert not (tmp_path / "fiber_metrics.json").exists()
    assert not (tmp_path / "fiber_spectra.npz").exists()
    assert not (tmp_path / "fiber_spectra_overlay.png").exists()
    assert list(tmp_path.iterdir()) == []


def test_write_fiber_spectrum_diagnostics_nonfinite_preflight_runs_before_any_plot(tmp_path: Path) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    good = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    bad = np.array([np.inf, 1.1, 1.2], dtype=np.float64)
    targets = {
        3: _spectrum_target(label="early", truth_flux_density=np.array([0.9, 1.0, 1.1])),
        9: _spectrum_target(label="late", truth_flux_density=np.array([0.9, 1.0, 1.1])),
    }
    with pytest.raises(ValueError, match="exposure 9 / label 'late'"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={3: (lam, good), 9: (lam, bad)},
            spec_diagnostics_by_exposure=targets,
            use_flux_density=True,
            run_dir=tmp_path,
        )
    # With individual plots enabled the earlier fiber would otherwise have
    # been drawn before the later one was inspected.
    assert list(tmp_path.iterdir()) == []


def test_write_fiber_spectrum_diagnostics_nonfinite_check_keeps_source_only_semantics(
    tmp_path: Path,
) -> None:
    lam = np.array([5000.0, 5100.0, 5200.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        diagnostics_mod.write_fiber_spectrum_diagnostics(
            fiber_spectra={3: (lam, np.array([1.0, np.nan, 1.2], dtype=np.float64))},
            spec_diagnostics_by_exposure={
                3: _spectrum_target(label="fiber", truth_flux_density=np.array([0.9, 1.0, 1.1]))
            },
            use_flux_density=True,
            run_dir=tmp_path,
            include_additive=False,
            spec_additive_by_exposure={3: np.array([0.4, 0.5, 0.6], dtype=np.float64)},
        )
    assert not (tmp_path / "fiber_metrics.json").exists()
