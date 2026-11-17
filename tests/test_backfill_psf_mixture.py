from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def _module():
    path = Path(__file__).parents[1] / "scripts" / "backfill_psf_mixture.py"
    spec = importlib.util.spec_from_file_location("backfill_psf_mixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest_payload(*, kl_qa_pass: bool = True, kl_fallback: bool = False) -> dict:
    return {
        "dgpsf": {
            "operator_kind": "dgpsf",
            "qa_pass": True,
            "fallback": False,
            "pixel_response": "analytic_unfolded_native",
        },
        "kl": {
            "operator_kind": "kl",
            "qa_pass": kl_qa_pass,
            "fallback": kl_fallback,
            "pixel_response": (
                "analytic_unfolded_native"
                if kl_fallback else "deconvolved_1_12_native"
            ),
        },
    }


def _wcs_header(crpix: tuple[float, float]) -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = crpix
    wcs.wcs.cdelt = np.array([-0.396 / 3600.0, 0.396 / 3600.0])
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs.to_header()


def _write_fixture(root: Path) -> tuple[Path, Path]:
    cutout = root / "cutout.fits"
    frame = root / "frame.fits"
    fits.PrimaryHDU(np.zeros((31, 31), dtype=np.float32), header=_wcs_header((16.0, 16.0))).writeto(cutout)
    fits.PrimaryHDU(np.zeros((64, 64), dtype=np.float32), header=_wcs_header((26.0, 30.0))).writeto(frame)

    n = 31
    center = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    stamps = []
    for sigma_x, sigma_y in ((1.3, 1.3), (2.5, 1.8), (4.0, 4.0)):
        stamp = np.exp(
            -0.5 * (((xx - center) / sigma_x) ** 2 + ((yy - center) / sigma_y) ** 2)
        ) / (2.0 * np.pi * sigma_x * sigma_y)
        stamps.append(stamp)
    eig = np.stack((0.7 * stamps[0], 0.2 * stamps[1], 0.1 * stamps[2]))
    coeff = np.zeros((3, 1), dtype=np.float64)
    coeff[:, 0] = 1.0
    kl_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="nrow_b", format="J", array=np.ones(3, dtype=np.int32)),
        fits.Column(name="ncol_b", format="J", array=np.ones(3, dtype=np.int32)),
        fits.Column(name="RNROW", format="J", array=np.full(3, n, dtype=np.int32)),
        fits.Column(name="RNCOL", format="J", array=np.full(3, n, dtype=np.int32)),
        fits.Column(name="c", format="1D", array=coeff),
        fits.Column(name="rrows", format=f"{n*n}D", array=eig.reshape(3, -1)),
    ])
    empty = fits.BinTableHDU.from_columns([])
    dg_values = {
        "psf_sigma1": [2.5] * 5,
        "psf_sigma2": [5.0] * 5,
        "psf_b": [0.029] * 5,
        "psf_p0": [0.0] * 5,
        "psf_beta": [3.11] * 5,
        "psf_sigmap": [1.94] * 5,
    }
    dg_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name=name, format="5D", array=np.asarray([values], dtype=np.float64))
        for name, values in dg_values.items()
    ])
    psfield_dir = root / "psfields"
    psfield_dir.mkdir()
    psfield = psfield_dir / "psField-000001-2-0003.fit"
    fits.HDUList([
        fits.PrimaryHDU(), kl_hdu, empty.copy(), empty.copy(), empty.copy(), empty.copy(), dg_hdu,
    ]).writeto(psfield)

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "image_exposures": [{
            "label": "u-000001-2-0003", "band": "u", "run": 1, "camcol": 2, "field": 3,
            "cutout_path": cutout.name, "frame_path": frame.name, "psf_fwhm_arcsec": 1.4,
        }]
    }), encoding="utf-8")
    return manifest, psfield_dir


def test_backfill_fits_dgpsf_and_kl_with_deterministic_digest(tmp_path: Path) -> None:
    module = _module()
    manifest, psfield_dir = _write_fixture(tmp_path)
    out_a = tmp_path / "out_a.json"
    out_b = tmp_path / "out_b.json"
    result_a = module.backfill(manifest, tmp_path, psfield_dir, out_a)
    result_b = module.backfill(manifest, tmp_path, psfield_dir, out_b)
    exposure = result_a["image_exposures"][0]
    assert result_a["psf_mixture_digest"] == result_b["psf_mixture_digest"]
    assert len(result_a["psf_mixture_digest"]) == 64
    assert result_a["psf_mixture_digest_version"] == 2
    assert np.asarray(exposure["psf_mixture_dgpsf"]).shape == (3, 4)
    assert np.asarray(exposure["psf_mixture_kl"]).shape == (3, 4)
    assert sum(row[0] for row in exposure["psf_mixture_dgpsf"]) == pytest.approx(1.0)
    assert sum(row[0] for row in exposure["psf_mixture_kl"]) == pytest.approx(1.0)
    moment = sum(
        row[0] * np.array([[row[1], row[2]], [row[2], row[3]]])
        for row in exposure["psf_mixture_kl"]
    )
    expected = 2.355 * np.sqrt(np.trace(moment) / 2.0) * 0.396
    assert exposure["psf_effective_fwhm_arcsec"] == pytest.approx(expected, rel=2e-6)
    assert exposure["psf_effective_fwhm_kl_arcsec"] == pytest.approx(expected, rel=2e-6)
    dg_rows = np.asarray(exposure["psf_mixture_dgpsf"])
    dg_variances = dg_rows[:, 1]
    truncate_px = module.DGPSF_TRUNCATE_ARCSEC / 0.396
    scaled_radius = truncate_px**2 / (2.0 * dg_variances)
    truncated_mass = 1.0 - np.exp(-scaled_radius)
    radial_second = 2.0 * dg_variances * (
        1.0 - np.exp(-scaled_radius) * (1.0 + scaled_radius)
    )
    expected_dg = 2.355 * np.sqrt(
        np.sum(dg_rows[:, 0] * radial_second)
        / np.sum(dg_rows[:, 0] * truncated_mass)
        / 2.0
    ) * 0.396
    assert exposure["psf_effective_fwhm_dg_arcsec"] == pytest.approx(expected_dg)
    assert exposure["psf_fwhm_arcsec"] == 1.4
    qa = result_a["psf_mixture_qa"]
    assert qa["scored_convention"] == "runtime_pixel_folded_covariance_1_12_native"
    assert qa["pixel_response"] == "deconvolved_1_12_native"
    assert qa["kl_stamp_pixel_response_evidence"] == "pixel_integrated_decided_upstream"
    assert qa["exposures"][0]["pixel_response"] == "deconvolved_1_12_native"
    assert qa["exposures"][0]["dgpsf"]["pixel_response"] == "analytic_unfolded_native"
    assert qa["exposures"][0]["kl"]["pixel_response"] == "deconvolved_1_12_native"
    assert qa["exposures"][0]["kl"]["tv_unfolded"] >= 0.0
    assert qa["exposures"][0]["kl"]["tv_folded"] >= 0.0
    assert qa["exposures"][0]["kl_fit_position"] == pytest.approx([29.0, 25.0])
    assert len(qa["exposures"][0]["kl_offcenter_positions"]) == 4
    assert len(qa["exposures"][0]["kl_offcenter_tv_folded"]) == 4
    assert qa["exposures"][0]["kl_offcenter_skipped"] == []
    assert qa["n_kl_fallback_dg"] == 0
    assert qa["kl_fallback_dg_labels"] == []


def test_backfill_refuses_in_place_and_kl_failure_falls_back(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    manifest, psfield_dir = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="Refusing to write.*in place"):
        module.backfill(manifest, tmp_path, psfield_dir, manifest)
    monkeypatch.setattr(
        module, "fit_kl_mixture",
        lambda _stamp, **_kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    result = module.backfill(manifest, tmp_path, psfield_dir, tmp_path / "fallback.json")
    exposure = result["image_exposures"][0]
    assert exposure["psf_mixture_kl"] == exposure["psf_mixture_dgpsf"]
    qa = result["psf_mixture_qa"]
    assert qa["n_kl_fallback_dg"] == 1
    assert qa["kl_fallback_dg_labels"] == ["u-000001-2-0003"]
    digest_payload = exposure["psf_mixture_digest_payload"]["kl"]
    assert digest_payload == {
        "operator_kind": "kl",
        "qa_pass": False,
        "fallback": True,
        "pixel_response": "analytic_unfolded_native",
    }


def test_backfill_skips_offcenter_positions_outside_frame_support(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _module()
    manifest, psfield_dir = _write_fixture(tmp_path)
    monkeypatch.setattr(module, "_cutout_frame_offsets", lambda *_args: (-14.0, -14.0))

    result = module.backfill(
        manifest, tmp_path, psfield_dir, tmp_path / "edge_diagnostics.json",
    )

    qa_row = result["psf_mixture_qa"]["exposures"][0]
    assert qa_row["kl_fit_position"] == [1.0, 1.0]
    assert qa_row["kl_offcenter_positions"] == [[8.5, 8.5]]
    assert len(qa_row["kl_offcenter_tv_folded"]) == 1
    assert len(qa_row["kl_offcenter_skipped"]) == 3
    assert all(
        skipped["reason"] == "outside_frame_support"
        for skipped in qa_row["kl_offcenter_skipped"]
    )


def test_fold_radial_density_matches_gaussian_identity() -> None:
    module = _module()
    radius = np.linspace(0.0, 12.0, 6001, dtype=np.float64)
    sigma_sq = 1.3
    density = np.exp(-0.5 * radius**2 / sigma_sq) / (2.0 * np.pi * sigma_sq)
    folded = module._fold_radial_density(
        radius, density, module.PIXEL_VARIANCE_NATIVE
    )
    expected_sq = sigma_sq + module.PIXEL_VARIANCE_NATIVE
    expected = np.exp(-0.5 * radius**2 / expected_sq) / (2.0 * np.pi * expected_sq)
    assert np.max(np.abs(folded - expected)) < 1e-5
    mass_in = np.trapezoid(density * 2.0 * np.pi * radius, radius)
    mass_out = np.trapezoid(folded * 2.0 * np.pi * radius, radius)
    assert abs(mass_out - mass_in) < 1e-6


def test_dgpsf_truncated_fit_passes_tv_and_encircled_energy_gates() -> None:
    module = _module()
    params = {
        "psf_sigma1": 2.5, "psf_sigma2": 5.0, "psf_b": 0.029,
        "psf_p0": 0.0, "psf_beta": 3.11, "psf_sigmap": 1.94,
    }
    _, qa = module.fit_dgpsf_mixture(params, 0.396)
    assert qa["tv"] < 0.01
    assert all(
        abs(value) < 0.005 for key, value in qa.items()
        if key.startswith("ee_error_")
        and not key.endswith(("_unfolded", "_folded"))
    )
    assert qa["tv_unfolded"] == qa["tv"]
    # Folded QA compares folded model to folded target, so it measures fit
    # quality in the runtime convention, not the fold cost itself.
    assert qa["tv_folded"] < 0.01


@pytest.mark.parametrize(
    ("target_folded", "should_pass"),
    ((True, False), (False, True)),
)
def test_dgpsf_gate_compares_folded_model_to_folded_target(
    monkeypatch, target_folded: bool, should_pass: bool,
) -> None:
    # The psField DG target is continuous (pre-pixel, verified against raw
    # KL stamps).  A fit whose components equal the continuous target passes
    # in the folded convention; a target that already carries the pixel
    # variance would make the stored fit double-count it, and must fail.
    module = _module()
    weights = np.array([0.6, 0.3, 0.1], dtype=np.float64)
    sigmas = np.array([0.7, 1.5, 3.0], dtype=np.float64)
    fitted_vector = np.concatenate((np.log(weights), np.log(sigmas)))

    def synthetic_target(radius, _params):
        target_sigmas = np.sqrt(
            sigmas**2 + (module.PIXEL_VARIANCE_NATIVE if target_folded else 0.0)
        )
        return module._radial_mixture_density(radius, weights, target_sigmas)

    monkeypatch.setattr(module, "_dgpsf_density", synthetic_target)
    monkeypatch.setattr(
        module,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(
            fun=np.zeros(1), x=fitted_vector, success=True, message="synthetic fit",
        ),
    )
    params = {
        "psf_sigma1": 1.0,
        "psf_sigma2": 2.0,
        "psf_b": 0.1,
        "psf_p0": 0.01,
        "psf_beta": 3.0,
        "psf_sigmap": 2.0,
    }

    if should_pass:
        _, qa = module.fit_dgpsf_mixture(params, 1.0)
        assert qa["tv_folded"] < 0.01
        assert qa["tv_unfolded"] < 0.01
    else:
        with pytest.raises(RuntimeError, match="DGPSF mixture TV"):
            module.fit_dgpsf_mixture(params, 1.0)


def test_pixel_integrated_gaussian_is_deconvolved_before_storage() -> None:
    module = _module()
    continuous_truth = [[1.0, 1.5, 0.1, 1.2]]
    target = module._render_mixture_stamp(
        (31, 31), continuous_truth, pixel_folded=True,
    )
    target /= target.sum()
    stored, qa = module.fit_kl_mixture(
        target, pixel_scale_arcsec=0.396, tv_threshold=0.08,
    )
    stored_moment = sum(
        row[0] * np.array([[row[1], row[2]], [row[2], row[3]]])
        for row in stored
    )
    np.testing.assert_allclose(
        stored_moment, np.array([[1.5, 0.1], [0.1, 1.2]]), atol=2e-5, rtol=0.0,
    )
    refolded = module._render_mixture_stamp(
        target.shape, stored, pixel_folded=True,
    )
    refolded /= refolded.sum()
    np.testing.assert_allclose(refolded, target, atol=2e-6, rtol=0.0)
    assert qa["tv_folded"] < 1e-4
    assert qa["tv_unfolded"] > qa["tv_folded"] + 0.01
    assert qa["central_peak_fractional_change_folded_vs_unfolded"] < -0.04
    assert qa["scored_convention"] == module.FINAL_QA_CONVENTION
    assert qa["pixel_response"] == "deconvolved_1_12_native"
    assert qa["n_covariance_floor_components"] == 0


def test_kl_deconvolution_floors_subpixel_components() -> None:
    module = _module()
    subpixel_components = [
        [0.5, 0.04, 0.0, 0.04],
        [0.3, 0.06, 0.0, 0.06],
        [0.2, 0.08, 0.0, 0.08],
    ]

    stored, n_floor_components = module._deconvolve_pixel_response(
        subpixel_components
    )

    for _, vxx, vxy, vyy in stored:
        eigenvalues = np.linalg.eigvalsh([[vxx, vxy], [vxy, vyy]])
        assert np.min(eigenvalues) == pytest.approx(
            module.KL_COVARIANCE_EIGENVALUE_FLOOR_NATIVE
        )
    assert n_floor_components == 3


def test_kl_padded_stability_and_clean_gaussian_pass() -> None:
    module = _module()
    target = module._render_mixture_stamp(
        (31, 31), [[1.0, 1.5, 0.1, 1.2]], pixel_folded=False,
    )
    target /= target.sum()
    _, qa = module.fit_kl_mixture(
        target, pixel_scale_arcsec=0.396, tv_threshold=0.08, qa_padded=True,
    )
    assert qa["tv_folded"] < 0.08
    assert qa["outside_mass"] < 1e-6
    assert qa["padded"]["pad_pixels"] > 0
    assert qa["padded"]["component_max_abs_drift"] >= 0.0
    assert "outside_mass_change" in qa["padded"]


def test_substantial_outside_mass_gates_and_backfill_falls_back(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    module = _module()
    manifest, psfield_dir = _write_fixture(tmp_path)
    yy, xx = np.mgrid[0:5, 0:5].astype(np.float64)
    broad = np.exp(-0.5 * ((xx - 2.0) ** 2 + (yy - 2.0) ** 2) / 3.0**2)
    broad /= broad.sum()
    with pytest.raises(module.KLFitGateError) as error:
        module.fit_kl_mixture(broad, pixel_scale_arcsec=0.396)
    assert error.value.qa["outside_mass"] > 0.5

    monkeypatch.setattr(module, "_kl_stamp", lambda *_args: broad)
    result = module.backfill(
        manifest, tmp_path, psfield_dir, tmp_path / "outside_fallback.json",
    )
    exposure = result["image_exposures"][0]
    qa_row = result["psf_mixture_qa"]["exposures"][0]
    assert exposure["psf_mixture_kl"] == exposure["psf_mixture_dgpsf"]
    assert qa_row["fallback"] is True
    assert qa_row["kl"]["outside_mass"] > 0.5
    assert "KL->DGPSF FALLBACK" in capsys.readouterr().out


def test_symmetric_tiny_perturbation_has_identical_canonical_rows_and_digest() -> None:
    module = _module()
    truth = [
        [0.2, 0.8, 0.0, 0.8],
        [0.5, 2.5, 0.0, 2.5],
        [0.3, 8.0, 0.0, 8.0],
    ]
    target = module._render_mixture_stamp((31, 31), truth, pixel_folded=False)
    target /= target.sum()
    yy, xx = np.mgrid[0:31, 0:31]
    radius2 = (xx - 15) ** 2 + (yy - 15) ** 2
    perturbed = target * (1.0 + 1e-6 * ((radius2 > 50) & (radius2 < 80)))
    perturbed /= perturbed.sum()
    rows_a, _ = module.fit_kl_mixture(target, tv_threshold=0.2)
    rows_b, _ = module.fit_kl_mixture(perturbed, tv_threshold=0.2)
    assert rows_a == rows_b
    dg = [[0.6, 1.0, 0.0, 1.0], [0.3, 4.0, 0.0, 4.0], [0.1, 9.0, 0.0, 9.0]]
    exposure_a = [{
        "label": "a", "psf_mixture_dgpsf": dg, "psf_mixture_kl": rows_a,
        "psf_mixture_digest_payload": _digest_payload(),
    }]
    exposure_b = [{
        "label": "a", "psf_mixture_dgpsf": dg, "psf_mixture_kl": rows_b,
        "psf_mixture_digest_payload": _digest_payload(),
    }]
    assert module.psf_mixture_digest(exposure_a) == module.psf_mixture_digest(exposure_b)


def test_digest_canonicalizes_exposure_order() -> None:
    module = _module()
    mixture_a = [[0.6, 1.0, 0.0, 1.0], [0.3, 4.0, 0.0, 4.0], [0.1, 9.0, 0.0, 9.0]]
    mixture_b = [[0.5, 2.0, 0.0, 2.0], [0.3, 5.0, 0.0, 5.0], [0.2, 10.0, 0.0, 10.0]]
    exposures = [
        {
            "label": "b", "psf_mixture_dgpsf": mixture_b,
            "psf_mixture_kl": mixture_a,
            "psf_mixture_digest_payload": _digest_payload(),
        },
        {
            "label": "a", "psf_mixture_dgpsf": mixture_a,
            "psf_mixture_kl": mixture_b,
            "psf_mixture_digest_payload": _digest_payload(),
        },
    ]
    assert module.psf_mixture_digest(exposures) == module.psf_mixture_digest(
        list(reversed(exposures))
    )


def test_v2_digest_binds_qa_status_and_requires_complete_payload() -> None:
    module = _module()
    mixture = [
        [0.6, 1.0, 0.0, 1.0],
        [0.3, 4.0, 0.0, 4.0],
        [0.1, 9.0, 0.0, 9.0],
    ]
    passing = [{
        "label": "a",
        "psf_mixture_dgpsf": mixture,
        "psf_mixture_kl": mixture,
        "psf_mixture_digest_payload": _digest_payload(kl_qa_pass=True),
    }]
    failing = [{
        **passing[0],
        "psf_mixture_digest_payload": _digest_payload(kl_qa_pass=False),
    }]

    assert module.psf_mixture_digest(passing) != module.psf_mixture_digest(failing)

    incomplete = [{
        **passing[0],
        "psf_mixture_digest_payload": _digest_payload(),
    }]
    del incomplete[0]["psf_mixture_digest_payload"]["kl"]["pixel_response"]
    with pytest.raises(ValueError, match="missing.*pixel_response"):
        module.psf_mixture_digest(incomplete)
