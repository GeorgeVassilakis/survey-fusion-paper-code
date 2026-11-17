from __future__ import annotations

from pathlib import Path

import pytest

from survey_fusion.training.checkpoints import (
    OPERATOR_SCHEMA_VERSION,
    PIXEL_RESPONSE_CONVENTION,
    checkpoint_psf_model,
    validate_checkpoint_operator_config,
)
from survey_fusion.utils.paths import build_run_dir


def _operator_config(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "operator_schema_version": OPERATOR_SCHEMA_VERSION,
        "psf_model": "mog",
        "psf_mixture_digest": "a" * 64,
        "psf_mixture_digest_version": 2,
        "psf_mixture_provenance_convention": (
            "digest_v2_operator_qa_pixel_response"
        ),
        "apply_spatial_psf": True,
        "pair_n_sigma": 8.0,
        "exact_pair_prune": True,
        "pixel_response_convention": PIXEL_RESPONSE_CONVENTION,
        "analytic_image_quadrature_order": 1,
        "analytic_images_enabled": True,
        "analytic_spectra_enabled": True,
        "spectral_response_digest": "c" * 64,
        "spectral_exposure_count": 2,
        "psf_jacobian_rel_tol": 0.01,
        "psf_jacobian_algorithm": "float64_one_pixel_tiled_v2",
        "psf_jacobian_max_tiles_per_axis": 16,
        "psf_jacobian_cap_policy": "error",
    }
    config.update(overrides)
    return config


def test_legacy_checkpoint_reads_as_gaussian() -> None:
    assert checkpoint_psf_model({}) == "gaussian"
    assert checkpoint_psf_model({"operator_config": {"pair_n_sigma": 8.0}}) == "gaussian"


@pytest.mark.parametrize(
    ("field", "trained_value"),
    [
        ("operator_schema_version", OPERATOR_SCHEMA_VERSION + 1),
        ("psf_model", "dgpsf"),
        ("psf_mixture_digest", "b" * 64),
        ("psf_mixture_digest_version", 1),
        ("psf_mixture_provenance_convention", "legacy_array_only"),
        ("apply_spatial_psf", False),
        ("pair_n_sigma", 3.5),
        ("exact_pair_prune", False),
        ("pixel_response_convention", "point_sample"),
        ("analytic_image_quadrature_order", 2),
        ("analytic_images_enabled", False),
        ("analytic_spectra_enabled", False),
        ("spectral_response_digest", "d" * 64),
        ("spectral_exposure_count", 3),
        ("psf_jacobian_rel_tol", 0.02),
        ("psf_jacobian_algorithm", "float32_center"),
        ("psf_jacobian_max_tiles_per_axis", 8),
        ("psf_jacobian_cap_policy", "warn"),
    ],
)
def test_resume_rejects_every_operator_identity_mismatch(
    field: str, trained_value: object,
) -> None:
    trained = _operator_config(**{field: trained_value})
    with pytest.raises(ValueError, match=field):
        validate_checkpoint_operator_config(
            {"operator_config": trained},
            current_operator_config=_operator_config(),
            context="Resume", checkpoint_path=Path("model.pt"), resume=True,
        )


def test_resume_psf_model_override_logs_loudly(capsys) -> None:
    checkpoint = {"operator_config": _operator_config(psf_model="dgpsf")}
    assert validate_checkpoint_operator_config(
        checkpoint, current_operator_config=_operator_config(),
        context="Resume", resume=True, allow_psf_model_change=True,
    )
    assert "WARNING: explicit resume operator override" in capsys.readouterr().out


@pytest.mark.parametrize(
    "legacy",
    [
        {},
        {
            "operator_config": {
                "psf_model": "mog",
                "psf_mixture_digest": "a" * 64,
            }
        },
    ],
)
def test_eval_rejects_unknown_or_partial_operator_identity(legacy) -> None:
    with pytest.raises(ValueError, match="incomplete.*allow_operator_reinterpretation"):
        validate_checkpoint_operator_config(
            legacy, current_operator_config=_operator_config(), context="Eval",
        )


def test_eval_allows_explicit_unknown_operator_reinterpretation(capsys) -> None:
    assert validate_checkpoint_operator_config(
        {},
        current_operator_config=_operator_config(),
        context="Eval",
        allow_operator_reinterpretation=True,
    )
    assert "explicit operator reinterpretation" in capsys.readouterr().out


def test_eval_digest_mismatch_raises_by_default() -> None:
    checkpoint = {
        "operator_config": _operator_config(psf_mixture_digest="b" * 64)
    }
    with pytest.raises(ValueError, match="fail-closed.*allow_operator_reinterpretation"):
        validate_checkpoint_operator_config(
            checkpoint, current_operator_config=_operator_config(),
            context="Holdout exposure evaluation",
        )


def test_eval_digest_mismatch_allows_explicit_reinterpretation(capsys) -> None:
    checkpoint = {
        "operator_config": _operator_config(psf_mixture_digest="b" * 64)
    }
    assert validate_checkpoint_operator_config(
        checkpoint, current_operator_config=_operator_config(),
        context="Holdout exposure evaluation",
        allow_operator_reinterpretation=True,
    )
    assert "explicit operator reinterpretation" in capsys.readouterr().out


def test_reinterpreted_evaluation_output_directory_has_suffix(tmp_path: Path) -> None:
    run_dir = build_run_dir(
        tmp_path, "eval", append_timestamp=False, suffix="_reinterpreted",
    )
    assert run_dir.name == "eval_reinterpreted"
