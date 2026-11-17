from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from survey_fusion.pgm.contracts import ExposureState, SkyDiskApertureSpec
from survey_fusion.training.checkpoints import operator_config_for_trainer, validate_checkpoint_operator_config


def _trainer():
    exposure = ExposureState(
        wcs=None,
        psf_fwhm_arcsec=1.5,
        lsf_fwhm_angstrom=np.array([2.0, 3.0]),
        bandpasses={},
        lam_spec=np.array([5000.0, 5100.0]),
        fiber_mask=np.ones((2, 2)),
        pixel_scale_arcsec=1.0,
        lam_edges=np.array([[4950.0, 5050.0], [5050.0, 5150.0]]),
        aperture=SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.0),
        spectral_source_scale=1.2,
        spectral_response_metadata={"provenance": {"source": "matching reduction"}},
    )
    data = SimpleNamespace(
        exposures={4: exposure},
        spec_label_by_exposure={4: "S4"},
        spec_role_by_exposure={4: "science"},
        holdout_region_summary={"spectral_response_mode": "metadata", "spectral_lsf_sampling": "native"},
    )
    return SimpleNamespace(data=data, exposures=data.exposures, analytic_spectra_enabled=True)


@pytest.mark.parametrize(
    "changed", ["seeing", "scale", "lsf", "membership", "provenance", "edges", "aperture", "policy"]
)
def test_checkpoint_rejects_changed_spectral_likelihood(changed):
    trainer = _trainer()
    original = operator_config_for_trainer(trainer)
    exposure = trainer.exposures[4]
    if changed == "seeing":
        exposure.psf_fwhm_arcsec = 2.0
    elif changed == "scale":
        exposure.spectral_source_scale = 1.3
    elif changed == "lsf":
        exposure.lsf_fwhm_angstrom[0] += 0.2
    elif changed == "membership":
        trainer.exposures[5] = trainer.exposures.pop(4)
    elif changed == "provenance":
        exposure.spectral_response_metadata["provenance"]["source"] = "different reduction"
    elif changed == "edges":
        exposure.lam_edges[0, 0] -= 1
    elif changed == "aperture":
        exposure.aperture = SkyDiskApertureSpec(center_radec_deg=(180.0, 0.0), radius_arcsec=1.5)
    elif changed == "policy":
        trainer.data.holdout_region_summary["spectral_lsf_sampling"] = "legacy_masked_spacing"
    current = operator_config_for_trainer(trainer)
    assert original["psf_model"] == current["psf_model"]
    assert original["spectral_response_digest"] != current["spectral_response_digest"]
    with pytest.raises(ValueError, match="spectral_response_digest"):
        validate_checkpoint_operator_config(
            {"operator_config": original}, current_operator_config=current, context="Resume", resume=True
        )


def test_checkpoint_identical_spectral_response_roundtrips():
    trainer = _trainer()
    saved = operator_config_for_trainer(trainer)
    current = operator_config_for_trainer(deepcopy(trainer))
    assert saved["spectral_exposure_count"] == 1
    assert not validate_checkpoint_operator_config(
        {"operator_config": saved}, current_operator_config=current, context="Resume", resume=True
    )


def test_old_image_only_identity_requires_explicit_resume_override(capsys):
    current = operator_config_for_trainer(_trainer())
    old = dict(current)
    old["operator_schema_version"] = 2
    for key in ("spectral_response_digest", "spectral_exposure_count", "analytic_spectra_enabled"):
        old.pop(key)
    with pytest.raises(ValueError, match="fail-closed"):
        validate_checkpoint_operator_config(
            {"operator_config": old}, current_operator_config=current, context="Resume", resume=True
        )
    assert validate_checkpoint_operator_config(
        {"operator_config": old},
        current_operator_config=current,
        context="Resume",
        resume=True,
        allow_operator_change=True,
    )
    assert "explicit resume operator override" in capsys.readouterr().out


def test_empty_spectral_identity_requires_no_metadata():
    identity = operator_config_for_trainer(SimpleNamespace(exposures={}))
    assert identity["spectral_exposure_count"] == 0
    assert len(identity["spectral_response_digest"]) == 64


@pytest.mark.parametrize("with_label_map", [True, False])
def test_image_wavelength_metadata_is_not_a_spectral_exposure(with_label_map):
    from dataclasses import replace

    trainer = _trainer()
    if not with_label_map:
        trainer.data.spec_label_by_exposure = {}
    original = operator_config_for_trainer(trainer)
    image = replace(
        trainer.exposures[4],
        bandpasses={"r": None},
        lam_edges=None,
        aperture=None,
        spectral_source_scale=None,
        spectral_response_metadata=None,
    )
    trainer.exposures[3] = image
    current = operator_config_for_trainer(trainer)
    assert current["spectral_exposure_count"] == 1
    assert current["spectral_response_digest"] == original["spectral_response_digest"]
    image.psf_fwhm_arcsec = 2.7
    assert operator_config_for_trainer(trainer)["spectral_response_digest"] == original["spectral_response_digest"]
