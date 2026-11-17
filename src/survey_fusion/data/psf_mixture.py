"""Canonical provenance digest for manifest PSF mixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np


PSF_MIXTURE_SHAPE = (3, 4)
PSF_MIXTURE_KEYS = ("psf_mixture_dgpsf", "psf_mixture_kl")
PSF_MIXTURE_DIGEST_VERSION = 2
PSF_MIXTURE_DIGEST_PAYLOAD_KEY = "psf_mixture_digest_payload"
PSF_MIXTURE_OPERATOR_KINDS = ("dgpsf", "kl")
PSF_MIXTURE_OPERATOR_PAYLOAD_FIELDS = (
    "operator_kind", "qa_pass", "fallback", "pixel_response",
)


def _validate_exposure_labels(
    exposures: Sequence[Mapping[str, object]],
) -> None:
    labels = [str(exposure.get("label", "")) for exposure in exposures]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("PSF mixture digest requires unique non-empty exposure labels.")


def _mixture_array(exposure: Mapping[str, object], key: str) -> np.ndarray:
    label = exposure.get("label")
    if key not in exposure:
        raise ValueError(f"{label!r} is missing required v2 field {key!r}.")
    array = np.asarray(exposure[key], dtype=">f8")
    if array.shape != PSF_MIXTURE_SHAPE:
        raise ValueError(
            f"{label!r} {key} has shape {array.shape}, "
            f"expected {PSF_MIXTURE_SHAPE}."
        )
    return array


def _legacy_psf_mixture_digest(
    exposures: Sequence[Mapping[str, object]],
) -> str:
    digest = hashlib.sha256()
    for exposure in sorted(exposures, key=lambda item: str(item["label"])):
        for key in PSF_MIXTURE_KEYS:
            digest.update(_mixture_array(exposure, key).tobytes(order="C"))
    return digest.hexdigest()


def _v2_operator_payload(
    exposure: Mapping[str, object], operator_kind: str,
) -> dict[str, object]:
    label = exposure.get("label")
    payload = exposure.get(PSF_MIXTURE_DIGEST_PAYLOAD_KEY)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{label!r} is missing required v2 field "
            f"{PSF_MIXTURE_DIGEST_PAYLOAD_KEY!r}."
        )
    missing_operators = set(PSF_MIXTURE_OPERATOR_KINDS) - set(payload)
    if missing_operators:
        raise ValueError(
            f"{label!r} v2 digest payload is missing operator fields "
            f"{sorted(missing_operators)}."
        )
    operator = payload[operator_kind]
    if not isinstance(operator, Mapping):
        raise ValueError(
            f"{label!r} v2 digest payload {operator_kind!r} must be a mapping."
        )
    missing = set(PSF_MIXTURE_OPERATOR_PAYLOAD_FIELDS) - set(operator)
    if missing:
        raise ValueError(
            f"{label!r} v2 digest payload {operator_kind!r} is missing "
            f"required fields {sorted(missing)}."
        )
    if operator["operator_kind"] != operator_kind:
        raise ValueError(
            f"{label!r} v2 digest payload {operator_kind!r} has operator_kind "
            f"{operator['operator_kind']!r}."
        )
    for key in ("qa_pass", "fallback"):
        if not isinstance(operator[key], bool):
            raise ValueError(
                f"{label!r} v2 digest payload {operator_kind!r} field "
                f"{key!r} must be boolean."
            )
    pixel_response = operator["pixel_response"]
    if not isinstance(pixel_response, str) or not pixel_response:
        raise ValueError(
            f"{label!r} v2 digest payload {operator_kind!r} field "
            "'pixel_response' must be a non-empty string."
        )
    return {
        key: operator[key] for key in PSF_MIXTURE_OPERATOR_PAYLOAD_FIELDS
    }


def psf_mixture_digest(
    exposures: Sequence[Mapping[str, object]],
    *,
    version: int = PSF_MIXTURE_DIGEST_VERSION,
) -> str:
    """Hash PSF arrays and bound provenance in stable exposure-label order."""
    _validate_exposure_labels(exposures)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in (1, PSF_MIXTURE_DIGEST_VERSION)
    ):
        raise ValueError(f"Unsupported PSF mixture digest version {version!r}.")
    if version == 1:
        return _legacy_psf_mixture_digest(exposures)

    records = []
    for exposure in sorted(exposures, key=lambda item: str(item["label"])):
        records.append({
            "label": str(exposure["label"]),
            "mixtures": {
                key: _mixture_array(exposure, key).tobytes(order="C").hex()
                for key in PSF_MIXTURE_KEYS
            },
            "operators": {
                operator_kind: _v2_operator_payload(exposure, operator_kind)
                for operator_kind in PSF_MIXTURE_OPERATOR_KINDS
            },
        })
    serialized = json.dumps(
        {"version": PSF_MIXTURE_DIGEST_VERSION, "exposures": records},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()
