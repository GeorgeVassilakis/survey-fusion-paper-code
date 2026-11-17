from __future__ import annotations

import pytest
import torch

from survey_fusion.pgm.contracts import KernelSamples, validate_kernel_samples


def _valid_samples(m: int = 3, k: int = 5) -> KernelSamples:
    omega_xyz = torch.zeros((m, k, 3), dtype=torch.float32)
    omega_xyz[..., 0] = 1.0
    loglam_val = float(torch.log(torch.tensor(5000.0)).item())
    loglam = torch.full((m, k), loglam_val, dtype=torch.float32)
    w = torch.full((m, k), 1.0 / float(k), dtype=torch.float32)
    z = torch.ones((m,), dtype=torch.float32)
    return KernelSamples(omega_xyz=omega_xyz, loglam=loglam, w=w, Z=z)


def test_validate_kernel_samples_accepts_valid_phase1_kernel() -> None:
    samples = _valid_samples()
    validate_kernel_samples(samples, require_z_close_to_one=True)


def test_validate_kernel_samples_rejects_non_unit_omega() -> None:
    samples = _valid_samples()
    samples.omega_xyz[..., 0] = 0.9
    with pytest.raises(ValueError, match="unit-normalized"):
        validate_kernel_samples(samples)


def test_validate_kernel_samples_rejects_nonfinite_loglam() -> None:
    samples = _valid_samples()
    samples.loglam[0, 0] = float("inf")
    with pytest.raises(ValueError, match="loglam must be finite"):
        validate_kernel_samples(samples)


def test_validate_kernel_samples_rejects_nonpositive_wavelength() -> None:
    samples = _valid_samples()
    samples.loglam[0, 0] = -1000.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_kernel_samples(samples)


def test_validate_kernel_samples_rejects_z_not_one_in_fluxdensity_mode() -> None:
    samples = _valid_samples()
    samples.Z[1] = 1.05
    with pytest.raises(ValueError, match="Z must be 1"):
        validate_kernel_samples(samples, require_z_close_to_one=True)
