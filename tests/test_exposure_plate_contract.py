from __future__ import annotations

import numpy as np
import pytest
import torch

from survey_fusion.data.bandpasses import Bandpass
from survey_fusion.pgm.contracts import DatumBatch, ExposureState, validate_batch_exposure_links


def _bandpass() -> Bandpass:
    lam = np.array([4500.0, 5500.0, 6500.0], dtype=np.float64)
    throughput = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return Bandpass(band="r", lam=lam, throughput=throughput, canonical_id="test_r")


def _exposure() -> ExposureState:
    class DummyWCS:
        def pixel_to_omega_xyz(self, x, y):
            shape = np.broadcast(x, y).shape
            omega = np.zeros(shape + (3,), dtype=np.float64)
            omega[..., 0] = 1.0
            return omega

    return ExposureState(
        wcs=DummyWCS(), psf_fwhm_arcsec=None, lsf_fwhm_angstrom=None, bandpasses={"r": _bandpass()},
        lam_spec=np.array([5000.0, 5100.0], dtype=np.float64), fiber_mask=np.ones((2, 2), dtype=np.float32),
        pixel_scale_arcsec=1.0,
    )


def _batch(exposure_ids: torch.Tensor) -> DatumBatch:
    n = int(exposure_ids.numel())
    return DatumBatch(
        y=torch.zeros(n, dtype=torch.float32), sigma=torch.ones(n, dtype=torch.float32),
        exposure_id=exposure_ids.to(dtype=torch.long), datum_kind=torch.zeros(n, dtype=torch.long),
        datum_index=torch.zeros(n, dtype=torch.long), kernel_samples=None,
    )


def test_validate_batch_exposure_links_accepts_valid_links() -> None:
    batch = _batch(torch.tensor([0, 1], dtype=torch.long))
    exposures = {0: _exposure(), 1: _exposure()}
    validate_batch_exposure_links(batch, exposures)


def test_validate_batch_exposure_links_rejects_unknown_exposure_id() -> None:
    batch = _batch(torch.tensor([0, 2], dtype=torch.long))
    exposures = {0: _exposure(), 1: _exposure()}
    with pytest.raises(ValueError, match="not present in exposure table"):
        validate_batch_exposure_links(batch, exposures)
