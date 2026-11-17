import math

import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.splat_spectrum import splat_band_amplitudes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize('rows',[8192,65536])
def test_compact_all_rows_matches_dense_pruning_amplitudes(monkeypatch,rows):
    from survey_fusion.pgm.compact_band_kernels import CompactBandContraction
    original=CompactBandContraction.apply
    calls=[]
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(CompactBandContraction,'apply',counted)
    torch.manual_seed(260922)
    field = SplatField(
        rows, 32, torch.linspace(-1, 1, 96), sed_off_grid="zero", sed_pin_gauge_knots=True
    ).cuda()
    with torch.no_grad():
        field.sed_shift.uniform_(-0.7, 0.7)
        field.sed_coeffs.normal_()
        field.raw_flux.normal_()
    bands = {}
    for name, lo, hi in [
        ("u", 3100, 4000),
        ("g", 4000, 5500),
        ("r", 5500, 7000),
        ("i", 6900, 8400),
        ("z", 8200, 10300),
    ]:
        wave = torch.linspace(lo, hi, 51, device="cuda")
        bands[name] = (wave, torch.sin(torch.linspace(0, math.pi, 51, device="cuda")).clamp_min(0))
    with torch.no_grad():
        monkeypatch.setenv("SPLAT_COMPACT_ALL_BANDS", "0")
        expected = splat_band_amplitudes(
            field, bands, loglam_min=math.log(2800), loglam_max=math.log(11000), include_flux=False
        )
        monkeypatch.setenv("SPLAT_COMPACT_ALL_BANDS", "1")
        actual = splat_band_amplitudes(
            field, bands, loglam_min=math.log(2800), loglam_max=math.log(11000), include_flux=False
        )
    for band in bands:
        torch.testing.assert_close(actual[band], expected[band], rtol=0, atol=0)
    assert bool(calls)==(rows==8192),'Unexpected compact-path selection'
