import numpy as np
import torch

from survey_fusion.data.random_holdout import (
    random_datum_holdout_mask,
    random_holdout_local_indices,
)


def test_mask_deterministic_and_correct_fraction():
    m1 = random_datum_holdout_mask(n_datums=100_000, fraction=0.02, seed=1234)
    m2 = random_datum_holdout_mask(n_datums=100_000, fraction=0.02, seed=1234)
    assert torch.equal(m1, m2)
    assert m1.dtype == torch.bool
    frac = m1.float().mean().item()
    assert abs(frac - 0.02) < 0.002


def test_different_seed_different_mask():
    a = random_datum_holdout_mask(10_000, 0.05, seed=1)
    b = random_datum_holdout_mask(10_000, 0.05, seed=2)
    assert not torch.equal(a, b)


def test_local_indices_deterministic_and_salted():
    a = random_holdout_local_indices(50_000, 0.02, seed=20260723, salt=0)
    b = random_holdout_local_indices(50_000, 0.02, seed=20260723, salt=0)
    c = random_holdout_local_indices(50_000, 0.02, seed=20260723, salt=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.int64
    assert (np.diff(a) > 0).all()


def test_adapter_random_holdout_excludes_and_records():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_sdss_partial_holdout_integration import (
        _bandpasses_with_i,
        _synthetic_sdss_scene,
    )
    from survey_fusion.data.adapters.sdss import build_sdss_datums

    device = torch.device("cpu")
    kwargs = dict(
        psf_fwhm_arcsec=1.0,
        device=device,
        random_holdout_cfg={"fraction": 0.1, "seed": 20260723},
    )
    store1 = build_sdss_datums(_synthetic_sdss_scene(), _bandpasses_with_i(), **kwargs, spectral_response_mode="legacy_imaging_median")
    store2 = build_sdss_datums(_synthetic_sdss_scene(), _bandpasses_with_i(), **kwargs, spectral_response_mode="legacy_imaging_median")
    assert store1.image_random_holdout_indices_by_exposure
    for exp_id, idx in store1.image_random_holdout_indices_by_exposure.items():
        assert idx.size > 0
        assert np.array_equal(idx, store2.image_random_holdout_indices_by_exposure[exp_id])
        excluded = store1.image_excluded_local_indices_by_exposure[exp_id]
        assert np.isin(idx, excluded).all()
