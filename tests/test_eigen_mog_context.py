from types import SimpleNamespace

import numpy as np
import torch

from survey_fusion.training.eigen_mog_renderer import EigenMogRenderer, eigen_mog_identity
from test_eigen_mog import write_sidecar


def test_cached_wcs_preserves_pixel_tiles_and_band_order(tmp_path, monkeypatch):
    path = write_sidecar(tmp_path / "sidecar.npz", k=10)
    _, identity = eigen_mog_identity(dict(eigen_mog_sidecar=str(path), eigen_mog_support_radius=18))
    matrices = torch.tensor(
        [
            [[2.0, 0.1], [0.05, 3.0]],
            [[2.1, 0.2], [0.1, 3.2]],
            [[1.9, -0.1], [0.03, 2.8]],
            [[2.05, 0.01], [-0.02, 3.1]],
        ],
        dtype=torch.float64,
    )
    caches = {
        i: dict(
            psf_jacobian_path="tiled",
            psf_jacobian_tile_grid=(2, 2),
            pixel_to_norm_jacobian_tiles=matrices * (i + 1),
            band_lam=torch.arange(3),
            band_throughput=torch.ones(3),
        )
        for i in range(2)
    }
    trainer = SimpleNamespace(
        eigen_mog_sidecar_path=str(path),
        eigen_mog_config=identity,
        eigen_mog_execution="cached",
        device=torch.device("cpu"),
        exposures={i: SimpleNamespace(label=f"e{i:04d}", image_width=8, image_height=8) for i in range(2)},
        image_band_by_exposure={0: "g", 1: "r"},
        _analytic_cache_for_exposure=lambda eid: caches[eid],
    )
    ids = np.array([1, 0, 1, 0, 0, 1])
    local = np.array([0, 7, 63, 56, 27, 36])
    tile = (local // 8 >= 4) * 2 + (local % 8 >= 4)
    jac = matrices[tile] * torch.as_tensor(ids + 1)[:, None, None]
    ref = EigenMogRenderer(trainer)
    monkeypatch.setenv("SPLAT_EIGEN_FAST_CONTEXT", "0")
    expected = ref.prepare(ids, local, jac)
    actual_renderer = EigenMogRenderer(trainer)
    monkeypatch.setenv("SPLAT_EIGEN_FAST_CONTEXT", "1")
    actual = actual_renderer.prepare(ids, local, None)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    tables = actual_renderer._wcs_tables
    actual_renderer.prepare(ids, local, None)
    assert actual_renderer._wcs_tables is tables
    bands, _, mapping = actual_renderer.batch_metadata([1, 0])
    assert bands == ["r", "g"] and mapping.tolist() == [0, 1]
    bands, _, mapping = actual_renderer.batch_metadata([1])
    assert bands == ["r"] and mapping.tolist() == [0]
