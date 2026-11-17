from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from train import _nested_source_observations


def test_source_observations_prefer_datastore_source_before_scene_images() -> None:
    scene = SimpleNamespace(images={"r": np.full((2, 2), 5.0, dtype=np.float32)})
    data_store = SimpleNamespace(
        image_band_by_exposure={3: "r"}, image_source_by_exposure={3: np.full((2, 2), 2.0, dtype=np.float32)},
    )

    nested = _nested_source_observations(scene, data_store, use_flux_density=True)

    np.testing.assert_allclose(nested[3]["r"], 2.0)


def test_source_observations_prefer_scene_source_flambda_when_available() -> None:
    scene = SimpleNamespace(
        image_source_flambda={"r": np.full((2, 2), 3.0, dtype=np.float32)},
        images={"r": np.full((2, 2), 5.0, dtype=np.float32)},
    )
    data_store = SimpleNamespace(
        image_band_by_exposure={3: "r"}, image_source_by_exposure={3: np.full((2, 2), 2.0, dtype=np.float32)},
    )

    nested = _nested_source_observations(scene, data_store, use_flux_density=True)

    np.testing.assert_allclose(nested[3]["r"], 3.0)


def test_source_observations_do_not_fall_back_to_total_scene_images() -> None:
    scene = SimpleNamespace(images={"r": np.full((2, 2), 5.0, dtype=np.float32)})
    data_store = SimpleNamespace(image_band_by_exposure={3: "r"}, image_source_by_exposure={})

    nested = _nested_source_observations(scene, data_store, use_flux_density=True)

    assert nested == {}
