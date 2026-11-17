from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from train import(_image_exposure_ids_for_band, _nested_observations, _nested_source_observations)


def test_train_postprocess_observations_use_exposure_store_for_repeated_bands() -> None:
    scene = SimpleNamespace(
        images_flambda={"r": np.full((2, 2), -1.0, dtype=np.float32)},
        image_source_flambda={"r": np.full((2, 2), -2.0, dtype=np.float32)},
    )
    data_store = SimpleNamespace(
        image_band_by_exposure={0: "r", 1: "r"},
        image_total_by_exposure={
            0: np.full((2, 2), 1.0, dtype=np.float32), 1: np.full((2, 2), 2.0, dtype=np.float32),
        },
        image_source_by_exposure={
            0: np.full((2, 2), 3.0, dtype=np.float32), 1: np.full((2, 2), 4.0, dtype=np.float32),
        },
    )

    total = _nested_observations(scene, data_store, use_flux_density=True)
    source = _nested_source_observations(scene, data_store, use_flux_density=True)

    assert np.all(total[0]["r"] == 1.0)
    assert np.all(total[1]["r"] == 2.0)
    assert np.all(source[0]["r"] == 3.0)
    assert np.all(source[1]["r"] == 4.0)


def test_sed_background_can_request_all_exposures_for_diagnostic_band() -> None:
    data_store = SimpleNamespace(image_band_by_exposure={0: "g", 1: "i", 2: "r", 3: "i", 4: "z"})

    assert _image_exposure_ids_for_band(data_store, "i") == [1, 3]
