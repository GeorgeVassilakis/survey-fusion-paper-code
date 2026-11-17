from types import SimpleNamespace

import numpy as np
import torch

from survey_fusion.pgm.eigen_mog import EigenMixture
from survey_fusion.training.eigen_mog_renderer import EigenMogRenderer, eigen_mog_identity
from test_eigen_mog import write_sidecar


def test_prepare_uses_frame_coordinates_and_per_exposure_tables(tmp_path):
    path = write_sidecar(tmp_path / "sidecar.npz")
    _, identity = eigen_mog_identity({
        "eigen_mog_sidecar": str(path),
        "eigen_mog_support_radius": 20,
        "eigen_mog_compute_dtype": "float64",
    })
    trainer = SimpleNamespace(
        eigen_mog_sidecar_path=str(path),
        eigen_mog_config=identity,
        eigen_mog_pair_chunk=100,
        device=torch.device("cpu"),
        exposures={0: SimpleNamespace(label="e0000", image_width=8)},
    )
    renderer = EigenMogRenderer(trainer)
    local = np.array([0, 11, 23])
    jacobian = torch.tensor([
        [[2.0, 0.1], [0.0, 3.0]],
        [[2.0, 0.1], [0.0, 3.0]],
        [[2.0, 0.1], [0.0, 3.0]],
    ], dtype=torch.float64)
    context = renderer.prepare(np.zeros(3, dtype=np.int64), local, jacobian)
    assert context["weights"].shape == (3, 8)
    torch.testing.assert_close(
        context["weights"].sum(1), torch.ones(3, dtype=torch.float64), rtol=0, atol=1e-12
    )
    torch.testing.assert_close(context["det"], torch.full((3,), 6.0, dtype=torch.float64))
    torch.testing.assert_close(context["wing"], 20 * jacobian.abs().sum(-1))
    representation = EigenMixture.load(path)
    expected_alpha = representation.normalized_kl_weights(
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 3.0, 7.0], dtype=torch.float64),
    )
    expected_weights = (
        expected_alpha[:, :, None]
        * torch.as_tensor(representation.weights[0], dtype=torch.float64)[None]
    ).reshape(3, -1)
    torch.testing.assert_close(context["weights"], expected_weights, rtol=0, atol=0)
