import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from survey_fusion.pgm.eigen_mog import ALGORITHM, EigenMixture, eigen_mog_pair_values
from survey_fusion.pgm.eigen_mog_triton import eigen_mog_pairs
from survey_fusion.training.checkpoints import (
    operator_config_for_trainer,
    validate_checkpoint_operator_config,
)
from survey_fusion.training.eigen_mog_renderer import eigen_mog_identity


def write_sidecar(path: Path, *, k=2, offset=0.0):
    means = np.zeros((410, 4, k, 2), dtype=np.float64)
    means[..., 0] = offset
    covariances = np.zeros((410, 4, k, 3), dtype=np.float64)
    covariances[..., 0] = 1.0
    covariances[..., 2] = 1.0
    eigen_sums = np.ones((410, 4), dtype=np.float64)
    weights = np.full((410, 4, k), 1.0 / k, dtype=np.float64)
    coefficients = np.zeros((410, 4, 5, 5), dtype=np.float64)
    coefficients[:, :, 0, 0] = np.array([1.0, 0.2, 0.1, 0.05])
    records = [
        {"label": f"e{index:04d}", "row0": float(index), "col0": float(2 * index)}
        for index in range(410)
    ]
    metadata = {
        "schema": 1,
        "algorithm": ALGORITHM,
        "pixel_response": "already_in_native_stamp",
        "atlas_sha256": "a" * 64,
        "records": records,
        "fit": {"k": k, "max_offset": 10.0, "max_sigma": 5.0},
        "validation": {"median_relative_l2": 0.01},
        "support": {"radius": 20},
    }
    np.savez_compressed(
        path,
        means=means,
        covariances=covariances,
        weights=weights,
        eigen_sums=eigen_sums,
        kl_coefficients=coefficients,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return path


def test_sidecar_load_weights_and_identity(tmp_path):
    path = write_sidecar(tmp_path / "sidecar.npz")
    representation = EigenMixture.load(path)
    row = torch.tensor([10.0, 20.0], dtype=torch.float64)
    col = torch.tensor([30.0, 40.0], dtype=torch.float64)
    tables = representation.pixel_tables(torch.tensor([0, 1]), row, col)
    assert tables[0].shape == (2, 8)
    torch.testing.assert_close(
        tables[0].sum(1), torch.ones(2, dtype=torch.float64), rtol=1e-15, atol=1e-15
    )
    assert representation.identity() == {
        "algorithm": ALGORITHM,
        "atlas_sha256": "a" * 64,
        "k": 2,
        "n_components": 8,
        "fit": {"k": 2, "max_offset": 10.0, "max_sigma": 5.0},
    }


def test_loader_rejects_bad_covariance_and_integral(tmp_path):
    path = write_sidecar(tmp_path / "sidecar.npz")
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values["covariances"] = values["covariances"].copy()
    values["covariances"][0, 0, 0] = [1.0, 2.0, 1.0]
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="positive definite"):
        EigenMixture.load(path)


def test_torch_pair_values_and_gradients_are_finite():
    torch.manual_seed(4)
    pairs, components = 11, 40
    geometry = [torch.randn(pairs, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    cxx = (torch.rand(pairs, dtype=torch.float64) + 0.5).requires_grad_()
    cxy = (0.05 * torch.randn(pairs, dtype=torch.float64)).requires_grad_()
    cyy = (torch.rand(pairs, dtype=torch.float64) + 0.5).requires_grad_()
    weights = torch.randn(pairs, components, dtype=torch.float64) / components
    means = [0.3 * torch.randn(pairs, components, dtype=torch.float64) for _ in range(2)]
    sxx = torch.rand(pairs, components, dtype=torch.float64) + 0.2
    sxy = 0.02 * torch.randn(pairs, components, dtype=torch.float64)
    syy = torch.rand(pairs, components, dtype=torch.float64) + 0.2
    value = eigen_mog_pair_values(*geometry, cxx, cxy, cyy, weights, *means, sxx, sxy, syy)
    gradients = torch.autograd.grad(value.sum(), (*geometry, cxx, cxy, cyy))
    assert value.shape == (pairs,)
    assert all(torch.isfinite(tensor).all() for tensor in (value, *gradients))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-9), (torch.float32, 3e-5)])
def test_triton_matches_reference_forward_and_gradients(dtype, tolerance):
    torch.manual_seed(8)
    device = torch.device("cuda")
    pixels, pairs, components = 9, 37, 40
    point = torch.randint(pixels, (pairs,), device=device)
    tables = [torch.randn(pixels, components, device=device, dtype=dtype) for _ in range(3)]
    sxx = torch.rand(pixels, components, device=device, dtype=dtype) + 0.4
    sxy = 0.03 * torch.randn(pixels, components, device=device, dtype=dtype)
    syy = torch.rand(pixels, components, device=device, dtype=dtype) + 0.4
    tables.extend([sxx, sxy, syy])
    base = [torch.randn(pairs, device=device, dtype=dtype) for _ in range(2)]
    base.extend([
        torch.rand(pairs, device=device, dtype=dtype) + 0.8,
        0.03 * torch.randn(pairs, device=device, dtype=dtype),
        torch.rand(pairs, device=device, dtype=dtype) + 0.8,
    ])
    reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in base]
    triton_inputs = [tensor.detach().clone().requires_grad_() for tensor in base]
    gathered = [table[point] for table in tables]
    reference = eigen_mog_pair_values(*reference_inputs, *gathered)
    actual = eigen_mog_pairs(*triton_inputs, point, *tables, implementation="triton")
    reference_grad = torch.autograd.grad(reference.sum(), reference_inputs)
    actual_grad = torch.autograd.grad(actual.sum(), triton_inputs)
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    for got, wanted in zip(actual_grad, reference_grad):
        torch.testing.assert_close(got, wanted, rtol=tolerance, atol=tolerance)


def test_identity_and_checkpoint_change_controls(tmp_path):
    path = write_sidecar(tmp_path / "sidecar.npz")
    with pytest.raises(ValueError, match="support_radius"):
        eigen_mog_identity({"eigen_mog_sidecar": str(path)})
    with pytest.raises(ValueError, match="float32 or float64"):
        eigen_mog_identity({
            "eigen_mog_sidecar": str(path),
            "eigen_mog_support_radius": 20,
            "eigen_mog_compute_dtype": "bfloat16",
        })
    _, identity = eigen_mog_identity({
        "eigen_mog_sidecar": str(path),
        "eigen_mog_support_radius": 20,
    })
    gaussian = operator_config_for_trainer(SimpleNamespace(
        psf_model="gaussian", analytic_images_enabled=True,
        analytic_image_quadrature_order=1,
    ))
    current = operator_config_for_trainer(SimpleNamespace(
        psf_model="eigen_mog", eigen_mog_config=identity,
        analytic_images_enabled=True, analytic_image_quadrature_order=1,
    ))
    checkpoint = {"operator_config": gaussian}
    with pytest.raises(ValueError, match="allow_psf_model_change"):
        validate_checkpoint_operator_config(
            checkpoint, current_operator_config=current, context="test", resume=True
        )
    assert validate_checkpoint_operator_config(
        checkpoint, current_operator_config=current, context="test", resume=True,
        allow_psf_model_change=True,
    )
    changed = dict(current)
    changed["eigen_mog_config"] = dict(identity, support_radius=16.0)
    with pytest.raises(ValueError, match="eigen_mog_config"):
        validate_checkpoint_operator_config(
            {"operator_config": current}, current_operator_config=changed,
            context="test", resume=True,
        )
