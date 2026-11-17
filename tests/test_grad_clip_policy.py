"""Per-splat parameters are exempt from global gradient clipping.

The global norm over a growing splat census scales with census size, so a
fixed grad_clip ceiling dilutes every row's update as the field grows and
pushes faint-row gradients below Adam's eps. Shared/calibration parameters
keep the standard global clip.
"""
import pytest
import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.training.datum_trainer import DatumTrainer


class _Shell:
    _base_model = DatumTrainer._base_model
    _refresh_clip_params = DatumTrainer._refresh_clip_params
    _gradient_norm = DatumTrainer._gradient_norm
    _clip_gradient_norm = DatumTrainer._clip_gradient_norm


def _shell_with_field():
    field = SplatField(
        n_splats=6,
        n_templates=2,
        loglam_grid=torch.linspace(-1.0, 1.0, 8),
        init_scale=0.05,
    )
    shell = _Shell()
    shell.model = field
    shell.spectral_calibrator = None
    shell.params = list(field.parameters())
    shell.grad_clip = 1.0
    shell._refresh_clip_params()
    return shell, field


def test_clip_set_excludes_per_splat_parameters():
    shell, field = _shell_with_field()
    splat_params = {
        id(p)
        for group in field.sparse_param_groups_gsplat()
        for p in group["params"]
    }
    assert splat_params.isdisjoint(shell._clip_param_ids)
    shared_params = {
        id(p) for group in field.shared_param_groups_gsplat() for p in group["params"]
    }
    assert shared_params <= shell._clip_param_ids


def test_clip_scales_shared_but_not_splat_gradients():
    shell, field = _shell_with_field()
    shared = next(iter(field.sed.parameters()))
    shared.grad = torch.full_like(shared, 10.0)
    field.mu.grad = torch.full_like(field.mu, 500.0)
    splat_before = field.mu.grad.clone()
    shared_norm = float(torch.linalg.vector_norm(shared.grad))

    preclip, postclip = shell._clip_gradient_norm()

    assert abs(float(preclip) - shared_norm) < 1e-4 * shared_norm
    assert float(postclip) <= 1.0 + 1e-6
    assert torch.equal(field.mu.grad, splat_before)
    assert float(torch.linalg.vector_norm(shared.grad)) <= 1.0 + 1e-5


def _host_side_clip(shell):
    """The host-decided clip the device-side one replaced, as the gate."""
    total_norm = float(shell._gradient_norm())
    max_norm = float(shell.grad_clip)
    scale = max_norm / (total_norm + 1.0e-6)
    if scale < 1.0:
        for parameter in shell.params:
            if id(parameter) not in shell._clip_param_ids:
                continue
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                gradient._values().mul_(scale)
            else:
                gradient.mul_(scale)
        return total_norm, total_norm * scale
    return total_norm, total_norm


@pytest.mark.parametrize("magnitude", [10.0, 1.0e-3])
def test_device_side_clip_matches_the_host_decision_bitwise(magnitude):
    """Clipping and non-clipping regimes must both reproduce the host path."""
    reference_shell, reference_field = _shell_with_field()
    candidate_shell, candidate_field = _shell_with_field()
    for field in (reference_field, candidate_field):
        for parameter in field.sed.parameters():
            parameter.grad = torch.full_like(parameter, magnitude)
        field.mu.grad = torch.full_like(field.mu, 500.0)

    expected_preclip, expected_postclip = _host_side_clip(reference_shell)
    preclip, postclip = candidate_shell._clip_gradient_norm()

    assert float(preclip) == expected_preclip
    assert float(postclip) == expected_postclip
    for reference_parameter, candidate_parameter in zip(
        reference_field.parameters(), candidate_field.parameters()
    ):
        if reference_parameter.grad is None:
            assert candidate_parameter.grad is None
            continue
        assert torch.equal(candidate_parameter.grad, reference_parameter.grad)


def test_device_side_clip_scales_sparse_gradient_values():
    parameter = torch.nn.Parameter(torch.zeros(4, 2))
    values = torch.full((2, 2), 10.0)
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]), values, size=(4, 2)
    ).coalesce()
    shell = _Shell()
    shell.params = [parameter]
    shell._clip_param_ids = {id(parameter)}
    shell.grad_clip = 1.0

    preclip, postclip = shell._clip_gradient_norm()

    expected_scale = 1.0 / (float(preclip) + 1.0e-6)
    torch.testing.assert_close(
        parameter.grad._values(),
        values * expected_scale,
        rtol=0.0,
        atol=1e-6,
    )
    assert float(postclip) <= 1.0 + 1e-6


def test_refresh_tracks_surgery_replacements():
    shell, field = _shell_with_field()
    old_ids = set(shell._clip_param_ids)
    with torch.no_grad():
        field.mu = torch.nn.Parameter(field.mu.data.clone())
    shell.params = list(field.parameters())
    shell._refresh_clip_params()
    assert id(field.mu) not in shell._clip_param_ids
    assert shell._clip_param_ids != old_ids or id(field.mu) not in old_ids
