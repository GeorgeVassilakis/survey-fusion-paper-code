import torch
from omegaconf import OmegaConf

from survey_fusion.models.splat_field import SplatField
from train import _init_model as init_train_model


def test_init_model_builds_splat_field_from_config():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "splat",
                "n_splats": 17,
                "n_templates": 5,
                "n_loglam_grid": 24,
                "init_extent": 0.5,
                "init_scale": 0.01,
            }
        }
    )
    model = init_train_model(cfg)
    assert isinstance(model, SplatField)
    assert model.mu.shape == (17, 2)
    assert model.sed_coeffs.shape == (17, 5)
    assert model.sed.loglam_grid.numel() == 24


def test_init_model_defaults_to_mlp_without_name():
    cfg = OmegaConf.create(
        {
            "model": {
                "hidden_dim": 8,
                "n_layers": 2,
                "softplus_beta": 2.0,
                "positional_encoding": None,
                "output_head": None,
            }
        }
    )
    model = init_train_model(cfg)
    assert not isinstance(model, SplatField)


def test_synthetic_recovery_two_sources():
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _run_synthetic_recovery()
    finally:
        torch.set_num_threads(n_threads)


def _run_synthetic_recovery():
    torch.manual_seed(0)
    truth = SplatField(
        n_splats=2, n_templates=4, loglam_grid=torch.linspace(-1, 1, 32)
    )
    with torch.no_grad():
        truth.mu.copy_(torch.tensor([[-0.3, 0.0], [0.4, 0.2]]))
        truth.raw_flux.fill_(2.0)
    n = 4096
    x = torch.rand(n) * 2 - 1
    y = torch.rand(n) * 2 - 1
    ll = torch.rand(n) * 2 - 1
    with torch.no_grad():
        target = truth(x, y, ll)

    model = SplatField(
        n_splats=16, n_templates=4, loglam_grid=torch.linspace(-1, 1, 32),
        init_scale=0.15,
    )
    # Mirror the production recipe: without decay the fit converges by ~step 400
    # and then diverges as the background term runs away (loss 1.6 -> 1400).
    steps, pos_lr0 = 1200, 5e-2
    groups = model.param_groups_gsplat()
    groups[0]["lr"] = pos_lr0
    opt = torch.optim.Adam(groups)
    for step in range(steps):
        opt.zero_grad()
        loss = ((model(x, y, ll) - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.param_groups[0]["lr"] = pos_lr0 * (0.01 ** (step / (steps - 1)))
    rel = loss.item() / (target.var().item() + 1e-12)
    assert rel < 0.08, f"synthetic fit failed: relative loss {rel:.3f}"
