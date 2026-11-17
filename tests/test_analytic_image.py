import torch

from survey_fusion.models.splat_field import SplatField
from survey_fusion.pgm.analytic_image import analytic_band_prediction


def test_analytic_matches_mc_within_tolerance():
    torch.manual_seed(0)
    f = SplatField(
        n_splats=16,
        n_templates=4,
        loglam_grid=torch.linspace(-1, 1, 32),
        init_scale=0.05,
    )
    with torch.no_grad():
        f.raw_flux.fill_(0.5)
        f.sed_shift.copy_(0.05 * torch.randn(16))
    px = torch.linspace(-0.5, 0.5, 20)
    xx, yy = torch.meshgrid(px, px, indexing="ij")
    psf_sigma = 0.02
    bandpass_ll = torch.linspace(-0.3, 0.1, 40)

    with torch.no_grad():
        pred = analytic_band_prediction(
            f, xx.flatten(), yy.flatten(), psf_sigma, bandpass_ll
        )

        n_mc = 4000
        offs = torch.randn(n_mc, 2) * psf_sigma
        ll = bandpass_ll[torch.randint(0, bandpass_ll.numel(), (n_mc,))]
        mc = torch.stack(
            [
                f(x + offs[:, 0], y + offs[:, 1], ll).mean()
                for x, y in zip(xx.flatten(), yy.flatten())
            ]
        )
    err = (pred - mc).abs().max() / mc.abs().max()
    assert err < 0.05, f"analytic vs MC mismatch: {err:.4f}"


def test_analytic_zero_psf_matches_pointwise_band_average():
    torch.manual_seed(1)
    f = SplatField(
        n_splats=8, n_templates=4, loglam_grid=torch.linspace(-1, 1, 32),
        init_scale=0.1,
    )
    x = torch.linspace(-0.4, 0.4, 15)
    y = torch.zeros(15)
    bandpass_ll = torch.linspace(-0.2, 0.2, 16)
    with torch.no_grad():
        pred = analytic_band_prediction(f, x, y, 0.0, bandpass_ll)
        direct = torch.stack(
            [f(x, y, ll.expand_as(x)) for ll in bandpass_ll]
        ).mean(0)
    assert torch.allclose(pred, direct, rtol=1e-3, atol=1e-6)
