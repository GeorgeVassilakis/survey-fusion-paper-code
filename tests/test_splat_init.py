import torch

from survey_fusion.models.splat_init import peaks_from_image


class _LinearObservationWCS:
    def __init__(self, x_offset: float = 0.0, y_offset: float = 0.0):
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)

    def pixel_to_omega_xyz(self, x, y):
        import numpy as np

        return np.concatenate(
            [x + self.x_offset, y + self.y_offset, np.ones_like(x)], axis=1
        )


class _LinearModelWCS:
    @staticmethod
    def omega_xyz_to_norm_torch(omega):
        return 0.001 * omega[:, 0], 0.001 * omega[:, 1]


def _peaks_image(shape, peaks):
    yy, xx = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
    )
    image = torch.full(shape, -1.0e6, dtype=torch.float32)
    for y, x, value in peaks:
        surface = float(value) - (yy - int(y)).float() ** 2 - (xx - int(x)).float() ** 2
        image = torch.maximum(image, surface)
    return image


def _linear_store(exposures):
    from types import SimpleNamespace

    return SimpleNamespace(
        image_band_by_exposure={i: band for i, (band, _, _) in enumerate(exposures)},
        image_source_by_exposure={i: image for i, (_, image, _) in enumerate(exposures)},
        exposures={
            i: SimpleNamespace(obs_wcs=obs_wcs, wcs=_LinearModelWCS())
            for i, (_, _, obs_wcs) in enumerate(exposures)
        },
    )


def test_peaks_found_on_synthetic_image():
    torch.manual_seed(1)
    img = torch.zeros(64, 64)
    img[10, 20] = 5.0
    img[40, 45] = 3.0
    img += 0.01 * torch.randn(64, 64)
    xy = peaks_from_image(img, n_peaks=2, min_sep=3)
    got = {tuple(map(int, p)) for p in xy.tolist()}
    assert got == {(10, 20), (40, 45)}


def test_min_sep_suppresses_neighbors():
    img = torch.zeros(32, 32)
    img[16, 16] = 5.0
    img[16, 18] = 4.0
    img[5, 5] = 3.0
    xy = peaks_from_image(img, n_peaks=2, min_sep=3)
    got = {tuple(map(int, p)) for p in xy.tolist()}
    assert got == {(16, 16), (5, 5)}


def test_requests_more_peaks_than_exist():
    img = torch.zeros(16, 16)
    img[8, 8] = 1.0
    xy = peaks_from_image(img, n_peaks=5, min_sep=2)
    assert xy.shape[0] >= 1
    assert (8.0, 8.0) in {tuple(p) for p in xy.tolist()}


def test_seed_positions_places_splats_at_peaks():
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_positions

    torch.manual_seed(0)
    f = SplatField(n_splats=10, n_templates=4, loglam_grid=torch.linspace(-1, 1, 16))
    peaks = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    seed_positions(f, peaks, floor_frac=0.2)
    seeded = f.mu[:8]
    d = torch.cdist(seeded, peaks).min(dim=1).values
    assert (d < 0.01).all()


def test_seed_from_store_places_splats_in_field():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_analytic_trainer_dispatch import _bandpasses, _realistic_scene
    from survey_fusion.data.adapters.sdss import build_sdss_datums
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    store = build_sdss_datums(
        _realistic_scene(), _bandpasses(),
        psf_fwhm_arcsec=1.0, device=torch.device("cpu"),
        spectral_response_mode="legacy_imaging_median",
    )
    f = SplatField(n_splats=32, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8))
    n_used = seed_from_store(f, store, prefer_band="r")
    n_r_exposures = sum(1 for b in store.image_band_by_exposure.values() if b == "r")
    assert n_used == n_r_exposures
    assert torch.isfinite(f.mu).all()


def test_seed_from_store_uses_observation_pixel_scale():
    from types import SimpleNamespace

    import numpy as np

    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    class ObservationWCS:
        @staticmethod
        def pixel_to_omega_xyz(x, y):
            return np.concatenate([x, y, np.ones_like(x)], axis=1)

    class ModelWCS:
        @staticmethod
        def omega_xyz_to_norm_torch(omega):
            return 0.001 * omega[:, 0], 0.001 * omega[:, 1]

    torch.manual_seed(4)
    image = 0.01 * torch.randn(16, 16)
    image[4, 5] = 3.0
    image[11, 12] = 2.0
    store = SimpleNamespace(
        image_band_by_exposure={0: "r"},
        image_source_by_exposure={0: image},
        exposures={
            0: SimpleNamespace(obs_wcs=ObservationWCS(), wcs=ModelWCS())
        },
    )
    field = SplatField(
        n_splats=4, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    seed_from_store(field, store)

    expected = torch.full_like(field.log_scales, 0.001)
    assert torch.allclose(torch.exp(field.log_scales), expected)


def test_multiband_dedupe_keeps_one_brighter_detection_within_radius():
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_store

    store = _linear_store([
        ("r", _peaks_image((13, 13), [(6, 4, 3.0)]), _LinearObservationWCS()),
        ("g", _peaks_image((13, 13), [(6, 6, 9.0)]), _LinearObservationWCS()),
    ])
    field = SplatField(
        n_splats=2, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    torch.manual_seed(17)
    seed_from_store(field, store, prefer_band=None)

    detections = torch.tensor([[0.004, 0.006], [0.006, 0.006]])
    exact_detections = (torch.cdist(field.mu, detections) < 1.0e-7).any(dim=1)
    assert int(exact_detections.sum()) == 1
    assert torch.allclose(field.mu[0], detections[1], atol=1.0e-7, rtol=0.0)
    with torch.no_grad():
        pred = field(detections[1, :1], detections[1, 1:], torch.zeros(1))
    assert torch.allclose(pred, torch.tensor([9.0]), rtol=0.05, atol=0.02)


def test_multiband_seed_count_is_same_order_as_r_only(monkeypatch):
    from types import SimpleNamespace

    from survey_fusion.models import splat_init

    image = _peaks_image((16, 16), [(4, 4, 5.0), (11, 11, 4.0)])
    store = _linear_store([
        ("r", image, _LinearObservationWCS()),
        ("g", image + 1.0, _LinearObservationWCS()),
        ("i", image + 2.0, _LinearObservationWCS()),
    ])
    counts = []

    def capture(_model, norm_xy, _peak_values, **_kwargs):
        counts.append(int(norm_xy.shape[0]))
        return counts[-1]

    monkeypatch.setattr(splat_init, "seed_calibrated", capture)
    model = SimpleNamespace(mu=torch.zeros(64, 2))
    splat_init.seed_from_store(model, store, prefer_band="r")
    splat_init.seed_from_store(model, store, prefer_band=None)

    r_only, multiband = counts
    assert 0.5 * r_only <= multiband <= 2.0 * r_only


class _LinearSkyWCS:
    @staticmethod
    def all_pix2world(x, y, _origin):
        import numpy as np

        return np.asarray(x) / 3600.0, np.asarray(y) / 3600.0

    @staticmethod
    def all_world2pix(ra, dec, _origin):
        import numpy as np

        return (
            np.asarray(ra) * 3600.0,
            np.asarray(dec) * 3600.0,
        )


def _catalog_store(images):
    from types import SimpleNamespace

    from survey_fusion.utils.wcs import AstropyWCSAdapter, TangentPlaneWCS

    raw_wcs = _LinearSkyWCS()
    model_wcs = TangentPlaneWCS(
        ra0_rad=0.0, dec0_rad=0.0, pixel_scale_arcsec=1.0,
        width=8, height=8, ref_x=0.0, ref_y=0.0,
    )
    return SimpleNamespace(
        image_band_by_exposure={i: band for i, (band, _) in enumerate(images)},
        image_source_by_exposure={i: image for i, (_, image) in enumerate(images)},
        exposures={
            i: SimpleNamespace(
                obs_wcs=AstropyWCSAdapter(raw_wcs), wcs=model_wcs,
                image_height=8, image_width=8,
            )
            for i in range(len(images))
        },
    )


def test_seed_from_catalog_maps_known_radec_to_normalized_positions(tmp_path):
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_catalog

    raw_wcs = _LinearSkyWCS()
    x = torch.tensor([2.0, 6.0], dtype=torch.float64).numpy()
    y = torch.tensor([3.0, 5.0], dtype=torch.float64).numpy()
    ra, dec = raw_wcs.all_pix2world(x, y, 0)
    catalog = tmp_path / "sources.csv"
    catalog.write_text(
        "ra,dec,flux\n"
        f"{ra[0]:.12f},{dec[0]:.12f},4.0\n"
        f"{ra[1]:.12f},{dec[1]:.12f},7.0\n",
        encoding="utf-8",
    )
    store = _catalog_store([("r", torch.zeros(8, 8))])
    field = SplatField(
        n_splats=2, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    n_rows = seed_from_catalog(field, store, catalog)

    expected = torch.tensor([
        [2.0 * 2.0 / 7.0 - 1.0, 2.0 * 3.0 / 7.0 - 1.0],
        [2.0 * 6.0 / 7.0 - 1.0, 2.0 * 5.0 / 7.0 - 1.0],
    ])
    assert n_rows == 2
    assert torch.allclose(field.mu, expected, atol=2.0e-5, rtol=0.0)


def test_seed_from_catalog_uses_median_covering_exposure_flux(tmp_path):
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_catalog

    raw_wcs = _LinearSkyWCS()
    ra, dec = raw_wcs.all_pix2world([2.0], [3.0], 0)
    catalog = tmp_path / "sources.csv"
    catalog.write_text(f"ra,dec\n{ra[0]:.12f},{dec[0]:.12f}\n", encoding="utf-8")
    first = torch.zeros(8, 8)
    second = torch.zeros(8, 8)
    first[3, 2] = 4.0
    second[3, 2] = 8.0
    store = _catalog_store([("r", first), ("g", second)])
    field = SplatField(
        n_splats=1, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    seed_from_catalog(field, store, catalog)

    with torch.no_grad():
        pred = field(field.mu[:, 0], field.mu[:, 1], torch.zeros(1))
    assert torch.allclose(pred, torch.tensor([6.0]), rtol=0.05, atol=0.02)


def test_seed_from_catalog_rejects_rows_outside_all_footprints(tmp_path):
    import pytest

    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_from_catalog

    catalog = tmp_path / "sources.csv"
    catalog.write_text(
        "ra,dec,flux\n0.0005,0.0005,2.0\n1.0,0.0,3.0\n",
        encoding="utf-8",
    )
    store = _catalog_store([("r", torch.zeros(8, 8))])
    field = SplatField(
        n_splats=2, n_templates=3, loglam_grid=torch.linspace(-1, 1, 8),
        init_scale=0.25,
    )

    with pytest.raises(ValueError, match=r"CSV rows \[3\]"):
        seed_from_catalog(field, store, catalog)


def test_seed_calibrated_matches_peak_amplitude():
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(0)
    f = SplatField(n_splats=16, n_templates=4, loglam_grid=torch.linspace(-1, 1, 16))
    peaks = torch.tensor([[0.2, -0.1], [-0.4, 0.3]])
    vals = torch.tensor([2.5, 0.8])
    n = seed_calibrated(f, peaks, vals, loglam_norm=0.0)
    assert n == 2
    with torch.no_grad():
        pred = f(peaks[:, 0], peaks[:, 1], torch.zeros(2))
    assert torch.allclose(pred, vals, rtol=0.05, atol=0.02), pred


def test_every_seeded_splat_stays_trainable():
    """Surplus splats used to be parked at raw_flux=-30, where softplus' is
    1e-13: no gradient reaches them and, with no densification pass to revive
    them, two thirds of the model sat dead for the whole run."""
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(0)
    f = SplatField(n_splats=60, n_templates=4, loglam_grid=torch.linspace(-1, 1, 16))
    peaks = torch.tensor([[0.2, -0.1], [-0.4, 0.3], [0.0, 0.6]])
    seed_calibrated(f, peaks, torch.tensor([2.5, 0.8, 1.2]))
    assert (torch.sigmoid(f.raw_flux) > 1e-8).all()
    f(peaks[:, 0], peaks[:, 1], torch.zeros(3)).sum().backward()
    assert (f.raw_flux.grad.abs() > 0).all()


def test_many_splats_per_peak_still_reproduce_amplitude():
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(3)
    f = SplatField(n_splats=40, n_templates=4, loglam_grid=torch.linspace(-1, 1, 16))
    peaks = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    vals = torch.tensor([3.0, 1.0])
    seed_calibrated(f, peaks, vals)
    with torch.no_grad():
        pred = f(peaks[:, 0], peaks[:, 1], torch.zeros(2))
    assert torch.allclose(pred, vals, rtol=0.05), pred


def test_splats_sharing_a_peak_are_not_stacked():
    """Exactly coincident splats have identical gradients and never separate,
    the same symmetry trap that collapsed the SED dictionary."""
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(0)
    f = SplatField(n_splats=12, n_templates=4, loglam_grid=torch.linspace(-1, 1, 16))
    seed_calibrated(f, torch.tensor([[0.0, 0.0], [0.4, 0.4]]),
                    torch.tensor([1.0, 1.0]))
    sharing = f.mu[torch.arange(12) % 2 == 0]
    assert sharing.std(dim=0).min() > 0


def test_seeding_spans_multiple_scales():
    """All splats at one narrow width reproduce each source's peak pixel but not
    its wings: bright sources keep a positive halo in the image residual, and a
    3 arcsec fibre integrating the model collects 4-5x too little flux."""
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(0)
    f = SplatField(n_splats=32, n_templates=4,
                   loglam_grid=torch.linspace(-1, 1, 16), init_scale=0.01)
    seed_calibrated(f, torch.tensor([[0.0, 0.0], [0.3, -0.2]]),
                    torch.tensor([1.0, 2.0]), n_scales=4)
    widths = torch.exp(f.log_scales).mean(dim=1)
    assert widths.max() / widths.min() >= 7.9, widths.max() / widths.min()
    assert torch.unique(widths.round(decimals=6)).numel() == 4


def test_multi_scale_seeding_still_reproduces_peak_amplitude():
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    torch.manual_seed(1)
    f = SplatField(n_splats=40, n_templates=4,
                   loglam_grid=torch.linspace(-1, 1, 16), init_scale=0.01)
    peaks = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    vals = torch.tensor([3.0, 1.0])
    seed_calibrated(f, peaks, vals, n_scales=4)
    with torch.no_grad():
        pred = f(peaks[:, 0], peaks[:, 1], torch.zeros(2))
    assert torch.allclose(pred, vals, rtol=0.05), pred


def test_wider_seeding_carries_more_flux_through_an_aperture():
    """The point of multi-scale seeding: integrated flux inside an aperture
    several times the PSF width, which is what a fibre measures."""
    from survey_fusion.models.splat_field import SplatField
    from survey_fusion.models.splat_init import seed_calibrated

    peaks = torch.tensor([[0.0, 0.0]])
    vals = torch.tensor([1.0])
    grid = torch.linspace(-0.15, 0.15, 61)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    inside = (xx**2 + yy**2) <= 0.1**2

    flux = {}
    for n_scales in (1, 4):
        torch.manual_seed(2)
        f = SplatField(n_splats=16, n_templates=4,
                       loglam_grid=torch.linspace(-1, 1, 16), init_scale=0.01)
        seed_calibrated(f, peaks, vals, n_scales=n_scales)
        with torch.no_grad():
            img = f(xx.reshape(-1), yy.reshape(-1), torch.zeros(xx.numel()))
        flux[n_scales] = float(img.reshape(xx.shape)[inside].sum())
    assert flux[4] > 2.0 * flux[1], flux
