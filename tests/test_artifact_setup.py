"""Verify the exported recipe against the recorded run and exercise portable input setup."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest

from survey_fusion.data.fpm_masks import check_sidecar_binding
from survey_fusion.data.prepared import prepared_digest, prepare_seed

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_recipe_keeps_the_actual_runs_science(tmp_path):
    output = tmp_path / 'local_joint.yaml'
    result = script('configure_paper').configure(tmp_path, tmp_path / 'prepared', output)
    actual = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
    reference = OmegaConf.to_container(OmegaConf.load(ROOT / 'provenance/original_resolved_config.yaml'), resolve=True)
    for group in ('model', 'physics', 'calibration', 'postprocess'):
        assert actual[group] == reference[group], group
    logistics = {'eigen_mog_sidecar', 'image_pixel_mask', 'field_overlap_table',
                 'checkpoint_archive_dir', 'checkpoint_archive_drain_timeout_s'}
    assert {k:v for k,v in actual['trainer'].items() if k not in logistics} == {
        k:v for k,v in reference['trainer'].items() if k not in logistics}
    assert actual['run']['seed'] == reference['run']['seed'] == 123
    assert len(actual['trainer']['exclude_spectrum_labels']) == 89
    assert len(actual['trainer']['exclude_exposure_labels']) == 10
    assert actual['data']['prepare_seed'] == 42
    assert actual['data']['expected_prepared_digest'] == result['prepared_digest']
    digest = prepared_digest(actual['data'], seed=prepare_seed(actual['data']),
                             bands=list(actual['physics']['bandpasses']),
                             manifest_path=actual['data']['scene_manifest_path'], holdout_band=None)
    assert digest == result['prepared_digest']
    manifest = json.loads((ROOT / 'data/paper/scene_manifest.json').read_text())
    labels = {i:row['label'] for i,row in enumerate(manifest['image_exposures'])}
    check_sidecar_binding(result['mask_descriptor'], store_digest=digest, image_labels=labels)
    with pytest.raises(ValueError, match='was built for store'):
        check_sidecar_binding(result['mask_descriptor'], store_digest='another', image_labels=labels)


def test_images_only_excludes_the_retained_spectra_and_keeps_imaging_split(tmp_path):
    setup = script('configure_paper')
    joint = tmp_path / 'joint.yaml'
    images = tmp_path / 'images.yaml'
    setup.configure(tmp_path, tmp_path / 'prepared', joint)
    setup.configure(tmp_path, tmp_path / 'prepared', images, seed=7, images_only=True)
    a, b = (OmegaConf.load(p) for p in (joint, images))
    assert len(b.trainer.exclude_spectrum_labels) == 889
    assert set(a.trainer.exclude_spectrum_labels) <= set(b.trainer.exclude_spectrum_labels)
    assert a.trainer.exclude_exposure_labels == b.trainer.exclude_exposure_labels
    assert b.run.seed == 7
    assert a.data.expected_prepared_digest == b.data.expected_prepared_digest


def test_setup_refuses_to_overwrite_configuration(tmp_path):
    output = tmp_path / 'existing.yaml'
    output.write_text('preserve me\n')
    with pytest.raises(FileExistsError):
        script('configure_paper').configure(tmp_path, tmp_path / 'prepared', output)
    assert output.read_text() == 'preserve me\n'


def test_download_inventory_is_complete_and_local(tmp_path):
    rows = script('download_sdss').inventory(tmp_path)
    assert sum(r['kind'] == 'image' for r in rows) == 410
    assert sum(r['kind'] == 'spectrum' for r in rows) == 910
    assert len({r['destination'] for r in rows}) == 1320
    assert all(Path(r['destination']).is_relative_to(tmp_path) for r in rows)
    assert all(r['url'].startswith('https://data.sdss.org/sas/dr17/') for r in rows)
    by_label = {r['label']: r for r in rows}
    assert '/eboss/spectro/redux/v5_13_2/' in by_label['S4737-55630-0367']['url']
    assert '/sdss/spectro/redux/104/' in by_label['S3154-54821-0192']['url']
    assert '/sdss/spectro/redux/26/' in by_label['S0500-51994-0475']['url']


def test_cutout_preserves_pixels_and_sky_positions(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10., 8.]
    wcs.wcs.cdelt = [-.00011, .00011]
    wcs.wcs.crval = [150., 2.]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
    source = tmp_path / 'frame.fits'
    output = tmp_path / 'cutout.fits'
    fits.PrimaryHDU(image, wcs.to_header()).writeto(source)
    script('download_sdss').make_cutout(source, output, dict(r0=3, c0=7, height=9, width=11))
    with fits.open(output) as hdul:
        np.testing.assert_array_equal(hdul[0].data, image[3:12, 7:18])
        np.testing.assert_allclose(WCS(hdul[0].header).all_pix2world([[0.,0.],[10.,8.]], 0),
                                   wcs.all_pix2world([[7.,3.],[17.,11.]], 0), atol=1e-12, rtol=0)
