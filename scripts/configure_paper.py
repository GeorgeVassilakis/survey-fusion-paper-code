"""Write a local paper configuration and bind the masks to its prepared-store identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from survey_fusion.data.bandpasses import load_bandpasses
from survey_fusion.data.prepared import prepared_digest, prepare_seed

ROOT = Path(__file__).resolve().parents[1]


def configure(observation_root: Path, prepared_dir: Path, output: Path, *, seed: int = 123,
              images_only: bool = False) -> dict:
    observation_root, prepared_dir, output = (
        p.expanduser().resolve() for p in (observation_root, prepared_dir, output)
    )
    if not observation_root.is_dir():
        raise FileNotFoundError(f"Observation root does not exist: {observation_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing configuration: {output}")
    inputs = ROOT / 'data' / 'paper'
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        cfg = compose(config_name='train_paper')
    cfg.paths.observation_root = str(observation_root)
    cfg.paths.bandpass_root = str(ROOT / 'data')
    cfg.data.observation_root = str(observation_root)
    cfg.data.scene_manifest_path = str(inputs / 'scene_manifest.json')
    cfg.data.prepared_dir = str(prepared_dir)
    cfg.data.spectral_response_metadata_path = str(inputs / 'spectral_response.json')
    cfg.trainer.eigen_mog_sidecar = str(inputs / 'eigen_mog.npz')
    cfg.trainer.field_overlap_table = str(inputs / 'field_overlap_table.json')
    cfg.run.seed = seed
    cfg.run.run_name = ('images_only' if images_only else 'joint') + f'_s{seed}'
    if images_only:
        manifest = json.loads((inputs / 'scene_manifest.json').read_text())
        # The manifest also has rejected candidates; only retained labels are valid exclusions.
        metadata = json.loads((inputs / 'spectral_response.json').read_text())
        retained = {str(row['label']) for row in metadata if row.get('loader_retained')}
        candidates = {str(row['label']) for row in manifest['spectra']}
        if len(retained) != 889 or not retained <= candidates:
            raise ValueError('Spectral membership does not match the recorded 889-spectrum inventory.')
        cfg.trainer.exclude_spectrum_labels = sorted(retained)
    bands = list(load_bandpasses(ROOT / 'data/registry/mbr.csv', cfg.physics.bandpasses))
    data = OmegaConf.to_container(cfg.data, resolve=True)
    digest = prepared_digest(data, seed=prepare_seed(data), bands=bands,
                             manifest_path=cfg.data.scene_manifest_path, holdout_band=None)
    cfg.data.expected_prepared_digest = digest
    local_dir = ROOT / '.runtime' / output.stem
    local_dir.mkdir(parents=True, exist_ok=True)
    mask_path = local_dir / 'image_pixel_mask.json'
    mask = json.loads((inputs / 'image_pixel_mask.json').read_text())
    mask['store'] = f'prepared_{digest}'
    mask['arrays'] = str(inputs / 'fpm_mask.npz')
    mask_path.write_text(json.dumps(mask, indent=2) + '\n')
    cfg.trainer.image_pixel_mask = str(mask_path)
    # Resolve all interpolations now: this file must be usable independently of defaults.
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(resolved), output)
    return {'config': str(output), 'prepared_digest': digest, 'seed': seed,
            'images_only': images_only, 'mask_descriptor': str(mask_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-root', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/local_paper.yaml')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--images-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(configure(args.observation_root, args.prepared_dir, args.output,
                               seed=args.seed, images_only=args.images_only), indent=2))


if __name__ == '__main__':
    main()
