"""List or fetch the SDSS observations named in the paper inventory."""
from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'https://data.sdss.org/sas/dr17'


def inventory(output: Path, limit: int | None = None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive.')
    manifest = json.loads((ROOT / 'data/paper/scene_manifest.json').read_text())
    geometry = json.loads((ROOT / 'data/paper/image_pixel_mask.json').read_text())
    by_label = {row['label']: row for row in geometry['exposures']}
    rows = []
    for row in manifest['image_exposures'][:limit]:
        frame_name = Path(row['frame_path']).name
        rows.append({'kind': 'image', 'label': row['label'],
                     'url': f"{ARCHIVE}/eboss/photoObj/frames/301/{row['run']}/{row['camcol']}/{frame_name}.bz2",
                     'destination': str(output / row['frame_path']),
                     'cutout': str(output / row['cutout_path']), 'geometry': by_label[row['label']]})
    for row in manifest['spectra'][:limit]:
        name = Path(row['path']).name
        survey = 'eboss' if row['run2d'].startswith('v5_') else 'sdss'
        rows.append({'kind': 'spectrum', 'label': row['label'],
                     'url': f"{ARCHIVE}/{survey}/spectro/redux/{row['run2d']}/spectra/lite/{int(row['plate']):04d}/{name}",
                     'destination': str(output / row['path'])})
    return rows


def fetch(row: dict) -> None:
    destination = Path(row['destination'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.partial')
        try:
            with urllib.request.urlopen(row['url'], timeout=120) as response, temporary.open('wb') as out:
                if row['kind'] == 'image':
                    with bz2.BZ2File(response) as decompressed:
                        shutil.copyfileobj(decompressed, out)
                else:
                    shutil.copyfileobj(response, out)
            # Validate the expected FITS container before accepting a completed download.
            from astropy.io import fits
            with fits.open(temporary) as hdul:
                hdul.verify('exception')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if row['kind'] == 'image':
        make_cutout(destination, Path(row['cutout']), row['geometry'])


def make_cutout(frame: Path, output: Path, geometry: dict) -> None:
    if output.exists():
        return
    from astropy.io import fits
    from astropy.wcs import WCS
    output.parent.mkdir(parents=True, exist_ok=True)
    r0, c0, height, width = (int(geometry[k]) for k in ('r0', 'c0', 'height', 'width'))
    with fits.open(frame) as hdul:
        image = hdul[0].data[r0:r0 + height, c0:c0 + width].copy()
        if image.shape != (height, width):
            raise ValueError(f"Cutout geometry extends outside {frame}")
        # Match the original rectangular crop and its WCS. The full frame supplies sky/calibration HDUs.
        header = WCS(hdul[0].header).slice((slice(r0, r0 + height), slice(c0, c0 + width))).to_header()
        if 'BUNIT' in hdul[0].header:
            header['BUNIT'] = hdul[0].header['BUNIT']
        temporary = output.with_suffix(output.suffix + '.partial')
        try:
            fits.PrimaryHDU(image, header).writeto(temporary, overwrite=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    rows = inventory(output, args.limit)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'download_inventory.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f"Inventory: {len(rows)} files in {output / 'download_inventory.json'}", flush=True)
    if args.download:
        for row in rows:
            print(f"Fetching {row['label']}: {row['url']}", flush=True)
            fetch(row)
    else:
        print('Add --download to retrieve observations; no survey files were downloaded.')


if __name__ == '__main__':
    main()
