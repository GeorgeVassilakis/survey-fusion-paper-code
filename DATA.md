# Data and supporting metadata

The included `data/paper/scene_manifest.json` lists 410 SDSS image exposures and 910 candidate spectra in the COSMOS region. The loader's quality and imaging-coverage selections retain 889 spectra. The joint-fit configuration excludes ten images and 89 retained spectra from the objective; the remaining 400 images and 800 spectra define the joint fit.

The inputs originate from the [SDSS Science Archive Server](https://data.sdss.org/sas/dr17/) and its [imaging documentation](https://www.sdss4.org/dr17/imaging/images/). The manifest identifies runs, camera columns, fields, plates, MJDs, and fibers. The spectral metadata include their source URLs and header identities.

## Included files

- `scene_manifest.json`: exposure order, field geometry, image filenames, and candidate spectra. Original installation paths have been made relative to the observation root.
- `spectral_response.json`: the declared Gaussian seeing, point-source calibration convention, and exposure-level provenance used for spectral responses. Internal recovery paths are recorded as archival filenames; the measured response values are retained.
- `eigen_mog.npz`: signed Gaussian expansions of the image PSF eigenimages.
- `image_pixel_mask.json` and `fpm_mask.npz`: pixel-mask geometry and indices for the INTERP, SATUR, CR, and GHOST planes. The NPZ is losslessly recompressed; the mask values are unchanged.
- `field_overlap_table.json`: duplicate-detector-sample geometry for adjacent fields. This is used to exclude withheld samples consistently.

The original input hashes are in `provenance/paper_run.json`. Making manifest paths portable and recompressing the mask changes file identities. The bundled descriptors refer to the exported files; `scripts/configure_paper.py` additionally binds the pixel masks to the locally configured prepared store. Neither operation changes the measured values, PSF coefficients, mask indices, split labels, or overlap pairs.

## Filter curves

The 19 transmission curves in `data/filters/` are listed in `data/registry/mbr.csv` and were retrieved from the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/). The registry records each curve's identifier, source URL, and retrieval date.

## Obtain observations

`scripts/download_sdss.py --output observations` writes a JSON inventory of public URLs and local destinations without downloading. Add `--download` to fetch the files. Image frames are downloaded from SDSS DR17's `eboss/photoObj/frames/301` directory and decompressed; exact rectangular cutouts are reconstructed from the pixel-mask geometry. Spectra use the manifest's `run2d`, plate, MJD, and fiber identifiers in the DR17 SDSS/BOSS `spectra/lite` archives.

Existing files are reused. Network failures stop the helper and identify the failing URL. The helper supports `--limit N` for an initial subset of images and spectra. Subsets are useful for inspecting file formats; the full paper recipe requires the full inventory.

The release checks include representative public-file retrieval and cutout reconstruction, rather than a fresh download and preprocessing of the entire dataset. Public archive availability can change. The manifest and original file identities should be used when checking a full reconstruction.

## Other observations and fitted models

HSC PDR3, DESI DR1, Gaia cross-check data, and fitted model checkpoints are not distributed here. `results/` contains numerical summaries from the original analysis, including the HSC aperture and DESI population comparisons. These summaries do not constitute a release of the external-survey evaluation pipeline or its observations.

The canonical joint checkpoint is identified by SHA-256 `c91c0a8d22bcb4888bd32568f878f54d3ff43db75eb75d5773b49edfb2929408`. A public checkpoint download is not supplied by this repository.
