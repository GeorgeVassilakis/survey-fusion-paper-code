# Export validation

## Test suite

Run on 17 November 2026 on Linux x86-64, in a virtual environment with an editable installation of this distribution.

| Package | Imported version |
| --- | --- |
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| SciPy | 1.17.1 |
| Astropy | 7.2.0 |
| Hydra | 1.3.2 |
| Matplotlib | 3.10.9 |
| pandas | 3.0.3 |
| tqdm | 4.67.3 |
| Pillow | 12.2.0 |
| pytest | 9.0.3 |

- With no GPU visible: **1,277 passed, 183 skipped**. Every skipped check is a CUDA-specific path.
- With one NVIDIA B200 visible: **1,463 passed, none skipped**.
- Compilation of Python source, scripts, and tests.

The suite includes the artifact setup checks: local configuration generation and Hydra composition without starting a fit, comparison of the paper science settings with the original resolved canonical configuration, and joint and images-only membership and mask bindings.

## Export checks

- All 74 runtime files compared with the recorded source commit. After the documented naming substitutions, each has the same syntax tree as its source once string literals are set aside.
- Publication-content check for obsolete branding, private installation paths, credentials, oversized files, and unrelated development artifacts.

## Data checks

Run on 16 November 2026 on macOS arm64.

- Representative SDSS frame retrieval, decompression, and rectangular cutout reconstruction. The downloaded cutout's pixels match the original input exactly, and its sampled WCS coordinates have zero difference.
- Representative spectral FITS retrieval from all three `run2d` families in the inventory: `26`, `104`, and `v5_13_2`.
- Lossless mask recompression: all arrays compared element by element with the original mask archive.

## Scope

The full survey download, billion-measurement preparation, 11-epoch training, and HSC/DESI analyses were not rerun for this export. Reference results and the quoted B200 runtime come from the original paper workflow. These checks establish the contents of the exported artifact and its unit-level behavior on CPU and on one GPU. They do not establish a fresh reproduction of every paper result or bitwise agreement across GPU/library versions.
