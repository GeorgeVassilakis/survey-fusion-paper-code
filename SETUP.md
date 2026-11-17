# Running the paper implementation

Technical notes for inspecting or rerunning the archived implementation. Release scope and limitations are described in [README.md](README.md).

## Installation

Use Python 3.12 and a virtual environment. Install a PyTorch build suitable for your hardware before installing this repository. Full-field training uses Linux and an NVIDIA CUDA GPU; the original run used one B200. CPU tests and small examples can run without CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
# Linux/CUDA execution also uses Triton:
python -m pip install -e '.[cuda]'
```

Dependency lower bounds are not a lockfile for the original training environment. See [VALIDATION.md](VALIDATION.md) for the environment and checks used for this export.

## Set up the paper inputs

The download helper writes an inventory by default. Downloading all survey files is a substantial transfer; use `--limit 1` for a small setup check.

```bash
python scripts/download_sdss.py --output observations
python scripts/download_sdss.py --output observations --download
python scripts/configure_paper.py --observation-root observations --prepared-dir prepared
```

The configuration helper writes `configs/local_paper.yaml` and a local pixel-mask descriptor. It retains the paper's science settings and split while binding the masks and prepared-cache digest to your filesystem. These local files are ignored by Git. It does not start training or download data.

## Prepare and train

Run these commands from the repository root:

```bash
python scripts/prepare_data.py local_paper
PYTHONHASHSEED=0 python src/train.py --config-name local_paper
```

The full prepared store contains about one billion measurements and needs substantial CPU memory and storage. The paper reports eight hours for the joint fit on one B200; that is the original measurement, not a benchmark of this export. The default run writes a timestamped directory under `outputs/` containing `config.yaml` and `model.pt`.

Seeds 42 and 7 use the same scientific recipe. To generate a separate local configuration, supply `--seed 42 --output configs/local_s42.yaml` to the configuration helper, then use `local_s42` in both preparation and training. For the images-only control, add `--images-only`; the helper excludes all spectra from the objective, including those already withheld by the joint split.

## Inspect and evaluate

```bash
python scripts/inspect_checkpoint.py outputs/RUN/model.pt
python scripts/evaluate_spectra.py --run-dir outputs/RUN --output outputs/spectra
python scripts/eval_holdout_exposures.py \
  --run-dir outputs/RUN --outdir outputs/held-images --save-arrays
```

The spectral helper renders all retained fibers and marks the fitted/withheld split in its output. These tools use the same instrument declarations as the local run. Existing checkpoints are bound to their recorded operators and inputs; the original private run's hashes cannot simply be assigned to relocated inputs. See [PROVENANCE.md](PROVENANCE.md).

## Checks

```bash
python -m pytest -q
```

CUDA-specific tests skip when suitable hardware is unavailable. Tests of omitted campaign launchers and historical configurations are not part of this artifact. The numerical implementation was retained without a scientific refactor.
