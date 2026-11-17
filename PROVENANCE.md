# Snapshot provenance

This is a new repository with a flattened history. It exports selected files from source commit `79b9d2db152f89e36a419b77a70134622ab85c28`, the source recorded for the paper's canonical joint fit. Later development is outside this artifact.

`provenance/source_files.json` records the original and exported SHA-256 of every copied file. The Python package and its environment-variable prefix were renamed to `survey_fusion` and `SURVEY_FUSION`. Package version lookup was changed to this distribution's name. Comments, docstrings, and a few message strings were edited to remove references to internal infrastructure and working notes. Apart from those changes, the runtime under `src/` is the source snapshot's implementation. Every Python file there has the same syntax tree as its source once string literals are set aside.

The source's generic paper configuration defaults to seed 42. `configs/train_paper.yaml` selects seed 123 for the displayed canonical run. `provenance/original_resolved_config.yaml` records the original run's settings with private filesystem paths and the host-specific platform name replaced. The CUDA platform profile preserves the original runtime policy while omitting host-specific filesystem defaults.

`scripts/configure_paper.py` resolves local paths, recomputes the path-dependent prepared-cache identity, and creates the corresponding mask descriptor. The base recipe retains its original digest for historical traceability. The local configuration uses the new digest and continues to enforce it during preparation and fitting. This is a new local input identity, not a declaration of bitwise equivalence to the original run.

The original checkpoint includes operator/input identities. Its identity checks remain in the implementation. Package renaming and portable input records do not make an original checkpoint automatically resumable on a new installation.

Campaign launchers, supervisors, sweep scripts, performance experiments, an earlier product renderer and its verifier, private working notes, historical alternative configurations, and tests specific to those omitted tools were excluded. Tests of the retained model and operators were preserved. The one environment-knob test for an omitted sweep configuration was removed from the refinement tests. Artifact setup checks compare the exported recipe with the actual resolved canonical configuration. The default logging project name in `scripts/prepare_data.py` was replaced.

The `results/` files are reference summaries from the original paper workflow. Private artifact paths and internal bookkeeping notes have been replaced by descriptive names. `results/numbers_hsc_aperture.json` records digests of its external protocol and calibration-evidence files. Those digests refer to the original files, and the calibration notes shown here carry the edits just described. These summaries were not recomputed during this export.
