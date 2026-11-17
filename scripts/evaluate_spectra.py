"""Render all retained SDSS fibers from a local run and record fitted/withheld membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_holdout_exposures import load_trainer
from survey_fusion.evaluation.diagnostics import write_fiber_spectrum_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg, store, trainer, entry, checkpoint, operator, reinterpreted = load_trainer(
        args.run_dir, args.output, args.prepared_dir, args.device,
    )
    fitted = set(trainer.spec_exposure_ids)
    trainer.spec_exposure_ids = sorted(store.spec_diagnostics_by_exposure)
    spectra = trainer.predict_fiber_spectra(include_additive=False, max_spectra=None)
    write_fiber_spectrum_diagnostics(
        fiber_spectra=spectra, spec_diagnostics_by_exposure=store.spec_diagnostics_by_exposure,
        use_flux_density=trainer.use_flux_density, run_dir=args.output,
        max_overlay_spectra=12, max_individual_spectra=0, include_additive=False,
        spec_additive_by_exposure=getattr(store, 'spec_additive_by_exposure', None),
        save_fiber_spectra=True, spec_sigma_by_exposure=entry._fiber_sigma_by_exposure(store),
    )
    membership = {str(eid): {'label': store.spec_label_by_exposure[eid],
                            'fitted': eid in fitted} for eid in spectra}
    (args.output / 'membership.json').write_text(json.dumps(membership, indent=2) + '\n')
    print(f"Rendered {len(spectra)} spectra; membership and diagnostics: {args.output}")


if __name__ == '__main__':
    main()
