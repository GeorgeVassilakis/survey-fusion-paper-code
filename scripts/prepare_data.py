"""Build the prepared scene+datum cache from a CPU job, so GPU runs start on a
warm cache instead of spending their allocation parsing FITS files.

Usage (from any checkout; paths resolve relative to this file's repo):
  python scripts/prepare_data.py <config-name> [hydra overrides...]

Uses the same code path as training (train.prepare_scene_and_datums), so the
digest is identical by construction.
"""

import os
import sys
import time


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    os.chdir(repo_root)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    cfg_name = sys.argv[1]
    overrides = sys.argv[2:]
    root = os.environ.get("SURVEY_FUSION_OBSERVATION_ROOT")
    if root:
        overrides = [f"data.observation_root={root}", *overrides]
    with initialize_config_dir(version_base=None, config_dir=os.path.join(repo_root, "configs")):
        cfg = compose(config_name=cfg_name, overrides=overrides)

    wandb_run = None
    if os.environ.get("WANDB_MODE", "disabled") != "disabled":
        import wandb

        wandb_run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "survey-fusion"),
            name=os.environ.get("WANDB_RUN_NAME", f"prepare_{cfg_name}"),
            mode=os.environ["WANDB_MODE"],
            config={"config_name": cfg_name, "overrides": overrides},
            settings=wandb.Settings(save_code=True),
        )

    import torch
    import train as train_mod
    from survey_fusion.data.bandpasses import load_bandpasses
    from survey_fusion.data.prepared import prepare_seed

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True, throw_on_missing=True)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True, throw_on_missing=True)
    paths = train_mod._resolve_paths(cfg)
    bandpasses = load_bandpasses(paths["registry_path"], cfg.physics["bandpasses"])
    holdout_band = train_mod._resolve_holdout_band(cfg)

    t0 = time.time()
    train_mod.prepare_scene_and_datums(
        data_cfg, bandpasses, trainer_cfg=trainer_cfg,
        seed=prepare_seed(data_cfg), device=torch.device("cpu"),
        holdout_band=holdout_band,
    )
    elapsed = time.time() - t0
    print(f"[prepare_data] done in {elapsed:.0f}s", flush=True)
    if wandb_run is not None:
        wandb_run.log({"prepare/elapsed_sec": elapsed})
        wandb_run.finish()


if __name__ == "__main__":
    main()
