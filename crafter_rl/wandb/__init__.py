"""The one reusable Weights & Biases convention for this repo.

Every training script calls :func:`init_run` instead of ``wandb.init`` directly,
so run naming, grouping, tags, and the local-files directory are consistent
across experiments and sweeps:

    name     = "{sweep_id}-{algo}-s{seed}"
    group    = "{sweep_id}/{algo}"
    job_type = "train"
    tags     = [sweep_id, algo, "env:crafter", exp_type]

Note: this module is named ``crafter_rl.wandb`` but the ``import wandb`` below
resolves to the real pip-installed package (absolute import), not this module.
"""


def init_run(*, algo, seed, sweep_id, exp_type, project, config,
             out_dir=None, name=None, notes=None):
    """Initialise and return a W&B run following the repo convention.

    Parameters
    ----------
    algo, seed, sweep_id, exp_type : run identity used for name/group/tags.
    project : W&B project name.
    config : dict of hyperparameters/metadata to record.
    out_dir : if given, passed as ``dir=`` so wandb's local files land in the
        trial's out dir (keeps artifacts co-located with the trial).
    name : override the default ``{sweep_id}-{algo}-s{seed}`` name.
    notes : free-text run notes.
    """
    import os
    import wandb  # absolute import -> the real pip package

    # wandb writes its local files to ``{dir}/wandb/``; if ``dir`` doesn't exist
    # yet it silently falls back to the cwd (repo root). Create it so runs stay
    # co-located with the trial that produced them.
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    run = wandb.init(
        project=project,
        name=name or f"{sweep_id}-{algo}-s{seed}",
        group=f"{sweep_id}/{algo}",
        job_type="train",
        tags=[sweep_id, algo, "env:crafter", exp_type],
        config=config,
        notes=notes,
        dir=out_dir,
        sync_tensorboard=True,
        save_code=True,
        reinit=True,
    )
    return run
