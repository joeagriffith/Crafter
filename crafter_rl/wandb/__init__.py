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

import re


def run_id(sweep_id, algo_label, seed):
    """Return a stable, wandb-safe run id for a (sweep, algo, seed) triple.

    Both the live streamer and ``report_results.py`` derive the W&B run id from
    this helper so they target the SAME run (no duplicates). Any character
    outside ``[A-Za-z0-9._-]`` is replaced by ``-``.

    The id uses a ``seed{n}`` token while the human-readable run *name* uses
    ``s{n}`` (see :func:`init_run`); keeping the immutable id distinct from the
    display name is deliberate -- the id is the stable join key across the live
    streamer and the post-hoc finaliser, the name is just what humans read.
    """
    import os

    raw = f"{sweep_id}-{algo_label}-seed{seed}"
    # Optional per-session suffix (env ``WANDB_RUN_SUFFIX``) appended to the id
    # ONLY -- never the display name. Lets a fresh session reuse the same human
    # name while getting a brand-new, unique id (avoids the wandb tombstone-hang
    # that reusing a deleted id triggers). Default empty -> fully backward-compat.
    suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
    if suffix:
        raw = f"{raw}-{suffix}"
    return re.sub(r"[^A-Za-z0-9._-]", "-", raw)


def init_run(*, algo, seed, sweep_id, exp_type, project, config,
             out_dir=None, name=None, notes=None, id=None, resume=None,
             sync_tensorboard=True):
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
    id : explicit, deterministic W&B run id (see :func:`run_id`). When set, the
        same id is reused across processes so the live streamer and the
        post-hoc finaliser join ONE run instead of creating duplicates.
    resume : passed through to ``wandb.init`` (e.g. ``"allow"`` to attach to an
        existing run id, or create it if absent). Defaults to ``None`` so
        existing callers are unaffected.
    sync_tensorboard : default ``True`` (mirror co-located TB event files).
        Set ``False`` when the caller logs metrics explicitly with its own
        ``step=`` -- TB syncing forces wandb's own step axis and ignores
        ``step=`` (the live streamer uses ``False`` to keep the env step).
    """
    import os
    import wandb  # absolute import -> the real pip package

    # wandb writes its local files to ``{dir}/wandb/``; if ``dir`` doesn't exist
    # yet it silently falls back to the cwd (repo root). Create it so runs stay
    # co-located with the trial that produced them.
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Bound init so a server-side hang (e.g. a tombstoned/in-trash run id, or a
    # network stall) raises promptly instead of blocking the caller forever.
    # This protects report_results.py's critical path: a wandb hiccup turns into
    # a caught exception, never a hang that prevents metrics.json + ledger.
    settings = None
    try:
        settings = wandb.Settings(init_timeout=45)
    except Exception:
        settings = None  # older/newer wandb without this knob -> default behaviour

    run = wandb.init(
        project=project,
        id=id,
        resume=resume,
        name=name or f"{sweep_id}-{algo}-s{seed}",
        group=f"{sweep_id}/{algo}",
        job_type="train",
        tags=[sweep_id, algo, "env:crafter", exp_type],
        config=config,
        notes=notes,
        dir=out_dir,
        sync_tensorboard=sync_tensorboard,
        save_code=True,
        reinit=True,
        settings=settings,
    )
    return run
