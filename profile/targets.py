"""Resolve a trial directory into a concrete launch recipe.

A *recipe* is everything a backend needs to spawn the training process:
the python interpreter, the cwd, the full argv, and the environment. Every
resolved recipe FORCES W&B off and skips our ledger/metrics writes so a
profiling run can never pollute ``experiments/ledger.jsonl``.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
DREAMER_DIR = os.path.join(REPO_ROOT, "third_party", "r2dreamer")
DREAMER_VENV_PY = os.path.join(DREAMER_DIR, ".venv", "bin", "python")
SB3_TRAIN = os.path.join(REPO_ROOT, "experiments", "20260624-sb3-baselines",
                         "train.py")

SB3_ALGOS = ("ppo", "a2c", "dqn")


@dataclass
class Recipe:
    """A concrete, spawnable training command."""
    slug: str                       # short id for output dir naming
    kind: str                       # "sb3" | "dreamerv3" | "raw"
    argv: list[str]                 # full argv (argv[0] is the interpreter)
    cwd: str
    env: dict = field(default_factory=dict)
    venv_python: str | None = None  # interpreter (== argv[0] for resolved trials)

    def full_env(self) -> dict:
        e = dict(os.environ)
        e.update(self.env)
        return e


def _base_env() -> dict:
    # Belt-and-braces: disable W&B at the env level too.
    return {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}


def infer_algo(trial: str) -> str | None:
    """Infer the algo/kind from a trial path (basename, then ancestors)."""
    p = os.path.normpath(trial)
    if "dreamerv3" in p or "20260625-dreamerv3" in p:
        return "dreamerv3"
    base = os.path.basename(p.rstrip("/"))
    if base in SB3_ALGOS:
        return base
    for part in p.split(os.sep):
        if part in SB3_ALGOS:
            return part
    return None


def resolve(trial: str, steps: int, out_dir: str,
            extra: str = "") -> Recipe:
    """Map a trial path to a Recipe. ``out_dir`` is the profiler output dir;
    the trial's own (throwaway) outputs are nested under it."""
    algo = infer_algo(trial)
    if algo is None:
        raise ValueError(
            f"Could not infer algo/kind from trial path: {trial!r}. "
            f"Expected one of {SB3_ALGOS} or a dreamerv3 trial.")
    extra_args = shlex.split(extra) if extra else []
    trial_out = os.path.join(out_dir, "trial_tmp")
    os.makedirs(trial_out, exist_ok=True)

    if algo in SB3_ALGOS:
        argv = [
            MAIN_VENV_PY, SB3_TRAIN,
            "--algo", algo,
            "--seed", "0",
            "--steps", str(steps),
            "--envs", "16",
            "--no-wandb",
            "--no-record",            # skip eval/metrics/ledger (see train.py)
            "--out-dir", trial_out,
        ] + extra_args
        return Recipe(slug=f"sb3-{algo}", kind="sb3", argv=argv,
                      cwd=REPO_ROOT, env=_base_env(), venv_python=MAIN_VENV_PY)

    # DreamerV3 (vendored r2dreamer): small model, no compile, cpu replay so a
    # bounded profiling run is quick. Hydra does not chdir, so absolute logdir
    # is honoured while relative imports keep working from DREAMER_DIR.
    argv = [
        DREAMER_VENV_PY, "train.py",
        "env=crafter",
        "model=size12M",
        "model.rep_loss=dreamer",
        "model.compile=False",
        "device=cuda:0",
        "buffer.storage_device=cpu",
        f"env.steps={steps}",
        f"logdir={trial_out}",
    ] + extra_args
    return Recipe(slug="dreamerv3", kind="dreamerv3", argv=argv,
                  cwd=DREAMER_DIR, env=_base_env(),
                  venv_python=DREAMER_VENV_PY)


def resolve_cmd(cmd: str, out_dir: str) -> Recipe:
    """Escape hatch: profile an arbitrary raw command verbatim."""
    argv = shlex.split(cmd)
    return Recipe(slug="rawcmd", kind="raw", argv=argv, cwd=REPO_ROOT,
                  env=_base_env(), venv_python=argv[0] if argv else None)
