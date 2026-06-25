"""Shared utilities: git/version introspection, seeding, and metrics/ledger I/O.

metrics.json schema (one object per trial/seed, written by ``write_metrics``):

    {
      "trial_id":          "<algo>/s<seed>",          # local id within experiment
      "experiment":        "20260624-sb3-baselines",  # experiment dir name
      "algo":              "ppo",
      "seed":              0,
      "sweep_id":          "bl0624",
      "steps":             1000000,                   # total env steps trained
      "n_envs":            16,                         # parallel envs used
      "eval_reward":       6.10,                       # mean eval episode reward
      "eval_achievements": 11,                         # distinct achievements unlocked
      "wandb_url":         "https://wandb.ai/.../runs/<id>",  # or null
      "git_sha":           "uncommitted",
      "status":            "ok",                       # ok / failed / skipped
      "created":           "2026-06-25"                # ISO date
    }

The same flat object is appended (compact, one line) to ``experiments/ledger.jsonl``
as the global, append-only index of every trial across every experiment.
"""
import importlib.metadata
import json
import os
import random
import subprocess

import numpy as np

# Project root = two levels up from this file (crafter_rl/utils/__init__.py).
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_git_sha():
    """Short git SHA of the project, or ``"uncommitted"`` if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PKG_ROOT,
            stderr=subprocess.DEVNULL)
        return sha.decode().strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def pkg_version(module, dist_name):
    """Library version from the module, falling back to package metadata."""
    version = getattr(module, "__version__", None)
    if version:
        return version
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return "unknown"


def set_seed(seed):
    """Seed Python ``random``, NumPy, and (if available) torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_metrics(out_dir, metrics: dict):
    """Write ``out_dir/metrics.json`` (pretty JSON). Creates ``out_dir``.

    See the module docstring for the expected schema. Returns the path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def append_ledger(record: dict, ledger_path="experiments/ledger.jsonl"):
    """Append one compact JSON line to the append-only ledger.

    Creates the parent dir if needed. Returns the path written.
    """
    parent = os.path.dirname(ledger_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return ledger_path
