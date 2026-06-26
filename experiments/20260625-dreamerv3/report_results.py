#!/usr/bin/env python3
"""Adapter: fold r2dreamer's DreamerV3 Crafter output into our result schema.

r2dreamer logs eval scalars two ways inside the trial out dir (== Hydra logdir):
  * ``metrics.jsonl`` - one JSON object per ``Logger.write()`` call; eval lines
    carry ``episode/eval_score`` (mean eval episode reward), ``episode/eval_length``
    and one ``episode/eval_<achievement>`` per Crafter achievement (mean count
    over the eval episodes).
  * TensorBoard event files in the same dir (tags ``episode/eval_*``).

This script is the bridge to OUR conventions. It runs in the MAIN project venv
(the one that has ``crafter_rl`` + ``wandb``), NOT r2dreamer's isolated venv:

  1. parse the last eval record from ``metrics.jsonl``;
  2. compute ``eval_reward`` (= eval_score) and ``eval_achievements``
     (= # achievements with mean count > 0);
  3. write ``metrics.json`` (13-field schema) + append to ``experiments/ledger.jsonl``
     via ``crafter_rl.utils``;
  4. (optional, --wandb) create a run with ``crafter_rl.wandb.init_run`` and replay
     the eval curves from metrics.jsonl into it.

Usage (invoked by run_dreamerv3.sh after training):
  python report_results.py --out-dir <trial_out> --seed 0 --sweep-id dv3MMDD \
      --pinned-sha <sha> [--wandb] [--no-ledger]
"""
import argparse
import datetime as dt
import json
import os
import sys

# Make crafter_rl importable regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from crafter_rl.utils import append_ledger, get_git_sha, write_metrics  # noqa: E402

EXPERIMENT = "20260625-dreamerv3"
ALGO = "dreamerv3"
DEFAULT_MODEL_SIZE = "size200M"
LEDGER_PATH = os.path.join(_REPO_ROOT, "experiments", "ledger.jsonl")

# Non-achievement eval scalars that must be excluded from the achievement count.
_NON_ACHIEVEMENT = {"episode/eval_score", "episode/eval_length", "episode/eval_success"}


def read_eval_records(out_dir):
    """Return all metrics.jsonl records that contain an eval score, in order."""
    path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - did r2dreamer training run and write to this logdir?")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "episode/eval_score" in rec:
                records.append(rec)
    return records


def summarise(rec):
    """eval_reward + distinct-achievements from one eval record."""
    eval_reward = float(rec["episode/eval_score"])
    achievements = 0
    for k, v in rec.items():
        if k.startswith("episode/eval_") and k not in _NON_ACHIEVEMENT:
            if float(v) > 0.0:
                achievements += 1
    return eval_reward, achievements


def maybe_wandb(out_dir, eval_records, *, seed, sweep_id, steps, eval_reward,
                eval_achievements, git_sha, pinned_sha, algo_label, model_size):
    """Create a W&B run and replay eval curves. Returns the run URL or None.

    ``algo_label`` (e.g. ``dreamerv3-size50M``) is used as the W&B ``algo`` so the
    run name becomes ``{sweep_id}-{algo_label}-s{seed}`` and the group becomes
    ``{sweep_id}/{algo_label}`` -- this keeps each size's runs distinguishable.
    """
    try:
        from crafter_rl.wandb import init_run, run_id
        import wandb
    except Exception as e:
        print(f"[report] W&B unavailable ({e}); skipping.")
        return None

    config = {
        "algo": ALGO, "seed": seed, "sweep_id": sweep_id, "exp_type": "baseline",
        "total_steps": steps, "n_envs": 1, "git_sha": git_sha,
        "impl": "NM512/r2dreamer", "impl_sha": pinned_sha,
        "model_size": model_size, "rep_loss": "dreamer",
        "crafter_version": _ver("crafter"),
        "torch_version": _ver("torch"),
    }
    # Deterministic id shared with the live streamer: passing it with
    # resume="allow" JOINS the streamer's already-created run (or creates it if
    # the streamer never ran) instead of spawning a duplicate run.
    rid = run_id(sweep_id, algo_label, seed)
    try:
        run = init_run(algo=algo_label, seed=seed, sweep_id=sweep_id, exp_type="baseline",
                       project="crafter", config=config, out_dir=out_dir,
                       id=rid, resume="allow",
                       name=f"{sweep_id}-{algo_label}-s{seed}",
                       notes=f"DreamerV3 ({model_size}, r2dreamer rep_loss=dreamer) Crafter baseline")
    except Exception as e:
        print(f"[report] init_run failed ({e}); skipping W&B.")
        return None

    # Replay every eval record so W&B shows the full eval curve, not just a
    # point. Guard the whole replay+finalise: a W&B/network hiccup here must
    # NEVER block the metrics.json + ledger writes that follow in main().
    url = getattr(run, "url", None)
    try:
        for rec in eval_records:
            step = int(rec.get("step", 0))
            payload = {f"eval/{k.split('/', 1)[1]}": float(v)
                       for k, v in rec.items() if k.startswith("episode/eval_")}
            if payload:
                wandb.log(payload, step=step)
        run.summary["eval/mean_reward"] = eval_reward
        run.summary["eval/achievements"] = eval_achievements
    except Exception as e:
        print(f"[report] W&B replay failed ({e}); metrics/ledger unaffected.")
    finally:
        try:
            run.finish()
        except Exception:
            pass
    return url


def _ver(dist_name):
    """Best-effort installed version of a dist, or 'unknown'."""
    import importlib.metadata
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="trial out dir == r2dreamer logdir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-id", default=None)
    ap.add_argument("--pinned-sha", default="unknown")
    ap.add_argument("--model-size", default=None,
                    help="r2dreamer model size (e.g. size50M). Differentiates the "
                         "run in W&B/ledger via algo label dreamerv3-<size>. "
                         f"Default {DEFAULT_MODEL_SIZE}.")
    ap.add_argument("--steps", type=int, default=None,
                    help="override trained env steps (default: max step in metrics.jsonl)")
    ap.add_argument("--wandb", action="store_true", help="also push to W&B")
    ap.add_argument("--no-ledger", action="store_true", help="skip ledger append (smoke/debug)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    sweep_id = args.sweep_id or f"dv3{dt.date.today():%m%d}"
    # Backward-compatible: fall back to $MODEL env then the historical default.
    model_size = args.model_size or os.environ.get("MODEL") or DEFAULT_MODEL_SIZE
    algo_label = f"{ALGO}-{model_size}"

    eval_records = read_eval_records(out_dir)
    if not eval_records:
        raise SystemExit(f"[report] no eval records in {out_dir}/metrics.jsonl")
    last = eval_records[-1]
    eval_reward, eval_achievements = summarise(last)
    steps = args.steps if args.steps is not None else int(last.get("step", 0))
    git_sha = get_git_sha()

    wandb_url = None
    if args.wandb:
        # Belt-and-braces: metrics.json + ledger are the critical deliverable,
        # so no W&B failure mode may prevent them from being written below.
        try:
            wandb_url = maybe_wandb(
                out_dir, eval_records, seed=args.seed, sweep_id=sweep_id, steps=steps,
                eval_reward=eval_reward, eval_achievements=eval_achievements,
                git_sha=git_sha, pinned_sha=args.pinned_sha,
                algo_label=algo_label, model_size=model_size)
        except Exception as e:
            print(f"[report] W&B step failed ({e}); continuing to metrics+ledger.")

    metrics = {
        "trial_id": f"{algo_label}/s{args.seed}",
        "experiment": EXPERIMENT,
        "algo": algo_label,
        "model_size": model_size,
        "seed": args.seed,
        "sweep_id": sweep_id,
        "steps": steps,
        "n_envs": 1,
        "eval_reward": round(eval_reward, 4),
        "eval_achievements": eval_achievements,
        "wandb_url": wandb_url,
        "git_sha": git_sha,
        "status": "ok",
        "created": f"{dt.date.today():%Y-%m-%d}",
    }
    path = write_metrics(out_dir, metrics)
    print(f"[report] wrote {path}")
    print(f"[report] eval_reward={eval_reward:.3f} eval_achievements={eval_achievements} steps={steps}")
    if not args.no_ledger:
        append_ledger(metrics, ledger_path=LEDGER_PATH)
        print(f"[report] appended {LEDGER_PATH}")
    else:
        print("[report] --no-ledger: ledger NOT touched")


if __name__ == "__main__":
    main()
