#!/usr/bin/env python
"""Profiling toolkit for Crafter RL training runs.

Point it at a trial, pick a profiler, and it launches a bounded training run
(W&B + ledger writes DISABLED) under that profiler, writing artifacts + a
human-readable summary into profile/out/.

  uv run python profile/run.py --doctor
  uv run python profile/run.py --trial experiments/20260624-sb3-baselines/trials/ppo \
      --profiler throughput --steps 3000
  uv run python profile/run.py --trial experiments/20260624-sb3-baselines/trials/ppo \
      --profiler pyspy --steps 3000

See profile/README.md for the backend table.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Make `import targets` / `import backends` work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doctor  # noqa: E402
import targets  # noqa: E402
from backends import (ncu, nsys, perf, pyspy,  # noqa: E402
                      throughput, torch_prof)

BACKENDS = {
    "throughput": throughput.run,
    "pyspy": pyspy.run,
    "torch": torch_prof.run,
    "nsys": nsys.run,
    "ncu": ncu.run,
    "perf": perf.run,
}

PROFILE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description="Profile a bounded Crafter training run.")
    ap.add_argument("--trial", help="Trial dir, e.g. "
                    "experiments/20260624-sb3-baselines/trials/ppo")
    ap.add_argument("--profiler", choices=list(BACKENDS), default="throughput")
    ap.add_argument("--steps", type=int, default=3000,
                    help="Bound the training run (default 3000).")
    ap.add_argument("--out", default=None, help="Output dir.")
    ap.add_argument("--cmd", default=None,
                    help="Raw command to profile instead of a resolved trial.")
    ap.add_argument("--extra", default="",
                    help="Extra k=v args passed through to the trial command.")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Hard wall-clock cap on the child (seconds).")
    ap.add_argument("--doctor", action="store_true",
                    help="Print profiler availability + install help, exit.")
    args = ap.parse_args()

    if args.doctor:
        doctor.run()
        return 0

    if not args.trial and not args.cmd:
        ap.error("one of --trial or --cmd is required (or use --doctor).")

    # Output dir: profile/out/<slug>-<profiler>-<timestamp>
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = (os.path.basename(os.path.normpath(args.trial)) if args.trial
            else "rawcmd")
    out_dir = args.out or os.path.join(PROFILE_DIR, "out",
                                       f"{slug}-{args.profiler}-{ts}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.cmd:
        recipe = targets.resolve_cmd(args.cmd, out_dir)
    else:
        recipe = targets.resolve(args.trial, args.steps, out_dir, args.extra)

    print(f"[run] profiler={args.profiler}  trial={args.trial or args.cmd}")
    print(f"[run] recipe.kind={recipe.kind}  cwd={recipe.cwd}")
    print(f"[run] argv: {' '.join(recipe.argv)}")
    print(f"[run] out: {out_dir}\n")

    summary = BACKENDS[args.profiler](recipe, out_dir, args.steps, args.timeout)
    if summary:
        print(f"\n[run] DONE -> {summary}")
    else:
        print(f"\n[run] backend produced no summary (see messages above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
