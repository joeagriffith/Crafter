"""torch backend: run the SB3 trial with the in-code torch.profiler hook."""
from __future__ import annotations

import os
import subprocess


def run(recipe, out_dir, steps, timeout):
    if recipe.kind != "sb3":
        print("[torch] torch.profiler is only wired into the SB3 train.py.\n"
              "        The DreamerV3 trial is vendored (r2dreamer) and not "
              "instrumented.\n"
              "        Use `--profiler pyspy` (python hotspots) or "
              "`--profiler nsys` (CUDA timeline) instead.")
        return None

    # Swap --no-record for --profile-torch <dir>; train.py wraps a short
    # instrumented learn and dumps the op table.
    prof_dir = os.path.join(out_dir, "torch_trace")
    argv = [a for a in recipe.argv if a != "--no-record"]
    # drop the --no-record had no value; now append the torch flag.
    argv = argv + ["--profile-torch", prof_dir]

    print(f"[torch] running {recipe.slug} under torch.profiler ...")
    rc = subprocess.run(argv, cwd=recipe.cwd, env=recipe.full_env(),
                        timeout=timeout + 60).returncode

    summary_path = os.path.join(out_dir, "summary.md")
    table = os.path.join(prof_dir, "key_averages.txt")
    lines = [f"# torch.profiler — {recipe.slug}", "",
             f"- return code: {rc}",
             f"- op table: `torch_trace/key_averages.txt`",
             f"- chrome/tensorboard trace: `torch_trace/` "
             "(open via `tensorboard --logdir torch_trace` or chrome://tracing"
             " on the .json).", ""]
    if os.path.exists(table):
        with open(table) as f:
            head = "".join(f.readlines()[:30])
        lines += ["## key_averages (head)", "", "```", head, "```", ""]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[torch] wrote {summary_path}")
    return summary_path
