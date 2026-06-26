"""perf backend: Linux perf stat (counters) + perf record (sampling)."""
from __future__ import annotations

import os
import shutil
import subprocess


def _paranoid() -> int | None:
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return int(f.read().strip())
    except Exception:
        return None


def run(recipe, out_dir, steps, timeout):
    if not shutil.which("perf"):
        print("[perf] perf not found. Install:\n"
              "   sudo apt-get install -y linux-tools-$(uname -r) "
              "linux-tools-generic")
        return None

    par = _paranoid()
    if par is not None and par > 2:
        print(f"[perf] WARNING: perf_event_paranoid={par} (>2). Sampling/"
              "counters may be denied. To allow:\n"
              "   echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid")

    stat_log = os.path.join(out_dir, "perf_stat.txt")
    perf_data = os.path.join(out_dir, "perf.data")

    # 1) perf stat -d : detailed CPU counters around the run.
    print("[perf] perf stat -d ...")
    with open(stat_log, "w") as f:
        rc_stat = subprocess.run(
            ["perf", "stat", "-d", *recipe.argv], cwd=recipe.cwd,
            env=recipe.full_env(), stderr=f, stdout=f,
            timeout=timeout + 60).returncode

    # 2) perf record -g : call-graph sampling for a flamegraph.
    print("[perf] perf record -g ...")
    rc_rec = subprocess.run(
        ["perf", "record", "-g", "-o", perf_data, "--", *recipe.argv],
        cwd=recipe.cwd, env=recipe.full_env(),
        timeout=timeout + 60).returncode

    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w") as f:
        f.write(
            f"# perf profile — {recipe.slug}\n\n"
            f"- perf stat rc: {rc_stat} -> `perf_stat.txt`\n"
            f"- perf record rc: {rc_rec} -> `perf.data`\n\n"
            "## Reading perf.data\n"
            "- `perf report -i perf.data`  (interactive TUI)\n"
            "- Flamegraph: `perf script -i perf.data | "
            "stackcollapse-perf.pl | flamegraph.pl > perf.svg`\n"
            "  (FlameGraph: https://github.com/brendangregg/FlameGraph)\n")
    print(f"[perf] wrote {summary}")
    return summary
