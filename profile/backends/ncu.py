"""ncu backend: Nsight Compute per-kernel analysis (VERY slow, targeted)."""
from __future__ import annotations

import os
import shutil
import subprocess


def run(recipe, out_dir, steps, timeout):
    if not shutil.which("ncu"):
        print("[ncu] ncu not found on PATH. Install:\n"
              "   sudo apt-get install -y nsight-compute\n"
              "   or via the CUDA Toolkit "
              "(https://developer.nvidia.com/nsight-compute).")
        return None
    print("=" * 70)
    print("[ncu] WARNING: Nsight Compute uses kernel REPLAY and is EXTREMELY")
    print("[ncu] slow (minutes/kernel). It is for TARGETED kernel analysis")
    print("[ncu] (e.g. our future custom CUDA kernels), NOT whole-run profiling.")
    print("[ncu] --launch-count 30 caps how many kernels are captured.")
    print("=" * 70)
    report = os.path.join(out_dir, "ncu_report")
    cmd = ["ncu", "--set", "basic", "--launch-count", "30",
           "--target-processes", "all", "-f", "-o", report, *recipe.argv]
    print(f"[ncu] {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=recipe.cwd, env=recipe.full_env(),
                        timeout=timeout + 600).returncode
    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w") as f:
        f.write(f"# ncu profile — {recipe.slug}\n\n"
                f"- return code: {rc}\n"
                f"- report: `ncu_report.ncu-rep` (open in Nsight Compute GUI, "
                "or `ncu --import ncu_report.ncu-rep --page details`)\n")
    print(f"[ncu] wrote {summary}")
    return summary
