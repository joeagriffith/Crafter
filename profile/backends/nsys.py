"""nsys backend: NVIDIA Nsight Systems timeline (CUDA/NVTX/OS runtime)."""
from __future__ import annotations

import os
import shutil
import subprocess


def run(recipe, out_dir, steps, timeout):
    if not shutil.which("nsys"):
        print("[nsys] nsys not found on PATH. Install one of:\n"
              "   sudo apt-get install -y nsight-systems\n"
              "   or the NVIDIA Nsight Systems installer "
              "(https://developer.nvidia.com/nsight-systems)\n"
              "   (it also ships with the CUDA Toolkit).")
        return None
    report = os.path.join(out_dir, "report")
    cmd = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=cpu",
           "--force-overwrite=true", "-o", report, *recipe.argv]
    print(f"[nsys] {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=recipe.cwd, env=recipe.full_env(),
                        timeout=timeout + 120).returncode
    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w") as f:
        f.write(f"# nsys profile — {recipe.slug}\n\n"
                f"- return code: {rc}\n"
                f"- report: `report.nsys-rep` "
                "(open in Nsight Systems GUI, or `nsys stats report.nsys-rep`)\n")
    print(f"[nsys] wrote {summary}")
    return summary
