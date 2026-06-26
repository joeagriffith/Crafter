"""Profiler tool availability check + install hints."""
from __future__ import annotations

import os
import shutil

from targets import MAIN_VENV_PY, REPO_ROOT

# name -> (purpose, install command if missing)
TOOLS = {
    "nvidia-smi": (
        "GPU util/VRAM sampling (throughput backend)",
        "ships with the NVIDIA driver (nvidia-utils)"),
    "py-spy": (
        "sampling python profiler (pyspy backend)",
        f"{MAIN_VENV_PY} -m uv pip install py-spy  "
        "(or: uv pip install py-spy)"),
    "nsys": (
        "Nsight Systems timeline (CUDA/NVTX/OS runtime)",
        "sudo apt-get install -y nsight-systems   "
        "(also ships with the CUDA Toolkit)"),
    "ncu": (
        "Nsight Compute per-kernel analysis (very slow)",
        "sudo apt-get install -y nsight-compute    "
        "(also ships with the CUDA Toolkit)"),
    "perf": (
        "CPU counters + sampling profiler (perf backend)",
        "sudo apt-get install -y linux-tools-$(uname -r) linux-tools-generic"),
}


def _which_pyspy() -> str | None:
    w = shutil.which("py-spy")
    if w:
        return w
    cand = os.path.join(REPO_ROOT, ".venv", "bin", "py-spy")
    return cand if os.path.exists(cand) else None


def status() -> dict[str, tuple[bool, str | None, str, str]]:
    """name -> (available, path, purpose, install_cmd)."""
    out = {}
    for name, (purpose, install) in TOOLS.items():
        path = _which_pyspy() if name == "py-spy" else shutil.which(name)
        out[name] = (path is not None, path, purpose, install)
    return out


def run() -> None:
    st = status()
    print("\n  Profiler toolkit — availability\n")
    print(f"  {'TOOL':<12} {'STATUS':<10} {'PURPOSE'}")
    print(f"  {'-'*12} {'-'*10} {'-'*40}")
    missing = []
    for name, (ok, path, purpose, install) in st.items():
        mark = "OK" if ok else "MISSING"
        print(f"  {name:<12} {mark:<10} {purpose}")
        if not ok:
            missing.append((name, install))
    if missing:
        print("\n  Install commands for missing tools:")
        for name, install in missing:
            print(f"    {name:<12}: {install}")
    else:
        print("\n  All profilers available.")
    print("\n  Notes:")
    print("    - perf record needs: echo -1 | sudo tee "
          "/proc/sys/kernel/perf_event_paranoid")
    print("    - nsys/ncu also come bundled with the CUDA Toolkit.\n")


if __name__ == "__main__":
    run()
