"""Shared helpers for backends: child process running + resource sampling."""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import threading
import time

# SB3 prints `|    fps    | 471 |` rows in its rollout table.
_FPS_RE = re.compile(r"\bfps\b\s*\|\s*([\d.]+)")


def _read_proc_stat() -> tuple[int, int]:
    """Return (idle, total) jiffies from /proc/stat aggregate cpu line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return idle, sum(vals)


def _cpu_percent_factory():
    """Return a callable giving instantaneous CPU% across samples.

    Prefers psutil; falls back to /proc/stat deltas (no extra deps)."""
    try:
        import psutil
        psutil.cpu_percent(None)  # prime

        def _p():
            return psutil.cpu_percent(None)
        return _p
    except Exception:
        state = {"prev": _read_proc_stat()}

        def _p():
            idle, total = _read_proc_stat()
            pidle, ptotal = state["prev"]
            state["prev"] = (idle, total)
            dt = total - ptotal
            if dt <= 0:
                return 0.0
            return 100.0 * (1.0 - (idle - pidle) / dt)
        return _p


def _gpu_sample() -> tuple[float, float]:
    """Return (gpu_util%, mem_used_MiB). (-1,-1) if nvidia-smi missing."""
    if not shutil.which("nvidia-smi"):
        return -1.0, -1.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        # first GPU line
        util, mem = out.strip().splitlines()[0].split(",")
        return float(util), float(mem)
    except Exception:
        return -1.0, -1.0


class Sampler:
    """Background thread sampling GPU util/VRAM + system CPU% every `interval`."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._cpu = _cpu_percent_factory()
        self._t0 = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            gpu, mem = _gpu_sample()
            cpu = self._cpu()
            self.samples.append({"t": round(t, 2), "gpu_util": gpu,
                                 "vram_mib": mem, "cpu_pct": round(cpu, 1)})
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def write_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["t", "gpu_util", "vram_mib",
                                              "cpu_pct"])
            w.writeheader()
            w.writerows(self.samples)


def run_child(recipe, log_path: str, timeout: int):
    """Run recipe.argv, tee stdout/stderr to log_path, parse SB3 fps.

    Returns (returncode, elapsed_s, fps_list, timed_out)."""
    fps_vals: list[float] = []
    t0 = time.time()
    timed_out = False
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            recipe.argv, cwd=recipe.cwd, env=recipe.full_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1)
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                m = _FPS_RE.search(line)
                if m:
                    try:
                        fps_vals.append(float(m.group(1)))
                    except ValueError:
                        pass
                if time.time() - t0 > timeout:
                    timed_out = True
                    proc.kill()
                    break
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    return proc.returncode, time.time() - t0, fps_vals, timed_out


def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
