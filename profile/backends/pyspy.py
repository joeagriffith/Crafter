"""pyspy backend: sampling profiler -> speedscope JSON + hottest frames."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from targets import REPO_ROOT


def _pyspy_bin() -> str | None:
    w = shutil.which("py-spy")
    if w:
        return w
    cand = os.path.join(REPO_ROOT, ".venv", "bin", "py-spy")
    return cand if os.path.exists(cand) else None


def _ensure_pyspy() -> str | None:
    bin_ = _pyspy_bin()
    if bin_:
        return bin_
    print("[pyspy] py-spy not found; installing into the main venv ...")
    # pip binary, no sudo needed.
    for cmd in (["uv", "pip", "install", "py-spy"],
                [os.path.join(REPO_ROOT, ".venv", "bin", "python"),
                 "-m", "pip", "install", "py-spy"]):
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            if _pyspy_bin():
                return _pyspy_bin()
        except Exception as e:
            print(f"[pyspy]   install via {cmd[0]} failed: {e}")
    return _pyspy_bin()


def _top_frames(speedscope_path: str, n: int = 5):
    """Parse a py-spy speedscope (sampled) profile -> top-n by self weight."""
    with open(speedscope_path) as f:
        data = json.load(f)
    frames = data.get("shared", {}).get("frames", [])

    def label(i):
        fr = frames[i]
        nm = fr.get("name", "?")
        file = os.path.basename(fr.get("file", "") or "")
        line = fr.get("line", "")
        loc = f"{file}:{line}" if file else ""
        return f"{nm}  ({loc})" if loc else nm

    self_w: dict[int, float] = {}
    total = 0.0
    for prof in data.get("profiles", []):
        if prof.get("type") != "sampled":
            continue
        weights = prof.get("weights", [])
        for stack, w in zip(prof.get("samples", []), weights):
            total += w
            if stack:
                self_w[stack[-1]] = self_w.get(stack[-1], 0.0) + w
    ranked = sorted(self_w.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(label(i), w, (100.0 * w / total if total else 0.0))
            for i, w in ranked], total


def run(recipe, out_dir, steps, timeout):
    bin_ = _ensure_pyspy()
    if not bin_:
        print("[pyspy] could not obtain py-spy. Install: uv pip install py-spy")
        return None

    speedscope = os.path.join(out_dir, "pyspy.speedscope.json")
    svg = os.path.join(out_dir, "pyspy.svg")
    summary_path = os.path.join(out_dir, "summary.md")

    # speedscope (machine-parseable) + flamegraph SVG (eyeball-friendly).
    base = [bin_, "record", "--subprocesses", "--rate", "200"]
    env = recipe.full_env()
    print(f"[pyspy] recording {recipe.slug} (speedscope) ...")
    rc1 = subprocess.run(
        base + ["--format", "speedscope", "-o", speedscope, "--",
                *recipe.argv],
        cwd=recipe.cwd, env=env, timeout=timeout + 60).returncode
    print(f"[pyspy] recording {recipe.slug} (flamegraph svg) ...")
    try:
        subprocess.run(
            base + ["--format", "flamegraph", "-o", svg, "--", *recipe.argv],
            cwd=recipe.cwd, env=env, timeout=timeout + 60)
    except Exception as e:
        print(f"[pyspy] flamegraph pass skipped: {e}")

    lines = [f"# py-spy profile — {recipe.slug}", ""]
    if os.path.exists(speedscope):
        try:
            top, total = _top_frames(speedscope, 5)
            lines += ["## Top 5 functions by self-time", "",
                      "| # | function | self % |",
                      "|---|----------|--------|"]
            for i, (lbl, w, pct) in enumerate(top, 1):
                lines.append(f"| {i} | `{lbl}` | {pct:.1f}% |")
            lines.append("")
        except Exception as e:
            lines.append(f"_could not parse speedscope: {e}_\n")
    else:
        lines.append(f"_speedscope not produced (rc={rc1})_\n")

    lines += [
        "## Artifacts",
        f"- speedscope: `pyspy.speedscope.json` — open at "
        "https://www.speedscope.app (drag-drop) or in Chrome.",
        f"- flamegraph: `pyspy.svg` — open in any browser.",
        "",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[pyspy] wrote {summary_path}")
    return summary_path
