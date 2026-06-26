"""throughput backend: run to completion while sampling GPU/CPU; verdict."""
from __future__ import annotations

import os
import statistics as st

from ._common import Sampler, percentile, run_child


def run(recipe, out_dir, steps, timeout):
    log_path = os.path.join(out_dir, "child.log")
    csv_path = os.path.join(out_dir, "samples.csv")
    summary_path = os.path.join(out_dir, "summary.md")

    print(f"[throughput] running {recipe.slug} ({steps} steps, "
          f"timeout {timeout}s) ...")
    with Sampler(interval=0.5) as sampler:
        rc, elapsed, fps_vals, timed_out = run_child(recipe, log_path, timeout)
    sampler.write_csv(csv_path)

    s = sampler.samples
    gpu = [x["gpu_util"] for x in s if x["gpu_util"] >= 0]
    cpu = [x["cpu_pct"] for x in s if x["cpu_pct"] >= 0]
    vram = [x["vram_mib"] for x in s if x["vram_mib"] >= 0]

    mean_gpu = st.mean(gpu) if gpu else float("nan")
    med_gpu = st.median(gpu) if gpu else float("nan")
    p95_gpu = percentile(gpu, 95) if gpu else float("nan")
    mean_cpu = st.mean(cpu) if cpu else float("nan")
    peak_vram = max(vram) if vram else float("nan")

    # steps/sec: prefer SB3's own fps (last value is most representative since
    # it is cumulative timesteps/elapsed); else fall back to steps/elapsed.
    if fps_vals:
        sps = fps_vals[-1]
        sps_src = "SB3 fps (last)"
    else:
        sps = steps / elapsed if elapsed > 0 else float("nan")
        sps_src = "steps / elapsed"

    # VERDICT heuristic.
    if mean_gpu == mean_gpu and mean_gpu > 70:
        verdict = "GPU-bound"
    elif mean_gpu == mean_gpu and mean_gpu < 40:
        verdict = "CPU-bound / input-bound"
    else:
        verdict = "mixed"

    def fmt(v):
        return f"{v:.1f}" if v == v else "n/a"

    lines = [
        f"# Throughput profile — {recipe.slug}",
        "",
        f"- **Verdict:** **{verdict}**",
        f"- **steps/sec:** {fmt(sps)}  _(source: {sps_src})_",
        f"- **steps requested:** {steps}",
        f"- **wall time:** {elapsed:.1f}s"
        + ("  (TIMED OUT — bound hit)" if timed_out else ""),
        f"- **child return code:** {rc}",
        "",
        "## GPU",
        f"- mean util: {fmt(mean_gpu)}%",
        f"- median util: {fmt(med_gpu)}%",
        f"- p95 util: {fmt(p95_gpu)}%",
        f"- peak VRAM: {fmt(peak_vram)} MiB",
        "",
        "## CPU",
        f"- mean util (system, all cores): {fmt(mean_cpu)}%",
        "",
        "## Notes",
        f"- samples: {len(s)} @ 0.5s  ->  `samples.csv`",
        f"- child log: `child.log`",
        "- Verdict heuristic: GPU-bound if mean GPU>70%; "
        "CPU/input-bound if mean GPU<40%; else mixed.",
        "",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[throughput] wrote {summary_path}")
    return summary_path
