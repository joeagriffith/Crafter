# Profiling

Finding RL throughput bottlenecks. The workload is **CPU-bound env stepping**
(Crafter render + `step` across 16 envs) racing the **GPU model update**.

The toolkit lives at [`profile/`](../profile/) — see
[`profile/README.md`](../profile/README.md) for the full backend table and
install notes. Quick start:

```bash
uv run python profile/run.py --doctor                      # tool availability
uv run python profile/run.py \
  --trial experiments/20260624-sb3-baselines/trials/ppo \
  --profiler throughput --steps 3000                       # CPU/GPU verdict
uv run python profile/run.py \
  --trial experiments/20260624-sb3-baselines/trials/ppo \
  --profiler pyspy --steps 3000                            # Python hotspots
```

Every profiling run forces **W&B off** and uses `train.py --no-record`, so it
never writes to `experiments/ledger.jsonl`. Artifacts go to `profile/out/`
(gitignored).

Decision flow:
1. `throughput` → CPU-bound or GPU-bound?
2. If CPU-bound → `pyspy` (Python) / `perf` (native) for the hot frames.
3. If GPU-bound → `torch` (SB3 op table) / `nsys` (timeline) / `ncu` (kernels).
