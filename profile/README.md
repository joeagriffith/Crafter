# Profiling toolkit

Point at a trial, profile a **bounded** training run (W&B + ledger writes
**disabled**), and get artifacts + a human-readable summary so we can find and
iterate on RL throughput bottlenecks.

The workload shape we care about: **CPU-bound env stepping** (Crafter render +
`step`, 16 parallel envs) vs **GPU model update** (the SB3 CNN / DreamerV3
world-model). Each backend below answers a different slice of that.

## Usage

```bash
# What's installed + how to install the rest
uv run python profile/run.py --doctor

# Throughput + CPU/GPU verdict (no extra tools needed)
uv run python profile/run.py \
  --trial experiments/20260624-sb3-baselines/trials/ppo \
  --profiler throughput --steps 3000

# Python hotspots (the key bottleneck finding)
uv run python profile/run.py \
  --trial experiments/20260624-sb3-baselines/trials/ppo \
  --profiler pyspy --steps 3000

# DreamerV3 trial (own venv, resolved automatically)
uv run python profile/run.py \
  --trial experiments/20260625-dreamerv3/trials/dreamerv3 \
  --profiler throughput --steps 3000

# Escape hatch: profile any raw command
uv run python profile/run.py --cmd "/path/python foo.py" --profiler pyspy
```

### Flags
| flag | meaning |
|------|---------|
| `--trial <path>` | trial dir; algo/kind inferred from the path |
| `--profiler` | `throughput` (default), `pyspy`, `torch`, `nsys`, `ncu`, `perf` |
| `--steps N` | bound the run (default 3000) |
| `--out <dir>` | default `profile/out/<slug>-<profiler>-<timestamp>` |
| `--cmd "<raw>"` | profile an arbitrary command instead of a trial |
| `--extra "<k=v ...>"` | pass-through extra args to the trial command |
| `--timeout N` | hard wall-clock cap on the child (s, default 900) |
| `--doctor` | print tool availability + install commands, exit |

## Backends — what each answers
| backend | tool | answers | needs |
|---------|------|---------|-------|
| `throughput` | nvidia-smi + psutil/proc | steps/sec, GPU/CPU util, peak VRAM, **CPU-vs-GPU verdict** | nvidia-smi |
| `pyspy` | py-spy | **which Python functions are hot** (env step / render?) | py-spy (auto-installed) |
| `torch` | torch.profiler | per-op CPU/CUDA time in the model update (SB3 only) | torch |
| `nsys` | Nsight Systems | CUDA/NVTX/OS timeline — overlap, stalls, H2D copies | nsys |
| `ncu` | Nsight Compute | per-kernel detail (targeted; **very slow**) | ncu |
| `perf` | Linux perf | CPU counters + native call-graph sampling | perf |

`throughput` → "is this CPU- or GPU-bound?"; `pyspy`/`perf` → "where in CPU?";
`torch`/`nsys`/`ncu` → "where on the GPU?".

## Target resolution
- **SB3 trials** (`.../trials/{ppo,a2c,dqn}`): main venv, runs `train.py
  --algo <algo> --seed 0 --steps N --envs 16 --no-wandb --no-record
  --out-dir <tmp>`. `--no-record` (added to that `train.py`) skips W&B, the
  final eval, `metrics.json`, and the ledger append.
- **DreamerV3 trial**: its own venv (`third_party/r2dreamer/.venv`), cwd
  `third_party/r2dreamer`, runs Hydra `train.py env=crafter model=size12M
  model.rep_loss=dreamer model.compile=False device=cuda:0
  buffer.storage_device=cpu env.steps=N logdir=<tmp>` (small + no compile so
  profiling is quick; writes nothing to our ledger).
- All recipes also set `WANDB_MODE=disabled`.

## Install notes (missing tools)
- **py-spy**: `uv pip install py-spy` (pip binary, no sudo; auto-installed by
  the `pyspy` backend).
- **nsys / ncu**: `sudo apt-get install -y nsight-systems nsight-compute`, or
  the NVIDIA installers — they also ship with the **CUDA Toolkit**.
- **perf**: `sudo apt-get install -y linux-tools-$(uname -r) linux-tools-generic`
  then `echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid`.

Outputs land in `profile/out/` (gitignored).
