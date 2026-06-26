# Repository Structure

This repo is organized for **mass experimentation** with an **experiment-then-promote**
workflow and **agent-friendly** conventions (everything queryable from the filesystem,
no UI required). Inspired by the Codeca `optim-loop` / experiment pattern.

## Top-level layout

```
crafter_rl/            # SHARED, PROMOTED library — thin, tested, importable everywhere
  env.py               #   CrafterGym (the promoted Gymnasium adapter for Crafter)
  wandb/               #   the W&B logging convention in one place (see wandb-conventions.md)
  utils/               #   git_sha, versions, set_seed, metrics.json + ledger helpers
  kernels/             #   Python bindings for custom CUDA ops (see cuda-kernels.md)
csrc/                  # CUDA / C++ sources (.cu/.cpp) compiled into crafter_rl._C
experiments/           # MASS EXPERIMENTATION — one dated directory per experiment
  ledger.jsonl         #   append-only global index, one flat JSON object per trial
  _template/           #   copy this to start a new experiment
  <YYYYMMDD-slug>/     #   a concrete experiment (a "trial suite") — also where new ideas are prototyped
third_party/           # vendored OSS impls that aren't pip-installable (+ thin adapters)
docs/                  # the conventions you are reading
runs are NOT stored at top level — every run writes into the trial that produced it.
```

## The two code tiers

| Tier | Where | Bar to entry | Lifetime |
|------|-------|--------------|----------|
| **Promoted** | `crafter_rl/` | conforms to interfaces, smoke-tested, reproduces a baseline | permanent, imported by everything |
| **Experiment** | `experiments/<dated>/` | follows the trial contract (new ideas are prototyped here) | kept as the historical record |

The promotion path (experiment-proven → `crafter_rl/`) is documented in
[promotion-checklist.md](promotion-checklist.md).

## Why this shape
- **Wrapping OSS**: `crafter_rl/` holds a thin common interface; OSS algorithms (SB3,
  and later vendored repos in `third_party/`) are adapted to it rather than forked in.
- **Own implementations**: each lives as a self-contained trial (`model.py` + `env.py`
  + `train.py`) — point an agent (or yourself) at a directory and run it. No framework.
- **Agentic exploration**: results land in `metrics.json` + `experiments/ledger.jsonl`,
  so an agent greps the filesystem to learn what's been tried. See
  [experiment-workflow.md](experiment-workflow.md).

See also: [wandb-conventions.md](wandb-conventions.md), [cuda-kernels.md](cuda-kernels.md).
