# Weights & Biases Conventions

The goal: with hundreds of runs, **any set of runs you care about is one filter away.**
All of this is applied automatically by `crafter_rl.wandb.init_run(...)` — don't hand-roll
`wandb.init`.

## The linchpin: `sweep_id`
Every launch gets a short, stable `sweep_id` stamped into the run's name, group, tags,
**and** config. One launch = one `sweep_id`. Examples: `bl0624` (baselines, Jun 24),
`dreamer0701`, `ppo-lr-sweep-0705`. If omitted, an `adhoc-<MMDD-HHMM>` id is auto-generated
so names never collide.

## The five W&B axes and how we use each

| Axis | Job | Value |
|------|-----|-------|
| `config` | the workhorse — every hyperparam + metadata → filterable/sortable columns | all algo hyperparams + `sweep_id, exp_type, seed, total_steps, n_envs, git_sha, crafter_version, sb3_version, torch_version` |
| `group` | aggregate seeds of one config into a mean±band | `f"{sweep_id}/{algo}"` |
| `job_type` | the run's role | `train` / `eval` / `debug` |
| `tags` | click-filters in the UI | `[sweep_id, algo, "env:crafter", exp_type]` |
| `name` | unique + human-readable | `f"{sweep_id}-{algo}-s{seed}"` → `bl0624-ppo-s0` |

`exp_type` ∈ {`baseline`, `ablation`, `tuning`, `debug`}.

## Finding runs among hundreds
- *"that sweep"* → filter `config.sweep_id = bl0624` (or click the `bl0624` tag)
- *"every PPO ever"* → `config.algo = ppo`
- *"best config"* → group the runs table by `group`, sort by `eval/mean_reward`
- *"seeds of one config"* → already one group with a variance band
- *"reproduce this run"* → `config.git_sha` + the logged hyperparameters

## Project layout
Single W&B project **`crafter`** (entity `joeagriffith-home`), sliced by `sweep_id`/tags.
We deliberately do **not** spin up a project per experiment — one project keeps
cross-experiment comparison trivial.

## Reproducibility metadata (always logged)
`git_sha` (short SHA, or `"uncommitted"`), `crafter_version`, `sb3_version`,
`torch_version`. Local W&B files are written into the trial's `out/` dir (`wandb.init(dir=...)`),
not a top-level `wandb/`.

> The first baseline sweep (`bl0624`, 10 runs) was retroactively migrated to this scheme.
