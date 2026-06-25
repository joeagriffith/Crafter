# Experiment Workflow

The loop for running experiments — designed so a human *or* an autonomous agent can drive
it the same way. Adapted from Codeca's `optim-loop`.

## 1. Start an experiment
Copy the template into a dated directory:
```bash
cp -r experiments/_template "experiments/$(date +%Y%m%d)-<slug>"
```
- **Experiment dir**: `YYYYMMDD-<kebab-slug>` (e.g. `20260625-policy-freeze`).
- **Trial dir**: `trials/NNN_<slug>` (e.g. `trials/001_baseline`, `trials/002_freeze-encoder`).

## 2. The trial contract (3 files)
Each trial is self-contained — point at the directory and run it:
| File | Role |
|------|------|
| `model.py` | the architecture / policy being tested |
| `env.py` | the environment (usually `from crafter_rl.env import CrafterGym`; override if the trial needs a custom env) |
| `train.py` | the training loop (often inherited from the suite root; override per trial when needed) |

Shared `env.py`/`train.py` can live at the **experiment root** and be reused by every trial;
a trial only adds its own file when it needs to differ. (For DL projects the third file is
`data.py`; for RL it's `env.py`.)

## 3. Run it
Every run writes **all** its outputs into the trial that produced it:
```
experiments/<exp>/trials/<trial>/out/[s<seed>/]   # tb logs, checkpoints, wandb-local, metrics.json
```
There is no top-level `runs/`. Use `crafter_rl.wandb.init_run(...)` for tracking and pass
`out_dir` so W&B's local files land there too.

## 4. Record results (the machine-readable contract)
On completion, write `out/.../metrics.json` and append one line to `experiments/ledger.jsonl`
via `crafter_rl.utils.write_metrics(...)` / `append_ledger(...)`.

**`metrics.json` schema:**
```json
{"trial_id": "<algo-or-trial>/s<seed>", "experiment": "<exp-dir>", "algo": "...",
 "seed": 0, "sweep_id": "...", "steps": 1000000, "n_envs": 16,
 "eval_reward": 0.0, "eval_achievements": 0, "wandb_url": "...",
 "git_sha": "...", "status": "ok|failed|partial", "created": "YYYY-MM-DD"}
```
`experiments/ledger.jsonl` holds the same object, one per line — so an agent learns the
whole history with `cat experiments/ledger.jsonl`, no W&B API call needed.

## 5. Agentic iteration (autoresearch loop)
- `README.md` at the experiment root states the **objective + current best result**.
- Keep a `hypotheses/index.md` — one line per hypothesis with a terminal status
  (`confirmed` / `refuted` / `blocked` / `partial`). **Append-only**; never rewrite history.
- An orchestrator agent: reads `ledger.jsonl` + `hypotheses/index.md` → proposes the next
  trial → spawns a sub-agent to run it → reads its `metrics.json` → appends the outcome.
- Cap trials per session and keep each hypothesis to one sub-agent for clean attribution.

## 6. Promote what works
When a trial's code earns its place, graduate it into `crafter_rl/` per
[promotion-checklist.md](promotion-checklist.md).
