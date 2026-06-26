# 20260625-dreamerv3 — DreamerV3 strong baseline (Crafter)

## Objective
A **strong model-based baseline** for Crafter to sit alongside the SB3 model-free
baselines (`20260624-sb3-baselines`, best PPO ≈ 6.1 reward / 11 achievements).
Target: **Crafter score ≈ 14%** (DreamerV3 territory), roughly **~14 eval reward**.

DreamerV3 here is the PyTorch reproduction in **`NM512/r2dreamer`** (ICLR 2026
R2-Dreamer repo; its DreamerV3 reproduction is selected with `model.rep_loss=dreamer`).
The repo also advertises a ~5x speedup over the popular `dreamerv3-torch` codebase.

- Vendored impl: `third_party/r2dreamer/` (git-ignored; rebuilt by `setup.sh`)
- **Pinned commit:** `546e4fab8146ea4b14e1d7726bbc1a8a1d50322f`
- Isolated env: `third_party/r2dreamer/.venv` (**Python 3.11**, `torch==2.8.0`).
  The main project `.venv` (SB3 / torch-cu130) is never touched.

## Reproduce the environment
```bash
bash experiments/20260625-dreamerv3/setup.sh
```
Idempotent: clones + pins r2dreamer, makes the isolated 3.11 venv, installs the
`crafter` extra. r2dreamer pins `requires-python >=3.11,<3.12`, so **3.11 is required**
(3.12 is rejected by dependency resolution).

## Launch the real run (NOT auto-executed)
```bash
# ~6-12h wall-clock, ~6-10GB VRAM on one RTX 4090
SEED=0 SWEEP_ID=dv30625 bash experiments/20260625-dreamerv3/run_dreamerv3.sh
```
which runs (from `third_party/r2dreamer/`, isolated venv):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    env=crafter model=size200M model.rep_loss=dreamer model.compile=True \
    device=cuda:0 buffer.storage_device=cuda:0 \
    seed=0 logdir=<exp>/trials/dreamerv3/out/s0
```
Budget is the Crafter standard **1.01e6 env steps** (`env.steps` in
`configs/env/crafter.yaml`); eval every 1e4 steps over 10 episodes.

## Outputs & where results flow
Everything lands in `trials/dreamerv3/out/s<seed>/` (== Hydra `logdir`):
`metrics.jsonl` (per-eval scalars), TensorBoard event files, `console.log`,
`train.log`, `latest.pt`.

After training, `run_dreamerv3.sh` calls **`report_results.py`** (in the MAIN
venv) which:
1. parses the last eval record from `metrics.jsonl`
   (`episode/eval_score` → `eval_reward`; count of `episode/eval_<achiev>` with
   mean > 0 → `eval_achievements`);
2. writes `out/s<seed>/metrics.json` (13-field schema) and appends one line to
   `experiments/ledger.jsonl` via `crafter_rl.utils`;
3. with `--wandb`, opens a run via `crafter_rl.wandb.init_run(algo="dreamerv3",
   exp_type="baseline", ...)` and **replays the eval curves** from `metrics.jsonl`
   into it (project `crafter`, entity `joeagriffith-home`, `sweep_id` naming).

### W&B caveat (read before launch)
r2dreamer trains in its **own subprocess + venv with no wandb**, so true in-process
`sync_tensorboard` is not available during training. Two supported paths:
- **Default (implemented):** `report_results.py` replays `metrics.jsonl` eval
  scalars into a wandb run after training — gives clean eval curves + summary.
- **Full TB curves (optional):** also mirror the raw TensorBoard scalars with
  ```bash
  wandb sync --sync-tensorboard --project crafter --entity joeagriffith-home \
      experiments/20260625-dreamerv3/trials/dreamerv3/out/s0
  ```
  (`init_run` already passes `sync_tensorboard=True`, harmless when no live writer.)

## Note on the Crafter "score"
Our schema records `eval_reward` + distinct `eval_achievements`. The canonical
Crafter **score** (geometric mean of per-achievement success rates, the ≈14%
target) needs success *rates* across training; r2dreamer logs per-achievement mean
*counts* per eval, so the headline score is best read off TensorBoard/W&B
(`episode/eval_*`) rather than reconstructed in `metrics.json`.

## Status
Scaffold + isolated env prepared and smoke-checked. **The 1M-step run has NOT been
started** — launch deliberately with the command above.
