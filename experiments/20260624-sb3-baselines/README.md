# 20260624-sb3-baselines

Model-free RL baselines on [Crafter](https://github.com/danijar/crafter) using
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/): **PPO**, **A2C**,
and **DQN** with a CNN policy on the raw 64x64 image observations.

## Objective

Establish reference scores (mean eval reward + distinct achievements unlocked)
for the three standard SB3 model-free algorithms on Crafter, trained for 1M env
steps each on a single machine, across multiple seeds. These numbers are the
baseline that later custom-agent experiments are measured against.

## Setup

- 1,000,000 env steps per run.
- On-policy (PPO/A2C): 16 parallel envs. Off-policy (DQN): 1 env + replay buffer.
- Eval: 5 deterministic episodes on a held-out seed.
- Hyperparameters: see `get_hyperparams` / `build_model` in `train.py`.
- sweep_id: `bl0624`.

## Results

5 eval episodes, mean reward ± standard deviation across seeds:

| algo | seeds | eval reward (mean ± std) |
|------|-------|--------------------------|
| PPO  | 0–3   | **5.75 ± 0.89**          |
| A2C  | 0–2   | **4.63 ± 0.09**          |
| DQN  | 0–2   | **3.50 ± 0.28**          |

Per-seed numbers live in each trial's `trials/<algo>/out/s<seed>/metrics.json`
and in the global `../ledger.jsonl`.

PPO is the strongest baseline; A2C is consistent but lower; DQN trails both,
as expected for a single-env value-based method on this sparse-reward task.

**2 planned seeds were skipped** (ran out of the launch budget): `a2c-s3` and
`dqn-s3`. The means above are over the seeds that completed.

## Reproduce

```bash
# single run
uv run python experiments/20260624-sb3-baselines/train.py \
    --algo ppo --seed 0 --steps 1000000

# full sweep (round-robin, deadline-budgeted)
experiments/20260624-sb3-baselines/run_sweep.sh
```

## Links

- W&B project: https://wandb.ai/joeagriffith-home/Crafter
