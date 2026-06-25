"""Custom-PyTorch training-loop skeleton (the starting point for a new experiment).

This is intentionally tiny and self-contained: a plain PyTorch loop (NOT
Stable-Baselines3) so you own the model and the update step. Copy the whole
`_template/` dir, then grow this file. The shared plumbing -- env, W&B
convention, seeding, metrics/ledger I/O -- comes from the promoted `crafter_rl`
package so every experiment logs consistently.

Run:
    uv run python experiments/<your-exp>/train.py --algo proto --seed 0 --steps 2000 --no-wandb
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch

from crafter_rl.env import CrafterGym
from crafter_rl.utils import append_ledger, get_git_sha, set_seed, write_metrics
from crafter_rl.wandb import init_run

# Trial's model lives next to its out/ dir. Adjust the import to your trial.
from trials.__import_your_model__ import Policy  # noqa: F401  # TODO: real import

EXPERIMENT = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="proto", help="trial/algo label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out-dir", default=None,
                    help="Default: trials/<algo>/out/s<seed>/ under this exp.")
    ap.add_argument("--project", default="Crafter")
    ap.add_argument("--sweep-id", default=None)
    ap.add_argument("--exp-type", default="ablation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # --- identity + paths --------------------------------------------------
    sweep_id = args.sweep_id or f"adhoc-{datetime.now():%m%d-%H%M}"
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        EXP_DIR, "trials", args.algo, "out", f"s{args.seed}"))
    os.makedirs(out_dir, exist_ok=True)

    # --- reproducibility ---------------------------------------------------
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- env + model -------------------------------------------------------
    env = CrafterGym(seed=args.seed)
    # TODO: instantiate your policy from trials/<id>/model.py
    # model = Policy(n_actions=env.action_space.n).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    config = dict(algo=args.algo, seed=args.seed, sweep_id=sweep_id,
                  exp_type=args.exp_type, total_steps=args.steps,
                  env="crafter", git_sha=get_git_sha())

    run = None
    if not args.no_wandb:
        run = init_run(algo=args.algo, seed=args.seed, sweep_id=sweep_id,
                       exp_type=args.exp_type, project=args.project,
                       config=config, out_dir=out_dir)

    # --- training loop (skeleton) -----------------------------------------
    obs, _ = env.reset()
    ep_reward, rewards = 0.0, []
    for step in range(args.steps):
        # TODO: replace random action with model.forward(obs) -> action,
        #       collect rollouts, compute loss, optimizer.step().
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        ep_reward += r
        if term or trunc:
            rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
        # if run: wandb.log({"train/loss": loss.item()}, step=step)

    eval_reward = float(np.mean(rewards)) if rewards else 0.0

    # --- persist results (same schema as every other experiment) ----------
    record = {
        "trial_id": f"{args.algo}/s{args.seed}",
        "experiment": EXPERIMENT,
        "algo": args.algo,
        "seed": args.seed,
        "sweep_id": sweep_id,
        "steps": args.steps,
        "n_envs": 1,
        "eval_reward": round(eval_reward, 4),
        "eval_achievements": 0,  # TODO: count distinct achievements at eval
        "wandb_url": run.get_url() if run else None,
        "git_sha": get_git_sha(),
        "status": "ok",
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    write_metrics(out_dir, record)
    append_ledger(record, ledger_path=os.path.join(
        os.path.dirname(EXP_DIR), "ledger.jsonl"))
    if run:
        run.finish()
    print(f"[{args.algo}-s{args.seed}] done -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
