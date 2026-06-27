"""Train a model-free RL baseline on Crafter (PPO / A2C / DQN via Stable-Baselines3).

All baselines use a CNN policy on Crafter's 64x64 image observations and log to
Weights & Biases. This is the standard Crafter model-free baseline setup, sized
to train on a single machine.

Shared infra (env, W&B convention, git/version/metrics/ledger helpers) lives in
the promoted ``crafter_rl`` package; only the SB3-specific hyperparameters and
model construction live here.

Usage:
    uv run python experiments/20260624-sb3-baselines/train.py --algo ppo --seed 0 --steps 1000000
    uv run python experiments/20260624-sb3-baselines/train.py --algo dqn --seed 1 --steps 1000000 --no-wandb
"""
import argparse
import os
from datetime import datetime

import crafter
import stable_baselines3
import torch
import wandb
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecMonitor)
from wandb.integration.sb3 import WandbCallback

from crafter_rl.env import CrafterGym
from crafter_rl.eval import evaluate as shared_evaluate
from crafter_rl.utils import (append_ledger, get_git_sha, pkg_version,
                              write_metrics)
from crafter_rl.wandb import init_run

ALGOS = {"ppo": PPO, "a2c": A2C, "dqn": DQN}
EXPERIMENT = "20260624-sb3-baselines"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def make_env(seed):
    def _init():
        return CrafterGym(seed=seed)
    return _init


def get_hyperparams(algo):
    """Single source of truth for each algo's hyperparameters.

    Keys use config-style names (e.g. ``lr``); ``build_model`` translates them to
    the matching SB3 keyword arguments so the model and the W&B config can never
    drift apart.
    """
    if algo == "ppo":
        return dict(lr=3e-4, gamma=0.99, ent_coef=0.01,
                    n_steps=256, batch_size=256, n_epochs=4)
    elif algo == "a2c":
        return dict(lr=7e-4, gamma=0.99, ent_coef=0.01, n_steps=5)
    else:  # dqn
        return dict(lr=1e-4, gamma=0.99, buffer_size=100_000,
                    learning_starts=10_000, batch_size=32, train_freq=4,
                    target_update_interval=1000, exploration_fraction=0.1,
                    exploration_final_eps=0.05)


def build_model(algo, seed, logdir, n_envs, hp):
    """Construct the vec-env and SB3 model for the given algorithm."""
    if algo in ("ppo", "a2c"):
        # On-policy: many parallel envs (CPU-bound is the bottleneck on Crafter).
        venv = SubprocVecEnv([make_env(seed * 1000 + i) for i in range(n_envs)])
    else:
        # DQN is off-policy / single-env with a replay buffer.
        venv = DummyVecEnv([make_env(seed * 1000)])
    venv = VecMonitor(venv)

    common = dict(policy="CnnPolicy", env=venv, verbose=1, seed=seed,
                  learning_rate=hp["lr"], gamma=hp["gamma"],
                  tensorboard_log=logdir, device="cuda")
    if algo == "ppo":
        model = PPO(n_steps=hp["n_steps"], batch_size=hp["batch_size"],
                    n_epochs=hp["n_epochs"], ent_coef=hp["ent_coef"], **common)
    elif algo == "a2c":
        model = A2C(n_steps=hp["n_steps"], ent_coef=hp["ent_coef"], **common)
    else:  # dqn
        model = DQN(buffer_size=hp["buffer_size"],
                    learning_starts=hp["learning_starts"],
                    batch_size=hp["batch_size"], train_freq=hp["train_freq"],
                    target_update_interval=hp["target_update_interval"],
                    exploration_fraction=hp["exploration_fraction"],
                    exploration_final_eps=hp["exploration_final_eps"], **common)
    return model, venv


def evaluate(model, n_episodes=10):
    """Standardized eval via the shared ``crafter_rl.eval`` module.

    Wraps the SB3 model's greedy ``predict`` as a plain ``act_fn(obs) -> int`` so
    SB3 baselines are scored on exactly the same code path (same seeds, same
    canonical Crafter score, same per-achievement success rates) as every other
    algorithm. ``n_episodes`` defaults to 10 to match the DreamerV3 eval.
    """
    def act_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return shared_evaluate(act_fn, n_episodes=n_episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS), default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="Trial out dir. Default: trials/<algo>/out/s<seed>/ "
                         "relative to this experiment dir.")
    ap.add_argument("--project", default="Crafter")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sweep-id", default=None,
                    help="Groups runs into a sweep; auto-generated if omitted.")
    ap.add_argument("--exp-type", default="baseline",
                    help="baseline/ablation/tuning/debug.")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--ledger", default=os.path.join(
        os.path.dirname(EXP_DIR), "ledger.jsonl"),
        help="Path to the append-only ledger.jsonl.")
    ap.add_argument("--no-wandb", action="store_true")
    # --no-record: profiling-only mode. Skip W&B init, the final eval, the
    # metrics.json write AND the ledger append, so a profiling run never
    # pollutes experiments/ledger.jsonl. Implies --no-wandb.
    ap.add_argument("--no-record", action="store_true",
                    help="Profiling mode: skip wandb, eval, metrics.json and "
                         "ledger append (keeps ledger.jsonl clean).")
    # --profile-torch DIR: wrap a short model.learn under torch.profiler and
    # exit. Writes a chrome/tensorboard trace + a key_averages() table to DIR.
    ap.add_argument("--profile-torch", default=None, metavar="DIR",
                    help="Run a short torch.profiler-instrumented learn into "
                         "DIR, dump the op table, then exit (no record).")
    args = ap.parse_args()

    # --no-record implies no W&B (and the torch profiler path never records).
    if args.no_record or args.profile_torch:
        args.no_wandb = True

    sweep_id = args.sweep_id or f"adhoc-{datetime.now():%m%d-%H%M}"
    algo = args.algo

    # All trial artifacts (tensorboard, checkpoint, wandb local files, metrics)
    # live under one out dir.
    out_dir = args.out_dir or os.path.join(
        EXP_DIR, "trials", algo, "out", f"s{args.seed}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    name = args.name or f"{sweep_id}-{algo}-s{args.seed}"

    hp = get_hyperparams(algo)
    config = dict(
        algo=algo, seed=args.seed, sweep_id=sweep_id, exp_type=args.exp_type,
        total_steps=args.steps, n_envs=args.envs, policy="CnnPolicy",
        env="crafter", git_sha=get_git_sha(),
        crafter_version=pkg_version(crafter, "crafter"),
        sb3_version=pkg_version(stable_baselines3, "stable_baselines3"),
        torch_version=pkg_version(torch, "torch"),
        **hp)

    run = None
    if not args.no_wandb:
        run = init_run(algo=algo, seed=args.seed, sweep_id=sweep_id,
                       exp_type=args.exp_type, project=args.project,
                       config=config, out_dir=out_dir, name=args.name,
                       notes=args.notes)

    # tensorboard_log routed under the trial out dir.
    model, venv = build_model(algo, args.seed, out_dir, args.envs, hp)
    print(f"[{name}] training {args.algo.upper()} for {args.steps:,} steps "
          f"on {model.device} ...")
    callback = WandbCallback(verbose=1) if run else None

    # --- torch profiler path: short instrumented learn, dump table, exit. -----
    if args.profile_torch:
        prof_dir = os.path.abspath(args.profile_torch)
        os.makedirs(prof_dir, exist_ok=True)
        from torch.profiler import (ProfilerActivity, profile, schedule,
                                     tensorboard_trace_handler)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        # wait/warmup/active steps -> a handful of profiled rollout+update cycles.
        sched = schedule(wait=1, warmup=1, active=3, repeat=1)
        prof_steps = min(args.steps, max(hp.get("n_steps", 256) * args.envs * 6,
                                         3000))
        with profile(activities=activities, schedule=sched,
                     on_trace_ready=tensorboard_trace_handler(prof_dir),
                     record_shapes=True, with_stack=False) as prof:
            class _Step:
                def _on_step(self_inner):
                    prof.step()
                    return True
            from stable_baselines3.common.callbacks import BaseCallback

            class StepCb(BaseCallback):
                def _on_step(self_cb):
                    prof.step()
                    return True

            model.learn(total_timesteps=prof_steps, callback=StepCb())
        sort_key = ("self_cuda_time_total" if torch.cuda.is_available()
                    else "self_cpu_time_total")
        table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
        table_path = os.path.join(prof_dir, "key_averages.txt")
        with open(table_path, "w") as f:
            f.write(f"sorted by {sort_key}\n\n{table}\n")
        print(table)
        print(f"[{name}] torch profiler trace + table -> {prof_dir}")
        venv.close()
        return

    status = "ok"
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
    except Exception as e:
        status = "failed"
        print(f"[{name}] training failed: {e}")
        raise
    finally:
        model_path = os.path.join(out_dir, "checkpoint")
        model.save(model_path)
        print(f"[{name}] saved model -> {model_path}.zip")

    # --no-record (profiling): stop here. No eval, no metrics.json, no ledger.
    if args.no_record:
        venv.close()
        print(f"[{name}] --no-record: skipped eval/metrics/ledger.")
        return

    # Standardized eval: 10 episodes (matches DreamerV3) on the shared eval path.
    ev = evaluate(model, n_episodes=10)
    mean_r = ev["return_mean"]
    crafter_sc = ev["crafter_score"]
    n_distinct = ev["achievements_unlocked"]            # legacy back-compat
    success_rates = ev["achievement_success_rates"]     # {name: fraction}
    print(f"[{name}] eval mean return     : {mean_r:.2f} "
          f"(+/- {ev['return_std']:.2f}, n={ev['n_episodes']})")
    print(f"[{name}] crafter score        : {crafter_sc:.2f}")
    print(f"[{name}] distinct achievements: {n_distinct}")

    wandb_url = run.get_url() if run else None
    if run:
        log = {"eval/mean_reward": mean_r,
               "eval/return_std": ev["return_std"],
               "eval/crafter_score": crafter_sc,
               "eval/distinct_achievements": n_distinct}
        # per-achievement success rates (fraction 0..1) -> W&B for parity with
        # the canonical Crafter per-achievement breakdown.
        for ach_name, rate in success_rates.items():
            log[f"eval/achievement/{ach_name}"] = rate
        wandb.log(log)
        wandb.summary["eval/mean_reward"] = mean_r
        wandb.summary["eval/crafter_score"] = crafter_sc
        wandb.summary["eval/distinct_achievements"] = n_distinct
        run.finish()
    venv.close()

    record = {
        "trial_id": f"{algo}/s{args.seed}",
        "experiment": EXPERIMENT,
        "algo": algo,
        "seed": args.seed,
        "sweep_id": sweep_id,
        "steps": args.steps,
        "n_envs": args.envs,
        "eval_reward": round(mean_r, 4),            # comparable mean return
        "crafter_score": round(crafter_sc, 4),      # canonical Crafter score
        "eval_achievements": n_distinct,            # distinct count (back-compat)
        "achievement_success_rates": {k: round(v, 4)
                                      for k, v in success_rates.items()},
        "wandb_url": wandb_url,
        "git_sha": get_git_sha(),
        "status": status,
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    write_metrics(out_dir, record)
    append_ledger(record, ledger_path=args.ledger)
    print(f"[{name}] wrote metrics.json + appended ledger -> {args.ledger}")


if __name__ == "__main__":
    main()
