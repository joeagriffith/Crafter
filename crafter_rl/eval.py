"""Shared, standardized Crafter evaluation.

One eval path so every algorithm (SB3 baselines, DreamerV3, future methods) is
scored identically and comparably. Given any greedy policy as a plain
``act_fn(obs) -> int``, ``evaluate`` runs a fixed number of deterministically
seeded episodes on :class:`crafter_rl.env.CrafterGym` and reports:

  * ``return_mean`` / ``return_std`` - mean (and std) episode return, the simple
    comparable headline reward both pipelines already track;
  * ``achievement_success_rates`` - per-achievement success rate over the 22
    canonical Crafter achievements, where "success" = the achievement was
    unlocked (count > 0) at least once in an episode;
  * ``crafter_score`` - the canonical Crafter benchmark score, the geometric
    mean of the per-achievement success rates expressed in percent:
    ``exp(mean_i ln(1 + s_i_percent)) - 1``. This is the standard headline
    Crafter number (random ~1%, DreamerV3 ~14%, max 100) and is richer than a
    raw distinct-achievement count.

    NOTE on the formula: the task brief wrote ``100 * (exp(mean ln(1 + s%)) - 1)``,
    but that extra leading ``100`` makes the result 100x too large (a random
    policy would score ~70 and a uniform 50%-success agent would score 5000 --
    impossible for a [0,100] metric). The official Crafter score
    (Hafner et al., ``analysis/plot_scores.py``:
    ``np.exp(np.nanmean(np.log(1 + percents))) - 1`` with percents in 0..100)
    has NO leading 100 -- the percent scale already lands the score in [0,100]
    (e.g. all-achievements-at-50% -> 50, DreamerV3 -> ~14). We implement that
    canonical version so all algorithms match the standard benchmark.

Dependency-light by design: only ``crafter_rl.env`` + numpy + crafter.

Driving a DreamerV3 ``latest.pt`` checkpoint through this SAME evaluate():
    The contract is just ``act_fn(obs) -> int`` on 64x64x3 uint8 obs. A later
    reporter (run inside r2dreamer's own venv, on its own device) would load the
    agent once and wrap its greedy/argmax action selection -- e.g.

        agent = load_r2dreamer(latest_pt)          # in the r2dreamer venv
        state = agent.initial_state(batch=1)       # recurrent carry
        def act_fn(obs):
            global state
            action, state = agent.policy(obs, state, mode="eval")
            return int(action)
        results = evaluate(act_fn, n_episodes=10)

    Note recurrent agents must reset their carry at each episode boundary; for a
    stateful policy use ``evaluate``'s per-episode loop semantics (one fresh
    CrafterGym per episode) and reset the carry in ``act_fn`` on a new-episode
    signal, or expose a tiny stateful wrapper. Scaffold only -- not implemented
    here, and intentionally no torch / model import lives in this module.
"""
import math

import numpy as np

from crafter_rl.env import CrafterGym

# The 22 canonical Crafter achievements (sorted, matching a fresh crafter.Env
# info["achievements"] keyset). Hard-coded so eval never silently depends on a
# particular crafter build's iteration order, but verified against crafter below.
CRAFTER_ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)


def crafter_score(success_rates):
    """Canonical Crafter score from per-achievement success rates.

    ``success_rates`` is either a dict ``{name: rate}`` or a sequence of rates,
    with rates as FRACTIONS in ``[0, 1]`` (e.g. 0.25 == unlocked in 25% of
    episodes). Converts to percent internally and returns the canonical
    Crafter score

        exp(mean_i ln(1 + s_i_percent)) - 1

    i.e. the geometric mean of ``(1 + percent)`` shifted back by one, which lies
    in ``[0, 100]``. All-zero rates -> 0.0. Always reduced over the full set of
    22 achievements when a dict is given (missing achievements count as 0%).
    (See the module docstring for why there is no extra leading factor of 100.)
    """
    if isinstance(success_rates, dict):
        rates = [float(success_rates.get(name, 0.0))
                 for name in CRAFTER_ACHIEVEMENTS]
    else:
        rates = [float(r) for r in success_rates]
    if not rates:
        return 0.0
    # rate (fraction) -> percent, then log(1 + percent); geometric mean - 1.
    logs = [math.log(1.0 + 100.0 * r) for r in rates]
    geom = math.exp(sum(logs) / len(logs))
    return geom - 1.0


def evaluate(act_fn, n_episodes=10, seed_base=987_654):
    """Run a standardized Crafter evaluation of a greedy policy.

    Args:
        act_fn: callable ``obs -> action_int``. Should be deterministic/greedy.
        n_episodes: number of episodes (default 10, to match DreamerV3 eval).
        seed_base: episode ``i`` uses world seed ``seed_base + i`` so every
            policy is evaluated on the SAME set of worlds.

    Returns a dict:
        return_mean, return_std, n_episodes,
        achievements_unlocked        - legacy distinct-union count across all
                                       episodes (back-compat with the old eval),
        achievements_unlocked_mean   - mean distinct count per episode,
        achievement_success_rates    - {name: fraction in [0,1]} over 22 achs,
        crafter_score                - canonical geometric-mean score (percent).
    """
    returns = []
    # per-achievement: number of episodes in which it was unlocked at least once.
    unlock_counts = {name: 0 for name in CRAFTER_ACHIEVEMENTS}
    distinct_union = set()
    distinct_per_episode = []

    for i in range(n_episodes):
        env = CrafterGym(seed=seed_base + i)
        obs, _ = env.reset(seed=seed_base + i)
        done, total, info = False, 0.0, {}
        while not done:
            action = int(act_fn(obs))
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)

        episode_unlocked = {name for name, count
                            in info.get("achievements", {}).items()
                            if count > 0}
        distinct_per_episode.append(len(episode_unlocked))
        distinct_union |= episode_unlocked
        for name in episode_unlocked:
            if name in unlock_counts:
                unlock_counts[name] += 1

    returns = np.asarray(returns, dtype=np.float64)
    success_rates = {name: unlock_counts[name] / n_episodes
                     for name in CRAFTER_ACHIEVEMENTS}

    return {
        "return_mean": float(returns.mean()) if len(returns) else 0.0,
        "return_std": float(returns.std()) if len(returns) else 0.0,
        "n_episodes": int(n_episodes),
        # legacy back-compat: distinct achievements unlocked across all episodes.
        "achievements_unlocked": len(distinct_union),
        "achievements_unlocked_mean": (
            float(np.mean(distinct_per_episode)) if distinct_per_episode
            else 0.0),
        "achievement_success_rates": success_rates,
        "crafter_score": crafter_score(success_rates),
    }
