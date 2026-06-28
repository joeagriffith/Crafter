# Evaluating Long-Term Memory in 3D Mazes (Memory Maze) — synthesis

- **Slug**: pasukonis_memory_maze_2022
- **Authors**: Pasukonis, Jurgis; Lillicrap, Timothy; Hafner, Danijar
- **Year / venue**: 2022 / arXiv 2210.13383
- **Source PDF**: ./pasukonis_memory_maze_2022.pdf
- **Citation**: Pasukonis, J., Lillicrap, T., & Hafner, D. (2022). Evaluating Long-Term Memory in 3D Mazes. arXiv:2210.13383.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
Memory Maze is a 3D first-person procedurally-generated maze benchmark designed to *isolate* long-term memory from the other capacities Crafter and Atari conflate. The agent must reach a sequence of coloured target objects whose locations were briefly visible earlier in the episode; success requires building and querying an internal map across hundreds-to-thousands of steps. Four maze sizes (9×9, 11×11, 13×13, 15×15) span memory horizons from ~1k to ~4k steps. Both **online RL** and **offline probing** protocols are provided. Headline: humans solve large mazes near-perfectly; even DreamerV2/V3 saturate small mazes but fall off sharply on 15×15, and the offline probes show that today's world-model latents do not encode positions beyond ~50 steps.

## Problem framing
The authors argue most RL benchmarks reward agents whose memory is "implicit and short": frame-stacked Atari has ~4-step memory; even DMC humanoid has horizon-shaped tasks that local controllers solve. Long-term memory has historically been tested with toy passive tasks (associative recall, copy task) that don't transfer to embodied control. Memory Maze sits explicitly inside the partial-observation MDP regime and decouples memory from exploration, perception, and dexterity by keeping the action space small and the visual perception trivial.

## Method (env spec)
- **Engine**: dm_control / MuJoCo, 3D first-person.
- **Observation**: 64×64×3 RGB ego-view. No proprioception, no top-down. Optional depth/segmentation for probing.
- **Action**: 6 discrete (forward, back, strafe left/right, turn left/right). Low-dimensional on purpose.
- **Episode**: agent is shown a coloured target (cue), navigates to it for reward, then a new target colour is requested, repeat. Maze layout fixed within episode, randomised between episodes. Wall layouts and target positions are procedurally generated each episode.
- **Sizes**: 9×9, 11×11, 13×13, 15×15 tile mazes with episode caps of 1 000 / 2 000 / 3 000 / 4 000 steps. Larger mazes hold more targets (3, 4, 5, 6) and demand keeping more positions in memory.
- **Reward**: +1 per correct target reach; sparse.
- **Online RL setup**: standard PPO / IMPALA / Dreamer training on the env.
- **Offline probing**: collect trajectories from a trained or random agent, freeze the agent's internal state stream, train a *linear* probe to predict (agent xy position, target xy positions, wall layout) from the latent at each step. This measures *what the world model remembers*, separate from how well it acts.

## Key results
Reported scores (mean episode reward; human baseline collected by authors):
- 9×9: Human ~36, DreamerV2 ~32 (close)
- 11×11: Human ~36, DreamerV2 ~25
- 13×13: Human ~32, DreamerV2 ~16
- 15×15: Human ~28, DreamerV2 ~9 (large gap)
- IMPALA: ~half of Dreamer at every size; basically chance on 15×15.
- TBPTT length (truncated backprop through time) is the dominant knob: doubling sequence length from 50 → 200 doubles 15×15 performance for Dreamer.

Offline probe results (more diagnostic):
- Agent position decodable from RSSM h_t with ~90% R² at horizon < 50 steps, decays to chance by ~500 steps.
- Target positions encoded only after recent observation; rapidly forgotten.
- Wall layout poorly encoded — agent reasons about local geometry, not a global map.

## Limitations
- Only one task family (find-target); no skill diversity à la Crafter.
- Perception is deliberately easy; the env doesn't stress representation learning.
- The probing is linear, which may underestimate non-linear encoding.
- Sparse reward + long horizon may interact with exploration in ways not cleanly separated from memory per se.
- Scaling beyond 15×15 not tested; unclear if the trend continues smoothly or has a phase transition.

## Idea seeds for our loop
- **Memory Maze is the natural transfer/diagnostic target after Crafter.** Crafter doesn't stress memory; Memory Maze does. An iterative-PC agent with timescale-separated layers (slow top layer = long-horizon memory) has a natural story here. **
- **Offline-probe methodology is gold for us.** We should adopt their linear-probe protocol on our PC agent: predict (position, target, wall-layout, recent obs) from the settled latents at each layer. If iterative inference *does* improve memory, the probe will show it before the reward curve does. This converts a noisy RL signal into a clean representation-learning signal. **
- **Timescale separation as a hypothesis**: PC with τ_top >> τ_bottom should outperform amortised RSSM (single GRU) on 15×15 specifically. Concrete trial: same world-model arch, same actor-critic, only vary whether the top latent has slow dynamics (PC settle with high effective τ) vs fast (DreamerV3 default). Probe at 50 / 200 / 1000-step horizons.
- **TBPTT-equivalent in PC**: their finding that TBPTT length is the dominant knob translates to "how many PC settle steps per env step" in our setup. We should sweep settle_K on 15×15 and look for the same monotone improvement.
- **PC's iterative settle as test-time memory consolidation**: on a long episode, additional settle iterations at observation-time could replay the past via top-down prior + current bottom-up evidence. Test: does adding settle steps *after each obs* help recall in 15×15?
- **Cross-task transfer**: train an iterative-PC agent on Crafter, then evaluate (zero-shot or fine-tune) on Memory Maze. If our iterative settle is a generic mechanism rather than task-specific, the transfer pattern should differ from DreamerV3's.

## Connections
- `[hafner_dreamerv3_2023]` — DreamerV3 paper used Memory Maze as a benchmark; this paper introduces it.
- `[hafner_crafter_2022]` — sister benchmark; Crafter stresses skill/exploration, Memory Maze stresses memory. Two halves of one capability spectrum.
- `[hafner_dreamerv2_2020]` — the headline baseline here, with TBPTT-length as the dominant knob.
- `[marino_iterative_amortized_policy_2020]` — iterative policy refinement at decision time may be where our PC settle pays off in long-horizon recall.

## Relevant themes
iteration-for-OOD-evidence; biological (cognitive-map / hippocampus framing); active-inference; supporting (env paper).

## Tier
high

## Out of scope
—

## Notes / follow-ups
- 2026-06-22 / agent-20260622: PDF fetched cleanly (3.16 MB). Probe-methodology numbers above are stated with my best recollection from the paper; should be cross-verified against the figures before any hypothesis depends on the exact decay rates. Authors note future work on harder memory tasks (key-pickup, multi-room) that haven't been released as of the paper.
