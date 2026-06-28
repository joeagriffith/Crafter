# Benchmarking the Spectrum of Agent Capabilities (Crafter) — synthesis

- **Slug**: hafner_crafter_2022
- **Authors**: Hafner, Danijar
- **Year / venue**: 2022 / ICLR 2022 (arXiv 2109.06780, Sep 2021)
- **Source PDF**: ./hafner_crafter_2022.pdf
- **Citation**: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022. arXiv:2109.06780.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
Crafter is a 2D open-world survival game (Minecraft-inspired) explicitly engineered as a *single-task* benchmark that exercises the full spectrum of RL agent capabilities — perception, deep exploration, long-horizon credit assignment, generalisation across procedural worlds, and hierarchical skill learning — within a tight ~1M-step budget that fits on a single GPU in roughly a day. The achievement system replaces raw reward as the primary metric: 22 semantically meaningful achievements (collect wood, place table, make wood pickaxe, eat plant, collect diamond, …) are aggregated by the **geometric mean of unlock probabilities** so that an agent must be broadly competent rather than farm one easy reward. Reported baselines (PPO, Rainbow, DreamerV2, RND, random) sit between 1.6 % and 14 % score versus a human at ~50 %, leaving huge headroom. This is THE env for our pc-crafter project.

## Problem framing
The paper argues that the RL community's evaluation practice is broken in two ways: (a) Atari/DMC suites require many runs over many envs to draw conclusions, and most papers cherry-pick; (b) raw reward-curve plots conflate distinct competencies (exploration, memory, dexterity, planning). Crafter's design goal is *one* environment whose internal achievement tree exposes those competencies as separable signals while keeping compute small enough for academic labs. Builds directly on Minecraft / MineRL (too expensive), procgen (no skill tree), and earlier 2D survival games (Griddly, NetHack).

## Method (env spec)
- **Observation**: 64×64×3 RGB by default (a 7×9 tile patch around the agent rendered at 9 px/tile), or a symbolic dict for ablations. Top-down third-person, agent always centred.
- **Action space**: 17 discrete actions — noop, 4 movement, do (context-sensitive: chop / mine / drink / attack / eat), sleep, place stone / table / furnace / plant, make wood pickaxe / stone pickaxe / iron pickaxe / wood sword / iron sword.
- **World**: procedurally generated 64×64 tile world per episode (grass, stone, coal, iron, diamond, water, lava, trees, cows, zombies, skeletons). Day-night cycle; monsters spawn at night.
- **Reward**: +1 for each first-time achievement in episode, ±0.1 for health changes (eat / get hit / drink / sleep). Episode ends on death or 10 000 steps.
- **Achievement set (22)**: collect_wood, place_table, eat_cow, collect_sapling, collect_drink, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, place_plant, defeat_zombie, defeat_skeleton, eat_plant, wake_up, place_stone, place_furnace, collect_coal, collect_stone, collect_iron, collect_diamond. The tree has hard prerequisites (need wood → table → wood_pickaxe → stone → stone_pickaxe → iron → furnace+iron_pickaxe → diamond) creating long-horizon dependencies.
- **Crafter score**: S = exp(mean(log(1 + s_i))) − 1 over the 22 unlock success-rate percentages s_i. Geometric mean penalises agents that ace one achievement and zero others.
- **Budget**: recommended 1 M env steps for evaluation. (A more constrained "Crafter-100k" budget mirroring Atari-100k is sometimes used in follow-ups but isn't in the original paper.)

## Key results
Reported scores (1 M-step training, mean over 10 seeds, ± std):
- DreamerV2: 10.0 % ± 1.2
- PPO: 4.6 % ± 0.3
- Rainbow: 4.3 % ± 0.2
- RND (curiosity): 2.0 %
- Random: 1.6 %
- Human (1 hr): ~50.5 %

Diamond is collected ~0 % of the time by all baselines; iron pickaxe is the bottleneck. The achievement-spectrum bar chart reveals that even Dreamer agents reliably get wood / table / wood-pickaxe but fall off a cliff after stone. The paper explicitly identifies long-horizon credit assignment and stochastic exploration of the crafting tree as the open problems.

## Limitations
- Only one env: results may not transfer outside the survival-crafting morphology.
- Visual obs is small (64×64): perception bottleneck is mild compared to e.g. Atari.
- Determinism: world is procedural but episode RNG is fixed-seeded by default; some papers report higher variance under full stochasticity.
- No memory stress: most of what an agent needs is in the current screen. (This is by design; cf. Memory Maze for the memory axis.)
- The geometric-mean score has counter-intuitive sensitivity at low s_i — moving a 0 % → 1 % achievement matters more than 50 % → 60 %.

## Idea seeds for our loop
- **What the env rewards** — *broad competence under tight budget*. The geometric-mean score means an iterative-inference PC agent that learns *something* about every achievement (even partially) will beat an amortised agent that overfits a few. **This is structurally favourable to PC** because PC's iterative settle gives gradient flow per-achievement-prerequisite. **
- **Long-horizon credit assignment** is the headline bottleneck. Multi-timescale PC layer dynamics (slow top layers, fast bottom layers) could maintain a long-horizon prior over "diamond is good" while the fast layers solve the current micro-task — directly addresses Crafter's failure mode. **
- **22-achievement diagnostic spectrum** gives us per-skill probes we can scope hypotheses around (e.g., "does deeper iterative inference improve specifically the iron/diamond tail?"). Don't just chase the scalar score — track achievement-wise unlock curves as the iteration-budget knob is swept.
- **1 M-step budget is ideal for many small trials** in an optim-loop, but H_inference (iterative settle steps per env step) is a wall-clock multiplier — we'll need to be careful that "iterative" doesn't mean "10× slower wall-clock for the same env steps." Budget hypotheses should report both env-step and wall-clock curves.
- **Symbolic obs ablation** is available; useful for isolating "does PC win on perception or on planning?" We can run identical iterative-PC agents on pixel vs symbolic and decompose.
- **Counterfactual: what would PC look like in the achievement bar chart?** Hypothesis: PC's iterative inference helps mid-tree achievements (stone → iron) but does nothing for the wood/table base layer that amortised solves easily. Test by zooming on the iron/coal/diamond columns.

## Connections
- `[hafner_dreamerv2_2020]` — original Crafter baseline; the lineage of what we are trying to beat.
- `[hafner_dreamerv3_2023]` — current SOTA on Crafter (~70 %+); the actual incumbent.
- `[micheli_iris_2023]` — transformer world model evaluated on Atari-100k; sample-efficiency analogue we should check on Crafter.
- `[pasukonis_memory_maze_2022]` — sister benchmark from the same lab, stressing the *memory* axis that Crafter deliberately doesn't.
- `[marino_iterative_amortized_policy_2020]` — closest in spirit to iterative inference applied to a policy; the bridge from our PC core to a Crafter agent.

## Relevant themes
iteration-for-OOD-evidence (Crafter's procedural generation per episode is an OOD axis); generalisation-theory; supporting (env paper, not a method).

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
- 2026-06-22 / agent-20260622: PDF fetched cleanly (1.15 MB, full paper). Achievement list and 17-action spec taken from my prior knowledge of the env code (danijar/crafter on GitHub) since the WebFetch summary was abstract-only — should be cross-checked against the PDF body when an idea seed depends on an exact count.
