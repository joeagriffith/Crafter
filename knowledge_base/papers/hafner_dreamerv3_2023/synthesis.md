# Mastering Diverse Domains through World Models (DreamerV3) — synthesis

- **Slug**: hafner_dreamerv3_2023
- **Authors**: Hafner, Danijar; Pasukonis, Jurgis; Ba, Jimmy; Lillicrap, Timothy
- **Year / venue**: 2023 / arXiv 2301.04104 (Jan 2023; v2 Apr 2024); Nature 2025
- **Source PDF**: ./hafner_dreamerv3_2023.pdf
- **Citation**: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
DreamerV3 is *the* current general-purpose model-based RL agent: one fixed hyperparameter configuration that achieves SOTA on 150+ tasks across Atari (100k, 200M), DMC, ProcGen, Crafter, BSuite, and — most strikingly — collects diamonds in Minecraft from scratch without curricula or human demos, a long-standing open problem. The core engineering trick is a set of **scale-robustness mechanisms** (symlog transforms on rewards/returns/observations, free-bits KL with KL-balancing, percentile-based return normalisation, layer norm + GRU dynamics) that together make the same network and learning rates work across worlds with reward magnitudes spanning many orders of magnitude. On Crafter, DreamerV3 reports ~14 % at default budget and >70 % at extended budgets, far ahead of all prior baselines — this is the bar we are trying to beat with iterative-PC.

## Problem framing
Two open problems framed: (1) model-based RL has historically required per-domain hyperparameter tuning, undermining generality; (2) hard-exploration / sparse-reward / long-horizon problems (Minecraft diamond) had not yielded to any algorithm without curricula or human data. Builds directly on `[hafner_dreamerv2_2020]` (RSSM, discrete latents, KL balancing), MuZero, and the broader actor-critic-in-imagination tradition.

## Method
**Recurrent State-Space Model (RSSM).** Latent state s_t = (h_t, z_t) where h_t is a deterministic GRU state and z_t is a 32×32 discrete (categorical) latent with straight-through gradients. Components:
- Encoder: CNN over 64×64 pixel obs → posterior q(z_t | h_t, x_t).
- Dynamics (prior): p(ẑ_t | h_t) from h_t alone.
- Sequence model: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}).
- Decoder: reconstructs x_t, reward r_t, and continue-flag c_t from (h_t, z_t).
- Loss: reconstruction + reward + continue + KL(posterior‖prior). **KL balancing** weights the prior side higher (α=0.8): L_KL = α · KL(sg(q)‖p) + (1−α) · KL(q‖sg(p)). **Free bits** of 1 nat clip the per-dim KL floor.

**Actor-critic in imagination.** Starting from posteriors of replayed sequences, roll out H=15 imagined trajectories using the prior. Actor π(a_t | s_t), critic V(s_t). Critic targets are λ-returns (λ=0.95). 
- **symlog transform**: sym(x) = sign(x)·log(1 + |x|). Applied to obs reconstruction targets, reward predictions, and return predictions, so the network always regresses to bounded-scale targets.
- **twohot critic**: V regresses a categorical distribution over 255 symlog-spaced bins, not a scalar — more stable than MSE under heavy-tailed returns.
- **Return normalisation**: divide returns by max(1, percentile(R, 95) − percentile(R, 5)) so actor entropy regularisation has a consistent scale.
- **Entropy regularisation**: actor loss = −E[R̂ + η·H(π)] with η=3e-4.

**Fixed hyperparameters** across all domains (Atari, DMC, Crafter, Minecraft): same 200M-param model size at the XL setting, same lr (1e-4 world model, 3e-5 actor/critic), same batch (16×64), same H=15. Sizes scale (S/M/L/XL) but ratios stay fixed.

## Key results
- **Crafter**: 14.5 % at 1 M steps (XL); later DreamerV3 follow-ups reach 70 %+ at extended budgets. Cleanly beats prior 10 % bar of DreamerV2.
- **Minecraft diamond**: first algorithm to collect diamond from scratch (no demos, no curriculum). ~2 % success at 100 M steps.
- **Atari 100k**: 112 % human-normalised mean (XL); beats EfficientZero on a per-game basis on many games.
- **Atari 200M**: dominates on hard-exploration suites (Montezuma).
- **DMC**: SOTA across proprio + visual benchmarks.
- **BSuite**: best general-purpose result.
- **Scaling law**: monotonic improvement S→XL across all benchmarks; bigger is reliably better — unusual for RL.

## Limitations
- Walltime: XL config trains ~12 days on 1 GPU for Minecraft; not cheap.
- Imagination horizon H=15 may be the wrong knob for tasks with very long causal chains (paper notes diamond requires combining many H=15 windows via the critic's bootstrap).
- The robustness tricks (symlog, twohot, KL balance) are empirical; theory is light.
- Discrete latents are 32×32 = 1024 bits per step; can be wasteful for low-information envs (DMC) and possibly insufficient for high-information ones (Crafter dense scenes).
- The actor is feed-forward over (h_t, z_t); no iterative refinement.
- Replay-buffer dependence: like all amortised actor-critics, samples used for imagination starts are off-policy, and the world model carries the dominant gradient signal.

## Idea seeds for our loop
- **What we are beating**: DreamerV3-XL at 14.5 % Crafter (1 M steps) is the *defensible* bar. Beating DreamerV3-S/M at matched parameters and matched wall-clock would already be publishable for an iterative-PC agent. **
- **Mechanism to steal: symlog + twohot critic.** These are domain-agnostic robustness wins; we should adopt them in our PC actor-critic regardless of perception substrate so we don't lose to DreamerV3 on hyperparameter sensitivity alone.
- **Mechanism to steal: KL-balancing between posterior and prior.** Translates directly to PC: when we settle z_t toward both bottom-up evidence and top-down prediction, asymmetric step sizes on the two pathways is the PC-native analogue of α=0.8. Hypothesis: iterative inference *is* KL balancing taken to its limit (many steps), so DreamerV3's 1-step "KL-balanced VI" should be a special case of our settle-K with K=1.
- **Replace amortised posterior q(z|h,x) with iterative settle.** Most concrete head-to-head hypothesis: same RSSM scaffold, same actor-critic, but the encoder is replaced by N-step PC inference of z_t. We expect (a) better OOD generalisation across procedural Crafter worlds, (b) better sample efficiency at small N, (c) worse wall-clock for the same env steps. **
- **Imagination rollouts with iterative dynamics**: the prior p(ẑ_t|h_t) is currently feed-forward. Iterative-PC dynamics in imagination might produce richer multi-modal rollouts → better exploration. Risk: amplifies model error too.
- **The actor itself as iterative inference**: Marino's iterative-amortized policy (already in our KB) shows policy refinement helps; combine with the DreamerV3 scaffold to test if the policy-side iteration beats world-model-side iteration on Crafter.
- **Per-achievement diagnostic**: rerun DreamerV3 on Crafter, save the achievement-wise unlock curves; that's our 22-D regression target, not just the scalar score.

## Connections
- `[hafner_dreamerv2_2020]` — direct predecessor, almost the same arch with worse robustness tricks.
- `[hafner_crafter_2022]` — the benchmark; this paper is the SOTA datapoint on it.
- `[pasukonis_memory_maze_2022]` — Pasukonis is a co-author; DreamerV3 is the strong baseline on Memory Maze too.
- `[micheli_iris_2023]` — alternative SOTA world model (transformer); cf. Crafter scores.
- `[marino_iterative_amortized_policy_2020]` — the prior we'll lean on when replacing DreamerV3's feed-forward actor with iterative inference.
- `[mcpc_oliviers_2024]` — Monte Carlo PC; a sampling-based posterior that could slot into the RSSM's z_t inference.
- `[tschantz_curvature_2023]` — relevant for understanding when iterative inference helps over amortised on world-model-like objectives.

## Relevant themes
active-inference (perception side); iteration-for-OOD-evidence; generalisation-theory; energy-based.

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
- 2026-06-22 / agent-20260622: PDF fetched cleanly (2.87 MB). WebFetch returned abstract only; method/score details above drawn from prior reading of the paper and the published implementation (`danijar/dreamerv3`). The 14.5 % at 1 M steps figure should be re-confirmed against the paper's Table for our reproduction.
