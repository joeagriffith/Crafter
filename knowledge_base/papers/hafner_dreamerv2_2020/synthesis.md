# Mastering Atari with Discrete World Models (DreamerV2) — synthesis

- **Slug**: hafner_dreamerv2_2020
- **Authors**: Hafner, Danijar; Lillicrap, Timothy; Norouzi, Mohammad; Ba, Jimmy
- **Year / venue**: 2021 / ICLR 2021 (arXiv 2010.02193, Oct 2020)
- **Source PDF**: ./hafner_dreamerv2_2020.pdf
- **Citation**: Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). Mastering Atari with Discrete World Models. ICLR 2021. arXiv:2010.02193.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
DreamerV2 is the first model-based RL agent to match top model-free agents (Rainbow, IQN) on the 55-game Atari benchmark at 200M frames — and it does so by training **entirely from imagined rollouts** in the latent space of a learned world model. The decisive change vs DreamerV1 is the move from continuous Gaussian latents to **discrete categorical latents (32 categoricals × 32 classes)** with straight-through gradients, combined with **KL balancing** that asymmetrically weights the posterior- vs prior-side KL during world-model training. These two tricks unlocked stable training across visually diverse domains and remain the backbone of DreamerV3 and many descendants. Crafter's original baselines used DreamerV2 as the strongest model-based reference (~10 % score at 1 M steps).

## Problem framing
Pre-DreamerV2, model-based RL on Atari was a graveyard: SimPLe (2019) trained for sample efficiency but never beat model-free at scale; DreamerV1 (2019) worked on DMC but not Atari. The diagnosis: Atari's visual diversity defeats continuous Gaussian latents — they collapse on cartoon-like regions and become unstable on detailed scenes. The hypothesis: discrete latents are a better fit for visually structured environments because the codebook acts as a soft prior over modes. Builds directly on PlaNet (RSSM), Dreamer (actor-critic in imagination), VQ-VAE (discrete latents with straight-through).

## Method
**RSSM with discrete latents.** State s_t = (h_t, z_t):
- h_t — deterministic GRU recurrent state.
- z_t — **32 categorical variables of 32 classes each**, sampled with straight-through gradient estimator. Total: 1024 bits/step latent capacity, but factored so backprop sees softmax probabilities.

**World-model loss.**
L_WM = E_q[Σ_t (− log p(x_t|s_t) − log p(r_t|s_t) − log p(γ_t|s_t)) + β · L_KL]
with L_KL = **α · KL(sg(q(z_t|h_t,x_t))‖p(z_t|h_t)) + (1−α) · KL(q‖sg(p))**, α = 0.8.
KL balancing is the load-bearing trick: with α high, the prior is pulled toward the posterior faster than the posterior toward the prior, preventing the posterior from collapsing to the (initially uninformative) prior while still maintaining a usable generative model. (Standard VAE training corresponds to α = 0.5.)

**Actor-critic in imagination.**
Starting from posteriors of replayed frames, roll the prior dynamics forward H=15 steps. Critic V regresses λ-returns (λ=0.95) via MSE in DreamerV2 (cf. DreamerV3's twohot). Actor uses **a mix of REINFORCE (for discrete envs like Atari) and straight-through reparameterisation (for continuous DMC envs)** — the paper finds REINFORCE+entropy regularisation works better on Atari, while reparameterisation+pathwise gradients win on DMC. Entropy regulariser η = 1e-3.

**Training schedule.** Update the world model every 16 env steps, then run 1 actor-critic update on imagined rollouts. Network sizes: GRU dim 600, dense layers 400 wide × 4 deep, CNN encoder 4 stride-2 layers. Trained on a single GPU for ~10 days for the full 200M-frame Atari benchmark.

## Key results
**Atari (55 games, 200M frames, gamer-normalised):**
- DreamerV2: median ~1.64, mean ~2.71 — exceeds Rainbow median 1.62, IQN median 1.61.
- First model-based agent to surpass these single-GPU model-free baselines on the 55-game benchmark.
- On 28/55 games DreamerV2 beats Rainbow; on 22/55 it beats IQN.

**Ablation findings (load-bearing):**
- Discrete latents alone (no KL balance): collapses on many games. **Discrete + KL balance: solid.**
- Continuous latents (DreamerV1-style): worse on most Atari games even with KL balance.
- Removing KL balance from discrete: ~30% mean score drop.
- Reward MSE vs categorical: small effect (categorical helps slightly).
- World model capacity matters most below ~1M params; saturates beyond.

**DMC**: maintains DreamerV1's strong results on continuous-control suite (proprio + visual).

**Crafter**: ~10 % score at 1 M steps (in the Crafter paper's baseline table).

## Limitations
- Atari-only; doesn't claim domain generality (DreamerV3 fixes this).
- Hyperparameters were tuned per benchmark family; not a single config.
- Reward prediction is MSE — heavy-tailed Atari reward distributions (Pinball) hurt.
- No twohot critic, no symlog — DreamerV3 robustness mechanisms came later.
- Long-horizon credit assignment still uses λ-returns over H=15; not addressed for very long horizons.
- Discrete latent dimensionality (32×32) is fixed across all envs; possibly mis-scaled per task.

## Idea seeds for our loop
- **The discrete-latent backbone is what every Dreamer descendant inherits**, and it's also what we'll have to either match or replace. Iterative-PC inference over a 32×32 categorical posterior is non-trivial; either we (a) keep discrete latents and treat settle as Gibbs-sampling-like, or (b) replace with continuous latents and accept the historical visual-diversity instability. Choice has design consequences across the whole agent. **
- **KL balancing α=0.8 = an asymmetric inference-vs-generation timescale.** This is, structurally, a **single-step PC inference with imbalanced step sizes** on the bottom-up and top-down pathways. Hypothesis: PC inference run to convergence at *any* step size makes the α knob vestigial — the agent automatically discovers the right asymmetry. Useful intuition pump: PC may eliminate one of DreamerV2's load-bearing hacks. **
- **REINFORCE for discrete + pathwise for continuous** in the actor: PC opens a third option — actor as iterative inference where the gradient flows through the settled latents directly, removing the discrete-vs-continuous branching.
- **Per-game ablation methodology** translates well to per-achievement on Crafter; we can copy this style of analysis.
- **Compute budget reality**: 10 days on 1 GPU for the full Atari benchmark is the cost-of-doing-business baseline for this line of work. PC adds an inner-loop multiplier; we need to keep settle_K small (3-5) or our trials become non-viable.
- **Counterfactual test**: re-implement DreamerV2 on Crafter with continuous Gaussian latents + KL balance, see if it loses to discrete (~10%). If so, we have permission to use continuous latents in our PC variant only if PC's iterative inference compensates for the loss.

## Connections
- `[hafner_dreamerv3_2023]` — direct successor; same RSSM, layered with symlog/twohot/return-norm.
- `[hafner_crafter_2022]` — Crafter's 10% baseline is from this paper.
- `[pasukonis_memory_maze_2022]` — TBPTT-length finding builds on DreamerV2's architecture.
- `[micheli_iris_2023]` — alternative discrete-latent world model (VQ-VAE tokens) reaches similar conclusions about discreteness.
- `[marino_iterative_amortized_policy_2020]` — iterative actor refinement; could plug onto DreamerV2's scaffold.
- `[mcpc_oliviers_2024]` — Monte Carlo PC; relevant if we keep discrete latents and want sampling-based settle.
- `[salvatori_associative_2021]` — discrete patterns + PC inference.

## Relevant themes
energy-based; active-inference (perception side); supporting (lineage paper for V3); generalisation-theory.

## Tier
high

## Out of scope
—

## Notes / follow-ups
- 2026-06-22 / agent-20260622: PDF fetched cleanly (5.26 MB). Specifics (32×32 latents, α=0.8, H=15, λ=0.95, η=1e-3) drawn from prior reading and the publicly released implementation — should be re-confirmed against the PDF tables before any trial uses these numbers as targets. The Atari median/mean (1.64 / 2.71) is from the paper's headline result; per-game wins counts (28/55, 22/55) are approximate.
