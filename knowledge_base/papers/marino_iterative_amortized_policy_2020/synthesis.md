# Iterative Amortized Policy Optimization — synthesis

- **Slug**: marino_iterative_amortized_policy_2020
- **Authors**: Marino, Joseph; Piché, Alexandre; Ialongo, Alessandro Davide; Yue, Yisong
- **Year / venue**: 2021 / NeurIPS 2021 (arXiv:2010.10670, submitted 2020-10-20)
- **Source PDF**: ./marino_iterative_amortized_policy_2020.pdf
- **Citation**: Marino, J., Piché, A., Ialongo, A. D., & Yue, Y. (2021). Iterative Amortized Policy Optimization. NeurIPS 2021. arXiv:2010.10670.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
Marino et al. recast SAC-style policy networks as *amortised variational optimisers* and show that the direct feedforward policy `f_phi(s) -> lambda` suffers an **amortisation gap** versus the in-class optimum. They replace it with an **iterative amortised optimiser** that reads current parameters and gradients of the soft-RL objective and emits gated updates over N inner steps per env transition. On MuJoCo this matches/beats SAC across 8 envs, recovers multi-modal action distributions from a uni-modal Gaussian, generalises to changed objectives without retraining, and is ~1 OOM more sample-efficient per inference step than Adam/CEM. For pc-crafter this is the canonical bridge between iterative-inference PC and RL: the gradient-encoder pattern PC uses for perception, applied to action selection.

## Problem framing
KL/entropy-regularised RL is variational inference over an optimality variable O, where a stochastic policy `pi_phi(a|s)` is the approximate posterior `pi(a|s, O)`. The policy network is therefore a *direct amortised inference network* — the encoder of a VAE — and inherits three pathologies: (1) an **amortisation gap** between `pi_phi` and the in-class optimum, (2) a single uni-modal estimate that blocks multi-modal exploration, and (3) inability to generalise across objectives because gradients arrive only *after* the forward pass. Marino's earlier "Iterative Amortized Inference" (arXiv:1807.09356) fixes (1)-(3) on the perception side by replacing the encoder with a gradient-reading optimiser. This paper transports that machinery to control. Prior art: Haarnoja SAC, Marino 2018, Levine 2018 (RL-as-inference), Fujimoto TD3.

## Method
The policy is a parametric Gaussian with parameters `lambda = [mu, sigma]`. Direct amortisation is `lambda <- f_phi(s)` (Eq. 10). The iterative variant is `lambda <- f_phi(s, lambda, grad_lambda J)` (Eq. 11), repeated N times per env step with a highway gate `omega_phi`: `lambda <- omega_phi ⊙ lambda + (1 - omega_phi) ⊙ delta_phi` (Eq. 12). The objective is the soft-RL bound `J = E_pi[Q_pi(s,a)] - alpha * KL(pi || p_theta)` (Eq. 8). Gradients `grad_lambda J` come from the pathwise estimator on a twin soft-Q network. The inner loop runs at both acting and training time, so the meta-network learns to behave like a gradient-coupled optimiser — exactly how PC's `mu` settles on free energy before weights update. Default N=5. A `beta`-weighted twin-Q lower confidence bound (Eq. 13) suppresses the value overestimation that iterative amortisation amplifies (more flexible policies more aggressively exploit Q-bias).

## Key results
- **Reduced amortisation gap** across Hopper, HalfCheetah, Walker2d, Ant at equal compute (Fig 6); training trajectories visit higher-value states.
- **Performance**: matches or beats SAC on 8 MuJoCo envs (Reacher, Hopper, HalfCheetah, Swimmer, Walker2d, Ant, Humanoid, HumanoidStandup; Fig 5) with lower seed variance.
- **Optimiser efficiency**: converges in ~5 steps where Adam needs ~100 and CEM more (Fig 4c).
- **Multimodality from uni-modal policy**: recovers two high-value modes on Ant-v2 from random Gaussian inits (Fig 2) — unavailable to direct even with normalising flows.
- **Objective generalisation**: trained with one Q and N=5, continues to improve with 5 *additional* eval iterations on a *different* objective (Fig 1, Fig 6).

## Limitations
- Inner loop multiplies wall-clock per env step by N.
- Value overestimation is *worse* than direct's; requires beta-LCB or careful Q-targets.
- All experiments MuJoCo continuous-control with Gaussian policies — no discrete actions, pixels, partial observability, or world models. Crafter has all four.
- N=5 inherited from Marino 2018; no depth-coupled iteration scaling study.
- Single-timescale optimiser; no exploration of layer-wise timescale separation or deep multi-point trajectories.

## Idea seeds for our loop
- **H-MARINO-1 — iterative amortised policy with PC perception, joint settling**: replace SAC's encoder *and* its policy head with two iterative-amortised optimisers that share a settling step. The perception side does PC inference on the world-model latent `z`, the policy side iterates on `lambda`, both at the *same* timescale on every env step. Test against amortised-only and PC-only ablations in Crafter. **(active-inference, iteration-for-OOD-evidence)**
- ****H-MARINO-2 — timescale-separated iteration depth across layers**: Marino uses one optimiser with N=5 outer iterations. We replace it with a stack where higher layers iterate fewer times than lower layers (or vice versa), aligning with hybrid_pc_2023's timescale-separation thesis. The hypothesis: deep multi-point dynamics let the action emerge from slow context conditioning slow perception conditioning fast sensorimotor loops. **(biological, active-inference)**
- **H-MARINO-3 — multi-modal exploration in Crafter's tech tree**: Crafter has explicit multi-modal optimal behaviours (mine-then-eat vs. plant-then-eat). Marino's mode-finding from a uni-modal Gaussian is the cheapest known mechanism to surface them without ensembling. Predict: iterative-amortised policy improves Crafter achievement-tree coverage over PPO/SAC at equal frames. **(counter-evidence target if it fails)**
- **H-MARINO-4 — train-time iteration as the load-bearing knob**: per Marino §4.2.4 and multistep_pc_simplicity_2025, the key knob may be *training-time* iteration count rather than test-time. Sweep N_train in {1,2,5,10} × N_test in {1,5,20} on Crafter; predict N_train monotone, N_test sub-linear — matching synthesis 02. **(generalisation-theory)**

## Connections
- `[hybrid_pc_2023]` — timescale-separated layer dynamics; Marino's single-timescale inner loop is the null to beat.
- `[multistep_pc_simplicity_2025]` — multi-step inference at training time as the generalisation lever; matches H-MARINO-4.
- `[jpc_innocenti_2024]` — JAX PC substrate for stacking an iterative-amortised policy head over PC perception.
- `[brain_like_vi_2024]` — iterative VI under biological constraints; same family, perception side.
- `[dacosta_active_inference_2020]` — EFE-based action selection; Marino's J is its entropy-regularised RL cousin.
- `[whittington_bogacz_2017]` — PC-backprop equivalence; relevant because training backprops through the inner loop.
- `[pvae_vafaii_2024]` — PC-as-VAE; policy-as-encoder reuses the same equivalence on the action side.

## Relevant themes
- iteration-for-OOD-evidence
- active-inference (action side — `dacosta_active_inference_2020`'s sibling)
- energy-based (soft Q + KL = free-energy-style functional)
- generalisation-theory (training-iter vs test-iter, see synthesis 02)

## Tier
**must-read** — this is the load-bearing reference for pc-crafter. Every hypothesis that touches the policy head should internalise §3.1 (iterative amortisation formulation) and Fig 1 (amortisation-gap visualisation). The operator has flagged it as the key paper for the project.

## Out of scope
Marino's experiments are all MuJoCo Gaussian continuous-control; Crafter is discrete-action, partially observed, pixel-input. Marino's hyperparameters (N=5, beta=2.5, twin-Q) are starting points, not transfers.

## Notes / follow-ups
- 2026-06-22 agent-20260622: operator-supplied arxiv 2009.01791 actually resolves to Hafner et al. "Action and Perception as Divergence Minimization"; the Marino paper is arxiv 2010.10670 — that is what's in this directory. Predecessor worth pulling for perception-side iterative amortisation: Marino, Yue & Mandt 2018, "Iterative Amortized Inference" (arXiv:1807.09356).
- Code: https://github.com/joelouismarino/variational_rl (PyTorch SAC backbone) — starting point for Crafter port.

—
