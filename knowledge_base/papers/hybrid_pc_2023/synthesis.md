# Hybrid Predictive Coding: Inferring, Fast and Slow — synthesis

- **Slug**: hybrid_pc_2023
- **Authors**: Tscshantz, Alexander; Millidge, Beren; Seth, Anil K.; Buckley, Christopher L.
- **Year / venue**: 2023 / PLOS Computational Biology 19(8):e1011280
- **Source PDF**: ./hybrid_pc_2023.pdf
- **Citation**: Tscshantz A, Millidge B, Seth AK, Buckley CL. Hybrid predictive coding: Inferring, fast and slow. PLOS Comp Biol 19(8):e1011280, 2023.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
HPC augments a standard hierarchical PCN with bottom-up *amortised* connections that learn a feedforward "best guess" of the iterative-inference equilibrium for each layer. At test time, the amortised pass initialises latents, then top-down iterative inference refines them by minimising free energy. The split is principled: both phases optimise the *same* variational free-energy objective and use local Hebbian updates. Empirically HPC matches PC accuracy/generation, needs orders-of-magnitude fewer inference iterations once trained, learns with as few as 100 MNIST examples (where pure amortised fails), and *adaptively spends more iterations on novel data* (Fig 7C — they explicitly test a 0-4 → 5-9 distribution shift). This is the single most on-thesis published precedent for our workspace's bet.

## Problem framing
Pure PC is iterative and slow; pure amortised inference (VAE-style) is fast but suffers the amortisation gap and ignores context. HPC unifies both within one variational objective. Built directly on the PC tradition (Rao & Ballard 1999; Bogacz 2017; Friston 2003/2005) and on amortised variational inference (Kingma & Welling 2013). Closely related to semi-amortised VAEs and iterative amortised inference (Marino et al), but uniquely cast in PC's local Hebbian frame. Note: SPE (symmetric predictive estimator) is flagged as nearest competitor; HPC differs by temporal asymmetry between amortised and iterative phases.

## Method
Architecture: L hierarchical layers, each with state μ_i and error ε_i. Two parameter sets per layer: top-down generative f_θ_i and bottom-up amortised f_φ_i. Single variational free energy

F(μ,x) = Σ ε_l²/(2σ_l²) + ½ ln(σ_p σ_l), with ε_l = μ_{l-1} − f_θ(μ_l)

Inference proceeds in two phases:
1. **Amortised sweep**: μ_0 = f_φ_0(x); μ_{i+1} = f_φ_i(μ_i). Layer-wise feedforward.
2. **Iterative phase**: for N steps, μ̇_i = −κ(ε_p − ∂f_θ(μ_i)/∂μ_i · ε_l) (standard PC dynamics).

Learning: generative weights θ updated on converged μ* (Eq 9). Amortised weights φ updated via a separate amortised prediction error ε^φ_i = μ*_i − f_φ_i(μ_{i-1}), pushing the feedforward function to predict the *equilibrium* of iterative inference — i.e., "learning to infer through inference." Adaptive computation: an energy threshold β stops inference when free energy drops below it; well-learned/stationary stimuli need few iterations, novel stimuli need many.

## Key results
On MNIST (4 layers, 784-500-500-10, tanh; supervised by clamping top to one-hot label):
- **Match in accuracy and generation**: HPC ≈ standard PC at convergence (Fig 3).
- **Fast inference**: HPC trained with N=10 iterations matches PC trained with N=100. PC at N=10 *fails to learn at all* (Fig 5A).
- **Data efficiency**: with 100 examples, HPC reaches ~60% accuracy, amortised-only reaches ~17% (Fig 6A). Iterative top-down + generative model gives strong few-shot capability.
- **Adaptive iterations**: number of iterations drops toward 0 as the network learns the data (Fig 7B). On a 0-4 → 5-9 distribution shift, the network *spontaneously* increases its iteration count again (Fig 7C). This is direct evidence that iterative inference's resource cost is automatically allocated to novelty.
- **Entropy of label distribution monotonically decreases through inference steps** (Fig 7A) — iterative inference sharpens beliefs from an initially-uncertain amortised guess.

## Limitations
- MNIST-only. No CIFAR, no corruptions, no proper OOD benchmark. The 0-4 → 5-9 split is a sanity test, not an OOD evaluation.
- Both pathways use the same nonlinearity stack (tanh). No exploration of mismatched amortised vs generative function classes.
- "Adaptive computation" is gated by an MSE threshold, not by a calibrated uncertainty estimate.
- Asymptotic MNIST accuracy is lower than usually reported because the model is fundamentally generative.
- The amortised path can in principle be made deeper / more powerful than the generative path; this trade-off is unexplored.

## Idea seeds for our loop

- **Direct port to F-MNIST-C (our primary axis).** ** HPC's adaptive-iteration mechanism is exactly what should help on corrupted data: more iteration on novel/corrupted samples. The 0-4 → 5-9 experiment is one stop short of MNIST-C. Trial idea: implement HPC at the H000 capacity floor, evaluate iteration count per corruption × severity cell, and check whether iteration count correlates with corruption severity. Tag: iteration-for-OOD-evidence.

- **Amortised initialiser + iterative refinement = explicit speed/quality knob.** Hold parameter count fixed; train an HPC; at eval time sweep N iterations from 0 to 200 and measure F-MNIST-C accuracy as a function of N. Direct test of the central thesis: does *more iteration buy more OOD generalisation*, and where does the curve saturate? The 10/25/50/100 ablation in Fig 5 is the analogous in-distribution sweep.

- **Iteration as uncertainty signal.** ** They use MSE energy as the stopping criterion. We could use the number of iterations a sample needed as a *secondary OOD signal* — high-iteration samples ought to be harder. This becomes its own measurable axis: correlate per-sample iteration count with per-sample corruption severity / cross-dataset OOD-ness. Useful even when the model doesn't improve the primary number.

- **Mismatched amortised / generative capacities.** The published HPC uses symmetric architectures. Bias the amortised path to be small/fast (forced amortisation gap), let iterative inference do most of the lifting. The bet: harder iterative load → cleaner test of the iteration lever. Tag: iteration-for-OOD-evidence.

- **Hybrid with a sparse generative path.** Combine HPC's amortised initialiser with a Laplace/top-k prior on the generative latents (SDPC-style). Two operator-flagged levers in one trial: hybrid inference *and* sparsity. The amortised initialiser may stabilise sparse PC's notoriously fragile early training. Tag: sparsity, iteration-for-OOD-evidence.

## Connections
- `[salvatori_associative_2021]` — same Salvatori-line PCN inference dynamics; HPC's iterative phase is mechanically what associative-memory PCNs do at retrieval.
- `[ngc_ororbia_2022]` — alternative hierarchical-generative PCN; NGC tests cross-dataset transfer (MNIST→KMNIST) that HPC has not. Pairing HPC with NGC's eval protocol is a natural extension.
- `[pvae_vafaii_2024]` — could swap HPC's Gaussian latents for Poisson; the amortised initialiser becomes especially load-bearing because Poisson MAP is non-trivial.
- `[mcpc_oliviers_2024]` — replace HPC's deterministic iterative phase with Langevin; amortised → MAP init, Langevin → posterior sampling.
- `[brain_like_vi_2024]` — parallel evidence: less amortised → better OOD. HPC gives us a knob (N iterations) to traverse the same axis.
- `[multistep_pc_simplicity_2025]` — theoretical link between iteration depth and representation simplicity; HPC offers an empirical test bed.
- `[hinton_forward_forward_2022]` — both replace BP with local-error schemes; FF's contrastive idea could be cross-pollinated into HPC's amortised path.
- `[whittington_bogacz_2017]` — training-side foundation HPC inherits.
- `[friston_fep_2009]` — variational free-energy framing HPC explicitly invokes.

## Relevant themes
iteration-for-OOD-evidence (primary), active-inference (perception side), energy-based, biological

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
- The PLOS code is at github.com/alec-tschantz/pybrid. Pull-then-improve: the trial-000 sub-agent should mirror their architecture exactly at H000 capacity, then experiment.
- Adam optimiser is used for both pathways; authors note this is biologically implausible but "could be treated as a black box." Sub-agents should not get distracted by this.
