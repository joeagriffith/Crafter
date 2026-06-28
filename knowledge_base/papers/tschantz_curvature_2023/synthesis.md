# Curvature-Sensitive Predictive Coding with Approximate Laplace Monte Carlo — synthesis

- **Slug**: tschantz_curvature_2023
- **Authors**: Zahid, Umais; Guo, Qinghai; Friston, Karl; Fountas, Zafeirios
- **Year / venue**: 2023 / arXiv preprint 2303.04976 (cs.LG)
- **Source PDF**: ./tschantz_curvature_2023.pdf
- **Citation**: Zahid, U., Guo, Q., Friston, K., & Fountas, Z. (2023). Curvature-Sensitive Predictive Coding with Approximate Laplace Monte Carlo. arXiv:2303.04976.
- **Synthesized by**: agent-20260622
- **Status**: initial

> **Slug correction flag**: Despite the slug, Tschantz is *not* an author — he is only cited (`hybrid_pc_2023`). The lead author is Umais Zahid, the same author as `zahid_critical_2023`. The slug should arguably be `zahid_almc_2023`; left as-is for stability, but downstream syntheses should be aware.

## TL;DR
Standard PC trains hierarchical Gaussian generative models by gradient
descent on the log-joint and ignores the Hessian (log-determinant)
term of the variational Laplace ELBO. The paper shows this is the
source of PC's poor log-likelihoods and washed-out samples: PC is a
Laplace-variational Bayes algorithm in which the variational posterior
covariance `Σ_q = -He(μ_z, θ)^-1` has been silently fixed to zero (Dirac
point mass). They add the missing term back via Monte Carlo — sampling
latents from the Laplace-optimal Gaussian `N(z_MAP, -H^-1)` at the end
of inference and using those samples in the learning gradient (LMC).
Because a full Hessian is `O(N^2)` and frequently non-PSD, they also
derive a *block-diagonal, PSD-by-construction* approximate Hessian
(ALMC) using only within-layer second-order relations. ALMC matches or
beats PC bits-per-dimension on MNIST/CIFAR10/CelebA/SVHN and produces
strictly more diverse samples; curvature gets implicitly regularised
even though they never optimise the log-det term directly. We care
because this is the cleanest formulation of *what the iterative
posterior actually is* in PC, plus a concrete recipe for going beyond
MAP.

## Problem framing
The paper sits inside the variational-Laplace reading of PC
(`friston_fep_2009`, `whittington_bogacz_2017`, `bogacz_2017`,
`millidge_backprop_2020`) and explicitly targets the gap diagnosed by
`zahid_critical_2023`: PC trained with the standard recipe produces
poor generative models because it optimises the wrong objective.
Under a quadratic (Laplace) approximation of the log-joint, the ELBO is

  F ≈ log P(x, μ_z | θ) + ½ log[(2π)^N det He(θ)^-1]

The PC literature drops the second term, equivalent to assuming a
Dirac-delta variational posterior, which collapses uncertainty over
latents. Recovering it is non-trivial because the Hessian is large,
expensive, and frequently non-PSD on real models.

## Method
The paper makes three moves.

**(1) LMC objective.** Rather than optimising the log-det term
directly, decompose the ELBO into expected-energy + entropy. Under the
Laplace approximation, the optimal posterior is `Q(z; μ_MAP, -He^-1)`
analytically, and its entropy is parameter-free. So sample
`z_i ∼ N(μ_MAP, -He^-1)` and learn θ via standard autodiff against the
MC average `LMC(θ) = (1/K) Σ log P(x, z_i | θ)` — no reparameterisation
trick required because the posterior parameters are deterministic
functions of `μ_MAP` and θ. This implicitly regularises curvature: they
empirically verify `log det He` stays bounded under LMC training while
PC blows it up.

**(2) PSD block-diagonal Hessian (ALMC).** Restrict to within-layer
second-order relations. For piecewise-linear `f` (leaky ReLU), second-
order cross-terms vanish, giving a per-block Jacobian-precision-Jacobian
form (eq. 7):

  -J(∇z(j) log P) = Σ(j) + Σ_k (∂f_k/∂z(j))^T Σ(k) (∂f_k/∂z(j)) + (obs term)

This is structurally a generalised Gauss-Newton block, PSD by
construction (gram matrix). Memory drops from `O(N^2)` to
`O(max(n_L^2, n_F·n_O))`.

**(3) Combination model.** Train the lowest layer with ordinary PC
(Dirac posterior) and only the upper layers with ALMC. Further cuts
memory; surprisingly *improves* BPD over full LMC, attributed to the
PSD guarantee (full Hessian was non-PSD ~90% of the time early in
training, forcing identity fallback).

Inference is unchanged from PC: MAP gradient descent on `log P(x, z|θ)`.
The curvature only enters at the *learning* gradient via Monte Carlo
samples from the Laplace posterior centred at the MAP.

## Key results
Bits-per-dimension on 5-layer (40/10→64→64→64→64) hierarchical models,
5 seeds (Table 1; lower is better):

| Dataset | PC | LMC | ALMC |
|---|---|---|---|
| MNIST  | 6.785 | 6.731 | **6.727** |
| CIFAR10 | 7.007 | 6.935 | **6.900** |
| CelebA  | 6.895 | 6.896 | 6.895 (tie) |
| SVHN   | 5.533 | 5.505 | **5.493** |

In the fixed-variance, non-combined setting the BPD gap widens
(MNIST 9.690 → 9.374). Sample diversity: CelebA samples under PC are
near-uniform "average faces"; LMC/ALMC samples preserve hair colour,
gender, lighting variability (Fig 3). Top-down interpolations through
the ALMC hierarchy show meaningful disentanglement by layer (global
features at the top, local at the bottom — Fig 4). Sharpness diagnostic
(Fig 2): PC's `log det He` rockets across training; LMC/ALMC stay
bounded → "Occam's razor" effect on posterior confidence.

## Limitations
- **Inference is still MAP.** Curvature enters the *learning* gradient
  but the inference loop itself remains first-order gradient descent on
  the log-joint. This is in tension with the paper's title ("curvature-
  sensitive predictive coding") — strictly only learning is curvature-
  sensitive. A Hessian-preconditioned (Newton/natural-gradient)
  inference step is a natural extension they don't take.
- **PSD violations even with the approximation.** Identity fallback
  was needed for non-PSD blocks during early training; they did not
  characterise when/where this happens.
- **Block-diagonal is a strong assumption.** Cross-layer covariance is
  thrown away, which is exactly the thing that makes hierarchical
  representations interesting.
- **Small models, no OOD eval.** 5 layers × 64 dims; the evaluation is
  log-likelihood and sample quality on in-distribution test sets. No
  corruption robustness, no cross-dataset NLL, no inference-step
  ablation. This is the gap our workspace can fill.
- **Compute scaling unstated.** Functorch-based Hessians are convenient
  but expensive; no wall-clock numbers reported.

## Idea seeds for our loop

- ****Curvature-preconditioned inference step on Poisson latents.**
  open_question #5 already flags `tschantz_curvature × pvae`. The
  geometry of the Poisson posterior (skewed, support `≥ 0`, variance =
  mean) makes Gaussian-Laplace approx a poor fit at small rates; an
  alternative is to keep the ALMC *learning* contribution but use a
  Fisher-information preconditioner for the rate parameter at
  inference. One trial, single non-trivial change. Theme:
  energy-based + generalisation-theory.

- ****Diagonal-Fisher curvature for cheap second-order inference.**
  Drop the full block-Jacobian and approximate with diag-Fisher
  (squared gradient running estimate, Adam-style). Becomes a per-latent
  adaptive step size — cheap, drop-in, compatible with our 60-min
  budget. Tests whether the *direction* of curvature use matters as
  much as the magnitude. Theme: energy-based.

- **Convergence-diagnostic via posterior sharpness.** Our guardrail
  measures `final_energy_slope`; their `log det He` is a strictly more
  informative scalar — it tells us whether inference *also* sharpened
  the posterior. Log it under `convergence` extra; flag trials whose
  energy converges but whose curvature diverges as "PC pathology"
  (likely the trials that look fine in-distribution but break on
  F-MNIST-C).

- **LMC sampling as a built-in OOD detector.** At eval time, draw K
  samples from `N(z_MAP, -H^-1)` for each test input and compute
  acceptance-rate-style statistics (e.g. spread of predicted logits
  across samples). Corrupted inputs should produce flatter posteriors
  (larger `-H^-1` eigenvalues) → higher logit variance. Same machinery
  doubles as a confidence score on F-MNIST-C. Theme: generalisation-
  theory.

- **Combination trick generalises.** "Lower layers Dirac, upper layers
  curvature-aware" is a powerful template — it says *the level at which
  uncertainty matters is the abstract representation layer*. Test the
  inverse for OOD: lower-layer curvature (where corruption noise
  enters), Dirac top. The asymmetry is testable.

- **Block-diagonal vs hierarchical-cross.** They throw away cross-layer
  curvature; under iterative inference, cross-layer uncertainty
  coupling may be exactly what propagates evidence up the hierarchy.
  A scratchpad question rather than a trial.

- **Lessons-style entry: never train PC with `log det He` unbounded.**
  Fig 2 is a clean experimental confirmation that PC's training-time
  numerical-stability lore (small inference learning rates, adaptive
  step sizes, layernorm-as-precision-rescaler) is fighting the same
  symptom. A trial-agnostic note for `lessons.md` once the loop is
  running.

## Connections
- `[whittington_bogacz_2017]` — the first-order PC this work corrects;
  the variational-Laplace reading inherited from Bogacz 2017 tutorial.
- `[zahid_critical_2023]` — same lead author. The "PC is bounded below
  by backprop" critique and this paper's "PC has the wrong objective"
  critique are two facets of the same Zahid programme.
- `[friston_fep_2009]` — Laplace approximation is FEP-canonical;
  Friston is a co-author here.
- `[mcpc_oliviers_2024]` — different Monte Carlo strategy (Langevin
  sampling on inference) where ALMC samples from a Gaussian fit at the
  MAP. Both are MC corrections to PC; they're complementary rather than
  competitive (Langevin during inference + LMC at learning).
- `[pvae_vafaii_2024]` — Poisson posterior is non-Gaussian; whether
  Laplace approx breaks gracefully or catastrophically is exactly
  open_question #5.
- `[sdpc_boutin_2021]` — sparse PC uses ISTA (first-order proximal);
  FISTA / second-order proximal lifts the same idea by adding a
  curvature term. Curvature × sparsity is unexplored.
- `[salvatori_associative_2021]` — large-scale PC that would presumably
  benefit from ALMC's PSD block-Hessian if memory is the bottleneck.
- `[multistep_pc_simplicity_2025]` — simplicity bias from inference
  depth; the "sharpness regularisation" / Occam framing here is a
  direct mechanistic candidate for *why* iterative PC simplifies.
- `[hybrid_pc_2023]` — explicitly cited; ALMC's "combination model"
  (lower-layer fast PC, upper-layer slow curvature-aware) is structurally
  the Hybrid PC system-1 / system-2 split applied to uncertainty rather
  than to amortisation.
- `[jpc_innocenti_2024]`, `[pinchetti_benchmark_2024]` — benchmarking
  / engineering layer; ALMC sits in the family of "what should PC's
  objective actually be" reforms they should be tested against.

## Relevant themes
energy-based; generalisation-theory.

## Tier
high — this is one of the cleanest available recipes for "iterative
variational PC in the non-equivalence regime" (the workspace's bet),
and it directly supplies one of the workspace's explicitly-flagged
combinations (open_question #5). Sub-agents working on alternative
inference rules, posterior geometry, or sample-quality / OOD diagnostics
should read.

## Out of scope
— (No spiking content; the curvature machinery is pure perception-side
inference, no active-inference policy machinery.)

## Notes / follow-ups
- 2026-06-22 agent-20260622: Slug is misleading — author is Zahid
  (same as `zahid_critical_2023`), not Tschantz. Worth a renaming
  issue or at minimum a `connections.md`-level cross-reference so the
  Zahid programme reads as one arc rather than two disconnected slugs.
- 2026-06-22 agent-20260622: ALMC's combination-model trick (mix Dirac
  + Laplace posteriors per layer) is a generalisable template; could
  be the spine of a "heterogeneous posterior PC" hypothesis where
  different layers get different posterior families (Dirac / Gaussian /
  Poisson / Laplace).
