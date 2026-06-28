# Monte Carlo Predictive Coding — synthesis

- **Slug**: mcpc_oliviers_2024
- **Authors**: Oliviers, Gaspard; Bogacz, Rafal; Meulemans, Alexander
- **Year / venue**: 2024 / PLOS Computational Biology 20(10): e1012532 (originally bioRxiv 2024.02.29.581455)
- **Source PDF**: ./mcpc_oliviers_2024.pdf
- **Citation**: Oliviers G, Bogacz R, Meulemans A. "Learning probability distributions of sensory inputs with Monte Carlo predictive coding." PLOS Comput Biol 20(10): e1012532 (2024). https://doi.org/10.1371/journal.pcbi.1012532
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
MCPC drops a single zero-mean noise term into the standard PC inference
dynamics, turning gradient descent on the free-energy into Langevin
sampling of the posterior `p(x|y;θ)` rather than MAP descent to its
mode. With weight updates evaluated on the sampled (not converged)
latents, the algorithm implements Monte-Carlo expectation-maximisation
and provably converges to a local optimum of the marginal likelihood
`p(y;θ)` — which Rao/Ballard-style PC famously *does not* (its weights
diverge under a Gaussian model). One extra line of code (`x ← x − η∇F
+ √(2η)·ξ`), fully local, beats PC on MNIST FID / log-likelihood /
masked-digit MSE and approaches a backprop-trained DLGM. The thesis-
relevant payoff: the inference dynamics now do real variational work
instead of collapsing to a Dirac mode.

## Problem framing
The paper sits at the intersection of two biological-Bayes traditions
the authors explicitly unify: (i) predictive coding as gradient
descent on prediction-error energies [`whittington_bogacz_2017`,
`friston_fep_2009`, Rao & Ballard 1999], which is local and plausible
but only ever produces MAP estimates; and (ii) neural sampling models
(Hoyer & Hyvärinen, Berkes et al, Hennequin et al) which give full
posteriors but lack local learning rules. Olshausen's variance argument
(also cited) frames the gap: PC's Dirac-delta variational approximation
ignores entropy, the marginal-likelihood bound becomes arbitrarily
loose, and weights blow up. MCPC fills both gaps with a single
modification.

## Method
**Generative model.** Hierarchical Gaussian, `p(y,x;θ) =
∏_{l=0}^{L−1} N(x_l; W_l f(x_{l+1}), σ²I) · N(x_L; μ, σ²I)`. Free
energy `F = −ln p(y,x;θ)` is the usual squared-prediction-error sum.

**Inference (the load-bearing change).** Replace PC's `∂x/∂t =
−∇_x F` with the **Langevin SDE**

  `∂x_l/∂t = −ε_l + f'(x_l) W_{l−1}^⊤ ε_{l−1} + n_l(t)`,

with white noise `E[n_i n_j^⊤] = 2σ_n² δ(t−t') I` (default σ_n²=1).
By the fluctuation-dissipation theorem, the steady-state distribution
is exactly `p^ss(x) = e^{−F}/Z = p(x|y;θ)`. Discretisation is
Euler-Maruyama: `x ← x − h ∂F/∂x + √(2h) n` with `n ~ N(0, σ_n² I)`.
Algorithm 1 prepends `K` MAP steps to shorten mixing, then runs `M`
mixing + `S` sampling steps.

**Learning.** Parameters integrate gradients over the sampled steady-
state: `ΔW_l ∝ ∫_{t_0}^{t_0+T} ε_l(t) f(x_{l+1}(t))^⊤ dt`. This is
Monte-Carlo E-M (Proposition 3): the E-step samples from the
posterior, the M-step ascends the expected joint log-likelihood,
converging to a local optimum of `ln p(y;θ)`. Computation and
plasticity remain strictly local; the only structural addition over
[`whittington_bogacz_2017`] is the noise injection in value-neuron
dynamics.

**Generation.** Unclamp `x_0` and run the same SDE: the marginal
`p(x_0; θ)` is the new steady state — one network does inference
*and* generation depending on whether `y` is clamped.

## Key results
- **Posterior fidelity (MNIST, half-masked digits).** KL divergence
  between an ideal-observer ResNet-9 class posterior and the linear-
  decoded latent class distribution: **MCPC ≈ 30, PC ≈ 60, random
  shuffled ≈ 140** (Fig 2e). MCPC visibly samples *multiple* digit
  interpretations on ambiguous inputs (Fig 2c-d); PC commits to one.
- **Generative quality (Table 1, MNIST, 3 seeds).** FID: PC 115.2,
  **MCPC 60.6**, DLGM 45.4. −ln p(y_eval): PC 168.9, **MCPC 144.6**,
  DLGM 126.0. Masked-digit reconstruction MSE (×10⁻²): **MCPC 8.29**,
  PC 8.73, DLGM 12.04. MCPC nearly halves the gap to a backprop-
  trained DLGM and beats both on inpainting.
- **Why PC fails.** Fig 4a-c: under PC, weight `W_0` diverges to ±∞
  (nullcline analysis); the learned variance is `W_0² + 1` and grows
  without bound. MCPC's nullcline intersects exactly at the data-
  matching parameters.
- **Captures cortical variability.** Reproduces Churchland et al's
  stimulus-onset variance quenching (Fig 6a) and Berkes et al's
  developmental spontaneous-evoked similarity *specific to trained
  stimuli* (Fig 6b). PC reproduces neither.
- **Robust to noise scale.** Within `σ_n² ≤ Σ_data/σ²`, MCPC recovers
  the true data variance across two decades of `σ_n²` (Fig 8).

## Limitations
- **Inference cost.** Mixing time scales badly with latent dimension —
  MNIST (~200-d) needs "significantly more inference steps" than the
  1-d linear toy (S2 Fig). The authors flag this as the dominant
  scaling concern and suggest momentum / preconditioning / Hennequin's
  E/I balanced recurrence as cures.
- **Learning floor on narrow data.** If `Σ_data < σ_n² σ²` MCPC cannot
  learn the data variance (weight collapses to zero). Authors propose
  learning precision matrices to fix; not done here.
- **Static inputs only.** "Zero-order variational filtering" — no
  temporal dynamics in `y`. Authors point to temporal PC for
  extension.
- **Noise mapping is hand-waved.** The biological source of the noise
  term is left unspecified (any zero-mean, time-uncorrelated, finite-
  variance noise satisfies fluctuation-dissipation).
- **No OOD benchmark.** They show generalisation on the training
  distribution and *learning* generalises across noise levels, but
  they don't test classification-under-corruption or any distribution
  shift — which is exactly the axis our spec scores on.

## Idea seeds for our loop

- ** **MCPC + Poisson latents.** Vafaii et al's metabolic-cost KL
  [`pvae_vafaii_2024`] gives a strongly non-Gaussian (and
  multimodal-friendly) posterior; MAP estimates of a Poisson posterior
  are particularly bad approximations. Swap MCPC's Gaussian generative
  model for Poisson and you get sampling under a sparse, biologically-
  motivated prior — covers two operator themes (energy-based +
  biological / sparsity) at once. Open question 5 already hinted at
  curvature × Poisson; sampling × Poisson is the cleaner first cut.
- ** **Posterior diversity as an OOD signal.** MCPC sample variance
  should *grow* on OOD inputs because the posterior is broader / more
  multimodal. Operationally: at eval, average classifier readout over
  `S` posterior samples and use the cross-sample entropy as a
  per-input uncertainty signal. Cheap to instrument; directly relevant
  to the corruption benchmark — gives a measurable OOD-vs-in-dist
  separation that an MAP-PC baseline cannot produce.
- **Noise schedule as a tunable knob.** `σ_n²` controls the trade
  between exploration and convergence sharpness; an annealed schedule
  (large → small over inference steps) is a one-line ablation that
  could compose with `multistep_pc_simplicity_2025`-style simplicity
  bias. Hypothesis: late-step low-noise sharpens in-distribution
  accuracy without losing the OOD smoothing of early-step high-noise.
- **MAP-warmstart + sampling tail.** Algorithm 1 already prepends `K`
  MAP steps before noise — a recipe transplantable to any PC variant.
  Worth ablating the split `(K, M, S)` against the F-MNIST-C primary;
  pure MAP and pure sampling are the corners, the middle may be the
  sweet spot.
- **Cross-sample averaging at the readout.** Train a linear classifier
  on `x_L` and report `mean_s p(class | x_L^{(s)})`. This is a Bayes-
  model-average over inference samples — a near-zero-cost way to turn
  Brain-like-VI's monotone iteration trend [`brain_like_vi_2024`] into
  a sample-ensemble trend.
- **Failure-mode warning to bake into our amortised baseline.** Their
  divergent-weight finding (Fig 4c) means any naive PC baseline with
  unregularised generative weights will silently blow up given enough
  training. H000 in our workspace is a Gaussian PCN — worth checking
  weight-norm growth as a sanity diagnostic.
- **MCPC × curvature.** [`tschantz_curvature_2023`]'s Hessian-scaled
  step composes naturally with MCPC: preconditioned Langevin is
  standard practice in MCMC. Direct port: `x ← x − η H⁻¹ ∇F + √(2η)
  L⁻¹ ξ` with `L L^⊤ = H`. Should fix MCPC's slow mixing on high-d
  latents (their #1 limitation) — and curvature-aware sampling is more
  faithful to the local posterior geometry.

## Connections
- [`whittington_bogacz_2017`] — Bogacz's own prior PC formulation; the
  exact "PC" that MCPC compares against and outperforms. The training
  mechanism MCPC inherits.
- [`friston_fep_2009`] — variational free-energy framing. MCPC is a
  literal MCMC version of FEP-inference.
- [`pvae_vafaii_2024`] — sibling "swap the inference machinery for
  something probabilistic" line. P-VAE swaps the *prior* (Gaussian →
  Poisson); MCPC swaps the *inference rule* (MAP → sampling). The two
  swaps are orthogonal; combining them is unclaimed territory.
- [`tschantz_curvature_2023`] — Hessian / Laplace-MC variant of PC.
  Preconditioned Langevin = curvature-aware sampling; the combination
  is open-question #5.
- [`brain_like_vi_2024`] — same "less amortised → better
  generalisation" thesis. MCPC is *least*-amortised (full posterior,
  no recognition net), so should sit at the favourable end of their
  axis if measured.
- [`hinton_cd_2002`] — contrastive-divergence is also Langevin on an
  energy; MCPC's E-step is structurally the same MCMC machinery, but
  on a hierarchical Gaussian rather than a Boltzmann/RBM.
- [`multistep_pc_simplicity_2025`] — provides theoretical support that
  more inference steps give simpler representations; MCPC requires
  many steps to mix, so its inductive bias should line up with their
  finding.
- [`salvatori_associative_2021`] — masked-input reconstruction /
  associative memory is one of MCPC's headline tasks (Fig 4e). Direct
  comparison if we touch associative memory.
- [`zahid_critical_2023`] — counter-evidence on PC↔backprop
  equivalence. MCPC is *not* a backprop-equivalence variant; its noise
  term breaks any deterministic-equivalence story and keeps the
  variational interpretation, which is precisely the regime our
  central thesis says matters. Discussion §3.4 notes Zahid's Langevin-
  PC, but flags it as "biologically implausible, singular latent,
  backprop-trained" — i.e. not the MCPC kind.
- [`hybrid_pc_2023`] — amortised-fast prior + iterative-slow
  refinement template. MCPC could slot in as the slow-refinement
  module; its MAP warmstart is already a hybrid-style scheme.

## Relevant themes
energy-based / biological / iteration-for-OOD-evidence (indirect —
posterior fidelity, not OOD acc) / generalisation-theory (MC-EM
convergence guarantee on `p(y;θ)`)

## Tier
high

## Out of scope
The paper itself is fully in scope. Two small carve-outs to flag:
- §3.6.3 floats a *spiking* MCPC variant as a way to constrain the
  noise source. Out of build-scope per spec.md (the Langevin / noise
  *idea* is in scope; a spiking substrate is not).
- §3.4 cites neural-sampling models tied to active-inference action
  selection; the inference-side parts only.

## Notes / follow-ups
- Author summary explicitly positions MCPC as a *unified* theory
  bridging predictive coding and neural sampling — useful framing if
  we ever write up an MCPC-based hypothesis.
- COI: R. Bogacz is a Fractile shareholder (declared); irrelevant to
  the algorithmic content but worth noting for citation hygiene.
- Algorithm 1 (page 21) is the canonical reference implementation;
  GitHub is gaspardol/MonteCarloPredictiveCoding. Pull this for the
  H001 baseline if a sub-agent runs MCPC.

—
