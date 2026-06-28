# Poisson Variational Autoencoder — synthesis

- **Slug**: pvae_vafaii_2024
- **Authors**: Vafaii, Hadi; Galor, Dekel; Yates, Jacob L.
- **Year / venue**: 2024 / NeurIPS 2024 (spotlight)
- **Source PDF**: ./pvae_vafaii_2024.pdf
- **Citation**: Vafaii, H., Galor, D., & Yates, J. L. (2024). Poisson Variational Autoencoder. *Advances in Neural Information Processing Systems 38 (NeurIPS 2024)*. arXiv:2405.14473.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR

P-VAE swaps the Gaussian latent of a standard VAE for a Poisson posterior over discrete count vectors `z ∈ ℤ_≥0^K`, with a *learned* rate prior `r` and a multiplicative encoder output `δr(x)` such that posterior rate `λ = r ⊙ δr(x)`. The derived KL is `r · f(δr)` with `f(y) = 1 − y + y log y` — a metabolic-cost term that explicitly penalises firing rate. With a linear decoder this exactly recovers an amortised, overcomplete sparse-coding objective; trained on natural-image patches it learns Gabor receptive fields rather than the PCA-like filters a Gaussian VAE produces. P-VAE avoids posterior collapse (~2% dead neurons vs ~80% for G/L-VAE), encodes inputs into higher-dimensional manifolds that are linearly separable, and is **5× more sample-efficient** on downstream MNIST KNN classification. Relevant to us because it pairs *sparsity that emerges from the prior* with a *non-Euclidean posterior geometry* — two of our highest-priority structural variations of PC.

## Problem framing

The authors argue VAEs are otherwise an excellent neuroconnectionist substrate (Bayesian, hierarchical, perception-as-inference, cortex-aligned representations) but their continuous Gaussian latents are misaligned with biological neurons that encode information via discrete, non-negative firing-rate-modulated spike counts. Prior work tried to inject sparsity into VAEs through Laplace / Cauchy priors (Geadah et al. 2024), spike-and-slab priors (Tonolini 2020) or post-hoc ISTA layers (Xiao 2024), but **none recovered the sparse-coding objective from the ELBO itself** — they bolt sparsity on. Predictive coding (Rao & Ballard, [`whittington_bogacz_2017`]) and sparse coding (Olshausen & Field) are treated here as *separate* theoretical lineages that P-VAE unifies under a single ELBO derivation. The Poisson assumption is motivated by Tolhurst-style empirical observations of mean ≈ variance in short cortical counting windows; the authors explicitly acknowledge real neurons are only *conditionally* Poisson.

## Method

**Latents.** `z ∈ ℤ_≥0^K`. Prior `p(z) = Pois(z; r)` with learnable rate vector `r ∈ ℝ_{>0}^K`. Approximate posterior `q(z|x) = Pois(z; r ⊙ δr(x))` where the encoder outputs a *multiplicative deviation* `δr(x)`, so the posterior rate is the prior rate scaled by an input-dependent residual. This *is* the predictive-coding assumption: feedforward carries deviation from the top-down expectation; feedback carries the expectation itself.

**The metabolic-cost KL.** A short manipulation of the Poisson PMFs gives a closed form (eq. 14, 18):

`D_KL[ q(z|x) ‖ p(z) ] = r · f(δr),   f(y) := 1 − y + y log y`

`f` is non-negative, zero at `y=1`, and ≈ ½(δr−1)² for small deviations. Because `r ≥ 0` and `f ≥ 0`, the KL functions as a **firing-rate penalty**: minimising it pushes either `r → 0` (dead prior neuron) or `δr → 1` (no residual to encode). This is precisely Olshausen-Field-style activity sparsity *derived* from variational inference, not added.

**Reparameterised Poisson sampling.** Algorithm 1 generates Poisson counts by drawing `n_exp` exponential inter-event times, accumulating to arrival times, and counting those below the unit window — using a sigmoid-relaxed indicator with temperature `T` to make the count differentiable. `T → 0` recovers true `torch.poisson`. They anneal `T: 1.0 → 0.05` over first half of training; at test time `T = 0`. Monte-Carlo gradients are nearly indistinguishable from the closed-form exact gradients (a Gaussian-quality reparameterisation, table 2).

**Architecture.** Encoder outputs `δr(x)`; with a linear decoder `x̂ = Φz` and overcomplete `K > M`, the full ELBO becomes (eq. 4, 24):

`L_SC-PVAE = ‖x − Φλ‖² + λᵀ diag(ΦᵀΦ) + β Σᵢ rᵢ f(δrᵢ)`

— literally the sparse-coding loss plus a basis-element-correlation correction term, with the activity penalty derived rather than chosen.

**Inference.** Amortised single-shot encoder. No iterative inner loop in the base model. The authors explicitly flag this as a limitation: "the amortization gap … could likely be closed … through iterative inference" (Marino et al.).

## Key results

- **Receptive fields.** Linear `P-VAE` on van Hateren patches: Gabor-like dictionaries comparable to LCA/ISTA (Fig. 4). Gaussian VAE: PCA-like / noisy. Laplace VAE: many dead basis elements (~80%).
- **Posterior collapse.** P-VAE: ~2% dead neurons. G-VAE / L-VAE: ~80%. Categorical VAE: ~0.4% (similarly avoids collapse).
- **Rate-distortion.** β-sweep: linear P-VAE achieves the same MSE as a conv G-VAE while being 1.7–2.4× sparser (Fig. 5a). At matched sparsity, P-VAE basis vectors run through LCA inference nearly close the gap to fully-trained LCA — points to "the encoder is the bottleneck, not the basis."
- **Downstream sample efficiency.** MNIST KNN, K=10 latents: P-VAE hits **82% with N=200 labelled samples**; G-VAE needs N=1000 for the same. ~5× sample efficiency (Table 4).
- **Shattering dimensionality.** Higher for P-VAE → discrete latents shatter MNIST into a higher-dimensional manifold that is linearly separable across all (10 choose 5) = 252 binary splits.

## Limitations

- **Marginal Poisson is approximate.** Authors acknowledge cortex over longer / hierarchical windows is super-Poisson; their proposal is that hierarchical / doubly-stochastic extensions would be conditionally Poisson but not marginally so.
- **Amortisation gap.** With the same dictionary, fully-iterative LCA still beats P-VAE at matched sparsity (Fig. 5b). The model leaves performance on the table by being amortised — a direct invitation to wire in iterative inference.
- **No OOD evaluation.** Despite the strong sample-efficiency result, **the paper never tests corruption robustness or distribution shift**. The "5× sample efficiency + higher-dimensional geometry" story is suggestive but unconfirmed for OOD.
- **No hierarchy.** Single-layer Poisson. The hierarchical version is future work.
- **Straight-through estimator underperforms** their relaxed sampler; surrogate-gradient methods (the SNN community's go-to) may be losing accuracy unnecessarily.

## Idea seeds for our loop

- **(★★ particularly strong) Poisson PC with iterative inference is the obvious load-bearing combination.** P-VAE keeps the encoder amortised and explicitly names this as the residual gap. The thesis says iterative variational inference is the OOD lever. Hypothesis: Poisson latents + iterative MAP/Langevin inference on `δr` (descending the same `r·f(δr) + recon` energy at test time) closes the amortisation gap *and* delivers OOD robustness on F-MNIST-C. The energy and its closed-form gradient w.r.t. `δr` are in the paper, so the inference loop is mechanical. [sparsity, biological, energy-based, iteration-for-OOD]
- **Sparsity-from-prior beats sparsity-as-regulariser.** The authors argue the activity penalty *emerges* from the ELBO under a Poisson prior, where Laplace VAEs needed an extra approximation step and collapsed anyway. Lesson for our loop: prefer priors whose KL is naturally sparsity-inducing over post-hoc L1 penalties. Wire this into the H000 Gaussian PCN as a drop-in prior swap rather than as an added regularisation term. [sparsity, energy-based]
- **Curvature-sensitive inference × Poisson posterior** (seeds open_questions #5). The Poisson posterior is decidedly non-Gaussian — Fisher information `1/λ` makes the natural-gradient geometry rate-dependent. [`tschantz_curvature_2023`]-style Hessian-scaled inference steps should compose cleanly because they were derived to respect arbitrary posterior curvatures. P-VAE's `f(y)` Taylor-expands to ½(δr−1)² near 1, so the local quadratic is `r·(δr−1)²`: curvature `r`, scale-dependent step size emerges naturally. Two-variable hypothesis with both knobs concrete. [biological, energy-based]
- **The shattering-dimensionality result predicts OOD wins.** Higher shattering = better linear separability across arbitrary class splits = better robustness to nuisance shifts of the input. P-VAE's K=50 representations win on shattering; this is the geometric mechanism by which OOD generalisation *could* follow. Operationalisation: track shattering dim per trial alongside primary F-MNIST-C accuracy; if the correlation holds, it gives us a cheap intermediate signal. [generalisation-theory]
- **Linear decoder closed-form is a sanity-check oracle.** The eq. 4 / eq. 24 closed-form `L_SC-PVAE` admits exact gradients without sampling. Useful as an exact-objective trial early in the loop to validate our Poisson reparameterisation matches theirs before scaling up. [supporting]
- **β as a sparsity dial we control.** Their β sweep produces a clean rate-distortion curve from dense to ~99% sparse on natural images. Our open question #1 (does sparsity regime matter for OOD) wants exactly this dial. Sweep β across {0.01, 0.1, 1.0, 4.0} on F-MNIST and read the OOD-accuracy/sparsity curve. [sparsity]
- **Surrogate gradients are leaving accuracy on the table.** Their "hard-forward" experiment underperformed the relaxed Poisson — useful lesson the spiking community usually doesn't internalise. Practical knob: keep `T > 0` during training, only set `T = 0` for the eval forward pass.

## Connections

- [`sdpc_boutin_2021`] — sibling sparsity-prior PC paper. P-VAE derives sparsity from a Poisson prior; SDPC chooses an explicit Laplace (L1) prior and uses ISTA inference. The clean head-to-head: do Poisson and Laplace sparsity buy the same OOD robustness, or are they categorically different?
- [`tschantz_curvature_2023`] — open_questions #5 already flags the combination. Poisson posterior is non-Gaussian; curvature-sensitive Hessian steps should compose. P-VAE makes this concrete by giving us the energy explicitly.
- [`mcpc_oliviers_2024`] — Langevin inference is well-defined on the Poisson NELBO if we treat `δr` (or `λ`) as the continuous variable being sampled. Adds a posterior-sampling story to P-VAE rather than amortised MAP-style encoding.
- [`brain_like_vi_2024`] — the OOD-via-iteration paper. P-VAE has the Poisson architecture but explicitly *no* iteration; brain-like VI has iteration but Gaussian latents. The intersection is unclaimed.
- [`whittington_bogacz_2017`] — Gaussian PCN that the H000 baseline implements. P-VAE is the Poisson analogue at the VAE level; the analogous swap at the PCN level is the natural H00x.
- [`numenta_sparsity_2019`] — hard-sparsity sibling. Numenta enforces top-k by construction; P-VAE elicits it via the metabolic-cost KL. Open question #1 wants both compared.
- [`hinton_forward_forward_2022`] — alternative EBM training. Not directly related but the closed-form linear-decoder ELBO is the kind of local-objective derivation FF aims for.
- [`hybrid_pc_2023`] — System 1 / System 2 framing fits P-VAE-as-encoder + iterative-refinement-as-inference perfectly.
- [`pinchetti_benchmark_2024`] — benchmark substrate; we'd want to drop Poisson-PC variants in there.

## Relevant themes

sparsity / biological / energy-based / iteration-for-OOD-evidence (by extension — P-VAE itself does not test OOD, but the geometric findings predict it)

## Tier

high

(Borderline must-read. Operator flagged P-VAE as on-thesis; the Poisson-prior swap is one of the cleanest single-variable variations of PC machinery available and gives us a closed-form energy and a reparameterised sampler off-the-shelf. Sub-agents in the probabilistic-variants or sparsity neighbourhoods should read this.)

## Out of scope

**P-VAE is NOT spiking; Poisson latents are continuous distributions over discrete counts, not spike trains. In scope per operator.** The model samples integer count vectors `z ∈ ℤ_≥0^K`, *not* temporally-resolved spike trains over a time grid. The "spike count" language and the neuroscience framing make this easy to misread, but the underlying object is a discrete distribution with a continuous (Poisson rate) parameterisation — exactly what spec.md §"Out-of-scope" carves *into* scope. Sub-agents reaching for spiking analogies should reread the operator's distinction in `01_central_thesis_iteration_for_OOD.md` §"Out of scope".

## Notes / follow-ups

- 2026-06-22 agent-20260622: Source code at github.com/hadivafaii/PoissonVAE per Fig. 1 caption — pull-then-improve baseline candidate per program.md §"Baselines: pull-first, then improve".
- 2026-06-22 agent-20260622: The closed-form linear-decoder objective (eq. 24) is what we want for a first trial — exact gradients, no sampling noise, lets us isolate the prior-swap effect from the reparameterisation effect.
- 2026-06-22 agent-20260622: Worth flagging to other sub-agents that f(y) = 1 − y + y log y is *not* the standard "soft-L1" any sparsity paper uses; it's specifically the Poisson KL derivative and only equals ½(δr−1)² near δr=1. Far from 1 the penalty grows linearly in δr·log δr — softer than L2, harder than L1. Empirically interesting.

—
