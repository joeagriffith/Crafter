# Predictive coding under the free-energy principle — synthesis

- **Slug**: friston_fep_2009
- **Authors**: Friston, Karl; Kiebel, Stefan
- **Year / venue**: 2009 / Philosophical Transactions of the Royal Society B
- **Source PDF**: ./friston_fep_2009.pdf
- **Citation**: Friston K, Kiebel S. Predictive coding under the free-energy principle. Phil Trans R Soc B 2009; 364:1211-1221. doi:10.1098/rstb.2008.0300
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
Friston & Kiebel cast perception as Bayesian inversion of hierarchical dynamical generative models. Under a Laplace approximation, inverting these models reduces to gradient descent on internal energy (variational free energy), and that descent can be implemented by a neuronally plausible message-passing scheme: prediction-error units and state (expectation) units exchanging precision-weighted residuals across cortical levels. The paper demonstrates the scheme on a synthetic birdsong model with two coupled Lorenz attractors, recovering both nuisance dynamics and slowly varying hidden causes, and reproduces empirical ERP-like signatures (mismatch negativity, omission responses, repetition suppression). This is the canonical statement of *why* iterative inference on a free-energy functional is meaningful rather than incidental.

## Problem framing
The paper sits inside the Helmholtz / "perception-as-inference" tradition (Helmholtz 1860; Dayan, Hinton, Neal & Zemel 1995, *the Helmholtz machine*; Rao & Ballard 1999) and the variational-Bayes tradition (Beal 2003; Friston 2008). It generalises Rao & Ballard's static hierarchical predictive coding to *dynamical* generative models written in generalised coordinates of motion, then asks: what neural circuit could invert such a model? The bridge work is Friston (2008) "Hierarchical models in the brain"; the present paper specialises that machinery to the Laplace approximation so that inference becomes a single gradient ascent on internal energy U(t) = ln p(ỹ, μ).

## Method
A generative model in generalised coordinates is given by
- y = g(x, v, θ) + z   (observations from causes v and hidden states x with observation noise z)
- ẋ = f(x, v, θ) + w   (state dynamics with process noise w)

Stacking levels yields an empirical-Bayes hierarchy where the output of level i serves as the input of level i+1. Under the Laplace approximation, the posterior q(ϑ) = N(μ, C) collapses onto its mean μ, and inference becomes
- μ̇ − Dμ̃ = U(t)_u  with U the internal energy.

Once expanded, this becomes the predictive-coding message passing rule (their Eq. 3.3): expectations μ are driven by prediction errors from the same level (bottom-up) and the level below (lateral), while error units ξ = Π(ε) carry the precision-weighted difference between the expectation and the top-down prediction. The mapping onto cortical anatomy: superficial pyramidal cells → error units (forward connections), deep pyramidal cells → state/expectation units (backward connections), with backward connections nonlinear/modulatory (consistent with the empirical asymmetry of forward vs feedback connections).

The free-energy F upper-bounds the negative log evidence, so descending the energy is descending an upper bound on surprise. Crucially this is *not* equivalent to backprop — the recognition density is a variational posterior, not a point estimate of a discriminative gradient, and the precision matrices Π are themselves learnable, giving the framework uncertainty-aware updates.

## Key results
- **Synthetic birdsong model**: two coupled Lorenz attractors drive a syrinx; the inversion scheme recovers the hidden state trajectory after ~600 ms of iteration and stably segments perceptual categories (song A/B/C) into well-separated clusters in cause-space (Fig 6).
- **Mismatch negativity / omission responses**: the precision-weighted error trace shows a vigorous transient at stimulus omission *despite no input being present* (Fig 5) — the same qualitative ERP signature observed in human MMN studies.
- **Conditional categorisation accuracy ~100 %** on the three-song toy at the second level, with shrinking 90 % posterior credible regions.

No quantitative comparison to other algorithms; the paper is a constructive demonstration, not a benchmark.

## Limitations
- Linear Gaussian / Laplace assumptions: the analysis assumes Gaussian innovations and a fixed-form Laplace posterior. The variational story degrades when posteriors are multimodal or heavy-tailed.
- All experiments are 1-D toy attractors. No image or classification benchmark; no quantitative OOD evaluation.
- Precision Π is parameter-free in the demos; the harder question of *learning* precisions (which is what makes the framework uncertainty-aware in any nontrivial sense) is deferred to "Friston 2008".
- The synthetic neurobiology is plausibility-grade, not predictive: the laminar-cell-population assignment is consistent with anatomy but not derived from it.

## Idea seeds for our loop
- **The free-energy interpretation is the lever, not the message-passing form.** Whatever architecture we build, the iterative inference should be derivable as gradient descent on a tractable F. This synthesises the central-thesis claim: "iteration alone is insufficient — it must be optimisation of *something*." Tag: **iteration-for-OOD-evidence**, **energy-based**.
- ****Precision (uncertainty) as a learnable per-level gain.** The Π matrices in Eq. 3.3 weight prediction errors and can themselves be learned. In a non-equivalence regime this is exactly the lever that distinguishes PC from backprop — under MSE-with-identity-precision they collapse. Sweep precision learning rules + initial precision schedule as a trial axis; expect OOD gains where precision adaptation matches the corruption channel (e.g. heteroscedastic noise vs structured occlusion). Tag: **iteration-for-OOD-evidence**, **generalisation-theory**.
- ****Generalised-coordinates inference for dynamic corruptions.** The paper's central technical claim is that posteriors over *trajectories* (not points) are tractable when you carry derivatives. Even in our static F-MNIST-C setting, treating an inference loop as descent on energy of *(state, motion)* could be a way to absorb temporally-flavoured corruptions (motion-blur, glass-blur, dotted-line) as inference over short trajectories rather than single states. Tag: **biological**, **iteration-for-OOD-evidence**.
- **Top-down predictions as modulatory rather than driving.** Bastos-style empirical anatomy says feedback connections are nonlinear/modulatory. This biases our architectural choice: a context-modulated decoder (gain modulation, FiLM) rather than additive concatenation matches the biology and may generalise differently. Tag: **biological**, **competition**.

## Connections
- `[bastos_microcircuits_2012]` — the cortical-implementation companion to this paper; same message-passing equations mapped onto laminar populations.
- `[whittington_bogacz_2017]` — operationalises the Laplace-Gaussian PC of this paper as a trainable network with local Hebbian plasticity.
- `[millidge_backprop_2020]` — extends the variational-Bayes derivation here to arbitrary computation graphs; relies on the same factorised Gaussian posterior.
- `[zahid_critical_2023]` — argues that the modified PC variants (FPA-PC, Z-IL) abandon the variational interpretation framed in this paper; counter-evidence for any hypothesis that builds on equivalence-regime PC.
- `[friston_active_inference_2017]`, `[parr_gfe_2019]`, `[sajid_active_inference_demystified_2021]` — extend the same F machinery to action; the perception side is the part directly relevant to our loop.
- `[brain_like_vi_2024]`, `[hybrid_pc_2023]` — modern empirical extensions of Friston-style iterative inference, with explicit OOD evaluation.

## Relevant themes
biological, energy-based, iteration-for-OOD-evidence

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
The paper's "what if we omit the last chirps" simulation (Fig 5) is the cleanest empirical foothold for an MNIST-C-style trial: the system predicts a percept *despite no input*, suggesting a way to evaluate our iterative networks on partial-occlusion corruptions in particular. Worth checking whether F-MNIST-C's `stripe` or `spatter` severities give a similar handle.
