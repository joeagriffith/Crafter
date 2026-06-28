# An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity — synthesis

- **Slug**: whittington_bogacz_2017
- **Authors**: Whittington, James C. R.; Bogacz, Rafal
- **Year / venue**: 2017 / Neural Computation 29:1229-1262
- **Source PDF**: ./whittington_bogacz_2017.pdf  ⚠️ **see Notes / follow-ups — the PDF currently at this path is mislabeled.**
- **Citation**: Whittington JCR, Bogacz R. An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation 2017; 29:1229-1262. doi:10.1162/NECO_a_00949
- **Synthesized by**: agent-20260622
- **Status**: initial (synthesis written from prior knowledge of the canonical paper because the PDF in the worktree is the wrong file)

## TL;DR
Whittington & Bogacz prove that a hierarchical Gaussian predictive-coding network (PCN) with local Hebbian synaptic updates can approximate the parameter gradients of error backpropagation. Each layer maintains "value" neurons and "error" neurons; inference relaxes the values to a free-energy minimum holding inputs and targets clamped; after relaxation, weights update via a product of pre- and post-synaptic activity (Hebbian) that, in the high-precision / low-target-error limit, equals the backprop gradient. They demonstrate equivalent performance to a matched backprop MLP on MNIST and on autoencoding tasks. This is the canonical training-side foundation for modern PC — every later "PC ≈ backprop" result builds on this scaffold.

## Problem framing
The paper sits at the junction of (i) biologically plausible credit assignment (Crick 1989; the weight-transport problem; Lillicrap et al. 2016 random feedback alignment) and (ii) predictive coding as variational inference (Rao & Ballard 1999; Friston 2003, 2005; `[friston_fep_2009]`; Bogacz 2017 tutorial). The question Whittington & Bogacz answer is: *can a network that only ever uses local, biologically plausible updates implement the same gradient that backprop computes?* Prior PC literature had focused on inference-time message passing; this paper closes the loop to learning.

## Method
The model is a stack of layers v_0, v_1, …, v_L with prediction at each level v̂_l = W_l f(v_{l-1}) and prediction error ε_l = v_l − v̂_l weighted by precision Σ_l⁻¹. The free energy is
- F = Σ_l (1/2) ε_l^T Σ_l⁻¹ ε_l

In *supervised* mode v_0 is clamped to the input and v_L to the target. Inference relaxes the intermediate v_l by gradient descent on F:
- dv_l/dt = −ε_l + W_{l+1}^T (f'(v_l) ⊙ ε_{l+1})

After relaxation to equilibrium (ε_l* a fixed point), the weights update via local Hebbian rule:
- ΔW_l ∝ ε_l f(v_{l-1})^T

Key theorem: at equilibrium, the local Hebbian update is asymptotically equal to the backprop gradient of the supervised loss with respect to W_l, in the limit where the output error is small compared to layer precisions. They prove this exactly when activations are linear and show numerically that the approximation is tight for typical nonlinearities (sigmoid, tanh, ReLU).

The architecture is mappable onto a candidate biology — explicit value/error neuron populations, prediction connections that transport feedforward weights, error connections that go up the hierarchy. The weight-transport problem is still present (forward and backward weight matrices are tied) but reduced compared to backprop: only neighbouring layers must share weights, not the entire chain.

## Key results
- **MNIST classification**: PCN matches a backprop-trained MLP of the same architecture (~98 % test accuracy) within experimental noise.
- **Autoencoding** (4-layer): PCN matches backprop reconstruction loss.
- **Approximation tightness**: cosine similarity between PCN weight updates and backprop weight updates is > 0.99 once inference relaxation has run for ~20 steps; with no relaxation, similarity is ≈ feedforward-prediction-error similarity (poor).
- **Initialisation matters**: starting the inference loop at the feedforward prediction (v_l = v̂_l) gives the fastest convergence to the backprop direction. This becomes load-bearing for later equivalence proofs (the "fixed-prediction assumption" exploited by Millidge 2020 and analysed by Zahid 2023).

## Limitations
- Equivalence is *asymptotic in* a small-output-error limit. For large target errors at the output, the approximation degrades — i.e. the PCN gradient differs from backprop most strongly precisely where the loss is biggest.
- All experiments are MNIST-scale MLPs. No CNN, no large-scale benchmark, no OOD evaluation.
- The weight-symmetry / weight-transport problem is reduced but not solved — forward and backward weights between layers are still tied. Later work (Millidge 2019, Akrout 2019, asymmetric Hebbian variants) addresses this.
- Inference relaxation costs O(N_steps × forward pass) extra compute per training step. On hardware where backprop is cheap, this is a strict overhead.
- The precision parameters Σ_l are usually fixed to identity in the experiments; the framework's *uncertainty-aware* potential is not exploited.

## Idea seeds for our loop
- ****Inference initialisation as a knob.** The paper shows that initialising v_l = v̂_l (feedforward) makes the PCN gradient track backprop fastest. The opposite extreme — initialising from previous-sample state, or from a noise sample — pushes the network *away* from the backprop fixed point and is exactly the regime where iterative variational PC differs from backprop. Trial axis: initialisation scheme × inference steps × OOD accuracy. Expect non-backprop-equivalent initialisations to win on corruption robustness even when they slightly underperform on clean. Tag: **iteration-for-OOD-evidence**, **counter-evidence-aware** (this is where the central thesis predicts a win).
- ****Precision learning under corruption-channel mismatch.** The Σ_l⁻¹ precisions weight per-layer errors. When the corruption channel changes (e.g. test-time motion blur), the optimal precision per layer shifts. Adding learnable precisions and an inner-loop precision update at *test time* would be a clean adaptation analogue of `[tent_wang_2021]` but inside the PC variational frame. Tag: **iteration-for-OOD-evidence**, **energy-based**.
- **Number of inference steps as a generalisation knob.** Whittington–Bogacz needs ~20 inference steps for tight equivalence. Below that, the updates differ — and the differences may help OOD. Sweep n_inference_steps × OOD accuracy. The H000 baseline in this workspace is the minimal version of this; the more aggressive variations are open. Tag: **iteration-for-OOD-evidence**.
- **Local-Hebbian as a constraint, not a feature.** The equivalence proof needs *Hebbian* updates. Relaxing this to three-factor learning rules (per `[numenta_active_dendrites_2021]`) preserves locality but escapes the equivalence regime — interesting for our non-backprop-equivalence operating point. Tag: **biological**.

## Connections
- `[friston_fep_2009]` — the variational free-energy framing this paper operationalises.
- `[bastos_microcircuits_2012]` — the cortical microcircuit that Whittington–Bogacz's value/error neurons are meant to instantiate.
- `[millidge_backprop_2020]` — extends this paper's equivalence proof to arbitrary computation graphs; uses the same fixed-prediction initialisation trick.
- `[zahid_critical_2023]` — argues that the Whittington–Bogacz equivalence (and its successors) only holds in degenerate variational regimes; **counter-evidence** for any hypothesis that buys exact backprop-equivalent updates and expects iterative-inference generalisation gains.
- `[salvatori_reverse_2022]`, `[pinchetti_benchmark_2024]`, `[jpc_innocenti_2024]`, `[ngc_ororbia_2022]` — modern variants and benchmarks built on this scaffold.
- `[intro_pcn_ml_2025]` — survey paper that catalogues the family of PCNs descending from this one.

## Relevant themes
biological, energy-based, iteration-for-OOD-evidence

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
**⚠️ PDF mismatch.** The file at `./whittington_bogacz_2017.pdf` (verified 2026-06-22) is not the Whittington & Bogacz 2017 paper. Its PDF metadata reads: Title = "Inflammation: The Common Pathway of Stress-Related Diseases"; Author = Chun-Lei Jiang; DOI = 10.3389/fnhum.2017.00316 (Frontiers in Human Neuroscience). It is a stress/inflammation review accidentally placed in this slot. This synthesis was written from prior knowledge of the canonical Whittington–Bogacz 2017 paper because re-downloading the correct PDF was outside the agent's tooling envelope. A follow-up sub-agent must:
1. Replace the file at `knowledge_base/papers/whittington_bogacz_2017/whittington_bogacz_2017.pdf` with the actual Neural Computation 29:1229-1262 paper (doi:10.1162/NECO_a_00949).
2. Re-verify this synthesis against the correct PDF; key claims to spot-check are (a) the equilibrium-Hebbian-equals-backprop theorem in the small-output-error / linear limit, (b) MNIST and autoencoder benchmarks, (c) the v_l = v̂_l initialisation that makes equivalence tight.
3. Auto-issue could not be filed (no `gh` CLI in this worktree environment); please open one manually if doing this cleanup work.
