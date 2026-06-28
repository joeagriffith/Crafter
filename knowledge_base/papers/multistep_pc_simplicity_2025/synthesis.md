# Prediction horizon shapes representations in predictive learning — synthesis

- **Slug**: multistep_pc_simplicity_2025
- **Authors**: Ratzon, Aviv; Barak, Omri
- **Year / venue**: 2026 (arXiv 2511.09290, v2 May 2026; preprint)
- **Source PDF**: ./multistep_pc_simplicity_2025.pdf
- **Citation**: Ratzon A, Barak O. Prediction horizon shapes representations in predictive learning. arXiv:2511.09290 (2025/2026).
- **Synthesized by**: agent-20260622
- **Status**: initial

> *Note on title*: the workspace slug refers to "multi-step PC simplicity"; the canonical paper title in the PDF is **"Prediction horizon shapes representations in predictive learning"**. The simplicity-bias framing is the discussion-section interpretation, not the paper's headline.

## TL;DR
Ratzon & Barak identify the *prediction horizon* — how many steps into the future a network is asked to predict — as a critical and under-appreciated structural lever in predictive learning. In a tractable linear setting and across nonlinear extensions (deep networks, MNIST cDCGAN, predictive coding networks), they show that *long* prediction horizons combined with the implicit low-rank bias of deep networks consistently drive representations to collapse onto the latent geometry of the environment, while short horizons leave representations unstructured. They give a mechanistic account via the spectrum of the OLS estimator: as the horizon grows, the data's effective rank shrinks, deep networks (which prefer low-rank solutions) lock onto the latent state direction, and OOD generalisation improves. The paper provides theoretical scaffolding for the central-thesis claim that *the structure of the iterative prediction problem* — not just optimisation — biases networks toward generalisable representations.

## Problem framing
The paper sits in the "implicit bias of gradient descent" tradition (Soudry et al. 2018; Ji & Telgarsky 2018; Lyu & Li 2019 — gradient descent on separable data converges to a max-margin solution; Dai/Karzand/Srebro 2021 representation costs on deep linear networks). It also engages with: predictive learning as world-model formation (Recanatesi et al. 2021 on low-dimensional latent extraction; Levenstein et al. 2024 hippocampal-prediction theory; Stachenfeld 2017); and the broader question of *when* predictive objectives recover latent state versus memorising input-output relations. The novel piece is making the *horizon* the explicit control variable.

## Method
The minimal setting: an "environment" with S latent states arranged on a line; at each step the agent sees a one-hot observation O(s) and a one-hot action g(a) with a ∈ [-A, A]; the network must predict O(s+a). A is the *prediction horizon* — the maximal action magnitude. This setup factors out time, making it tractable.

A deep linear network with cross-entropy loss is trained on all valid (state, action) pairs. The implicit bias of gradient descent on deep linear nets with multiclass classification corresponds to optimisation of a Schatten 2/L quasi-norm of the effective weight W:
- argmin_W ‖W‖^{SC}_{2/L} = (Σ_i σ_i^{2/L})^{L/2}, s.t. margin constraints.

For L > 1 this biases toward low-rank solutions (sparse singular value spectra). To connect this to the prediction horizon, the authors study the OLS estimator Σ = (XᵀX)⁺ Xᵀ Y. Its participation ratio (PR) — a measure of effective rank — they compute analytically:
- PR(A, S) = S³ / [(2A + 1)(S² + 2AS + S − 4A² − 4A − 1)].
- Large A: PR ∝ S/A → rank shrinks linearly with horizon.

So long horizons → low-rank OLS solution → deep networks lock onto the dominant singular vector → that vector corresponds to the latent-state direction → representations recover the latent geometry. The paper validates this on:
- Linear toy: spontaneous collapse to a 1-D manifold in last-hidden-layer activations once A ≥ A_thresh ∝ S, and only for sufficiently deep networks (Fig 2).
- Multiple environments: representations align across environments under long horizons (Fig 4).
- Continuous environments with discontinuities (multi-room): long horizons stitch rooms into a coherent 1-D manifold (Fig 5).
- MNIST cDCGAN with held-out (state, action) pairs: long horizon → lower-dim, more linearly decodable representations → better OOD accuracy (Fig 6).
- **Predictive Coding Network** (5-layer PCN, Fig 7): identical qualitative pattern. As horizon grows, dimensionality drops, decodability of latent from first PC rises. A "mesoscale optimum" at A ≈ S/2 is observed but not theoretically explained.
- Hyperbolic (tree) geometry: longer horizons + deeper networks → higher OOD accuracy (Fig 8).

## Key results
- **Phase transition in representation structure** as A crosses A_thresh ∝ S; only deep networks (L ≥ 2 in nonlinear case) exhibit collapse.
- **Analytical formula** PR(A, S) = S³/[(2A+1)(...)] linking horizon to effective dimensionality.
- **MNIST OOD**: cDCGANs with larger A achieve higher accuracy on held-out (digit, action) combinations, with OOD accuracy positively correlated with linear decodability of latent state.
- **PCN-specific** (their Fig 7): same horizon-driven dimensionality reduction occurs in a predictive coding network — but with a mesoscale optimum around A ≈ S/2 above which dimensionality starts to grow again. Important caveat the authors flag as unexplained.
- **Generalisation in tree geometry**: deeper networks + longer horizons → linear scaling of OOD test accuracy, suggesting compositional representations.

## Limitations
- The theoretical analysis is *deep-linear-network specific*. Extensions to nonlinear / CNN settings are empirical, not analytical.
- The PCN result is a single sweep; no explanation of why PCN has a mesoscale optimum (where linear nets monotonically improve with A). This is an open question the authors call out.
- All "environments" are extremely stylised — discrete linear, or two-room, or tree. The OOD evaluation is "held-out state-action pairs", not corruption-style shift.
- No comparison against amortised baselines on the OOD task to verify that iterative-PC's gains are above amortised matched-capacity baselines.
- The "predictive coding network" here is treated as one architectural choice among several; the paper does not deeply analyse the variational-Bayes implications.

## Idea seeds for our loop
- ****Prediction horizon as an explicit architectural knob in PC.** This paper makes A (horizon) a first-class design dimension. The cleanest mapping to our F-MNIST-C setting: define an auxiliary self-supervised task on the latent (e.g. predict a *transformation* of the input several steps deep — rotation chain, blur chain) and vary the prediction depth. Hypothesis: deeper auxiliary prediction → lower-rank latent → better OOD. Tag: **generalisation-theory**, **iteration-for-OOD-evidence**.
- ****Implicit low-rank bias as a generalisation lever.** Their theorem says deep networks under gradient descent are biased toward low-rank solutions of the OLS estimator. If we add a *rank-constraining* regulariser (Schatten 2/L norm, nuclear norm, or just deeper effective depth via PC inference unrolling) to a shallow PC network, we should see OOD wins even at fixed parameter count. Tag: **generalisation-theory**, **sparsity** (low rank is a structural-sparsity analogue).
- ****The PCN mesoscale optimum is an unexplained empirical anomaly.** Their Fig 7 shows PR rising again at very long horizons in a PCN — opposite to the linear-network monotone trend. If we replicate this in our F-MNIST-C loop, it implies a sweet spot in inference depth that *neither* shallow amortised nor maximally iterative networks hit. Worth a sweep of n_inference_steps × OOD accuracy with non-monotone hypothesis in mind. Tag: **iteration-for-OOD-evidence**, **generalisation-theory**.
- **Multi-environment alignment (Fig 4) as a sparsity-of-representation prior.** When the same network sees two environments, long horizons make the representations *share* a low-rank substrate that exposes the common structure. This is a clean motivation for an experimental setup where we train on F-MNIST + a related dataset (KMNIST? a coloured version?) and check whether long-horizon prediction in the joint latent yields better cross-dataset NLL — directly aligned with our secondary metrics. Tag: **iteration-for-OOD-evidence**.

## Connections
- `[friston_fep_2009]` — provides the variational framing; this paper provides the implicit-bias-of-deep-nets framing. The two are orthogonal mechanisms that point in the same direction.
- `[whittington_bogacz_2017]`, `[millidge_backprop_2020]` — the PCN architecture used in their Fig 7 is in this family. Their result transfers.
- `[zahid_critical_2023]` — Zahid argues equivalence kills the variational lever; Ratzon & Barak show that the *prediction-horizon* lever survives even when latent-Bayes structure isn't explicit. Complementary.
- `[brain_like_vi_2024]` — direct OOD-via-iteration evidence; the simplicity-bias-from-horizon story here gives the theoretical scaffolding for the Brain-like-VI empirics.
- `[hybrid_pc_2023]` — amortised-fast + iterative-slow may correspond to short-horizon + long-horizon prediction respectively.
- `[ngc_ororbia_2022]` — generative-PC alternative whose horizon dependence is also worth studying.
- `[ttt_sun_2020]`, `[tent_wang_2021]` — test-time iteration with auxiliary objectives; their objectives are essentially short-horizon (single-step) self-supervised. The Ratzon & Barak result suggests longer-horizon test-time objectives could buy more.

## Relevant themes
generalisation-theory, iteration-for-OOD-evidence, sparsity (via low-rank implicit bias)

## Tier
must-read

## Out of scope
—

## Notes / follow-ups
The single most actionable observation for our loop is Fig 7: a PCN exhibits a *non-monotone* PR-vs-horizon curve with an optimum near A ≈ S/2. If reproducible in our setting (where S = 10 F-MNIST classes), it suggests the optimal inference depth for the H000 baseline is *not* "as many steps as we can afford" but a structured sweet spot. A trial that scans inference_steps ∈ {2, 4, 8, 16, 32, 64} × OOD accuracy is cheap and directly tests this. The Ratzon & Barak paper does not connect their horizon to *inference unrolling depth* explicitly, but the analogy (predicting further into the future ≈ unrolling inference deeper) is the load-bearing translation step our loop should make.
