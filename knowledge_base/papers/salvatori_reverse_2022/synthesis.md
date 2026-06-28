# Reverse Differentiation via Predictive Coding — synthesis

- **Slug**: salvatori_reverse_2022
- **Authors**: Salvatori, Tommaso; Song, Yuhang; Xu, Zhenghua; Lukasiewicz, Thomas; Bogacz, Rafal
- **Year / venue**: 2022 / AAAI 2022 (arXiv:2103.04689)
- **Source PDF**: ./salvatori_reverse_2022.pdf
- **Citation**: Salvatori T, Song Y, Xu Z, Lukasiewicz T, Bogacz R. Reverse Differentiation via Predictive Coding. AAAI 2022.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
The authors generalise Z-IL (zero-divergence inference learning, a PC training variant that exactly reproduces BP on MLPs) to *any* computational graph — CNNs, RNNs, ResNets, transformers — by inserting "identity vertices" to level the DAG so error signals arrive in lock-step. The resulting weight updates have exactly zero divergence from BP. Z-IL is comparable to BP in wall-clock (3.81 ms vs 3.72 ms on MLPs; 12.53 vs 12.43 on ResNet-18) and several orders of magnitude faster than standard inference learning. This is the strongest "PC ≡ BP" result; it is **counter-evidence** to our central thesis's positive direction but essential context for understanding the equivalence-regime carve-out.

## Problem framing
Build directly on Whittington & Bogacz 2017 (PC approximates BP asymptotically), Millidge et al 2020 (PC approximates BP along arbitrary graphs), and Song et al 2020 (Z-IL exactly equals BP on MLPs). The new contribution: extend Z-IL exactness from MLPs to arbitrary computational graphs. Frame as "bridging neuroscience and deep learning" — the same architectures used in modern ML can in principle be trained by a biologically plausible local learning rule with no efficiency penalty.

## Method
A computational graph G=(V,E) where each internal node v_i has an elementary function g_i. The PC counterpart attaches a value node x_{i,t} and error node ε_{i,t} = μ_{i,t} − x_{i,t} to every node, with energy F_t = ½ Σ ε² and inference update Δx_{i,t} = −γ ∂F_t/∂x_{i,t}. Standard IL converges in T steps to BP-equivalent weight updates.

Z-IL modifies IL in three ways: (i) initialise all errors to zero via a forward pass; (ii) update each weight parameter ζ_l only at time step t=l (one specific step); (iii) update value nodes continuously. **Identity vertices**: any edge e_{i,j} where the distance d_j − d_i > 1 gets padded with (d_j − d_i) identity nodes so that the error signal reaches every node in a single time step regardless of graph structure. This "levels" the DAG.

Theorem 4 (main result): on the levelled graph, Δζ_i (Z-IL) = Δz_i (BP) exactly, for every node and every iteration. Empirical Table 1: BP-IL divergence is 0 on MLP/CNN/RNN, 4.53×10⁷ on ResNet-18 *before* their fix, 0 after. Table 2: per-update wall-clock times for BP and Z-IL are within 5% across MLP, AlexNet, RNN, ResNet-18, Transformer; IL is 100–200× slower.

## Key results
- **Exact BP equivalence on every architecture tested**: MLP, AlexNet (CNN), many-to-one RNN, ResNet-18, single-layer Transformer.
- **Wall-clock parity with BP**: 3.81/3.72 (MLP), 8.86/8.61 (AlexNet), 5.67/5.64 (RNN), 12.53/12.43 (ResNet-18), 20.53/20.43 (Transformer) ms per update.
- **IL is ~100–200× slower**: the orders-of-magnitude advantage of Z-IL over IL is the practical contribution.
- The skip-connection issue (ResNet) is solved by adding identity vertices that delay paths to be lockstep.

## Limitations
- The "biological plausibility" claim is technical — Z-IL still requires synchronised cross-graph timing. The identity-vertex trick reinterprets these as transmission delays, but the precise scheduling is non-trivial.
- **For our workspace, the load-bearing limitation is conceptual**: Z-IL achieves exact BP equivalence at the cost of *losing the variational interpretation*. Per the central thesis (and `[zahid_critical_2023]`), this means iterative inference no longer optimises free energy — it just computes the BP gradient. The lever the thesis depends on is given up.
- No tests on OOD, no corruption benchmarks, no test-time iteration benefits — by construction, because Z-IL is a training-time algorithm whose inference equals BP's forward pass.

## Idea seeds for our loop

Most "ideas" here are **caution flags, not directions**. The paper is essential context but the hypothesis space it opens is largely *off-thesis*.

- **Counter-evidence anchor.** ** Any sub-agent tempted to propose "use Z-IL / FPA-PC / prospective-config" as a hypothesis should read this paper first and re-read `[zahid_critical_2023]`. The training-equivalence variants don't move the OOD lever; they give up the variational interpretation that does. File this synthesis as evidence whenever the loop drifts toward "let's just train standard MLPs with PC."

- **Cleanly separate training-side and inference-side claims.** The paper distinguishes (a) PC as a training rule that approximates BP, and (b) PC as iterative inference at test time. The OOD lever lives in (b). A practical follow-on: even when *training* via Z-IL (matching BP's training efficiency), keep iterative inference at *test time*. This combination has not been published — could be a clean trial. Tag: iteration-for-OOD-evidence.

- **Identity-vertex levelling as a tool.** The paper's identity-vertex idea generalises naturally to skip connections in our generative paths. If we want skip connections in a PCN (often improves training stability), Z-IL's levelling tells us how the local-update rule must change. Useful engineering knowledge for any sub-agent building deeper PCNs. Tag: implementation-detail.

- **Compute time on the wall clock argument.** The Table 2 numbers tell us Z-IL is ~100× faster than IL on a Transformer. If a sub-agent wants iterative inference *only at test time*, the cost is bounded — the iteration overhead is small relative to a forward pass. This is a feasibility argument for the iterative-inference-as-OOD-lever bet.

## Connections
- `[zahid_critical_2023]` — primary companion. Zahid criticises exactly the equivalence regime this paper expands, arguing it sacrifices the variational interpretation.
- `[millidge_backprop_2020]` — direct precursor. This paper extends Millidge's "PC approximates BP along arbitrary graphs" to *exact* equivalence on levelled graphs.
- `[whittington_bogacz_2017]` — the original PC-≈-BP result.
- `[hybrid_pc_2023]` — orthogonal direction: keep iterative inference in the regime where it's *not* BP-equivalent, but speed it up via amortisation.
- `[pinchetti_benchmark_2024]` — same lab; Pinchetti's benchmark includes iPC which is an extension of these ideas.
- `[jpc_innocenti_2024]` — JPC implements the inference dynamics this paper sidesteps; the libraries are complementary tools.
- `[hinton_forward_forward_2022]` — different family of "BP alternative" that *doesn't* claim equivalence; useful comparison for what we're sacrificing.

## Relevant themes
counter-evidence, generalisation-theory (negatively)

## Tier
counter-evidence

## Out of scope
—

## Notes / follow-ups
- The discussion-section claim that "deep learning may actually be more closely related to information processing in the brain than commonly thought" is best read as a *training-side* claim, not an inference-side one. The split matters for our workspace.
