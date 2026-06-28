# Benchmarking Predictive Coding Networks — Made Simple — synthesis

- **Slug**: pinchetti_benchmark_2024
- **Authors**: Pinchetti, Luca; Qi, Chang; Lokshyn, Oleh; Oliviers, Gaspard; Emde, Cornelius; Tang, Mufeng; M'Charrak, Amine; Frieder, Simon; Menzat, Bayar; Bogacz, Rafal; Lukasiewicz, Thomas; Salvatori, Tommaso
- **Year / venue**: 2024 / ICLR 2025 (arXiv:2407.01163)
- **Source PDF**: ./pinchetti_benchmark_2024.pdf
- **Citation**: Pinchetti L, Qi C, Lokshyn O, Oliviers G, Emde C, Tang M, M'Charrak A, Frieder S, Menzat B, Bogacz R, Lukasiewicz T, Salvatori T. Benchmarking Predictive Coding Networks — Made Simple. ICLR 2025.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
The PCX library (JAX/Equinox, JIT) and a large-scale benchmark of PC variants (PC-CE/SE, iPC, MCPC, PN/NN/CN nudging variants) on MNIST, F-MNIST, CIFAR-10/100, CelebA, Tiny ImageNet, with VGG-5/7/9 and ResNet-18 backbones. Headline findings: (i) nudging variants (PN/NN/CN) match BP up to VGG-7 on CIFAR-100; (ii) PC fails to scale past VGG-9; (iii) the load-bearing knobs are state learning rate γ and width-dependent stability; (iv) energy imbalance across layers grows exponentially with depth and limits PC's reach. The benchmark is *the* current floor for PCN performance — what we have to beat.

## Problem framing
The PC literature suffers from non-comparable evaluations: each paper uses its own architecture / dataset / metric. The authors argue this masks the field's real open problem: **scalability**. Built on Salvatori et al's PC line, Whittington-Bogacz 2017's training-side theory, Equilibrium Propagation (Scellier & Bengio 2017), and the recent variant papers (Salvatori 2024 iPC, Oliviers 2024 MCPC). They explicitly state: "the field is avoiding what we believe to be the most important open problem: scalability."

## Method
**Tool — PCX**: JAX library following Equinox conventions; PC components as PyTrees; modular Layers, Optimizers, Nodes; JIT compilation. Up to 50× speedup over prior PC libraries; 5.5 s/epoch on CIFAR-100 with VGG-5 (vs ~110 s for Eqprop). Code: github.com/liukidar/pcx.

**Variants tested**:
- **PC**: standard, with cross-entropy (CE) or squared-error (SE) loss. F = ½ Σ (h_l − μ_l)².
- **iPC** (Salvatori 2024): every weight updated at every inference step (no inner-outer loop).
- **MCPC** (Oliviers 2024): Langevin sampling — Gaussian noise η ~ N(0, √2γ) added to each state update.
- **PN/NN/CN**: Positive / Negative / Centered nudging from the Eqprop literature. PN: perturb output toward true label; NN: away from target with inverted weight sign; CN: alternate epochs of PN and NN.

Discriminative mode clamps h_0=x, h_L=y, and infers internal h_l to F minimum, then updates θ. Generative mode clamps only h_L=x (no label), infers h_l, reads reconstruction at h_0.

**Initialization**: forward-pass init of h_l = μ_l (so ε_l = 0 for hidden layers, error concentrated at output) consistently outperforms zero or Gaussian init.

## Key results
- **Top model**: CN-trained VGG-5 — CIFAR-10 88.42% (vs BP 89.43%); CIFAR-100 Top-5 86.60% (vs BP 85.85%); CIFAR-100 Top-1 67.19% vs BP 60.82%. CN actually *beats BP* on CIFAR-100.
- **PC matches BP up to VGG-5/7; lags from VGG-9 onward**: on Tiny ImageNet (VGG-9), PC achieves 31.5%, BP 88.4%.
- **ResNet-18 with PC fails badly** (PC-CE 43.19% vs BP 92.83% on CIFAR-10). PC does not yet scale to residual architectures.
- **MCPC achieves lower FID than VAE on Iris** (FID 2.53 vs 4.19) at matched model size — sampling-based PC has generative quality.
- **iPC is best on small architectures, worst on deep ones** — its parallel weight updates work for shallow nets, lose stability as depth grows.
- **Critical analysis (Fig 6)**: small state learning rate γ ≪ 1 gives best test accuracy but causes 6-orders-of-magnitude energy imbalance between first and last layer. Large γ=1 gives balanced energies but poor accuracy. Identified as the central tension that blocks scaling.
- **AdamW destabilises wide layers** at small γ (Fig 7). SGD is more robust but slower.

## Limitations
- **Scalability ceiling at VGG-9**: PC can't currently train deep modern architectures. ResNet-18 with PC is broken.
- **No OOD/corruption benchmarks**: the entire benchmark is in-distribution test accuracy. The PC-vs-BP comparison is silent on the generalisation lever our workspace cares about.
- **No depth-aware energy normalisation** is offered — the energy-imbalance problem is diagnosed but not solved.
- **Adam still recommended despite stability issues** — they conclude better optimisation is needed but stop short of proposing it.
- Generative mode uses decoder-only architectures; no comparison to modern strong generators.

## Idea seeds for our loop

- **Use PCX-style JIT in the trial harness.** ** Their 50× speedup over the Song reference implementation translates directly to more trials per session. If a trial author is iterating inference 100+ times per sample, JIT-compiling the inference loop is the difference between 5-min and 5-hour wall-clock. PCX (JAX) and JPC `[jpc_innocenti_2024]` both demonstrate this. Trial-harness suggestion: pull pcx or jpc into the H000 baseline for the speed gain. Tag: implementation-efficiency.

- **Test the nudging variants on F-MNIST-C.** CN-PC beat BP on CIFAR-100 in-distribution. The OOD behaviour is unknown. Trial idea: re-run their CN-PC at F-MNIST scale and measure on F-MNIST-C. If CN-PC also wins OOD, it might be one of the strongest signal points in the workspace. If it doesn't, that's still valuable (CN's win was specific to in-dist).

- **Energy-imbalance regularisation.** ** Their Fig 6c-d shows γ controls in-dist accuracy and γ controls energy balance, but they pull in opposite directions. Hypothesis: add an explicit per-layer energy-normalisation term so γ can be small (high in-dist accuracy) without exponential energy decay. Could enable PC training of much deeper models. High novelty, on-thesis (depth → more iterative compute → more OOD potential). Tag: depth, energy-based.

- **OOD ablation of the nudging variants.** Their PN/NN/CN nudging are inspired by Eqprop. They each subtly change what the iterative inference is optimising. Hypothesis: PN keeps the variational interpretation (perturbing toward target keeps the energy gradient meaningful); CN may or may not (alternating perturbations might be off-thesis). Worth a small ablation. Tag: iteration-for-OOD-evidence, energy-based.

- **MCPC + amortised initialiser = MCPC-HPC.** They show MCPC matches VAE on generation. Sampling-based PC with HPC-style amortised init could be a clean hybrid: posterior sampling at test time, amortised speed-up. Combines two on-thesis directions. Tag: iteration-for-OOD-evidence, energy-based.

## Connections
- `[jpc_innocenti_2024]` — JPC and PCX are complementary JAX PC libraries; JPC's ODE-solver inference may compose with PCX's structured benchmark.
- `[hybrid_pc_2023]` — HPC isn't in this benchmark but should be; their methodology applies cleanly.
- `[salvatori_associative_2021]` — same lab; the associative-memory experiments in Fig 5 / Table 3 of this paper directly extend the 2021 result on a larger benchmark.
- `[salvatori_reverse_2022]` — Z-IL not directly tested; iPC is the closest variant.
- `[ngc_ororbia_2022]` — NGC is mentioned as alternative PC framework but not benchmarked head-to-head.
- `[mcpc_oliviers_2024]` — MCPC is one of the benchmarked variants here; this paper provides MCPC's first cross-dataset comparison.
- `[whittington_bogacz_2017]` — training-theory foundation.
- `[zahid_critical_2023]` — relevant critique of equivalence variants like iPC.
- `[multistep_pc_simplicity_2025]` — theoretical link between inference depth and simplicity bias.

## Relevant themes
generalisation-theory, energy-based, iteration-for-OOD-evidence (indirectly — gives us the floor)

## Tier
high

## Out of scope
—

## Notes / follow-ups
- The library code (github.com/liukidar/pcx) and the benchmark recipe (CIFAR variants + VGG architectures + AdamW with state-LR sweep) are exactly the kind of pull-then-improve starting point our spec calls for.
- The "cookbook" appendix has the hyperparameter recipes — worth a careful read by any sub-agent that pulls PCX.
