# Associative Memories via Predictive Coding — synthesis

- **Slug**: salvatori_associative_2021
- **Authors**: Salvatori, Tommaso; Song, Yuhang; Hong, Yujian; Frieder, Simon; Sha, Lei; Xu, Zhenghua; Bogacz, Rafal; Lukasiewicz, Thomas
- **Year / venue**: 2021 / NeurIPS 2021 (arXiv:2109.08063)
- **Source PDF**: ./salvatori_associative_2021.pdf
- **Citation**: Salvatori T, Song Y, Hong Y, Frieder S, Sha L, Xu Z, Bogacz R, Lukasiewicz T. Associative Memories via Predictive Coding. NeurIPS 2021.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
Salvatori et al. reframe a generative PCN as an associative memory: training images are stored as *attractors* of the energy dynamics, and a corrupted or partial input flows downhill via iterative inference to the nearest stored point. With 2–10 hidden layers of 256–8192 units they retrieve full ImageNet (224×224) images from 1/8 of the pixels — outperforming overparameterised autoencoders and modern Hopfield networks (MHNs) by large margins on Tiny ImageNet, CIFAR-10, SVHN. The result establishes the modern PCN flagship: iterative-inference-as-retrieval scales to natural images.

## Problem framing
Builds on the associative-memory tradition (Hopfield 1982; modern Hopfield networks, Ramsauer 2020) and the PC tradition of Rao & Ballard 1999, Whittington & Bogacz 2017. The novelty is reading a *trained generative* PCN as an associative memory whose attractors are the training points. They explicitly position against (i) MHNs (one-shot but limited capacity for natural images), and (ii) over-parameterised autoencoders (work but inefficient). The conceptual move: a PCN trained to generate a dataset has the dataset embedded as energy minima — therefore retrieval is just inference.

## Method
A generative PCN of L layers. Sensory layer 0 is fixed to the image s̄; the memory vector b̄ at layer L is trainable. Prediction at layer l is μ_{i,t}^l = Σ θ_{i,j}^{l+1} f(x_{j,t}^{l+1}), error ε_{i,t}^l = x_{i,t}^l − μ_{i,t}^l, global energy E_t = ½ Σ (ε_{i,t}^l)². Inference learning (IL):

Δx_{i,t}^l = γ · (−ε_{i,t}^l + f'(x_{i,t}^l) Σ_k ε_{k,t}^{l-1} θ_{k,i}^l)
Δθ_{i,j}^{l+1} = α · ε_{i,T}^l f(x_{j,T}^{l+1})

Training: clamp x̄^0 = s̄, run inference to convergence, then one Hebbian weight step. Each datum becomes a local minimum of E_t.

Retrieval (denoising): fix x̄^0 to corrupted c̄, run inference for T steps with weights frozen, read back μ̄_T^0. Retrieval (completion): fix only the *known* pixels of partial input s̄'; let the rest evolve via inference until convergence; read μ̄_T^0. The iterative inference dynamics implement a flow toward the nearest stored attractor.

## Key results
- **Capacity vs autoencoders**: 2-layer PCN with 256 hidden neurons stores 250 CIFAR-10 / Tiny ImageNet images with near-100% denoising retrieval (Fig 3); equivalent 3-layer AE with 2048 neurons fails to retrieve a single Tiny ImageNet image. PCNs win in network-efficient parameter use.
- **Depth helps**: a 10-layer PCN reconstructs >98% of 500 Tiny ImageNet images given half the pixels (vs 72% for matched AE); 74% given 1/4 of pixels (vs 48%) (Fig 8).
- **ImageNet at scale**: 8192-hidden-unit network perfectly reconstructs 224×224 ImageNet images from just 1/8 of the pixels.
- **Beats MHNs**: MHNs only retrieve 1–9 images correctly when given corrupted CIFAR-10 inputs (out of 50–500 stored); PCN retrieves nearly all. MHNs converge to "wrong-but-confident" attractors; PCNs converge to "fuzzy-but-correct" attractors.
- **Hetero-associative memory**: image ↔ 25-dim caption-embedding retrieval. PCNs handle this; baselines fail.

## Limitations
- The "OOD" notion here is *partial / noisy version of stored data*, not distribution shift. Excellent direct relevance to the iterative-inference-as-attractor view, but not a clean cross-distribution test.
- No comparison to a *backprop-trained* feedforward classifier on the same data.
- The retrieval threshold is hand-tuned (0.001 or 0.005 MSE) and ad hoc.
- Capacity scales with hidden dimension but the relationship is empirical, not theoretically bounded.
- No exploration of *which* corruptions iterative inference handles best — they only test Gaussian noise and random masking.

## Idea seeds for our loop

- **PCN-as-attractor for F-MNIST-C.** ** A trained generative PCN should naturally treat each corrupted F-MNIST-C image as a noisy initial condition that iterates toward its nearest stored attractor. Trial idea: train a generative PCN on F-MNIST clean, classify corrupted samples by which class-conditional retrieval has lowest reconstruction error. This is a free-energy-based classifier — directly testing the central thesis on the primary axis. Tag: iteration-for-OOD-evidence, energy-based.

- **Iteration-step ablation on retrieval.** Their Fig 7 shows capacity changes with hidden dim; they don't show the analogue for *number of inference steps T*. Sweep T on F-MNIST-C and ask: does more iteration buy more corruption-robust retrieval? If yes — direct evidence for the thesis. If no (saturates at small T) — partial counter-evidence with high signal value.

- **Multi-attractor competition for classification.** ** Train one PCN per class on F-MNIST clean. At test time run iterative inference under each class-conditional model and assign the class whose retrieval reaches lowest energy. This is energy-based classification via class-conditioned PCN retrieval. Combines associative-memory + iterative inference + (implicit) competition between class models. Tag: competition, energy-based, iteration-for-OOD-evidence.

- **Depth-vs-iteration trade-off.** Deeper PCNs have higher capacity (Fig 8). Match parameter budget: is a deeper, fewer-step network or a shallower, more-step network better on F-MNIST-C? Helps disentangle "compute via depth" from "compute via iteration" — directly relevant to the DEQ counter-evidence in the central thesis.

## Connections
- `[hybrid_pc_2023]` — uses the same iterative-inference machinery; amortised initialiser would speed associative retrieval here too.
- `[ngc_ororbia_2022]` — NGC's pattern-completion experiments are the parallel result with a different (lateral-competition) PCN variant.
- `[hinton_dbn_2006]`, `[hinton_boltzmann_1985]` — classical energy-based / attractor memory tradition. PCN is the modern reincarnation; cross-pollination opportunities (CD-style training, mean-field updates) remain unexplored.
- `[pinchetti_benchmark_2024]` — replicates associative-memory experiments on a more complete benchmark; useful for reference numbers.
- `[salvatori_reverse_2022]` — same lab; complementary on the training-side equivalence question.
- `[multistep_pc_simplicity_2025]` — theoretical bridge: attractors-as-simplicity-bias.
- `[hawkins_columns_2017]`, `[numenta_thousand_brains_project_2024]` — Hawkins' columnar attractors are conceptually similar; cross-pollination into multi-column PCN attractor models is an under-explored direction.

## Relevant themes
energy-based, iteration-for-OOD-evidence (partial), biological

## Tier
high

## Out of scope
—

## Notes / follow-ups
- Open code is referenced in the paper supplement.
- The hetero-associative experiments (image ↔ caption) are a hint that the PCN attractor frame might transfer to multi-modal tasks — out of scope for our F-MNIST-C primary, but worth noting.
