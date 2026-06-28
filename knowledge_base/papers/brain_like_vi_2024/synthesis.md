# Brain-like Variational Inference — synthesis

- **Slug**: brain_like_vi_2024
- **Authors**: Vafaii, Hadi; Galor, Dekel; Yates, Jacob L.
- **Year / venue**: 2025 / NeurIPS 2025 (arXiv 2410.19315v3)
- **Source PDF**: ./brain_like_vi_2024.pdf
- **Citation**: Vafaii, H., Galor, D., & Yates, J. L. (2025). Brain-like Variational Inference. *NeurIPS 2025*. arXiv:2410.19315.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR

The authors derive a family of "iterative VAE" architectures (iP-VAE, iG-VAE, iG_relu-VAE) from first principles by prescribing **natural gradient descent on variational free energy** with online belief updating. The headline empirical finding is the single most direct support for this workspace's thesis: across MNIST, EMNIST, Omniglot and cross-dataset (e.g. MNIST → ImageNet32) shifts, **the less amortised the inference, the better the OOD generalisation** — a monotone trend from fully-amortised P/G-VAE through hybrid semi-amortised → iterative VAE. iP-VAE also beats Gaussian PCNs on a sparsity-reconstruction Pareto and learns Gabor-like V1 features that recombine to represent novel inputs.

## Problem framing

Sits squarely in the variational-inference / `[whittington_bogacz_2017]`-style PC tradition, but reframes it through Khan & Rue's Bayesian Learning Rule. Builds on `[pvae_vafaii_2024]` (Poisson VAE; amortised cousin) and on Rao-Ballard PC, sparse coding (LCA), and standard G-VAE. Their unification framework (FOND = Free energy Online Natural-gradient Dynamics) is explicitly prescriptive: pick (a) distributions for posterior/prior/likelihood and (b) parameterisation, then **all three** of natural-gradient / online / iterative dynamics fall out automatically.

## Method

Variational free energy F = E_q[||x − Φz||²/2] + β·D_KL(q ‖ p₀). For Poisson posterior and prior with rate r = exp(u) (u = membrane potential), Fisher metric G(u) = exp(u), giving the natural-gradient inference dynamics:

  u̇ ∝ Φᵀ x  −  Φᵀ Φ z(u)  −  β (u − u₀)

with z ∼ Pois(z; exp(u)). The three terms read as feedforward drive, recurrent "explaining away" via sampled spikes (not membrane potentials — biologically tighter than PC), and a homeostatic KL leak. Inference is online: each step the current posterior becomes next step's prior (rolling-prior). Training uses T_train inference steps per batch then a single weight update through the unrolled trajectory (i.e. BPTT through inference, T_train as "effective model depth"). Linear decoder Φ in the main paper, MLP/conv decoders in appendix.

The two control knobs are T_train (inference depth at train time) and β (rate-distortion / sparsity trade-off). T_test = 1000 in reported results; the iteration *consistently converges beyond the training horizon* — a striking generalisation-of-the-dynamics result on its own.

## Key results

- **Monotone amortisation→generalisation trend (D.5.1, this is the load-bearing finding for our workspace):** fully amortised < heavily-amortised hybrid (ia-VAE) < semi-amortised (sa-VAE) < fully iterative (iP-VAE) on OOD reconstruction. Authors' words: "the less amortized the inference process, the better the generalization capability."
- **Cross-domain (MNIST → ImageNet32):** iP-VAE's reconstruction MSE *continues to decrease with more inference iterations* on natural-image patches it has never seen, while hybrid ia-VAE diverges and sa-VAE plateaus (Fig. 16, p.62). The model trained only on digits captures structural detail of natural images by recombining its learned Gabor-like primitives.
- **Sparsity-reconstruction Pareto (Fig. 4, p.10):** iP-VAE and LCA dominate the Pareto frontier; amortised models are strictly worse. iP-VAE achieves R² ≈ 0.83 with 77 % zeros on natural patches; PCN/iPC are dominated.
- **CelebA 128×128 (Table 4, p.64):** iP-VAE achieves the best reconstruction-sparsity trade-off with 0.7 M parameters vs 1.9 M for amortised P-VAE.
- **MNIST classification via downstream linear probe:** P-VAE reaches ~98 % comparable to supervised PCNs (appendix C.6).

## Limitations

- All quantitative OOD numbers are on **reconstruction / sparsity**, not classification accuracy. The downstream-classification OOD case is sketched in appendix but not the main story.
- T_test = 1000 is generous. Iteration runtime is roughly 2× amortised training time and the inference cost remains an open scaling problem (D.5.1, D.5.5).
- The generative model is single-layer; "hierarchical extensions with realistic synaptic transmission delays" is flagged as future work (D.5.4).
- Adaptation is purely retrospective ("predictive dynamics" with anticipatory updates left as future work, D.5.3).
- Operator-relevant carve-out: the canonical iP-VAE is *spiking* (Poisson-sampled integer spike counts in the recurrent term). Spiking is **out of scope** for our build (`spec.md`). The non-spiking siblings — iG-VAE and especially iG_relu-VAE — are in-scope analogues to study (operator distinction also notes P-VAE itself is in-scope because Poisson latents ≠ spike trains).

## Idea seeds for our loop

- **Direct port: monotone-amortisation sweep on F-MNIST-C (our primary axis).** Replicate the Vafaii Fig. 13/15/16 ablation: amortised reference (already in `common/baselines/amortised.json`) → semi-amortised (warm-started by amortised encoder, then K refinement steps) → fully iterative iG_relu-VAE-style. If the monotone trend holds on F-MNIST-C *classification accuracy*, that's our central thesis confirmed at our scale and our primary metric. iteration-for-OOD-evidence.
- **iG_relu-VAE in 3-file form.** Strip iP-VAE's spiking part — keep Gaussian latents with post-sampling ReLU and the natural-gradient update u̇ ∝ Φᵀx − ΦᵀΦ·ReLU(z) − β(u−u₀). Operator-permitted, biology-flavoured, EBM-adjacent, and explicitly shown to beat hybrid sa/ia models. Strong H001 candidate. energy-based, sparsity, biological.
- **"Inference converges beyond training horizon" as a diagnostic.** Train at T_train ∈ {8, 16, 32}; report energy-vs-step traces for T_test up to 200 on F-MNIST and on F-MNIST-C. If our PC trial *also* shows monotonically decreasing energy on OOD past its training horizon, we have a clean qualitative signature aligned with brain_like_vi — and our `convergence` guardrail enforces it natively.
- **Online / rolling-prior inference.** Even on a static image, treating the posterior at step t as the prior at step t+1 lets the leak/KL term *vanish in the single-step limit* (eq. 7). If we adopt the rolling-prior scheme our PC variant would be one of the very few in our literature corpus implementing the "online" leg of FOND. Cheap, structural, and tightly motivated. active-inference (perception-side only).
- ** β as a rate-distortion / sparsity dial.** Their β sweep traces a clean trade-off curve; ours is a 1-D sweep we should always run alongside a new model. Cheap insight per trial.
- **Compositional code as the *mechanism* of OOD generalisation.** Vafaii attributes iP-VAE's cross-domain transfer to Gabor-primitive recombination (Fig. 17). Add a feature-visualisation pass to our evaluator (top-K activated filters per OOD corruption) so we can tell when an iterative win is structural vs. accidental. generalisation-theory.

## Connections

- `[whittington_bogacz_2017]` — classical PC training framework; brain_like_vi's iG-VAE is essentially their PC with natural-gradient preconditioning and ReLU sample-nonlinearity. Direct extension.
- `[pvae_vafaii_2024]` — same lab's amortised Poisson VAE; this paper's iP-VAE is the iterative counterpart. The amortised/iterative pair within one architecture family is exactly the controlled comparison the operator wants.
- `[hybrid_pc_2023]` — sa-VAE is in spirit a hybrid PC; brain_like_vi reports the fully iterative variant beats it, refining the operator's view of the hybrid trade-off.
- `[zahid_critical_2023]` — counter-evidence on PC equivalence variants; brain_like_vi's non-equivalence iterative regime is precisely where Zahid says the variational lever survives.
- `[sdpc_boutin_2021]` — Laplace-prior sparse PC; iP-VAE's Poisson posterior is an alternative sparsity mechanism (exponential nonlinearity emerges naturally, divisive normalisation as stabiliser).
- `[mcpc_oliviers_2024]` — MCPC adopts Langevin sampling; appendix D.4 explicitly connects FOND to sampling-based inference as complementary.
- `[tschantz_curvature_2023]` — curvature-aware PC; FOND's natural-gradient prescription is the same family of ideas (Fisher preconditioning).
- `[multistep_pc_simplicity_2025]` — provides theoretical link between inference depth and representation simplicity; brain_like_vi is the empirical companion piece.
- `[ttt_sun_2020]`, `[tent_wang_2021]` — different framing (test-time fine-tuning of weights vs. test-time inference over latents) but same direction of evidence for iteration-at-test-time.
- `[hinton_cd_2002]`, `[hinton_dbn_2006]` — EBM-lineage prior; the recurrent "explaining away" term and the analysis-by-synthesis framing both descend from this lineage (eq. 7.1 in CD matches eq. 5 in brain_like_vi structurally).

## Relevant themes

iteration-for-OOD-evidence (primary), energy-based, sparsity, biological, active-inference (perception side only), generalisation-theory.

## Tier

**must-read**. This is the workspace's nearest published precedent.

## Out of scope

The spiking iP-VAE variant itself is out-of-scope for *building* (operator carve-out). The iG-VAE / iG_relu-VAE non-spiking siblings, and the iterative-vs-amortised methodology and ablations, are fully in scope. The "Poisson VAE / P-VAE" component is in scope (operator note: Poisson latents are continuous distributions over counts, not spike trains).

## Notes / follow-ups

—
