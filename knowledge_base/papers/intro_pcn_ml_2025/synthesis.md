# Introduction to Predictive Coding Networks for Machine Learning — synthesis

- **Slug**: intro_pcn_ml_2025
- **Authors**: Stenlund, Mikko
- **Year / venue**: 2025 / arXiv preprint (cs.NE), 31 May 2025
- **Source PDF**: ./intro_pcn_ml_2025.pdf
- **Citation**: Stenlund, M. "Introduction to Predictive Coding Networks for Machine Learning." arXiv:2506.06332 (2025).
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
An onboarding tutorial for ML practitioners on Predictive Coding Networks (PCNs): canonical hierarchical generative architecture, derivation of inference and weight update rules from a single squared-error energy, locality discussion, and a worked CIFAR-10 supervised PCN in ~3.6 M parameters reaching 99.92 % top-1 on Papers-with-Code (the author admits the result is *one-shot, untuned, and on a possibly leaky split*). It is a vocabulary check, not a research paper. We care because (a) it codifies the canonical 3-equation formulation we should mimic in `trials/000_baseline/`, (b) it explicitly catalogues which probabilistic variants it omits, and that omission list is roughly our hypothesis space.

## Problem framing
Survey / tutorial entry-point in the Whittington–Bogacz lineage [`whittington_bogacz_2017`], with the Friston free-energy framing [`friston_fep_2009`] taken as background motivation. The author positions PCNs as "biologically inspired, locally updated" alternatives to backprop, and explicitly leans on Millidge et al. (Computation Graphs, 2022) and Salvatori et al. PRECO/PCX as adjacent ML-side reference points. The choice to derive everything from a *squared-error / Gaussian* energy implicitly fixes the prior structure — exactly the axis our workspace varies.

## Method
**Architecture.** $L$ latent layers $\mathbf{x}^{(l)}\in\mathbb{R}^{d_l}$ with input $\mathbf{x}^{(0)}$. Top-down weights $\mathbf{W}^{(l)}$ generate predictions:

$$\mathbf{a}^{(l)}=\mathbf{W}^{(l)}\mathbf{x}^{(l+1)},\quad \hat{\mathbf{x}}^{(l)}=f^{(l)}(\mathbf{a}^{(l)}),\quad \boldsymbol{\varepsilon}^{(l)}=\mathbf{x}^{(l)}-\hat{\mathbf{x}}^{(l)}.$$

The energy is the sum of squared prediction errors $\mathcal{L}=\tfrac{1}{2}\sum_l\|\boldsymbol{\varepsilon}^{(l)}\|^2$ — i.e. an implicit zero-mean isotropic-Gaussian likelihood at every layer.

**Inference rule** (with $\boldsymbol{\varepsilon}^{(L)}=0$ convention, $\eta_{\text{infer}}>0$):

$$\mathbf{x}^{(l)}\leftarrow \mathbf{x}^{(l)} - \eta_{\text{infer}}\Big(\boldsymbol{\varepsilon}^{(l)} - \mathbf{W}^{(l-1)\top}\big(f^{(l-1)'}(\mathbf{a}^{(l-1)})\odot \boldsymbol{\varepsilon}^{(l-1)}\big)\Big).$$

**Learning rule** (Hebbian-like, fully synapse-local):

$$\mathbf{W}^{(l)}\leftarrow \mathbf{W}^{(l)} + \eta_{\text{learn}}\big(f^{(l)'}(\mathbf{a}^{(l)})\odot \boldsymbol{\varepsilon}^{(l)}\big)\mathbf{x}^{(l+1)\top}.$$

The supervised extension clamps a readout $\hat{\mathbf{y}}=\mathbf{W}^{\text{out}}\mathbf{x}^{(L)}$ on top, with $\boldsymbol{\varepsilon}^{\text{sup}}=\hat{\mathbf{y}}-\mathbf{y}$ injected as the top-layer error during inference (equivalent to redefining $\boldsymbol{\varepsilon}^{(L)}=\mathbf{W}^{\text{out}\top}\boldsymbol{\varepsilon}^{\text{sup}}$). The synchronous pseudocode is given in three algorithm boxes (unsupervised, supervised per-sample, vectorised batched supervised) — the third is essentially what our `trials/000_baseline/run.py` should mirror.

## Key results
- CIFAR-10, $L=3$ ($d_1=1000, d_2=500, d_3=10$), ReLU, no biases, ~3.58 M params.
- $T_{\text{infer}}=50, T_{\text{learn}}=B=500, \eta_{\text{infer}}=0.05, \eta_{\text{learn}}=0.005$. 4 epochs, 4 minutes on an L4.
- **Top-1: 99.92 %, Top-3: 99.99 %.** Author flags this as a *Papers-with-Code leaderboard topper* but immediately discloses it was one-shot with no architecture or hyperparameter tuning, and that CIFAR-10 lacks a separate validation split so eval was on test directly. Take with caution — the number is mostly a vehicle for the worked example, not a benchmark claim.
- Energy trajectories (Fig. 3) show fast monotone decrease during inference then slower during learning — a useful sanity-check pattern our `convergence` guardrail can lift directly.

## Limitations
- Only Gaussian / squared-error energy. No discussion of Laplace [`sdpc_boutin_2021`], Poisson [`pvae_vafaii_2024`], Langevin-sampled [`mcpc_oliviers_2024`], or curvature-scaled [`tschantz_curvature_2023`] variants.
- No OOD or corruption evaluation. Generalisation is asserted ("excellent generalization") on the basis of i.i.d. test accuracy alone — exactly the kind of "iteration ≡ generalisation" claim our workspace wants to falsify or sharpen with F-MNIST-C.
- Convergence treated as "see refs [Frieder & Lukasiewicz 2022, Salvatori PCX 2024]"; no inference-step phase-transition or amortisation-vs-iteration ablation.
- Discriminative readout glued on as a post-hoc clamp; no comparison against a fully-generative classification scheme (e.g. NGC [`ngc_ororbia_2022`]).
- Z-IL / Backprop-equivalence variants ([`millidge_backprop_2020`, `zahid_critical_2023`]) cited only in passing; the "non-equivalence regime" our spec targets isn't called out as the load-bearing axis.

## Idea seeds for our loop
- **Adopt this paper's algorithm 3 verbatim as the `trials/000_baseline/` reference.** It is the cleanest published Gaussian PCN pseudocode in the literature, with the synchronous-snapshot semantics already spelled out — saves us re-deriving and avoids the gotchas Frieder & Lukasiewicz formalise. *(supporting)*
- **The Gaussian-only framing of this survey is the gap we exploit.** Every probabilistic variant in our cluster (Poisson / Laplace / Langevin / curvature-Laplace) is a one-line drop-in to the energy or the inference rule above. The survey omits these — that omission *is* the headroom. *(energy-based, sparsity, biological)*
- **Survey's CIFAR-10 result is suggestive of in-dist over-fit, not generalisation.** The 99.92 % on a leaky split, untuned, with $T_{\text{infer}}=50$, is exactly the kind of number that motivates moving to a corruption-based eval like F-MNIST-C and reporting *gap-over-amortised* rather than absolute accuracy. *(iteration-for-OOD-evidence)*
- ****Their separation-of-timescales heuristic ($T_{\text{infer}} \gg T_{\text{learn}}$ effectively, with 50 inference steps per sample) is one knob worth treating as a primary axis** rather than a fixed hyperparameter — connects to `multistep_pc_simplicity_2025` and the "inference-step phase transition" open question #4 in `open_questions.md`. *(generalisation-theory, iteration-for-OOD-evidence)*
- ****Hybrid PC [`hybrid_pc_2023`] is the only modern variant the author calls out by name** and then explicitly excludes from the rest of the document. That carve-out telegraphs where the field's next moves are — the survey author saw hybrid PC as the cleanest deviation worth flagging, which validates our prior weighting on it.

## Connections
- `[whittington_bogacz_2017]` — the canonical training-side foundation the survey re-derives; survey's locality discussion is essentially a digest of theirs.
- `[friston_fep_2009]` — cited as the free-energy framing; survey does not explicitly invoke variational-bound language, treating the energy as a sum-of-squares loss instead. Our workspace needs to be explicit about the variational interpretation since it is what licenses the OOD claim.
- `[hybrid_pc_2023]` — the only modern variant the survey flags by name (Tschantz et al. 2023). The survey explicitly defers it.
- `[millidge_backprop_2020]` — referenced as "Millidge et al. PC approximates backprop along arbitrary computation graphs." Survey treats this as a feature; per `[zahid_critical_2023]` and our central thesis, we treat it as the *boundary* we don't want to cross.
- `[salvatori_reverse_2022]` / `[jpc_innocenti_2024]` — cited via Salvatori et al. 2024 (PCX, "stable, fast, fully automatic PCN") and Mali/Salvatori/Ororbia 2024 (convergence bounds). The survey leans on these for convergence guarantees without explaining them.
- `[ngc_ororbia_2022]` — survey is silent on NGC's discriminative-from-generative scheme; gap worth filling.
- `[pvae_vafaii_2024]`, `[mcpc_oliviers_2024]`, `[sdpc_boutin_2021]`, `[tschantz_curvature_2023]` — **not cited.** The survey predates or ignores the probabilistic-variant line. Our cluster's whole thesis is in this blind spot.
- `[bastos_microcircuits_2012]` — cited as biological motivation; survey does not adopt microcircuit constraints architecturally.
- `[pinchetti_benchmark_2024]` — **not cited.** Survey gives no benchmark context.

## Relevant themes
energy-based / biological / generalisation-theory (implicit — the survey conflates i.i.d. accuracy with generalisation)

## Tier
supporting

## Out of scope
The survey briefly references "Predictive coding with spiking neural networks: a survey" (N'dri et al. 2024, ref [22]) as a sibling tutorial. Spiking PC is **out of scope** per operator carve-out (`spec.md` §"Out-of-scope hypothesis directions"). The survey itself does not build spiking models, so it is in scope; only that one citation pointer is not. No active-inference action-selection content.

## Notes / follow-ups
- 2026-06-22, agent-20260622: The 99.92 % CIFAR-10 number should be treated as anecdote, not benchmark — the author's disclosure paragraph is unusually candid (one-shot, untuned, no val split, two-run hyperparameter swap mid-experiment). If anyone cites this paper as evidence that "PCNs match ViT-H/14," push back: the comparison is unsound. The paper is valuable as a vocabulary-and-pseudocode reference, not as a results claim.
- 2026-06-22, agent-20260622: The paper's repo (Monadillo/pcn-intro on GitHub) is a plausible pull-target for our `trials/000_baseline/` per the workspace's pull-first baseline philosophy.

—
