# 03 — The Hopfield ↔ attention ↔ predictive coding bridge

- **Authored**: literature-supplement sub-agent, 2026-06-22
- **Status**: initial (navigational; orchestrator may rename / merge)

This is a navigational synthesis, not a curated reference. The
operator (Joe) flagged that the knowledge base was missing the
Hopfield side of an important three-way bridge that the loop has
been quietly using all along. This synthesis points at the
connections.

## The three-way bridge

> **Hopfield fixed-point dynamics, transformer attention, and PC
> iterative inference are three faces of the same mathematical
> object: iterated optimisation on a structured energy.**

The papers that pin down each pairwise bridge:

- **Hopfield ↔ PC**: [`hopfield_1982`], [`hopfield_tank_1986`] (the
  energy-descent template) bridge directly to PC inference via
  [`whittington_bogacz_2017`] (PC inference IS gradient descent on
  free energy) and especially [`salvatori_associative_2021`] (a
  generative PCN's training images become attractors of the
  inference dynamics — i.e., a PCN is literally a Hopfield network
  with a structured, learned, hierarchical energy).
- **Hopfield ↔ attention**: [`krotov_hopfield_2020`] and
  [`ramsauer_hopfield_2020`] (simultaneously, 2020) prove the
  identity: continuous Hopfield retrieval with softmax memory layer
  IS one step of transformer self-attention. β = 1/√d. Keys = values
  = stored patterns. Single attention step = single Hopfield update.
- **Attention ↔ PC**: [`millidge_hopfield_2022`] — Millidge & Bogacz
  are PC researchers; they unified all of Hopfield, DAM, modern
  Hopfield, attention, and SDM under one similarity-separation-
  projection schema. Their paper is the bridge **from inside the PC
  community**.

The chain composes: PC inference is iterated descent on a
free-energy; the free-energy has structural pieces that look like
Hopfield retrieval; one step of Hopfield retrieval IS one attention
operation; therefore **iterated attention with the right energy IS
PC inference**.

## Why this matters for the loop

The central thesis (`syntheses/01_central_thesis_iteration_for_OOD.md`)
claims that **iterative variational inference is the algorithmic
mechanism that buys OOD generalisation**. The Hopfield bridge tells
us that transformer attention is *one step* of this mechanism. So:

- Standard transformers are doing one Hopfield retrieval step per
  layer (feedforward).
- A "PC-attention" architecture would *iterate* the attention step
  until convergence — i.e., do many Hopfield retrievals at the same
  layer until the basin is reached.
- This is exactly the "iteration as inference" lever the central
  thesis predicts buys OOD robustness.

Tested by:

- Universal Transformers (Dehghani 2018) iterate attention at the
  same layer — but they don't frame it as energy descent and don't
  test OOD systematically.
- Deep Equilibrium Models (Bai 2019) iterate to a fixed point — but
  the counter-evidence (arXiv:2306.01429) says DEQs don't get OOD
  gains, **because the iteration isn't doing variational
  optimisation**. Iteration on a non-variational energy isn't
  enough.

The frame from this synthesis: **iterated attention WITH an explicit
free-energy interpretation should buy OOD gains**. Untested in this
clean form.

## How this bears on the active hypotheses

### H010 (input-conditioned learned-precision attention on PE)

H010 was just spawned (2026-06-22). The architecture proposed:
input-conditioned softmax temperature β(x) over prediction errors
in a deep PCN.

**Per the bridge, H010 IS one step of modern Hopfield retrieval
embedded in PC inference.** Specifically:

- Similarity = prediction error magnitude (or signed PE inner
  product against learned templates).
- Separation = softmax with input-conditioned temperature β(x).
- Projection = weighted combination back into the latent.

This is exactly the `softmax(β · ⋅)` retrieval step of
[`ramsauer_hopfield_2020`] / [`krotov_hopfield_2020`] /
[`millidge_hopfield_2022`]. **H010's mechanism has a 40-year-old
theoretical foundation.** This is good news — the prediction is that
H010 should buy OOD because variance-weighted retrieval IS the
attention-as-Hopfield retrieval step, and iterating that step is
iterative variational inference.

The sub-agent attacking H010 should read, in order:

1. [`millidge_hopfield_2022`] — establishes the framework and the
   PC connection.
2. [`ramsauer_hopfield_2020`] — concrete softmax-attention =
   Hopfield identity with the cleanest equations.
3. [`krotov_hopfield_2020`] — alternative derivation; suggestive
   two-layer feature+memory architecture.

### Other hypotheses bearing on this bridge

- **H007 (column-voting, REFUTED on F-MNIST)**: GLOM-style subspace
  voting was the column-voting variant — i.e., attention-style
  agreement-finding. The H007 result that voting-matrix M doesn't
  learn on F-MNIST is *partially* explained by this synthesis: GLOM
  attention is one Hopfield step, but without iteration over an
  *energy* it's just unrolled compute — and the counter-evidence
  on DEQs says that's not enough. **Open**: did H007 fail because
  the energy structure was missing, not because column-voting is
  wrong?
- **H008 (Forward-Forward) / H009 (active inference perception)**:
  Both are alternative-training schemes; the bridge says they
  *could* be reframed as Hopfield-style retrieval, but neither
  uses softmax-attention machinery directly.
- **Future**: a sub-agent inspired by this synthesis could explicitly
  build a "Hopfield-PC" — a PCN whose top layer is a continuous
  dense AM with learnable prototype dictionary, and whose inference
  iterates the attention step. Strong candidate for a follow-up to
  H010.

## Limits of the bridge

- The Hopfield-attention identity holds at *one step*; multi-step
  iteration of attention is mathematically clean but empirically
  under-explored. Universal Transformers and DEQs are partial probes.
- Hopfield retrieval stores patterns in *one* large lookup; PC builds
  a hierarchical generative model. The bridge unifies the retrieval
  step but leaves the generative-model structure of PC as a richer
  story than classical Hopfield.
- [`salvatori_associative_2021`] empirically shows that PCN beats
  modern continuous Hopfield on natural-image retrieval. The bridge
  is not "PC = Hopfield" — it's "PC contains Hopfield-retrieval-like
  steps as part of a richer iteration." PC's hierarchical generative
  model is the additional structure.
- [`zahid_critical_2023`] warns that PC variants engineered to be
  backprop-equivalent abandon the variational interpretation. The
  same warning applies here: attention with a soft prior is one
  thing; attention iterated on a non-variational pseudo-energy is
  not the thesis's mechanism.

## What to do with this synthesis

This is a navigational piece. When a sub-agent is:

- Designing an attention-flavoured PCN → start here, then the three
  modern-Hopfield papers.
- Doubting whether H010 has theoretical grounding → start here.
- Asked "is the central thesis just a restatement of universal
  transformers / DEQs?" → start here. The answer is no, because
  those don't iterate on a variational energy.

The orchestrator may rename or merge this synthesis into one of the
other two as the loop matures. For now, it's an explicit pointer to
a bridge that was implicit in the earlier syntheses.

## Connections

- `syntheses/01_central_thesis_iteration_for_OOD.md` — the loop's
  bet; this synthesis explains the Hopfield/attention slice of it.
- `syntheses/02_training_iteration_vs_test_iteration.md` —
  complementary: this synthesis is about retrieval-step iteration;
  02 is about training-time vs test-time iteration.
- `papers/hopfield_1982/`, `papers/hopfield_tank_1986/` — foundations.
- `papers/krotov_hopfield_2016/`, `papers/krotov_hopfield_2020/`,
  `papers/ramsauer_hopfield_2020/`, `papers/millidge_hopfield_2022/`
  — the modern bridge papers.
- `papers/salvatori_associative_2021/` — PC-side empirical bridge;
  PCN-as-associative-memory at ImageNet scale.
- `papers/hinton_glom_2021/` — GLOM's attention-driven island-forming
  is an architectural cousin.

— End of synthesis 03.
