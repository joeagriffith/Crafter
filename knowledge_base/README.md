# knowledge_base/ — Crafter / pc-crafter research knowledge base (imported from pc-crafter 2026-06-27)

> **Imported from the pc-crafter project** on 2026-06-27; it backs the
> iterative-inference (predictive-coding) vs amortised (DreamerV3/IRIS)
> research program. See `.claude/skills/knowledge-base/SKILL.md` for the
> use/contribute/synthesize protocol.

A **mutable** literature knowledge base for the pc-crafter project.
Sub-agents have write access for additive contributions (new papers,
new notes, new meta-syntheses, new index entries). The orchestrator
curates and may consolidate.

This file is the **entry point**. Navigate via progressive disclosure.

## Layout

```
knowledge_base/
├── README.md                  this file — navigation entry
├── SYNTHESIS_TEMPLATE.md      schema for a per-paper synthesis
├── open_questions.md          mutable; sub-agents append
├── papers/<slug>/
│   ├── <slug>.pdf             the source paper
│   ├── synthesis.md           per-paper synthesis
│   └── notes.md               (optional) sub-agent follow-up notes
├── indices/
│   ├── by_subfield.md         technical area
│   ├── by_mechanism.md        algorithmic mechanism
│   ├── by_theme.md            project-flagged themes
│   ├── by_relevance.md        tiers (must-read / high / medium / supporting)
│   └── by_methodology.md      theory / empirical-RL / empirical-PC / survey
└── syntheses/                 cross-paper meta-reviews
    ├── 01_central_thesis_iterative_inference_faster_learning.md
    ├── 02_training_iteration_vs_test_iteration.md
    └── 03_hopfield_attention_pc_bridge.md
```

## How to navigate

1. **Read** `syntheses/01_central_thesis_iterative_inference_faster_learning.md` —
   the founding document for this workspace. The high-level synthesis of
   what we're testing and why. ~5 min.
2. **Skim** `open_questions.md` — what's currently being chased.
3. **Read** `syntheses/02_*.md` and `03_*.md` for the carry-over from
   codeca's F-MNIST iteration-for-OOD experiment — the priors we bring in.
4. **Slice** along an axis using `indices/` — whichever cuts closest to
   your direction.
5. **Open** specific paper syntheses at `papers/<slug>/synthesis.md`.
6. **Open** the PDF at `papers/<slug>/<slug>.pdf` only when the
   synthesis isn't enough.

**Synthesis before PDF.** PDFs are 10-30 pages; a well-written
synthesis is 2-4 pages. Use the PDF for the parts that matter —
usually the method section or specific results, not the whole thing.

## How to contribute (sub-agents)

- **Add a paper**: create `papers/<slug>/`, drop the PDF in, write
  `synthesis.md` using `SYNTHESIS_TEMPLATE.md`, append a row to each
  relevant index.
- **Add a note to an existing paper**: append to `papers/<slug>/notes.md`
  with `## <date> · agent-<handle>` header. Don't overwrite others'
  notes.
- **Revise a synthesis**: if you read a paper deeply and find the
  existing synthesis is wrong or incomplete, edit `synthesis.md`,
  bump the `Status: revised <N>` line, and note what you changed at
  the bottom.
- **Add a meta-synthesis**: drop a new file at
  `syntheses/<NN>_<topic>.md`. Pick the next sequential number.
- **Add an open question**: append to `open_questions.md`. The
  orchestrator prunes resolved questions periodically.
- **Extend an index**: indices are live. Add a row when you add a paper.

## What's static here

- `syntheses/01_central_thesis_iterative_inference_faster_learning.md` —
  the orchestrator revises this only on a paradigm shift. Sub-agents
  file an issue rather than editing.
- `SYNTHESIS_TEMPLATE.md` — the schema. File an issue to change it.

Everything else is sub-agent-mutable under additive discipline.

## Carry-over from codeca

This KB inherits selectively from the codeca F-MNIST iteration-for-OOD
experiment. Carry-over papers are tagged in `indices/by_relevance.md`
as `codeca-carry-over`. Most relevant carry-overs:

- `whittington_bogacz_2017` — foundational PC training
- `marino_iterative_amortized_policy_2020` — the project's load-bearing
  reference (iterative-after-amortised; was the F-MNIST star, also is
  the pc-crafter star)
- `hybrid_pc_2023` — System-1 / System-2 amortised+iterative split
- `pinchetti_benchmark_2024` — deep-PC depth-pathology paper
- `pvae_vafaii_2024` — variance-preserving Poisson sampling (the
  strongest F-MNIST OOD lever)
- `multistep_pc_simplicity_2025` — inference-depth vs simplicity bias
- The Hopfield bridge: `hopfield_1982`, `ramsauer_hopfield_2020`,
  `millidge_hopfield_2022`, etc.

Codeca-specific papers (MNIST-C, TENT/TTT, refuted-on-FMNIST: sparsity,
column-voting, Forward-Forward) were intentionally NOT carried over.

## Status snapshot

- **Papers indexed**: ~26 (5 seed Crafter/world-model papers + 21
  codeca carry-overs)
- **Per-paper syntheses**: written for the 5 seed papers + the Marino
  load-bearing reference; codeca carry-overs ship with their existing
  syntheses
- **Meta-syntheses**: 3 (01 founding + 02 + 03 carry-over)
- **Open questions**: seeded; loop adds more
- **Last orchestrator curation**: 2026-06-22 (project scaffold)
