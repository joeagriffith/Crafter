---
name: knowledge-base
description: >-
  Work with the project's literature knowledge base (`knowledge_base/`): find
  prior work for a research question, ingest/synthesize a new paper, or run a
  cross-paper synthesis that detects patterns (contradictions, gaps, untested
  combinations) and turns them into novel, testable research directions. Use
  when reviewing literature, adding a paper, distilling an experiment finding
  back into the KB, or turning the corpus into research hypotheses to test
  against the baselines.
---

# knowledge-base

A **living research knowledge base** (`knowledge_base/`) that pools literature
and synthesizes it into *novel, testable research directions* — for the
iterative-inference / predictive-coding vs amortised (DreamerV3 / IRIS) program
on Crafter, and model-based RL more broadly. It is not an archive; it is a
machine for turning papers into experiments.

It has three jobs: **find** prior work, **add** papers, and — the point —
**synthesize across papers** into research.

> **Core principle — synthesis before PDF.** A PDF is 10–30 pages; a good
> `synthesis.md` is 1–2. Read syntheses + indices first; open a PDF only for the
> specific method or result the synthesis can't give you (usually one section).

## Layout — navigate via `knowledge_base/README.md`

```
knowledge_base/
├── README.md              navigation entry (read it for the full map)
├── SYNTHESIS_TEMPLATE.md  schema for a per-paper synthesis (follow it exactly)
├── open_questions.md      live; what we're chasing — append, don't overwrite
├── papers/<slug>/         <slug>.pdf + synthesis.md + (optional) notes.md
├── indices/               by_mechanism / by_theme / by_methodology /
│                          by_relevance / by_subfield — cross-cutting slices
└── syntheses/             cross-paper meta-reviews (01 = founding thesis)
```

Slug convention: `firstauthor_topic_year` (e.g. `hafner_dreamerv3_2023`).

---

## 1. USE — answer a research question from the KB

Progressive disclosure, cheapest first:

1. **Read the founding synthesis** `syntheses/01_*.md` — the thesis we're testing.
2. **Skim `open_questions.md`** — what's already live; don't re-derive it.
3. **Slice** via the `indices/` file closest to your question (by_mechanism for
   "who does iterative updates how", by_theme for operator-flagged themes, etc.).
4. **Read the relevant paper syntheses** — focus on their *Idea seeds* and
   *Connections* sections.
5. **Open a PDF** only when the synthesis is insufficient.

Output a focused answer that cites `[slug]`s. If you surfaced a gap or idea,
either append it to `open_questions.md` or trigger a synthesis (§3).

## 2. ADD — ingest a paper

1. `mkdir knowledge_base/papers/<slug>/`, put `<slug>.pdf` in it (fetch it if
   needed; PDFs are gitignored — local only).
2. Write `synthesis.md` following `SYNTHESIS_TEMPLATE.md` — **terse, 600–1000
   words.** The **"Idea seeds for our loop"** section is load-bearing: each seed
   = one-sentence claim + brief reasoning, tied to *our* program (does this
   method / prior / failure mode suggest a hypothesis to test against the
   DreamerV3 baseline?).
3. **Add a row** to each relevant index (`by_mechanism`, `by_theme`,
   `by_relevance` tier, `by_methodology`, `by_subfield`).
4. If the paper raises or answers a question, update `open_questions.md`.

**Quality bar for a synthesis:** a reader skims it in 2 min and leaves with ≥1
actionable idea *without opening the PDF*. Name the actual mechanism (the energy,
the update rule, the architecture) — no hand-waving like "PC-style updates".
Honest *Limitations*. Real `[slug]` *Connections*.

## 3. SYNTHESIZE ACROSS PAPERS — the frontier engine

This is how the KB produces *novel research* rather than storage. A cross-paper
synthesis (`syntheses/NN_<topic>.md`, next sequential number) mines the corpus
for a pattern that **no single paper states**.

**Run one when** you have ≥3 related syntheses, a recurring tension, or an
`open_questions.md` entry the literature can now address.

**Method:**

1. **Frame** the question/tension in one sentence.
2. **Gather** the relevant syntheses (slice an index). Read their *Method*,
   *Key results*, *Limitations*, *Idea seeds*, *Connections*.
3. **Mine for patterns** — explicitly hunt for:
   - **Contradictions** — paper A claims X, paper B refutes or bounds it. *The
     most generative finding: a contradiction is an experiment.*
   - **Unfilled gaps** — A cites B; the obvious step between them is untested.
   - **Under-explored combinations** — method from A + prior/regime from B = a
     clean two-variable hypothesis.
   - **Convergences** — ≥3 papers independently flag the same load-bearing
     factor → a real lever (or a shared blind spot worth attacking).
   - **Scale / transfer gaps** — proven at scale/domain N, untested at M (e.g.
     iteration-for-OOD shown on F-MNIST, untested on Crafter sample-efficiency).
4. **Write the meta-synthesis** — state the pattern sharply, cite `[slug]`s,
   and distill into **idea seeds → testable hypotheses → open questions.** Mark
   especially strong seeds with `**` so they surface to `open_questions.md`.
5. **Promote** the best hypotheses: append them to `open_questions.md`; if one
   is ripe, scaffold an experiment (`cp -r experiments/_template ...`) to test it
   against the DreamerV3 baseline (12M 8.2 / 25M 10.3 / 50M 10.6 return; use
   `crafter_rl/eval.py` for return + Crafter score).

**Rubric for a *good novel direction*** (elegance = maximal insight, minimal
apparatus):

- **Sharp** — a falsifiable claim, not a vibe. *"Iterative inference at multiple
  depths with timescale separation beats an amortised world model at matched
  compute on Crafter sample-efficiency"* is testable; "PC is promising" is not.
- **Matched** — the comparison isolates one variable (matched params / compute /
  eval protocol). An unmatched win teaches nothing.
- **Non-redundant** — check the corpus + `open_questions.md`: is it already done
  or refuted? Cite the `counter-evidence`-tier papers before betting.
- **Load-bearing** — targets a factor multiple papers flag, or resolves a real
  contradiction.
- **Frontier** — plausibly unpublished; the gap is real, not merely unread.

## 4. CLOSE THE LOOP — experiments feed the KB back

When an experiment yields an actionable finding: distill the *generalisable*
claim into `syntheses/` (new file or amend an existing one), reflect
paper-specific clarifications in `papers/<slug>/notes.md`, and **prune** entries
the finding contradicts. The corpus should get *sharper* over time, not just
bigger — that's what keeps it a research instrument and not a graveyard.

## Contribution discipline (additive)

- **Append, don't overwrite.** Notes and `open_questions.md` additions get a
  `## <YYYY-MM-DD> · <your-handle>` header; preserve earlier contributions.
- **Revise a synthesis** only after reading the paper deeply and finding it
  wrong/incomplete — bump `Status: revised <N>` and note what changed at the
  bottom.
- **Meta-syntheses** take the next sequential `NN`.
- **Indices are live** — add a row whenever you add a paper.
- **The founding synthesis (`01`) and `SYNTHESIS_TEMPLATE.md`** change only on a
  paradigm shift — propose it (open question / note), don't silently edit.

## Tagging (keep consistent across the KB)

- **Themes:** sparsity / competition / biological / active-inference /
  energy-based / iteration-for-OOD-evidence / generalisation-theory /
  counter-evidence.
- **Tiers:** must-read / high / medium / supporting / counter-evidence (default
  to *medium* when unsure).
- **Connections:** `[other_slug]` for in-KB papers; full citation otherwise.
