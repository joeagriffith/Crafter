# Per-paper synthesis template

When you (sub-agent or synthesis swarm) write a paper's
`synthesis.md`, follow this schema. Keep each section terse —
the synthesis should be 1-2 pages, not a re-read of the paper.

## Length target

~600-1000 words, with the "Idea seeds for our loop" section being
the load-bearing part. A reader should be able to skim the synthesis
in 2 min and find at least one actionable idea without opening the
PDF.

---

```markdown
# <Paper title> — synthesis

- **Slug**: <slug>
- **Authors**: <last, first; last, first; ...>
- **Year / venue**: <YYYY / venue>
- **Source PDF**: ./<slug>.pdf
- **Citation**: <full citation, single line>
- **Synthesized by**: agent-<YYYYMMDD>
- **Status**: initial

## TL;DR
<3-5 sentences. What problem did they tackle, what did they propose,
what did they find, why do we care.>

## Problem framing
<What problem are they solving? In what tradition? Cite the 1-3 prior
works they build on, by slug if also in our knowledge_base, by
external citation otherwise.>

## Method
<The technical core. Mathematical or algorithmic form in 1-3
paragraphs. Be specific — name the prior, the energy, the update
rule, the architecture. Avoid hand-waving like "PC-style updates."
If a key equation matters, transcribe it.>

## Key results
<Numbers if available, qualitative findings otherwise. Reproduce the
essence of 1-3 tables or figures in compact form. Specify datasets +
baselines they compared against.>

## Limitations
<What's missing or under-specified in this paper. What didn't
replicate. What they didn't test. What the authors flagged as future
work. Be honest — limitations are useful for our hypothesis design.>

## Idea seeds for our loop
<**The load-bearing section.** What ideas from this paper could
become a hypothesis in our workspace? Be specific. Examples of useful
seeds:
- "Their X mechanism + our Y prior would be a clean two-variable
  hypothesis."
- "They didn't measure OOD; we should — their method might shine or
  break."
- "Their failure mode at Z suggests an alternative formulation worth
  testing."
- "Their hyperparameter ablation suggests N is the load-bearing knob;
  we should sweep it more aggressively."
- "They cite [other slug] as prior; the gap between those is unfilled."

Each idea seed: one-sentence claim + brief reasoning. Tag with our
operator-flagged themes (sparsity, competition, biological,
active-inference, energy-based) when applicable.>

## Connections
<Other papers in `knowledge_base/papers/` this paper bears on. Use
`[slug]` notation:
- `[salvatori_associative_2021]` — cited heavily; method extends theirs
- `[brain_like_vi_2024]` — same OOD-via-iteration thesis at different scale
- `[zahid_critical_2023]` — counter-evidence on the equivalence claim
...>

## Relevant themes
<One or more of: sparsity / competition / biological / active-inference
(perception side) / energy-based / iteration-for-OOD-evidence /
generalisation-theory / counter-evidence. Pick all that apply.>

## Tier
<must-read / high / medium / supporting / counter-evidence>

Choose based on:
- **must-read** — foundational; pretty much every hypothesis needs to
  internalise this.
- **high** — load-bearing for the thesis or the operator's flagged
  themes; sub-agents working in this neighbourhood should read.
- **medium** — useful when you happen to be near this niche.
- **supporting** — context, background, framing.
- **counter-evidence** — the paper argues against part of our thesis
  or a popular intuition. Critical to read before betting on the
  intuition.

## Out of scope
<Mark this section "—" if the paper has no out-of-scope content
relative to our loop. Mark explicitly if relevant, e.g.:

"This paper is partially about spiking implementation, which the
operator has carved out as build-out-of-scope (see `spec.md`§Themes).
Ideas about timing / dynamics are still in scope; the spiking
substrate itself is not.">

## Notes / follow-ups
<Free space. Sub-agents who engage with the paper may append below.
Each addition prefixed with date + agent handle. Earlier
contributions are preserved.>

—
```

## Tagging conventions

For consistency across syntheses:

- **Themes** match the list in `spec.md`§"Themes the operator wants
  biased toward."
- **Tier** uses one of the 5 values above. When in doubt, prefer
  "medium" — it's the default.
- **Connections** use slug brackets `[other_slug]`. When linking to a
  paper not yet in `knowledge_base/papers/`, use the full citation.
- **Idea seeds** that the author of the synthesis believes are
  particularly strong may be marked with a `**` prefix; the
  orchestrator can use this to surface them in `open_questions.md`.
