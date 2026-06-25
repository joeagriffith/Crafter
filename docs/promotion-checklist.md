# Prototype → Promote Checklist

Code matures in three stages: **`prototypes/`** (scratch) → proven in an
**`experiments/<dated>/`** trial → **`crafter_rl/`** (shared, imported everywhere).
Promote only when a thing has earned permanence; until then, keep it in an experiment.

## When to promote
A component is a promotion candidate when it is reused (or clearly will be) across more than
one experiment, AND it has proven correct on a real run.

## Acceptance criteria (all must pass)
- [ ] **Conforms to the shared interface** in `crafter_rl/` (e.g. an env is a `CrafterGym`-style
      `gymnasium.Env`; an agent exposes the common `train`/`predict` surface).
- [ ] **Smoke test passes**: a short run completes clean, e.g.
      `uv run python <trial>/train.py --steps 2000 --no-wandb` → exits 0, writes a
      schema-valid `metrics.json`.
- [ ] **Reproduces a known baseline number** (within noise) — link the W&B run / ledger line.
- [ ] **Imported, not copy-pasted**: callers do `from crafter_rl... import ...`; no duplicated logic.
- [ ] **Lint / type-check clean.**
- [ ] **Docs updated**: mention it in [repo-structure.md](repo-structure.md) and, if it adds a
      convention, the relevant doc.

## Mechanics
1. Move the file(s) into `crafter_rl/` with `git mv` to **preserve history** (once the repo
   has commits).
2. One small commit per structural change (move, rename, delete legacy) — keep diffs reviewable.
3. Replace the experiment's local copy with an import from `crafter_rl/`; the originating
   trial becomes the reference example in that module's docstring.
4. Delete the now-dead prototype/legacy files — don't leave two copies.

## What NOT to promote
- One-off experiment glue, plotting scratch, anything used by a single trial.
- Unproven ideas — those stay in `prototypes/` until an experiment validates them.
