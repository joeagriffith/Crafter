# 02 — Training-time iteration vs test-time iteration: the H002 reframing

- **Authored**: orchestrator, 2026-06-22 (post-H002)
- **Status**: live (revisable by orchestrator when more evidence arrives;
  sub-agents file an issue rather than editing)

A follow-up to
[`01_central_thesis_iteration_for_OOD.md`](01_central_thesis_iteration_for_OOD.md).
H002 produced a clean 2×2 ablation that pressures the central thesis as
originally stated. The thesis isn't dead — but it's mis-localised. This
synthesis captures the revised frame so future sub-agents start from the
correct picture.

## The H002 result

A matched-parameter (1,590,010), matched-architecture, matched-data 2×2
factorial isolated two factors:

| Train regime  | Hidden noise  | Classify mode | F-MNIST-C | Trial |
|---------------|--------------|---------------|-----------|-------|
| CE backprop   | none          | feedforward   | 0.5763    | 014   |
| CE backprop   | Langevin σ=0.04 anneal=0.75 | feedforward | 0.5761 | 011   |
| PCN T=20      | none          | feedforward   | 0.5694    | 017   |
| **PCN T=20**  | **Langevin σ=0.04 anneal=0.75** | **feedforward** | **0.5872** | **016** |
| PCN T=20      | Langevin (same)               | **iterative T=20** | 0.5826 | 009 (v9) |

Amortised reference: 0.5878. H000 (PCN, no Langevin, iterative classify):
0.5693.

Three observations are load-bearing:

1. **Langevin alone (011 vs 014):** -0.02 pp. Adding hidden-state Langevin
   noise to a vanilla backprop-trained MLP does effectively nothing.
2. **PCN training alone (017 vs 014):** -0.69 pp. Training under PCN's
   iterative-inference loss but without Langevin noise is *worse* than
   backprop CE training.
3. **Both together (016 vs 014):** +1.09 pp on OOD. Above amortised by
   only -0.06 pp, but achieved by FEEDFORWARD classify. The interaction
   between PCN training and Langevin noise is non-additive: each alone
   is a no-op or negative; together they shift the representation in a
   way the simple sum cannot predict.
4. **Test-time iteration HURTS the same weights (009 vs 016):** -0.46 pp.
   With identical weights trained under PCN+Langevin, running iterative
   inference at test time *underperforms* running a single feedforward
   pass. The iteration during test removes 0.46 pp of OOD accuracy.

## What the original thesis claimed, and what survives

The original central thesis ([`01_central_thesis_iteration_for_OOD.md`]):

> Iterative inference — gradient descent on a variational posterior /
> energy / free-energy at test time — is the algorithmic mechanism that
> lets neural representations generalise beyond their training distribution.

**What survives:** iteration *as a mechanism* is load-bearing. PCN
training (which is gradient descent on a free-energy with iterated
inference at each batch) shapes representations that generalise better
than backprop CE — *but only when paired with Langevin noise*. The
variational-gradient frame remains the right lens on what's happening
during training.

**What doesn't survive:** the locus is the wrong layer. Iterative
inference at *test time* is not the lever — it's either a no-op (on
H000-style weights, per H001/v4 TENT) or actively negative (on
PCN+Langevin weights, per H002's 016 vs 009). Inference iteration at
training shapes the weight matrices; once the matrices are frozen, a
single feedforward pass extracts the representation more accurately.

## The two-fingerprint mystery

Trial 016 (PCN+Langevin, FF classify) and trial 009/v9 (same weights,
iterative classify) have OPPOSITE per-corruption fingerprints:

| Corruption type      | v9 (iter) vs amortised | 016 (FF) vs amortised |
|----------------------|------------------------|------------------------|
| rotate               | flat                   | **+3.4 pp**            |
| zigzag               | flat                   | **+4.5 pp**            |
| defocus_blur         | flat                   | **+6.1 pp**            |
| motion_blur          | flat                   | **+3.3 pp**            |
| gaussian_blur        | flat                   | **+4.3 pp**            |
| gaussian_noise       | **+6.6 pp**            | flat / negative        |
| impulse_noise        | **+5.0 pp**            | -6.6 pp                |
| brightness           | **+3.9 pp**            | flat                   |
| contrast             | **+5.3 pp**            | -2.2 pp                |
| fog                  | **+4.6 pp**            | -3.7 pp                |

The iterative classify selectively activates a *noise/intensity-robust*
representation; the feedforward classify selectively activates a
*geometric/blur-robust* representation. Same weights, two readouts.

**Hypothesised mechanism (speculative — sub-agents should test):**

- Iterative inference at test pulls latents toward the energy basin
  most consistent with the input. On noisy inputs, this means averaging
  over the noise toward the deterministic mode of the prior — denoising.
- Feedforward classify reads the first-pass encoder output. On
  geometrically-perturbed inputs, this first-pass output is the most
  "literal" reading of the image structure; subsequent iteration would
  pull it toward a clean-data mode the encoder has *learned*, but that
  mode may be the wrong one for a geometric perturbation.

This is conjecture. The HIGH-EV experiment is to engineer a system that
exposes both representations.

## The new active hypothesis space

Reframed from the original synthesis's five directions:

1. **Fusion of FF and iterative readouts.** Single model trained PCN +
   Langevin. At test, combine FF logits and iterative-classify logits
   (mean, learned mixture, gated by corruption type, etc.). Plausible
   ceiling: 0.60+ if the fingerprints compose.

2. **Mixed-T training.** Vary T_infer at training time (sometimes T=1,
   sometimes T=20). Plausibly produces a single set of weights whose
   FF representation captures both fingerprints. Cheap to test.

3. **T-curriculum.** Anneal T from large to small (or vice versa) over
   training. May escape the in-dist-floor vs OOD Pareto seen at fixed T
   (T=1 → 0.6093 OOD but 0.85 clean invalid; T=20 → 0.5872 valid).

4. **Operator-flagged themes remain UNDER-EXPLORED.** Sparsity,
   column-voting, P-VAE, Forward-Forward. None have been tried as
   primary hypotheses; H001 attempted sparse-PC + lateral inhibition
   but pivoted away from custom architectures because the in-dist floor
   was too tight. With H002's lever (PCN+Langevin produces meaningful
   gains), the floor may now be easier to clear with these themes.
   When the loop next stagnates, these are the high-leverage move.

5. **Score-based / denoising-style PC.** Still an empty literature
   space. The H002 mechanism (training-time iteration on a free-energy)
   is conceptually close to score matching's training-time iterative
   refinement. May port across.

## Out-of-scope (operator carve-outs unchanged)

- **Spiking NN — do not build.** (P-VAE is in scope; see
  [`01_central_thesis_iteration_for_OOD.md`].)
- **Active-inference action-selection / planning.** Perception-as-
  inference parts only.

## How sub-agents should use this synthesis

This sits next to `01_central_thesis_iteration_for_OOD.md` — not above
it, not replacing it. The original synthesis defines the variational-
gradient frame; this one localises where the gradient actually buys
generalisation (training, not test). When designing a hypothesis:

1. Is the lever you're proposing acting on the training side
   (representation shaping) or the test side (inference refinement)?
   If the latter, the H002 priors are heavily against you — expect
   no-op or negative results. To make a test-time lever work, you need
   to argue *why* it would (a) not be a no-op on BP-equivalent weights
   and (b) not actively underperform a single feedforward pass on the
   non-BP-equivalent weights.

2. If you're proposing a training-side lever, does it interact with
   either of the H002 ingredients (iterated inference + Langevin noise)?
   The interaction at training is non-additive — recombinations are
   high-EV territory.

3. Does the move plausibly compose the FF-robustness and iter-robustness
   fingerprints, or does it produce one to the exclusion of the other?
   The fingerprint complementarity is the most underexplored axis the
   loop has surfaced so far.

---

## Revision 2026-06-22 (post-H003) — the two-fingerprint contrast was largely RNG noise

H003 ran an exhaustive ensemble-of-readouts sweep on the v9 weights (14
combiners; mean, max, soft-vote, weighted, learned, gated, per-class
temperature, …) and **none beat the FF-alone classify**. The follow-up
diagnostic uncovered why: a `quick_eval` call inside several trials'
training loops iterates the test dataloader, perturbs the CPU RNG state,
and shifts the final F-MNIST-C accuracy by ~+1 pp. The same code with
`quick_eval` removed (trial 028, seed=42) reaches **0.5994 vs trial 016's
0.5872** — a +1.22 pp shift attributable to nothing but RNG.

This means:

1. **The "FF vs iter classify +0.46 pp" gap and the per-corruption
   fingerprint contrast were inside variance noise.** Both classify
   modes operate on the same weights; the apparent specialisation
   (noise-robust under iter, geometric-robust under FF) was a sampling
   artifact of which run produced which leaderboard entry, not a clean
   mechanistic split.
2. **The lever (training-time PCN + Langevin) is STRONGER than the
   original H002 estimate.** Trial 028 — same hyperparameters as v9,
   FF classify, RNG-isolated — clears amortised by +1.16 pp on
   F-MNIST-C with the in-dist floor held. This is the first PC-family
   trial above the amortised reference on this workspace's primary
   axis.
3. **The mechanism question remains open.** What does training-time
   PCN iteration *with* Langevin noise actually do to the representation
   that backprop-CE + Langevin doesn't? The two are matched in
   architecture, parameter count, and training data — yet the
   F-MNIST-C gap is ~+2.3 pp. Open candidates: noise-coupled iterated
   inference acts as a Tikhonov-style data-dependent regulariser on the
   *encoder Jacobian*; PCN's local error gradients produce a flatter
   loss landscape than backprop CE; iterated inference during training
   approximates a Hamiltonian-Monte-Carlo step that averages over a
   neighbourhood of weight configurations. None has been instrumented.

## What now survives, restated

The variational-gradient mechanism remains the right frame (synthesis 01
is intact on that). The localisation is **purely training-time** —
post-H003, there is no surviving evidence that test-time iteration adds
anything on top of the trained representation. Sub-agents proposing
test-time-iteration levers should treat this as a near-decisive prior
unless they have a mechanism for why their lever would behave differently
from H001/v4 (TENT), H002's 015 (test-time iteration on amortised), or
H003's logit-fusion (the FF and iter heads agree where they're correct).

The HIGH-EV open territory:

- **Multi-seed variance characterisation.** Every claim made by the loop
  rests on whether the effect is above ~±1 pp variance noise. A cheap
  3-5-seed replicate of trial 028 anchors all future claims.
- **Mechanism instrumentation on trial 028 vs 011 (matched-MLP + Langevin):**
  encoder-Jacobian norms, loss-landscape curvature, weight-distribution
  statistics. What does the +2.3 pp F-MNIST-C lift correspond to in the
  parameter geometry?
- **Layering operator-flagged themes on top of the now-working lever.**
  Sparse activations, column-voting, P-VAE Poisson latents, Forward-
  Forward goodness — all so far untested as primary hypotheses, and
  all now layer onto a working baseline (0.5994) rather than a losing
  one.
- **Score-based / denoising-style PC.** Still an empty literature space
  and conceptually adjacent to training-time iteration on a free-energy.

The pivot from "iterative inference at test" → "iterated inference at
training" is the largest reframe the loop has produced. The synthesis
should not be cleaned up further until the multi-seed replicate either
confirms (treat as load-bearing) or breaks (reopen the synthesis) the
trial 028 result.

---

*Update history:*
*  2026-06-22 (initial) — authored post-H002, claiming the test-vs-train iteration distinction was a clean mechanism.*
*  2026-06-22 (revised) — H003 dissolved the FF-vs-iter fingerprint contrast as RNG-variance noise, but RE-AFFIRMED the underlying lever (training-time PCN+Langevin) more strongly: trial 028 = 0.5994 clears amortised by +1.16 pp.*
*  2026-06-22 (re-revised post-H004) — 3-seed bracket of the substrate (seed=42 0.5994, seed=43 0.6019, seed=44 0.5787) gives a mean of 0.5933 = **+0.55 pp over amortised**, with a 2.3 pp seed-to-seed spread. The +1.16 pp number was a lucky single-seed. The lever is real but smaller than H003's headline. ALL future hypothesis claims must use ≥3-seed replicates; the variance band is ~±1.2 pp not ±0.5 pp.*
