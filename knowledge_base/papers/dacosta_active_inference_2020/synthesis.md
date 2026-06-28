# Active Inference on Discrete State-Spaces: A Synthesis — synthesis

- **Slug**: dacosta_active_inference_2020
- **Authors**: Da Costa, Lancelot; Parr, Thomas; Sajid, Noor; Veselic, Sebastijan; Neacsu, Victorita; Friston, Karl
- **Year / venue**: 2020 / Journal of Mathematical Psychology (arXiv 2001.07203)
- **Source PDF**: ./dacosta_active_inference_2020.pdf
- **Citation**: Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology, 99, 102447. arXiv:2001.07203.
- **Synthesized by**: arXiv-2001.07203.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
A comprehensive mathematical synthesis of active inference on
discrete state-space generative models. Derives the entire process
theory — perception, planning, action selection, learning, structure
learning — from first principles, with clear notation and a unified
view that subsumes Friston 2017
[`friston_active_inference_2017`] as a special case. For our workspace
the in-scope portion is **Sec. 4-5** (variational Bayesian inference
and perception), which give the most rigorous derivation of the
gradient-descent inference dynamics in our corpus. The planning,
expected-free-energy, and structure-learning material is out of scope.

## Problem framing
The active-inference literature accumulated theoretical refinements
and process-theoretic instantiations across many papers (Friston et
al. 2015, 2017a; Parr & Friston 2019 [`parr_gfe_2019`];
Schwartenbeck et al. 2019). Da Costa et al. unify these into a single
notational framework, derive the dynamics from first principles, and
explicitly map each variable to plausible neuronal mechanisms. The
paper deliberately positions itself as the technical reference for
people who want to *implement* active inference rather than read
piecemeal Friston papers.

## Method
**Markov-blanket framing (Sec. 2 + Fig. 1).** A self-organising system
is partitioned by a Markov blanket into internal states µ, external
states η, sensory states o, and active states u. Internal-state
dynamics minimise a variational free energy that bounds surprise on
sensory states — perception. Active-state dynamics minimise expected
free energy — action (out of scope).

**Generative model (Sec. 3, Fig. 2).** A discrete POMDP:
P(o_{1:T}, s_{1:T}, A, π) = P(π)P(A)P(s_1) ∏_τ P(s_τ|s_{τ-1}, π)
P(o_τ|s_τ, A), with categorical priors and Dirichlet conjugates over
A and B matrices.

**Variational free energy (Sec. 4, eq. 2-3).**

    F[Q(s_{1:T}, A, π)] = D_KL[Q ‖ P_prior] - E_Q[log P(o_{1:t}|s_{1:T}, A, π)]
                       = Complexity − Accuracy

With a structured mean-field factorisation Q(s_{1:T}, A, π) =
Q(A)Q(π) ∏_τ Q(s_τ|π) (eq. 4), the per-policy VFE becomes:

    F_π(s_{π1}, …, s_{πT}) = ∑_τ s_{πτ} · log s_{πτ}
                           − ∑_τ o_τ · log A s_{πτ}
                           − s_{π1} · log D
                           − ∑_τ s_{πτ} · log B_{π_{τ-1}} s_{π_{τ-1}}    (eq. 7)

**Perception dynamics (Sec. 5, eq. 8-9).** Gradient descent on F_π
with respect to the sufficient statistics s_{πτ} gives

    ∇_{s_{πτ}} F_π = 1 + log s_{πτ}
                   − {o_τ · log A + s_{πτ+1} · log B_{πτ} + log D    if τ=1
                      o_τ · log A + s_{πτ+1} · log B_{πτ} + log B_{πτ-1} s_{πτ-1}   if 1 < τ ≤ t
                      s_{πτ+1} · log B_{πτ} + log B_{πτ-1} s_{πτ-1}    if τ > t}    (eq. 8)
    v̇ = -∇F_π,  s_{πτ} = σ(v)         (eq. 9)

— state-estimation as a softmax of accumulated negative-free-energy
gradients, biologically interpretable as v as average membrane
potential and s as average firing rate.

**Learning (Sec. 8, eq. 17-21).** Synaptic plasticity comes out as
Hebbian:
    ȧ = -a + a_prior + ∑_τ o_τ ⊗ s_τ
which formally matches associative / Hebbian plasticity at slow
timescales.

## Key results
- The single mean-field gradient descent on F (eq. 8-9) reproduces
  variational message passing as a special case (Sec. 5.1); under the
  Bethe approximation it reduces to belief propagation. Active
  inference's perception step is *not* a novel inference algorithm —
  it's a generic discrete-state variational inference with a
  particular factorisation.
- The dynamics are biologically faithful: place-cell-like activity,
  mismatch negativity, evidence-accumulation race-to-bound,
  theta-gamma coupling, dopaminergic discharges all emerge.
- Winner-take-all architectures (Sec. 6.3) are recovered as the
  high-precision limit of the softmax over policies, with the
  precision parameter γ controlling sharpness. **Useful framing for
  competition-flavoured PC hypotheses** (in scope, though Da Costa
  applies it to action).

## Limitations
- Limited to discrete states; mixed continuous/discrete generative
  models are deferred to a companion paper (Parr 2019).
- Mean-field is a strong factorisation; the marginal / Bethe variants
  (briefly discussed) are biologically more plausible and not derived
  in full.
- All claims about biological plausibility are coarse (network-level,
  not single-neuron).
- The structure-learning material (Bayesian model reduction /
  expansion, Sec. 9) is theoretical with no benchmarked
  implementation.

## Idea seeds for our loop

**Operator carve-out applied:** in-scope is Sec. 4-5 (VFE,
perception), the softmax winner-take-all observation in Sec. 6.3, the
Hebbian-learning result in Sec. 8 limited to the A-matrix update, and
the Markov-blanket framing in Sec. 2.

- **Equation (8) is a concrete prescription for PC inference dynamics
  that we have not seen written this cleanly in PC-mainline papers.**
  In particular: a state's belief is driven by a sum of (a) the
  current likelihood log A · o_τ, (b) the *forward* prediction
  log B_{πτ} · s_{πτ+1}, and (c) the *backward* prediction log
  B_{πτ-1} · s_{πτ-1}. For static-image PC, transcribing this to a
  hierarchical (spatial) PC means each layer's belief is updated from
  the input *and* both adjacent layers simultaneously, not just from
  bottom-up errors as in standard PC. **This is a clean inference-
  loop variant to test.** Tag: active-inference (perception),
  biological, energy-based. **
- **Softmax-of-log-belief activation function (eq. 9).** The natural
  activation for a categorical posterior is softmax (a generalisation
  of sigmoid). For PC layers that represent categorical hypotheses
  (e.g. the top classification layer), using softmax rather than the
  usual sigmoid / linear is a small architectural tweak with a
  principled motivation; it also provides "lateral inhibition" /
  competition between alternative hypotheses for free. Tag:
  active-inference (perception), competition.
- **Winner-take-all in the high-precision softmax limit (Sec. 6.3) is
  a quantitative bridge between active inference and the Hawkins
  column-voting framing** ([`numenta_thousand_brains_project_2024`]).
  Hawkins' columns vote and a winner emerges; Da Costa's softmax over
  hypotheses is the soft version of the same operation, with γ as the
  knob between "many hypotheses survive" and "one wins." Worth
  exploring whether PC with annealed precision γ (start low,
  competition mild; end high, near-winner-take-all) generalises
  better than PC with fixed precision. Tag: active-inference
  (perception), competition. **
- **Hebbian A-matrix update (eq. 17-21) as a PC weight-update analog.**
  The update Δa ∝ o ⊗ s is the discrete-state equivalent of the
  Hebbian local-error updates in Whittington-Bogacz PC. Useful
  framing: "Hebbian plasticity is what falls out of free-energy
  minimisation in the discrete case; the continuous case
  ([`whittington_bogacz_2017`]) is structurally analogous." Tag:
  biological, energy-based.
- **The accuracy / complexity decomposition** (same as
  [`millidge_efe_2021`]) is the right lens for diagnosing what an
  inference loop achieves on a corrupted observation. Re-iterating
  the recommendation: **log the two terms separately during
  inference, not just total F**.

## Connections
- `[friston_active_inference_2017]` — the previous-generation process
  theory this paper unifies.
- `[sajid_active_inference_demystified_2021]` — the readable
  introduction; Sajid and Da Costa are complementary.
- `[parr_gfe_2019]` — Parr's generalised free energy; Da Costa is the
  natural follow-up read.
- `[millidge_efe_2021]` — interrogates the EFE side of Da Costa's
  formalism.
- `[friston_fep_2009]`, `[bastos_microcircuits_2012]` — foundational
  PC / FEP papers; Da Costa's Sec. 5.1 mentions that the resulting
  prediction-error dynamics are similar in form, though predicated
  on a different generative model.
- `[whittington_bogacz_2017]` — continuous-state PC analog of Da
  Costa's discrete-state perception dynamics.
- `[numenta_thousand_brains_project_2024]` — winner-take-all in the
  high-precision limit (Sec. 6.3) is a bridge to column-voting.

## Relevant themes
active-inference (perception side), energy-based, biological,
competition (via winner-take-all in the high-precision softmax
limit).

## Tier
**medium-supporting** — for the perception-side material. Most
rigorous discrete-state derivation we have, but most of the paper's
content is action-side and out of scope. Sec. 4-5 + Sec. 6.3 are the
sections to direct sub-agents at; the rest is reference.

## Out of scope
**Per operator carve-out**, the following are out of scope:

- **Sec. 6 entire (Planning, decision-making and action selection)**
  — expected free energy minimisation, action selection via Bayesian
  model averaging, biological plausibility of policy selection,
  pruning of policy trees, action-perception cycle discussion. All
  action-side.
  - Exception: Sec. 6.3 paragraph on **winner-take-all as the high-
    precision softmax limit** is conceptually re-usable for
    competition-flavoured PC hypotheses (kept in scope by the
    operator's competition theme).
- **Sec. 7 Properties of the expected free energy** — risk /
  ambiguity / extrinsic / intrinsic value decompositions. Entire
  section is about EFE.
- **Sec. 9 Structure learning** (Bayesian model reduction /
  expansion) — out of scope; the operator is not chasing structure-
  learning hypotheses.
- **Appendix A.2-A.4 (complexifying prior over policies, multiple
  outcome modalities under EFE, deep temporal models for policies)**
  — action-side.
- **Appendix B-C (expected free energy and its computation)** —
  entirely about EFE.

**In scope:** Sec. 1-2 (overview, Markov blanket framing), Sec. 3
(generative model), Sec. 4 (variational Bayesian inference), Sec. 5
(perception), Sec. 6.3 paragraph on winner-take-all only, Sec. 8
limited to the A-matrix Hebbian update (eq. 17-21).

## Notes / follow-ups
The Sec. 5 perception dynamics (eq. 8-9) are the most actionable
specification of an active-inference-flavoured PC inference loop we
have. Worth transcribing into a trial draft.

—
