# Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models — synthesis

- **Slug**: millidge_hopfield_2022
- **Authors**: Millidge, Beren; Salvatori, Tommaso; Song, Yuhang; Lukasiewicz, Thomas; Bogacz, Rafal
- **Year / venue**: 2022 / ICML 2022 (arXiv:2202.04557)
- **Source PDF**: ./millidge_hopfield_2022.pdf
- **Citation**: Millidge B, Salvatori T, Song Y, Lukasiewicz T, Bogacz R. Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models. ICML 2022. arXiv:2202.04557.
- **Synthesized by**: agent-20260622 (literature-supplement)
- **Status**: initial

## TL;DR
**The bridge paper.** Millidge and Bogacz (predictive-coding insiders) show that classical Hopfield, dense associative memory, modern continuous Hopfield/attention, and sparse distributed memory (Kanerva 1988) are **all special cases of a three-step framework**: (1) similarity computation between query and stored patterns, (2) separation function applied to similarities, (3) projection onto a stored value set. Choosing different similarities (dot-product, Euclidean, Manhattan) and different separation functions (identity, polynomial, softmax, max) recovers each historical model. This **unifies associative memory with attention** under one schema and — critically for us — places PC-style retrieval in the same family. **Most load-bearing paper for our loop's purposes** because the authors are PC researchers and they explicitly frame Hopfield-style retrieval as adjacent to PC inference.

## Problem framing
The Hopfield literature exploded in 2020 ([[krotov_hopfield_2020]], [[ramsauer_hopfield_2020]]) with the attention-equivalence result, but the literature had a zoo of related models (classical Hopfield, DAM, modern Hopfield, sparse distributed memory, Kanerva) that all behaved similarly without a unifying schema. Millidge et al. — coming from the PC tradition where they had just shown PCNs are associative memories ([[salvatori_associative_2021]]) — saw the unifying structure. The paper's contribution is taxonomic and analytical: pick a similarity, a separation, a projection, and you reproduce any of the historical models. They go further: empirically compare combinations on retrieval tasks, demonstrating the framework is *generative* (new combinations work).

## Method
**Single-shot retrieval**:

retrieval(q) = projection(separation(similarity(q, K)), V)

where:
- `q` is the query (corrupted/partial input)
- `K` are stored keys (patterns)
- `V` are stored values (often = K for auto-associative)
- `similarity` is a function `ℝ^d × ℝ^{N×d} → ℝ^N` (e.g., dot product `q K^T`, negative Euclidean `-||q - K||²`, negative Manhattan `-||q - K||_1`)
- `separation` is a function `ℝ^N → ℝ^N` (e.g., identity, polynomial `x^n`, softmax `softmax(βx)`, max)
- `projection` combines the separated scores with values: `result = sep(sim(q,K)) · V / norm`

**Special cases recovered**:
| Model | Similarity | Separation |
|---|---|---|
| Classical Hopfield (linearised) | dot product | identity |
| DAM polynomial ([[krotov_hopfield_2016]]) | dot product | `x^n` |
| Modern Hopfield ([[ramsauer_hopfield_2020]]) | dot product | softmax |
| Sparse Distributed Memory (Kanerva 1988) | Manhattan / Hamming | max / threshold |
| Transformer self-attention | dot product (scaled) | softmax |

**Iterative retrieval**: iterate the single-shot step with `q ← retrieval(q)` until fixed point — recovers multi-step Hopfield dynamics.

**Energy formulation**: each (similarity, separation, projection) combination has a corresponding Lyapunov function (when the operators are sufficiently regular); iteration descends this energy.

## Key results
- **Unification**: 4+ historical AM models recovered as special cases.
- **New combinations work**: Manhattan-similarity + softmax-separation (an unstudied combination) gives higher retrieval accuracy on MNIST than the classical (dot+identity) and matches modern Hopfield (dot+softmax) on some corrupted tasks.
- **Separation function is the key knob**: nonlinear separation (softmax, polynomial, max) gives high capacity; identity gives only linear capacity.
- **Similarity matters less than separation**: dot vs Euclidean vs Manhattan have second-order effects compared to separation choice.
- **PC connection**: the authors explicitly note that PCN retrieval ([[salvatori_associative_2021]]) doesn't fit cleanly in the single-shot schema because PC's retrieval is *iterative inference on a learned generative model*, not single-shot. This puts PCN-as-AM in a distinct, more flexible class.

## Limitations
- The framework is *single-shot*; iterative dynamics (which PC uses) are folded in only by repeated application — doesn't directly model PCN's iterative inference equation.
- Doesn't include sampling / probabilistic retrieval (Boltzmann-machine-style); deterministic only.
- Empirical comparison is on synthetic MNIST-scale retrieval — not natural-image-scale.
- "Universal" oversells — there are AM models (e.g., spiking AM, multi-layer DAM with hidden representations) that don't fit cleanly.
- The PC connection is *gestured at* but not formally derived in the paper.

## Idea seeds for our loop

- **★★ Strongest seed: PCN retrieval IS a generalised similarity-separation-projection step, but with an iterative/variational similarity.** The authors stop short of formalising this; we could. Specifically:
  - PCN's "similarity" is *prediction error* between current state and the generative prediction.
  - PCN's "separation" is the *gradient step* on this error (effectively a kind of nonlinear weighting).
  - PCN's "projection" is the message-passing back through the weight matrix.
  - **And it's iterated.**
  
  Rewriting PCN inference in Millidge et al.'s schema makes it a natural sibling of modern Hopfield retrieval, with the key difference being that PC iterates the step and uses a learned generative model rather than fixed stored patterns. **This is the conceptual frame for understanding why iterative variational PC works for OOD.** Tag: energy-based, iteration-for-OOD-evidence.

- **H010 = modern Hopfield retrieval in the PC schema.** H010 proposes input-conditioned learned-precision attention on prediction errors. In the universal-Hopfield schema, this is:
  - similarity: prediction error magnitude (or signed PE)
  - separation: softmax with input-conditioned temperature β(x)
  - projection: weighted combination back into the latent
  
  This is *modern Hopfield retrieval with input-conditioned softmax temperature*. Millidge et al. (and the underlying [[ramsauer_hopfield_2020]] / [[krotov_hopfield_2020]] equivalences) provide the formal foundation for H010 — the sub-agent attacking it should read this paper to ensure the formulation is correct. **★ Strongest H010 seed.**

- **New combinations to try.** Manhattan-similarity + softmax-separation hasn't been tested in PC. What if PC's prediction error used L1 instead of L2? Combined with softmax-style retrieval at the top layer? This is a novel PC-attention variant.

- **Iterated single-shot AM as a baseline.** Before committing to a deep PCN-Hopfield hybrid, run the *single-shot* Millidge-Salvatori universal-Hopfield retrieval on F-MNIST-C as a baseline. The retrieval accuracy gives us an upper bound on what "associative memory with one attention step" achieves — anything beyond requires the iterative/variational machinery PC brings. Tag: iteration-for-OOD-evidence.

- **Sparse distributed memory port.** Kanerva's SDM (Manhattan + threshold) hasn't been tried in our corpus. SDM's hard threshold is similar in spirit to kWTA — which H001 already tested unsuccessfully. But: SDM uses sparse *addressing* not sparse *latents*, which is a different mechanism. Possibly worth a follow-up trial.

## Connections
- `[salvatori_associative_2021]` — same lab (Millidge, Salvatori, Bogacz, Lukasiewicz); PCN-as-AM is the PC-side; this paper is the unification. Read together.
- `[ramsauer_hopfield_2020]` — modern Hopfield = attention; one of the four models unified here.
- `[krotov_hopfield_2020]` — continuous DAM = attention; another of the unified models.
- `[krotov_hopfield_2016]` — polynomial DAM; one of the unified models.
- `[hopfield_1982]` — classical Hopfield; the foundational special case.
- `[hinton_boltzmann_1985]` — Boltzmann machines are stochastic relatives; not unified here but conceptually adjacent.
- `[whittington_bogacz_2017]` — Bogacz is a co-author of both; canonical PC framing of [[salvatori_associative_2021]] connects PC to this paper's framework.
- `[hybrid_pc_2023]` — same lab; amortised+iterative PC is the next step beyond single-shot retrieval.
- `[friston_fep_2009]` — variational inference framing; the iterative version of single-shot retrieval is what PC inference becomes.
- `[brain_like_vi_2024]` — iterative VAE family with explaining-away; structurally similar to iterated AM retrieval.
- `[multistep_pc_simplicity_2025]` — long inference horizons = many iterated retrieval steps; theoretical scaffolding.

## Relevant themes
energy-based, competition (separation as winner-take-all), iteration-for-OOD-evidence (in the iterated-retrieval interpretation), biological (mild)

## Tier
high

## Out of scope
—

## Notes / follow-ups
- Synthesis written from prior knowledge; PDF fetched from arXiv (2202.04557).
- **Read first if attacking H010 or any "PCN + attention" hypothesis.** This paper formalises why those ideas should compose.
- Millidge & Bogacz are PC literature insiders; their unified framework is the closest published bridge from Hopfield-attention back to PC.
- Possible meta-synthesis warranted (`syntheses/03_hopfield_attention_pc_bridge.md`) — see below.
