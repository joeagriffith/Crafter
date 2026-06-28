# Central Thesis: Iterative Inference for Faster and More Performant Learning on Crafter

**Status**: founding document — initial. Author: agent-20260622-setup.

This is the founding synthesis for **pc-crafter**. Every hypothesis tested in this workspace should sharpen, extend, or falsify the claim below. When the leaderboard becomes hypnotic, re-read this.

## The bet

> **Iterative inference — gradient descent on a posterior/energy/free-energy at test time — can produce *faster and more performant* learning than amortised approaches on Crafter, when the iterative mechanics occur at *multiple points* inside a *deep* network with *timescale-separated* layer dynamics (deeper layers update more slowly than shallower).**

Three narrowings carry load:

1. **Iterative beats amortised on sample efficiency, not just final score.** The benchmark question is "how few env steps to reach a given achievement count," not "what asymptote can you reach with infinite data." DreamerV3 sets the bar at ~5 achievements / score ~6 within 1M steps; we are aiming to either match that bar in fewer steps or to exceed the score at matched budget.
2. **Multiple iterative points, not one.** A single iterative latent on top of a feedforward trunk is the easy case and largely matches amortised baselines (Marino 2020). The bet is that iterative dynamics at multiple depths inside the network compose into something the amortised version cannot match.
3. **Timescale separation is load-bearing.** Deeper layers update slower than shallower; this is the structural prior the model uses to allocate "fast perceptual correction" to early layers and "slow context revision" to late layers. Without timescale separation, the multi-point iteration collapses back into uniform unrolled compute (cf. DEQs' OOD null result on classification).

## Why this is the right bet

The most direct empirical precedent is **Marino 2020** (`papers/marino_iterative_amortized_policy_2020`, arXiv:2009.01791). Marino's central result: iterating a posterior *after* amortised initialisation outperforms either approach alone, on both reconstruction and policy-conditioned distributions. Iteration alone is slower than amortised; amortisation alone leaves a gap to the variational optimum; the combination closes the gap cheaply, because the amortised network does the heavy lifting and a few iteration steps polish the answer. The framing pc-crafter inherits from Marino is therefore **iterative-after-amortised** as a default, with iteration as a refinement of an amortised guess rather than a from-scratch posterior search.

Extending to RL world models: an amortised world-model prior (e.g., DreamerV3's RSSM) gives a coarse rollout; a few iterative refinement steps applied at each predicted timestep — and crucially at multiple latent depths — should re-localise the posterior toward the data manifold cheaply. The compute cost per step rises modestly; the sample efficiency (env steps per achievement) is the metric we expect to move.

Crafter is the right testbed because it has:

- a small enough state and action space that the iteration's compute cost stays manageable on a single GPU;
- 22 explicit achievements with a clean geometric-mean score that resolves sample efficiency at fine granularity;
- a published 100k-step regime where sample efficiency separates methods (PPO vs DreamerV3 is a >2× score gap);
- temporal structure rich enough that a memory module's quality matters, but short enough that we don't need a Memory-Maze-scale architecture from day one.

## What survives from the F-MNIST experiment

Three findings from the closing-out F-MNIST work in codeca carry into pc-crafter as durable priors. They are recorded in detail in `syntheses/02_training_iteration_vs_test_iteration.md` and `syntheses/03_hopfield_attention_pc_bridge.md`; the punchlines:

**Training-time variance injection is the lever.** The H002 2×2 ablation showed that injecting noise *during* iterated PE descent at training time — not at test time — is what moves OOD performance. Test-time iteration without train-time variance injection bought nothing on F-MNIST-C. The mechanism is best understood as "the model learns to invert the noise process," and only the variance-injected training regime exposes it to that signal. **Implication for pc-crafter**: variance injection during iterated rollouts of the world model is a high-prior intervention. Plain test-time iteration without it is the wrong ablation to focus on.

**Variance-preserving Poisson sampling was the strongest single OOD lever.** Of the sampling variants tried (Gaussian Langevin, Poisson, mixture priors), variance-preserving Poisson dominated on F-MNIST-C OOD. The mechanism is consistent with Vafaii et al's P-VAE: Poisson latents are continuous distributions over discrete counts (not spikes), and the variance-preserving parameterisation prevents the prior from collapsing as the mean shrinks. **Implication for pc-crafter**: when reaching for a non-Gaussian latent distribution in the world model, Poisson with variance preservation is the first thing to try.

**Sparsity, Forward-Forward, and column-voting were refuted on the simpler problem.** The hypothesis space the orchestrator entered F-MNIST with included three operator-flagged themes that did not survive: hard sparsity (Numenta-style) showed no consistent OOD gain over soft Laplace priors; Forward-Forward as a training scheme did not compose with PC's variational frame in a useful way; column-voting with semi-independent units underperformed shared-trunk baselines at matched parameters. **Implication for pc-crafter**: these directions are not in the starting hypothesis space. They may re-enter later if a specific Crafter-structured argument re-motivates them, but they don't start with priors.

The fourth carry-over is architectural rather than empirical: **the Hopfield ↔ attention ↔ PC bridge** as a lens on the model's memory module. A continuous Hopfield retrieval IS one step of transformer self-attention (Ramsauer 2020, Krotov 2020), and PC inference is gradient descent on an energy that — for the right prior — has the same fixed-point structure. The implication for pc-crafter is that the model's episodic-memory module can be framed *interchangeably* as attention-over-keys, Hopfield retrieval, or iterative PC inference. We get to pick the framing that's most tractable per design decision and convert between them when convenient. See `syntheses/03_hopfield_attention_pc_bridge.md`.

## What is new in pc-crafter

The problem differs from F-MNIST in four ways that matter for design:

- **Time is first-class.** Crafter is a sequential decision problem. Temporal structure (recurrent state, rollout-time iteration, multi-step prediction) is part of the model, not a wrapping.
- **Deep networks with multiple iterative points.** F-MNIST experiments used 2–3 hidden layers and a single iterative latent. pc-crafter targets networks with more than three hidden layers, iterative dynamics at several depths, and the question of how those dynamics compose.
- **Timescale separation between layers.** Deeper layers update on a slower clock than shallower ones — implemented as either a smaller learning rate, fewer iteration steps, or a literal time-constant in a continuous-time formulation. This is the structural mechanism we expect to make multi-point iteration buy more than single-point iteration.
- **Success metric shift: sample efficiency, not OOD accuracy.** The number we move is *env steps to a target Crafter score* (and/or *Crafter score at a fixed step budget*). OOD is not the framing here — generalisation matters insofar as it improves sample efficiency.

## Open hypothesis space

Five directions are open from day one. Sub-agents should locate any new hypothesis inside one of these (or argue explicitly for a sixth).

1. **Deep PCN architectures with multi-point iterative inference.** What's the right depth? Where in the network do iterative latents pay off — sensory layers, mid-network feature layers, world-model latent, all of the above? Does the gain compose linearly or saturate? The baseline failure mode to rule out is "single iterative latent does just as well as four."

2. **Timescale separation (slow-deep, fast-shallow).** Implement timescale separation three ways — layerwise learning rates, layerwise iteration counts, continuous-time τ — and ablate which mechanism actually carries the gain. The question is whether the gain is about the *separation* itself or about a specific implementation of it.

3. **Iterative-after-amortised (Marino-style) for world-model rollouts.** Initialise each rollout step with an amortised RSSM-like prior; refine with a small number of iterative steps before sampling the next latent. Ablate iteration count (0, 1, 3, 7) against rollout horizon and against single-step prediction accuracy. The first goal is to *replicate* Marino's "iteration-after-amortisation wins" on Crafter at small scale before scaling up.

4. **Hopfield/attention-as-retrieval framing for the memory module.** The model's episodic memory can be a Hopfield-style energy with the trajectory store as patterns, or equivalently attention-over-stored-states. Pick the framing per design decision; ablate energy form (modern Hopfield vs softmax attention vs continuous PC) at matched parameters. The bridge synthesis (`03_*`) is the navigational document.

5. **Variance-injection-at-training carry-over.** Inject variance during iterated world-model rollouts at training time — Langevin-style noise on iterative refinement, Poisson variance-preserving on the world-model latent. The F-MNIST result was that test-time iteration without train-time variance injection is largely empty calories; this prior transfers directly. Default to variance-preserving Poisson if a non-Gaussian latent is needed.

## Out of scope

Hard carve-outs. Treat as constraints.

- **No spiking neural networks.** We may read spiking-PC papers for ideas about timing, dynamics, and local plasticity, but we do not build spiking networks. Continuous-activation analogues are preferred. As in codeca: Poisson VAE is **not** spiking — Poisson latents are continuous distributions over discrete counts. P-VAE is in scope.
- **No active-inference action-selection / expected-free-energy-under-policies machinery.** Friston's perception-as-inference is on-thesis; the planning machinery (EFE under policies, reflex-arc action selection, etc.) is off-thesis. Read selectively from active-inference papers — the inference-side chapters are useful; the policy/action sections are not.
- **Standard RL value/policy methods are IN scope as amortised reference baselines.** PPO is the bottom-of-curve baseline. Actor-critic heads on top of the PCN world model are fine. The carve-out is specifically against active-inference *planning* — not against amortised RL machinery as a comparison.
- **Crafter specifically.** No Atari, no DMC, no Procgen at the start. Memory Maze (Pasukonis 2022, `papers/pasukonis_memory_maze_2022`) is on deck if memory-module work matures and the project has time — but only after Crafter results land.

## Success criteria

The project succeeds if **any one** of the following is achieved:

1. **Beat DreamerV3 Crafter score at matched env-step budget.** Specifically, exceed the DreamerV3-reported geometric-mean Crafter score at 1M env steps (or at whatever budget the operator declares as the comparison point), on the same Crafter version, with seeds averaged appropriately.
2. **Reach a fixed Crafter score (~5 achievements / score ~6) in fewer env steps than DreamerV3.** Sample-efficiency framing. This is the bet's natural success condition — the curve crosses DreamerV3's curve to the left.
3. **Show clear sample-efficiency curve separation from a PPO baseline at fixed compute budget.** The minimum bar: our method beats a strong amortised RL baseline by a clear margin (>30% fewer env steps to a fixed score, averaged over seeds) at the same wall-clock or FLOP budget. This is the floor; the ceiling is criterion 1 or 2.

Failure modes to recognise early:

- **Matched-amortised performance** — i.e., the iterative version ties an equivalent amortised model. This refutes the bet as stated and forces a rewrite of this document. It does **not** refute iterative inference broadly, but it does refute *this* specific framing of when iterative wins.
- **Iterative slows learning** — i.e., the iterative version is sample-efficient on a per-update basis but the per-step compute makes total wall-clock slower than amortised at matched budget. This is a *real* outcome to watch for and is partially addressed by Marino-style iterative-after-amortised initialisation, but if it persists it would push the project toward a "small iteration count, well-targeted" framing rather than the multi-point deep framing.

## How sub-agents should use this synthesis

This is not a hypothesis list — it's the *frame* through which any hypothesis should be evaluated. When deciding what to test, ask:

1. Does the hypothesis sit inside the bet — iterative inference, multi-point, deep, timescale-separated, on Crafter?
2. Is it motivated by direct evidence (`papers/marino_iterative_amortized_policy_2020`, `papers/hybrid_pc_2023`, F-MNIST carry-overs) or by an explicit cross-pollination from the KB?
3. Would confirming or refuting it move sample efficiency on Crafter, **or** generate a load-bearing insight about *where* iteration helps even if the headline number doesn't move?
4. Is it inside scope (no spiking, no active-inference planning, no other environments yet)?

If at least three of those four are "yes," the hypothesis is worth a session. If fewer, the hypothesis is probably outside the bet — re-anchor to this document before proceeding.
