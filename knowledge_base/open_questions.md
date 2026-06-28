# open_questions.md

Mutable. Sub-agents append. Orchestrator prunes resolved questions
periodically. One question per entry; reference the paper / synthesis
that motivated it.

---

## Founding questions (seeded at scaffold time, 2026-06-22)

1. **Multi-point iteration vs single bottleneck.** Marino 2020
   (`papers/marino_iterative_amortized_policy_2020/`) iterates at a
   single posterior. Pinchetti 2024 (`papers/pinchetti_benchmark_2024/`)
   surfaces depth-pathology in deep PC networks. How does
   iterative inference at *multiple* points (every block, not just
   one) interact with the depth-pathology failure mode? Is the
   pathology absent when iteration is distributed?

2. **Timescale-ratio sweet spot.** With block timescales ``[1, τ, τ², ...]``,
   what ratio τ optimises learning speed on Crafter? Does it depend on
   network depth? Is it stable across seeds, or seed-fragile like
   F-MNIST sparsity findings were?

3. **Iterative refinement after amortised init in a world model.** Marino's
   result is on a static inference problem. On an RL world model with
   temporal latent dynamics, does amortised-init + iterative-refine
   preserve the gains, or does the temporal recurrence already absorb
   what iteration would do?

4. **Sample efficiency vs final score.** The bet is *faster + more
   performant*. Are these the same axis (better gradient signal earlier
   ⇒ better final score), or independent (iteration helps the early
   regime, amortisation catches up)? Plot the learning curve, not just
   the asymptote.

5. **Variance-injection-at-training carry-over.** The F-MNIST experiment
   found that variance-preserving Poisson sampling inside the iterated
   PE descent was the strongest single OOD lever
   (`syntheses/02_training_iteration_vs_test_iteration.md`). Does
   variance injection inside the world-model rollout help Crafter
   exploration, or does the env's own stochasticity dominate?

6. **Hopfield-as-memory framing for the world model.** The codeca
   experiment surfaced a Hopfield ↔ attention ↔ PC bridge
   (`syntheses/03_hopfield_attention_pc_bridge.md`). Does framing the
   world model's recurrent memory as a Hopfield-style retrieval
   network buy anything on Crafter's medium-tier achievements (which
   require recalling crafted items)?

7. **Encoder fairness invariant.** Are matched encoders (impala_small
   shared across PPO and iterative-inference trials) sufficient to
   isolate the inference-mechanic effect, or do amortised methods need
   a heavier encoder to compensate for not iterating? Run a paired
   ablation.

8. **The DreamerV3 ceiling.** What specific Crafter score does DreamerV3
   actually report on the configuration we'll match? Pin the number
   from `papers/hafner_dreamerv3_2023/synthesis.md` before the first
   trial so we have a concrete target.
