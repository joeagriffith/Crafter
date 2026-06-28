# by_relevance.md

Papers grouped by how directly they bear on the pc-crafter thesis
(iterative inference for faster + more performant learning on Crafter).
Tiers — must-read, high, medium, supporting, counter-evidence.

The orchestrator owns this index. Sub-agents append a row when they
add a paper; the orchestrator may re-tier as evidence accumulates.

---

## Must-read (load-bearing for the founding thesis)

- [`marino_iterative_amortized_policy_2020`](../papers/marino_iterative_amortized_policy_2020/synthesis.md)
  — iterative-after-amortised inference; the project's anchor paper.
- [`hafner_crafter_2022`](../papers/hafner_crafter_2022/synthesis.md)
  — the env paper. Defines the score and the 22-achievement set.
- [`hafner_dreamerv3_2023`](../papers/hafner_dreamerv3_2023/synthesis.md)
  — the dominant Crafter baseline. The target we beat.
- [`hybrid_pc_2023`](../papers/hybrid_pc_2023/synthesis.md)
  — System-1 / System-2 amortised+iterative split. Direct precedent.
- [`pinchetti_benchmark_2024`](../papers/pinchetti_benchmark_2024/synthesis.md)
  — deep-PC depth-pathology paper. We must understand the failure mode
  before we can design around it.

## High relevance

- [`whittington_bogacz_2017`](../papers/whittington_bogacz_2017/synthesis.md)
  — foundational PC training mechanism.
- [`micheli_iris_2023`](../papers/micheli_iris_2023/synthesis.md)
  — strong alt baseline; transformer world model.
- [`hafner_dreamerv2_2020`](../papers/hafner_dreamerv2_2020/synthesis.md)
  — DreamerV3 lineage; introduces discrete-latent world models.
- [`jpc_innocenti_2024`](../papers/jpc_innocenti_2024/synthesis.md)
  — Heun ODE solver for fast PC inference.
- [`multistep_pc_simplicity_2025`](../papers/multistep_pc_simplicity_2025/synthesis.md)
  — inference-depth vs simplicity bias.
- [`brain_like_vi_2024`](../papers/brain_like_vi_2024/synthesis.md)
  — direct empirical evidence for iterative-inference's OOD value.
- [`pvae_vafaii_2024`](../papers/pvae_vafaii_2024/synthesis.md)
  — variance-preserving Poisson sampling; strongest F-MNIST OOD lever.

## Medium relevance

- [`mcpc_oliviers_2024`](../papers/mcpc_oliviers_2024/synthesis.md)
  — Langevin / MCMC predictive coding.
- [`tschantz_curvature_2023`](../papers/tschantz_curvature_2023/synthesis.md)
  — Hessian-as-precision.
- [`salvatori_associative_2021`](../papers/salvatori_associative_2021/synthesis.md)
  — associative-memory PC.
- [`salvatori_reverse_2022`](../papers/salvatori_reverse_2022/synthesis.md)
  — algorithmic variants of PC.
- [`intro_pcn_ml_2025`](../papers/intro_pcn_ml_2025/synthesis.md)
  — recent ML-focused PC review.
- [`pasukonis_memory_maze_2022`](../papers/pasukonis_memory_maze_2022/synthesis.md)
  — companion env if/when we extend.

## Supporting (Hopfield bridge, foundational)

- [`hopfield_1982`](../papers/hopfield_1982/synthesis.md)
- [`hopfield_tank_1986`](../papers/hopfield_tank_1986/synthesis.md)
- [`krotov_hopfield_2016`](../papers/krotov_hopfield_2016/synthesis.md)
- [`krotov_hopfield_2020`](../papers/krotov_hopfield_2020/synthesis.md)
- [`ramsauer_hopfield_2020`](../papers/ramsauer_hopfield_2020/synthesis.md)
- [`millidge_hopfield_2022`](../papers/millidge_hopfield_2022/synthesis.md)
- [`friston_fep_2009`](../papers/friston_fep_2009/synthesis.md)
- [`dacosta_active_inference_2020`](../papers/dacosta_active_inference_2020/synthesis.md)
- [`numenta_active_dendrites_2021`](../papers/numenta_active_dendrites_2021/synthesis.md)

## Counter-evidence / cautionary

(Add entries here as the loop accumulates failure-mode evidence.)
