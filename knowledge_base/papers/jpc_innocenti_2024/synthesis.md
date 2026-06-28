# JPC: Flexible Inference for Predictive Coding Networks in JAX — synthesis

- **Slug**: jpc_innocenti_2024
- **Authors**: Innocenti, Francesco; Kinghorn, Paul; Yun-Farmbrough, Will; De Llanza Varona, Miguel; Singh, Ryan; Buckley, Christopher L.
- **Year / venue**: 2024 / arXiv:2412.03676
- **Source PDF**: ./jpc_innocenti_2024.pdf
- **Citation**: Innocenti F, Kinghorn P, Yun-Farmbrough W, De Llanza Varona M, Singh R, Buckley CL. JPC: Flexible Inference for Predictive Coding Networks in JAX. arXiv:2412.03676, 2024.
- **Synthesized by**: agent-20260622
- **Status**: initial

## TL;DR
JPC is a <1000-line JAX library (Equinox + Diffrax + Optax) for training PCNs of every common flavour — discriminative, generative, hybrid. Its single distinctive contribution: replace the standard Euler-discretised inference loop with proper ODE solvers (Heun, a second-order Runge–Kutta). Empirically, Heun converges in significantly fewer wall-clock steps than Euler across MNIST/F-MNIST/CIFAR-10 and depths H∈{3,5,10}, with comparable accuracy. Includes theoretical-energy tools (closed-form energy at inference equilibrium for deep linear networks). Tooling, but tooling matters because iterative-inference sweeps are our loop's compute bottleneck.

## Problem framing
The PC literature lacked unified, fast, modular implementations (acknowledged earlier libraries: pypc, Torch2PC). Built on Equinox (Kidger & Garcia 2021), Diffrax (Kidger 2022), and the Buckley-group line on PC fundamentals (Buckley et al 2017, Millidge et al 2021). Frames PC inference as a gradient flow ż_l = −∂F/∂z_l and asks whether higher-order ODE solvers beat the default Euler integration. Complementary to PCX `[pinchetti_benchmark_2024]` which used Euler throughout.

## Method
PCN energy F = Σ_ℓ ||z_ℓ − f_ℓ(W_ℓ z_{ℓ-1})||². Bi-level optimisation: inference (Eq 2, infer arg min_{z_l} F) then learning (Eq 3, learn arg min_{W_l} F). Inference dynamics are the gradient flow ż = −∂F/∂z. Standard PC uses Euler integration (z ← z − dt · ∂F/∂z). JPC instead uses Diffrax's adaptive solvers (Heun by default), with a Proportional–Integral–Derivative step-size controller.

API surface:
- `jpc.make_pc_step(model, optim, opt_state, y, x)` — single-call training step. Integrates inference dynamics to equilibrium using ODE solver, then updates parameters.
- `jpc.make_hpc_step(generator, amortiser, optims, opt_states, y, x)` — Hybrid PC `[hybrid_pc_2023]` step with separate generator and amortiser.
- `jpc.solve_inference`, `jpc.update_params`, `jpc.init_activities_with_ffwd` — exposed primitives for custom step functions.

Theoretical tool: for deep linear networks, energy at inference equilibrium has closed form (Innocenti et al 2024):

F* = (1/2N) Σ (y_i − W_{L:1} x_i)^T S^{-1} (y_i − W_{L:1} x_i), S = I + Σ_{ℓ=2}^L (W_{L:ℓ})(W_{L:ℓ})^T.

Comparing numerical F to theoretical F* during training tells you whether sufficient inference has been performed at each train step. Used to diagnose under-converged inference (Fig 2, Fig 7).

## Key results
- **Heun beats Euler in wall-clock per training iteration** across MNIST/F-MNIST/CIFAR-10 at H∈{3,5,10} (Fig 1). Effect is larger at deeper networks.
- **Comparable accuracy at convergence** between Heun and Euler at matched hyperparams (Fig 3, Figs 4-6 in appendix).
- **Theoretical energy criterion**: when the gap between numerical and theoretical F closes, accuracy plateaus (Fig 2 MNIST, Fig 7 F-MNIST). Suggests a principled stopping criterion: integrate inference until F ≈ F*_theoretical.
- **Library is small (<1000 LoC) and modular**: PCNs as PyTrees, JIT-compiled, swappable solver/optimizer.

## Limitations
- Runtime experiments use only one epoch — not a serious benchmark of final accuracy.
- Heun's gains are noted as "could be different with other optimiser-specific hyperparameters."
- Theoretical energy tool only works for deep *linear* networks; nonlinear case is open.
- No comparison to PCX's JAX implementation (the two are parallel competitors).
- No stochastic-differential-equation solvers — extension noted as future work (especially relevant for MCPC / Langevin PC `[mcpc_oliviers_2024]`).
- No experiments on OOD or corruption — purely tooling/runtime paper.

## Idea seeds for our loop

- **Adopt JPC (or Heun-style ODE inference) in our trial harness.** ** The single highest-leverage idea: if our trials do many inference iterations, switching from Euler to Heun (or any second-order RK) speeds them up at constant accuracy. Especially useful at H=10-style depths if a sub-agent tries deeper PCNs. Tag: implementation-efficiency.

- **Use the closed-form deep-linear energy as an "inference convergence" diagnostic.** Our spec's guardrail 4 requires non-increasing energy across the final 25% of inference. JPC's theoretical F* gives a stronger criterion: an inference loop is *under-converged* if F > F*_theoretical, regardless of slope. For linear-init or near-linear PCNs (and for sanity-checking the inference loop), this is a free guardrail. Tag: implementation, diagnostics.

- **Stochastic-differential-equation inference for MCPC.** They flag SDE solvers as future work. Combining JPC's ODE/SDE infrastructure with `[mcpc_oliviers_2024]`'s Langevin PC gives proper Monte Carlo PC inference — sampling-based, not point-MAP. On-thesis, and directly enabled by JPC's plumbing. Tag: energy-based, iteration-for-OOD-evidence.

- **Adaptive step-size controller as a per-sample compute knob.** Diffrax's PID controller adapts step size to local curvature. On corrupted samples (high curvature, poor convergence), it would naturally spend more compute. Similar to `[hybrid_pc_2023]`'s adaptive iteration count but more principled. Worth measuring whether per-sample step count correlates with corruption severity on F-MNIST-C. Tag: iteration-for-OOD-evidence.

- **HPC implementation drop-in.** They already expose `make_hpc_step`. A sub-agent implementing HPC can use JPC's HPC primitives directly rather than rewriting. Tag: implementation, must-have.

## Connections
- `[hybrid_pc_2023]` — JPC has first-class HPC support; this is the natural HPC training stack.
- `[pinchetti_benchmark_2024]` — parallel JAX PC library (PCX). The two are complementary; PCX has the benchmark, JPC has the ODE solver innovation.
- `[mcpc_oliviers_2024]` — Langevin PC; SDE solvers in JPC are the natural infrastructure.
- `[salvatori_associative_2021]` — same iterative inference dynamics that JPC integrates more efficiently.
- `[salvatori_reverse_2022]` — Z-IL is a different (training-side) speedup; JPC speeds up *inference*, Z-IL speeds up *training*. Complementary, not competing.
- `[whittington_bogacz_2017]` — training-side foundation; JPC's solver is for the inference half.
- `[ngc_ororbia_2022]` — NGC's inference dynamics could be integrated by JPC-style ODE solvers (NGC's lateral terms would need custom handling).
- `[multistep_pc_simplicity_2025]` — theoretical work on inference-depth/representation; JPC's theoretical-energy tool is a sibling diagnostic.

## Relevant themes
implementation-efficiency, energy-based, iteration-for-OOD-evidence (indirectly — enables more iteration cheaply)

## Tier
high

## Out of scope
—

## Notes / follow-ups
- Code: github.com/thebuckleylab/jpc. Documentation and example notebooks (referenced in §4.1) are the right starting point for any sub-agent considering it as a runtime base.
- The "Only strict saddles" paper (Innocenti et al 2024, arXiv 2408.11979) is the source of the closed-form energy result and worth a glance for theoretical context — it argues PCN's optimisation landscape has only strict saddles, no spurious minima, which is encouraging for our loop.
