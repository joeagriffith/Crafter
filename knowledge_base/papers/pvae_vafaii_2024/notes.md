# P-VAE — sub-agent notes

Append-only notes from sub-agents who used the synthesis.

---

## 2026-06-22 — agent-H005 (sub-agent 5)

Implemented a PCN+Langevin+Poisson hybrid (option A: encoder predicts
δr; δr is a free PCN latent that gets settled by gradient descent on
the energy `F = 0.5 ||δr - pred_δr||² + β · r · f(δr) + 0.5
||top - W_1 z + b_1||²`). The Poisson sampling uses Vafaii's reparam-
eterised algorithm (exponential inter-event times + sigmoid-relaxed
indicator). Refimpl pull was skipped — Vafaii's published code is conv
VAE on van Hateren / MNIST patches with a linear sparse-coding decoder,
which would have replaced the substrate rather than layering on it.

### Empirical findings against the synthesis predictions:

- **"Poisson PC with iterative inference is the obvious load-bearing
  combination"** (synthesis Idea Seed ★★): **partially confirmed**.
  The 3-seed mean on F-MNIST-C lifted from substrate's 0.5933 to
  **0.6420** (+4.87 pp). However, the clean F-MNIST floor broke by
  ~6.5 pp across all 9 variants, making the hypothesis invalid as
  stated. The OOD signal is real and large; the floor break is
  architectural (softplus-based encoder + non-negative rate latents
  + linear readout).

- **"Sparsity-from-prior beats sparsity-as-regulariser"**: irrelevant
  here — the gains came from the Poisson SAMPLING variance, not from
  the sparsity induced by the KL term. A deterministic-mean ablation
  (trial 045, no sampling, same architecture) gave 0.5492 — worse
  than substrate. The "sparsity emerges from KL" property of P-VAE is
  not the load-bearing piece for OOD.

- **"Curvature-sensitive inference × Poisson"**: untested in H005,
  remains the natural follow-up (open question 5+10).

- **"Shattering dimensionality predicts OOD wins"**: not directly
  measured in this loop's evaluator. Worth instrumenting if a
  future sub-agent pursues P-VAE more deeply.

- **"β as a sparsity dial"**: tested only at β=0.001 (settled on after
  β=1.0 led to immediate r→0 collapse with learnable_r=True). With
  fixed r=1 and β=0.001 the system is stable.

### Practical implementation gotchas:

- **r→0 collapse is fast with learnable_r=True.** The KL gradient
  `β · r · f(δr)` w.r.t. log_r is `r · β · f(δr) ≥ 0`, always pushing
  r down. Without a decoder that creates pressure to use z, r drops
  to ~0.07 in 40 epochs and the network can no longer classify.
  Fixing r=1 (non-learnable) was essential.

- **Softplus on the inference-step δr is the wrong projection.**
  softplus(0) ≈ 0.69, so it adds a constant offset to values near
  zero — δr explodes from ~1 to >1.29 in one step, energy blows up.
  Use `clamp(min=1e-6)` instead to project to non-negative while
  preserving positive values.

- **Test-time multi-sample averaging does NOT help.** The mean rate
  λ at test gives equivalent (slightly better) accuracy than averaging
  K=8 exact-Poisson samples. The train/test mismatch is NOT about
  sampling; it's structural.

- **Poisson warmup → deterministic curriculum doesn't preserve the
  OOD gain.** Active maintenance of multiplicative variance throughout
  training is required.

### Mechanism distilled

The variance-preserving property of Poisson sampling (Var = mean) IS
the OOD lever — the structural changes (softplus, learnable rate,
multiplicative encoder, KL term) actually HURT clean accuracy on their
own (trial 045 substrate = 0.55). Adding sampling on top gives +9.7 pp
OOD. This is the dual of H004's variance-killing soft-Laplace
refutation: variance-preserving works where variance-killing fails.

Follow-up for the next sub-agent: try multiplicative-noise injection
on the substrate's tanh-based encoder (no softplus, no KL term, no
non-negative latents — just a multiplicative N(1, σ²) or
relaxed-Bernoulli mask during PCN training inference). This should
combine the variance-preserving OOD lever with the floor-preserving
substrate architecture.
