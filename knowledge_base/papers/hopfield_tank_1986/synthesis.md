# Computing with Neural Circuits: A Model — synthesis

- **Slug**: hopfield_tank_1986
- **Authors**: Hopfield, John J.; Tank, David W.
- **Year / venue**: 1986 / Science 233:625-633
- **Source PDF**: ./hopfield_tank_1986.pdf (PDF MISSING — synthesis from prior knowledge)
- **Citation**: Hopfield JJ, Tank DW. Computing with Neural Circuits: A Model. Science 1986;233:625-633.
- **Synthesized by**: agent-20260622 (literature-supplement)
- **Status**: initial

## TL;DR
The continuous-state successor to the 1982 Hopfield paper. Replaces binary units with graded sigmoidal neurons that obey a deterministic continuous-time ODE; the same Lyapunov-function structure carries over, so dynamics still descend an energy to fixed points. They show this lets a neural circuit *compute*: solving constraint-satisfaction problems (analog-to-digital conversion, the traveling salesman problem) by encoding the cost function as an energy and letting the dynamics relax. This is the direct mathematical ancestor of PC inference dynamics — continuous-time gradient flow on an energy that mixes data-fit and prior terms.

## Problem framing
Builds on [[hopfield_1982]]; the binary-state model was elegant but biologically unrealistic and computationally limited. Hopfield-Tank generalises to graded neurons with RC dynamics (Hodgkin-Huxley-flavoured passive circuit) and demonstrates that the energy-descent property is robust to the discretisation choice. The framing is explicitly *constructive* — they show how to design a network whose energy minimum encodes the answer to an arbitrary optimisation problem, opening up "neural computation as constraint satisfaction." Sits in the same lineage as later constraint-satisfaction work (Boltzmann machines [[hinton_boltzmann_1985]], mean-field theory in neural nets).

## Method
N continuous units `u_i ∈ ℝ` (membrane potentials) with sigmoidal output `V_i = g(u_i)` where `g` is a steep sigmoid. ODE:

C_i du_i/dt = -u_i/R_i + Σ_j T_ij V_j + I_i

with symmetric synaptic matrix `T_ij = T_ji`. The Lyapunov function:

E(V) = -½ Σ_ij T_ij V_i V_j - Σ_i V_i I_i + Σ_i (1/R_i) ∫₀^{V_i} g⁻¹(v) dv

decreases monotonically under the dynamics (dE/dt ≤ 0, with equality only at fixed points). For sharp sigmoids the integral term is small and behaviour approximates the binary network. To solve an optimisation problem: choose `T_ij` and `I_i` so that E is the cost function plus a soft-constraint barrier; let the circuit relax. Originally implemented as an actual analog electronic circuit.

## Key results
- **Convergence to fixed points** under continuous dynamics — same property as the binary network.
- **TSP demonstrator**: 10-city traveling salesman solved (approximately, within ~few % of optimal) by mapping the cost function to network energy.
- **A/D conversion**: a 4-bit ADC implemented as a small Hopfield-Tank network.
- **Speed scales as O(1)** circuit relaxation time (constant in number of digital ops), independent of problem size — the analog speed claim that motivated decades of neuromorphic work.

## Limitations
- Solutions are local minima, not necessarily global — TSP solutions are approximate, often poor.
- Soft-constraint barriers in E mean some "valid" fixed points violate problem constraints.
- Symmetric `T` and gradient-flow assumption — biologically still implausible; later "non-symmetric" extensions sacrifice the Lyapunov guarantee.
- No probabilistic interpretation; deterministic descent gets stuck.
- Doesn't scale: per-pattern capacity inherited from [[hopfield_1982]].

## Idea seeds for our loop

- **Continuous-time PC = Hopfield-Tank dynamics with a structured energy.** PCN inference is literally what Hopfield-Tank described: graded units with sigmoidal nonlinearity, continuous-time descent on a Lyapunov energy. Our PCN inference loops are a discretised version of the same ODE Hopfield-Tank wrote down 40 years ago. This anchors PC firmly in the energy-descent tradition.

- **The integral term `∫ g⁻¹(v) dv` is a prior on activations.** Hopfield-Tank's "natural" energy includes a per-neuron integral term that penalises extreme activations — i.e., it's a *prior over the latent state*. This is exactly the structural role played by L1 priors in [[sdpc_boutin_2021]], by Poisson priors in [[pvae_vafaii_2024]], and by Gaussian priors in canonical PCN. The Hopfield-Tank energy is the prototype "data-fit + activation-cost" decomposition our PC variants are exploring. Tag: energy-based, sparsity.

- **Constraint-satisfaction framing for OOD inference.** OOD generalisation can be reframed as constraint satisfaction: the input gives partial constraints (corrupted pixels), the prior gives partial constraints (training distribution), and the energy encodes their joint cost. Inference *is* finding the configuration that satisfies both. This is exactly the Hopfield-Tank computational frame — and exactly what a generative PCN should do at test time. The central thesis is a restatement of Hopfield-Tank's claim with a richer (learned, hierarchical) prior.

- **Analog inference speed.** The original argument was that *relaxation time is independent of pattern dimension*. PC inference loops are likewise iterative, not sequential-per-element. If we want fast OOD inference at deployment, the analog-relaxation framing argues for parallel inference hardware (not new — but worth remembering for [[friston_fep_2009]]-style neuromorphic deployment).

## Connections
- `[hopfield_1982]` — binary-state predecessor; this paper is the continuous-time extension.
- `[krotov_hopfield_2020]` — modern continuous dense AM; same continuous-state framing, polynomial energy.
- `[hinton_boltzmann_1985]` — adds stochasticity to graded recurrence; probabilistic counterpart.
- `[friston_fep_2009]` — free-energy gradient-flow as the general form; PCN dynamics inherit this.
- `[whittington_bogacz_2017]` — PC equations are explicitly the Hopfield-Tank form with an error-propagation structure.
- `[salvatori_associative_2021]` — PCN-as-associative-memory directly inherits Hopfield-Tank's retrieval-via-relaxation.
- `[brain_like_vi_2024]` — natural-gradient FOND dynamics generalises the Hopfield-Tank Lyapunov to information geometry.

## Relevant themes
energy-based, biological, iteration-for-OOD-evidence

## Tier
supporting

## Out of scope
—

## Notes / follow-ups
- PDF missing — synthesis from prior knowledge. Science 1986 paper, accessible via JSTOR / institutional libraries.
- Foundational anchor; the continuous-state Lyapunov result is what every later "energy-descent inference" paper relies on.
