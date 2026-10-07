# Pure-state metric preconditioning: kernel and development protocol

Status: implementation and q1-q3 engineering validation only. No scientific
training run is authorized or executed by this artifact. Candidate 7 in the
existing registry predeclared the two damping values below. A later allocated
driver must freeze this protocol, benchmark custody, starting records, and
all imported source hashes before running the four-target panel.

## Mathematical identity and provenance

For the ideal logical circuit applied to the all-zero input, compute the
state psi and every complete scalar-parameter derivative d_i. The real metric
is the full Fubini-Study tensor

`G_ij = Re(<d_i|d_j> - <d_i|psi><psi|d_j>)`.

This convention has no factor of four; pure-state quantum Fisher information
would be `4G`. Implementation projects each derivative orthogonally to psi,
then forms the real Gram matrix. This removes parameter-dependent global
phase components and preserves positive semidefiniteness up to floating-point
roundoff. Each shared parameter receives the sum of every occurrence's
derivative, including cross terms in the metric. Signed scales, offsets,
unused parameters, and zero-scale gates are handled explicitly.

This is a known geometric construction and optimizer adaptation, with no
novelty claim. [Stokes, Izaac, Killoran, and Carleo, Quantum 4, 269 (2020)](https://quantum-journal.org/papers/q-2020-05-25-269/)
connect quantum natural gradient to the real quantum geometric tensor. Their
efficient block approximation is distinct from the full dense metric here.
The broader noisy-state formulation in
[Koczor and Benjamin, _Quantum natural gradient generalised to noisy and non-unitary circuits_](https://arxiv.org/abs/1912.08660)
is relevant prior art, but is not implemented in this experiment.

The metric describes the ideal pure state. The trained objective is exact
infidelity under the benchmark's calibrated local depolarizing channel.
Consequently this is a surrogate preconditioner, not the natural gradient
of the resulting mixed state. Gauge or weak directions of the ideal state
may be poorly aligned with the noisy objective. A small regularizer can
amplify a noisy gradient component along a nearly null ideal direction.
This mismatch is a central falsification mechanism, not a hidden assumption.

## Implemented kernel and optimizer

`state_jacobian(circuit, theta)` returns psi, parameter-by-state derivative
rows, and cost metadata. A single batch propagates psi and all tangent rows
through every ideal unitary. For a trainable occurrence, insert
`angle_scale * D * psi_after`, where `dU/dangle = D U`. The checked local
statevector kernel is reused without constructing full-system operators.
Physical noise flags do not alter this ideal metric. Circuit and parameter
inputs are not mutated.

`pure_state_metric(circuit, theta)` adds the projected Gram construction.
`precondition(gradient, G, damping)` solves
`(G + damping I) direction = -gradient` without forming an inverse. It checks
symmetry, positive semidefiniteness and finite descent, and reports spectral
conditioning and solve CPU. Damping is restricted to 0.001 or 0.01; there is
no adaptive damping, rank cutoff, pseudoinverse, or additional search grid.

`minimize(circuit, initial, target, damping=..., max_evaluations=...)` provides
the complete deterministic local-noisy optimizer for at most 300 true
channel-gradient calls. Initial step length is one, Armijo coefficient 1e-4,
backtracking factor 0.5, at most 20 trials per direction, and infinity-norm
gradient tolerance 1e-8. Every trial evaluates the same true calibrated
local-depolarizing loss and gradient. Rejected and nonfinite trials count
against the budget. The pure metric is rebuilt once per new accepted base
point, and its direction is reused during backtracking.

Returned state contains the accepted parameters and noisy gradient, all
counters, pending direction/step/slope/trial count, metric-construction and
solve records, every line-search decision, and the accepted-iterate curve.
It is JSON serializable, including rejected nonfinite trials, whose loss
is represented by an explicit null rather than an invalid JSON infinity.
A callback receives the complete private state after the initial evaluation
and each finished trial. A resumed pending search uses precisely the same
next trial and does not charge or compute its metric a second time.

The caller owns authenticated durable storage and physical attempt accounting.
Checkpointing at every ten completed channel calls is sufficient for the
planned driver. If interruption occurs after the last durable checkpoint,
replayed work and interrupted-attempt CPU must remain in its external ledger.
The kernel does not create a background job, silently launch scientific runs,
or modify any frozen optimizer/ensemble module.

## Planned fixed development comparison

Use the same original four q6 development targets, construction replica 11,
original saved parameters, and fixed 108-gate logical architecture. Every
arm starts independently from those originals; no improved structural or
ensemble endpoint is imported as a favorable warm start.

| Arm                      | Damping or optimizer setting             | True channel-gradient cap |
| ------------------------ | ---------------------------------------- | ------------------------: |
| Pure-metric Armijo       | 0.001                                    |                       300 |
| Pure-metric Armijo       | 0.01                                     |                       300 |
| Exact local-noisy L-BFGS | Existing Armijo optimizer, history 10    |                       300 |
| Exact local-noisy Adam   | Existing Adam update, learning rate 0.02 |                       300 |

L-BFGS retains its existing numerical defaults and an accepted-iteration cap
of 300 so the joint-call cap, convergence, or explicit line-search failure
governs stopping. Adam retains beta1 0.9, beta2 0.999 and epsilon 1e-8, and
uses its final update. No best-checkpoint selection or additional rate grid
is allowed. Fixed final iterates are scored under all twelve unchanged
benchmark conditions. The mean over eleven noisy conditions and noiseless
fidelity are reported separately; local-depolarizing fidelity remains visible
as the actual trained objective.

The equal channel-call cap does not make metric computation free or imply
equal CPU. A meaningful comparison must also select the latest available
iterate at common measured training-CPU limits and rescore those parameters
under the same fixed conditions. It must include failed searches and all
unused budgets, avoid interpolation or quality-based checkpoint selection,
and charge endpoint/common-CPU scoring separately from training.

## Cost, tests, and falsification

For L ideal gates, P parameters, and n qubits, the forward tangent batch costs
`(P+1)*L` local row-operator applications plus one derivative insertion for
each trainable occurrence with nonzero scale. It stores a `(P+1) by 2**n`
batch; metric formation adds a P-by-P Gram matrix and costs O(P squared times
2\*\*n). Spectral validation and the damped solve each have cubic parameter
cost. Explicit array sizes are working-array sizes, not a peak-RSS claim.
These statevector operations are deterministic tangent propagation, not
stochastic trajectories or density direction passes.

The returned ledger separates true noisy-gradient CPU, metric construction
CPU, solve/spectral-check CPU, and total optimizer CPU. Each joint noisy
gradient contributes two density direction passes. Every failed line-search
trial is counted. A reused pending direction adds no extra metric evaluation.
Metric construction/solve cost must remain present when comparing the two
geometric arms to L-BFGS and exact Adam.

Engineering tests compare all scalar-gate Jacobian rows with independent
Qiskit state finite differences, including reversed supports and shared
parameters. They verify the metric formula, phase-gauge invariance, analytic
shared-RX cross terms, pure gauge/null directions, positive semidefiniteness,
the nontrivial q2 rank bound, and the local-infidelity quadratic form. Further
tests check regularized descent, invalid-input rejection, monotone true-noisy
Armijo steps, explicit nonfinite trial costs, budget-limited pending searches,
and identical numerical optimizer paths after JSON serialization and callback
interruption.

These tests establish implementation correctness only. Retire the candidate
if its metric and solve costs erase its fidelity gain at common CPU, if it
systematically consumes the budget in backtracking, or if the pure/noisy
geometry mismatch produces unstable or ineffective directions. Any favorable
development outcome would require a separate frozen confirmation experiment.
