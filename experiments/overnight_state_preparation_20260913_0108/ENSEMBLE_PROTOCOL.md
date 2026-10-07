# Fixed-circuit ensemble objectives: development protocol

This is a follow-up development experiment motivated by the focused curvature
screen: optimizing the local depolarizing condition improved its score while
reducing transfer to the mean of the eleven noisy conditions. That observation
motivates testing the already catalogued ensemble-control candidate. It is not
confirmation evidence, and these four targets cannot become a held-out test set.

The experiment uses the four original, fixed 108-gate q6 circuits and saved
parameters from construction replica 11 in the authenticated `starts.json`:
Gaussian, critical TFIM, Haar random 1, and random MPS with bond dimension two.
No structural deletion, insertion, warm-start transfer, parameter-grid search,
or selection of a previous favorable checkpoint is performed.

## Frozen arms and stopping rules

| Arm                | Training objective                                   | Joint gradient-call cap | Individual channel-gradient cap |
| ------------------ | ---------------------------------------------------- | ----------------------: | ------------------------------: |
| Local L-BFGS       | Local depolarizing                                   |                    1100 |                            1100 |
| Mean-two L-BFGS    | Equal mean of local depolarizing and local dephasing |                     550 |                            1100 |
| Mean-eleven L-BFGS | Equal mean of all eleven noisy benchmark conditions  |                     100 |                            1100 |
| Local Adam         | Local depolarizing                                   |                    1100 |                            1100 |
| Noiseless Adam     | Noiseless                                            |                    1100 |                            1100 |
| Noiseless L-BFGS   | Noiseless                                            |                    1100 |                            1100 |

All six arms are included. L-BFGS uses the existing deterministic resumable
Armijo implementation, history 10, and unchanged line-search, curvature and
gradient-tolerance defaults. Its accepted-iteration cap equals its joint-call
cap, so a hidden 100-iteration default cannot prematurely govern the larger
budgets. Convergence and line-search failure are retained stopping outcomes;
unused calls are reported and are not reassigned. Rejected line-search trials
consume the same channel-gradient budget as accepted trials.

Both Adam arms use the existing bias-corrected Adam update implementation,
learning rate 0.02, beta1 0.9, beta2 0.999, and epsilon 1e-8. They perform 1100
updates using exactly 1100 joint gradients. The final Adam parameters follow
the final update. Its pre-update loss is not presented as its endpoint score.
Adam is a known comparator ([original Adam paper](https://arxiv.org/abs/1412.6980)).
L-BFGS is likewise an established method ([Nocedal's original references](https://users.iems.northwestern.edu/~nocedal/lbfgs.html)).
Optimizing over several noise conditions is an adaptation of known ensemble
and robust control ideas ([Kosut, Grace, and Brif](https://arxiv.org/abs/1301.3235)),
not a new optimization algorithm or a minimax objective.

Single-channel local and noiseless arms use the existing fast exact density
adjoint, avoiding a generic-channel implementation handicap for those strong
baselines. Mean-two and mean-eleven use the independently tested general
ensemble adjoint. Ballarin includes derivatives of both local probability
factors, uses the frozen native materializer's canonicalization and pruning,
and rejects trainable parameters at its documented nonsmooth boundaries.
Its one-qubit operations remain noiseless; retained two-qubit noise follows
the materializer even when a logical noise flag is disabled.

Each target runs its arms serially. The six-arm order is rotated by the
target's position in the frozen target list. Target workers may run in
parallel only under the coordinator's CPU allocation and single-thread BLAS
limits. There is no stochastic training or final trajectory randomness.

## Endpoints, common CPU comparisons, and diagnostics

Every reported endpoint is the final accepted L-BFGS iterate or final Adam
update. All twelve unchanged benchmark conditions are scored exactly on
that same physical circuit and parameter vector. The primary quantity is
the equal-target mean of the eleven noisy-condition fidelities. Noiseless
fidelity is separate. Individual profiles, especially local dephasing and
local depolarizing, remain visible so a mean cannot hide a transfer failure.

The individual-channel cap is matched across arms; equal CPU is not assumed.
Every successful arm retains every accepted iterate with measured cumulative
process CPU and channel-call counts. For each target, define the common CPU
limit as the minimum measured training CPU among all six completed arms.
At fractions 0, 0.25, 0.5, 0.75 and 1 of that limit, select the latest available
iterate using CPU only. Rescore the selected parameters under all twelve
conditions afterward. There is no interpolation of parameters, extrapolation
beyond measured time, or selection by training or evaluation fidelity.
If an arm fails, that target's six-way common-CPU comparison is explicitly
unavailable; its surviving final endpoints and the failed arm remain reported.

Exact score records are cached by circuit, target and complete parameter vector.
Repeated requests for the same point reuse a single scored record. Their
physical forward-channel calls and CPU are counted once in a separate scoring
ledger. Ballarin native and logical gate counts are both reported: its
angle-threshold deletions can change the evaluated native count even though
the trainable logical architecture stays fixed at 108 gates.

Before training, a separate diagnostic computes all eleven exact channel
gradients at the shared original starting parameters, their norms, and every
pairwise cosine. This costs eleven individual gradient evaluations and 22
density direction passes per target, excluded from training budgets but
reported. Zero-norm cosines are undefined. Negative alignment is evidence of
a local first-order conflict, not a proof of global incompatibility. The
diagnostic is not used to choose channels, rates, starts, or stopping points.

## Accounting and reproducibility

One joint gradient of an m-profile objective costs m individual gradient
evaluations and 2m density direction passes. Report logical consumed calls,
attempted and completed physical calls, replayed completed calls, CPU, wall
time, and optimizer stop reason. No statevector trajectory count is inferred
from a density direction pass. The six training caps total 6600 individual
channel gradients per target and 26,400 across the four-target panel, before
separate diagnostics and exact endpoint/common-CPU scoring.

Durable checkpoints are written every ten accepted steps (and the initial
L-BFGS accepted boundary). They retain complete L-BFGS secant memory or Adam
moments, step counters, parameters, and the available-iterate curve. A final
L-BFGS state also retains any partially consumed pending line search. Restart
reuses completed artifacts, and resumes unfinished work from its last durable
optimizer state. Caught interruptions write an attempt ledger in a finally
block; completed work after the last durable checkpoint is explicitly charged
when replayed. SIGKILL and host failure require the coordinator's external
resource ledger because a terminated process cannot report its final CPU.

Training CPU includes optimization and intermediate checkpoint serialization.
Imports, custody checks, final/attempt-ledger serialization, the alignment
diagnostic, and post-training scoring are outside that timer; numerical
diagnostic and scoring costs are reported separately. Allocation-sensitive
wall times are retained, and CPU comparisons should still be interpreted in
light of thermal and scheduling variation on the shared laptop.

Preparation authenticates the benchmark files, original start records, the
optimizer and all imported numerical source hashes, and this protocol text.
Execution rechecks custody before and after each target. Every retained JSON
record has a content checksum. Numerical failures produce explicit per-arm
records and are never silently removed from the panel or retried under a
different configuration. Scientific execution requires the coordinator's
allocation and successful coherent repository lint; preparing does not launch
the experiment.

```sh
uv run python -m experiments.overnight_state_preparation_20260913_0108.ensemble_experiment \
  --output output/overnight-state-preparation-20260913-0108/ensemble-wave1 --prepare-only
```

Replacing `--prepare-only` with `--run --workers 1` executes or resumes the
frozen protocol. BLAS thread limits are supplied by the coordinator. Smaller
budgets and q1-q3 circuits are restricted to automated engineering tests and
are not scientific panel results.
