# Two-channel entropic-risk development screen

This is a new objective family, prepared before its scientific execution.
It does not modify the benchmark, frozen mean-objective experiments, channels,
starts or optimizer implementations. Only the coordinator can allocate a run.

## Provenance and mechanism

Risk-sensitive decision and control objectives are established methods; see
[Howard and Matheson, 1972](https://pubsonline.informs.org/doi/10.1287/mnsc.18.7.356).
For quantum control, [Ge and Wu, 2021](https://arxiv.org/html/2104.01323v1)
derive exponential-utility weighting of individual uncertainty gradients, with
normalized weights in their equations 5-7. Their sampled Hamiltonian-uncertainty
setting motivates this adaptation to two exact calibrated channel expectations.
Their performance or convergence claims are not transferred to this experiment.
[Ge et al., 2020](https://arxiv.org/abs/1909.02296) provides related adversarial
quantum-control context; this implementation does not add adversarial samples.

For the unchanged local all-support depolarizing and dephasing channels, define
`loss_j(theta) = 1 - F_j(theta)`. Minimize

`L_tau(theta) = tau * log((exp(loss_1/tau) + exp(loss_2/tau))/2)`.

Its exact gradient is `w_1*g_1 + w_2*g_2`, where
`w_j = exp(loss_j/tau)/sum_k exp(loss_k/tau)` and each g is the existing exact
density adjoint. Larger losses receive greater weight. The equal-channel mean
is the infinite-temperature limit; the maximum channel loss is the limit as
temperature approaches zero. For finite positive temperature,
`mean(loss) <= L_tau <= max(loss)` and
`L_tau >= max(loss) - tau*log(2)`.

Equivalently this maximizes a smooth minimum channel fidelity. The risk is over
complete expected losses of channel conditions, not stochastic path losses;
applying this nonlinear objective to noisy estimates would introduce a separate
estimation-bias problem. No such approximation is used here. It supplies no
guarantee about the other nine noisy benchmark conditions.

Use a maximum shift, expm1/log1p for the small-spread correction, and shifted
softmax. Numerical stability follows the established shifted-evaluation approach
analyzed by [Blanchard, Higham and Higham, 2019](https://arxiv.org/abs/1909.03469).
The expm1/log1p refinement preserves the equal-mean limit for large temperature.
The temperature and normalized objective scale remain fixed within each arm.

## Frozen six-arm comparison

Use the authenticated original 108-gate q6 starts for Gaussian, critical TFIM,
Haar 1 and random MPS bond 2, construction replica 11. Protect benchmark custody;
every circuit physically starts from zero, with unchanged metadata and targets.
There is no structural search, compilation or target-dependent setting in this
mechanism screen. Temperature values are exactly 0.02 and 0.1, replacing the
earlier scout's unexecuted tentative 0.05/0.1 proposal before this screen starts.

Each arm receives a cap of 1100 individual channel-gradient calls, equivalent
to 550 joint calls, from the identical original theta:

- entropic-risk L-BFGS with temperature 0.02;
- entropic-risk L-BFGS with temperature 0.1;
- entropic-risk exact Adam with temperature 0.02;
- entropic-risk exact Adam with temperature 0.1;
- equal-mean L-BFGS;
- equal-mean exact Adam.

L-BFGS uses the unchanged resumable two-loop implementation: history 10,
Armijo coefficient 1e-4, halving backtracking, at most 20 trials, gtol 1e-8.
Every rejected line-search gradient call is charged. Adam uses exactly 550
updates, rate 0.02, beta1 0.9, beta2 0.999, epsilon 1e-8; moments persist through
the entire arm. Each successful joint gradient uses two exact channel calls and
four density directions. Early L-BFGS convergence reports actual unused budget.
No additional settings, restarts, temperature changes or best-checkpoint choices.

The mean arms use the same component oracle and aggregation path as the risk
arms, with explicit mean aggregation. Existing mean2 results are provenance
references until identities and resource scopes can be authenticated; rerunning
a control is charged as new physical work. Six new arms isolate the objective
change from optimizer opportunity, but are not the strongest stochastic controls.

## Endpoints, costs, failure and resume

The final endpoint is the last accepted L-BFGS iterate or final Adam update.
Score it independently under all twelve unchanged exact benchmark conditions.
Record logical and Ballarin-materialized gate counts, depth, parameter count,
complete per-channel fidelities, training/evaluation CPU and wall time, individual
and joint calls, attempted/completed physical calls, and density directions.
Component fidelities and softmax weights from gradient work are diagnostic only;
they do not replace final exact evaluation.

Store the circuit once and retain theta-only accepted-iterate curves. Current
checkpoints preserve full optimizer memory, objective identity, counters and
temperature. Intermediate checkpoints are written every ten accepted iterations.
Resume from the latest durable state with identical numerical trajectory; retain
all interrupted and replayed work in raw attempt ledgers. Unknown killed work
requires external resource accounting, never a zero-cost entry. Failures remain
visible and cannot be omitted from panel comparisons.

For common-CPU comparison use the smallest completed-arm training CPU for the
same target and choose each arm's last available iterate at or before that CPU.
Selection uses time alone, then scores all twelve exact conditions. Charge every
failed/rejected call and interruption replay; do not equate two directions with
one forward-only evaluation. Optimization and intermediate-checkpoint CPU are
included, while final persistence and orchestration are separately recorded.
Historical panels with different I/O scopes require explicit scope labels.
The 1100-call cap applies to the reproducible logical optimization trajectory;
interruption replay is additional physical work and is reported separately,
never erased or presented as cap-equivalent measured compute. Case invocation
timing includes training/evaluation persistence and orchestration within the
case; final case-record writes and external custody remain launcher overhead.

`risk_experiment.py --prepare-only --output <directory>` freezes one protocol
without propagation. Its `--run --workers 1` mode is reserved for a coordinator
allocation after full lint; each numerical worker must use one BLAS thread.
The production protocol fixes all six methods and both temperatures. Smaller
configurations accepted by the Python API are engineering tests, not extra
scientific settings. The output uses `overnight.entropic_risk.v1`, signed
protocol/case/attempt records and explicit stable method/channel CSV ordering.

## Metrics, promotion and falsification

The primary score remains the equally weighted eleven-noisy-condition mean,
first per target and then equally across all four targets. Noiseless and the
two training channels are diagnostic, not substitutes for that primary score.
Tail metrics are the minimum over all eleven noisy fidelities per target and
the mean of its three lowest noisy fidelities. Also show each channel family
(N1 angle, N2 dephasing, N3 depolarizing) and its worst family average.

Compare each risk arm to the equal-mean control with the same optimizer, and
show the strongest applicable matched control rather than selecting only a weak
one. A primary-mean promotion requires at least 0.5 percentage points over that
strongest control, without an unexplained failure or a target loss above one
point. A tail-oriented development promotion instead requires at least 0.5
points in the target-balanced all-eleven worst-channel metric with no more than
0.1 points loss in the all-eleven mean versus the corresponding mean control.
Report this route explicitly as a tail tradeoff and show the common-CPU result;
it is not a claim of a superior primary-mean algorithm. Meaningful resource
advantages with comparable mean/tail remain separately described evidence.

Retire settings that worsen the primary mean without this declared tail benefit,
or improve only the two training conditions. Temperature 0.02 may overweight a
specialist bad channel, distort curvature or impair line search; temperature
0.1 may be too close to the mean to change outcomes. Improved training-channel
tail can coexist with worse unseen-condition tail. Broad superiority still
requires the equally treated strong weighted/adaptive objective baselines and
independent confirmation targets/replicas. This screen establishes neither.
