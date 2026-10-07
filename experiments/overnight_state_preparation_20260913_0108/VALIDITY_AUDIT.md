# Independent audit of numerical benchmark semantics

The selected `original` confirmation recipe is consistent with the immutable
benchmark's corrected channel definitions on the scalar circuits audited here.
No numerical semantics defect was found in this bounded review. This is a
source and engineering-test audit, not a second calculation of confirmation
results. It does not alter the acceptance rule or establish that the candidate
passed it.

The review traced the frozen evaluator, exact objective, sampled controls,
physical initialization and compilation. It reran 111 selected preexisting
synthetic q1–q3 tests: all passed in 1.56 seconds with pytest workers disabled
and all four numerical thread limits set to one. No confirmation target was
decoded or loaded, no confirmation fidelity was recomputed, and no q12 work
was performed. Twenty-six relevant numerical/protocol sources match their
frozen DESIGN hashes. The benchmark text retains SHA-256
`40685856bb052b88aab924792df74159a2b3bdf47d047adfb84183f5f4076f11`.

The [machine receipt](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/overnight-state-preparation-20260913-0108/validation/NUMERICAL_SEMANTICS_AUDIT.json)
contains source hashes, all twenty standard profile/arity calibrations, the
complete selected test command, test-file hashes, JUnit/log hashes and a circuit
metadata inventory. Its content checksum is
`a176f83f4c3c8284b78da600edb825befd287e24ce3899fba3d989020214553c`.
The independent inventory covered 982 content-addressed circuit-bank files
across all ten cases, including intermediate and rejected structures. Every
stored gate was a scalar RY, RZ, RXX, RYY or RZZ rotation, with valid support,
finite affine metadata, `noise_enabled=True`, no data map and no arbitrary
unitary payload. These bank occurrences are not deployed gate-count totals.

## Physical state, gates and fidelity

Both exact evaluators allocate a density matrix with only `rho[0, 0] = 1`.
They apply each physical unitary and then its profile-specific local channel;
they never accept a free prepared initial state. Their fidelity is
$F=\langle\psi|\rho|\psi\rangle$, and loss is $1-F$. Under the noiseless profile
this reduces to $|\langle\psi|U|0\cdots0\rangle|^2$. This is the benchmark's
pure-target fidelity, without a square root or a per-qubit normalization.
Finite values, trace, Hermiticity, positivity and the fidelity interval are
checked to floating-point tolerances.
See [exact evaluation](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/channels.py:166)
and [canonical density propagation](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation/exact_density.py:79).

The target loader validates the little-endian convention: bit $i$ of amplitude
index $k$ denotes qubit $i$. It validates normalization and the stored global
phase, then returns the vector without changing its order. Local circuit
matrices use ascending-site tensor factors; reversed gate sites swap the local
matrix. Density contractions use axis `n - 1 - site`, and the independent
Qiskit references reverse the corresponding `qargs`. Synthetic q3 circuits
exercise nonadjacent and reversed sites with complex targets and shared,
scaled and offset parameters. A global phase cannot change the reported
fidelity.
See [target validation](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation/targets.py:137),
[matrix conventions](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/src/mqt/yaqs/optimization/parameterized_circuit.py:324)
and [independent profile placement tests](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_channels_20260913.py:149).

The legacy constructor starts its MPS backend at the all-zero state and emits
the full scalar preparation circuit. The conditional alternative reverses
rank-two disentanglers, then lowers every one- and two-qubit unitary into scalar
rotations before noise is evaluated. Its unsimplified Weyl decomposition uses
`fidelity=None`; all additional Euler rotations initially have noise enabled.
The independent q1–q3 preparation tests start Qiskit at zero, and the two-qubit
lowering tests compare complete unitaries, including reversed and nonadjacent
sites, up to the recorded global phase.
See [legacy input](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/construction.py:271),
[primitive lowering](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/synthesis.py:156)
and [physical synthesis tests](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_synthesis_20260913.py:49).

## All twelve evaluation conditions

For standard profiles, `1q` and `2q` below refer to the gate's complete support,
not to independently selected spectator sites. Define local-two as
$\{X\otimes I,Y\otimes I,Z\otimes I,I\otimes X,I\otimes Y,I\otimes Z\}$ and correlated-two as
$\{X,Y,Z\}\otimes\{X,Y,Z\}$. A dash means no channel window after that arity.

| Profile                 | After one-qubit gate | After two-qubit gate                                                             |
| ----------------------- | -------------------- | -------------------------------------------------------------------------------- |
| `noiseless`             | —                    | —                                                                                |
| `ballarin_coupled`      | Noiseless            | Native retained RZZ: two independent angle-dependent local depolarizing channels |
| `dephasing_1s_1q`       | Z                    | —                                                                                |
| `dephasing_1s_2q`       | —                    | $Z\otimes I$, $I\otimes Z$                                                       |
| `dephasing_1s_all`      | Z                    | $Z\otimes I$, $I\otimes Z$                                                       |
| `dephasing_2s_2q`       | —                    | ZZ                                                                               |
| `dephasing_1s2s_all`    | Z                    | $Z\otimes I$, $I\otimes Z$, $Z\otimes Z$                                         |
| `depolarizing_1s_1q`    | X, Y, Z              | —                                                                                |
| `depolarizing_1s_2q`    | —                    | Local-two                                                                        |
| `depolarizing_1s_all`   | X, Y, Z              | Local-two                                                                        |
| `depolarizing_2s_2q`    | —                    | Correlated-two, nine generators                                                  |
| `depolarizing_1s2s_all` | X, Y, Z              | All fifteen nonidentity two-qubit Paulis                                         |

The canonical [operator and placement definitions](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation/noise.py)
select these supports. The new [window implementation](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/channels.py:69)
independently calibrates the integrated coefficient of each active generator
to the corrected complete-support gate-average error: $0.00064$ for one qubit
and $0.0051$ for two qubits. It exponentiates the Pauli-Lindblad channel, rather
than interpreting those errors as per-Pauli probabilities. In dimension $D$,
the transfer eigenvalue for Pauli $Q$ is
$\eta_Q=\exp[-2\sum_{P:\{P,Q\}=0}a_P]$, and the gate-average error is
$(D^2-\sum_Q\eta_Q)/(D(D+1))$. The Pauli character transform gives normalized
channel probabilities, including multiple-event cancellation.

The independent tests transcribe the immutable table literally, construct
each generator with Qiskit Paulis, and compare against a dense matrix
exponential. They verify both arities of all ten profiles, including inactive
windows, to $2\times10^{-13}$. The receipt's maximum gate-error discrepancy
is $1.05\times10^{-16}$. Complete q3 profile placement also agrees with an
independently factored Qiskit density calculation to $10^{-13}$.
See [generator tests](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_channels_20260913.py:54).

There is a material distinction from the historical executable provider.
`StandardNoiseProvider` deliberately retains raw legacy per-operator
coefficients and the old finite-step TJM path. The immutable benchmark itself
documents that discrepancy in its implementation-status section. The selected
exact evaluator uses its profile sets, not that provider's execution or rate
assignment. Its separate `legacy_one_jump` convention is not selected by the
confirmation scoring path. These results therefore represent the corrected
mathematical calibration; they must not be relabeled as reproducing the
general legacy runner's noise strengths or finite-step dynamics.
See [benchmark implementation distinction](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation_benchmarks.md:682).

## Ballarin mapping and differentiation

The exact scorer delegates lowering and pruning to the unchanged canonical
[Ballarin materializer](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation/ballarin.py:191).
It wraps angles into $[-\pi,\pi)$, uses their magnitude, lowers RXX with H basis
changes and RYY with RX basis changes, and prunes the entire native round trip
at the fixed small-angle threshold. Single-qubit operations are noiseless in
this profile only. Every retained native RZZ receives its two local channels,
including when an input logical noise flag was disabled; this is the canonical
Ballarin rule. Standard profiles instead preserve such flags. All stored
confirmation circuit-bank gates have their input noise flags enabled.

For $a=|\operatorname{wrap}(\alpha)|$, the fit is
$\epsilon(a)=2.1\times10^{-4}+1.43\times10^{-3}a$ and
$r=(1-\sqrt{1-5\epsilon/4})/3$. Each site uses probabilities
$(1-3r,r,r,r)$. Their product gives sixteen branches and complete two-qubit
gate-average error $\epsilon$. The independent synthetic Qiskit sum verifies
the exact scorer against the materialized circuit.

For the exploratory mean-eleven objective, the adjoint includes the derivative
of these probabilities. Away from pruning and wrapping boundaries,
$dr/d\alpha=5(1.43\times10^{-3})\operatorname{sign}(\alpha)/(24\sqrt{1-5\epsilon/4})$,
using the wrapped angle and multiplying by the gate's affine scale. The noise
tangent includes both terms of the product rule for the two site channels.
Isotropic local depolarization permits the logical-gate adjoint representation
to agree with the native basis-change circuit. Tests include signed and
wrapped angles, all three entanglers, shared parameters, and RZZ on zero,
where the unitary contributes only global phase and the entire loss derivative
comes from noise. Trainable pruning-boundary and angle-wrap cusp points are
explicitly rejected. The selected confirmation training objective is mean-two,
so this derivative is relevant to earlier all-eleven training, while Ballarin
final evaluation remains relevant to confirmation.
See [noise tangent](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_objective.py:96)
and [probability-only derivative test](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_ensemble_objective_20260913.py:102).

## Physical compilation, training and final evaluation

Generic compilation preserves the ideal unitary up to global phase and its
reported numerical error bound, while changing the physical gate sequence.
It removes near-zero rotations modulo $2\pi$ and fuses equal generators only
across disjoint gates with compatible noise metadata. Removing a physical gate
removes its channel window. Setting an angle to zero without removing its gate
does not remove a standard noise window. This intended physical change is
covered by independent q1–q3 complete-unitary tests and explicit noisy-window
tests. Every role receives the same compiler opportunity. Ballarin native
counts are reported separately from the underlying physical circuit counts.
See [compiler](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/compilation.py:50)
and [window-removal test](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_compilation_20260913.py:170).

Exact L-BFGS and exact Adam differentiate the equally weighted mean of the two
frozen training profiles, `depolarizing_1s_all` and `dephasing_1s_all`. The
noiseless ablation uses the corresponding zero-error exact objective. Weighted
Adam and adaptive-online controls instead use 32 paths per training channel:
one deterministic no-factor stratum and 31 conditional paths, with channel
weights one half. Their independent-factor law exactly represents the same
calibrated channel; their finite sample gradients are stochastic. Exhaustive
q1–q3 factor enumeration matches the exact density loss and gradient. The
sampler retains cancelling multi-factor paths, validates the circuit identity,
and rejects a nonzero supplied backend input state. Finite online sweeps use
stale adjoints and are not claimed to be unbiased finite-step gradient updates.
See [mixture law and metadata](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/trajectory_mixture.py:265)
and [exhaustive comparison](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/tests/optimization/test_overnight_trajectory_mixture_20260913.py:100).

Every final or common-cutoff endpoint is scored by
[the shared all-profile scorer](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/composed_experiment.py:141),
which calls `channels.evaluate` once per identifier through the exact operation
store. All twelve profiles receive the same stored physical circuit and theta;
only the prescribed Ballarin materialization differs. The mean-noisy score
excludes `noiseless` and equally averages eleven conditions. The parent then
averages two replicas within each target and weights targets equally. The
scoreboard's `exact_calibrated_channel` label describes final channel summation,
including for stochastic training arms. It does not label their optimization
signals as exact or remove variation between independently trained replicas.
See [exact dispatch](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/ensemble_structural_experiment.py:225)
and [score aggregation and labels](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/best_candidate.py:971).

## Limits of the audit

“Exact” means finite-dimensional channel summation up to floating-point error,
not exact arithmetic. Probability cleanup and physicality checks have explicit
tolerances. The canonical Ballarin implementation inherits a $10^{-12}$
roundoff band around its $10^{-4}$ pruning threshold; the literal sharp
threshold is therefore implemented with that bounded numerical convention.
The generic compiler likewise has an explicit $10^{-12}$ identity threshold
and reports its ideal-operator error bound.

The calibration is a homogeneous mathematical model motivated by gate-average
errors. Neither its support-dependent strengths nor the Ballarin fit establish
native-device accuracy for arbitrary physical hardware. All-profile correctness
on the selected scalar domain does not prove correctness for arbitrary gates,
data-dependent channels, routing schedules, correlated temporal noise or
unmodeled hardware effects.

This review did not independently regenerate target states or recalculate the
ten q6 confirmation endpoints. It relies on the separate completed artifact
authentication for those values. Its 26 source checks are a relevant direct
subset, not a transitive import-closure audit. The separate archive-closure
audit addresses the previously reported direct-map omissions without changing
the frozen design. Synthetic tests are finite examples supported by the channel
derivation, not a universal numerical proof. Finally, the selected noisy
refinement and evaluation are dense q6 calculations; using MPS construction
and validating a separate trajectory bridge do not establish q12 scalability
of this complete method.
