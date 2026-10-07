# Twelve-qubit competitor readiness

**2026-09-28 campaign update:** the user selected 400 bases × 1,024 shots for
RobustState, with its sampling limitations explicitly retained. The distributed
eleven-group matrix excludes Ballarin and is defined in
[Q12_CAMPAIGN.md](Q12_CAMPAIGN.md). Preparation/staging are separate from launch;
the full matrix has not been started. Earlier audit findings below remain
historical evidence and do not imply that 400 bases are sampling-converged.

This work prepares numerical backends and bounded feasibility checks. It does
not launch the comparison matrix or supply new competitive fidelity results.
Use the existing IBM logical Pauli channel for every training and evaluation
window. The original competitors' objectives, optimizer policies and gate
schedules are preserved. The synthetic logical channel is a benchmark
adaptation, not the hardware noise model in each original paper.

## Exact mixed-state execution

`checkpointed_dense.CheckpointedDenseObjective` supports one to twelve qubits.
It uses recursive recomputation for the exact noisy adjoint, retaining a
logarithmic number of density checkpoints instead of one per gate. Recomputed
gates cost additional time. The channel is the same correlated Pauli mixture,
contracted as a local superoperator; no inverse channel, density truncation,
trajectory approximation or replacement gradient is introduced.

Noiseless fidelity gradients and RobustState's ideal-density Jacobian use ket
propagation. For a fixed Hermitian costate C, the ket costate is C|psi>, and
2 Re <C psi|d psi> equals Tr(C d rho). This holds for indefinite shadow
estimates as well. A twelve-qubit complex density still occupies 256 MiB;
checkpoint counts do not include all temporary arrays. Measure actual peak RSS.

Fontana and RobustState select this backend above six qubits. The original
six-qubit objective stays in place. Generic target validation remains capped
at six unless a caller explicitly opts into twelve. Other competitors are not
silently enabled at larger sizes.

## Ballarin Gaussian startup and compression

The current compressed candidate pipeline uses [published two-site TCI](IQSP_TCI.md)
at every interpolation coordinate. Pass `tci_options=TCIOptions()` to the startup
audit for this target source. The older preflight receipts below used dense SVD
and remain historical diagnostics. They do not certify a TCI campaign.

Use the original twelve-qubit one-dimensional Gaussian fixture, represented
as `CombLayout(12, 1)`. `interpolation_target` now derives the resolution from
the supplied fixture instead of hardcoding six qubits. The current Gaussian
TCI path queries the analytic amplitude oracle and independently checks the
result against the original fixture. The optional legacy dense-SVD conversion
also preserves target amplitudes but must not be labeled TCI. Report target
construction settings, error and setup cost separately in either case.

The identity SU(4) derivative caveat in `IQSP_COMB.md` remains real. The
`q12_readiness.audit_gaussian_initialization` check runs at lambda=0.05 from
unchanged zero parameters, using normal Adam at learning rate 0.01. An exact
statevector derivative is an audit oracle only; it never drives training.
Within twenty startup updates, the run must show actual exact-fidelity
progress, fidelity discrepancy below 1e-5 throughout, and final relative
gradient error below 1%. Record the large initial gradient discrepancy rather
than hiding it. These thresholds are readiness criteria, not paper parameters
or a global accuracy guarantee. Failing cases must not enter the ranking.

For twelve-qubit noisy gradients, use explicit autograd checkpointing and
verify target representation precision before timing.
For TCI, keep `target_tolerance=0`; its own interpolation controls and independent
target error are recorded separately. For historical dense-SVD diagnostics,
`CombOptions(checkpoint_interval=8, target_tolerance=1e-28)` removes only
numerical null directions from the Gaussian target, with an independently
checked vector error below 8e-14. This is optional and recorded; zero target
tolerance preserves every singular direction as before. Final evaluation
always uses the original exact fixture. Only the explicitly selected legacy
conversion is a dense-fixture SVD adapter; the default now uses published TCI.
The vendored SVD backward also applies null-space projectors in factored form,
using `W - U (U† W)` and `W - (W V) V†`. It no longer builds the large square
projector of a tall or wide purification split. This preserves the original
regularized derivative. Tests independently verify linear-loss derivatives and
reject any allocation of the corresponding full square projector.

The opt-in recomputes gate/channel segments during backward evaluation instead
of retaining their complete autograd graphs. It preserves all forward tensors,
rank choices, optimizer gradients and diagnostics; parity tests cover pure,
IBM and H2 execution. Zero retains the previous full-tape behavior. A local
tensor-size bound alone does not bound the full autograd allocation.
On macOS, monitor physical footprint including compressed memory, not RSS alone.

SU(4) logical lowering repairs only a Qiskit eigensolve failure caused by
roundoff nonunitarity: the polar correction must be at most 1e-12 per matrix
element, and the independent local/global circuit-equivalence checks still
apply. The correction is recorded. No parameter jitter, additional circuit
gates or modified training update is introduced.

Noisy compressed execution also needs comparisons against tighter bond and
environment settings. Forward accuracy does not establish gradient accuracy.
Record both discrepancies, local discarded weights and actual ranks. The
campaign must continue these checks at its selected endpoints; passing a
startup check does not certify an entire optimization trajectory.

## RobustState sampled feedback

Sampling uses integer base-three indices without constructing the entire
3^n basis list. A known CPTP simulator output can skip a redundant cubic
positive-semidefiniteness eigensolve; the public default still validates
untrusted density matrices. The original raw shadow estimator and all sampled
counts are unchanged, and no positivity projection is added.

Forty bases with 1,024 shots per basis at twelve qubits is a proposed pilot
budget. The paper demonstrates its 40-basis example at four qubits. The new
post-training audit uses 32 independent draws at each of two twelve-qubit
checkpoints after the original 500 ideal Adam steps, preserving moments.
Mean gradient cosines are -0.03180 for Haar 1 and -0.00345 for TFIM critical;
both 95% intervals contain zero. The mean predicted Adam descent intervals
also contain zero. The budget is therefore not validated for the main ranking.
This does not prove failure of fifty-step stochastic training or inferiority
of the published method. The exact-feedback TFIM diagnostic step also slightly
reduces fidelity, so sampled-feedback noise alone does not explain every decline.
Report preparation noise, ideal measurement rotations/readout, basis count,
shots, surrogate-gradient variability and classical simulation time explicitly.

## Independent final assessment

`logical_evaluation.validation_from_paths` now streams at most 256 statevectors
at a time. Chunking preserves the sampled paths, exact no-jump multiplicities,
weighted mean and raw-sample standard error. Validation and final test banks
must remain independent of training and each other. Keep actual jump counts;
do not condition the sample on observing jumps.

Before launching a campaign, require passing tests, recorded noisy feasibility
and compression checks, and a frozen protocol listing exclusions. One optimizer
seed and four depths remain a pilot, not a converged publication comparison.

## Launch status

The implementation permits preparation of a resource-bounded pilot, but the
original full-depth, full-budget comparison is not cleared. In particular,
RobustState's proposed 40-basis twelve-qubit budget did not establish reliable
average feedback at the two trained checkpoints above. A different declared
measurement budget needs evidence; exact density is a separately labeled
diagnostic oracle, not a silent replacement for sampled feedback.

Ballarin's new TCI depth-two comparison has relative gradient discrepancy
0.01814%, but absolute fidelity discrepancy 4.0616e-5 exceeds the predeclared
1e-5 criterion. This is a small 0.00406-percentage-point difference, not evidence
of catastrophic inaccuracy. Only completed matched comparisons in the latest
report establish agreement at the stated checkpoints. Failed memory attempts
provide no accuracy result and must not enter the ranking as low fidelities.
The diagnostic's optional compact checkpoint storage preserves tensor values
and derivatives while releasing oversized SVD backing views; its twelve-qubit
default comparisons and small-system parity tests are recorded separately.

The resumed audit also tested accumulation of the same SVD derivative terms
in one buffer. All 153 targeted tests pass, including eight frozen derivatives;
the twelve-qubit depth-three default gradient matches its original to relative
error 5.03e-15. The tighter depth-three retry still reached the 8 GiB memory
guard after 812 seconds. The depth-four default retry was then interrupted
when the host ran out of free disk space, and its tighter partner was not
started. Earlier completed depth-four defaults remain available. Temporary
disk offloading was considered but not implemented or run. Thus, deeper
compression accuracy remains unresolved; no failed or interrupted probe is
treated as low fidelity. The full matrix remains on hold.

A subsequent [memory diagnostic](../../output/q12-ballarin-memory-20260927/REPORT.md)
found that tighter depth-three forward-only propagation also exceeds 8 GiB,
without retaining an autograd tape. At the interrupted SVD, the compressed
state occupies 6.49 MiB, but its temporary complex128 matrix occupies 649.69
MiB. The same matrix's thin SVD completes in a fresh process at a sampled
5.15 GiB physical footprint. The isolated derivative attempt stopped at the
disk guard before backward execution. Thus, checkpoint offloading alone is
not a demonstrated solution, and derivative workspace remains unmeasured at
that input. No target, noise, compression or optimization rule changed.

Current numerical results, resource receipts and qualification of the findings:
[`output/q12-final-checks-20260925/REPORT.md`](../../output/q12-final-checks-20260925/REPORT.md).
The older [preflight report](../../output/q12-prerequisites-20260923/REPORT.md)
remains a historical non-TCI diagnostic. Fontana's earlier depth-one gradient
cost about 173 seconds; none of these single-gradient checks measures the cost
of a complete optimizer run. The full comparison matrix has not been launched.

## Leonhard checks, 2026-09-27

The remote [report](../../output/q12-remote-checks-20260927/REPORT.md) records
the frozen snapshots, resources and completed numerical diagnostics.
Leonhard-13 and Leonhard-16 have sufficient observed memory capacity;
the checks run on Leonhard-13 through the configured georg gateway. The
recorded node survey is not a reservation or an exhaustive cluster inventory.

The TCI depth-three default/tighter pair now completes: absolute fidelity
difference 2.66697e-4 exceeds the unchanged 1e-5 criterion, while relative
gradient difference 0.07564% passes the 1% criterion. The tighter calculation
took 61.41 minutes and 6.58 GiB sampled RSS versus 2.03 minutes and 1.47 GiB
for default. Identical parameter, target and source hashes were verified.
The tighter result is approximate and cannot establish absolute error.
The initial remote tighter depth-four attempt stopped at the local tensor
allowance after 29.45 minutes, with only 6.12 GiB sampled peak RSS. No paired
accuracy result exists. A documented resource-only retry raises that guard
from `2**26` to `2**28` elements, preserving the 128 GiB process budget,
90-minute deadline, compression settings and all scientific inputs. Its
status must not be inferred from the depth-three result.

The extended RobustState audit has completed 32 independent 400-basis trials
per target after the same 500-step warm stage. Mean gradient cosines are
0.01811 (Haar; 95% interval [-0.02599, 0.06221]) and 0.00952 (TFIM critical;
[-0.04369, 0.06272]). Mean predicted Adam descent intervals also contain zero.
Thus even 400 bases have not established reliable average feedback at these
checkpoints. The reference remains the published surrogate with exact noisy
feedback, not the true noisy-fidelity gradient. This is not evidence that
fifty-step stochastic training must fail or that the published method loses.
Each feedback calculation took about 330 seconds on one remote thread.
The 400-basis budget is an explicit twelve-qubit extrapolation, not a claimed
published twelve-qubit setting or a change to our own CTJM trajectory count.

The full comparison is not cleared by these results. Retain the compression
and sampling limitations explicitly; do not replace sampled feedback with an
exact oracle, exclude an inconvenient competitor, or relax the accuracy
criterion and retrospectively label the original check as passed.
