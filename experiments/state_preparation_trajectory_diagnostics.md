# Compact statevector development studies

Implementation protocol, 2026-09-11. This driver implements the four proposed
follow-ups without running them automatically. It is a development screen, not a
held-out confirmation study or evidence of state-of-the-art performance.

Execution follow-up (2026-09-11): gradient/cross diagnostics, the tail screen and
the CRN screen are complete. Adaptive tails were deferred under the planned
gradient-quality condition; optional backtracking was not run. See the
[findings and artifact links](state_preparation_trajectory_diagnostics_findings.md).

## Fixed scientific setup

- Read the completed Experiment 2 output, including its linked original hybrid
  construction source. Authenticate manifests, selected checkpoint checksums,
  configurations, target fixtures and starting parameters. Source files are read
  only; their historical MPS fingerprint is preserved as provenance.
- Use only the four old development targets: Gaussian amplitudes, critical TFIM,
  complex Haar and random bond-2 MPS. Their vectors are dense even for the MPS
  target family. Each uses the frozen compact circuit from construction replica
  11: six qubits, 108 parameters/gates, including 30 two-qubit gates.
- No new construction or pruning. No confirmation targets are read for tuning.
  Each refinement starts at the saved development **update 0**, not update 25 or
  the best observed iterate. Recompute the online prefix with the new paired
  study streams; do not splice a prefix trained with a different noise policy.
- Calibrated homogeneous local Pauli-Lindblad noise is unchanged:
  `r1=0.00064`, `r2=0.0051`, exact independent Pauli-factor sampling after every
  retained gate, unit time window. Input is the zero state; training loss is
  `1 - mean(|<target|psi_r>|**2)`. All training uses complex128 statevectors.
- Every refinement has 50 updates and R=32. Unless noted below, it has 25 online
  sweeps at 0.7 followed by 25 independent-gradient batch steps at 0.1.
- Refinements use two noise replicas (11, 29) per target, with the **same** fixed
  construction parameters. These are noise replicas, not new target states or
  independent construction initializations. Method-independent seeds pair all
  variants; the seed role separates diagnostics from refinement. Root seed is
  `2026091101`. All variants have the same per-target update/path streams where
  their refresh policies agree, including identical prefixes in the tail study.

## Choose one study

The CLI deliberately has no `all` mode. Each study uses a separate new output
directory and its own manifest. No cross refinement is automatically promoted.

| Selection                  | Predeclared work                                                                                     | Default jobs                                       |
| -------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `gradient`                 | Independent gradient at saved updates 0, 25, 50; R=8,32,128; eight independent ensembles per setting | 12 checkpoint jobs, 288 sampled-gradient estimates |
| `gradient --include-cross` | Same diagnostic with both independent and cross signals on identical maps                            | 12 jobs, 576 estimates sharing 288 ensembles       |
| `tail`                     | Batch-tail rates 0.1,0.2,0.3 plus 50 online sweeps at 0.7                                            | 32 refinements                                     |
| `crn`                      | Refresh each update, each five updates, or fully fixed                                               | 24 refinements                                     |
| `backtracking`             | Fixed-rate hybrid baseline versus backtracking in its batch tail                                     | 16 refinements                                     |
| `adaptive`                 | Fixed-rate hybrid, momentum tail, diagonal-scaling tail, Adam tail, and full trajectory Adam control | 40 refinements                                     |

The source reader fixes the four-target panel and predeclares the old
`trajectory_hybrid_r0p7_t0p1` setting rather than choosing another winner.
The Python `Config` accepts smaller engineering budgets; the CLI uses the
scientific defaults above. The generated manifest includes all configuration
values and variant definitions. Nothing is launched merely by importing the
module or reading this protocol.

### 1 and 3: gradient quality and cross-estimator diagnostic

Compare against the existing **exact-density** adjoint at identical parameters.
Check its fidelity against the saved checkpoint before accepting the job output.
For every estimate retain the gradient vector, absolute/relative L2 error, cosine
similarity, gradient norm, exact-gradient dot product, and separate sampling and
gradient CPU/wall times. Ratios/cosines are null for near-zero reference norms,
not misleading zeros. Report bias of the empirical mean, variance trace, mean
squared error, and fraction pointing in a true descent direction across repeats.

Across R, samples are nested prefixes of the same per-repeat stream. Within R,
independent and cross use the **same map object**; execution order alternates
across repeats. Each estimator's cost includes sampling, even when that sampling
was physically shared by the paired diagnostic. Exact reference cost is separate.
Record sampled Pauli-factor counts and factor-free trajectory counts; several
sampled factors can cancel, so these counts do not claim net nonidentity errors.

`MSE * mean CPU seconds` is a descriptive cost/error measure (lower is better),
not a guarantee for averaging a biased estimator. Inspect the individual
checkpoints and bias/variance alongside it, especially near convergence. The
cross signal is not asserted to be the derivative of the same finite-ensemble
loss as independent averaging. Promote cross to full training only after this
diagnostic supports doing so; this driver does not make that decision.

### 2: CRN and optional backtracking

Every forward/backward/updated-prefix calculation replays one ensemble within
an update. Across updates, the ensemble counter is:

```text
fresh:       epoch(k) = k
five-update: epoch(k) = 1 + 5 * floor((k - 1) / 5)
fixed:       epoch(k) = 1
```

Changing parameters does not resample retained maps. On resume, rebuild a retained
ensemble from the original initial-parameter metadata, base seed and global
epoch. Probabilities are parameter independent. Replay reconstruction is logged
as resume/setup cost, not an extra algorithmic ensemble refresh.

Backtracking changes only the batch tail; its baseline and online prefix use
fresh maps each update. Start each tail update at rate 0.3, halve up to six times
(seven candidates at most), and accept the first candidate satisfying

```text
sample_loss(theta - rate * gradient)
    <= sample_loss(theta) - 1e-4 * rate * ||gradient||²
```

Reuse the baseline loss from the gradient forward pass. Every candidate uses the
same maps and every trial objective call, including rejected ones, is timed and
counted. If all trials fail, retain the old parameters and record a rejected
update with effective rate zero. Exact noisy fidelity is **never** the acceptance
criterion. Sample-loss descent is not a guarantee of expected-fidelity descent.

### 4: adaptive tails

All hybrid variants retain the same 25-update prefix. Tail moments are initialized
to zero at the switch; the first tail step has counter **1**, not 26.

| Variant                 | Tail update                                                                     | Rate  |
| ----------------------- | ------------------------------------------------------------------------------- | ----- |
| Hybrid baseline         | `theta -= rate*g`                                                               | 0.1   |
| Krotov-momentum         | `m = 0.9*m + 0.1*g; theta -= rate*m`                                            | 0.1   |
| Krotov-diagonal scaling | `v = 0.999*v + 0.001*g²; theta -= rate*g/(sqrt(v/(1-0.999**t)) + 1e-8)`         | 0.005 |
| Krotov-Adam             | Standard Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, tail-local bias correction | 0.005 |
| Trajectory Adam control | Same Adam from update 1, without an online prefix                               | 0.005 |

The diagonal-scaling variant is a **bias-corrected RMSProp-style** control, not
an assertion of equivalence to every library's RMSProp defaults. Momentum is an
uncorrected exponential average, not the unnormalized heavy-ball sum. These
rates are predeclared exploratory choices, not separately tuned optima. Label
the combined methods honestly; do not attribute an Adam-tail gain to pure Krotov.

## Execution, cost and artifacts

Start with the diagnostic; adding `--include-cross` is optional:

```sh
uv run python -m experiments.state_preparation_trajectory_diagnostics \
  --source-root output/state-preparation-trajectory-comparison-20260910 \
  --output-root output/statevector-gradient-development \
  --study gradient --include-cross --workers auto
```

For a later screen, change `--study` and use a **new output directory**. Use
`--resume` only with the same study/configuration/source/runtime. Worker count
may change; seed identities and scientific manifests exclude it. Independent
jobs use bounded process workers with one numerical thread each. Resume after
a driver change intentionally fails source-fingerprint validation.

Each refinement update is saved atomically, including parameters, moments,
tail counter, noise epoch, event counts, accepted rate, trial rates and work
counters. Diagnostics are atomic per target/checkpoint job: interruption during
one diagnostic recomputes that unfinished job, not previously completed jobs.

Training CPU/wall cost includes map generation, online/gradient/adaptive updates
and all candidate losses. Engine setup and resume replay are stored separately.
Exact noisy scoring runs at update 0 and after every update, recorded separately
and used only descriptively, never to choose updates or stop training. It can be
a substantial fraction of a small dense run's wall time. `trajectory_passes`
counts R times two forward/adjoint sweeps for a batch gradient, three sweeps for
online updates, and one per candidate loss; it is not a count of individual local
matrix products or a memory measurement.

Every output root contains a frozen `manifest.json`, authenticated per-job JSON,
`result.json`, and a readable `summary.md`. The summary uses equally weighted
target/checkpoint or target/noise-replica means, without selecting best iterates.
Raw rows retain paired outcomes and separate scoring/training costs. Assess
across-family consistency before spending compute on new held-out targets.
The reported post-split gain always compares updates 25 and 50 under the default
configuration, including all-online and full Adam. It is a matched time interval,
not a claim that those controls contain a batch tail.

## Verification scope

Automated tests cover all presets, shared seeds/maps, nontrivial-noise resume
inside and across refresh boundaries, all-online and Adam kernel equivalence,
adaptive formulas, Armijo acceptance/rejection and evaluation counts, invalid
inputs, source authentication, source-fidelity checks, multi-target orchestration,
serial/parallel reproducibility, and rejection of incompatible checkpoints.
The tests use small synthetic engineering cases; they are not new research runs.
