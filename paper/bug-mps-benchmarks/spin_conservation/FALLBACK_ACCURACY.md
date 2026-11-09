# Accuracy-controlled fallback experiments

These experiments ask whether fallback is needed while a simulation is still
accurate. The prepared plan has **not been executed**. Creating a plan reads old
metadata and hashes input files; it does not evolve states, diagonalize a
Hamiltonian, or compute the old-event accuracy audit.

## Prepared protocol

All new trajectories use BUG to `T=8`, hard bond caps, zero SVD cutoff, conservation
tolerance `1e-13`, and Krylov tolerance `1e-14`. Each configuration compares
uncorrected compression, the original scalar correction, and scalar correction
with fallback, starting from identical tensors.

1. **Audit the five saved events.** Reconstruct the independent reference at the
   actual correction time, measure pre-fallback error, replay the correction,
   and measure post-fallback error. Verify the original counters, final energy,
   gauge, and bond dimensions. Retain the old final infidelity separately.
   A BUG correction after the first sweep is at `(step - 1/2) * dt`; the second
   is at `step * dt`. In particular, the old `step_end_time` values `1.8`, `2.1`,
   and `2.06` correspond to physical event times `1.75`, `2.05`, and `2.05`.
2. **Converge the four distinct eventful physical setups.** The two original
   timesteps of the L=8 Haldane–Shastry case share one convergence grid. Reuse
   the saved initial tensors and saved Hamiltonian scaling exactly.

   | Setup | Caps | Timesteps |
   |---|---|---|
   | L=8, ring power law, alpha=1, Delta=1 | 4, 8, 16 | .1, .05, .02, .01 |
   | L=8, ring power law, alpha=1, Delta=2 | 4, 8, 16 | .1, .05, .02, .01 |
   | L=6, Haldane–Shastry | 4, 8 | .1, .05, .02, .01 |
   | L=8, Haldane–Shastry | 8, 16 | .1, .05, .02, .01 |

   These are **40 configurations / 120 trajectories**, all domain walls. Probe
   every arm at the original event times even when no new fallback occurs.
   Full Schmidt-rank caps are explicitly labeled as controls.
3. **Larger systems.** Use `L=12,16`, caps `16,32,64`, and `dt=.02,.01` for:
   - Haldane–Shastry domain walls (`power_a2_d1`, ring chord couplings).
   - Open dimerized antiferromagnetic Heisenberg chains with weak/strong bond
     ratios `.05,.2,.5`, quenched from a product of nearest-neighbor singlets.

   These are **48 configurations / 144 trajectories**. For the dimer chains,
   `H = sum_i J_i S_i . S_(i+1)`, with `J_i=1` for even zero-based `i` and the
   selected weak coupling for odd `i`, using spin-half operators. The singlet
   product is the ground state at weak coupling zero. The actual quench state
   has fixed seed `20260812` and tiny `1e-10` tensor padding to bond two across
   inter-dimer cuts, followed by right QR canonicalization and normalization.
   This permits rank growth; all caps use exactly the same prepared state.
   Its infidelity to the exact singlet product is recorded. The domain walls
   use the existing seeded rank-two preparation.

In total: **88 configurations / 264 trajectories**, plus the five-event audit.
There are no cap-one configurations. L=12, cap=64 is a full-rank control; L=16,
cap=64 still truncates. The studies may return no accurate fallback events.

## References and interpretation

Normalize each Hamiltonian to spectral width per site one. New models use
matrix-free extremal eigensolves with fixed starting vectors and checked Ritz
residuals; convergence reruns and audits reuse the old scale. Record both scale
and normalization diagnostics. The independent full-state reference applies
spin terms through bit flips, with site zero the least significant bit. Check
its action against the MPO on two deterministic full-state probes. Unit tests
also check full matrices and dense exponential propagation on tiny systems.

The reference uses `expm_multiply` with exact trace zero. It streams half-step
vectors without constructing a dense square Hamiltonian or retaining the full
reference time history. Each comparison arm regenerates its reference stream.
Abort if reference norm drift exceeds `1e-9`, or energy drift exceeds
`1e-9 * max(1, abs(initial energy))`. These are numerical sanity checks, not a
mathematical error certificate. Runs are serial and can be expensive at L=16;
no runtime estimate has been measured for this new grid.

Measure normalized global infidelity, all local `Sx,Sy,Sz`, and all
nearest-neighbor `SxSx,SySy,SzSz` correlations. The default accuracy criterion
requires both infidelity <= `1e-3` and maximum absolute observable error <=
`1e-3` in spin units. These are adjustable study thresholds, not a universal
accuracy standard. Reference and BUG trajectories start from the actual
prepared tensors, including padding.

For **every correction call**, measure accuracy **before** applying the
correction. This supplies an unbiased denominator for the conditional rate:

- All-call rate: fallback attempts / correction calls.
- Accuracy-qualified rate: attempts whose inputs satisfy both thresholds /
  all correction calls whose inputs satisfy both thresholds.

Calls already satisfying conservation remain in these denominators. A zero
qualified denominator is reported as undefined. An accurate input whose
fallback produces an inaccurate output still counts as a qualified attempt;
post-correction error is shown separately. Preserve scan recovery, full-frame
recovery, infeasibility, and the other numerical failure outcomes separately.
An event being accurate locally does not imply the whole trajectory is accurate.

Compare adjacent caps at fixed timestep and adjacent timesteps at fixed cap,
matching the physical parameters and initial vector hash. Report observable
differences at common physical times and independently report reference errors
at every recorded endpoint. Observable agreement alone is not evidence of
accurate or asymptotically converged states. Missing runs and incomplete
common-time coverage are explicit. The old-event probes also permit direct
comparisons at those half-step times, across all three arms.

## Commands for later use

From the repository root in an environment with YAQS and the benchmark
requirements installed, select the runner:

```sh
export PYTHONPATH=src
export NUMBA_NUM_THREADS=1
runner=paper/bug-mps-benchmarks/spin_conservation/fallback_accuracy.py
```

Metadata-only planning (safe without starting experiments):

```sh
python "$runner" plan \
  --source ../fallback-stress-2026-09-27/main \
  --output ../fallback-accuracy-plan-2026-09-27.json
```

The existing prepared plan already has that name. A plan is never overwritten.
Use another filename to change the thresholds via `--infidelity-tol` and
`--observable-tol`; the reference tolerance is also stored in the plan.

The following are **execution commands for later**, not part of planning:

```sh
python "$runner" audit \
  --plan ../fallback-accuracy-plan-2026-09-27.json \
  --output ../fallback-accuracy-audit

python "$runner" run \
  --plan ../fallback-accuracy-plan-2026-09-27.json \
  --study convergence --output ../fallback-accuracy-runs

python "$runner" run \
  --plan ../fallback-accuracy-plan-2026-09-27.json \
  --study larger --output ../fallback-accuracy-runs

python "$runner" report ../fallback-accuracy-runs
```

`--limit N` selects the first N configurations in a chosen study; omit it to
complete that study. Repeating an identical command resumes completed cases;
interrupted cases without a final result are recomputed. The output manifest
pins the full plan, implementation hashes, Python/platform and dependency
versions, and archives the source. Changed inputs or protocols require a fresh
output directory. Old source files are read-only and outputs cannot be nested
inside the source results. No scheduler or automatic launch is installed.

The audit writes `audit.json`. The runs save initial tensors, pre/post fallback
snapshots, stage counters, event-time errors, original-event probes, full-step
observable/error traces, and reference checks in each case's `result.json`.
The report writes `accuracy_summary.json`, `accuracy_events.csv`,
`accuracy_probes.csv`, and `ACCURACY_REPORT.md`. Timing includes diagnostics and
reference propagation and is not an algorithm speed comparison.

## Distributed execution

`fallback_dispatch.py prepare --plan PLAN --output NEW_OUTPUT` pins the code,
inputs and numerical dependency versions, then creates 264 independent arm jobs
for the full plan. An optional `--estimates` uses the Mac calibration JSON to
queue longer trajectories first. Preparation does not evolve states.

On each Linux host sharing the source and output directories, run
`fallback_dispatch.py host --output NEW_OUTPUT --workers 64 --reserve-gib 96`
with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMBA_NUM_THREADS=1` and `PYTHONPATH=src`. Each spawned worker uses one physical
core at nice level 19; eight physical cores must remain unassigned. Workers
check MemAvailable before claiming another job. Running arms finish normally;
no other user's processes are inspected for control or modified.

Only one `watch --output NEW_OUTPUT` process should collect results. It produces
`dispatch-status.json` and the standard report once every arm has a terminal
result. Individual trajectories publish progress every ten steps. An atomic
claim directory prevents duplicate execution across hosts. Failed jobs retain
tracebacks; interrupted claims are never stolen automatically. Inspect the
claim owner and host log before manually resuming interrupted work.

All arms independently reconstruct deterministic normalized Hamiltonians and
initial states. Collection checks initial-vector hashes, specifications,
thresholds, variant labels and Hamiltonian scales before combining triples.
Each result records its worker host, runtime, versions and CPU affinity.
