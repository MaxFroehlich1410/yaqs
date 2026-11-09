# Reproducing the conservation manuscript

This opt-in pipeline reproduces Tables I-III and Figure 1 of the manuscript dated
September 27, 2026. Nine unique trajectories cover all existing table rows and
curves. Four additional scalar energy trajectories compare recovery disabled and
enabled for BUG and two-site TDVP. The two uncorrected baselines are shared, so
there are **13 trajectories, not separate reruns for each output**.

The validated, completed dimer accuracy study remains a separate source of data
for the new fallback subsection. Its normalization, initial states and archived
implementation must remain explicit; it is not rerun by this pipeline.

## Unchanged physical and numerical protocol

- Open isotropic Heisenberg chain, L=16, Hamiltonian scale 1 (no bandwidth rescaling).
- Rotated Neel state, tilt pi/4, original fixture seed and noisy initial bond 4.
- T=6, dt=0.01, maximum bond dimension 32.
- Relative discarded-weight threshold 1e-12, Krylov tolerance 1e-12, conservation
  tolerance 1e-13 with the original target scaling.
- BUG corrects twice per timestep; TDVP corrects once per timestep.
- The original joint solver and one-center/every-center schedules are retained.
  Scalar fallback is not added to the joint solver, even if its active constraint
  set temporarily reduces to one. A sequential scalar repair is not a substitute
  for satisfying simultaneous constraints.

The manuscript's original `l16_joint_table.make_params` and
`l16_joint_table.install_correction` construct the schedules. `l16_observables`
provides the original Table III measurement convention, including its correlator
center index. `plot_trajectory.py` regenerates Figure 1 without modifying its style.

## Explicit stages

Use one Python environment with pinned numpy, scipy, numba and qiskit versions.
Set `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, and `NUMBA_NUM_THREADS=1` before starting Python.

From the repository root:

```sh
runner=paper/bug-mps-benchmarks/spin_conservation/paper_reproduction.py
python "$runner" plan --output /absolute/path/plan.json \
  --manuscript /absolute/path/current-manuscript.pdf
python "$runner" prepare --plan /absolute/path/plan.json \
  --output /absolute/path/new-study
```

Planning performs no numerical work. Preparation creates the manifest and source
archive and refuses an existing output directory. After preparation, changes to
any production source, benchmark script, or numerical dependency invalidate the
workers. No implementation should be edited in a prepared run directory.

For a local serial check:

```sh
python "$runner" reference --output /absolute/path/new-study
python "$runner" run --output /absolute/path/new-study --arm bug-none-none
```

Run each remaining identifier in the plan exactly once, then:

```sh
python "$runner" report --output /absolute/path/new-study
```

For a deliberately small validation fixture, plan with
`--length 4 --cap 4 --dt 0.05 --total-time 0.1`. Its reports explicitly identify a
smoke protocol and are never compared with the published L=16 numbers.

## Reference and trajectory records

One shared `expm_multiply` reference is evaluated at all half steps and stored as
a read-only memory-mapped array. At L=16 and T=6 this occupies about 1.17 GiB.
Every arm loads the same saved initial tensors. Source, reference, input and
result hashes guard against accidental mixing of configurations or code versions.
The reference checks independent MPO actions, norm/energy drift and agreement
between the streamed endpoint and a one-shot propagation.

Each trajectory records every full timestep, all pinned invariants, infidelity,
and the four observable families in Table III. Final errors and maxima over time
are distinct. The original table values are transcribed from the supplied PDF;
non-round-off comparisons allow half the last printed digit, and round-off
entries use a 1e-12 comparison floor. Discrepancies are reported, not hidden or
silently accepted. Original numeric Figure 1 data were not supplied, so curve
regeneration does not certify pixel-level agreement with the old figure.

For joint runs, `joint_unresolved` counts individual center solves, while
`schedule_unresolved` counts complete correction schedules still above tolerance.
These are different denominators: an intermediate center can fail before a later
center restores the constraints. Joint counters are not scalar fallback rates.

For scalar runs, all calls and accuracy-qualified calls are reported separately.
Qualification requires pre-correction infidelity <=1e-3 and maximum local-spin or
nearest-neighbor same-axis correlation error <=1e-3, using the actual BUG half-step
or TDVP endpoint reference. Every fallback stores before/after tensors, the exact
time, input accuracy, recovery stage, and checks of bond dimensions and gauge.
Scan recovery, full-frame recovery, infeasibility and numerical failures remain
separate. A zero qualified denominator is undefined, not a zero probability.

Timing separates measurement diagnostics from instrumented evolution and
correction calls. The tracker itself adds work, particularly to joint solves, so
these timings must not support a claim of negligible correction overhead without
a separate uninstrumented benchmark.

## Shared queue and safe server distribution

```sh
dispatch=paper/bug-mps-benchmarks/spin_conservation/paper_reproduction_dispatch.py
python "$dispatch" host --output /absolute/shared/new-study --workers 7 --reserve-gib 96
```

An explicit `host` invocation on a second server can use six workers on the same
shared study. There are only 13 independent trajectories; more than 13 workers
cannot accelerate this workload. A single server with 13 free physical cores can
also run all trajectories concurrently. The slowest every-center joint arm sets
the critical path; each time evolution is sequential. Work is ordered with the
expensive joint sweeps first.

The queue first assigns one reference job. Trajectories become eligible only
when its completion record exists. Atomic directory claims prevent duplicate
simulations. Each worker uses one physical core, runs at nice level 19, and waits
before claiming work if available memory is below the specified reserve. At
least eight physical cores and 64 GiB of memory are left outside admission limits.
Optional `--arms` with comma-separated plan identifiers restricts a host to an
explicit assignment. Cover every identifier exactly once across the hosts. This
can place the three expensive every-center arms on different servers.
These bounds do not replace a fresh load check before launch. Existing processes
are never signaled, reprioritized or modified. Failed or interrupted claims are
never automatically stolen or restarted.

```sh
python "$dispatch" status --output /absolute/shared/new-study
python "$dispatch" watch --output /absolute/shared/new-study
```

The collector fails explicitly if a job fails. It produces the final report only
after every required trajectory is complete and checksummed. A claimed job with
no completion record is reported as `claimed_or_interrupted`, never complete.

## Outputs

- `report/REPORT.md`: Tables I-III, figures, limitations and comparison status.
- `report/trajectory.pdf` and `.png`: regenerated manuscript Figure 1.
- `report/scalar_ablation.pdf` and `.png`: BUG/TDVP energy-only comparisons.
- `report/table_i.csv`, `table_ii.csv`, `table_iii.csv`: corresponding final values.
- `report/trajectory_metrics.csv`: final errors and maxima over the full trajectory.
- `report/correction_counts.csv`: stage counts, denominators and unresolved schedules.
- `report/published_comparison.csv`: every printed-number comparison for production runs.
- `report/validation.json`: completion status and explicit discrepancies.
- `arms/*/result.json`, `final.npz`, and event snapshots: raw reviewable results.
- `manifest.json` and `source.tar.gz`: pinned protocol, dependencies and implementation.

Tests live in `tests/core/methods/test_paper_reproduction.py`. They exercise all
13 real solver configurations on a tiny chain, compare them against direct
legacy evolution, check half-step events and pre-correction qualification, and
verify queue exclusivity and report completeness.
