# Optimizer versus trajectory-budget screen

Frozen development protocol, 2026-09-12, before scientific execution.

Execution complete: see the [results and weighted follow-up](state_preparation_sampling_budget_findings.md).

Run step 1 of the post-diagnostic plan on the same four old development targets,
frozen construction-replica-11 circuits and initial parameters. No new targets,
construction, pruning, noise strengths or rate selection. Preserve the calibrated
local Pauli-Lindblad channel and exact Bernoulli-factor sampler. This screen does
not claim device performance, general nonunitary-jump support or an optimum.

## Primary comparison

For each target and noise replica 11/29, run 50 updates with fresh maps every
update at R=32 and R=128. Use the old development root seed `2026091101` and
unchanged seed derivation, so methods share paths and R=32 is a nested prefix of
R=128 at a fixed update. All training is complex128 statevector computation.

| Method            | Updates 1-25  | Updates 26-50                     |
| ----------------- | ------------- | --------------------------------- |
| All-online Krotov | Online at 0.7 | Online at 0.7                     |
| Hybrid Krotov     | Online at 0.7 | Independent batch gradient at 0.3 |
| Trajectory Adam   | Adam at 0.02  | Adam at 0.02                      |

Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, zero initial moments, global
bias-correction counter. The rate 0.02 is the earlier selected development rate,
not the untested adaptive-screen default 0.005. No adaptive Krotov tail is added.

There are 48 primary target/method/R/noise-replica cases. Reuse the 16 compatible
R=32 all-online and hybrid-0.3 cases from the completed 2026-09-11 tail screen,
after authenticating source manifests, cases, runtime and record signatures.
Preserve original timings and explicit provenance; do not call these new runs.

## Controls

- Eight R=32 hybrid cases with the same online prefix but batch tail **0.7**.
  Comparison with all-online 0.7 isolates continuation ordering at matched rate.
- Four exact-noisy Adam references, rate 0.02 and 50 updates, one per target.
  Both noise replicas share the same starting circuit, so repeating a
  deterministic reference would create no independent evidence. These use the
  existing density adjoint and are attainable references, not global optima.

Total: 60 result rows, including 16 reused cases and 44 newly executed cases.

## Evaluation, cost and decision

Use final update-50 exact channel fidelity, never best-iterate selection.
Trajectory curves retain an independent exact density score after every update;
exact-Adam references retain the established scores at 0,10,25,50. All records
pair scored iterates with cumulative training CPU. Report fidelity versus
training CPU as well as equal-update endpoints. Do not substitute concurrent
elapsed time for CPU or hide sampling cost. Exact scoring, setup and I/O are
excluded from training cost; reused timings are observational from the earlier
run. Exact-reference scoring reports wall time, not an invented CPU measurement.

Report paired R=128-minus-R=32 and Krotov-minus-Adam differences, family means,
and consistency across noise replicas. One target per family is insufficient
for a population-level generalization claim. Do not use the old confirmation
panel for further tuning, and do not launch new targets automatically.

After this screen, assess whether trajectory variance remains a useful target:
inspect R-dependent progress and gaps to the matched exact-gradient reference
alongside the completed gradient diagnostics. If so, prioritize a **weighted
no-factor/at-least-one-factor estimator diagnostic** at saved development
checkpoints, validating unbiased weights and measuring error per CPU second
before any new weighted optimizer sweep. This is a qualitative development
decision, not a predeclared statistical pass/fail test. No broad adaptive,
cross-refinement, backtracking or fresh-target run is implied.

## Execution

```sh
uv run --no-sync python -m experiments.state_preparation_sampling_budget \
  --source-root output/state-preparation-trajectory-comparison-20260910 \
  --tail-root output/statevector-tail-development-20260911 \
  --output-root output/statevector-sampling-budget-20260912 \
  --workers auto
```

Use a new output root; `--resume` requires identical sources/configuration/runtime.
The wrapper leaves the previous drivers and their source fingerprints unchanged.
Worker count may change on resume. All scientific jobs run with one numerical
thread per process; no test or lint process should overlap their timing.
