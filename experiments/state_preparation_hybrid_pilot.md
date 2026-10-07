# Hybrid Krotov screen of the compression-first pipeline

Version `yaqs.compression_hybrid_screen.v1`, specified 2026-09-06 before launch.

The preceding [batch-only pilot](state_preparation_pilot.md) favored layerwise
construction, magnitude pruning, repair, and noiseless continuation. It did
not test whether an early gate-sequential online phase improves that pipeline.
This screen addresses the author's request for 5–20 online updates and online
learning rates from 0.1 to 0.5. These are newly specified screening settings,
not a claimed reproduction of hyperparameters from the optimization paper.

## Fixed setting grid and common controls

Use the same four six-qubit target vectors, seeds 11 and 29, root seed
2026090520, initial parameters, scalar BMPD architecture, protected initial
U3 layer, exact local Pauli noise, and no-truncation settings as the batch pilot.
Every target is optimized independently. No new target or initialization is
selected based on its performance.

| Setting family | Initial online updates | Online learning rate     | Remaining updates |
| -------------- | ---------------------- | ------------------------ | ----------------- |
| B              | 0                      | Not used                 | All batch         |
| H5             | 5                      | 0.1, 0.2, 0.5 separately | Batch tail        |
| H10            | 10                     | 0.1, 0.2, 0.5 separately | Batch tail        |
| H20            | 20                     | 0.1, 0.2, 0.5 separately | Batch tail        |

There are ten settings, eight target/seed pairs, and **80 independent jobs**.
Each job produces four exact-channel endpoints across the two comparisons
below: 320 endpoints in total. There is no winner-selection stage, early
stopping, retuning, extra restart, or best-checkpoint selection. Report every
setting and target, including negative results.

Online here means the existing **gate-sequential stale-adjoint Krotov sweep**:
compute costates at the old parameters, then update gates in circuit order
using the already updated prefix and old costates. In the noisy version,
average eight independent trajectory signals once at each gate. Do not update
parameters separately for each trajectory. Keep the sampled maps fixed within
one sweep, and resample them at the next update. This is not the `cross`
trajectory estimator.

## A. Hybrid during early layerwise construction

For each setting, rebuild the best magnitude-pruning pipeline from the same
initial master vector as the batch pilot:

1. Grow through depths 1, 2, 3, 4. At **each** depth, run 100 total noiseless
   updates: the setting's first 0/5/10/20 updates are online at its specified
   rate, and all remaining updates are batch at constant rate 0.2. The online
   prefix restarts when the architecture grows. Thus H20 has 80 online and
   320 batch updates over the four source stages, not just 20 online updates
   over the entire pipeline.
2. Perform the unchanged six magnitude-pruning rounds. In each round, delete
   ten one-qubit and five two-qubit gates, then run **20 batch-only noiseless
   repair updates at 0.1**. No hybrid restart is inserted into these short
   repair blocks. Preserve all original gate IDs, scores, quotas, and real
   removal of the gate and its noise window.
3. Record `cold_construct` after 520 total updates. Then run **50 batch-only
   noiseless updates at 0.1**, yielding `cold_final` at 570 updates.

The final circuit has 78 one-qubit and 30 two-qubit gates for every setting.
Pruning may select different gates because the source parameters changed;
that is part of the construction pipeline, not an uncontrolled extra budget.
The source-layer traces contain the cold-start and newly grown-layer learning
curves, including every early online update.

The main contrast is each hybrid `cold_final` minus B `cold_final` within
the same target/seed. This asks whether early hybrid training improves the
previously best complete construction pipeline. It does not rerun the
expensive deletion-impact scorer or the full depth/method benchmark matrix.

## B. Hybrid refinement from identical saved parameters

Take the **original batch pilot's M construction endpoint at update 520** for
each target/seed. Every setting, including B, starts from that exact circuit
and parameter vector; do not substitute the hybrid construction from A.

Fork each start into:

- `warm_0`: 50 noiseless updates, first 0/5/10/20 online and the remainder
  batch at constant rate 0.1.
- `warm_N`: 50 noisy updates with the same phase schedule and batch rate 0.1,
  eight independent trajectories per update, refreshed at every update.

For each hybrid setting the online rate is its grid value. All warm arms
have a nonempty batch tail. Retain the starting parameter checksum and
evaluate only the final iterate. There are 80 noisy and 80 noiseless warm
refinements, totaling 4000 noisy gradient updates / 32000 gradient trajectories.

The noisy seed keys reproduce the original pilot's `refine_M_N` streams and
are shared across settings on the same target/seed. This pairs the random
maps for the comparison while still resampling across optimizer updates; it
does **not** freeze one training ensemble for the entire run. No independent
Monte Carlo evaluation ensemble is needed because endpoint evaluation is exact.

Contrasts are hybrid minus B separately for `warm_0` and `warm_N`, and
`warm_N` minus `warm_0` within each setting. A positive change over the frozen
start alone is insufficient to establish a noise-aware training advantage.

## Interpretation and work accounting

The cold rate-0.2 comparisons and warm rate-0.1 comparisons isolate the switch
and update ordering at unchanged learning rates. Other rates screen a joint
schedule/step-size change. A win at online rate 0.5 cannot by itself prove
that online ordering, rather than the larger early step size, caused the gain;
a subsequent matched batch-prefix rate control would be needed for that claim.

Match optimizer-update budgets, not wall-clock time. The public training API
also evaluates post-update training metrics and one initial metric per chunk;
these extra forward evaluations are included in measured runtime. They are
diagnostics, never selection data. Noisy gradient averaging remains R=8.
For one uninterrupted block with T noisy updates and C chunks, the current
public API samples R*T gradient trajectories, R*T post-update diagnostic
trajectories, and R\*C initial diagnostic trajectories. The latter two sets
do not contribute parameter updates. Saved chunk counts make this cost visible.

This remains a development-data screen with one state per family and two
seeds. Exact channel evaluation removes endpoint Monte Carlo uncertainty,
not training variability or selection bias from comparing nine hybrid settings.
Do not claim general target independence or state-of-the-art performance.

## Reproducibility and execution

The runner uses existing public hybrid-training functions. Constant-rate
training is checkpointed in ten-update chunks with phase boundaries and noisy
iteration counters preserved across chunks and resume. Tests compare chunked
training to uninterrupted public-API runs at switches 5, 10, and 20, including
high-noise fixtures, and test interruption on both sides of the switch.

The recomputed B control must reproduce the original M, M_0, and M_N parameter
vectors, circuits, and noisy fidelities within 1e-11, or the job fails. The
manifest validates source artifacts, source/configuration/fixture checksums,
noise calibration, and dependency versions. Each setting/target/seed owns its
directory; the parent alone aggregates completed jobs. Corrupt or incompatible
output is rejected, and failed jobs are not omitted from a mean.

```bash
uv run python -m experiments.state_preparation_hybrid_pilot \
  --source-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-compression-pilot-20260905 \
  --output-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-compression-screen-20260906 \
  --workers auto
```

Use `--resume` only for the identical interrupted screen. Worker count may
change on resume. All workers have one numerical thread; eight jobs run
concurrently on this eight-core laptop. The original pilot artifacts and
runtime implementation are left unchanged.
