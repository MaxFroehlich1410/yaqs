# Overnight resource accounting

`resource_accounting.py` performs a read-only audit of signed experiment artifacts. It never constructs a target, evaluates a circuit, trains a method, or selects a better checkpoint. Its output is a checksummed `LEDGER.json` and a flat `EVENTS.csv` for the same physical events. The source hash of the accounting adapter is included in the result.

## What the total means

The principal number is the **subtotal of recorded physical CPU measurements**. It includes every represented method and setting, rejected repair branches, durable failed/interrupted attempts, unused diagnostics, and actual reruns. It is not a claim to know complete project compute. Historical construction, unrecorded interrupted work, and engineering work have unknown timings. `complete_development_cpu_seconds` therefore remains null.

Each field reports `known_subtotal`, `unknown_count`, and `total`. `total` is null when any contributing event lacks that field. In CSV, an empty cell means unknown, never zero. A measured zero means the operation was cached or the measured procedure deliberately did no work of that kind. Counts absent from older instrumentation remain unknown even when CPU was recorded.

Three views are deliberately nonadditive:

1. Full-procedure charges describe what each competing procedure must pay, including a shared constructor or warmup. They do not imply another physical execution.
2. Reused reference endpoints carry their original procedure costs, but no new training is charged to the later panel. Fresh scores of those endpoints are physical work and are charged.
3. Covering invocation timers and nested diagnostic/metric timers describe overlapping scopes. The ledger either partitions a covering timer into disjoint components plus a residual, or records it only as a nonadditive observation.

Physical identity uses the signed run/operation identity, durable attempt identity or index, and measured costs. Source paths and reporting annotations are not execution identity. Copies of identical reports merge their proof references; different durable attempts still count separately, including equal-duration attempts. Repeated final/common-CPU views of one score cache count once. A separate score operation at identical parameters counts again.

This is content-integrity authentication, not a digital signature from a secret signing key. The frozen protocol identifies source SHA-256 hashes and benchmark fingerprints; each JSON record verifies its content checksum, and the complete panel index must cover the expected case grid exactly. Child attempt and score checksums are verified wherever provided by their schema. Structural schemas that precede child hash indexes retain both the authenticated case aggregate and separately authenticated primitive attempt files as evidence.

## Units

- An exact individual-channel gradient is a forward density propagation and its adjoint: two density direction passes.
- An exact forward fidelity score costs one forward channel call and one direction pass.
- Finite-deletion ranking uses prefix/suffix propagation. It has its own ranking-channel count and two direction passes per channel; it is not an ordinary gradient call.
- A joint two-profile gradient costs two individual-channel gradient calls and four density direction passes.
- Statevector trajectory paths and direction passes are separate units. Weighted gradients use forward/adjoint work; adaptive online updates have their own additional propagation. Where instrumented, attempted and completed paths and local operator applications are recorded separately.
- Coordinate sweeps, geometry metric row applications, and MPS optimizer updates are separate units. They are never converted into fictitious channel-gradient equivalents.
- A failed call can be attempted but not completed. Completed direction counts exclude unfinished calls. Older conservative reserved-update counts are observations labeled as bounds when a block failed; they are not advertised as measured propagation counts.

All12 endpoint scores are normally twelve forward calls. The original runner, synthesis through that runner, strong local baseline, and compilation drivers also execute an independent local-channel Qiskit audit: thirteen forward calls. The all11 noisy metric is a performance aggregation, not a computational cost conversion.

## Explicit frozen-schema adapters

| Panel/schema                                                     | Physical CPU and count source                                                                                                           | Reuse and scope rule                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wave1`, `focused_curvature`                                     | Per-run retained optimization, diagnostic and final-audit timers; recorded path/direction counters; worker peak RSS                     | Cached results add no work. Checkpoint I/O and discarded uncheckpointed work were not completely instrumented. Invocation wall is a covering observation.                                                                                                                                                                                                       |
| `overnight.exact_structural_lbfgs.v1` (seed11 and seed29)        | Signed case physical training aggregate, every branch attempt, and whole-search attempts; fresh all12 evaluations                       | Warmup is physically once and charged to both arms. Search timers include nested repairs and replay/cache I/O; repair CPU is not added again. Historical endpoints have fresh evaluations but no new historical training. More than one search attempt leaves retained ranking/score counts incomplete.                                                         |
| `overnight.rank_two_development_panel.v1`                        | One physical synthesis, each fresh runner result, every outer invocation attempt                                                        | Charge synthesis once physically and to all three procedures. Reused focused-panel scores incur no new scoring. Outer CPU minus retained runner components includes I/O and any discarded replay; its uninstrumented call count remains unknown.                                                                                                                |
| `yaqs.overnight.svrg_adam.v1`                                    | Raw `ATTEMPTS.json` counters and invocation CPU; training, diagnostic, evaluation, residual bookkeeping partition                       | Every exact anchor refresh is charged. Correction and weighted path counts include all actual propagations. Before-call counters on incomplete attempts are not called completed work.                                                                                                                                                                          |
| `overnight.ensemble_objective_comparison.v1`                     | Per-method raw exact-gradient attempts, initial all11 gradient diagnostic, unique signed score cache                                    | Final and common-CPU views share cache costs. All local, noiseless, mean2, and mean11 methods remain in tuning cost.                                                                                                                                                                                                                                            |
| `overnight.strong_structural_baselines.v1`                       | Four fresh arm aggregates, shared adaptive prefix, raw block/search attempts, fresh endpoint scores                                     | Adaptive warmup is physically once. Copied structural/fixed L-BFGS training and full-cap scores are reference-only. All six common-CPU endpoints are newly evaluated. Failed block counts remain reserved bounds.                                                                                                                                               |
| `overnight.pure_geometry_comparison.v1`                          | Raw channel/metric/solve attempts and unique score cache                                                                                | Outer optimization CPU already includes metric, solves and checkpoint I/O. Metric/solve timing breakdown is nonadditive.                                                                                                                                                                                                                                        |
| `overnight.physical_compilation_panel.v1`                        | Compilation timer, independent ideal-state audit, two fresh endpoint scores, invocation residual                                        | Source construction/refinement costs are references. Compilation performs no refinement. Failed invocations retain their measured outer cost and unknown internal breakdown.                                                                                                                                                                                    |
| `overnight.ensemble_structural.v1`                               | Every indexed stage attempt, stage invocation, ranking/score attempt, final/common score attempt                                        | Each arm executes its warmup independently. Every failed or rejected branch is included. Stage invocation residual adds measured wrapper work without duplicating optimizer time.                                                                                                                                                                               |
| `yaqs.overnight.exact_anchor_diagnostic.v1` (original and typed) | Four exact snapshots, shared setup, displaced exact oracles, all paired CV/weighted estimates at every distance                         | Both physical diagnostic executions are charged. They reuse the same scientific streams and do not provide independent replication. Each execution reuses its own anchor across settings. Distance-zero oracle reuse costs measured zero.                                                                                                                       |
| `overnight.legacy_construction_development_batch.v1`             | Batch protocol, case manifest, signed AUDIT/RESULT, every constructor attempt                                                           | Four fresh MPS reproductions count once. Internal MPS diagnostics are already in construction CPU. New external state audits and measured wrapper residual are additional. Historical constructor CPU stays null.                                                                                                                                               |
| `overnight.entropic_risk.v1`                                     | All six raw training attempts, exact final/common operation attempts, case invocation residual                                          | Both temperatures and both mean controls count even after the risk family is retired. Every exact component gradient is charged.                                                                                                                                                                                                                                |
| `overnight.mean2_stochastic_baselines.v1`                        | All artifact byte hashes, stochastic stage attempts, exact ranking/score operations                                                     | Four new controls execute warmups independently. Three copied deterministic arms add no training; their new common-CPU scores do add work. Untimed outer case bookkeeping remains unknown.                                                                                                                                                                      |
| `overnight.initializer_resource_policy_development.v1`           | Shared initializer attempts and components, four fresh training/deployment ledgers, unique score cache, complete dependency byte hashes | Archived legacy construction is charged to declared procedures but never physically added again. New rank-two probe and compilation count once. Accepted-point deployment compilation is separate from optimization CPU. Scores cost thirteen forward calls including the independent audit. Unavailable common-total-CPU points remain explicitly unavailable. |

The original estimator protocol expects the pre-typing source hash. Only its `frozen_driver.py` with exactly that expected SHA-256 is accepted as a substitute for `estimator_diagnostic.py`. The current typed driver is authenticated normally. `ENGINEERING_NOTE.json`, `estimator_reproduction_audit.json`, `ANALYSIS.json`, plots and scoreboards are derived artifacts, not extra scientific executions. An earlier stdout-only or unrecorded failed invocation cannot be assigned fabricated compute.

The fresh construction reproduction is available as a separate declared-procedure cost reference. It does not retroactively replace unknown historical construction CPU, and summing all training procedure charges would multiply shared construction work incorrectly.

## Completeness and failures

The expected targets, replicas, and methods come from each known protocol schema. Missing whole cases, missing methods, changed source bytes, mismatched completion hashes, and unsupported schemas abort the audit with a clear error. No apparently complete smaller total is emitted. Durable failed branch or oracle attempts within supported completed panels are retained. An unsupported terminal failed-case layout likewise aborts and requires an explicit adapter; it is not silently omitted.

The adapters intentionally enumerate every frozen arm rather than the eventual winning method. Raw score caches and indexed primitive-operation files are checked for unindexed work. Unknown future protocol versions require another adapter before discovery can claim coverage. Non-panel directories such as figures, validation, and an output ledger directory without a protocol are not scientific panels.

Peak RSS is the maximum observed worker process high-water mark, not a sum, incremental per-job memory, or concurrent machine peak. Older/missing memory observations stay unknown. Summed wall durations are worker elapsed time, not parallel experiment makespan. Outer wall observations must not be added to their inner timers.

Engineering shell commands, listings, literature work, edits, lint, tests, exports and unrecorded failed attempts remain a separate unmeasured category. The accounting pass itself measures and reports its read-only CPU/wall as engineering work; it does not add that cost to scientific events. No engineering time is inferred from an overnight wall-clock interval.

## Usage and validation

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run python -m experiments.overnight_state_preparation_20260913_0108.resource_accounting \
  --discover output/overnight-state-preparation-20260913-0108 \
  --output output/overnight-state-preparation-20260913-0108/resource-accounting
```

Alternatively repeat `--panel PATH` for an explicit frozen subset. The result lists its exact covered panels and proof files, so subset totals cannot be mistaken for the entire development run. JSON is the authoritative artifact; CSV exposes the same disjoint events without hierarchical proof metadata.

Small checksummed-fixture tests verify copy deduplication, distinct attempts, unknown timing propagation, maximum rather than summed RSS, nested timer partitioning, missing case/method/index guards, source and child checksum rejection, complete snapshot reuse accounting across both physical estimator runs, cached evaluation reuse, constructor diagnostic overlap, and failed exact-channel attempts. The initial read-only smoke covered seventeen panels. Subsequent complete collections preserve those physical events and extend the covered panel list; they perform no numerical experiments.

## Later completed diagnostics and composed procedures

The adapter now recognizes the original MPS overlap diagnostic, its separately frozen physical follow-up, long-ensemble6050, and conditional-composition. Each still requires authenticated source bytes, the full protocol case/method grid, completed result checksums, and all durable child-attempt hashes. An unknown additional attempt layout fails closed.

For both MPS diagnostics, each recorded stage is a disjoint diagnostic event. Successful gradient stages contribute eight paths and sixteen directions; successful forward stages contribute eight paths and eight directions. Failed partial path counts remain unknown. Inclusive case attempts contribute only the residual after stage timers. This counts fresh target factorization, regeneration, backends and checkpoint I/O once. Original failed acceptance remains visible and does not erase its physical cost; the follow-up is a separate physical diagnostic, not an independent scientific replication. Total tensor storage of one observed MPS state is not relabeled as process peak memory.

Long ensemble training uses primitive ATTEMPTS, invocation residuals and unique score ATTEMPTS. Copied final or common-CPU points add no training, and shared score-cache entries count once. Untimed outer case construction, target loading and record assembly remain explicitly unmeasured.

Composition partitions its inclusive CASE_ATTEMPTS into exact-gradient stages, stochastic/adaptive stages, finite-deletion rankings, forward scores, final compiler attempts and the remaining bookkeeping. Nested stage invocation timers are nonadditive observations. Reused conditional initializer construction is a declared procedure charge with zero fresh execution in this panel; its original physical source remains separately charged in the initializer/construction panels. Common-primary and all-trained scores share the same physical operation cache.

These extensions do not change the original and typed estimator replay accounting: both physical runs remain charged and are not independent replication. The complete development compute total remains unknown because engineering and some historical costs were never measured. No wall-clock inference fills those gaps.

## Compiler composition and complete confirmation extensions

The common-control audit adds only its new exact score attempts and the remaining wrapper CPU. The source's four structural scores, all previous training and initialization remain nonadditive references. The known-zero compiler panel charges every initial/final source pair: both exact endpoint evaluations, the compiler with its independent ideal-state audit, and the remaining inclusive pair CPU. Neither panel repeats source training costs.

Lazy initializer reproduction charges each actual lazy attempt, including a discarded rank-two probe and a fresh fallback constructor. Nested constructor timers are partitioned from the inclusive lazy timer; internal constructor diagnostics remain included. The new independent equivalence audit and separately timed target input are charged. Earlier interrupted input loads lack durable instrumentation and stay unknown. The external physical-state audit has CPU but no separate wall timer; its wall work remains within the inclusive residual, never imputed as a measured zero.

Compiled composition uses its authenticated compact EXECUTION receipt and adds both generic and state-preparation compiler attempts to the existing complete primitive-stage accounting. A shared initial compiler executes once physically while its declared cost offsets every arm once. Original initializer-policy arrays are references, with no fresh construction cost in this panel. All rejected repair branches and optimizer retries remain charged.

The complete confirmation adapter authenticates the sealed parent phase receipts and exact child file sets. Batch vector preparation is physical work once, with only a nonadditive amortized per-arm charge. Each case partitions its outer physical invocation around input, actual lazy initialization, and training/evaluation phases. The original and compiled kernels retain their measured gradients, trajectories, rankings and compiler costs; phase wrappers contribute only residuals. Per-case worker lifetime RSS is a nonadditive observation rather than a sum or incremental memory estimate. Future schema changes, incomplete confirmation panels and unsupported failed-case layouts fail closed; they cannot silently yield a smaller favorable cost total.

No central replacement ledger is generated until the coordinator authorizes the complete explicit panel list. Previously published ledgers and all physical event identities remain preserved.

## Retained MPS memory and the historical 27-panel ledger

The `overnight.one_path_retained_mps_memory.v1` adapter requires the complete
eight-endpoint development diagnostic, every raw measurement and completion
seal, and the entire child-file registry. Each endpoint contributes its inclusive
attempt CPU once. Successful and failed gradient stages, input and setup,
retained-memory inspection, target and norm audits, and residual bookkeeping
partition that attempt. A single successful gradient here is one path, unlike
the eight-path gradients in the older overlap diagnostics. Unknown stage or
attempt durations block a complete total.

Retained forward-tape tensor storage, the exact target, their deduplicated union,
and process RSS are separate observations. Tape payload is never substituted for
peak RSS or counted as extra compute. The eight successful endpoint attempts used
1.032438 CPU seconds. Their 0.367551 seconds of MPS gradients, 0.027735 seconds
of dense gradients, and 0.042394 seconds of inspection are nested in that total.
All fixed paths had no selected noise factors; these measurements do not estimate
full weighted-batch cost or a q12 speedup.

The preserved historical 27-panel ledger is
[`resource-accounting-27panels/LEDGER.json`](../../output/overnight-state-preparation-20260913-0108/resource-accounting-27panels/LEDGER.json).
Its content checksum is
`0e71745c8758f9f9bd890a9ea29a886a8a62d09f5e13bc0006393971fb6ac1fe`.
It contains 10,624 disjoint physical events and preserves all 6,674 event
identities from the 25-panel ledger. Confirmation adds 2,818.474129 CPU seconds;
the memory diagnostic adds 1.032438 seconds. The measured scientific CPU subtotal
is therefore **13,718.222034 seconds (3.810617 hours)** across all represented
development, confirmation and diagnostic work.

| Disjoint category | Measured CPU seconds |
| ----------------- | -------------------: |
| Optimization      |         12810.036432 |
| Construction      |           335.585177 |
| Diagnostic        |           206.629355 |
| Evaluation        |           120.507105 |
| Compilation       |            11.536556 |
| Setup             |             2.230784 |
| Bookkeeping       |           231.696625 |

This is not complete overnight CPU: `complete_development_cpu_seconds` remains
null, including after confirmation. Per-operation count fields also retain
their own unknown coverage, so their known subtotals must not be combined into
one total number of evaluations or compared as if they shared a denominator.
The ledger's read-only accounting pass used 80.080853 CPU seconds and
112.582478 seconds wall time, reported as engineering work outside the scientific
subtotal. See the
[collection audit](../../output/overnight-state-preparation-20260913-0108/validation/RESOURCE_27PANELS_AUDIT.json)
for preserved event identities and the corrected coordinator-only postcollection
reduction; neither scientific work nor the accounting collection was repeated.

## Independent confirmation verification

The separately frozen `overnight.independent_sealed_confirmation_validation.v1` adapter retains all 210 saved roles and 2,520 role/profile rows, even for numerical failure or budget incompleteness. It reconstructs the complete saved input panel through read-only parent APIs, authenticates source bytes and nested call receipts, and checks public rows against the raw independent measurements. It never invokes the independent oracle or its calibration routine during accounting. Extra durable attempts, missing role products, unknown interrupted cost and changed public cost receipts fail closed.

Preparation attempts and per-profile call attempts are disjoint physical diagnostic events. Inclusive run CPU and wall time contribute only their residual after all profile attempts; endpoint invocation clocks are covering observations. Thus initial calibration, saved-input authentication, numerical replay, failed/interrupted work, I/O and row reduction are charged once. Identical endpoint/profile reuse reduces physical call counts without reducing the 2,520 reported comparisons. Independent oracle forward calls are separate from the already charged original training and endpoint scoring. A failed partial oracle has one attempted call, zero completed calls and an unknown completed-direction count.

The adapter accepts a sealed budget-incomplete result only when every expected row remains explicit, and retains its failed acceptance. Unknown hard-kill clocks still abort. Final receipt publication, later cache authentication and whole-project engineering time remain outside the original measured invocation. Absent process RSS remains null; no memory figure is inferred from density dimensions or runtime. Synthetic fixtures cover full logical-panel reuse, inclusive timer partitioning, failed preparation, interrupted oracle replay, numerical failure, budget omissions, missing costs, extra attempts and re-signed public-row corruption.

## Final 28-panel ledger

The final [28-panel ledger](../../output/overnight-state-preparation-20260913-0108/resource-accounting-28panels/LEDGER.json)
contains **12,318 disjoint physical events** and a measured scientific CPU subtotal
of **13,765.937587 seconds (3.823871552 CPU hours)**. Its content checksum is
`925da91a2d27f29ea17d0246edce2025349bf5fdacaef91ff8fb9f5102e2106d`.
All 10,624 events from the historical 27-panel ledger are preserved exactly;
the independent confirmation verification adds 1,694 events and 47.715553 CPU
seconds. Its 1,692 unique oracle calls serve all 2,520 saved comparisons, with
preparation and inclusive run residual counted once. Original and typed
estimator executions remain charged separately; neither reused training nor
annotation-only reports add another physical execution.

| Disjoint category | Measured CPU seconds |
| ----------------- | -------------------: |
| Optimization      |         12810.036432 |
| Construction      |           335.585177 |
| Diagnostic        |           254.344908 |
| Evaluation        |           120.507105 |
| Compilation       |            11.536556 |
| Setup             |             2.230784 |
| Bookkeeping       |           231.696625 |

Only the diagnostic category changed from the historical 27-panel collection.
The [reconciliation audit](../../output/overnight-state-preparation-20260913-0108/validation/RESOURCE_28PANELS_AUDIT.json)
confirms that all 12,318 CSV measurement rows match the authoritative JSON events;
its content checksum is
`fc7cbfdaee4acbbd1e68440078e7b3a3d1248b1b13435f64036929cf1ae2f80e`.
The read-only collection used 87.241795 CPU seconds and 149.320147 seconds wall
time. The separate reconciliation used 0.271477 CPU seconds and 0.312168 seconds
wall time. These are measured engineering audits outside the scientific subtotal.

Complete overnight CPU remains unknown: `complete_development_cpu_seconds` is
null, and unrecorded engineering or historical work is not inferred from elapsed
time. Per-operation counts retain their different missing-data coverage. Recorded
worker RSS remains an observed process high-water mark, with unknown observations
left null; it is neither summed memory nor a complete simultaneous machine peak.
The final automated cohort includes all 52 resource-accounting tests among
1,409 passing owned tests. The accounting code and scientific sources were not
changed during this final documentation update.
