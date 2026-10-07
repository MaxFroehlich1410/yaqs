# Completed overnight research handoff

**Status: complete; no candidate passed the frozen q6 confirmation gate.**
All ten confirmation cases and all seven roles are retained. The primary
candidate mean is 88.534223%, versus 88.209294% for the strongest aggregate
control, structural adaptive online: +0.324929 percentage points and only three
of five target wins. The fixed weighted-Adam comparison is +0.439079 points,
also three wins. Both miss the required 0.5-point and all-five-target conditions.
TFIM para and random MPS bond 3 lose. The positive mean is concentrated in Haar;
both TFIM and MPS family means lose to adaptive online. Better full-cap scores
do not rescue this negative result, and no q8/q12 scientific run is justified.

Repository: `/Users/maximilianfrohlich/Desktop/GitHub/yaqs`.
Source package: `experiments.overnight_state_preparation_20260913_0108`.
Output root: `output/overnight-state-preparation-20260913-0108`.

[REPORT.md](REPORT.md) contains the complete thirteen-section scientific report;
[METHOD.md](METHOD.md) specifies the chosen algorithm;
[LITERATURE.md](LITERATURE.md) and [CANDIDATES.md](CANDIDATES.md) preserve source
provenance, hypotheses and every retired setting. This is a combination of known
methods, with no external state-of-the-art or Krotov-specific superiority claim.

## Preserve the frozen study

Do not edit the benchmark Markdown, target fixture, selected numerical sources,
per-run protocols, sealed outputs, manuscript or unrelated user work. The
manifest records authoritative startup hashes. No unfavorable seed may be
omitted or rerun to replace its score. Changed numerical code requires a newly
versioned output; never silently resume an old protocol with new source bytes.

The selected recipe remains `original`: lazy target-overlap initialization,
generic initial compilation, mean-two noisy training, exact deletion and
symmetric repair, then generic final compilation. All serious controls receive
the corresponding initializer, objective, estimator, structural and compilation
opportunities. The known-input compiler recipe was rejected during development
because its +0.377773-point gain over adaptive online had only two target wins,
despite a higher raw candidate fidelity. No confirmation outcome chose between
those recipes.

The complete parent covers `tfim_ferro`, `tfim_para`, `haar_random_2`,
`haar_random_3` and `random_mps_bond3`, each with replicas 11/29, seven roles,
twelve exact conditions and three separately reported endpoint strata. Primary
CPU uses the four-structural minimum with all seven roles evaluated at the same
chronological cutoff; the all-six-trained minimum is sensitivity only. Final
compilation is available only after its recorded cost. Exact noisy expectations
have no trajectory evaluation sampling error, but five target identities and
two replicas do not establish broad target-population generalization.

## Independent verification of all saved endpoints

The separately implemented Qiskit density evaluator passes all 2,520 saved
fidelity comparisons: 210 role endpoints, 141 unique target/circuit/parameter
combinations and 1,692 independent profile calls. Maximum absolute error is
9.992007221626409e-15 against the frozen 1e-10 tolerance, with zero missing rows,
failed calls or cap overruns. It constructs gates and Pauli-Liouvillian channel
matrix exponentials independently, including separate calibration and Ballarin
lowering. This verifies the saved circuits under the fixed benchmark semantics;
it does not change the candidate's failed promotion gate.

[INDEPENDENT_VERIFICATION_FINDINGS.md](INDEPENDENT_VERIFICATION_FINDINGS.md)
links the complete raw comparison and
[independent result audit](../../output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_VERIFICATION_RESULT_AUDIT.json).
All physical/Ballarin counts match, every published comparison is reconstructed,
and all 12,080 original confirmation files remain unchanged. The later audit
uses stored scalar diagnostics; final density matrices are not retained, so it
does not independently repeat eigenvalue calculations.

The verification's inclusive cost is 47.715553 CPU / 68.135319917 wall seconds,
including preparation. It is already included once in the authenticated
28-panel consolidated ledger; do not add it to that subtotal again. Its read-only reduction costs
4.255584 CPU seconds of engineering. The public verifier CLI cache resume
preserves all 7,197 new verification files and makes no new oracle calls;
its separate engineering cost is 7.037975 CPU / 16.456421 wall seconds.
[Verifier resume audit](../../output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_VALIDATION_CLI_RESUME_AUDIT.json).

## Reproduce and authenticate the selected result

Run from the repository root with all numerical libraries limited to one thread:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
uv run python -m experiments.overnight_state_preparation_20260913_0108.best_candidate --run --workers 1 --output output/overnight-state-preparation-20260913-0108/confirmation-best-candidate
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation --output output/overnight-state-preparation-20260913-0108/independent-confirmation-validation
```

The best-candidate command has already passed a complete cache resume: all ten
cases report complete, all 12,080 files remain byte-identical, and every
attempt/score receipt is unchanged. No scientific rerun occurred. See
[CONFIRMATION_CLI_RESUME_AUDIT.json](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_CLI_RESUME_AUDIT.json).
This proves recovery of saved results; it is distinct from fresh-run bitwise
endpoint equivalence. Measured process CPU can move a cutoff between deterministic
checkpoints on another execution or machine.

A fresh independent numerical reproduction needs an unused output directory,
preserved configuration and explicit resource allocation. The CLI sequence is
shown below for documentation, not as a new run requested by this handoff:

```sh
uv run python -m experiments.overnight_state_preparation_20260913_0108.best_candidate --freeze-design --recipe original --development-root output/overnight-state-preparation-20260913-0108/common-control-audit --lazy-root output/overnight-state-preparation-20260913-0108/lazy-initializer-reproduction --output output/selected-original-independent-reproduction
uv run python -m experiments.overnight_state_preparation_20260913_0108.best_candidate --prepare-only --output output/selected-original-independent-reproduction
uv run python -m experiments.overnight_state_preparation_20260913_0108.best_candidate --run --workers 2 --output output/selected-original-independent-reproduction
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation --prepare-only --parent output/selected-original-independent-reproduction --output output/selected-original-independent-validation
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation --output output/selected-original-independent-validation
```

The verifier preparation follows completion of the fresh parent. Both illustrative
destinations must be unused. Inspect each driver's
`--help`; historical drivers have different CLI schemas. Source and environment
validation may reject a changed installation. Do not bypass a rejection by
rewriting stored hashes. No more than two single-thread workers are accepted by
the complete parent. A fresh repetition is reproduction or fixed-benchmark
development, not untouched confirmation.

Configuration SHA256:
`7ca2de674a9e13e863dfaa0c1b4312dee90a1d7cb437fa1efc9a3bad302d9068`.
DESIGN canonical content checksum:
`203ee474a2500b752c0f46063a32838e51aa73dd2bd1dad48f3797916e9ca976`.
DESIGN file-byte SHA256:
`b606b4fd1b0d6024f567de06abb216911e01e1d605d9e241bcf13888f4b2a024`.
Parent canonical checksum:
`5a8a4980116a955ab5146c4e146c423d7f84ff92ce66a3c23167cfe394e0cb4e`.
Confirmation SUMMARY canonical checksum:
`426c02e029dc7db5bf0664520f46db556364f45c2879f5d5a516a6ec6a5820c9`.

## Final evidence and cost coverage

The central scoreboard authenticates **20 source panels, 10,632/10,632 cells,
zero missing**. Budget strata remain separate; scores, profile-specific counts,
source receipts and missing-value semantics are preserved. Central confirmation
reconciliation checks all 2,520 cells exactly for fidelity, counts and evaluation
hashes; the 21 aggregate points differ only by at most 1.11e-16 from alternate
floating-point reduction order.

- [Central SCOREBOARD.csv](../../output/overnight-state-preparation-20260913-0108/SCOREBOARD.csv)
  and [SCOREBOARD.md](../../output/overnight-state-preparation-20260913-0108/SCOREBOARD.md)
- [Confirmation SUMMARY](../../output/overnight-state-preparation-20260913-0108/confirmation-best-candidate/SUMMARY.json)
  and [strict authentication](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_AUTHENTICATION.json)
- [Complete case/history diagnosis](CONFIRMATION_DIAGNOSTICS.md)
- [Final resource/Pareto projections](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2_scatter.png)
  and [paired heatmaps](../../output/overnight-state-preparation-20260913-0108/figures/confirmation_tradeoffs_v2_heatmaps.png)
- [Development original/compiled Pareto comparison](../../output/overnight-state-preparation-20260913-0108/figures/development_pareto_primary_v3.png)

The complete confirmation physically costs 2,818.474129 CPU seconds, counting
preparation once plus ten inclusive case invocations. The final
[28-panel ledger](../../output/overnight-state-preparation-20260913-0108/resource-accounting-28panels/LEDGER.json)
retains **12,318 disjoint events and 13,765.937587 measured CPU seconds
(3.823872 hours)**, including confirmation, MPS memory and independent verification.
All 10,624 earlier 27-panel events are preserved exactly; 1,694 events add the
47.715553-second independent-verification cost. Its canonical checksum is
`925da91a2d27f29ea17d0246edce2025349bf5fdacaef91ff8fb9f5102e2106d`.
Complete overnight/development CPU remains unknown; engineering and some old
coverage are unmeasured. Do not add nested stage timers, old subtotals or
confirmation/verification totals again, or equate differently covered counters.

The new resource audit costs 87.241795 CPU / 149.320147 wall seconds, and the
central collection audit costs 317.282637 / 360.936062 seconds. These are
engineering work outside scientific clocks. All 12,318 event CSV rows match
their JSON measurements; [RESOURCE_28PANELS_AUDIT.json](../../output/overnight-state-preparation-20260913-0108/validation/RESOURCE_28PANELS_AUDIT.json)
preserves the independent check. Historical scoreboards and ledgers remain
available, including `scoreboard-snapshots/18-panels`.

## Exact source recovery

[The selected-source archive](../../output/overnight-state-preparation-20260913-0108/reproducibility/selected-original-7ca2de674a9e.tar.gz)
is 1,175,328 bytes with SHA256
`5cb516c704242262a8976b072a88405e2a7cd8343da9a392df3818808e00e4ba`.
[Readback verification](../../output/overnight-state-preparation-20260913-0108/reproducibility/selected-original-7ca2de674a9e.verification.json)
covers every one of its 153 members. It preserves all 114 directly frozen
source files, the immutable benchmark inputs, support files, 19 compact
receipts, dirty-Git state and 130 installed distribution names/versions without
credential URLs. An extracted source-only import resolves 101 local modules,
rebuilds the exact chosen configuration and passes `runner.custody()` without
loading targets or running scientific computation.

The original direct source map omitted three transitive imports. Trajectory
comparison and the benchmark generator match startup hashes; the lazy wrapper
is bound by a development protocol itself frozen in DESIGN evidence. All three
are included as supplementary recovery sources with these proofs. No drift was
found, and the omission was not hidden through a retrospective DESIGN edit.

[Final code/benchmark custody](../../output/overnight-state-preparation-20260913-0108/validation/FINAL_CODE_AND_BENCHMARK_CUSTODY.json)
verifies 287 startup files after removing only the exact owned additive
CHANGELOG/UPGRADING blocks, 114 selected sources, 97 selected evidence files,
104 final source/test files and all 153 archive members. No scientific source
or immutable benchmark changed during final verification.

This is source recovery, **not a fully relocatable raw-data deposit**. Full
training/evidence graphs remain separate and contain absolute paths. The archive
README records those dependencies, environment limits and exact verification.
Do not invoke a Hatch build that regenerates frozen source version files.

## Verification and outstanding limits

Targeted checks use the repository toolchain:

```sh
uv run pytest -n 0 --no-cov tests/optimization/test_overnight_best_candidate_20260913.py tests/optimization/test_overnight_reporting_20260913.py
uv run ty check experiments/overnight_state_preparation_20260913_0108/reporting.py tests/optimization/test_overnight_reporting_20260913.py
uvx --from ruff==0.15.15 ruff check experiments/overnight_state_preparation_20260913_0108/reporting.py tests/optimization/test_overnight_reporting_20260913.py
```

The authoritative corrected full owned cohort passes **1,409 tests in 116.09
seconds**, with zero failures, errors or skips. All 53 source modules and 51 test
files retain their 104 frozen hashes. Corrected combined coverage is 80.835381%
(81% displayed), with 10,140/12,298 statements (82.452431%) and 3,020/3,982 branches
(75.841286%). The 169 actual exclusions are independently verified as 52
TYPE_CHECKING headers and 117 import lines, with no excluded numerical function
lines. The 2,158 uncovered statements and 962 missing branches remain explicit. This is not complete repository or
behavioral coverage.

The temporary coverage configuration anchors ellipses to standalone statements,
without modifying repository settings or scientific sources. All fifteen
previously omitted function bodies are now measured. The older 1,249-test and
separate 152-test reports remain historical and must not be added to 1,409.
[TEST_COVERAGE_REVIEW.md](TEST_COVERAGE_REVIEW.md),
[corrected coverage JSON](../../output/overnight-state-preparation-20260913-0108/validation/coverage-final-corrected.json)
and [the final cohort audit](../../output/overnight-state-preparation-20260913-0108/validation/FINAL_CORRECTED_COHORT_REVIEW.json)
record the current module map, restored bodies, remaining gaps and source proofs.

Full `uvx nox -s lint` batches use complete owned snapshots so autoformatters
cannot touch frozen or unrelated user files. Batch 32 records zero new-code
violations, all typing checks passing and reviewed snapshot equality. Unrelated
legacy Ruff/license/toolchain/manuscript findings and a frozen spelling false
positive remain recorded; full repository nox still exits nonzero. The exact
selected-source archive is not a complete repository lint snapshot.

The initial MPS parameter-agreement gate remains failed. The separate physical
follow-up passes, and the later eight-endpoint retained-memory audit has near
roundoff norm/gradient error, but MPS gradients are about 13.25 times slower
than dense gradients at q6. The tapes occupy 122,816–365,952 bytes plus 2,688
target bytes; these are older endpoints and a no-factor path, not the complete
selected noisy batch. Worker RSS is lifetime peak, not incremental optimizer
memory. No full-method memory superiority is established.

The canonical fixture offers q6 and q12 only. The q8 design is ineligible for
missing inputs; do not generate replacement targets. Failed confirmation and
modest resource tradeoffs do not authorize q12 scaling.

The next useful proposal is a single development-only repair/polish allocation
comparison on the original four targets, with all controls given the same
opportunity and all search costs charged; see REPORT section 12. It has not
been executed. All nine q6 identities are now used, and new seeds cannot restore
untouched status. A changed recipe needs a separately authorized, independently
fixed untouched panel for any new confirmation claim.

## Source-only checkout

Historical run receipts (`MANIFEST.json`, `STARTUP.diff`, and
`STARTUP_FILES.json`) and the output archive stay local. Custody-checked
replays and archive-driven reports need those original local inputs.
The experiment source and JSON protocol configurations are published.
