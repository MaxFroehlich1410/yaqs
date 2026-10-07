# Independent verification of sealed confirmation endpoints

This is a numerical verification of the already completed, failed confirmation.
It is not a candidate method, training run, endpoint-selection opportunity or a
new confirmation panel. The old failed promotion gate remains unchanged.

Before any endpoint score, freeze all ten existing confirmation cases (five target
identities, replicas 11/29), all seven roles, all three saved panels (primary,
full cap, all-trained sensitivity), and all twelve normative noise conditions:
210 role endpoints and 2520 expected fidelity rows. A missing/failed case, role or
condition aborts preparation. Authenticate each case through frozen
`best_candidate.read_case`; read target bytes only from its sealed input phase.
No benchmark target loader, constructor, optimizer, compiler or original scoring
function is called. No files beneath the original confirmation root are written.

The new `independent_qiskit_oracle.evaluate_profile` uses Qiskit gates and density
operators plus independently constructed Pauli-Liouvillian matrix exponentials,
calibration and Ballarin lowering. The numerical oracle imports no YAQS channel
or gate-matrix implementation. Its dedicated engineering tests check independent
calibration, placement, ordering, density physicality and literal angle semantics.
Parent API imports are used only for artifact custody, metadata and saved arrays.

Freeze source/dependency hashes, environment, complete input file identities,
ordered profile list, every circuit/parameter/target vector and each expected
fidelity in new `INPUTS.json` and `PROTOCOL.json`. Saved target bytes must reproduce
the parent's complex128 vector hash and unit norm. Precompute and retain the
independent standard-profile calibration metadata, after the complete panel is
validated; its CPU is charged to preparation. No tolerance is chosen from results.

Deduplicate only identical target-vector bytes, full physical circuit records,
parameter vectors and profile identities. Different roles/panels/replicas remain
separate comparison rows even when the same numerical evaluation is reusable.
All 2520 rows compare the independent fidelity to their own saved expected value.
The 600-second budget is unchanged by the deduplication rate.

Acceptance requires every expected row to be present and successful, absolute
fidelity difference at most `1e-10`, density trace error `abs(trace(rho)-1)` at most
`1e-10`, maximum Hermitian residual at most `1e-10`, and minimum Hermitian-part
eigenvalue at least `-1e-10`. Values are not clipped, states are not renormalized,
and global discrepancies are not silently converted to passes. A complete panel
with disagreement stays complete with `accepted=false`; numerical failures and
uncomputed rows are explicit and also fail the all-row gate.

Use exactly one explicitly single-thread worker. All four OpenBLAS, OpenMP, MKL
and Accelerate thread variables must equal one. **The cumulative CPU admission
cap is 600 seconds**, including all measured preparation attempts, input/source
validation, run attempts, failed/replayed work, oracle calls, endpoint handling,
checkpoint I/O inside those attempts and final row reduction. Check the remaining
cap before every new oracle call. An already admitted call may finish across the
cap; record the actual overrun and launch no further call. This is an admission
cap, not a preemptive signal timer. If rows remain, report `incomplete_budget`
with all expected rows still visible and `accepted=false`. Finishing all calls
within the admission rule retains the full completion/physicality status and any
measured overrun. No budget increase or second configuration is allowed on resume.

Persist preparation, run, unique-endpoint and per-profile attempts before their
work starts. Preparation plus inclusive run costs are the physical total;
endpoint/per-call/oracle clocks are nested diagnostic breakdowns and must not be
added again. Final receipt publication and later cache-only authentication are
outside the original measured invocation. A caught interruption retains its
measured duration; only missing calls may resume. A durable `running` marker with
unknown elapsed work blocks resume instead of becoming zero cost. Terminal
numerical failures are cached and never retried for a better result. A completed
cache binds all descendant files, raw oracle results and reconstructed comparison
rows; cache validation performs no new numerical scoring.

Synthetic q1-q3 parents and injected failures are allowed only in engineering
tests. They do not access held-out vectors. The scientific driver is prepared and
executed only after coordinator source freeze and full lint. The independent q3
108-gate engineering timing forecast V2 is bound by byte hash in the new protocol
and source inventory. Its receipt is
`validation/INDEPENDENT_ORACLE_ENGINEERING_TIMING_V2.json` (content checksum
`db0ea2a4428812f424ba171f0fd97e493766d8c93a7ac7fa9417e607f9cc0343`).
The projected 240.991 CPU seconds for all 2520 rows without deduplication, or
481.982 seconds with a twofold planning margin, is planning evidence, not a guarantee of q6
runtime or correctness. If the cap prevents completion, report that outcome; do
not shrink the predeclared panel.

Example coordinator commands after source freeze:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation \
  --prepare-only --parent output/overnight-state-preparation-20260913-0108/confirmation-best-candidate \
  --output output/overnight-state-preparation-20260913-0108/independent-confirmation-validation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.overnight_state_preparation_20260913_0108.independent_confirmation_validation \
  --output output/overnight-state-preparation-20260913-0108/independent-confirmation-validation
```
