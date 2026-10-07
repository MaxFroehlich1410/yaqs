# Shared-adjoint scoring and reinvested refinement: findings

Completed 2026-09-12 under the frozen
[verification and reinvestment protocol](state_preparation_adjoint_restructuring.md).
All eight target/construction cases and 32 optimizer/structure arms completed.
Historical exhaustive-search artifacts were read-only and remain byte-identical.

## Main result

**Cheaper, equivalent deletion scoring plus continued noisy refinement makes
the structural procedure competitive with the fixed-circuit controls on this
development panel.** Both structural optimizers now exceed the predeclared
+0.5 percentage-point (pp) mean improvement over compute-matched fixed Adam,
with positive target-level gains in three of the four families.

Adaptive online Krotov has the higher mean, but only by **+0.115733 pp** over
structural Adam, winning **2/8** paired cases. Both wins are on Haar; Adam wins
all six structured-target cases. This supports the shared structural procedure,
not a general Krotov superiority claim. Haar recovers substantially but its
family mean still trails the fixed-circuit controls.

## What changed, and what did not

The shared density adjoint replaces the repeated whole-circuit evaluator in
the new experiment's deletion-ranking step. The old exhaustive scorer remains
available as the independent reference. For each gate k, one forward tape stores
rho_before(k), while the backward observable B_after(k) contains the unchanged
suffix. The exact finite-deletion fidelity is

`F_delete(k) = Tr(B_after(k) * rho_before(k))`.

Contract before propagating the current channel/unitary adjoint. This deletes
both the gate and its noise window; it is not an approximation based on a small
parameter change. All other gate noise windows remain unchanged.

Everything else was retained from the
[previous restructuring experiment](state_preparation_noisy_restructuring_findings.md):

- The same four six-qubit development targets: Gaussian, critical TFIM, Haar
  random 1 and bond-2 random MPS, each with construction starts 11 and 29.
  No new target, construction or hyperparameter search was introduced.
- The same 108-gate `H25_lr0p7:cold_final` starting circuits, calibrated local
  Pauli-Lindblad noise proxy (abstract-gate errors 0.00064 / 0.0051), weighted
  R32 trajectory sampler, fresh ensembles and root seed 2026091207.
- Adam at 0.02; adaptive online Krotov with 25 unscaled online updates at 0.7,
  then RMS-scaled online updates at 0.02. This is the existing adaptive-online
  schedule, **not an online-to-batch tail**. Moment constants are unchanged.
- 50 warmup updates, five deletion rounds, two shortlisted deletions, ten
  repair updates per candidate, protected original 18 product-layer gates,
  unchanged repaired/unrefined controls and the same ranking/acceptance rules.
- Preserved parameters, moments and update counters on retained gates; no
  optimizer restart after pruning. Additional refinement starts after the last
  consumed structural noise epoch, rather than replaying discarded work.

The sole new use of saved CPU is continued parameter refinement of the selected
structure. Both fixed-circuit controls were rerun too. Within a case all four
arms receive **the same frozen CPU budget**, the larger of the two historical
structural costs: 52.899165-58.241136 seconds. Both methods use best scored
checkpoints available before the cutoff; crossing blocks are excluded without
interpolation. All arms covered their budgets without hitting the 4000-update cap.

## Equivalence and speed verification

Verification finished for the entire panel before any refinement job started.

- **7040/7040** archived eligible deletion scores at all **80** round-input
  states agree. Maximum absolute discrepancy: **4.662937e-15**, against a
  predeclared tolerance of 1e-10.
- Every full ordered ranking, including the 1e-12 rounding/tie rule, is identical.
- Sixteen first-round states were freshly scored by both implementations, with
  their execution order alternated by optimizer. Median paired CPU speedup:
  **317.02x**, range **240.70-364.75x**. Summed paired CPU was 153.236302 s for
  exhaustive evaluation and 0.489466 s for shared scoring.
- Freshly rerunning the search reproduced all 80 accepted gate choices,
  every repaired candidate's complete optimizer state, and all 16 final search
  states exactly under canonical JSON hashes. No archived optimizer state was
  substituted for the freshly timed search.

The 317x figure compares these two implementations: the new local-array kernel
also avoids per-candidate Qiskit propagation and reconstruction overhead. It
is **not** a claim that adjoint reuse alone, or Krotov updates, are 317x faster.
At fixed qubit count, the gate-propagation work changes from quadratic to linear
in circuit length; the shared density tape still needs O(L \* 4^n) entries.
There is no demonstrated trajectory/MPS memory-scaling advantage here.

Across all 16 five-round searches, including identical warmups and candidate
repairs, old recorded search CPU fell from **859.041725 s to 141.599013 s**,
about **6.07x** overall. This is a cross-run comparison, not the fresh paired
kernel benchmark above. Deletion scoring fell from 731.672356 s to 2.455954 s:
from **85.17% to 1.73%** of search CPU. Per-method mean search CPU is now
7.577910 s for Adam and 10.121967 s for adaptive online Krotov.

## Matched-budget fidelities

Fidelities are exact noisy target fidelities in percent. Each family has two
starts and contributes equally to the overall mean. Wins are paired against
fixed Adam, not independent target samples.

| Method                                            | Mean fidelity (%) | Gain over fixed Adam (pp) | Wins / 8 | Gates |
| ------------------------------------------------- | ----------------: | ------------------------: | -------: | ----: |
| Fixed-circuit Adam                                |         79.800750 |                         0 |       -- |   108 |
| Fixed-circuit adaptive online Krotov              |         79.828662 |                 +0.027912 |        3 |   108 |
| Shared search + Adam refinement                   |         80.839281 |                 +1.038531 |        6 |   103 |
| Shared search + adaptive online Krotov refinement |         80.955015 |                 +1.154264 |        7 |   103 |

Against its own fixed-circuit optimizer, the Krotov structural procedure gains
+1.126353 pp. Both structural methods pass the +0.5 pp / three-family screen;
neither result uses a per-target mixture of optimizers.

| Family            | Fixed Adam (%) | Fixed Krotov (%) | Pruned + Adam (%) | Pruned + Krotov (%) |
| ----------------- | -------------: | ---------------: | ----------------: | ------------------: |
| Gaussian          |      82.799938 |        82.636192 |         84.577541 |           84.406072 |
| Critical TFIM     |      82.735584 |        82.654714 |         84.578994 |           84.490446 |
| Haar random       |      70.806833 |        71.186281 |         69.368531 |           70.258620 |
| Bond-2 random MPS |      82.860646 |        82.837461 |         84.832059 |           84.664921 |

The structured-family gains over fixed Adam are +1.78 to +1.97 pp for structural
Adam and +1.61 to +1.80 pp for structural Krotov. All six structured cases improve
with both methods. Krotov's mean lead over structural Adam comes from +0.890088 pp
on Haar, offset by losses of 0.09-0.17 pp in each structured family.

### Did extra refinement resolve the previous problem?

It helped, without changing which gates were removed. The unchanged five-round
search means were 79.819091% for Adam and 80.149843% for Krotov. Additional
refinement gains **+1.020191 pp** and **+0.805171 pp**, respectively.

The largest recovery is Haar: **+2.937876 pp** for Adam and **+2.254595 pp**
for Krotov relative to their old selected structures before continuation.
The Haar deficit to fixed Adam is now -1.438301 pp for structural Adam and
-0.548213 pp for structural Krotov. Krotov's Haar start 29 now wins by +0.903111 pp;
start 11 still loses by -1.999538 pp.

Thus inadequate refinement was a real contributor to the previous deficit,
but longer refinement does not completely solve Haar. The selected Krotov
Haar circuits have ideal fidelities 88.780322% and 84.447882%, versus 93.756201%
and 85.938207% for the fixed-Adam selections. Optimization basin and reduced
representational capacity remain possible explanations; this run does not
separate them or prove either a noise ceiling or eventual convergence.

Every selected structural circuit retains exactly the old 103 gates. Krotov
retains 25 two-qubit gates in every case, versus 30 in the fixed controls.
Adam retains 25 on structured targets and 26/28 on the Haar starts. The 16.7%
two-qubit reduction refers to these abstract gates, not compiled native gates.

At the CPU cutoffs, structural Adam has completed 1380-1480 accepted-path updates
and structural Krotov 990-1060, versus only 100 before reinvestment. Exported
best checkpoints occur at updates 1070-1440 and 720-1040, respectively; the
last iterate is not silently substituted for the best available circuit.
Fixed Adam completes 1410-1550 updates and fixed Krotov 1050-1120.
Discarded structural repair branches are additional charged work, not included
in these accepted-path update counts.

## Cost accounting

The run spent **738.047612 CPU seconds** on post-pruning continuation across
the 16 structural arms. This uses the saved search budget plus the predeclared
common-case budget adjustment; it is not an additional unreported optimization
allowance. Both fixed-circuit continuations cost 848.548944 s in total.

Physical operation CPU totals **1728.195569 s**, counting each reused warmup
once, plus separately measured verification cost of 155.526304 s. Charged arm
costs include setup, sampling, exact scoring and rejected repairs. Construction,
imports, launch, artifact I/O and post-run audit are excluded as declared.
Crossing blocks are recorded and paid physically, but unavailable at the primary
cutoff. Each selected curve point is within 0.436 s of its budget.

Actual computed updates total **40870**: 800 warmup, 2400 candidate repair,
18070 post-pruning continuation and 19600 fixed continuation. Recorded
trajectory-direction passes total **3174400**; these are not gate operations
or hardware shots. Eight single-threaded workers were used. The manifest-to-result
span was **355.55 s (5 min 56 s)**, including score verification and the pause
to inspect it before launching refinement; it is not isolated training time.

## Interpretation and next decision

1. **Keep the shared scorer.** It reproduces the old finite-deletion decisions
   and removes the dominant implementation bottleneck.
2. **Treat search plus substantial refinement as the candidate procedure.**
   On this panel it gives a meaningful fidelity/compression gain at fixed CPU,
   unlike the earlier search-only endpoint comparison.
3. **Keep Adam as an equally serious comparator.** Krotov has a slightly higher
   mean and helps the difficult family, but loses six of eight direct pairs.
   A Krotov-specific robustness or cost advantage has not been established.
4. **Freeze both current configurations before testing new states.** The
   development screen is now passed, so a predeclared fresh multi-state panel
   per family is a better next confirmation than tuning these same four targets
   further. No new-target run was launched by this experiment.

This is still exact-density-assisted structural selection with trajectory-based
parameter optimization, not a fully trajectory-based structural algorithm.
One repeatedly used state per family and two construction starts cannot establish
target-independent generalization. A noisy-versus-noiseless structural-ranking
control, scalable structural estimators, published synthesis baselines, native
compilation and realistic scheduling/calibration uncertainty remain separate
requirements for the broader paper claim. This result alone is not state of the art.

## Engineering verification and artifacts

- **17 new tests; all 639 optimization tests passed** in 139.78 s. Coverage includes
  every supported gate generator, fixed/shared parameters, disabled channels,
  reversed/nonadjacent sites, zero and high noise, complete gate/noise deletion,
  score/ranking mismatches, exact repair replay, moments/epochs, verification
  barriers, parallel execution and authenticated resume.
- The post-run audit independently recomputed **472 stored scores**, covering
  **456 distinct density evaluations**, with the local exact-density adjoint
  rather than the Qiskit selection oracle. Maximum error: **2.886580e-15**.
  This covers source and warmup scores, shortlisted raw deletions, every repaired
  candidate, search endpoints and all 32 selected matched-budget endpoints.
- All source/runtime/protocol fingerprints, case signatures, complete optimizer
  replay, continuation epochs, unchanged structures and cutoff coverage passed.
- Completed serial resume preserved all **19** new scientific JSON files byte
  for byte. All **10** historical source artifacts remained byte-identical.
- Changed-file checks pass. Required repository-wide `uvx nox -s lint` remains
  blocked by pre-existing manuscript/plotting/MPS and old optimizer-test issues,
  a broken Homebrew Node `simdjson.28.dylib` dependency, and missing matplotlib
  in the isolated lint environment. Unrelated automatic edits were restored.

Driver: [state_preparation_adjoint_restructuring.py](state_preparation_adjoint_restructuring.py).
Kernel: [exact_density.py](../benchmarks/state_preparation/exact_density.py).
Artifacts: `output/statevector-adjoint-restructuring-20260912/`, containing
the manifest, verification aggregate/eight records and result/eight optimization
records. Canonical SHA-256 values below exclude the checksum envelope.

| Artifact     | Canonical SHA-256                                                  |
| ------------ | ------------------------------------------------------------------ |
| Manifest     | `baa8d1f5c27cf9bf887cff10c2b5b92bbc0c40596a316ca5e00115d8d777c150` |
| Verification | `ae18cde201014350698279e5b9c4e1c58f266b791021fbb193001950ea12be7f` |
| Result       | `44713751ab99c23c125c64ded70d6988639647adbe1369d742df0bcb0e232908` |
