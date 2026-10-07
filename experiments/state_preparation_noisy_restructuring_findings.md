# Noisy deletion-and-repair: results and interpretation

Completed 2026-09-12 under the frozen
[headroom protocol](state_preparation_noisy_restructuring.md).
All eight target/construction cases and **32 optimizer/structure arms** completed.
The manifest-to-result interval was **269.28 seconds (4 min 29 s)** using eight
single-threaded worker processes. Implementation, tests and the final audit are
outside this interval. No old experiment output was overwritten.

## Main finding

**Selecting deletions by noisy fidelity is useful on three structured target
families, but the complete procedure does not yet pass the overall competitive
screen.** Both structural optimizers beat compute-matched fixed Adam on the
Gaussian, critical TFIM and bond-2 random-MPS targets, in both construction
replicas. Both lose on Haar, in both replicas.

Adaptive online Krotov gives the higher mean final structural fidelity, but at
slightly higher CPU cost and with wins in only three of eight endpoint pairs
against structural Adam. At the predeclared common CPU cutoff, structural Adam
has the higher mean. These results do not establish a general Krotov advantage.

## Matched conditions

- Four old six-qubit development targets, one state per family, construction
  replicas 11 and 29. These are genuinely different construction starts, not
  two noise replicas of the same start. Each start has one paired training
  stream; initialization and trajectory-stream variability are not separated.
- Starts are the earlier `H25_lr0p7:cold_final` 108-gate checkpoints, not the
  later four starts from the adaptive-rate experiment. Do not interpret changes
  from that experiment's mean as a matched continuation.
- All circuits initially have 78 one-qubit / 30 two-qubit gates. Parameter
  gradients and online updates use weighted R32 statevector trajectories.
- Adam rate 0.02; adaptive online starts with 25 unscaled updates at 0.7, followed
  by online RMS-scaled updates at 0.02. Moments are preserved after deletion.
- Same homogeneous calibrated abstract-gate local Pauli-Lindblad noise proxy:
  complete-window average errors 0.00064 / 0.0051. This is not a native hardware
  noise model with compilation, scheduling and idle decoherence.
- Each arm begins with 50 noisy warmup updates. Five structural rounds each
  exhaustively rank eligible single deletions, repair the top two plus an
  unchanged candidate for ten updates, and select by exact noisy fidelity.
  The original 18 product-layer gates are protected. No parameter/rate search.
- Exact density-matrix fidelity is used for structural decisions, not just
  reporting. This is therefore **exact-score-assisted trajectory optimization**,
  not an entirely trajectory-based circuit-synthesis algorithm.
- Fixed controls receive matching CPU, including the structural arm's exhaustive
  scoring and unsuccessful repair costs. They may export their best exactly
  scored checkpoint available before the cutoff. No future checkpoint or
  interpolation is used. Every fixed control covered its required budget;
  none hit the 4000-update safety cap.

## Primary endpoint comparison

Each fixed-control value below uses the corresponding structural arm's CPU
cutoff, so the two fixed-Adam means differ slightly. Fidelities are percentages;
gains are percentage points (pp). Average the two starts within each target,
then weight all four targets equally.

| Structural optimizer   | Structural fidelity (%) | Fixed Adam at matched CPU (%) | Gain over fixed Adam (pp) | Wins / 8 | Mean CPU (s) |
| ---------------------- | ----------------------: | ----------------------------: | ------------------------: | -------: | -----------: |
| Adam                   |               79.819091 |                     79.797865 |                 +0.021225 |        6 |    52.555114 |
| Adaptive online Krotov |               80.149843 |                     79.803891 |                 +0.345952 |        6 |    54.825102 |

The predeclared screen required at least **+0.5 pp** mean gain over fixed Adam
and positive gains in at least three families. Both methods meet the family
criterion, but **neither meets the magnitude criterion**. No new-target
confirmation or broader parameter sweep was launched.

Against their own fixed-structure optimizer, structural Adam gains +0.021225 pp;
structural adaptive online gains +0.307504 pp. At the adaptive structural cutoff,
fixed adaptive online averages 79.842339%, versus 79.803891% for fixed Adam.
This small longer-continuation mean advantage is not a contradiction of the
previous 50-update experiment: starts, budgets and checkpoint selection differ.

### Target-level results

Both gain columns compare against fixed Adam at the respective structural
arm's own CPU cutoff.

| Family            | Structural Adam (%) | Structural Krotov (%) | Adam gain (pp) | Krotov gain (pp) |
| ----------------- | ------------------: | --------------------: | -------------: | ---------------: |
| Gaussian          |           84.180888 |             84.057441 |      +1.385187 |        +1.253951 |
| Critical TFIM     |           84.384722 |             84.241038 |      +1.653877 |        +1.500736 |
| Haar random       |           66.430656 |             68.004025 |      -4.373613 |        -2.807102 |
| Bond-2 random MPS |           84.280097 |             84.296869 |      +1.419452 |        +1.436223 |

The three structured-family means improve by +1.486172 pp with Adam and
+1.396970 pp with adaptive online Krotov. This descriptive subgroup does not
replace the predeclared four-family primary mean, and must not be used to
silently drop the difficult Haar target.

## What the deletions actually did

Every structural run accepted one deletion in each of its five rounds: **80
accepted deletions**, with all final circuits at **103 gates/parameters**.
No run reached a no-further-benefit stopping point; five rounds was the imposed
limit, not evidence of an optimal final circuit size.

- Adaptive online removed five two-qubit gates in every case: 30 to 25,
  a 16.7% reduction in abstract two-qubit count.
- Adam also reached 25 two-qubit gates on all six structured-target cases.
  On the two Haar starts it retained 26 and 28, removing some one-qubit gates
  instead. Its mean final two-qubit count is 25.5.
- Every accepted deletion improved noisy fidelity **even before repair**.
  These immediate gains ranged from +0.012670 to +0.427359 pp for Adam's choices
  and +0.066068 to +0.431381 pp for Krotov's choices.
- After repair, each accepted deleted candidate also beat the equally repaired
  unchanged candidate from the same parent. The average local margin was
  +0.329716 pp for Adam and +0.377879 pp for adaptive online. These selected
  local comparisons are not independent samples or population evidence.

Thus the Haar failure is **not** that the accepted local deletions lowered the
current noisy fidelity. They raised it, but the fixed circuit used the search
budget for much longer parameter optimization and finished further ahead.

The surviving structural paths contain only **100 updates**: 50 warmup plus
50 accepted repair updates. Each structural run actually paid for **200 updates**
including the discarded repair branches. At the structural CPU cutoffs, fixed
Adam had completed **1390-1590 updates**. Selection among those checkpoints was
allowed in both the protocol and implementation.

Independent ideal-fidelity checks reinforce this distinction: the two Haar
structural-Krotov circuits have ideal fidelities 87.553485% and 80.595639%,
versus 93.743219% and 85.938207% for their compute-matched fixed-Adam selections.
This is not a demonstrated noise ceiling. The experiment does not separate
unfinished repair/local optimization from any loss of representational capacity
after pruning; further optimization of the selected structures would be needed.

## Does Krotov help inside the structural procedure?

At five completed rounds, adaptive online exceeds structural Adam by
**+0.330752 pp on average**, at approximately **4.3% more CPU**, winning **3/8**
endpoint pairs. Both Haar starts favor Krotov (+2.525293 and +0.621445 pp);
one MPS start does too. The other five pairs favor Adam.

At each case's smaller structural CPU budget, only four Krotov rounds have
completed while all five Adam rounds have completed:

| Structural method      | Mean available fidelity at common CPU (%) | Wins / 8 |
| ---------------------- | ----------------------------------------: | -------: |
| Adam                   |                                 79.819091 |        6 |
| Adaptive online Krotov |                                 79.659822 |        2 |

The common-CPU mean difference is -0.159269 pp for Krotov. Round-sized
checkpoints make this a coarse comparison: a modest total timing difference
excludes the entire fifth Krotov round. We do not interpolate an unfinished
decision, and we do not treat this cutoff effect as a definitive intrinsic
optimizer-speed ranking. The equal-round endpoint and CPU-limited results are
both reported. There is a useful Haar signal, not a consistent Krotov win.

## Main computational bottleneck

**85.17% of structural CPU was spent on exhaustive exact deletion scoring.**
Across both optimizers and all cases:

| Charged operation                                         | CPU seconds |
| --------------------------------------------------------- | ----------: |
| Initial exact scores                                      |    1.672608 |
| Warmup                                                    |   27.162487 |
| Exhaustive deletion scores                                |  731.672356 |
| Unchanged-circuit repair                                  |   32.482757 |
| Deleted-candidate repairs, including discarded candidates |   66.051517 |

The driver evaluated **7040 raw deletions**. Each was a separate complete
density propagation. This is an implementation cost of this deliberately simple
headroom diagnostic, not a proven lower bound on structural search cost.

Measured physical operation CPU, including fixed controls and counting reused
warmups only once, totaled **1710.55 seconds**. Summed per-worker operation wall
time was 1969.03 seconds; concurrent wall times must not be added and presented
as the 269.28-second experiment duration. Historical construction, file I/O,
worker import/startup and reporting/audit are excluded from the charged CPU.

## Recommended next step

Keep the same development panel and test **cheaper structural scoring followed
by adequate noisy refinement of the chosen circuit**, not a larger rate sweep
or new targets yet.

For this fixed local-channel model, all exact single-deletion scores can in
principle reuse a forward density tape and backward observables. If rho_before
is the state before gate k and B_after is the target observable propagated
back through the unchanged suffix after gate k, deleting the gate and its
channel gives `F_delete(k) = Tr(B_after * rho_before)`. This follows from
linearity and assumes the deletion does not alter any other noise window.
It suggests replacing repeated whole-circuit simulation with shared adjoint
environments. Verify those scores against the retained exhaustive results
before using the faster implementation for another experiment.

The saved computation could then be spent refining the selected circuit,
testing whether Haar recovers while the structured-family compression gains
remain. A scalable trajectory estimator for the deletion scores would be a
subsequent step, not something this exact-score experiment has implemented.

The remaining comparison must still include Adam within the same procedure.
These findings support a noise-aware synthesis direction, but not yet a
publishable superiority claim for trajectory-based Krotov. They also do not
isolate noisy scoring from noiseless structural search: there was no matched
noiseless-ranking arm in this experiment. Native compilation, realistic idle
noise, held-out targets and comparisons with published synthesis methods remain
necessary for the larger research claim.

## Verification and artifacts

- **22 new targeted tests passed; all 622 optimization tests passed** in 123.70 s.
  Tests cover physical deletion, moment/counter compaction, sampler recalculation,
  both optimizer kernels, deterministic ties, rejected-work costs, CPU cutoffs,
  safety caps, authenticated resume and serial/parallel structure decisions.
- The post-run audit authenticated all eight case records, source provenance,
  protocol/code fingerprints, identical optimizer streams during the shared
  circuit prefixes, all 80 decisions and the complete cost ledgers.
- It checked **464 stored scores**, covering **442 distinct density evaluations**,
  with the independent exact-density adjoint implementation rather than the
  Qiskit evaluator used for selection. Maximum discrepancy was **2.443e-15**.
  This includes all 160 raw shortlisted scores, all 240 repaired candidates,
  warmups and selected controls; it does not independently recompute all 7040
  raw ranking entries.
- Actual computed updates: 800 warmup, 2400 candidate repair and 20360 fixed
  continuation updates, totaling **23560**. Recorded trajectory-direction passes
  total **1835520**, not elementary gate operations or hardware shots.
- Completed serial resume preserved all **10 scientific artifact files byte
  for byte**. All fixed controls covered their required budgets without reaching
  the safety cap. Source and protocol fingerprints remained unchanged.
- Applicable changed-file hooks passed. Required repository-wide
  `uvx nox -s lint` was run; unrelated manuscript/plotting/MPS-test failures,
  a broken Homebrew Node `simdjson.28.dylib` dependency, and missing matplotlib
  in the isolated lint environment remain. Unrelated automatic edits were
  restored; no production code defaults changed.

Driver: [state_preparation_noisy_restructuring.py](state_preparation_noisy_restructuring.py).
Artifacts: `output/statevector-noisy-restructuring-20260912/`, containing
`manifest.json`, `result.json`, and one complete private JSON record per case.

| Artifact | Canonical SHA-256, excluding checksum envelope                     |
| -------- | ------------------------------------------------------------------ |
| Manifest | `e987eba5e0cdcd4095fab08ceb3d1eda5b4d838cd93a2aefb55ecb7fcc3b8fa7` |
| Result   | `169c353f8baf4c56b62d62b63b3941a9117c47990724c0ceec603eeb9cf9da19` |
