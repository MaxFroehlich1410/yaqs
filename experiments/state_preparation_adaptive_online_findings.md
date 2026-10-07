# Adaptive online Krotov: results and interpretation

Completed 2026-09-12 under the
[frozen 64-run protocol](state_preparation_adaptive_online.md).

## Main conclusion

**Adaptive scaling improves the current gate-sequential Krotov method, but does
not make it competitive with weighted Adam on this development panel.**

The best adaptive endpoint setting, rate 0.02 after a 25-update online prefix,
reaches 78.044484% mean exact noisy fidelity, versus 77.782097% for unscaled
all-online Krotov and 78.209294% for Adam at 0.02. Adaptive wins all eight
pairs against unscaled online Krotov, but loses all eight against Adam.
It costs approximately 45.1% more training CPU than Adam. The primary
matched-CPU comparison also favors Adam in every pair at every adaptive rate.

Thus the scaling hypothesis has useful support as an improvement to this
scheme. The hypothesis that the proposed adaptive schedule would beat Adam
at matched cost is not supported. These statements are restricted to the
fixed circuits, schedules, rates and four previously used development states.

## Execution and fixed parameters

There are **64 complete runs and 3,200 updates**: eight configurations on four
q6 targets, with noise replicas 11 and 29. The targets are the original
Gaussian, critical TFIM, Haar and random bond-2 MPS development states, each
with its frozen 108-gate construction-replica-11 circuit and parameters.
There was no new target generation, construction, pruning or gate removal.

All methods use weighted R=32 sampling: one deterministic no-selected-factor
path with probability 0.7658428756735005, plus 31 conditional paths with at
least one selected factor, each carrying probability weight `(1-p0)/31`.
All eight methods receive identical physical replay maps at each update for
the same target/noise replica. Maps refresh every update and are replayed
consistently for forward and adjoint propagation. Root seed is 2026091101.

The calibrated local Pauli-Lindblad model is unchanged: one-/two-qubit average
gate errors 0.00064/0.0051, integrated per-axis strengths
0.00032020497493061084/0.0010664702651014273. It is a state- and parameter-
independent random-unitary channel, not general normalized nonunitary jumps.
Training uses complex128 statevectors, with exact density scoring kept separate.
The common initial mean exact noisy fidelity is 77.278521%.

Both rate screens use 0.005, 0.01 and 0.02. Adaptive online means 25 ordinary
online updates at 0.7, then 25 gate-sequential RMS-scaled updates. The second
moment is initialized to zero at the switch, uses beta2=0.999 and epsilon=1e-8
outside the square root, with per-occurrence bias correction and no first
moment. Ensemble signals are summed before squaring/updating moments.
Every gate still sees updated forward prefixes and old costates.

Adam runs all 50 updates at its screened rate, with beta1=0.9, beta2=0.999,
epsilon=1e-8, zero initial moments and standard bias correction. The fixed
Krotov-Adam control uses 25 online updates at 0.7 followed by 25 Adam updates
at 0.02, resetting Adam moments/counter at the switch. It is a combination,
not a pure Krotov result. Unscaled online Krotov runs 50 updates at 0.7.

## All predeclared endpoint results

Each row equally averages the same eight target/noise-replica pairs.
Endpoints are update 50, never the best visited iterate. Training CPU includes
engine/sampler setup, sampling and updates. It excludes exact scoring,
physical-replay hashing, disk I/O and orchestration. All controls were rerun
under this revision, in fixed shuffled order with eight single-threaded workers.
These are refinement costs from common saved starts; the historical circuit
construction/pretraining cost is not part of this run or an end-to-end speed claim.

| Configuration                      | Mean exact noisy fidelity (%) | Mean training CPU (s) | Mean gain during updates 26-50 (pp) |
| ---------------------------------- | ----------------------------: | --------------------: | ----------------------------------: |
| Unscaled all-online Krotov, 0.7    |                     77.782097 |              1.890569 |                            0.147565 |
| Adaptive online, 0.005             |                     77.883536 |              1.885214 |                            0.249004 |
| Adaptive online, 0.01              |                     77.986944 |              1.888617 |                            0.352411 |
| Adaptive online, 0.02              |                     78.044484 |              1.886924 |                            0.409951 |
| Krotov-Adam combination, tail 0.02 |                     77.911776 |              1.595055 |                            0.277243 |
| Adam, 0.005                        |                     77.916677 |              1.292785 |                            0.188517 |
| Adam, 0.01                         |                     78.086628 |              1.284090 |                            0.237473 |
| Adam, 0.02                         |                     78.209294 |              1.300116 |                            0.289623 |

The adaptive 0.02 endpoint improves on unscaled Krotov by **0.262387 pp**,
with eight wins and individual improvements from 0.046103 to 0.926224 pp.
It closes approximately **61.4% of the previous endpoint gap** to Adam, at
essentially the same measured CPU as unscaled online Krotov. This is a real
development-panel improvement, not merely a lower noisy training loss.

However, adaptive 0.02 remains **0.164810 pp below Adam 0.02**, with eight
losses ranging from -0.350694 to -0.033063 pp. Adam 0.01 also has a higher
mean endpoint (78.086628%) and substantially lower CPU than adaptive 0.02.
The comparison is therefore not dependent only on Adam's highest rate.

The larger late gain for adaptive 0.02 than Adam 0.02 does not establish a
superior tail optimizer: the methods enter update 26 at different parameters
and different fidelities, with different remaining headroom and moment states.

## Primary result: matched training CPU

For each configuration versus Adam 0.02, use the smaller measured final CPU
total in each target/replica pair, and the last completed checkpoint within
that budget. Adam completes all 50 updates in every such comparison. Do not
interpolate or choose the maximum-fidelity checkpoint. The shared budget
averages 1.300116 s of training CPU per pair.

| Configuration           | Mean fidelity at common CPU (%) | Difference versus Adam (pp) |  Wins / 8 | Updates completed |
| ----------------------- | ------------------------------: | --------------------------: | --------: | ----------------: |
| Unscaled all-online     |                       77.692765 |                   -0.516529 |         0 |             32-34 |
| Adaptive online, 0.005  |                       77.736322 |                   -0.472972 |         0 |             32-35 |
| Adaptive online, 0.01   |                       77.791555 |                   -0.417739 |         0 |             32-35 |
| Adaptive online, 0.02   |                       77.786734 |                   -0.422560 |         0 |             31-36 |
| Krotov-Adam combination |                       77.673520 |                   -0.535774 |         0 |             33-40 |
| Adam, 0.02              |                       78.209294 |                           0 | Reference |                50 |

The best adaptive rate by this primary mean comparison is **0.01**, not the
endpoint winner 0.02. Their common-CPU difference is only about 0.0048 pp,
so do not overinterpret the rate ranking on noisy laptop timings. The main
conclusion is unchanged: **no adaptive rate beats Adam in any pair at common
CPU**, and all three lose on mean endpoint fidelity too.

At common CPU against unscaled online Krotov, all three adaptive rates still
win eight of eight pairs, improving by 0.097110, 0.196811 and 0.228341 pp for
rates 0.005, 0.01 and 0.02 respectively. This supports retaining the adaptive
implementation as an improved Krotov candidate, without claiming competitiveness.

Both online rules need 3R trajectory-direction passes per sweep, compared with
2R for Adam. The adaptive scalar arithmetic does not require another batch
gradient pass. The observed roughly 1.45x training-CPU ratio is consistent
with the online method needing greater progress per sweep to offset its extra
propagation work; it is not a proof that this ratio holds at larger sizes.
No peak-memory comparison was measured. All-core laptop CPU timings and these
retrospective cutoffs are descriptive, not precision microbenchmarks.

## Target-family differences

Mean endpoint fidelities within each family, averaging the two noise replicas:

| Target family | Unscaled online (%) | Adaptive 0.01 (%) | Adaptive 0.02 (%) | Krotov-Adam (%) | Adam 0.02 (%) |
| ------------- | ------------------: | ----------------: | ----------------: | --------------: | ------------: |
| Gaussian      |           82.134548 |         82.242641 |         82.214878 |       82.127436 |     82.322062 |
| TFIM          |           82.278246 |         82.394263 |         82.347391 |       82.414529 |     82.532533 |
| Haar          |           64.713725 |         65.229865 |         65.558094 |       65.113064 |     65.849124 |
| Random MPS    |           82.001871 |         82.081006 |         82.057572 |       81.992075 |     82.133458 |

Rate 0.01 is better than 0.02 on Gaussian, TFIM and random MPS. The global
endpoint preference for 0.02 comes from the Haar target; approximately 80.4%
of its total mean improvement over unscaled online comes from this one target.
It would be misleading to call 0.02 uniformly superior, or to report a
per-target best-rate mixture as a single target-independent method.

The Krotov-Adam combination improves on unscaled Krotov in six of eight pairs
at the endpoint, but loses to Adam 0.02 in every pair, both at the endpoint
and at common CPU. Its Gaussian and random-MPS family means are slightly worse
than unscaled online. This control does not establish an advantage from the
25-update online prefix.

Both optimizer families received three globally fixed candidate rates. Total
training CPU spent on the adaptive rate screen was **45.286040 s**, versus
**31.015929 s** for the Adam rate screen; these tuning costs must not be hidden
when describing an algorithm-selection budget. The fixed combination control
was not given a three-rate sweep and is interpreted only as that control.

## Decision and limits

The predeclared competitive check was **not passed**. As agreed, pause expansion
of these schedules rather than launching a wider rate grid, more trajectories,
or a new-target confirmation panel. Keep weighted Adam as the working baseline
and retain adaptive online as the strongest tested Krotov-style candidate in
this particular matched development setup.

The result narrows the research question: update scaling helps, but a
Krotov-specific advantage in noisy fidelity, CPU or memory has still not been
demonstrated. It does not isolate the benefit of sequential ordering from
adaptivity, because this screen did not include a matched RMS-scaled **batch**
control with the same prefix. Nor does it establish that 25 unscaled prefix
updates are optimal. These are unresolved mechanism questions to consider in
a research-design reassessment, not grounds for automatically extending this
screen or claiming the overall approach impossible.

Only four previously used states were tested, one per family. Two noise
replicas do not make eight independent target instances. There is no held-out
generalization, significance, large-system scaling or state-of-the-art claim.

## Artifacts and checks

- Driver: [state_preparation_adaptive_online.py](state_preparation_adaptive_online.py),
  reusing the weighted experiment runner and shared statevector online sweep.
- Output: `output/statevector-adaptive-online-20260912`, containing 64 private
  checkpoints, manifest, aggregate result and generated summary.
- Manifest checksum:
  `5af3d3eab196eec17693bc1191adb9ee0e728cf55beefc2016ee1fd65ce76933`.
- Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Qiskit 2.4.1.
- Total training CPU: 104.186960 s; exact scoring CPU: 325.108016 s.
  Manifest-to-result wall interval before resume: approximately 104.2 s.
  Startup/source authentication occur before the manifest; this interval is
  not a training-only measurement.
- All 64 checksums, all 3,200 updates, 51 scores per run, adaptive occurrence
  counters, phases and 3R/2R pass budgets were audited. Final exact scores were
  recomputed from saved parameters with zero difference.
- All eight methods share replay hashes at every update in each of eight
  target/noise groups. The five Krotov-prefix configurations have identical
  signals/scores through update 25.
- Sixteen repeated controls (all-online and Adam 0.02) reproduce the previous
  weighted screen's final parameters and fidelities bit for bit. Earlier
  scientific artifacts were not rewritten or resumed under the new revision.
- Explicit serial resume of the completed parallel run preserves every byte
  of all **67 output files** while revalidating sources and signatures.
- **600 optimization tests passed** before launch, including dense-reference
  weighted sequential updates, shared-parameter counter/bias correction,
  moment reset at the switch, malformed-state rejection, serial/parallel
  equivalence, paired streams, cost accounting and resume tests.
- Changed-file formatting/lint/type hooks pass. `uvx nox -s lint` continues to
  report the pre-existing manuscript/old-code issues, missing Homebrew Node
  `simdjson` library and matplotlib missing from its isolated lint environment.
  Unrelated automatic lint edits are restored, not included in this change.
