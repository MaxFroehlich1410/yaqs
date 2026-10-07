# Weighted trajectory optimization findings

Follow-up: the adaptive **online** scaling screen is now complete; see the
[64-run adaptive-online findings](state_preparation_adaptive_online_findings.md).
The recommendations below describe the preceding weighted-sampling study.

Completed 2026-09-12 following the
[frozen protocol](state_preparation_weighted_optimization.md).
Weighted sampling has now been used for actual circuit optimization, with the
same estimator and paired physical noise maps supplied to Krotov and Adam.

## Conclusion

**Weighted sampling works as a shared estimator improvement, but does not make
the tested Krotov schedules competitive with Adam.** Adam benefits much more in
final fidelity, remains cheaper, and wins every matched weighted comparison
against both all-online and hybrid Krotov at both trajectory budgets.

This is a useful negative result for the current Krotov update schedules and a
positive result for weighted trajectory optimization. It is not evidence that
all possible Krotov schedules fail, nor a new-target or state-of-the-art result.

## What was actually run

- Four existing q6 development targets: Gaussian, critical TFIM, Haar and
  random bond-2 MPS, one state per family. Each uses its frozen 108-gate
  construction-replica-11 circuit and initial parameters. No new pruning,
  circuit growth, construction starts or target states.
- Noise replicas 11 and 29; calibrated local Pauli-Lindblad noise with
  one-/two-qubit average gate errors 0.00064/0.0051, unchanged from the previous
  screen. The mean initial exact noisy fidelity is 77.278521%.
- All-online Krotov: 50 gate-sequential sweeps at 0.7. Hybrid Krotov: 25 online
  sweeps at 0.7 followed by 25 batch updates at 0.3. Adam: 50 updates at 0.02,
  beta1=0.9, beta2=0.999, epsilon=1e-8, zero initial moments. No retuning.
- Ordinary and weighted sampling, each at total path budgets R32/R128;
  fresh maps every update, replayed consistently within each update.
- **96 complete runs, 4,800 updates.** Forty-eight ordinary controls were
  rerun under the same source revision and timing convention. Execution used
  eight single-threaded processes with fixed shuffled job order.

Weighted sampling propagates one no-factor path with its exact probability
`p0=0.7658428756735005`, and R-1 paths conditional on at least one selected
Pauli factor, each weighted by `(1-p0)/(R-1)`. Multiple events and cancellations
are retained. The channel is parameter-independent and random-unitary; this
does not implement general normalized nonunitary jumps.

All methods optimize the weighted mean of **individual trajectory
infidelities**. In online Krotov, weights are applied once in the old costates,
and all trajectory contributions are summed before each gate update. This is
not separate optimization of each trajectory followed by parameter averaging.
For Adam and the batch tail, the signal is the gradient of exactly the same
weighted sampled loss. Exact density scores never enter training or stopping.

## Endpoint fidelity and training cost

Fidelity is the exact noisy score at update 50, not the best visited checkpoint.
Each entry averages the same eight target/noise-replica pairs. CPU includes
engine/sampler setup, sampling and updates; excludes exact scoring, replay-hash
construction and disk logging. It is not end-to-end runtime or a memory result.

| Method            |   R | Ordinary fidelity (%) | Weighted fidelity (%) | Ordinary CPU (s) | Weighted CPU (s) |
| ----------------- | --: | --------------------: | --------------------: | ---------------: | ---------------: |
| All-online Krotov |  32 |             77.768017 |             77.782097 |         2.908364 |         1.878182 |
| Hybrid Krotov     |  32 |             77.694438 |             77.705545 |         2.612377 |         1.599715 |
| Adam              |  32 |             78.002904 |             78.209294 |         2.345248 |         1.331458 |
| All-online Krotov | 128 |             77.779174 |             77.781913 |         9.905157 |         6.154636 |
| Hybrid Krotov     | 128 |             77.701232 |             77.703811 |         8.947497 |         5.257415 |
| Adam              | 128 |             78.146122 |             78.262976 |         8.136951 |         4.291828 |

| Weighted minus ordinary |   R | Mean change (pp) | Positive pairs / 8 | CPU reduction |
| ----------------------- | --: | ---------------: | -----------------: | ------------: |
| All-online Krotov       |  32 |        +0.014080 |                  7 |         35.4% |
| Hybrid Krotov           |  32 |        +0.011107 |                  6 |         38.8% |
| Adam                    |  32 |        +0.206390 |                  8 |         43.2% |
| All-online Krotov       | 128 |        +0.002739 |                  5 |         37.9% |
| Hybrid Krotov           | 128 |        +0.002579 |                  4 |         41.2% |
| Adam                    | 128 |        +0.116855 |                  8 |         47.3% |

The sampling implementation matters to the speed comparison: the conditional
sampler precomputes probabilities and builds sparse selected-factor maps,
whereas ordinary sampling retains the existing per-gate provider path.
Consequently, the measured speedup combines estimator design and implementation;
it is not a pure variance-reduction speedup or a Krotov-specific benefit.
Timing is descriptive on an all-core, heterogeneous-core laptop, not a precision
microbenchmark. Both samplers were timed in the same interleaved screen.

Weighted R32 Adam exceeds ordinary R128 Adam by **0.063172 pp**, using **16.4%
of its training CPU** in this implementation. This is a promising practical
small-system baseline, but still an exploratory four-target comparison.

## Does Krotov beat Adam with the same weighted estimator?

No, for these schedules, starts and budgets:

| Krotov minus weighted Adam |   R | Mean fidelity difference (pp) | Krotov wins / 8 | Krotov / Adam CPU |
| -------------------------- | --: | ----------------------------: | --------------: | ----------------: |
| All-online                 |  32 |                     -0.427197 |               0 |             1.411 |
| Hybrid                     |  32 |                     -0.503749 |               0 |             1.201 |
| All-online                 | 128 |                     -0.481064 |               0 |             1.434 |
| Hybrid                     | 128 |                     -0.559165 |               0 |             1.225 |

At R32, all-online's eight differences range from -1.157590 to -0.098363 pp;
at R128, from -1.326477 to -0.140992 pp. Thus the aggregate loss is not caused
by a single pair reversing an otherwise favorable comparison. The previous
ordinary all-online gap was -0.234887/-0.366948 pp at R32/R128: **weighting
widens rather than closes the gap**, because it helps Adam more.

For a descriptive equal-CPU check, each pair uses the smaller of the two
measured final CPU budgets and the last completed checkpoint within that
budget. No interpolation, best-checkpoint selection or further training:

| Weighted Krotov schedule |   R | Krotov minus Adam at common CPU (pp) | Krotov completed updates | Adam updates |
| ------------------------ | --: | -----------------------------------: | -----------------------: | -----------: |
| All-online               |  32 |                            -0.506106 |                    33-37 |           50 |
| Hybrid                   |  32 |                            -0.529446 |                    36-41 |           50 |
| All-online               | 128 |                            -0.562309 |                    34-35 |           50 |
| Hybrid                   | 128 |                            -0.589699 |                    37-39 |           50 |

All four comparisons have zero Krotov wins among eight pairs. These retrospective
CPU cutoffs are an analysis of saved curves, not a CPU-based training rule.

## Family-level results and remaining bottleneck

Weighted endpoint fidelities, averaged over the two noise replicas:

| Target family | All-online R32 (%) | Hybrid R32 (%) | Adam R32 (%) | All-online R128 (%) | Hybrid R128 (%) | Adam R128 (%) |
| ------------- | -----------------: | -------------: | -----------: | ------------------: | --------------: | ------------: |
| Gaussian      |          82.134548 |      82.095749 |    82.322062 |           82.130327 |       82.090021 |     82.313909 |
| TFIM          |          82.278246 |      82.197832 |    82.532533 |           82.285444 |       82.201038 |     82.571844 |
| Haar          |          64.713725 |      64.549567 |    65.849124 |           64.710278 |       64.546487 |     65.992415 |
| Random MPS    |          82.001871 |      81.979031 |    82.133458 |           82.001602 |       81.977699 |     82.173738 |

R128 minus R32 weighted fidelity is -0.000185 pp for all-online and -0.001734 pp
for hybrid, versus +0.053682 pp for Adam. More trajectories do not materially
help these Krotov schedules. The Haar target remains the largest optimizer gap.

During updates 26-50, weighted R32 all-online gains 0.147565 pp, hybrid gains
0.071012 pp, and Adam gains 0.289623 pp. At R128 the corresponding gains are
0.146099, 0.067998 and 0.316819 pp. This is consistent with an update-scaling or
schedule limitation beyond raw trajectory variance. It does not by itself
identify conditioning, learning rate, or stale-adjoint bias as the cause.

Weighted R128 Adam's mean 78.262976% is slightly above the previous exact-noisy
Adam reference of 78.232451%. That reference is one finite 50-update run at a
fixed rate, **not an upper bound or global optimum**. Different stochastic
optimization paths can end above it; no exact-gradient superiority claim follows.

## Recommended next decision

Keep weighted R32 as the economical development estimator and weighted Adam
as the comparison baseline. Do not spend a larger trajectory budget on the
unchanged Krotov schedules, and do not promote them to a new-target confirmation
panel on this evidence.

The next small Krotov-specific test should isolate **update scaling**, using
the same weighted estimator, circuits and development targets: a diagonal
adaptive or momentum tail, with an online-Krotov-to-Adam tail as an explicitly
labeled practical control. Charge all work to both methods and compare to
weighted Adam at equal cost. A Krotov-Adam combination must not be presented
as a pure Krotov win. This follow-up is proposed, not launched here.

Only a configuration with a credible matched-cost benefit should be frozen and
evaluated on genuinely new targets. The current evidence establishes neither a
Krotov-specific fidelity/cost advantage nor large-system memory superiority.
The known no-jump/conditional-stratum sampling idea is not itself a novelty claim.

## Artifacts and verification

- Driver: [state_preparation_weighted_optimization.py](state_preparation_weighted_optimization.py).
- Source: `output/state-preparation-trajectory-comparison-20260910`.
- Output: `output/statevector-weighted-optimization-20260912`, including frozen
  manifest, 96 per-run checkpoints, aggregate result and generated summary.
- Manifest checksum:
  `30a488a5cb473d4f6a9f5cbb34964637e284be9e2237c26a36c88799126454b5`.
- Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Qiskit 2.4.1, complex128
  statevector training; exact density evaluation remains separate.
- Aggregate training CPU: 442.950611 s. Exact scoring CPU: 488.679518 s.
  Manifest-to-result wall interval: approximately 136.5 s; this is not the
  training-only time. Source loading and command startup precede the manifest.
- Audited all 96 checksums, 4,800 updates and 51 scores per run. Recomputed all
  final scores from saved parameters with zero difference. All 32 paired
  method groups share physical replay hashes at every update. Both Krotov
  schedules have identical signals/scores through update 25 in each group.
- All 48 ordinary controls reproduce the historical final parameter vectors
  and fidelities **bit for bit**. Earlier scientific artifacts were not rewritten.
- Explicit completed resume validates sources, configuration and checkpoints
  and preserves every byte of all 99 output files.
- The full 552-test optimization suite passed before launch. Additional
  regression coverage includes weighted adaptive/backtracking resume; the
  expanded targeted subset is checked separately after the run. Changed-file
  lint/type/format hooks pass. `uvx nox -s lint` still reports pre-existing
  manuscript/old-code issues and unavailable Homebrew Node/matplotlib tooling;
  its unrelated automatic edits are restored rather than included in this work.

Four target families here mean **four previously used states**, not broad
family-level generalization. The two noise replicas do not make eight
independent target instances. No formal significance or publication-level
competitive claim is inferred from this small development panel.
