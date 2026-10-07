# Completed legacy weighted-trajectory experiment: all q6 targets

All **nine normative six-qubit target states** were tested at optimizer seeds
**300 and 400**. Of 108 paired fine-tuning conditions, **102 completed** and
**six stopped at the existing no-jump coverage check**. The 102 include 16
empty-circuit comparisons. There were no construction-threshold failures.

For the archived 150-gate top-down route, mean exact IBM fidelity rises from
**66.8730% to 69.6116%**, a **+2.7386 percentage-point** change. **11 of 18**
paired runs improve and seven decrease. The median change is only **+0.1001
percentage points**: several large gains dominate the mean. This is a mixed
result over two fixed optimizer seeds, not a universal improvement claim.

## Archived top-down: each benchmark target

The first table averages the two optimizer seeds for each fixed target.
All fidelities are exact noisy density-matrix fidelities under the unchanged
IBM sampler law, expressed in percent. Every circuit in this table has 150 gates.

| Target             | 3 histories | 100 weighted histories | Change (pp) |
| ------------------ | ----------: | ---------------------: | ----------: |
| Gaussian           |     80.6536 |                83.4912 |     +2.8376 |
| TFIM ferromagnetic |     51.5671 |                69.4099 |    +17.8428 |
| TFIM critical      |     56.7662 |                58.7214 |     +1.9552 |
| TFIM paramagnetic  |     81.7731 |                81.8586 |     +0.0856 |
| Haar 1             |     53.4041 |                60.5232 |     +7.1191 |
| Haar 2             |     68.9024 |                64.8360 |     -4.0663 |
| Haar 3             |     70.7020 |                67.8666 |     -2.8354 |
| Random MPS, bond 2 |     63.4914 |                63.4425 |     -0.0490 |
| Random MPS, bond 3 |     74.5973 |                76.3552 |     +1.7578 |

The largest gain is TFIM ferromagnetic at optimizer seed 300: 22.4270% to
58.0083%. Haar states 2 and 3 decrease at both seeds. The bond-2 random-MPS
mean is nearly unchanged. These differences are training outcomes; exact test
scoring has no trajectory sampling error. Two optimizer seeds provide a small
replication check, not a precise estimate of variability over all initializations.

| Target             | Seed 300: 3 | Seed 300: 100 | Seed 400: 3 | Seed 400: 100 |
| ------------------ | ----------: | ------------: | ----------: | ------------: |
| Gaussian           |     82.8589 |       83.1100 |     78.4484 |       83.8725 |
| TFIM ferromagnetic |     22.4270 |       58.0083 |     80.7071 |       80.8114 |
| TFIM critical      |     72.3500 |       76.2578 |     41.1823 |       41.1850 |
| TFIM paramagnetic  |     83.6116 |       83.5670 |     79.9345 |       80.1503 |
| Haar 1             |     42.5504 |       58.8215 |     64.2577 |       62.2249 |
| Haar 2             |     74.5185 |       73.0530 |     63.2862 |       56.6191 |
| Haar 3             |     68.8331 |       65.0929 |     72.5708 |       70.6402 |
| Random MPS, bond 2 |     81.8836 |       81.9794 |     45.0993 |       44.9056 |
| Random MPS, bond 3 |     78.7774 |       81.8718 |     70.4172 |       70.8385 |

## All method families

These means use completed pairs only, always comparing the same paired subset.
Cutoff means exclude coverage failures and therefore do not cover all 18
target/seed cases; their failure counts are part of the outcome. Empty circuits
remain included. Methods differ in gate counts and construction/update budgets,
so cross-method means are descriptive, not a matched-budget comparison.

| Method                  | Noise  | Completed / planned |  Mean 3 | Mean 100 | Mean change (pp) |
| ----------------------- | ------ | ------------------: | ------: | -------: | ---------------: |
| Archived top-down       | IBM    |             18 / 18 | 66.8730 |  69.6116 |          +2.7386 |
| Archived layerwise      | IBM    |             18 / 18 | 74.2781 |  74.2826 |          +0.0045 |
| Unpruned hybrid control | IBM    |             18 / 18 | 84.5774 |  84.8239 |          +0.2465 |
| Unpruned hybrid control | Google |             18 / 18 | 76.5651 |  77.0637 |          +0.4985 |
| Literal cutoff          | IBM    |             14 / 18 | 13.4251 |  14.3085 |          +0.8833 |
| Literal cutoff          | Google |             16 / 18 | 11.9173 |  12.8507 |          +0.9334 |

Archived layerwise circuits have 108 gates; archived top-down retains its
150-gate floor. The archived VQA references have 198 gates, no separate noisy
fine-tuning intervention, and retain three-trajectory training. Their mean exact
IBM fidelity is 54.2151%. These are the historical reference implementations;
no new claim of canonical ADAPT-VQE implementation is made.

## Unchanged cutoff and coverage failures

The cutoff is still `abs(theta * dF/dtheta) >= 0.01`. It deletes all gates in
**eight target/seed cases** below. Both noise profiles then give an empty-circuit
comparison, accounting for 16 of the 102 completed pairs. These are completed
diagnostics, not successful state-preparation results.

| Target             | Gates after cutoff, seed 300 | Gates after cutoff, seed 400 |
| ------------------ | ---------------------------: | ---------------------------: |
| Gaussian           |                            4 |                            4 |
| TFIM ferromagnetic |                           17 |                            8 |
| TFIM critical      |                            0 |                            0 |
| TFIM paramagnetic  |                            0 |                            1 |
| Haar 1             |                            0 |                            1 |
| Haar 2             |                            0 |                            0 |
| Haar 3             |                           27 |                            7 |
| Random MPS, bond 2 |                            0 |                            0 |
| Random MPS, bond 3 |                            3 |                            5 |

The six nonempty cutoff trials below sampled 100 histories with **zero jumps**.
The existing guard stopped them before 100-history parameter updates. Their
100-history fidelity is **missing**, not zero and not silently replaced by a
noiseless result. The same fixed banks were independently reproduced during
audit, confirming zero events. No alternate seeds or forced jumps were used.

| Target            | Optimizer seed | Noise  | Remaining gates |
| ----------------- | -------------: | ------ | --------------: |
| Gaussian          |            300 | IBM    |               4 |
| Gaussian          |            300 | Google |               4 |
| Gaussian          |            400 | IBM    |               4 |
| Gaussian          |            400 | Google |               4 |
| TFIM paramagnetic |            400 | IBM    |               1 |
| Haar 1            |            400 | IBM    |               1 |

This establishes that a fixed budget of 100 histories does not guarantee jump
coverage for very short pruned circuits. Neither the sample budget nor the
pruning rule was adjusted in response to these outcomes.

## Representation, fixed settings, and validation

- Exact fixture vectors were loaded directly in little-endian order, including
  the Gaussian, three uniform TFIM, three Haar, and two random-MPS vectors.
  Random-MPS target generation does not mean MPS-based training.
- Training uses **64-amplitude dense statevectors**. Final Qiskit evaluation
  uses **64-by-64 density matrices**, integrating all channel outcomes
  independently of the training histories. It never selects optimizer settings
  or checkpoints. Both IBM/Google and sampler/full-Lindblad scores are retained.
- Target generation and optimizer seeds are distinct. The only extension to the
  prior experiment is target identity and register size; construction, pruning,
  seed formulas, update counts, rates, and CRN policies are unchanged.
- The fixture is from the benchmark definition. Noise remains the previous
  IBM/Google logical Pauli profiles. This is a separate noise-aware extension,
  not the base benchmark’s Ballarin/dephasing/depolarizing matrix.
- In all **1,868 completed nonempty-circuit bank uses**, at least one of the
  100 sampled histories contains a jump. Weights and nested first-three histories
  were verified in every completed bank. All three/100 pairs share identical
  initial circuits and parameters.
- The **1,884 completed bank uses** contain 188,400 sampled histories, compressed
  to **24,597 retained histories**. This includes empty circuits. It is a bank
  inventory, not a count of all propagated path-steps or independent banks:
  frozen banks repeat across updates and CRN can be shared across targets.
- All **318 checkpoint** circuit/parameter checksums passed; source and fixture
  hashes remained unchanged. Maximum vector/Qiskit fidelity discrepancy is
  `3.77e-15`; maximum matrix trace error `2.22e-14`, maximum Hermiticity error
  `8.05e-16`, and minimum eigenvalue `-1.28e-15` are floating-point roundoff.
- **36 targeted tests passed**, including reduced-budget parameter-exact parity
  with the previous construction/pruning code, real q6 weighted/refreshed banks,
  empty circuits, coverage failures, threshold failures, and authenticated resume.
  Fixture regeneration check, targeted Ruff, and project type checks passed.
- Required repository-wide `uvx nox -s lint` still fails on pre-existing
  manuscript/plot/MPS findings, the broken Homebrew Node/simdjson dependency, and
  archived-output type checks. Unrelated automatic edits were restored.

## Artifacts and reproduction

[Frozen protocol](legacy_topdown_q6_protocol.md) records the exact settings.
The driver is [legacy_topdown_q6.py](legacy_topdown_q6.py).

```sh
uv run python -m experiments.legacy_topdown_q6 \
  --output NEW_DIRECTORY --workers 4
```

For the existing run, add `--resume` and use
`output/legacy-topdown-q6-weighted100-20260914`. Resume requires identical
sources, fixture and package versions. Each job authenticates its checkpoints.
Jobs ran concurrently with four processes and one BLAS thread per process;
recorded wall times should not be compared directly with prior serial q8 timing.

- [All 108 paired conditions, including failures](../output/legacy-topdown-q6-weighted100-20260914/all_pairs.md)
- [Aggregate result data](../output/legacy-topdown-q6-weighted100-20260914/results.json)
- [Source and fixture manifest](../output/legacy-topdown-q6-weighted100-20260914/manifest.json)
- [Complete numerical and trajectory audit](../output/legacy-topdown-q6-weighted100-20260914/validation/audit.json)
