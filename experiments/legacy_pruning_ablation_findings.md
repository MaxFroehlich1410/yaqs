# Completed four-setting pruning comparison

All nine six-qubit fixture targets were evaluated with optimizer seeds 300 and
400 and both unchanged logical noise profiles: **144 planned conditions**.
There are **136 completed fine-tuning endpoints and eight coverage failures**.
All starting circuits and parameters are the same authenticated checkpoints
used in the previous q6 experiment. Baseline A reproduces the saved pruning,
training maps and final parameters exactly (maximum parameter discrepancy zero).

**Replacing saliency by actual removal sensitivity produces the largest gain.**
Sequential deletion alone gives smaller circuits with slightly lower final
fidelity on matched cases. Adding repair gives the highest mean fidelity, but
costs more, retains more gates, and has a large failure on one TFIM seed.

## What was changed

| Setting | Difference from preceding setting                                                       |
| ------- | --------------------------------------------------------------------------------------- |
| A       | Current literal absolute parameter-times-gradient cutoff, simultaneous deletion         |
| B       | Replace only the score by signed exact noiseless gate-removal damage                    |
| C       | Delete one gate at a time and recompute scores                                          |
| D       | Add ten noiseless repair updates after each deletion: five online, five batch, rate 0.7 |

The strict cutoff remains 0.01. There is no cumulative fidelity floor,
rollback, noise-aware pruning or additional parameter search. Noisy fine-tuning
remains 200 hybrid cross updates at rate 0.2 with 100 frozen empirical maps
and weighted no-jump compression. See the [frozen protocol](legacy_pruning_ablation_protocol.md).

## Final fidelity and circuit resources

Fidelity means below use completed cases; parentheses give the number out of 18. Missing outcomes are not zero. Resource means cover all 18 cases, including
empty circuits and post-pruning circuits when fine-tuning fails its coverage check.

| Setting | IBM fidelity (%) | Google fidelity (%) | Logical gates | Logical 2q gates | Compiled CX | Compiled 2q depth |
| ------- | ---------------- | ------------------- | ------------- | ---------------- | ----------- | ----------------- |
| A       | 14.3085 (14)     | 12.8507 (16)        | 4.28          | 2.17             | 4.33        | 2.78              |
| B       | 86.6564 (18)     | 82.7451 (18)        | 47.44         | 16.39            | 32.56       | 16.00             |
| C       | 86.0682 (17)     | 82.3910 (17)        | 42.89         | 14.83            | 29.44       | 14.67             |
| D       | 89.1669 (18)     | 84.1610 (18)        | 58.50         | 21.17            | 42.33       | 21.33             |

The compiler uses the same reference `rz/sx/x/cx` basis, six-qubit line,
identity layout, level 1 and seed 1729 for every condition. These are reference
CX resources, not calibrated Heron/Willow native resources. Final noisy fidelity
is evaluated on the logical circuit under the unchanged sampler law, not on
the compiled circuit. Compiled circuits have independently verified equivalent
unitaries, up to global phase.

Mean noiseless fidelity immediately after pruning/repair is **7.6614%,
78.3592%, 81.8573%, and 96.0240%** for A, B, C and D respectively. Thus B still
exhibits accumulated deletion damage, despite replacing the defective score.

## Matched changes

Every row compares identical completed target/seed cases. Positive changes
favor the later method. Ties use a 1e-8 percentage-point tolerance.

| Comparison | IBM change (pp) | IBM wins / ties / losses | Google change (pp) | Google wins / ties / losses |
| ---------- | --------------- | ------------------------ | ------------------ | --------------------------- |
| A to B     | +71.6014, n=14  | 14 / 0 / 0               | +68.4923, n=16     | 16 / 0 / 0                  |
| B to C     | -0.5200, n=17   | 3 / 1 / 13               | -0.1432, n=17      | 6 / 1 / 10                  |
| C to D     | +2.7015, n=17   | 16 / 0 / 1               | +1.2019, n=17      | 15 / 0 / 2                  |
| B to D     | +2.5105, n=18   | 15 / 0 / 3               | +1.4159, n=18      | 14 / 0 / 4                  |

C reduces mean logical gate count by about 9.6% relative to B. D retains
about 23.3% more logical gates and 30.0% more compiled CX gates than B.
D reaches at least 90% exact noisy fidelity in **11/18 IBM and 9/18 Google**
cases, compared with **6/18 and 3/18** for B.

## Repair cost and instability

Mean CPU seconds from the shared construction checkpoint through final
fine-tuning are **2.066 / 2.626** for B (IBM / Google),
**2.633 / 2.984** for C, and **7.606 / 8.378** for D. C's mean uses its 17
completed cases; B and D use all 18. Scoring, sampling and compression are
included; independent evaluation and compilation are excluded. Historical
construction CPU is unavailable and is shared by all variants.

D performs a mean **620 additional repair updates per target/seed circuit**
(11,160 in total over the 18 noiseless pruning runs). Its mean optimization
CPU cost is about **3.68 times B for IBM and 3.19 times B for Google**.
This comparison does not establish a gain at equal gate or compute budgets.

The important counterexample is **ferromagnetic TFIM, optimizer seed 400**.
Adding repair changes final IBM fidelity from **92.1554% (C) to 63.0256% (D)**,
and Google fidelity from **88.6614% to 60.6442%**. The recorded first deletion
actually improves ideal fidelity from 91.9308% to 92.1228%. Its repair then
reaches 96.5193% before the remaining fixed-rate batch steps reduce it to
61.0050%. Similar collapses recur during pruning. Final noiseless fidelity
after D's pruning is 66.0479%. This identifies unstable repair updates in
this run, rather than damage from the first gate removal or test trajectories.
No rollback or rate change was introduced after observing this outcome.

## Coverage and validation

- A deletes every gate for the same eight target/seed cases as before, yielding
  16 empty-circuit conditions. B, C and D never produce an empty circuit.
- A reproduces its six prior zero-jump coverage failures: both Gaussian seeds
  under both profiles, plus TFIM paramagnetic seed 400 and Haar 1 seed 400
  under IBM. C additionally fails coverage for TFIM paramagnetic seed 400
  under both profiles (16 remaining single-qubit gates). No bank was replaced.
- All **120 completed nonempty-circuit training banks contain jumps**.
  The 136 completed banks, including 16 empty-circuit banks, contain 13,600
  sampled histories compressed to **1,282 retained histories**. Eight further
  sampled banks failed coverage. All weights reproduce empirical multiplicities.
- Training and pruning use dense **64-amplitude vectors**. Final evaluation
  independently propagates **64-by-64 density matrices**, integrating every
  noise outcome rather than reusing training trajectories. Random-MPS targets
  are dense fixture vectors; no MPS propagation is used.
- The audit verifies **2,169 initial removal candidates** and **2,513 sequential
  deletions**, plus their repair endpoints, using independent Qiskit statevector
  calculations. Maximum fidelity discrepancy is **4.22e-15**.
- All **208 checkpoint** checksums pass. Maximum vector/matrix fidelity error
  is **1.11e-15**, maximum trace error **4.44e-15**, maximum Hermiticity error
  **3.89e-16**, and minimum density eigenvalue **-6.71e-16**. The maximum compiled
  unitary discrepancy is **7.34e-15**. Inputs and frozen simulation sources
  remain unchanged.
- **54 targeted tests pass**, covering exact deletion against exhaustive
  Qiskit calculations, parameter permutations, simultaneous accumulated damage,
  sequential repair, frozen weighted updates, resource counts, missing-data
  pairing, checkpoint authentication, and QASM round trips. Targeted Ruff and
  type checks pass. Repository-wide lint retains pre-existing manuscript/plot/MPS
  findings, the broken Node/simdjson dependency, and archived-output type errors;
  unrelated automatic changes were restored.

## Interpretation and complete artifacts

These runs favor **B as the simple replacement for the current cutoff**.
**D is the strongest mean-fidelity candidate**, with a material repair-stability
problem and higher resource use. C offers a smaller-circuit tradeoff but does
not improve mean final fidelity over B. The next repair experiment should
address stability explicitly; it should not be silently folded into these results.
Two optimizer seeds per fixed target are a development comparison, not a claim
of performance over arbitrary targets or initializations.

- [Full comparison, including all nine targets under both profiles](../output/legacy-pruning-ablation-q6-20260914/comparison.md)
- [All 144 individual conditions](../output/legacy-pruning-ablation-q6-20260914/all_conditions.md)
- [Machine-readable results](../output/legacy-pruning-ablation-q6-20260914/conditions.json)
- [Numerical audit](../output/legacy-pruning-ablation-q6-20260914/validation/audit.json)
- [Experiment driver](legacy_pruning_ablation.py) and [analysis driver](legacy_pruning_analysis.py)

![All completed conditions](../output/legacy-pruning-ablation-q6-20260914/comparison.png)
