# Frozen four-setting q6 pruning comparison

This experiment changes pruning only, using all nine normative six-qubit
fixture targets and optimizer seeds 300 and 400. Each setting starts from the
same authenticated `unpruned_initial` checkpoint in
`output/legacy-topdown-q6-weighted100-20260914`. The checkpoints were grown
one layer at a time with the existing noiseless hybrid Krotov routine until
at least 90% fidelity. No target amplitudes or construction parameters change.

## Predeclared settings

| Setting | Score                                                                 | Deletion schedule                                            | Intermediate repair                                         |
| ------- | --------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| A       | Absolute parameter times noiseless loss derivative                    | Simultaneous                                                 | None                                                        |
| B       | Signed exact noiseless fidelity loss from deleting each complete gate | Simultaneous                                                 | None                                                        |
| C       | Same as B                                                             | Delete the smallest score, then recompute all scores         | None                                                        |
| D       | Same as B                                                             | Delete the smallest score, repair, then recompute all scores | 10 noiseless updates: 5 online, 5 batch, rate 0.7, no decay |

All settings use the strict threshold `score < 0.01`. For B-D this is one
fidelity percentage point per candidate deletion, not a cumulative budget.
Negative removal damage is eligible. Ties follow original gate order. C/D
stop when no gate is eligible or the circuit is empty. There is no fidelity
floor, rollback, gate-count floor or additional hyperparameter search. D uses
repair seed 42; repair is deterministic and skips an empty circuit. Its ten
updates are unconditional, with no early stop at 90% and no checkpoint selection.

Scores for B-D use one shared forward/backward **statevector** tape: the
overlap between the state before a gate and the backwards-propagated target
after it gives the exact deleted-circuit amplitude. Squaring its modulus
gives the finite removal fidelity. Every accepted deletion and every candidate
score is recorded, including cumulative damage and repair traces.

## Unchanged fine-tuning and evaluation

All 144 planned conditions (4 settings x 9 targets x 2 seeds x 2 noise profiles)
are run, including a fresh baseline A parity check against the saved results.
The existing IBM/Google logical Pauli coefficients and at-most-one-jump gate
windows are unchanged. These are historical simulation profiles, not a new
hardware calibration or the base benchmark's standard noise matrix.

Each final circuit samples 100 maps with noise counter `optimizer_seed + 5000`.
Only identical no-jump histories are merged, with their empirical multiplicity
weight; each jump history remains separate. Maps stay frozen during 200 hybrid
cross-trajectory updates (100 online, 100 batch; rate 0.2; no decay). The same
seed formula does not align physical gate events between different circuits.

The existing coverage guard stops a nonempty noisy circuit if its 100 maps
contain no jumps. No replacement bank or forced jump is used. Such a condition
is reported as missing final fidelity, with its post-pruning score and resource
counts retained. Empty circuits remain explicit diagnostics. Their histories
cannot contain gate jumps. Neither kind of failure is silently excluded.

Final evaluation independently propagates a 64-by-64 Qiskit density matrix
under the exact sampler channel, including all jump and no-jump outcomes.
Full Lindblad and noiseless scores are supplementary. The fixed final iterate
is scored; final noisy scores never choose deletions, steps or checkpoints.
The cross-trajectory update is preserved as implemented; this experiment does
not newly establish it as the gradient of the mean trajectory fidelity.

## Resource comparison

Report logical one-/two-qubit counts and logical two-qubit depth. Also compile
each final numerical circuit with Qiskit to `rz, sx, x, cx`, bidirectional
linear connectivity, identity initial layout, optimization level 1,
`seed_transpiler=1729`, and `approximation_degree=1.0`. Count compiled CX gates,
two-qubit depth (dependency-aware filtering), and total depth. Use the same
reference compiler for both noise profiles; this is not Heron/Willow native
compilation. Verify the complete compiled unitary after layout correction,
up to a global phase. Save compiled QASM as a resource artifact.

Engineering validation found that Qiskit's level-3 two-qubit synthesis changed
a near-identity test circuit by about 2e-5 in unitary entries even with
`approximation_degree=1.0`. Level 1 reproduced it to floating-point roundoff,
so level 1 is fixed for every condition before the scientific runs.

Noisy fidelity is evaluated on the original logical circuit, not on this
compiled reference. Thus the experiment does not claim noisy compiled-device
fidelity, and the logical noise model has no idle-duration decoherence term.

Record pruning CPU/wall time, scored candidate counts, repair update counts,
and fine-tuning CPU/wall time, including sampling and compression. Record
deterministic local-vector application counts for scoring, repair and ideal
gate propagation during fine-tuning; the latter excludes noise-operator
applications and cross-trajectory overlap contractions. CPU cost is compared
after the shared construction checkpoint. Historical construction CPU is
unavailable; its update count is reported separately. Evaluation and compilation
time are excluded from optimization costs. Four workers each use one numerical
thread. D spends additional repair compute, so it is not compute-matched to C.

## Analysis and reproducibility

Report all 144 rows and per-target/per-seed results. For every comparison use
the intersection of completed target/seed cases and disclose its size. Report
empty-circuit and coverage-failure counts separately. Use two-seed means as a
small development comparison, not a statistical generalization claim.

Inputs, source files and package versions are frozen before execution. Atomic
checksummed checkpoints permit identical-source resume; historical artifacts
are read-only. Run with:

```sh
uv run python -m experiments.legacy_pruning_ablation \
  --output NEW_DIRECTORY --workers 4
```

Add `--resume` only for the same output and identical configuration.
