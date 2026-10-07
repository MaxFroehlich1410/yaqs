# Independent six-qubit pruning comparison

Use all nine saved target states and seeds 300/400/500 (27 selected-depth
unpruned checkpoints). Authenticate the compact inputs from the matched
unpruned campaign, including the original source and fixture hashes. Every
method starts independently from the same original circuit and parameters.
There is no new construction, depth selection, technique composition, or noisy
fine-tuning. Submit every checkpoint/method pair independently (297 jobs), with
bounded concurrency based on physical cores, available RAM, and existing load.

Primary fidelity is ideal fidelity at the end of pruning. The repair arm's
endpoint includes its own noiseless repair; mark it explicitly when comparing
costs. Secondary exact IBM noisy fidelity is assessment only. Evaluate logical
gate count and two-qubit count/depth, plus two-qubit count/depth after the same
fixed-reference compilation used by the incumbent. Report every arm, all seeds,
and all targets. A strict win against B requires fewer logical total gates and
higher ideal fidelity for the same checkpoint; report two-qubit and compiled
resource changes separately. Do not infer a universal win from average fidelity
or choose a different method for each target.

For fidelity F and gate set S, signed deletion damage is F(C)-F(C without S).
Negative damage is an improvement. Threshold comparisons are strict.

| Arm                             | Frozen independent rule                                                                                                                                                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B                               | Score every single deletion on the full original circuit; delete all damages below 0.01 simultaneously. Reproduce the archived B circuit and retained angles exactly.                                                                        |
| Sequential ideal                | Recompute ideal single-deletion scores after each minimum-damage deletion; stop when none is below 0.01.                                                                                                                                     |
| Magnitude                       | Simultaneously remove actual affine rotation angles with magnitude below 0.2 radians after wrapping into [-pi, pi).                                                                                                                          |
| Sequential noisy                | Exact IBM sampler-law density scores; sequentially delete the gate and its noise window only if noisy damage is below -1e-12. No parameter optimization.                                                                                     |
| Pair                            | Score only same-support gate pairs separated by at most three gate positions; sequentially remove the minimum joint ideal damage below 0.01. No single-gate deletion or repair.                                                              |
| Resource weighted               | Recompute ideal single-deletion damage divided by fixed cost 1 for one-qubit and 2 for two-qubit gates; sequentially delete the minimum normalized damage below 0.01. Costs are estimates; actual compiled resources are measured afterward. |
| Ideal repair                    | Sequential ideal single-deletion cutoff 0.01; after each accepted deletion, at most ten safeguarded Armijo L-BFGS noiseless updates, retaining the best including the unrepaired input.                                                      |
| B 0.0025 / 0.005 / 0.015 / 0.02 | Each is a separate simultaneous one-pass B arm with that cutoff.                                                                                                                                                                             |

Tie order follows the circuit order, and pair order is lexicographic. Use these
same settings on every target. No per-target fallback or hyperparameter fitting.

Save the complete decision trace, original gate identities, compact retained
angles, repair updates, and CPU/wall cost. Seal pruning and assessment separately
for recovery. Cross-check every final ideal score against independent Qiskit
density evolution and untruncated MPS compilation equivalence; test the noisy
deletion adjoint and joint pair adjoint against exhaustive independent replays.
Cached results must match the input and method identity and their pruning seal.
