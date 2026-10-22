# Guarded adaptive ideal pruning

This experimental, pruning-only policy uses the same saved construction angles
and targets as simultaneous B and sequential C. No construction, ideal repair,
noisy training or noisy assessment is performed. Production defaults are unchanged.

The first comparison fixes signed deletion cutoff 0.01 and an absolute cumulative
ideal-fidelity loss budget of 0.05 for all nine targets, optimizer seeds 300/400/500,
at both six and twelve qubits. These are declared engineering choices, not tuned
optimal hyperparameters. The fidelity floor is the original input fidelity minus
0.05, including when a budget-exhausted Haar input starts below 99%.

1. Compute exact signed single-gate deletion damages on the current circuit.
2. Propose all gates with damage strictly below 0.01, as in B.
3. Replay the proposed batch exactly. Accept only if its fidelity is at least the
   original fidelity minus 0.05 (numerical tolerance 1e-12).
4. On rejection, halve the batch size, keeping the lowest-damage prefix with
   stable original-order ties. Repeat down to C's minimum-damage single gate.
5. Rescore after every accepted batch. Stop when there are no eligible scores,
   the minimum-damage single gate fails the cumulative guard, or the circuit is empty.

The batch search is heuristic: it does not exhaust all subsets or find a globally
minimal circuit. The guard bounds actual endpoint loss despite interacting gate
deletions. Retained parameter values and gate order remain unchanged. Statevector
scoring avoids a dense density matrix, including at twelve qubits.

Report arithmetic means and sample variability across the three optimizer seeds,
with per-seed data. Pruning CPU time excludes input authentication, serialization,
construction and later noisy refinement. Identify archived timings separately from
timings collected on the current machine; circuit and fidelity comparisons always
share each case's exact construction parent. Six- versus twelve-qubit trends also
reflect the different stored preparation policies and budgets.
