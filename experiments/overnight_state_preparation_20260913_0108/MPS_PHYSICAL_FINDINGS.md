# Separate physical follow-up of the failed MPS gate

The new, frozen q6 physical diagnostic passes all 24 step units across four development targets and both source endpoints. The original unconditional parameter gate remains failed for all eight endpoints. No optimizer constant, gradient, source circuit, old tolerance or saved acceptance flag was modified.

The independent saved-data audit authenticated the new protocol, summary, all four results and 56 case dependencies: 62 files in total. The driver also authenticates the complete original input manifest and numerical sources. The audit recomputed gradient discrepancies from saved arrays and phase-aligned state distances using the algebraically equivalent alignment on the opposite vector. No further circuit propagation or optimization was performed. Full derived values, locations, per-step bounds and hashes are in `output/overnight-state-preparation-20260913-0108/mps-physical-followup/ANALYSIS.json`.

## Numerical evidence

| Quantity                                           | Maximum observed |
| -------------------------------------------------- | ---------------: |
| Same-theta gradient discrepancy                    |     2.290269e-15 |
| Same-theta weighted loss/fidelity discrepancy      |     5.995204e-15 |
| Same-theta individual path-fidelity discrepancy    |     8.881784e-15 |
| Same-theta phase-aligned state distance            |     8.784010e-15 |
| Squared-norm drift                                 |     1.287859e-14 |
| Different saved-theta phase-aligned state distance |     1.509320e-10 |
| Different saved-theta target-fidelity difference   |     2.189693e-12 |
| Raw parameter discrepancy, retained unchanged      |      3.951687e-9 |
| Largest parameter-only state-distance bound        |      1.039383e-8 |

All same-theta errors satisfy the original `1e-10` gradient/physical-state/norm thresholds by a wide margin. The largest same-theta gradient difference occurs at structural Haar before update 2. The largest same-theta state discrepancy occurs at original108 MPS bond2 after update 1, path 5.

The largest different-theta state distance occurs for structural TFIM after update 2, path 7, comparing the two MPS outputs. Its ratio to the declared `Btheta + 2e-10` threshold is **0.01458193**, the largest ratio in the panel. The largest target-fidelity change is **2.189693e-12** for structural TFIM after update 3, path 5. The numerical evidence therefore supports small physical drift from the slightly different saved parameters; it does not assert that all Gaussian/TFIM structural drift is an exact gauge freedom.

For exceptionally tiny non-gauge parameter changes, the observed distance can exceed the parameter-only bound: the largest ratio is about **102.38**, at original108 TFIM after update 1. That bound is about machine-roundoff size and describes an exact unitary circuit's change with parameters. It excludes numerical propagation error. The predeclared explicit numerical allowance is essential in these cases and has not been fitted after observing results. Raw unaligned state distance reaches `1.850877e-8`, principally reflecting the initial global phases diagnosed from the unchanged Adam recurrence.

Normalized infidelities computed as `1 - normalized_overlap_squared` are at the double-precision cancellation floor, at most `7.771562e-16`. They cannot resolve a physical infidelity of order the squared `1e-10` state distance. Phase-aligned vector distance is the informative measure here.

## Tensor and resource audit

Every one of **23,280** recorded splits discards exactly zero squared singular-value weight; there are no hard-cap reductions. Maximum internal bond is **8**, including target/forward/adjoint work. Maximum total tensor storage for one observed MPS state is **2,688 bytes**, excluding simultaneous tapes, Python arrays and process memory. Per-path output bond lists, raw norms, split ranges, ordered map hashes, weights and all complex output vectors are preserved in the compressed raw step records. Instrumented MPS work contains **97,468** local operator applications.

The physical follow-up costs **12.871975 CPU seconds** and **13.084335 summed worker-wall seconds**. It executes 48 weighted gradient calls, 384 gradient-path calls, 960 additional forward-path calls and **1,728 directional paths**. These include all four output replays per saved step and the independent pre-update path scores. Target factorization, setup, regeneration, checkpoint writes and compression are inside the outer case attempts; nested timings are not added again. The independent saved-array analysis costs approximately 0.17 CPU seconds and is separate engineering work. The original diagnostic's **11.089131 CPU seconds** remains an additional physical cost, not a reused or independent scientific replication.

This result supports the diagnosis of near-zero-gradient roundoff amplified by Adam's fixed epsilon, and validates the tested q6 identical-input MPS bridge. It does not establish large-system behavior, performance of a noisy optimizer, or scalability of the exact-density structural candidate. Any larger-backend study needs a separately frozen protocol and allocation.
