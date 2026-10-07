# Full six-qubit logical-noise comparison

This extends the completed [logical pilot](LOGICAL_NOISE.md) to all nine fixed
six-qubit benchmark targets. The decision to use logical noise follows the
pilot; it is an explicit model choice, not evidence of performance on a common
native gate set. The separate native-noise results remain part of the study.

Use the pinned noisy-competitor environment and one numerical thread per worker:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-noisy-competitors/bin/python -m experiments.state_preparation_competitors.run_logical_suite \
  --output output/logical-noise-all-q6-ibm-20260920
```

Only target coverage changes. Keep the synthetic IBM coefficients, optimizer
seed 300, fixed candidate grids, optimizer budgets and independent validation
rules from the pilot. Noise is applied after each original elementary gate,
including zero-angle rotations. Its arity-dependent Pauli law is unchanged:
per acted site, X/Y/Z rates are each 0.0001; two-qubit windows also receive
correlated XX/ZZ rates of 0.0015 each. A window has at most one jump with
no-jump probability `exp(-sum(rates))`. There is no idle or readout noise.
Logical RXX/RYY/RZZ gates and logical CX gates each receive a two-qubit window;
this does not make their physical implementation costs equivalent.

The targets are Gaussian, ferromagnetic/critical/paramagnetic TFIM, Haar 1–3,
and random MPS with bond dimensions two and three. Here “random MPS” describes
target generation: these experiments use dense state vectors and matrices.

| Method                                    | Unchanged candidate and training policy                                                                                                                                                                                                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Our B + online Krotov/L-BFGS construction | Reuse the archived logical-noise selected B endpoint. Construction retained its best noiseless iterate. Noisy refinement retained its historical final iterate: 100 online then 100 batch cross updates, rate 0.2, M=100 with empirical no-jump compression. L-BFGS applies to noiseless construction. |
| NACL                                      | Structural checkpoints 0, 10, 50; one restart; 100 inner OrthoMADS objective evaluations; 300-gate cap. Original RZ/SX/CX schedule already matches this noise placement.                                                                                                                               |
| RobustState                               | Depths 2 and 4; 500 ideal and 50 noisy Adam updates, rate 0.005, exact-density feedback adaptation; original logical schedule.                                                                                                                                                                         |
| Fontana                                   | Depths 2 and 4; 200 ideal and 200 noisy BFGS iterations; original RY/CX schedule. Real targets only.                                                                                                                                                                                                   |
| Fixed VQSP control                        | Depths 2 and 4; independent initializations; 200 ideal and 200 noisy L-BFGS-B iterations; M=100 compressed weighted histories.                                                                                                                                                                         |
| Unpruned layerwise control                | Depths 1–4; 200 ideal and 200 noisy updates per depth, 100 online in each phase; M=100 compressed weighted histories.                                                                                                                                                                                  |

Fontana supports both random-MPS fixtures, which are real, but excludes all
three complex Haar fixtures. AQCE and BAA-LRSP remain excluded because they do
not optimize noisy fidelity. IQSP remains excluded while its implementation
issue is unresolved. Fixed VQSP and layerwise are internal controls, not
operator-pool ADAPT-VQE reconstructions.

## Reuse and missing work

The completed Gaussian/TFIM-critical/Haar-1 logical pilot owns 17 supported
method-target groups and 40 candidates. Reuse their exact circuits, parameters,
validation selections, final scores and historical costs unchanged. Preserve
source references to their compact binary histories; do not rerun validation,
reselect with exact scores, or count historical fresh-training time as new
work in this extension.

Our six additional endpoints already exist in the authenticated B/L-BFGS
archive. Reuse their trained circuits and original logical depth selections;
only assess them within the common logical comparison. The remaining six
targets require 28 fresh method-target groups and 74 trained candidates.
Reduced smoke results have incompatible training budgets and are not reused.
The full comparison therefore has **51 applicable groups, 120 candidates and
three explicit Fontana/Haar exclusions**.

All new selected circuits receive the unchanged independent 8,192-trajectory
logical validation procedure. Our baseline's archived depth winner is fixed;
its new validation is diagnostic. Other methods select among their declared
grids by validation fidelity, with the existing compiled-CX/depth/ID tie rule.
Compilation is used for resource diagnostics and tie-breaking only. Store the
selection lock before exact Qiskit density-matrix evaluation of the same
logical channel. Exact test scores include all jump outcomes and must not be
used to replace a validation-selected circuit. Do not condition sampling on
jump occurrence: missing coverage is a recorded failure rather than a reason
to resample.

## Reporting boundaries

Count all visited training candidates and all discarded proposals in runtime.
Charge historical full-grid construction/pruning/refinement costs when reused;
keep new wall time, new CPU work, validation and final assessment separate.
Parallel target workers retain independent process RNG and single-threaded
numerical libraries; parent-process CPU is not total worker CPU.

Report exact fidelity, logical one-/two-qubit gates and two-qubit depth, plus
separately compiled CX resources. Compare fidelity/runtime Pareto fronts per
target. State coverage explicitly when summarizing Fontana's six real targets;
do not compare its six-target mean with a nine-target mean as if they used the
same sample. The two controls should be distinguished from published-method
adaptations.

These six additional comparisons are not untouched algorithm-development
holdouts: our method was previously studied on the full target fixture. One
optimizer seed and limited method budgets do not establish a general SOTA
ranking. Keep the known native-noise limitations visible, and do not tune
individual methods or budgets after inspecting these results.
