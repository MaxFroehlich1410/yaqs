# Haar initialization matrix: continuation to depth 30

Continue all **18** completed twelve-qubit Haar cases: three fixed targets,
near-zero and single-random-product initialization, seeds 300/400/500. Each
case starts from its own sealed depth-16 best parameters in the initialization
matrix. Use fresh output directories and the parent's original frozen numerical
source, adding only the continuation wrapper and its tests. Keep the historical
results immutable.

For each new depth $d=17,\ldots,30$, append the original small-angle draw with
`RandomState(seed + 1000 + d)` and scale 0.001. Reuse the unchanged `train_depth`
call: 100 online Krotov sweeps and up to 2,000 safeguarded L-BFGS updates, rate
0.7, with curvature reset once per depth and retained throughout each batch call.
Return the best iterate at each depth and retain the best overall circuit for
selection. Continue the same exact-statevector, noiseless, unpruned experiment;
the new product-state screening option is a separate experiment.

Stop at the original **99% ideal fidelity** target, completed depth 30, or nine
additional CPU hours, checked between completed depths. Use one numerical thread,
nice priority 10, 16-GiB memory and twelve-hour wall guards per worker. Resource
exhaustion is not convergence or threshold attainment. Historical and new costs
are reported separately; setup includes the new parent authentication.

Authentication checks the declared parent command, successful supervisor exit,
fixture/source hashes, matrix completion hashes, final and selected-depth seals,
the depth-16 input inherited from depth 15, and independent exact replay. Planning
authenticates every parent before emitting 18 explicit commands, without launching
them. Deployment and scheduling remain separate operations.

```sh
python -m experiments.topdown_haar_matrix_continuation \
  --parent output/q12-initialization-matrix-20261005 \
  --output output/q12-haar-depth30-20261005

python -m experiments.topdown_haar_matrix_continuation --action run \
  --parent output/q12-initialization-matrix-20261005/jobs/haar_random_1__near_zero__seed300 \
  --output output/q12-haar-depth30-20261005/jobs/haar_random_1__near_zero__seed300
```

The example requires the original numerical snapshot; current working source can
differ from that snapshot. Completed stages can be authenticated and reused.
An interrupted continuous optimizer call cannot resume from a live angle snapshot,
because it lacks L-BFGS history; use an explicitly new branch directory for any
restart. Atomic launch claims prevent duplicate submissions.
