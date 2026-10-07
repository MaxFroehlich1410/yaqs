# Frozen M=400 rerun of the six-qubit depth experiment

The sole scientific intervention is **400 instead of 100 raw training histories**.
The reference is `output/topdown-depth-search-q6-20260914-v2`.
The [original depth protocol](topdown_depth_search_protocol.md) otherwise remains
fixed, including all nine targets, seeds 300/400, B/D pruning, IBM/Google profiles,
200 noiseless and noisy hybrid updates, rates, final-iterate retention, validation
entropy and 32,768 samples, patience, depth cap, compilation and exact final scoring.

Both the depth search and its first-90% single-circuit control use M=400. Retain
seed+5000, so the original 100 training histories are the first 100 of the larger
bank for the same circuit. Append 300 unconditional independent histories and
compress only no-jump duplicates, with weights m0/400 and 1/400. Freeze this bank
for all noisy updates. Keep the existing zero-jump coverage-failure policy.

Recompute construction, pruning and fine-tuning; do not initialize from M=100
fine-tuned parameters. Verify archived construction and pruned starting parameters,
as well as the common training-history prefix. M=400 endpoints need not equal
M=100 endpoints. The default M=100 rerun retains its stricter endpoint-parity check.
The historical protocol's baseline-parity description is interpreted accordingly.

The same stopping rule can visit/select different depths as a consequence of
changed validation scores. Report the complete selected-pipeline comparison and
also compare shared depths to distinguish fine-tuning effects from depth selection.
Exact final density fidelity integrates all modeled outcomes; no training or
validation samples are used to estimate the final mean.

Charge all construction, pruning, noisy training and validation CPU to each
standalone pipeline, including discarded candidates. Compare with archived M=100
timings on the same machine and report timing variability as a limitation. These
are paired results on two optimizer seeds, not a new independent-seed study.

```sh
uv run python -m experiments.topdown_depth_search \
  --output output/topdown-depth-search-q6-m400-20260915 \
  --training-paths 400 --workers 4
uv run --with matplotlib python -m experiments.topdown_depth_analysis \
  --output output/topdown-depth-search-q6-m400-20260915 --workers 4
```

The manifest records M=400 and source/input checksums before execution. This
experiment does not change the recommended default or introduce a growing-M schedule.
