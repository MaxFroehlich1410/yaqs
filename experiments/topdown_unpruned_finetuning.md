# Matched unpruned noisy fine-tuning

The current nine-target table comes from the completed six-qubit screened-product
campaign's **near-zero control**, averaged over seeds 300/400/500 under IBM logical
noise. This ablation reuses each of those 27 runs' exact unpruned construction
checkpoint at its already selected depth. It omits pruning B and changes nothing
in noisy training: 100 online plus 100 batch cross-trajectory updates at rate 0.2,
100 frozen histories, empirical no-jump compression, original seed plus 5000,
and final-iterate retention. Restoring gates also restores their noise windows.
The same seed formula does not imply identical histories for different circuits.

Selection is frozen before this experiment. This measures pruning's effect at
the incumbent's selected depth; it does not establish the optimal unpruned depth
or compare independently selected full searches. No construction is repeated.

Prepare into a fresh directory:

```console
uv run python -m experiments.topdown_unpruned_finetuning --action prepare \
  --parent output/q6-product-screening-20261005/retrieved/jobs \
  --output output/q6-unpruned-finetuning-20261006
```

Preparation copies sealed construction, pruning, incumbent refinement and final
stages, locks their file hashes and authenticates the fixture, identity, settings,
selection and ideal replay. Use the parent campaign's frozen numerical source
with the new wrapper for remote execution. Record a deployment manifest and
verify it before launching. Original sources and outputs remain available.

Execute one case from the prepared inputs:

```console
uv run python -m experiments.topdown_unpruned_finetuning --action run \
  --input output/q6-unpruned-finetuning-20261006/inputs/haar_random_2__seed300 \
  --output output/q6-unpruned-finetuning-20261006/jobs/haar_random_2__seed300
```

All 27 cases are independent and can run concurrently, each with one numerical
thread. Use fresh atomic launch claims, capacity checks, nice priority 10 and
bounded supervisors. Never relaunch an existing claim without checking its
process receipt. Checkpoint recovery resumes assessment after a committed
refinement and refuses changed input or saved arrays.

Final assessment independently propagates exact Qiskit density matrices through
the unchanged sampler-law channel. It reports unpruned noisy fidelity before
and after fine-tuning, paired differences from the incumbent, logical gate and
two-qubit counts, and separate new fine-tuning/assessment and historical costs.
Archived pruned before/after endpoints and unpruned ideal fidelity are replayed
independently. A zero-jump training bank remains a reported coverage failure;
it is never replaced or resampled and is not counted as a completed endpoint.
Report all seeds and all targets, plus per-target means and sample standard
deviations. Only complete pairs enter fidelity summaries.
