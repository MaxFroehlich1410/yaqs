# Matched twelve-qubit sequential pruning

This independent 18-case campaign reuses the six structured targets and seeds
300/400/500 from the October 7 simultaneous-B campaign. Haar targets are excluded.
All runs start from the same authenticated screened-product ideal endpoints;
construction and depth selection are not repeated.

Sequential C deletes the gate with minimum signed exact ideal deletion damage
strictly below 0.01 and recomputes all deletion damages after each deletion.
Original gate order breaks ties. It stops when no candidate qualifies or the
circuit is empty. There are no noiseless repairs and no cumulative fidelity
guard. The complete scores, deletion order and surviving parameter coordinates
are sealed. This is the existing archived C rule, using statevectors rather
than density matrices.

The original q12 continuation workflow then performs 100 online and 100 batch
noisy cross-trajectory updates, rate 0.2, with the same unconditional weighted
100-history IBM logical-noise law and optimizer seed + 5000. Online and batch
share one bank. No-jump banks fail without resampling. The final iterate is
assessed in four independent batches of 8,192 histories. Assessment never
selects training iterates. A distinct protocol/source identity prevents reuse
of simultaneous pruning or refinement stages.

Compare C and B by target and optimizer seed: ideal fidelity immediately after
pruning and after refinement, independently assessed noisy fidelity before and
after refinement, logical and compiled resources, pruning CPU and refinement
CPU. Historical ideal preparation is shared; evaluation costs are separate.
The same evaluation entropy rule and noise law apply, but different retained
circuits imply different noise banks across methods. Only before/after endpoints
of one retained circuit share an identical assessment bank. Report seed
variability separately from trajectory sampling uncertainty, and interpret
fidelity/gate/time tradeoffs rather than selecting a method on fidelity alone.

Prepare with `python -m experiments.topdown_q12_sequential_noisy_continuation
--action prepare --references references.json --output campaign`. Run each
sealed input with `--action run --input campaign/inputs/<job_id> --output
campaign/jobs/<job_id>`. Launch into a fresh root with independent atomic claims,
single-thread workers, measured capacity and process receipts. Account for
remaining guard headroom of the already running Haar retries. Preserve all
original jobs, snapshots, logs and results.
