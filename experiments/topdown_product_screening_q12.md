# Twelve-qubit screened product initialization

Run 27 new noiseless statevector cases: all nine fixture targets and seeds
300/400/500. Before depth one, score 1,000 independently sampled product states
and exact zero using the existing batched overlap selector. Encode the winner
in the existing initial U3 layer; all other initial angles, extension draws,
circuit topology and numerical updates match the historical controls.

At each depth, use 100 online Krotov updates and up to 2,000 safeguarded L-BFGS
updates in one continuous call, restoring the best iterate. Stop at the first
accepted ideal fidelity of 99% or after completing depth 30. No noise, pruning
or depth patience is used. A 14-hour accounted CPU ceiling checked between
depths and 16-hour wall/16-GiB deployment guards are safety limits; exhaustion
is not convergence or evidence that a target is unreachable.

Reuse all 54 authenticated near-zero/single-product controls. Structured-state
controls that attained 99% before depth 16 are valid for the extended cap. Haar
controls reference their original depth-16 paths and completed depth-30
continuations; account both historical segments, excluding unrelated runs.
Preserve every original folder and use its frozen source for authentication.

`experiments.topdown_product_screening_q12` plans by default and requires an
audited `--references` JSON file. `--action run --target-id ... --seed ...`
explicitly trains one new case. Plan commands are dispatched independently
only after source, test, CPU and RAM checks; one numerical thread per worker.
Store each selection and optimizer history once in sealed binary archives.
Sampling CPU is included once in the final preparation cost. Successful
completion requires final checkpoint seals and a matching successful supervisor
exit receipt. Live snapshots cannot resume continuous optimizer curvature;
interrupted calls require an explicit restart in a new directory.

Compare fidelity, first-success depth, total preparation CPU and wall time
pairwise by target and seed. Reaching a better starting overlap guarantees
neither faster optimization nor higher final fidelity. These runs test ideal
preparation only and do not establish noisy-fidelity improvements.
