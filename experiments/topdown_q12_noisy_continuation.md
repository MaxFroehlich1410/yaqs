# Twelve-qubit pruning and noisy continuation

This experiment reuses saved ideal checkpoints for nine targets and seeds
300/400/500. It adds no construction updates and performs no depth selection.
The October 7 panel uses screened-product endpoints for the six structured
targets and near-zero depth-30 endpoints for the three Haar fixtures.

Preparation verifies original final and selected-stage seals, successful
matching supervisor receipts, frozen numerical sources, target hashes and
exact statevector fidelity. It copies the selected construction into fresh
sealed inputs while retaining historical preparation CPU costs separately.

Each independent worker applies simultaneous pruning B with the existing
signed ideal deletion score and cutoff 0.01. It then performs 100 online
updates followed by 100 batch updates at rate 0.2, using the unchanged noisy
cross-trajectory signal. Both phases share the original seed + 5000 fixed
unconditional 100-history synthetic IBM bank. No-jump multiplicity weights
are preserved. A bank with no jumps fails without resampling.

The final iterate is evaluated with four fresh independent banks of 8,192
histories each. Pruned and refined endpoints share each assessment bank for
a paired comparison. Scores never select training iterates. Exact
statevectors are used throughout; no twelve-qubit density matrix is formed.
Report sampling uncertainty and keep logical noise placement distinct from
compiled CX resource counts. Haar parent endpoints around 87.4% reflect
depth-budget exhaustion, not attainment of the 99% target.

Use `python -m experiments.topdown_q12_noisy_continuation --action prepare
--references references.json --output campaign` to prepare the panel.
References specify `target_id`, `seed`, original `parent` directory and `host`.
Launch each sealed input independently using `--action run --input
campaign/inputs/<job_id> --output campaign/jobs/<job_id>`.

The Leonhard dispatch uses one numerical thread, nice priority 10, 16 GiB
per-worker memory and a 72-hour wall guard. Capacity is rechecked before
launch, with 64 GiB RAM and physical CPU headroom reserved. Original campaigns
are read-only. Atomic launch claims prevent duplicate submissions; inspect
existing claims after an ambiguous SSH outcome instead of rerunning dispatch.
Completed phase seals can be reused after explicit recovery. An interrupted
phase is incomplete and must not be reported as a finished result.
