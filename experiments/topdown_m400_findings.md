# M=400 rerun: essentially unchanged fidelity at higher cost

**Keep M=100 as the current default.** Increasing the frozen training bank to
400 histories produced negligible mean fidelity changes in this experiment.
The optional depth search changed from **88.64782% to 88.64626%** on average
(-0.00156 percentage points). The single-circuit pipeline changed from
**85.68235% to 85.70322%** (+0.02086 points). These observations do not establish
that the two budgets are statistically equivalent on new targets or seeds.

## Controlled comparison

All nine six-qubit fixture targets, optimizer seeds 300/400, pruning B/D and
the two existing logical noise profiles were rerun: 72 depth searches and
72 single-circuit controls. **Only M changed.** Construction, pruning, update
counts/rates, final-iterate retention, noise law, validation seeds and sample
count, stopping, selection and compilation remained fixed.

Each noisy candidate sampled 400 histories once, then used the same compressed
bank throughout 100 online plus 100 batch updates. The original 100 histories
are a prefix of the new bank at every shared circuit. Training uses dense
64-amplitude statevectors; final scoring uses exact 64-by-64 density matrices
after selection is locked. Random-MPS targets are also simulated densely.

The [frozen M=400 protocol](topdown_m400_protocol.md) gives the execution command.
The [full comparison](../output/topdown-depth-search-q6-m400-20260915/M_COMPARISON.md)
contains every target/seed endpoint, per-target means, resource counts and audits.
The reference is `output/topdown-depth-search-q6-20260914-v2`.

## Optional depth search

Each row averages 18 paired conditions. Fidelity differences are percentage
points. CPU ratios include all construction, pruning/repair, fine-tuning and
validation, including discarded/failed candidates; exact assessment and
compilation are excluded.

| Profile | Pruning | M=100 fidelity | M=400 fidelity | Change   | CPU ratio |
| ------- | ------- | -------------- | -------------- | -------- | --------- |
| IBM     | B       | 90.21351%      | 90.23897%      | +0.02546 | 1.71x     |
| IBM     | D       | 92.04025%      | 92.04989%      | +0.00964 | 1.11x     |
| Google  | B       | 85.14343%      | 85.10437%      | -0.03906 | 1.99x     |
| Google  | D       | 87.19408%      | 87.19179%      | -0.00229 | 1.26x     |

There are 50 improvements and 22 decreases; most changes are tiny. Selected
depth is unchanged in 70 of 72 searches. The two changes both lower final
fidelity: Gaussian/seed400/D/Google moves from depth 1 to 2 (-0.21444 points),
and TFIM-ferromagnetic/seed300/B/Google moves from depth 4 to 6 (-1.17506 points).

The latter is a fine-tuning sensitivity, not a pruning change: at the identical
depth-4 starting circuit, fidelity falls from 84.35402% to 80.82746% with M=400.
The unchanged depth search then selects depth 6, recovering to 83.17897%.
Larger sample banks do not guarantee a better finite-budget optimization result.

## Default single-circuit pipeline

| Profile | Pruning | M=100 fidelity | M=400 fidelity | Change   | CPU ratio |
| ------- | ------- | -------------- | -------------- | -------- | --------- |
| IBM     | B       | 86.65641%      | 86.66136%      | +0.00494 | 1.09x     |
| IBM     | D       | 89.16687%      | 89.18964%      | +0.02278 | 1.11x     |
| Google  | B       | 82.74510%      | 82.78118%      | +0.03608 | 1.29x     |
| Google  | D       | 84.16103%      | 84.18068%      | +0.01965 | 1.22x     |

There are 51 improvements and 21 decreases. Starting circuits and selected
depths are identical in this comparison. Mean gains of 0.005–0.036 points are
small relative to the additional work. The experiment provides no compelling
fidelity reason to replace M=100 with M=400 in the current method.

## Compute and sampling

Across 411 successful depth candidates common to both runs, noisy fine-tuning
alone costs **2.50–2.77x** as much CPU, depending on pruning/noise profile.
Mean same-depth fidelity changes range from -0.015 to +0.039 points across
the four groups. These candidates are correlated within searches, not
independent statistical replicates.

The M=400 depth searches attempted 417 candidates, the same total as M=100,
although three branches changed their stopping depth. There were **415
successful training banks and two zero-jump coverage failures**. Successful
banks sampled 166,000 histories but propagated only **20,522 representatives**
(49.45 per bank on average). Failed banks consumed another 800 sampled histories.
No jumps were forced and no banks were resampled. Two of the four original
coverage failures resolved; TFIM-ferromagnetic/seed300/depth1/D still fails
under both noise profiles. No circuit was pruned to zero gates.

All 415 successful training banks and all 415 validation banks contained jumps.
Validation used **13,598,720 independent histories** in total. Exact final
density scoring integrates every modeled outcome and does not reuse a finite
training or validation bank to estimate final fidelity.

The full rerun took **30.86 minutes** with four numerical workers, including
controls, compilation and retrospective exact scoring; audit/report generation
is additional. Across all searches, the summed standalone CPU ratio is 1.40x;
for controls it is 1.18x. Ratios use archived M=100 timings on the same machine;
runtime conditions and timings of unchanged phases can differ. They are
descriptive measurements, not a controlled hardware performance benchmark.

## Verification and limits

The comparison authenticated both runs and confirmed that the only changed
scientific setting was M. It checked 164 identical construction checkpoints,
250 identical pruned checkpoints, 483 identical 100-history prefixes and
411 identical validation banks at shared candidates, including all 72 controls
where relevant. All 417 stopping decisions and all validation banks were replayed.
Maximum validation replay error was zero; independent Qiskit trajectory replay
agreed within 3.34e-16 and vector/density ideal fidelity within 1.34e-15.

All **41 targeted tests** and focused Ruff/type checks pass. Full repository
lint still reports 2,468 existing diagnostics; unrelated hook edits were restored.
The output directory contains the manifests, source snapshots, checkpoints,
raw validation banks, reports, machine-readable comparisons and verification logs.

Only two optimizer seeds and previously used targets were tested. Exact final
evaluation removes Monte Carlo test error, not training variability. The
cross-trajectory signal's gradient interpretation and the noiseless optimizer's
loss of good intermediate iterates remain unresolved and unchanged. More
accurate sampling alone does not resolve those optimization questions.
