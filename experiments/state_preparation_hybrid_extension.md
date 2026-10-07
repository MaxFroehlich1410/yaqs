# Hybrid construction extension: longer, larger online prefixes

Version `yaqs.compression_hybrid_construction_extension.v1`, specified
2026-09-06 before launch. This is an exploratory follow-up to the
[completed hybrid screen](state_preparation_hybrid_findings.md).

The author's final clarification requests **only four new settings**:
25 or 30 online updates crossed with rates 0.7 or 0.9 (not 0.92).
Do not rerun the old settings or the separate warm-refinement study.

## Fixed experiment

Use precisely the preceding [construction pipeline](state_preparation_hybrid_pilot.md):
four fixed q6 targets (Gaussian, critical TFIM, Haar random 1, random MPS bond 2),
initialization seeds 11 and 29, root seed 2026090520, and identical master
parameter vectors. Each target/seed/setting is an independent job: **32 jobs**.

At each of the four BMPD growth stages (depths 1 through 4), restart the
online prefix. Keep exactly 100 updates per stage:

| Online prefix | Online rate | Batch tail at rate 0.2 |
| ------------- | ----------- | ---------------------- |
| 25            | 0.7         | 75                     |
| 25            | 0.9         | 75                     |
| 30            | 0.7         | 70                     |
| 30            | 0.9         | 70                     |

Keep the protected initial U3 layer, parameter-growth initialization, and
six magnitude-pruning rounds unchanged. Each round deletes ten one-qubit
and five two-qubit gates, including their noise windows, then performs
20 batch-only repair updates at rate 0.1. Finish with 50 batch-only updates
at rate 0.1. Every run therefore has **570 total updates** and ends with
78 one-qubit and 30 two-qubit gates. The 25-prefix runs have 100 online
updates overall; the 30-prefix runs have 120. Longer prefixes replace batch
updates rather than adding work to the update budget.

Training is noiseless, untruncated complex128 MPS hybrid Krotov. Online means
the existing gate-sequential update using old costates and an updated prefix;
there is no change to that algorithm. This tests improved circuit construction,
**not noise-aware trajectory fine-tuning**. Evaluate the fixed endpoints after
520 and 570 updates using the same exact local Pauli-Lindblad channel:
whole-window average gate-error targets r1 = 0.00064 and r2 = 0.0051, with
per-site/per-Pauli integrated coefficients 0.00032020497493061084 and
0.0010664702651014273. No noise strength or pruning budget is retuned.

## Comparisons and safeguards

The primary endpoint is exact noisy fidelity at update 570. Compare every
new setting against the **already completed** H20/rate-0.5 endpoint on each
of the same eight target/seed pairs; also show the original batch control.
Their target-balanced means were 74.92614319% and 71.50271265%, respectively.
Keep reference checksums and paths with the findings; references are not new
jobs. Verify common targets, source records, seeds, noise, environment, and
construction budgets before combining runs.

Report all four means, per-target means, paired differences, win counts, and
any deterioration or unstable training. Use final iterates only: no best-epoch
selection, early stopping, silent failed-job exclusion, or further adaptive
grid expansion. Inspect growth-stage traces to distinguish faster early
progress from final circuit improvement. Preserve both noiseless and noisy
endpoint fidelities and each pruning path.

This remains a development-data screen, not independent confirmation. There
is only one state per family and two optimizer seeds. A matched batch-only
high-rate-prefix control is still needed to separate update ordering from
learning-rate effects. These comparisons alone cannot establish a pruning-
specific gain, a noise-aware-training advantage, or state-of-the-art performance.

## Execution and verification

Reuse the public hybrid optimizer, atomic block checkpoints, checksummed
manifests, and bounded process executor. Eight single-threaded workers use
all eight laptop cores. Chunked optimization must match uninterrupted training
at the new phase boundaries and rates; construction-only endpoints must match
the corresponding endpoints of the full runner in engineering tests. Resume
must reject changing between full and construction-only modes.

```bash
caffeinate -i uv run python -m experiments.state_preparation_hybrid_pilot \
  --construction-extension \
  --source-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-compression-pilot-20260905 \
  --output-root /Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-construction-extension-20260906 \
  --workers auto
```

The original pilot and completed hybrid screen are read-only references.
No existing scientific outputs are overwritten.

Completed results and the post-run audit are in the
[extension findings](state_preparation_hybrid_extension_findings.md).
