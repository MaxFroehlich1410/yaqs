# Frozen q6 construction-optimizer stability comparison

This experiment changes only **noiseless construction optimization** within the
existing optional depth-search pipeline. It compares three construction policies
with both pruning B and D, the nine six-qubit fixture targets, optimizer seeds
300/400, and both historical logical noise profiles. The six optimizer/pruning
combinations therefore produce 216 validation-selected searches. All simulations
use dense state vectors; final noisy fidelity uses exact density matrices.

## Construction policies

| Policy              | Updates at each depth                                                    | Returned parameters                                    |
| ------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| `original_final`    | 100 online and 100 batch updates, rate 0.7                               | Final iterate                                          |
| `best_only`         | Identical updates and rate                                               | Best actual parameter vector, including iteration zero |
| `best_backtracking` | Identical 100 online updates; 100 batch updates with Armijo backtracking | Best actual parameter vector, including iteration zero |

For the batch loss $L=1-F_0$ and its independent noiseless gradient $g$, try
$\eta_j=0.7\,2^{-j}$, $j=0,\ldots,12$, accepting the first finite candidate for which

$$
L(\theta-\eta_jg)\leq L(\theta)-10^{-4}\eta_j\|g\|^2.
$$

Compute the gradient once per batch update and reuse it across trials. If no
trial is accepted, leave the parameters unchanged and record an exhausted line
search. Do not label that event convergence. The 200-update budget and disabled
90% early exit remain fixed; there is no additional stopping rule. Each batch
update starts its trials at 0.7. The online trajectory and the online-to-batch
transition stay unchanged; best retention affects only the returned checkpoint,
not the intervening trajectory. Exact ties retain the earliest best iterate.

## Stage 1: identical-input diagnostics

Authenticate and replay all 124 archived construction checkpoints in
`output/topdown-depth-search-q6-20260914-v2`. For each depth, reconstruct its
original input from the archived preceding unpruned final parameters and the
original deterministic layer-extension seed. Each of the three policies receives
exactly that same input; do not propagate a new policy's preceding result into
this stage. There are 372 fixed-depth optimizations.

Require the original policy's parameters and full fidelity trace to reproduce
the archive within numerical tolerance. Require best-only to have the same raw
optimization trace. Independently verify returned-state fidelity with Qiskit.
Check noiseless directional derivatives around the online-to-batch transition,
including the TFIM ferromagnetic seed-400 depth-3 and Haar-3 seed-400 depth-10
failures. Record initial, online-final, best, final and returned fidelities;
actual checkpoint parameters; accepted step sizes; rejected trials; exhausted
line searches; gradient/objective calls; and CPU and wall time.

## Stage 2: downstream B/D comparison

Run the same optional depth-search protocol separately for each construction
policy. Each policy now grows from its own returned unpruned checkpoint. The
following settings remain those of the
[existing depth-search protocol](topdown_depth_search_protocol.md):

- Original brickwall circuit, parameter initialization and ordering, depth cap 16.
- B/D pruning with exact deletion threshold 0.01. **D's five online plus five
  batch repair updates at rate 0.7 remain unchanged and retain their final iterate.**
- Noisy fine-tuning with 100 online plus 100 batch cross-trajectory updates at
  rate 0.2, retaining the final iterate, and 100 unconditional frozen histories.
- Exact empirical compression of duplicate no-jump histories; zero-jump banks
  fail without resampling or forced events.
- Separate 32,768-history validation banks with the original deterministic seed
  formula, three-step patience and the original material-improvement rule.
- Maximum validation fidelity selects the circuit; earliest depth wins exact
  ties. Lock all branch decisions before exact noisy density evaluation.
- Original logical noise coefficients and reference CX compilation. These
  coefficients are not calibrated IBM/Google hardware predictions; noise is
  applied to logical gates, with compiled counts reported separately.

Use the same RNG recipes across optimizer policies. Because pruning can change
the circuit, a shared seed does not imply gate-aligned jump events across
different circuits. Training and validation remain separate. Exact assessment
integrates every modeled noise outcome, including jumps, without sampling or
reusing the training objective.

Replay the original-policy search against the previous M=100 archive: verify
construction/pruning/fine-tuning parameters, training-bank identities, validation
banks and stopping/selection decisions. The previous first-90% control is not
rerun: the reference for this experiment is the original final-iterate **depth
search**. This experiment does not independently establish changes to the
cheaper first-90% pipeline or to D's repair optimizer.

## Reporting and reproducibility

Freeze source, fixture, reference-artifact hashes and software versions before
running. Check cached artifacts on resume. Report stage-1 optimizer CPU separately
from stage-2 search cost; stage-1 wall time also includes independent fidelity
and derivative checks. Each standalone search is charged its full
construction, pruning, noisy fine-tuning and validation cost, including discarded
and failed candidates; shared computation does not reduce the reported standalone
cost. Exact assessment and compilation are reported separately. Count every
backtracking trial in objective-evaluation totals.

Report every target/seed outcome, exact noisy fidelity, logical and compiled
two-qubit counts and compiled two-qubit depth, selected construction depth,
failures, CPU cost, and paired differences against both original and best-only.
Distinguish stabilization of noiseless optimization from improvement in the final
noisy circuit. Noisy fine-tuning and D repair can still be unstable in this
controlled comparison. Two optimizer seeds do not support broad significance
claims or guarantee generalization to other targets, sizes or noise models.
The 124 depths and 216 searches are correlated experimental conditions, not
independent statistical replicates. These previously examined fixture targets
are a method-development benchmark, not an unseen-target confirmation set.

Run with:

```sh
uv run python -m experiments.topdown_optimizer_stability --output NEW_DIRECTORY --workers 4
uv run --with matplotlib python -m experiments.topdown_optimizer_stability_analysis --output NEW_DIRECTORY
```
