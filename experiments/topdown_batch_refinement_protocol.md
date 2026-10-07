# Frozen q6 batch-refinement comparison

This study compares BB and L-BFGS batch refinement with the **current best
working construction policies**. It changes only the noiseless batch phase of
the optional every-depth pipeline. All four arms retain the best actual
construction parameters, including iteration zero. The original unprotected
final-iterate optimizer is not the baseline.

## Baselines and scope

The [completed stability study](topdown_optimizer_stability_findings.md) motivates
the predeclared primary comparison:

| Pruning | Current practical baseline | Reason                                                                            |
| ------- | -------------------------- | --------------------------------------------------------------------------------- |
| B       | `best_backtracking`        | Highest mean noisy fidelity, fewer two-qubit gates and comparable search CPU      |
| D       | `best_only`                | Better practical cost/resource tradeoff; backtracking's fidelity effect was mixed |

Both controls run with both B and D, so every new policy is also compared with
the other existing strong control. This preserves the slightly higher IBM D
mean from `best_backtracking` as an explicit comparison, rather than hiding it.
No target-specific or noise-profile-specific baseline is selected after seeing
the new results.

The nine six-qubit fixture targets, optimizer seeds 300/400 and both historical
IBM/Google logical noise profiles are unchanged. Four construction policies
times these conditions give **288 standalone searches**. Training uses dense
statevectors; independent final assessment uses exact density matrices.

## Four construction policies

All arms use the same 100 online sweeps at rate 0.7 and up to 100 attempted
batch updates. There is no 90% early exit or new gradient-norm stopping rule.
Best retention affects the returned checkpoint, not the active trajectory.
The batch phase starts from the online endpoint. Curvature history starts empty
at that transition and resets at every new circuit depth. Angles remain
unwrapped. No online signal is substituted for a true endpoint gradient.

Let $L=1-F_0$ and $g_k=\nabla L(\theta_k)$. Every safeguarded arm accepts a finite
trial $\theta_k+\alpha p_k$ only if

$$
L(\theta_k+\alpha p_k)\leq L(\theta_k)+10^{-4}\alpha g_k^Tp_k.
$$

Use at most 13 trials per attempted batch update, halving the trial step after
each rejection. Reuse the current gradient across trials. If all trials fail,
retain the previous state, count the work and record failure. A failed search
is not convergence. The fixed-step control keeps its existing unsafeguarded
trajectory while returning its best checkpoint.

| Policy              | Batch direction and initial trial step                     |
| ------------------- | ---------------------------------------------------------- |
| `best_only`         | $p_k=-g_k$, fixed $\alpha=0.7$, unchanged accepted updates |
| `best_backtracking` | $p_k=-g_k$, restart every trial search at $\alpha=0.7$     |
| `bb_backtracking`   | $p_k=-g_k$, BB1 curvature-based initial step, then Armijo  |
| `lbfgs`             | Limited-memory inverse-Hessian direction, then Armijo      |

For BB1, use consecutive accepted batch states and their true gradients:

$$
s=\theta_k-\theta_{k-1},\qquad y=g_k-g_{k-1},\qquad
\alpha_{\mathrm{BB1}}=\frac{s^Ts}{s^Ty}.
$$

Require finite pairs and $s^Ty>10^{-10}\lVert s\rVert\lVert y\rVert$ with
nonzero norms. Clip initial proposals and fallbacks to $[10^{-6},10]$. Start at 0.7 without
history; when curvature is unsuitable, fall back to the previous accepted
step, or 0.7 if none exists. Record invalid pairs, clipping and rejected
trials. Backtracking may reduce an initial proposal below the proposal bound.

L-BFGS uses the existing tested two-loop recursion with at most ten positive
curvature pairs, the same curvature threshold, and its inverse-Hessian scalar
initialization clipped to $[10^{-8},10^8]$. Skip unsuitable curvature pairs.
Reset history and use $-g_k$ if the proposed direction is nonfinite or is not a
descent direction. Start line searches at 0.7 with empty/reset history and at
1 otherwise. This is **L-BFGS with Armijo**, not a claim to reproduce SciPy
L-BFGS-B or a strong-Wolfe implementation. It changes direction as well as step
selection. Cache a gradient only for the exact parameter vector it belongs to.

The BB bounds, history length, acceptance constant, trial limits and update
budgets are fixed before the target runs; no tuning grid follows inspection of
their results.

## Stage 1: identical-input construction screen

Use the 124 **pipeline construction inputs** saved under `best_backtracking` in
`output/topdown-optimizer-stability-q6-20260919`. These are not the older study's
fixed-stage inputs, which coincidentally also number 124. Each of four policies
receives the same archived initial vector at each depth: **496 optimizations**.
Their later replay inputs do not depend on newly optimized parents.

Require all four online prefixes to match. Require the backtracking control's
full trace and returned parameters to reproduce the reference, and independently
verify each returned fidelity with primitive Qiskit statevector propagation.
Report best/final/returned fidelity, full update traces, accepted step sizes,
objective and gradient calls, rejected trials, failed searches, curvature
fallbacks/skips, direction resets, and measured CPU.

Primary endpoints use the same attempted-update budget. This is **not a
compute-matched comparison**. Also report fixed-input best fidelity versus
measured cumulative batch CPU, with each alternative restricted to the current
backtracking control's batch CPU budget. Use only completed iterates within the
budget; do not interpolate an unobserved fidelity. Record whether the alternative
used its entire update budget before that time. CPU curves are descriptive
single-run measurements, not exact hardware-independent work counts. Objective
and gradient calls remain separate because their costs differ.

## Stage 2: full B/D pipeline

Each construction policy grows from its own retained, unpruned checkpoint.
Reuse the existing circuit extension and seed recipes. Preserve B/D pruning
with cutoff 0.01, including **D's unchanged five-online/five-batch fixed-rate
final-iterate repair**. Noisy fine-tuning remains 100 online plus 100 batch
cross-trajectory updates at rate 0.2, final-iterate retention and **100 frozen
unconditional histories with empirical no-jump compression**.

Preserve separate 32,768-history validation banks, coverage-failure handling
without resampling, the original three-step patience rule and depth cap 16.
Lock validation-based circuit selection before exact density evaluation.
The same RNG recipe does not imply gate-aligned noise across different pruned
circuits. Final exact assessment integrates all modeled noise outcomes,
including jumps, without reusing training samples as a test set.

Freshly rerun both controls and require construction, pruning, noisy parameters,
training/validation banks, stopping and winner parity against their respective
reference-policy archives. Charge each standalone search its full construction,
pruning, noisy training and validation CPU, including discarded and failed
candidates. Shared computation is charged in full to each standalone branch;
parallel suite wall time is reported separately. Exact assessment, compilation
and general checkpoint bookkeeping are separate from search CPU.

## Reporting and reproducibility

Freeze protocol, numerical source, fixture, reference artifact hashes and
software versions before execution. Authenticate resume artifacts and preserve
the old experiment sources. A new analysis module may be developed separately,
but its source is saved with the exported report.

Report paired gains against each pruning method's current baseline and both
existing controls, wins/losses/ties, per-target and per-seed fidelities, final
logical/compiled two-qubit count and compiled depth, selected depth, coverage
failures, all phase costs and optimization failures. Distinguish improved
noiseless construction from improved final noisy circuits. D repair and noisy
fine-tuning remain potential downstream instabilities, so their behavior must
not be attributed solely to batch construction quality.

The historical profile labels are logical simulation coefficients, not
calibrated hardware predictions. Nine development targets and two optimizer
seeds do not establish generalization or statistical significance; depth
checkpoints and profile results are correlated. Recommendations should weigh
fidelity, regressions, resources and compute rather than promote a policy from
its ideal fidelity alone. This study does not rerun the cheaper first-90%
pipeline or test adaptive online updates.

References: [BB step sizes](https://optimization-online.org/wp-content/uploads/2018/05/6628.pdf),
[L-BFGS](https://users.iems.northwestern.edu/~nocedal/lbfgs.html).

```sh
uv run python -m experiments.topdown_batch_refinement --output NEW_DIRECTORY --workers 4
uv run --with matplotlib python -m experiments.topdown_batch_refinement_analysis --output NEW_DIRECTORY
```
