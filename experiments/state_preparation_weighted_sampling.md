# Weighted no-factor trajectory diagnostic

Execution complete: see the [combined findings](state_preparation_sampling_budget_findings.md).

Protocol frozen 2026-09-12 after the optimizer/trajectory-budget screen and before
this diagnostic. Step 1 showed R=128 improved Adam by 0.143217 pp in all eight
pairs, but all-online Krotov by only 0.011157 pp in four of eight pairs. Thus
variance remains relevant, but reducing it alone did not remove the Krotov gap.
This motivates a shared estimator improvement, not a claimed Krotov-specific win.

## Estimator and scope

For the existing fixed, state-/parameter-independent local Pauli factor channel,
let `p_j=(1-exp(-2*a_j))/2` for every enabled gate/site/XYZ factor and let
`p0=prod_j(1-p_j)`. The event stratum is **at least one sampled factor**, not
necessarily a net nonidentity error. Pauli cancellations remain in that stratum.

At each fixed parameter vector estimate the exact gradient of mean infidelity by

```text
g_hat = p0 * g(no factors) + (1-p0) * mean(g(path) | at least one factor).
```

The no-factor gradient is deterministic and computed once. The conditional
gradient is still a trajectory mean. Weights do not depend on parameters for
this model, so this is an unbiased decomposition without a missing probability
derivative. It is not a general amplitude-damping/no-jump gradient formula.

Draw the index j of the first selected factor with probability
`p_j * prod_{i<j}(1-p_i)/(1-p0)`, set all earlier factors to false, select j,
then independently sample later factors with their original probabilities.
This exact conditional sampler uses no rejection loop or single-jump truncation.
The zero-noise kernel returns the noiseless gradient without conditional draws.

This is an adaptation of an established sampling idea, not a novelty claim:
[QuTiP improved sampling](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html#using-the-improved-sampling-algorithm)
also treats the no-jump contribution separately. Here the event law is the
discrete commuting Pauli-factor channel, not continuous jump-time sampling.

## Frozen diagnostic design

- Same four old development targets and saved updates 0,25,50 as the previous
  gradient diagnostic; no new targets or optimizer updates.
- R=8,32,128 **total propagated paths**. Ordinary uses R independent paths;
  weighted uses one deterministic path and R-1 conditional paths.
- Eight ensemble repeats per target/checkpoint/R. Two estimators, 576 estimates
  across 12 atomic checkpoint jobs. No cross estimator or optimizer tuning.
- Separate deterministic diagnostic seed role, shared base seeds between the
  estimators. Counts use trajectory-index-stable nested streams. Conditional
  and ordinary maps are not claimed to be identical.
- Recompute ordinary baselines in the same run and alternate estimator order.
  The ordinary baseline is the unchanged existing sampler. Any CPU advantage
  includes both conditioning/sampling implementation and gradient costs; it is
  not automatically attributable solely to statistical variance reduction.
- Charge weighted sampler setup once **per estimate**, even though physically
  shared within a job; also charge its no-factor gradient every time. No hidden
  reuse of the exact reference or noiseless gradients. Reference scoring is
  separate. Frozen maps and physical channel remain unchanged during evaluation.
- Retain raw gradient vectors, sample losses, error, cosine, true-descent signs,
  sampled-factor counts, empirical mean bias, variance and MSE times CPU.
- Validate exhaustive small-system expectations against the exact density
  channel, conditional pattern frequencies, multiple-event cancellations,
  finite differences, zero noise, malformed inputs and resume before launching.

Any apparent improvement is an estimator diagnostic, not yet evidence of better
noisy-state preparation. A weighted online/batch optimizer integration and a
matched weighted-Adam comparison would be separate subsequent work. No
configuration is frozen for confirmation, and no fresh targets are evaluated.

```sh
uv run --no-sync python -m experiments.state_preparation_weighted_sampling \
  --source-root output/state-preparation-trajectory-comparison-20260910 \
  --output-root output/statevector-weighted-diagnostic-20260912 \
  --workers auto
```

Explicit `--resume` authenticates unchanged source/configuration/runtime.
Scientific jobs use one numerical thread per process and do not overlap tests.
