# Mean-two weighted trajectory controls

Status: kernel and engineering tests only. This protocol authorizes no scientific
run. The coordinator must freeze the complete optimizer/structure driver, source
hashes, benchmark custody hashes, budgets and stopping rules before launch.

## Purpose and scope

Give weighted Adam and adaptive online updates the same training objective as
the existing equal-mean two-channel exact optimizer:

```math
L(\theta)=\tfrac12 L_{\mathrm{depolarizing\_1s\_all}}(\theta)
          +\tfrac12 L_{\mathrm{dephasing\_1s\_all}}(\theta).
```

These are the unchanged calibrated exact benchmark channels. The input is
\(|0\ldots0\rangle\), and all physical one- and two-qubit gates are retained.
Noise acts after each enabled logical gate on its support. Disabled windows stay
disabled. Gate order, parameter sharing, signed scales, angle offsets, fixed
parameters and logical/native identifiers are retained in the circuit identity.
The sampler is independent of theta. Ballarin and parameter-dependent noise are
excluded from training. Final evaluation still uses every unchanged benchmark
condition, including Ballarin's native materialization and pruning semantics.

The primary engineering hypothesis is that the sampled gradient has exactly the
same channel expectation as the exact mean-two gradient. No optimizer advantage
is predicted by this identity. Any later scientific comparison must assess
whether estimator variance and adaptive online's stale adjoints prevent useful
progress at matched measured compute.

## Exact factor law

For a Hermitian Pauli P and integrated coefficient a, the channel is

```math
e^{a(\mathcal P-I)}=(1-p)I+p\mathcal P,
\qquad \mathcal P(\rho)=P\rho P,
\qquad p=(1-e^{-2a})/2.
```

Pauli conjugation superoperators commute, including when the Paulis themselves
anticommute. The exact channel therefore factors into independent Bernoulli
conjugations. Every selected factor is retained in its original order; multiple
selected factors may cancel physically. Such paths belong to the conditional
event stratum and are not merged into the deterministic empty-factor stratum.

The depolarizing component reuses the existing `FactorSampler` and default
`CalibratedLocalDepolarizingProvider`. Each enabled site has ordered X, Y and Z
factors. Setup checks that both provider coefficients agree with
`channels.window("depolarizing_1s_all", arity, "calibrated_exact")` to 2e-14.
Only these coefficients are used from the provider. Its TJM execution method is
not called: execution is the exact independent factor law, not the provider
metadata's legacy at-most-one-jump execution label.

`DephasingFactorSampler` reuses the same conditional draw/map/ensemble methods
with its complete factor-law initialization replaced. It takes each coefficient
from `channels.window("dephasing_1s_all", arity, "calibrated_exact")`. There is one
Z factor per enabled site, ordered by gate and then the original site tuple.
For one-qubit average error r1=0.00064, the independent closed-form check is
p1=3*r1/2. For two-qubit average error r2=0.0051, the two local Z probabilities
are p2=1-sqrt(1-5*r2/4), because the identity probability is (1-p2)^2 and the
complete-channel error is `4/5*[1-(1-p2)^2]`. Depolarizing strengths must never be
substituted into the dephasing law.

For channel c, let p0,c be the product of all empty-factor probabilities. One
deterministic empty-factor path gets weight p0,c. Each of R-1 random paths is
drawn conditional on at least one selected factor and gets weight
(1-p0,c)/(R-1). The first selected factor j has probability

```math
\Pr(j\text{ first}\mid\text{event})=
\frac{p_j\prod_{i<j}(1-p_i)}{1-p_{0,c}}.
```

Factors after j retain their original independent Bernoulli law. This requires
neither rejection sampling nor a truncation of multi-factor events. Each
channel's weights sum to one. Concatenating the two channel ensembles and
halving every weight therefore gives weights summing to one for the stated
mean objective. At fixed theta, the loss, fidelity and gradient estimates are
unbiased. The all-disabled limit keeps the entire requested path allocation,
including zero-weight duplicate paths.

## Frozen sampler opportunity and streams

The intended training allocation is **32 total paths per channel**: one
deterministic plus 31 conditional paths for each channel, 64 paths overall.
Weighted Adam and adaptive online receive this same sampler opportunity. There
is no channel-allocation tuning, variance-adaptive allocation, control variate,
fixed-ensemble reuse, importance tilt or cross-trajectory estimator in this
protocol. An alternative of 16 paths per channel is reserved only for a later
explicitly preregistered compute screen and is not activated by this kernel.

The deterministic order is all depolarizing paths, then all dephasing paths,
with each channel's empty-factor path first. Both empty paths are propagated
and charged separately. A root seed is paired across optimizer arms. A fresh
update index is used on each update and persisted through exact resume.
`stream_seed` hashes the fixed method-domain string, purpose and root seed,
then assigns each channel 2^64 integer seeds and each update 2^32 seeds. Path r
uses its channel/update base plus r through the existing `FactorSampler.draw`.
The channel and update intervals cannot overlap; hashes separate distinct root
seeds and purposes. Namespaces are `training`, `development_diagnostics` and
`final_evaluation`. Training and debugging must not consume final streams.

The base seed does not depend on path count, so the first 15 conditional draws
at R=16 are the first 15 at R=32; the conditional weights appropriately change.
Seed integers are recorded as decimal strings to preserve all bits through
JSON consumers. Same-structure paired arms get identical maps at a shared
root seed and update index. Once architectures differ, common integer seeds
do not imply identical physical events. A structural change requires rebuilding
the sampler; it must not reset the globally fresh update counter.

## Backend semantics and accounting

`MixtureSampler(circuit).ensemble(seed=..., update_index=...)` returns a
`MixtureBatch`. The batch exposes `gradient`, `online_update` and
`adaptive_online_update`, all forwarding **both maps and fixed weights** to the
existing independent `StatevectorBackend` methods. The batch validates the full
circuit signature and all-zero input state before replay. The adaptive method
uses the backend's existing second-moment update after the weighted local
signal has been averaged. Shared parameters update per occurrence with the
existing per-occurrence bias-correction counter. Finite online displacements,
nonlinear Adam scaling and finite optimizer trajectories are not claimed to be
unbiased because the underlying fixed-theta gradient estimator is unbiased.

Setup CPU and wall times belong to the sampler and are charged once per setup.
Each batch records sampling CPU and wall time. The later driver must measure
its full enclosing update, preserving common inclusion of setup, sampling,
backend calls, optimizer work, serialization and checkpointing across arms;
sampling sub-times are not added again to an enclosing timer. Failed attempts
and replayed work are charged separately and never erased by resume.

For N=2R paths, G gates, T trainable occurrences, and K selected Pauli factors
across all paths, `resource_counts` records the following successful-call costs:

| Operation                | Directional paths | Gate applications | Tangent applications | Noise-factor applications |
| ------------------------ | ----------------: | ----------------: | -------------------: | ------------------------: |
| gradient                 |                2N |               2GN |                   TN |                        2K |
| online / adaptive online |                3N |           (3G+T)N |                   TN |                        3K |
| extra sampled metrics    |                 N |                GN |                    0 |                         K |

Counts include zero weights, shared occurrences, zero angle scales, online gate
reevaluations and cancelling factors. These are verified against the actual
statevector primitive, and are valid for the currently frozen backend source.
Extra metrics and exact-channel scoring require their own charged calls. Later
results must report equal-update and common-CPU comparisons against serious
exact Adam and exact mean-two L-BFGS controls with matched starting circuits,
targets, initialization and any structural rules. This kernel supplies no
scientific endpoint, optimizer learning-rate selection, or stopping rule.

## Engineering acceptance and provenance

Small q1-q3 tests enumerate all 8+2 or 64+4 channel-specific Pauli outcomes and
compare their exact weighted gradient and fidelity to the unchanged density
mean-two objective. Tests independently check the average-error formulas, every
nonempty dephasing conditional pattern, sampled finite differences, nested and
fresh streams, channel weight normalization, metadata preservation, reversed
sites, signed/shared/unused/zero-scale parameters, disabled gates, empty windows,
cancelling factors, invalid inputs, stale circuit rejection and all backend cost
counts. Mathematical acceptance requires gradient/fidelity agreement within
3e-14 on these cases; it does not establish statistical optimizer performance.

This is a known estimator adaptation. The improved trajectory-sampling idea of
separately weighting a deterministic no-jump contribution and a sampled event
contribution appears in [Abdelhafez, Schuster and Koch, Physical Review A 99,
052327 (2019)](https://link.aps.org/accepted/10.1103/PhysRevA.99.052327).
Here the existing repository's exact Pauli-factor implementation is extended
to the calibrated local dephasing law. Concatenation with half weights follows
linearity of expectation and is not claimed as novel. More broadly, optimizing
over uncertain quantum dynamics is established robust-control practice; see
[Kosut, Grace and Brif, Physical Review A 88, 052326
(2013)](https://arxiv.org/abs/1301.3235). Our equal-mean objective is not their
worst-case sequential-convex-programming algorithm. No state-of-the-art or
tensor-network scalability claim follows from these engineering tests.
