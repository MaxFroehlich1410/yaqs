# Diagnosis of the failed q6 MPS overlap gate

The frozen gate remains **failed** for all eight endpoints. No tolerance, optimizer, gradient, source circuit or acceptance flag has been changed. This diagnosis uses saved gradients, optimizer states and compressed truncation audits only. It performs no circuit propagation or new dense/MPS gradient evaluation.

The dominant parameter discrepancy is consistent with double-precision gradient roundoff amplified by Adam near zero gradients. Six endpoints fail the parameter threshold only in provable initial global-phase coordinates. Gaussian and TFIM structural endpoints also have small non-gauge drift, so a claim that every discrepancy is physically irrelevant would be premature.

The authoritative derived artifact is `output/overnight-state-preparation-20260913-0108/mps-overlap/ANALYSIS.json`. It preserves every offending parameter index, gate identity, gradient pair, corrected moment, update bound, and source hash. The frozen protocol, summary and case hashes, 24 raw audit hashes, and every case dependency were checked; 145 numerical-source/source-endpoint/dependency SHA-256 checks matched. Scalar replay of all saved gradients reproduces **both complete final Adam states exactly**, including theta and both moments, for all eight endpoints.

## What failed and where

Indices below are zero-based parameter indices, not gate positions. In every source, indices `2, 5, 8, 11, 14, 17` are the first `RZ` on qubits 0–5, at physical positions `0, 3, 6, 9, 12, 15`. Their angle scales and complete gate metadata are retained in the JSON diagnosis.

An initial `RZ(phi)` acting on untouched `|0>` contributes only `exp(-i phi/2)`. Every later fixed unitary noise/gate replay retains that scalar global phase. Thus the exact fidelity gradient of these six coordinates is zero. Their small saved nonzero gradients are numerical residuals, not evidence of a physical descent direction.

The following table reports **after update 3**, rather than the maximum over all three updates. In particular, Gaussian structural has a smaller final parameter discrepancy than its earlier maximum; these are different measurements.

| Target    | Endpoint                | Max parameter difference | Max difference outside six phase coordinates | Parameter-only phase-aligned state-distance bound |
| --------- | ----------------------- | -----------------------: | -------------------------------------------: | ------------------------------------------------: |
| Gaussian  | original108             |              3.951687e-9 |                                 3.518019e-14 |                                      1.583343e-13 |
| Gaussian  | structural mean2 L-BFGS |             8.586072e-10 |                                 3.126858e-10 |                                      2.883334e-10 |
| TFIM      | original108             |              3.175299e-9 |                                 3.383405e-14 |                                      9.362303e-14 |
| TFIM      | structural mean2 L-BFGS |              3.035225e-9 |                                  2.091679e-9 |                                       1.039383e-8 |
| Haar      | original108             |              2.020943e-9 |                                 4.440892e-15 |                                      4.171316e-14 |
| Haar      | structural mean2 L-BFGS |              3.837088e-9 |                                 4.096723e-14 |                                      8.931050e-14 |
| MPS bond2 | original108             |              3.350056e-9 |                                 1.611211e-14 |                                      5.231319e-14 |
| MPS bond2 | structural mean2 L-BFGS |              1.254622e-9 |                                 8.515966e-13 |                                      9.944654e-13 |

All four original108 endpoints, plus structural Haar and structural MPS bond2, exceed `1e-10` only in the six initial phase coordinates. Their remaining differences are much smaller than the frozen parameter tolerance, but are not asserted to be identically zero.

Gaussian structural additionally exceeds `1e-10` at indices **13 and 16**, both later RZ gates. Their first gradients are approximately `1.23868665e-8` and `-4.98246056e-10`; the dense/MPS differences are respectively `-5.26462331e-16` and `-1.72316604e-16`. These small absolute discrepancies are amplified because the initial gradients are comparable to or smaller than epsilon.

TFIM structural has non-gauge failures after update 1 at indices **1, 4, 7, 10, 13, 16, 51, 52, 79, 80, 84, 85, 89, 90**. All are RZ gates with initial gradient magnitudes approximately `1.19e-12`–`9.89e-11`. The initial backend differences are of order `1e-16`–`1e-15`. Index **0**, an RY gate, additionally exceeds the parameter tolerance after update 3, with a difference `1.721146e-10`. This later coupled drift must not be relabeled as an exact phase freedom.

For example, TFIM structural index 16 has first gradients `2.2827240098166612e-11` and `2.2826189049393595e-11`. Their difference is only `1.05104877e-15`, yet the first Adam parameter difference is `2.09253354e-9`.

## Adam sensitivity explains the magnitude

The unchanged implementation is `state_preparation_exact_adam.adam_step`, using rate `a=0.02`, epsilon `e=1e-8`, beta1 `0.9`, beta2 `0.999`, and fresh moments. At the first update, bias correction gives

```math
\theta_1=\theta_0-a\,\frac{g}{|g|+e}.
```

The scalar function `g/(|g|+e)` is globally Lipschitz with constant `1/e`. Therefore, for the same initial theta,

```math
|\Delta\theta_1|\le (a/e)|\Delta g_1|=2\times10^6|\Delta g_1|,
```

apart from the final floating-point subtraction. This bound was checked **coordinatewise for every first-step parameter of all eight endpoints**, with a small subtraction-roundoff allowance. Every coordinate satisfies it.

For Gaussian original108 index 17, the first gradients are `2.08166817e-17` and `-5.69143706e-16`. The simple bound is approximately `1.17992077e-9`; the saved parameter difference is `1.17992072e-9`. The mismatch has the size predicted by the unchanged optimizer, despite excellent absolute gradient agreement.

For later updates, let `mD,mM` be bias-corrected first moments and `sD,sM` the square roots of the corresponding bias-corrected second moments. A coordinatewise algebraic bound on one update difference is

```math
a\left[
\frac{|m_D-m_M|}{s_D+e}
+\frac{|m_M|\,|s_D-s_M|}{(s_D+e)(s_M+e)}
\right].
```

Summing these bounds and the subtraction-roundoff allowances over steps also bounds every reconstructed parameter difference. This checks consistency of the recorded updates; it does not establish a new numerical-gradient accuracy tolerance.

In the six phase coordinates, `s` is tiny compared with epsilon, so the three-step drift is approximately

```math
\Delta\theta_3\simeq-2\times10^6[
1.772577199\,\Delta g_1+
0.858419111\,\Delta g_2+
0.369003690\,\Delta g_3].
```

Across all saved phase-coordinate steps, this epsilon-dominated prediction differs from the actual parameter difference by at most **4.24e-16**. This is strong evidence for the stated amplification mechanism. It does not justify changing epsilon or zeroing small gradients in the frozen run.

## Gradient comparisons after update 1 are confounded

Before update 1, dense and MPS evaluate exactly the same parameter vector. The maximum absolute initial gradient discrepancy over all eight endpoints is **2.259020e-15**.

Before updates 2 and 3, the vectors have already diverged. Their saved gradient differences combine backend numerical differences with the response of the objective to different parameters. Consequently, TFIM structural's maximum `1.403340e-10` gradient difference, observed **before update 2**, is not a same-theta backend-error measurement. Its before-update-3 maximum is `1.000837e-10`. Gaussian structural similarly reaches `2.996499e-12` later, while its initial agreement is at roundoff scale.

Across the saved points, the largest weighted loss/fidelity difference is **1.032507e-12** and the largest individual trajectory-fidelity difference is **2.833400e-12**, both TFIM structural. These are below the frozen `1e-10` objective tolerance. Full first- and second-moment differences also remain below `1e-10`. The parameter criterion fails everywhere, and TFIM structural also fails the gradient criterion after the parameter vectors diverge.

## What can be bounded physically without propagation

For a scalar Pauli rotation with angle scale `s_i`, the generator norm is `|s_i|/2`. After removing only the six proven initial global phases, the telescoping product bound gives, for the **ideal exact circuit at the two saved parameter vectors** and every identical fixed unitary noise path,

```math
\|\psi_D-e^{i\phi}\psi_M\|_2
\le B_\theta=\frac12\sum_{i\notin\mathrm{initial\ phase}}|s_i\Delta\theta_i|.
```

The table gives this parameter-only upper bound after update 3. The pure-state infidelity is at most `B_theta^2` (capped at one), and any target projector-fidelity difference has the conservative bound `2 B_theta`. The worst final state-distance bound is `1.039383e-8` for TFIM structural, corresponding to a parameter-only state-infidelity bound approximately **1.081e-16**. No cancellation among later rotations was assumed, so this bound is conservative.

These are analytical bounds on changing theta in an exact unitary circuit. They **do not** bound the unobserved numerical error of the actual MPS final state. Saved target fidelities alone cannot certify full phase-aligned state agreement. That distinction motivates the separate follow-up below.

## Raw tensor evidence and actual diagnostic costs

All **13,968** saved SVD truncation decisions were read and checked against their compact audits. Their total discarded squared singular-value weight is **exactly zero**, their maximum discarded tail is zero, and no hard-cap reduction occurred. Every endpoint has maximum internal bond **8**, including forward, adjoint, derivative and target work, within the q6 Hilbert-space bound. Target MPS bonds are `[2,4,8,4,2]`.

The largest forward final norm error is **8.437695e-15**. The largest dense-to-target-MPS factorization error is **1.193490e-15**. The audit records **58,418** local operator applications and maximum individual MPS tensor storage **2,688 bytes**. The latter excludes complete tapes and simultaneous states, and is not peak process memory.

The original completed diagnostic cost **11.089131 CPU seconds** across four target cases, including its measured input, setup, replay, checks and checkpoint I/O. Nested phase CPU includes **10.115356 s** for MPS gradients, **0.165723 s** for dense gradients and **0.252228 s** for the additional dense path-fidelity diagnostics. Those phase timings must not be added again to the total. Both backends made 24 weighted gradient calls with eight paths each; dense additionally propagated 192 individual diagnostic forward paths. This instrumented q6 diagnostic is not an optimization benchmark or scalability result.

## Proposed separate engineering audit — requires allocation and freeze

Keep the failed result and frozen code untouched. Use a new versioned output directory and a new protocol. Retain all four development targets, both endpoints, the same eight paths and `mps_overlap` stream identities, the same untruncated bridge, the same three Adam updates, and the original optimizer constants. No confirmation target, gradient clipping, gauge removal from training, epsilon change, or new optimizer setting is introduced.

Predeclare the following independent checks:

1. At each of the three saved dense pre-update vectors, evaluate **both backends at that identical theta** on the same regenerated, hash-checked maps. Retain all coordinate gradients, weighted losses and individual path fidelities. Require each original gradient/loss/path-fidelity discrepancy to remain at most `1e-10`.
2. Propagate the physical post-update states for both saved backend theta vectors on the same paths. Record norms, phase-aligned state distances, overlaps and target fidelities. Also compare dense and MPS at a common post-update theta, separating backend propagation error from parameter sensitivity. Require same-theta phase-aligned state agreement and norm error at most `1e-10`; retain different-theta results alongside the analytical `B_theta` bound rather than call them same-input errors.
3. Independently replay Adam from the recorded gradients and verify the first-step `a/e` bound and full coordinatewise recurrence bound, including stated floating-point subtraction allowance. This is a derived sensitivity check, not a silent replacement of the old unconditional parameter threshold. Preserve raw parameter differences in every result.
4. Retain the original exact-zero discarded-tail requirement, bond limit 8 and all raw split/norm/resource audits. Charge regenerated maps, every new propagation, both backend gradients, target factorization and failed attempts. Use one single-threaded worker only after coordinator allocation.

Report success or failure of this **new engineering audit** separately. Even success would diagnose the original gate's optimizer conditioning; it would neither turn the old failure into a pass nor, by itself, authorize scaling or establish optimization superiority.
