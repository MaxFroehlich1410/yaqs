# State-Preparation Benchmark Definition

This document is the normative scientific definition of the state-preparation
benchmark. It fixes the mathematical experiment, targets, noise channels, and
reporting requirements while keeping the implementation in the minimal branch
small. Names of Python functions appear only in the implementation-status
section and are not part of the mathematical definition.

## Scientific experiment

The separately labelled
[next Krotov/pruning experiments](../experiments/state_preparation_next_experiments.md)
specify multi-target architecture-changing comparisons, exact method budgets,
and noise-aware refinements. They extend, rather than replace, the base
train-once/evaluate-many definition below; the new protocol is not yet a runner.

### Scope

- Qubit counts: 6 and 12.
- Target set: nine target states from four families for each qubit count.
- Ansatz family: nearest-neighbor brickwall matrix-product disentangler (BMPD).
- Noise families:
  1. Ballarin/Quantinuum angle-dependent local depolarizing noise.
  2. Dephasing noise.
  3. Depolarizing noise.
- Primary metric: target-state fidelity after optimization.
- Secondary metrics: circuit depth, one- and two-qubit gate counts, number of
  trainable parameters, optimization wall time, and trajectory or shot budget.

Algorithms may use different update rules, but comparisons are meaningful only
when the target, ansatz instance, initialization, optimization budget, and test
conditions are held fixed. Every method must report both noiseless and noisy
test fidelities.

### Objective

For target state |phi>, initial state |0>^{tensor n}, and a parameterized
unitary U(theta), the noiseless fidelity and loss are

```{math}
F_0(\theta)
  = \left|\langle\phi|U(\theta)|0\rangle^{\otimes n}\right|^2,
\qquad
L_0(\theta) = 1-F_0(\theta).
```

A noisy circuit realization is indexed by omega. Its trajectory fidelity and
the channel-averaged objective are

```{math}
F_\omega(\theta)
  = \left|\langle\phi|\psi_\omega(\theta)\rangle\right|^2,
\qquad
F_{\mathcal N}(\theta)
  = \mathbb E_\omega[F_\omega(\theta)],
\qquad
L_{\mathcal N}(\theta) = 1-F_{\mathcal N}(\theta).
```

The base benchmark trains each target without noise and evaluates the same
final parameter vector under every test condition. This train-once/evaluate-many
rule prevents noise evaluations from silently using different optimized
circuits. Noise-aware training may be studied as a separately labelled
extension; it must report its training channel and use training randomness
disjoint from all final test randomness.

### Shared brickwall ansatz

For each neighboring pair, one nine-parameter BMPD block is

```{math}
B(\vartheta)
  =
  R_{XX}(\vartheta_6)
  R_{YY}(\vartheta_7)
  R_{ZZ}(\vartheta_8)
  \left[
    U_3(\vartheta_0,\vartheta_1,\vartheta_2)
    \otimes
    U_3(\vartheta_3,\vartheta_4,\vartheta_5)
  \right].
```

A BMPD depth d contains 2d alternating nearest-neighbor brickwall layers.
For the even qubit counts in this benchmark, this gives d(n-1) two-qubit
blocks. An optional initial trainable U3 layer contributes three parameters per
qubit. With that layer enabled, the logical parameter count is

```{math}
P(n,d) = 3n + 9d(n-1).
```

The state-preparation circuit applies the adjoints of the disentangling blocks
in reverse layer order. All methods must use the same value of d and the same
choice for the initial U3 layer in a comparison. Report both logical BMPD
statistics and, when compilation changes the circuit, final native statistics.

Required run metadata are:

- qubit count and target identifier;
- BMPD depth, equivalently 2d brickwall layers;
- whether the initial U3 layer is enabled;
- logical and native parameter and gate counts;
- parameter initialization rule and seed;
- optimizer name, hyperparameters, stopping rule, and budget;
- tensor-network truncation settings, if any.

## Noise models

### Common rate convention

The hardware-motivated benchmark specifies the average error of a complete
gate-noise channel, then converts that error into Lindblad coefficients for
each chosen noise model. The reference targets are

| Gate context      | Target average gate infidelity |
| ----------------- | -----------------------------: |
| Single-qubit gate |          r_1 = 0.064% = 6.4e-4 |
| Two-qubit gate    |           r_2 = 0.51% = 5.1e-3 |

These are not probabilities or rates to assign separately to every jump
operator. Dephasing, local depolarizing, and correlated Pauli models must be
calibrated separately to the same target error on the same gate support.
Adding operators without recalibration increases the overall noise and does
not isolate the effect of correlations.

Calibration correction recorded on 2026-09-05: this supersedes the earlier
scientific convention that assigned the two numbers to every operator. The
standard executable profiles still use those legacy coefficients for
reproducibility. A separate opt-in provider now calibrates the focused causal
and hybrid reruns without changing the simulator. See the legacy-run derivation
below and the implementation-status section before using these commands.

#### Hardware evidence and strength range

[Google's Willow specification, published 2024-12-09](https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf),
reports simultaneous mean operation errors, explicitly identified as average
errors measured using randomized benchmarking:

| Configuration                       | Single-qubit average error |       Two-qubit average error |
| ----------------------------------- | -------------------------: | ----------------------------: |
| Willow QEC chip                     |          0.035% +/- 0.029% |          0.33% +/- 0.18% (CZ) |
| Willow random-circuit-sampling chip |          0.036% +/- 0.013% | 0.14% +/- 0.052% (iswap-like) |

The reference values exactly equal 0.035% + 0.029% = 0.064% and
0.33% + 0.18% = 0.51%. This suggests a possible origin for the original
benchmark numbers, but does not establish their provenance. The sums are
benchmark choices, not demonstrated hardware maxima or confidence bounds.

[IBM reported a Heron r3 median two-qubit error of 0.117% on 2026-05-04](https://newsroom.ibm.com/2026-05-04-ibm-a-decade-of-quantum-on-the-cloud).
This provides another hardware scale, not a complete calibration of our
circuits or a universal value for IBM processors.

A proposed sensitivity grid, separate from the fixed reference pair, is:

- single-qubit average errors: 0.01%, 0.035%, and 0.064%;
- two-qubit average errors: 0.10%, 0.14%, 0.33%, and 0.51%;
- noiseless reference; optionally 1% two-qubit error as a labelled stress test.

These are selected hardware-motivated operating points, not measured device
quantiles. Report the selected pairs explicitly. A device-calibrated study
instead requires a dated calibration snapshot or learned noise model, including
the processor, physical qubits, native gates, and layer context. Keep that
study distinct from homogeneous noise models matched only in average error.

#### Lindblad rates, integrated coefficients, and channel errors

The [original TJM paper, Sec. II](https://arxiv.org/html/2501.17913v2#S2),
uses the Lindblad master equation

```{math}
\dot\rho=-i[H,\rho]+\sum_m\gamma_m
\left(L_m\rho L_m^\dagger-\frac12\{L_m^\dagger L_m,\rho\}\right).
```

For the Pauli noise profiles here, L_m = P_m is an unscaled Pauli string,
P_m^dagger P_m = I. Define the dimensionless integrated coefficient
alpha_m = gamma_m Delta t for a gate or layer window of duration Delta t.
Rates gamma_m have inverse-time units when Delta t is a physical duration;
with a unit circuit window their numerical values equal alpha_m. Subdividing
a window must preserve its total integrated coefficients.

The exact noise channel is

```{math}
\mathcal E=\exp\!\left[\sum_m\alpha_m\mathcal D_{P_m}\right],
\qquad \mathcal D_P(\rho)=P\rho P-\rho.
```

For Pauli generators it factorizes into commuting channels

```{math}
\mathcal E=\bigcirc_m\mathcal E_m,\qquad
\mathcal E_m(\rho)=(1-q_m)\rho+q_mP_m\rho P_m,\qquad
q_m=\frac{1-e^{-2\alpha_m}}{2}.
```

This is the [circuit TJM paper's Eqs. (8)-(10)](https://arxiv.org/html/2607.01323v2#S2.SS3).
The probability q_m describes odd jump parity for that factor: two identical
Pauli jumps cancel. It is not the probability of at least one MCWF jump.

Let D = 2^k for a k-qubit noise window; D is unrelated to BMPD depth d.
For a Pauli channel, let p_I be its final identity-Pauli probability, including
multiple-jump cancellations. The standard fidelity relation gives

```{math}
F_{\mathrm{avg}}=\frac{D p_I+1}{D+1},\qquad
r=1-F_{\mathrm{avg}}=\frac{D}{D+1}(1-p_I).
```

See [IBM's average-gate and process-fidelity definitions](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#average_gate_fidelity).
For a supplied Pauli error p = 1-p_I, first convert r = D p/(D+1).
[Google distinguishes Pauli errors in calibration data from average errors in its specification sheet](https://quantumai.google/cirq/google/calibration).

For any N2 or N3 operator set, an explicit calibration equation follows from
the Pauli transfer eigenvalues. For each phase-free k-qubit Pauli Q, including
identity,

```{math}
\eta_Q=\exp\!\left[-2\sum_{m:\,P_mQ=-QP_m}\alpha_m\right],\qquad
p_I=\frac{1}{D^2}\sum_Q\eta_Q,\qquad
r=\frac{D^2-\sum_Q\eta_Q}{D(D+1)}.
```

For the standard equal-coefficient profiles, set every active alpha_m to a
common a and solve this scalar equation for the target r_1 or r_2. For a
specified nonuniform model, fix its relative coefficients and solve for an
overall nonnegative scale. A single error measurement cannot determine those
relative coefficients or distinguish dephasing from correlated noise.

#### Worked exact-channel conversions

The following formulas are derived from the preceding eigenvalue and fidelity
relations, not additional hardware measurements.

For equal X, Y, Z coefficients a on one qubit, each nonidentity Pauli
anticommutes with two generators. Hence its eigenvalue is exp(-4a), giving

```{math}
p_I=\frac{1+3e^{-4a}}4,\qquad
r_1=\frac{1-e^{-4a}}2,\qquad
a=-\frac14\log(1-2r_1).
```

For equal coefficients on the six local Paulis (X, Y, and Z on each qubit),
the exact two-qubit channel is the tensor product of those one-qubit channels.
Thus

```{math}
p_I=\left(\frac{1+3e^{-4a}}4\right)^2,\qquad
r_2=\frac45\left[1-\left(\frac{1+3e^{-4a}}4\right)^2\right],\qquad
a=-\frac14\log\!\left(\frac{4\sqrt{1-5r_2/4}-1}{3}\right).
```

For all 15 nonidentity two-qubit Paulis with equal coefficients, every
nonidentity Q anticommutes with eight generators. Therefore

```{math}
r_2=\frac34(1-e^{-16a}),\qquad
a=-\frac1{16}\log(1-4r_2/3).
```

At the reference hardware-error targets, the resulting coefficients are:

| Noise window and operator set         | Target average error | Integrated coefficient a per operator |
| ------------------------------------- | -------------------: | ------------------------------------: |
| One qubit, X/Y/Z                      |               0.064% |                       3.2020497493e-4 |
| Two qubits, six local Paulis          |                0.51% |                       1.0664702651e-3 |
| Two qubits, all 15 nonidentity Paulis |                0.51% |                       4.2645158426e-4 |

These equal gamma only for Delta t = 1. They calibrate the exact Lindblad
channel, not the finite-step one-jump sampler. The 15-operator formula applies
to the combined N3 profile, not the nine-operator two_site-only profile.
For the latter and for N2, use their own operator sets in the general equation;
do not reuse the six- or 15-operator coefficients.

#### What noise the recent tests actually used

The original local-depolarizing causal and hybrid tests used the legacy
per-operator values gamma_1 = 6.4e-4 and gamma_2 = 5.1e-3 with Delta t = 1.
For Pauli jumps, the no-jump drift is exp(-A/2) I, where
A = sum_m alpha_m. Its squared norm is exp(-A). The current discrete sampler
therefore chooses no jump with probability exp(-A), or exactly one operator
with conditional probability alpha_m/A. Its ensemble channel is

```{math}
\mathcal E_{\mathrm{one\ jump}}(\rho)
  =e^{-A}\rho+(1-e^{-A})\sum_m\frac{\alpha_m}{A}P_m\rho P_m.
```

The A = 0 case is identity. Because every sampled P_m here is nonidentity,
p_I = exp(-A), so this sampler's average error is exactly

```{math}
r_{\mathrm{one\ jump}}=\frac{D}{D+1}(1-e^{-A}).
```

There were three operators after a one-qubit gate and six after a two-qubit
gate. Substitution gives

```{math}
r_{1,\mathrm{legacy}}=\frac23(1-e^{-3(0.00064)})=0.00127877198605,
\qquad
r_{2,\mathrm{legacy}}=\frac45(1-e^{-6(0.0051)})=0.0241092473011.
```

| Gate window | Intended average error | Legacy sampler average error | Ratio to intended error |
| ----------- | ---------------------: | ---------------------------: | ----------------------: |
| One qubit   |                 0.064% |                   0.1278772% |                 1.9981x |
| Two qubits  |                  0.51% |                   2.4109247% |                 4.7273x |

The corresponding jump-event probabilities are 0.1918158% and 3.0136559%;
these are not average gate errors. The table is specific to the recent
local-depolarizing profile and must not be applied to all historical noise
models. The runs remain useful as higher-noise stress tests, but their small
optimization gains do not establish failure at the intended hardware levels.
Lower noise does not by itself guarantee a gain from noise-aware training over
the frozen noiseless-trained baseline; that comparison must be rerun.

The one-jump channel and the exact Lindblad channel generally differ at order
A^2 per window. The circuit TJM paper explicitly uses an
[at-most-one-jump discretization in Sec. III](https://arxiv.org/html/2607.01323v2#S3).
Monte Carlo sample count does not remove this finite-step bias. Before claiming
hardware-calibrated results, compare small-system trajectory evolution with
the exact channel and report a numerical tolerance. Use sufficient substeps
at fixed total alpha_m or exact Pauli-factor sampling when necessary. Do not
replace the exact-generator coefficients above with a sampler-specific fit
without explicitly declaring the different calibration convention.

The calibrated local-depolarizing rerun introduced on 2026-09-05 uses the
exact-generator coefficients in the table, retaining the original unit-window
sampler and gate placement for a controlled comparison. Its actual finite-step
average errors are 0.0640102455% and 0.5102714183%. Independent dense generator
exponentiation tests recover the exact target errors to absolute tolerance
1e-13. For the q6 frozen checkpoint, independent density-matrix evolution gives
fidelity 0.6779627012 for the one-jump channel and 0.6779908161 for the exact
channel, an absolute difference of 2.81e-5. See the
[corrected-run findings](../experiments/state_preparation_calibrated_rerun_findings.md)
for the protocol, validation, and results. These are hardware-motivated homogeneous abstract-gate experiments,
not native-device simulations of Google or IBM hardware.

#### Device calibration and reporting safeguards

[IBM's noise learner](https://quantum.cloud.ibm.com/docs/en/guides/noise-learning)
can supply fitted Pauli-Lindblad coefficients lambda_m for an entangling
layer. In the convention above, alpha_m = lambda_m, or
gamma_m = lambda_m/Delta t. Preserve the calibrated operators, qubits, and
window; do not repeat an entire learned layer channel after each gate.

Match measurements to native operations before assigning noise. IBM's
two-qubit metric includes accompanying single-qubit Clifford errors, and
virtual RZ gates have zero reported gate error. Avoid double counting that
single-qubit contribution or charging an abstract U3 or RZ operation a
calibration for a different native instruction.
See [IBM's calibration metric definitions](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information).
Adding separate T1/T2 noise to an already calibrated total gate channel also
requires an explicit error-budget decomposition.

For every calibrated run, record the source and calibration date, measured
metric, selected r_1/r_2 or learned layer coefficients, complete jump-operator
set and normalization, Delta t, resulting alpha_m/gamma_m, native placement,
and exact-channel versus finite-step validation. Distinguish hardware-motivated
model assumptions from measured device-specific noise structure. The N1
Ballarin channel retains its separate angle-dependent calibration below.

### N1: Ballarin/Quantinuum angle-dependent local depolarizing noise

Source: Marco Ballarin, Juan Jose Garcia-Ripoll, David Hayes, and Michael
Lubasch, “Efficient quantum state preparation of multivariate functions using
tensor networks,” arXiv:2511.15674, Eqs. (8)–(9).

Use benchmark identifier ballarin_coupled and the following definition:

1. Express each two-qubit entangler in the native form

   ```{math}
   R_{ZZ}(\theta) = \exp[-i\theta(Z\otimes Z)/2].
   ```

   Compile RXX and RYY with noiseless single-qubit basis changes and one RZZ
   rotation with the same angle. Any other entangler must be lowered to an
   explicit RZZ sequence.

2. Canonicalize each native angle to [-pi, pi), then set a = |theta|. Taking
   the absolute value is the benchmark convention for extending the paper’s
   nonnegative calibration fit to signed rotations.

3. Remove a native RZZ rotation when a <= 1e-4 before counting gates or applying
   noise.

4. Treat every single-qubit gate, including compilation basis changes, as
   noiseless.

5. Immediately after each retained ideal RZZ(theta) on qubits i and j, apply
   K_i(a) tensor K_j(a), where

   ```{math}
   \epsilon(a) = 2.1\times 10^{-4} + 1.43\times 10^{-3}a,
   ```

   ```{math}
   r(a)
     = \frac{1}{3}\left(1-\sqrt{1-\frac{5}{4}\epsilon(a)}\right),
   ```

   ```{math}
   K_q(a)(\rho)
     = [1-3r(a)]\rho
       + r(a)\left(
           X_q\rho X_q + Y_q\rho Y_q + Z_q\rho Z_q
         \right).
   ```

An exact Kraus representation of K_q is
{sqrt(1-3r) I, sqrt(r) X, sqrt(r) Y, sqrt(r) Z}. The two-qubit channel
therefore has the 16 tensor-product Kraus operators formed from the two local
sets. A trajectory implementation must independently sample I, X, Y, or Z on
each participating qubit with probabilities 1-3r, r, r, and r. This is the
product of two local channels, not a correlated two-qubit Pauli channel.

Do not approximate these categorical branch probabilities with TJM rates and
do not additionally apply the common N2/N3 gate-noise channels. The compatibility
identifier contains “coupled” because both local channels are attached to the
same native gate, not because their Pauli outcomes are correlated.

### N2: Dephasing noise

The site-support choices are:

- single_site: one Z process on every affected qubit;
- two_site: one correlated ZZ process on the qubit pair of a multi-qubit gate;
- single_site plus two_site: both process sets.

The placement choices are single-qubit gates, multi-qubit gates, or all gates.

| Identifier         | Site support              | Placement          | Complete-window calibration target |
| ------------------ | ------------------------- | ------------------ | ---------------------------------- |
| dephasing_1s_1q    | single_site               | Single-qubit gates | r_1                                |
| dephasing_1s_2q    | single_site               | Multi-qubit gates  | r_2                                |
| dephasing_1s_all   | single_site               | All gates          | r_1 after 1q; r_2 after 2q         |
| dephasing_2s_2q    | two_site                  | Multi-qubit gates  | r_2                                |
| dephasing_1s2s_all | single_site plus two_site | All gates          | r_1 after 1q; r_2 after 2q         |

For a single_site profile after a multi-qubit gate, add one Z process for each
gate qubit and calibrate their joint channel to r_2, not each process to r_2.
These tables specify the corrected scientific calibration; existing executable
identifiers still resolve to legacy rates as documented below.
The minimum profile uses dephasing_1s_all and
dephasing_2s_2q.

### N3: Depolarizing noise

A single-site depolarizing set contains X, Y, and Z on every affected qubit.
The frozen two-site set contains exactly

```text
XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
```

Each of the nine operators receives the same coefficient, determined by
calibrating their joint channel to r_2. Operators with identity on either
qubit are excluded. A combined
single-site plus two-site profile after a multi-qubit gate contains the six
single-site processes, three on each gate qubit, and all nine two-site
processes.

| Identifier            | Site support              | Placement          | Complete-window calibration target |
| --------------------- | ------------------------- | ------------------ | ---------------------------------- |
| depolarizing_1s_1q    | single_site               | Single-qubit gates | r_1                                |
| depolarizing_1s_2q    | single_site               | Multi-qubit gates  | r_2                                |
| depolarizing_1s_all   | single_site               | All gates          | r_1 after 1q; r_2 after 2q         |
| depolarizing_2s_2q    | two_site                  | Multi-qubit gates  | r_2                                |
| depolarizing_1s2s_all | single_site plus two_site | All gates          | r_1 after 1q; r_2 after 2q         |

The minimum profile uses depolarizing_1s_all and
depolarizing_2s_2q.

### Exact channels and finite-trajectory estimates

The mathematical channel definitions are exact. Direct Ballarin sampling
selects an exact product-channel branch. The current standard dephasing and
depolarizing evaluator instead samples the finite-step circuit-TJM no-jump
and one-jump channel derived above; it is not generally the exact exponential
of the stated generator. Separate this deterministic bias from sampling
uncertainty and tensor-network truncation.

For N fresh test trajectories, report the arithmetic mean

```{math}
\bar F = \frac{1}{N}\sum_{k=1}^{N} F_k.
```

For N >= 2, also report the sample standard deviation s with ddof = 1, the
standard error s/sqrt(N), and the clipped normal 95% interval

```{math}
\left[
  \max(0,\bar F-z_{0.975}s/\sqrt N),
  \min(1,\bar F+z_{0.975}s/\sqrt N)
\right].
```

With one trajectory, the mean is valid but sampling uncertainty is undefined.
A finite-trajectory mean is a Monte Carlo estimate, not an exact density-matrix
fidelity. Compare methods with the same test budget and resolved random streams.

## Fixed target states

The checked-in file benchmarks/state_preparation_target_states.json is the
normative target fixture. It contains every target for n = 6 and n = 12. The
generator is benchmarks/generate_state_preparation_targets.py.

Each complex amplitude is encoded as [real, imaginary]. Amplitude index k uses
little-endian computational-basis ordering: qubit i is bit i of k. The global
phase is fixed so that the largest-magnitude amplitude is positive real.

Verify the fixture from the repository root with:

```bash
uv run python benchmarks/generate_state_preparation_targets.py --check
```

The check compares all benchmark-defining metadata exactly, ignores NumPy and
SciPy version strings as environment provenance, and uses tight numerical
tolerances only for state amplitudes, norms, and TFIM ground energies.

### T1: Gaussian amplitude state

Identifier: gaussian_mu0p5_sigma0p1.

Use the endpoint-excluded quantics grid on [0,1):

```{math}
x_k
  = \sum_{i=0}^{n-1}\operatorname{bit}_i(k)2^{-(i+1)}.
```

Qubit 0 is the coarsest quantics bit. With mu = 0.5 and sigma = 0.1, encode the
square root of the Gaussian probability density:

```{math}
\psi(x)
  \propto
  \exp\left[-\frac{(x-0.5)^2}{4(0.1)^2}\right].
```

Normalize the amplitude vector to unit Euclidean norm. This follows the
probability-distribution construction in arXiv:2602.12042, Section 3.2.1.

### T2–T4: transverse-field Ising ground states

Use the uniform open-chain one-dimensional TFIM

```{math}
H = -J\sum_{i=0}^{n-2}Z_iZ_{i+1}-h\sum_{i=0}^{n-1}X_i,
\qquad J=1.
```

| Identifier    | Regime                       | h/J | Eigensolver base seed |
| ------------- | ---------------------------- | --: | --------------------: |
| tfim_ferro    | Ferromagnetic                | 0.5 |                  1729 |
| tfim_critical | Thermodynamic critical point | 1.0 |                  2718 |
| tfim_para     | Paramagnetic                 | 1.5 |                  3141 |

For n qubits, initialize the sparse eigensolver with seed
base_seed + 10000 n. This is a reproducibility seed, not physical disorder.
The fixture records the uniform couplings, uniform fields, finite-size ground
energy, and resolved initial-vector seed.

### T5–T7: dense Haar-random states

| Identifier    | Seed |
| ------------- | ---: |
| haar_random_1 | 4001 |
| haar_random_2 | 4002 |
| haar_random_3 | 4003 |

Draw real and imaginary components independently from a standard normal
distribution and normalize the complex vector. Record the NumPy random
generator and seed.

### T8–T9: random MPS states

| Identifier       | Internal bond dimension | Seed |
| ---------------- | ----------------------: | ---: |
| random_mps_bond2 |                       2 | 5002 |
| random_mps_bond3 |                       3 | 5003 |

Draw real open-boundary MPS tensors with independent standard-normal entries,
using constant internal bond dimension 2 or 3, contract to the little-endian
dense vector, and normalize. This matches the default tensor distribution of
Quimb’s MPS_rand_state construction.

## Benchmark matrix and reproducibility

The base target set has

```text
2 qubit counts x 9 targets = 18 target instances.
```

The full test set has 12 conditions: noiseless, Ballarin, five dephasing
profiles, and five depolarizing profiles. The minimum set has six conditions:

1. noiseless;
2. ballarin_coupled;
3. dephasing_1s_all;
4. dephasing_2s_2q;
5. depolarizing_1s_all;
6. depolarizing_2s_2q.

Consequently, the minimum matrix has 108 result cells and the full matrix has
216 result cells per method, BMPD depth, initialization seed, and one fixed
standard-noise calibration pair. A strength sweep adds a separate calibration
axis for N2/N3; noiseless and Ballarin cells need not be duplicated. Optimization
runs are shared across the test-noise fan-out; a result cell is an evaluation,
not necessarily a separate training run.

Target-generation seeds are fixed by the fixture. For an initialization seed
s, use s for parameter initialization, (s+1) mod 2^64 for optimizer ordering,
and (s+2) mod 2^64 as the root of final noisy evaluation streams. Derive test
streams deterministically by condition. Within each result cell, trajectories
are indexed deterministically and gate-local draws are consumed in fixed circuit
order. Do not consume a shared mutable random generator across cells. Training,
checkpoint selection, and final test streams must be disjoint.

### Reference Krotov profile

The minimal branch’s reference method uses:

| Setting                       | Value                                                                |
| ----------------------------- | -------------------------------------------------------------------- |
| Method                        | Full-batch noiseless Krotov-inspired discrete adjoint                |
| BMPD depth                    | 1, hence two brickwall layers                                        |
| Initial single-qubit U3 layer | Enabled                                                              |
| Initialization                | Independent normal parameters with mean 0 and standard deviation 0.1 |
| Initialization seed           | 11                                                                   |
| Iterations                    | 100                                                                  |
| Step size                     | Constant 0.2                                                         |
| Optimizer seed                | 12                                                                   |
| Tensor-network truncation     | None: no bond cap and zero SVD threshold                             |
| Noisy test trajectories       | 100                                                                  |
| Test root seed                | 13                                                                   |

Additional algorithms must publish an equally explicit profile. If a method
cannot use the same scalar BMPD parameterization, it must demonstrate that its
logical unitary family and initial circuit are equivalent before being included
in a paired comparison.

## Reporting

Use one row per method, target, initialization, ansatz, and test condition.

| Field                             | Meaning                                                                                                                    |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| method                            | Algorithm name and implementation version                                                                                  |
| num_qubits                        | 6 or 12                                                                                                                    |
| target_id                         | One of the nine fixed identifiers                                                                                          |
| noise_id                          | noiseless or a noise-family identifier plus calibration version                                                            |
| noise_calibration                 | Source/date, metric, target errors or learned coefficients, operator set, window, resolved rates, and numerical validation |
| initialization                    | Rule and seed                                                                                                              |
| ansatz                            | BMPD depth, initial U3 choice, and convention                                                                              |
| num_parameters                    | Logical trainable parameter count                                                                                          |
| logical_gate_counts               | Logical one- and two-qubit counts                                                                                          |
| native_gate_counts                | Final materialized counts for the evaluated condition                                                                      |
| optimizer_budget                  | Iterations and method-specific hyperparameters                                                                             |
| train_trajectories_or_shots       | Zero for the base noiseless-training benchmark                                                                             |
| train_fidelity                    | Final training objective                                                                                                   |
| test_noiseless_fidelity           | Same final parameters evaluated without noise                                                                              |
| test_noisy_fidelity               | Mean over fresh trajectories under noise_id                                                                                |
| test_trajectories_or_shots        | Fresh final-evaluation budget                                                                                              |
| noisy_fidelity_standard_deviation | Sample standard deviation when defined                                                                                     |
| noisy_fidelity_standard_error     | Standard error when defined                                                                                                |
| confidence_interval               | Reported level and clipped bounds when defined                                                                             |
| wall_time_seconds                 | End-to-end runtime                                                                                                         |
| notes                             | Deviations, failures, truncation, or compilation details                                                                   |

Store raw per-trajectory fidelities or a checksum-addressed sidecar so that
uncertainty statistics can be recomputed. A failed cell remains a reported
cell with its failure reason; it must not disappear from aggregate results.

## Minimal-branch implementation and usage

The scientific definition above is independent of the Python layout. The
minimal branch provides the following focused components. Hardware-error
calibration is implemented for the local-depolarizing causal and hybrid
experiments only. The standard provider still assigns 6.4e-4 after one-qubit
gates and 5.1e-3 after two-qubit gates to every active operator. At Delta t = 1,
it therefore reproduces the legacy errors above. The general benchmark runner
presets below still use legacy coefficients; use the separate calibrated
experiment commands when reproducing the corrected local-depolarizing tests.

| Requirement                                                | Implementation                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| Target generation and freshness check                      | benchmarks/generate_state_preparation_targets.py                    |
| Validated target selection and conversion                  | benchmarks.state_preparation.targets                                |
| Scalar-parameter BMPD circuit                              | create_brickwall_matrix_product_disentangler_parameterized_circuit  |
| Reference noiseless optimization                           | KrotovBatchMethod and train_krotov_state_preparation_batch          |
| Ten standard noise operator/placement profiles             | StandardNoiseProvider; legacy per-operator coefficients only        |
| Local-depolarizing hardware-error conversion               | benchmarks.state_preparation.calibration; exact generator inversion |
| Other-family calibration and general-runner strength sweep | Not yet implemented; equations and proposed grid specified above    |
| Calibration and finite-step metadata                       | Emitted by calibrated causal/hybrid scripts; not general runner     |
| Ballarin lowering, pruning, and channel                    | materialize_ballarin_circuit and BallarinNoiseProvider              |
| Circuit and uncertainty statistics                         | benchmarks.state_preparation.statistics                             |
| Matrix expansion, checkpoints, and rows                    | benchmarks.state_preparation.runner                                 |
| Competing-method extension point                           | StatePreparationMethod                                              |

Code locations for auditing the discrepancy are
benchmarks/state_preparation/constants.py (legacy coefficients),
benchmarks/state_preparation/noise.py (per-operator assignment),
src/mqt/yaqs/optimization/krotov.py (drift and one-jump sampling), and
experiments/noisy_krotov_causal_study.py (local-depolarizing profile, dt = 1).
Calibrated configurations have distinguishable identities and provider
checksums; changing the calibration rejects resume. Future calibrated families
must retain this separation. Retain the original manifests and results; do not
silently relabel or resume legacy runs as calibrated experiments.

### Corrected local-depolarizing reruns

The causal rerun preserves the original q6 critical-TFIM target, depth-1/2/3/4
noiseless warm start, five arms, three training replicates, 200 updates,
three training trajectories, 12 validation trajectories, and 500 held-out
paired trajectories per replicate. Only noise calibration changes; the
optional worker count changes execution, not seeds or the scientific setup.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.noisy_krotov_causal_study \
  --noise-calibration hardware_average_error \
  --workers 3 \
  --output-root state_preparation_results/q6-causal-calibrated

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run python -m experiments.noisy_krotov_hybrid_smoke \
  --source-root state_preparation_results/q6-causal-calibrated \
  --output-root state_preparation_results/q6-hybrid-calibrated
```

The hybrid command requires a completed canonical causal source with a matching
pretraining checksum. An original legacy causal output can also be the source:
pretraining is noiseless and identical. This small diagnostic uses 10 online
and 10 batch updates, three training trajectories, resampling, cross updates,
and 200 fresh paired test trajectories, preserving the historical seeds.
Its held-out interval measures evaluation uncertainty for one training run,
not variation across independent training replicates. Existing hybrid outputs
are never overwritten.

### General benchmark runner (legacy coefficient presets)

Install the package and verify the fixed targets from the repository root:

```bash
uv sync
uv run python benchmarks/generate_state_preparation_targets.py --check
```

Inspect the fully resolved minimum matrix without creating output:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset minimum \
  --dry-run
```

Run the bounded 12-row smoke configuration or the canonical matrices:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset smoke \
  --output-dir state_preparation_results/smoke

uv run python -m benchmarks.state_preparation.runner \
  --preset minimum \
  --output-dir state_preparation_results/minimum

uv run python -m benchmarks.state_preparation.runner \
  --preset full \
  --output-dir state_preparation_results/full
```

Resume an interrupted run with the same scientific configuration:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset minimum \
  --output-dir state_preparation_results/minimum \
  --resume
```

The runner refuses an incompatible retained manifest. The overwrite option
removes only benchmark-owned manifests, rows, and checkpoints; unrelated files
in the output directory remain untouched. Results are stored in results.jsonl,
with one deterministic run identifier per matrix cell, raw trajectory
fidelities, uncertainty statistics, logical and native circuit counts, seeds,
and a shared content-addressed training checkpoint.

Repeated num-qubits, target-id, noise-id, method, bmpd-depth, and
initialization-seed options select subsets or extensions of a preset. Explicit
optimizer-iterations, optimizer-step-size, test-trajectories, and tjm-dt options
override its budgets.

The bundled registry contains the Krotov reference method. A competing
algorithm implements StatePreparationMethod and is supplied under its stable
method identifier to run_benchmark; the runner then applies the same target,
ansatz, initialization, test channels, seed separation, and reporting path.
No general experiment framework or historical Phase II machinery is required.
