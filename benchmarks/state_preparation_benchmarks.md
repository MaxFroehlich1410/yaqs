# State-Preparation Benchmark Definition

This document is the normative scientific definition of the state-preparation
benchmark. It fixes the mathematical experiment, targets, noise channels, and
reporting requirements while keeping the implementation in the minimal branch
small. Names of Python functions appear only in the implementation-status
section and are not part of the mathematical definition.

## Scientific experiment

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

The standard dephasing and depolarizing profiles use these strengths:

| Gate context              | Strength of each jump operator |
| ------------------------- | -----------------------------: |
| After a single-qubit gate |                0.064% = 6.4e-4 |
| After a multi-qubit gate  |                 0.51% = 5.1e-3 |

A listed strength belongs to each jump operator; it is not a total channel
strength divided among the operators. For example, single-site depolarizing
noise after a one-qubit gate has X, Y, and Z processes, each with strength
6.4e-4. YAQS interprets the values as Lindblad-process rates used with the
reported circuit-TJM time step. Any other parameterization must document its
conversion from this per-operator convention.

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
do not additionally apply the common 6.4e-4 or 5.1e-3 rates. The compatibility
identifier contains “coupled” because both local channels are attached to the
same native gate, not because their Pauli outcomes are correlated.

### N2: Dephasing noise

The site-support choices are:

- single_site: one Z process on every affected qubit;
- two_site: one correlated ZZ process on the qubit pair of a multi-qubit gate;
- single_site plus two_site: both process sets.

The placement choices are single-qubit gates, multi-qubit gates, or all gates.

| Identifier         | Site support              | Placement          | Per-operator strengths                                            |
| ------------------ | ------------------------- | ------------------ | ----------------------------------------------------------------- |
| dephasing_1s_1q    | single_site               | Single-qubit gates | 6.4e-4                                                            |
| dephasing_1s_2q    | single_site               | Multi-qubit gates  | 5.1e-3                                                            |
| dephasing_1s_all   | single_site               | All gates          | 6.4e-4 after 1q; 5.1e-3 after multi-qubit gates                   |
| dephasing_2s_2q    | two_site                  | Multi-qubit gates  | 5.1e-3                                                            |
| dephasing_1s2s_all | single_site plus two_site | All gates          | 6.4e-4 after 1q; 5.1e-3 for every process after multi-qubit gates |

For a single_site profile after a multi-qubit gate, add one Z process for each
gate qubit. The minimum profile uses dephasing_1s_all and
dephasing_2s_2q.

### N3: Depolarizing noise

A single-site depolarizing set contains X, Y, and Z on every affected qubit.
The frozen two-site set contains exactly

```text
XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
```

Each of the nine operators receives the full applicable per-operator strength.
Operators with identity factors, such as XI or IS, are excluded. A combined
single-site plus two-site profile after a multi-qubit gate contains the six
single-site processes, three on each gate qubit, and all nine two-site
processes.

| Identifier            | Site support              | Placement          | Per-operator strengths                                            |
| --------------------- | ------------------------- | ------------------ | ----------------------------------------------------------------- |
| depolarizing_1s_1q    | single_site               | Single-qubit gates | 6.4e-4                                                            |
| depolarizing_1s_2q    | single_site               | Multi-qubit gates  | 5.1e-3                                                            |
| depolarizing_1s_all   | single_site               | All gates          | 6.4e-4 after 1q; 5.1e-3 after multi-qubit gates                   |
| depolarizing_2s_2q    | two_site                  | Multi-qubit gates  | 5.1e-3                                                            |
| depolarizing_1s2s_all | single_site plus two_site | All gates          | 6.4e-4 after 1q; 5.1e-3 for every process after multi-qubit gates |

The minimum profile uses depolarizing_1s_all and
depolarizing_2s_2q.

### Exact channels and finite-trajectory estimates

The channel definitions are exact. Direct Ballarin sampling selects an exact
product-channel branch. Standard dephasing and depolarizing evaluation samples
circuit-TJM no-jump and jump trajectories for the stated process sets.

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
216 result cells per method, BMPD depth, and initialization seed. Optimization
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

| Field                             | Meaning                                                  |
| --------------------------------- | -------------------------------------------------------- |
| method                            | Algorithm name and implementation version                |
| num_qubits                        | 6 or 12                                                  |
| target_id                         | One of the nine fixed identifiers                        |
| noise_id                          | noiseless or one frozen noise identifier                 |
| initialization                    | Rule and seed                                            |
| ansatz                            | BMPD depth, initial U3 choice, and convention            |
| num_parameters                    | Logical trainable parameter count                        |
| logical_gate_counts               | Logical one- and two-qubit counts                        |
| native_gate_counts                | Final materialized counts for the evaluated condition    |
| optimizer_budget                  | Iterations and method-specific hyperparameters           |
| train_trajectories_or_shots       | Zero for the base noiseless-training benchmark           |
| train_fidelity                    | Final training objective                                 |
| test_noiseless_fidelity           | Same final parameters evaluated without noise            |
| test_noisy_fidelity               | Mean over fresh trajectories under noise_id              |
| test_trajectories_or_shots        | Fresh final-evaluation budget                            |
| noisy_fidelity_standard_deviation | Sample standard deviation when defined                   |
| noisy_fidelity_standard_error     | Standard error when defined                              |
| confidence_interval               | Reported level and clipped bounds when defined           |
| wall_time_seconds                 | End-to-end runtime                                       |
| notes                             | Deviations, failures, truncation, or compilation details |

Store raw per-trajectory fidelities or a checksum-addressed sidecar so that
uncertainty statistics can be recomputed. A failed cell remains a reported
cell with its failure reason; it must not disappear from aggregate results.

## Minimal-branch implementation and usage

The scientific definition above is independent of the Python layout. The
minimal branch implements it with the following focused components:

| Requirement                               | Implementation                                                     |
| ----------------------------------------- | ------------------------------------------------------------------ |
| Target generation and freshness check     | benchmarks/generate_state_preparation_targets.py                   |
| Validated target selection and conversion | benchmarks.state_preparation.targets                               |
| Scalar-parameter BMPD circuit             | create_brickwall_matrix_product_disentangler_parameterized_circuit |
| Reference noiseless optimization          | KrotovBatchMethod and train_krotov_state_preparation_batch         |
| Ten standard noise profiles               | StandardNoiseProvider                                              |
| Ballarin lowering, pruning, and channel   | materialize_ballarin_circuit and BallarinNoiseProvider             |
| Circuit and uncertainty statistics        | benchmarks.state_preparation.statistics                            |
| Matrix expansion, checkpoints, and rows   | benchmarks.state_preparation.runner                                |
| Competing-method extension point          | StatePreparationMethod                                             |

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
