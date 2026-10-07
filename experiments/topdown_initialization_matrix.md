# Twelve-qubit initialization comparison: complete target coverage

Compare historical near-zero and independent Haar single-qubit product starts
for **all nine** hardcoded twelve-qubit targets: Gaussian; TFIM ferromagnetic,
critical and paramagnetic; three complex Haar fixtures; and random MPS with bond
dimensions two and three. Use predetermined optimizer seeds 300, 400 and 500.
This is 54 matched cases. Reusing the six authenticated near-zero/random-product
bond-three MPS runs requires **48 new training jobs**. The earlier largest-basis
arm is outside this two-arm comparison.

## Frozen numerical procedure

The new driver delegates every numerical update to
`topdown_random_mps_initialization.run_case`. Only the target identifier and
initialization arm vary. Exact statevectors, the original initial U3 layer and
brickwall architecture, initial/extension random draws, 100 online Krotov sweeps,
up to 2,000 safeguarded L-BFGS updates per depth, rate 0.7, ten curvature pairs,
Armijo 1e-4, up to twelve backtracking reductions, and best-parameter retention
remain identical to the completed random-MPS experiment. Each new depth starts
with its own preceding best parameters. Curvature resets once at each new depth
and remains in memory throughout that continuous call.

Grow from depth one, stopping at the first accepted **99% ideal-fidelity**
crossing, depth 16, or six accounted CPU hours checked between completed depths.
There is no depth-patience stop in this bounded initialization diagnostic.
A cap is budget exhaustion, not convergence or an unattainability claim. Compare
Haar fidelity at matched depths and CPU even if neither arm reaches 99%.
Neither noise nor pruning is applied. These runs do not establish 99.9%.

Train every case from scratch except the six explicitly authenticated reference
runs. Shorter historical optimization budgets are not matched controls. Reference
authentication checks the fixture/target and numerical source hashes, exact
settings, initial parameters, every saved depth and inherited input, final seals,
and successful supervisor commands/exits. Run this authentication on the original
Leonhard numerical platform; bitwise random draws can differ slightly across
platforms. Existing histories remain read-only and appear by reference in the new
plan rather than being copied into duplicate training archives.

## Separate overlap-distribution diagnostics

For each target, sample **1,000** independent Haar single-qubit product states
with diagnostic seed 730001. This seed is separate from the training seeds and
the samples never select a training start. Store only polar/azimuth angles and
overlaps, once in a sealed binary archive. Report mean with Monte Carlo standard
error, median, quantiles and the fraction exceeding the zero-state overlap.
These diagnostics concern the product layer before entangling gates; each
training initialization also records its actual complete-circuit fidelity.

For any fixed normalized target the exact random-product mean overlap is
`2**(-12)`. This exceeds the zero-state fidelity only when the latter is below
that value; no universal initialization advantage or optimization guarantee is
assumed. TFIM therefore provides a useful case with a favorable zero-state start.

## Planning and explicit execution

```sh
python -m experiments.topdown_initialization_matrix \
  --output output/q12-initialization-matrix-20261005 \
  --reuse-random-mps /path/to/original/q12-random-mps-initialization-20261005
```

Planning writes explicit commands and launches nothing. Without the reference
option the plan contains 54 new cases; with validated references it contains 48.
One job is explicit:

```sh
python -m experiments.topdown_initialization_matrix --action run \
  --target-id tfim_critical --arm random_product --seed 300 \
  --output output/q12-initialization-matrix-20261005/jobs/tfim_critical__random_product__seed300
```

Use one numerical thread per worker, nice priority 10, an eight-hour wall guard
and a 16-GiB memory guard through the existing `bounded_probe` supervisor.
Capacity checks must permit simultaneous jobs with at least 64 GiB free RAM
reserved per selected node even if every worker reaches its memory guard, plus
substantial physical CPU headroom. Inspect existing load/processes and require
strict verified SSH host keys. Keep each target/seed pair on the same node and
spread pairs over nodes. Claims are atomic and must prevent duplicate launch.
Guard termination is incomplete; completion requires sealed final records,
wrapper completion hashes and a successful supervisor exit receipt.

Frozen source, manifests, launch claims, logs, source hashes and target hashes
must be preserved. Within-depth live snapshots lack optimizer curvature and
cannot resume an interrupted call. Only complete sealed depths may be reused.
No existing experiment or environment is changed or interrupted.

## Reporting

For each target and initialization report success count across three seeds,
CPU/wall time to matched fidelity, depth, logical two-qubit rotations, initial
overlap, fidelity-versus-depth/CPU curves and resource usage. Record failures,
budget exhaustion and continuing progress separately. Recheck original sealed
histories with exact statevector replay. Count reference costs as historical
preparation costs, without charging them as newly executed matrix jobs.
Interpret mean-overlap advantages and observed optimization speed separately.
