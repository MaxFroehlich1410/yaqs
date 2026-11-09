# Small-system fallback experiment

The default pilot runs 72 configurations, each with three matched trajectories:
uncorrected compression, the original scalar correction, and the correction
with the manuscript fallback. The experiment uses the actual BUG/TDVP dispatch
and production conservation functions.

| Parameter | Default |
| --- | --- |
| Models | TFIM, XXZ (Delta = 0.5), XXX, Haldane-Shastry |
| Sites | 6 |
| Bond caps | 1, 2, 4 |
| Initial states | Tilted Neel, domain wall, seeded random product |
| Time steps | 0.02, 0.1 |
| Final time | 2 |
| Integrator | BUG |
| Conservation tolerance | 1e-13 |
| Krylov tolerance | 1e-14 |
| SVD threshold | 0 (only the hard cap forces truncation) |

From the repository root, with YAQS and its test dependencies installed:

```bash
python paper/bug-mps-benchmarks/spin_conservation/fallback_pilot.py \
  --output /path/to/fallback-pilot --workers 2
python paper/bug-mps-benchmarks/spin_conservation/fallback_replay.py \
  /path/to/fallback-pilot
python paper/bug-mps-benchmarks/spin_conservation/fallback_report.py \
  /path/to/fallback-pilot
```

The plotting command needs the optional `matplotlib` package. Pass `--no-plots`
to generate tables and the text report without it. If using an editable source
checkout without an installed YAQS package, set `PYTHONPATH=src`.

Reusing the same output directory resumes completed configurations. The runner
refuses to mix a different grid or changed source with an existing manifest.
Use a new directory after changing the code. Each triple is checkpointed
atomically; an interrupted triple is rerun. A failed triple writes `error.json`
and makes the command exit with a nonzero status.

## Measurements

- The primary rate is fallback attempts divided by correction calls.
- The conditional rate excludes calls already within tolerance.
- BUG calls the correction twice per timestep; TDVP calls it once.
- Scalar-only rejections are reported separately from actual fallback calls.
- Scan recovery, full-frame recovery, and unresolved outcomes have separate counts.
- Energy drift and infidelity are recorded after every complete timestep.
  Before/after-correction energy maxima are additional, separate diagnostics.
- Timings separate production evolution from measured observer overhead. These
  single-run timings are descriptive: compilation, process contention and
  divergent trajectories prevent interpreting them as a precise overhead ratio.

The full-state reference uses an independently assembled sparse Hamiltonian
and `expm_multiply`. Every matrix entry is checked against the MPO's physical
ordering. The reference starts from the actual prepared MPS, and reference
energy drift is recorded as a numerical check.

Initial states retain the manuscript fixture's seeded padding of size 1e-10,
with starting bond dimension `min(cap, 4)`. This avoids relying on exact product
manifold degeneracies to seed BUG's basis expansion. The same prepared tensors
are copied into all three arms. The physical difference from the unpadded state
is recorded; comparisons across caps should account for this preparation detail.

Every fallback entry stores its compressed tensors before correction, center,
orientation, step and half-sweep call, target, failure reason of the scalar
direction, recovery stage, energy residual and displacement. NPZ archives use
numeric arrays only and load with `allow_pickle=False`.

The replay command applies the scalar-only and fallback modes to independent
copies of each snapshot. It checks rejection by the original scalar method,
reproducible recovery counters, energy residuals, preserved bond sizes and gauge,
and exact rollback for unresolved cases. Snapshot targets in the pilot are
always the initial energy, not artificially chosen failure targets.

## Follow-up runs

Repeat only the eventful pilot configurations with a looser constraint tolerance:

```bash
python paper/bug-mps-benchmarks/spin_conservation/fallback_pilot.py \
  --eventful-from /path/to/fallback-pilot --conserve-tol 1e-11 \
  --output /path/to/fallback-tolerance --workers 2
```

Extend the grid with `--lengths 4 8`, `--seeds 20260812 20260813 20260814
20260815 20260816`, `--initials random_product random_entangled`,
`--integrator tdvp`, or a longer `--total-time`. The CLI also accepts explicit
model, bond-cap and time-step subsets. The final time must be an integer
multiple of each time step.

If the pilot sees no events, extend time and include initially entangled states
before increasing the timestep further. Do not present a zero event count on
this finite grid as a universal guarantee. Calls within a trajectory are
correlated; uncertainty across initial-state seeds is more informative than
treating every compression as an independent Bernoulli trial.

The report pools numerators and denominators within each model/cap cell. It
does not average per-run percentages. Since smaller time steps generate more
calls at fixed final time, also inspect the disaggregated `runs.csv`.

## Output files

- `manifest.json` and `source.tar.gz`: full parameters, versions, source hashes,
  and a copy of the implementation used, including local uncommitted changes.
- `cases/*/result.json`: all three trajectories, counts and events.
- `cases/*/initial.npz`: the exact prepared initial MPS.
- `cases/*/fallback/event-*.npz`: every pre-fallback MPS.
- `cases/*/replay.json` and `replay_summary.json`: replay checks.
- `runs.csv`, `groups.csv`, `summary.json`, `report.md`: results for analysis.
- `fallback_frequency.{png,pdf}` and `accuracy_comparison.{png,pdf}`: figures.

## Normalized stress grid

`--models stress` selects 17 additional parameter settings:

- Chord-distance power-law XXZ: alpha = 1, 1.5, 2, 3 and Delta = 1, 2.
  Distance is `(L/pi) sin(pi |i-j|/L)` on a ring. Alpha = 2, Delta = 1
  reproduces the Haldane-Shastry Hamiltonian.
- Open J1-J2 Heisenberg: J1 = 1, J2 = 0.3, 0.5, 1.
- Open tilted Ising: `-sum ZZ - 1.05 sum X - hz sum Z`, hz = 0.2, 0.5, 1,
  in Pauli operators.
- Open nearest-neighbor XXZ: Delta = 2, 4, 8, in spin-half operators.

The MPO uses exact finite-state channels without compression; an independent
spin-flip assembly supplies the full-state reference. `--width-per-site 1`
multiplies both operators by `L / (Emax - Emin)`, using the exact small-system
spectrum. No constant is added. Thus T and dt use the same bandwidth-per-site
energy unit for every model and size. Each result records raw extrema, scale,
physical parameters, and an all-matrix-entry agreement check.

```sh
python fallback_pilot.py --output /path/to/stress \
  --models stress --lengths 6 8 --caps 2 4 8 --total-time 8 \
  --initials domain_wall tilted_neel random_entangled \
  --dts 0.02 0.1 --seeds 20260812 --initial-chi 2 --width-per-site 1 \
  --workers 4
python fallback_replay.py /path/to/stress
python fallback_report.py /path/to/stress
```

This is 612 configurations and 1,836 matched BUG trajectories. The initial
bond dimension is fixed at 2 for every cap: the seeded random entangled state
is identical across caps, as are the padded product states. This is one random
state per size, not a random-state ensemble. `initial_vector_sha256` makes
cross-cap identity checkable. The dense reference starts from those actual
tensors; there is no product-state reference for the random entangled state.

`slices.csv` pools fallback-enabled calls by each experimental factor.
`runs.csv` retains every factor for joint filtering. Call frequencies at the
two timesteps have different denominators and are reported separately there;
pooled percentages are descriptive, not an estimate over independent trials.
Event stages distinguish scan, full frame, infeasible, no minimizer, solver
failure, and gauge failure. Exact replay checks stage counts, residuals,
original gauge and dimensions, and bit-exact rollback on failure.
