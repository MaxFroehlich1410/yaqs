# Twelve-qubit noiseless continuation

This diagnostic continues our Gaussian, random-MPS bond-three and complex-Haar
targets from the frozen campaign's **unpruned depth-four construction**.
TFIM is excluded. It changes depth stopping only: no pruning, noisy refinement,
competitor training, or changes to the current comparison's files occur.

Each new depth retains the original circuit family, seed 300, extension seed
`seed + 1000 + depth`, extension scale 0.001, 100 online Krotov sweeps at 0.7,
100 safeguarded L-BFGS updates, and best-iterate restoration. The default backend
is untruncated MPS; `--backend statevector` trains every new extension with exact
statevectors. The previous unpruned checkpoint initializes the next depth.
Every endpoint is replayed by a fresh statevector evaluator; disagreement with
the saved optimizer score above 1e-9 stops the run. For vector training this is
a checkpoint-consistency check, not an independent-backend comparison.

Let R be the latest material ideal-fidelity record. A new depth resets patience
only if its best ideal fidelity F satisfies

```text
F - R > max(1e-12, 0.01 * min(R, 1 - R)).
```

This means a 1% relative improvement in fidelity at small overlaps, or a 1%
relative reduction in infidelity near one, above a numerical floor. Gains below
the threshold accumulate relative to R. Any better checkpoint remains eligible
for the actual winner, even if it does not reset patience; ties keep the earlier
depth. The baseline is the best of original depths 1–4, with fresh patience for
the new extensions. Historical noisy patience cannot terminate construction.

Stop after three unsuccessful new extensions, depth 16, or six additional CPU
hours per target (checked between depths). Depth and CPU limits are recorded
as **budget exhausted**, distinct from **stagnated**. Stagnation is an empirical
stopping heuristic, not proof of a global optimum or sufficient fidelity.
Deployment adds a 24-hour wall limit and 16-GiB memory limit per worker; a killed
worker remains an incomplete resource-limited run with recoverable checkpoints.

The new run manifest authenticates the old manifest, construction seals,
target and numerical sources. Original histories are referenced, not copied.
New fidelity curves and parameter snapshots are stored in sealed NPZ files,
with JSON references, optimizer diagnostics, timings and exact replay errors.
Resume reuses only authenticated complete stages and replays stopping decisions.
The manifest distinguishes the original parent backend from the selected
execution backend. Version 2 requires a fresh output directory; old version-1
results retain their frozen driver and must not be overwritten or mixed with it.

```sh
python -m experiments.topdown_noiseless_continuation \
  --parent-campaign /path/to/original/campaign \
  --output /path/to/separate/continuation \
  --target gaussian_mu0p5_sigma0p1
```

The three target chains may run independently. Checkpoints within a chain are
sequential. Use a fresh source snapshot and leave active comparison workers alone.

## Matched Gaussian and Haar statevector continuations

The 2026-09-29 experiment reruns Gaussian and complex Haar only. Each starts at
the exact same authenticated, unpruned depth-four checkpoint used by the earlier
MPS continuation. Depths 5 through 16 are trained independently in statevector
format, carrying this new run's own best parameters between depths. This is
training, not just evaluation of the MPS-trained endpoints; depths 1 through 4
remain explicitly labelled reused MPS history. No optimizer, step size, iteration
budget, seed, pruning rule, noise setting or ideal stopping rule changes.

```sh
python -m experiments.topdown_noiseless_continuation \
  --parent-campaign /path/to/original/campaign \
  --output /path/to/separate/statevector/results \
  --target gaussian_mu0p5_sigma0p1 --backend statevector
```

Repeat with `--target haar_random_1`. Compare fidelity at each depth, the best
checkpoint, stopping reason, measured CPU/wall time and worker peak memory.
The matched host assignments use leonhard-16 for Gaussian and leonhard-13 for
Haar, with one numerical thread each, as in the archived MPS continuations.
These are separate-time measurements on shared hosts, not isolated-machine
performance trials. Backend roundoff can change nonlinear optimization paths;
do not assume identical final parameters or fidelity from exact representation
equivalence alone. Original results remain preserved.
