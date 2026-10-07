# Twelve-qubit Haar refinement at fixed depths 66 and 87

Two independent runs reuse the authenticated growth checkpoints from
`output/q12-haar-capacity-20260930/results/grow`. The target is the unchanged
12-qubit complex `haar_random_1` fixture. The depths start at ideal fidelities
0.9511649849365087 and 0.980503241892958, respectively. Their circuits contain
2,178 and 2,871 logical two-qubit rotations (6,570 and 8,649 parameters).

Each run makes one continuous call to the existing safeguarded L-BFGS with
`online=0`, rate 0.7, history length 10, Armijo coefficient 1e-4 and at most
12 backtracking reductions. Curvature history resets once at the branch start
because the saved parent does not preserve it. The new call retains its history.
Circuit structure and parameter coordinates stay fixed; there is no pruning,
noise, circuit growth, random restart, or additional online Krotov training.

The budget is 2,000 additional attempted updates (failed line searches count).
Record the first accepted iterate reaching 0.99, then continue until the first
iterate reaching 0.999 or the update budget. These are exact comparisons before
display rounding. Always return the best iterate, including the starting point.
Resource guards are 6 wall hours and 16 GiB per worker; a guard interruption is
an incomplete experiment, never convergence or a completed budget-limited result.

Both runs use statevectors, one numerical thread, low scheduling priority, and
separate available Leonhard hosts. Before dispatch, inspect current CPU load,
available RAM and free disk, preserve substantial headroom, and atomically claim
each job. Frozen sources, target hashes, parent seals, exact commands, capacity
snapshots, logs and supervisor receipts accompany the run. Existing work is not
interrupted. Two concurrent workers exhaust the independent work in this protocol;
splitting an L-BFGS call into restarting chunks would change the experiment.

```sh
python -m experiments.topdown_haar_refinement \
  --parent /path/to/q12-haar-capacity-20260930/results/grow \
  --output /path/to/new/results/depth_066 --depth 66
python -m experiments.topdown_haar_refinement \
  --parent /path/to/q12-haar-capacity-20260930/results/grow \
  --output /path/to/new/results/depth_087 --depth 87
```

The driver independently replays the starting and final circuits. Report the
first 99% and 99.9% crossings, best fidelity, gradient norm, update/evaluation
counts and CPU/wall times. Save parameters and optimizer histories once in
`training/arrays.npz`; JSON records reference this compact archive. Incoming
accepted-iterate snapshots appear every 50 gradient calls in `live/` and
`progress.json`. They lack L-BFGS history and must never be resumed as a continuous
call. Incomplete calls require a new output directory and explicit replay from
the original parent; completed sealed training can be finalized without retraining.

Compare with the earlier growth run's first completed-depth 99% crossing at
depth 107 (99.0165%). Report refinement-only costs and total construction plus
refinement costs separately, counting the shared historical depth-16 cost once
per hypothetical preparation path and excluding growth beyond the selected depth.
Plots should show fidelity versus added updates and CPU time. Non-attainment
with continuing improvement differs from a plateau; neither establishes a global
fidelity ceiling. One target and initialization do not determine minimal depth.
Fewer gates at high ideal fidelity would motivate a separate matched noisy test.
