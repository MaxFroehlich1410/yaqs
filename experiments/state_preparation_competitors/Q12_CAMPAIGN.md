# Twelve-qubit logical-noise campaign

Prepared on 2026-09-28. **RobustState uses 400 bases × 1,024 shots per noisy
update. Ballarin is excluded. Preparation and staging do not launch experiments.**

| Method                                                           | Gaussian | TFIM critical | Random MPS, bond 3 | Complex Haar 1            | Training jobs                   |
| ---------------------------------------------------------------- | -------- | ------------- | ------------------ | ------------------------- | ------------------------------- |
| Ours: B + online Krotov / noiseless L-BFGS + weighted refinement | Yes      | Yes           | Yes                | Yes                       | 4 sequential depth chains       |
| Fontana                                                          | Yes      | Yes           | Yes                | Outside real-state ansatz | 12 independent depth candidates |
| RobustState, sampled feedback                                    | Yes      | Yes           | Yes                | Yes                       | 16 independent depth candidates |

Eleven method–target groups, up to 44 depth candidates, become **32 training
jobs plus seven competitor selection/final-assessment jobs**. Our four chains
include their own selection and final assessment. Depths are 1–4 and the seed
is 300. One seed and four depths are not a converged publication comparison.

## Frozen science

All methods train and are assessed with the existing IBM **logical-gate**
Pauli law, including one- and two-qubit windows. Targets come from the existing
twelve-qubit JSON fixture. Competitors receive no new pruning or optimizer.

- Ours reuses `topdown_mps.run_case` with untruncated MPS: 100 online + 100
  L-BFGS construction updates, best noiseless iterate, original pruning B,
  then 100 online + 100 batch noisy updates with 100 raw training histories.
  Empirical no-jump compression and **both cross-trajectory weights** stay
  unchanged. The unpruned best parent initializes the next depth. Existing
  validation/patience selection and final noisy-iterate retention stay intact.
- Fontana reuses `fontana_candidates`: independently initialized depths,
  200 ideal and 200 noisy BFGS iterations, existing evaluation caps,
  tolerances and endpoint policy, exact density checkpoint recomputation.
  The real RY/CX ansatz is not extended to complex Haar targets.
- RobustState reuses `robuststate_candidates`: 500 ideal + 50 noisy Adam
  updates at 0.005, continuous moments, raw Pauli-shadow feedback, final
  iterate. Each feedback call samples 400 distinct random bases without
  replacement, with 1,024 shots per basis: **20,480,000 shots per candidate**.
  This budget is resource-limited, **not established as sampling-converged**
  by the prior smoke tests. No exact-feedback substitution or readout noise.

Validation uses 32,768 independent histories per candidate. Selection is locked
before four fresh final batches of 8,192 histories are sampled. Seeds have
separate training, validation and test roles; server placement never affects
them. Jump counts, raw-sample standard errors and 95% normal intervals are
reported. There is no conditioning on jumps or silent resampling. Nonempty
circuits with zero-jump banks are flagged. Competitor validation ties prefer
fewer logical two-qubit gates, then the stable candidate identifier.

The competitor adapters retain the already documented adaptations to our
targets and logical noise. This is not a reproduction of each paper's original
hardware/noise experiment. Our compiled gate counts remain resource diagnostics;
they do not determine the logical noise windows.

## Distribution

The shared NFS queue uses **leonhard-13 and leonhard-16**, strict SSH host-key
checks and the existing numerical environment. Both had 144 logical CPUs and
substantial free memory at the read-only preflight; leonhard-12/-14/-15 rejected
login. Capacity is rechecked at launch and before dispatch; it is not reserved.

At most 16 workers per host, one numerical thread each, niceness 10. Admission
reserves 10% of total RAM and 25% of estimated physical cores, subtracts current
load, and budgets 16 GiB per worker. Each child has a 16 GiB RSS guard and a
seven-day operational wall-time ceiling. Exceeding either is recorded as an
incomplete run, never as low fidelity or successful early stopping.

Independent jobs use a longest-estimated-first heuristic, with ready finalizers
taking priority. Initial host preferences alternate the ordered jobs so the
first supervisor cannot fill its slots with all the longest candidates; nodes
can subsequently steal remaining work. Nodes take new jobs as slots free up. Atomic NFS directory
creation prevents duplicate claims. Our warm-start chains remain sequential.
This schedule avoids a static partition that leaves one node waiting behind
a long task; unknown BFGS iteration counts and shared memory bandwidth mean
that optimal wall time or linear parallel speedup cannot be guaranteed.

All sources, numerical package versions, fixtures, budgets and seeds are
authenticated. Histories are stored once in compact arrays. Each job has a
fresh child process, stage CPU timing, wall-time/peak-RSS/exit-status receipts.
Search costs include discarded candidates. Parallel campaign wall time must
not be confused with each method's compute cost.

## Commands and artifacts

From the repository root, use the established scientific environment:

```sh
export PYTHONPATH=.:src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=.venv-iqsp-tn/bin/python
CAMPAIGN=output/q12-logical-matrix-20260928

# New manifest; the default verb, when omitted, is only "plan".
$PYTHON -m experiments.state_preparation_competitors.q12_cluster prepare --root "$CAMPAIGN"
$PYTHON -m experiments.state_preparation_competitors.q12_cluster plan --root "$CAMPAIGN"

# Copy and verify an immutable snapshot. Still no training.
$PYTHON -m experiments.state_preparation_competitors.q12_cluster stage --root "$CAMPAIGN"

# Only after the user authorizes starting the experiments:
$PYTHON -m experiments.state_preparation_competitors.q12_cluster launch --root "$CAMPAIGN"
```

The launcher is `q12_cluster.py`; scientific job adapters are in
`q12_campaign.py`. The local campaign contains `manifest.json`, its checksum,
`capacity-plan.json`, the source bundle and staging receipt. Shared deployment:
`/Home/varioper/froehlich/yaqs-checks/q12-logical-matrix-20260928/`, containing
immutable `source/` and result directory `campaign/`. Supervisors detach on the
servers, so closing the laptop does not terminate them. Run the `status` action
against that remote campaign to inspect progress.

`campaign/jobs/JOB_ID/` contains competitor `training/` (QPY, compact arrays,
seal), `validation/` and `result/`. Finalizers add `selection_locked.json`
and `test_00/`–`test_03/`. Our chains retain `chain/depth_XX/`, their selection
lock, final batches and final record. Logs, ownership and resource receipts
are in `logs/`, `claims/`, `done/` and `failed/`.

Completed jobs are skipped; adapters authenticate and reuse sealed stages.
Interrupted competitor training restarts from its original seed, avoiding a
parameter-only restart with lost Adam/BFGS state. Failures are not automatically
retried, and ownership does not expire merely because a worker is slow. Before
requeuing an interrupted job, verify its worker/supervisor has stopped, archive
the old logs/receipt, and remove only that job's failed/claim markers. Inspect
stale supervisor locks similarly. Source/configuration changes require a new
campaign; never overwrite an active snapshot.
An existing `launch.json` also blocks accidental duplicate launch; inspect and
archive that receipt before intentionally relaunching stopped supervisors.
