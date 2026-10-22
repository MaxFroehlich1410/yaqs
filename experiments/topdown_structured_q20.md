# Structured twenty-qubit B/G pilot with final MPO assessment

The two ideal-preparation failures and their verified SVD repair are documented
in the [numerical failure analysis](topdown_svd_failure_analysis.md).

This is a fresh exploratory scaling campaign on Leonhard-13/16. It contains all
six structured fixed targets, one initial optimizer seed (300), and both requested
policies. Each target has one shared screened-product noiseless parent. Neither
policy is selected using the final noisy results. Existing campaigns, processes,
and source snapshots are not modified.

## Frozen numerical protocol

- Targets: Gaussian; TFIM ferro/critical/para; random MPS bond 2/3, at 20 qubits.
- Preparation: zero-protected best of 1,000 product starts, screening batch size 8;
  the original scalar BMPD circuit and extension stream. Each depth uses 100 online
  updates at rate 0.7 plus up to 2,000 continuous safeguarded L-BFGS updates.
  Stop on 99% ideal fidelity, depth 30, or a 14-hour preparation CPU allowance
  checked between complete depths. Setup/screening CPU is charged. Both policies
  consume the same sealed best parent.
- B: original simultaneous signed deletion cutoff 0.01; 100 online and 100 batch
  noisy updates at rate 0.2. Retain the earliest best fixed-training-bank iterate,
  including the initial pruned parameters, matching the current best-retention
  continuation convention.
- G: unchanged adaptive scalar G, cutoff 0.01 and cumulative ideal loss allowance
  0.05, with no repair. Run 100 online updates, then at most their CPU again in
  extra online updates. Stop earlier on a decline greater than 1e-12 or five
  consecutive gains below 1e-5. Complete and charge a crossing/stopping update.
  Retain the earliest best training iterate.
- Each branch freezes 100 unconditional historical logical IBM Pauli histories,
  compressing no-jump duplicates with their multiplicities. No jump coverage
  causes an explicit failure, never resampling. The cross-trajectory update is
  unchanged. Every complete noisy iterate is saved and sealed. The G CPU
  allowance measures online update plus fidelity-evaluation kernels; persistence,
  sampling/setup, and full stage CPU/wall are recorded separately by receipts.
- Preparation, pruning, and noisy updates use untruncated MPS propagation without
  bond caps. This avoids dense trajectory tapes at twenty qubits. The optional
  G backend adapter is regression-tested against the original dense policy.
- Final noisy scores: deterministic forward MPO at both the pruned starting
  parameters and the selected refined parameters for B and G. These scores never
  choose an iterate. No dense density matrix or final trajectory bank is formed.
  MPO SVD tolerance is zero and there is no bond cap or automatic fallback.

## Fixed targets and numerical representation

The hard-coded binary arrays remain authoritative and unchanged. Random-MPS
native tensors are loaded directly. Other target MPS representations use a fixed
absolute TT-SVD tail budget of 5e-13, followed by phase-preserving normalization.
Every representation is independently contracted back to its full vector and
must agree with the fixed array in absolute vector 2-norm to at most 1e-12.
This is an explicitly bounded target-representation approximation, not a change
to the target fixture. Reports include the measured error and twice that error
as a conservative pure-target fidelity error bound for the untruncated physical
output density. The final MPO report includes this target error separately from
any MPO discarded-tail error. Exact MPO propagation does not erase target error.

The remote fixture manifest is an authenticated projection onto qubit count 20,
with all nine original q20 arrays staged to satisfy collection completeness.
Only the six structured targets are run. The deployment plan records the original
manifest checksum, projected-manifest checksum, and every unchanged payload hash.
The 6/12 fixture is also preserved in the frozen source snapshot for regression
checks; twenty-four-qubit data are not transferred or executed.

## Execution and measurements

Leonhard-13 owns Gaussian, critical TFIM, and random MPS bond 2; Leonhard-16 owns
ferromagnetic/paramagnetic TFIM and random MPS bond 3. Each host runs one fresh
leaf process at a time, with one numerical thread and nice priority 10. Each
leaf has a 96-GiB process-tree RSS guard and a 48-hour wall guard. Admission leaves
64 GiB available beyond the leaf allowance; a node waits for capacity without
stopping any existing work. The complete local-map MPO workspace estimate is
limited to 32 GiB. These limits are recorded scientific outcomes when exceeded;
resource exhaustion is not convergence or target unreachability.

The phases are `prepare`, `B`, `B_pruned_mpo`, `B_selected_mpo`, `G`,
`G_pruned_mpo`, and `G_selected_mpo`. Each runs in a fresh Python process, giving
an OS lifetime peak RSS including imports. Leaves save process CPU, stage wall,
and OS peak RSS. Supervisors additionally sample the owned process tree every
second and record end-to-end wall time, peak observed tree RSS, exit code, and
any guard reason. An external kill retains the supervisor receipt even if the
leaf could not write its own final receipt. These sampling peaks are labelled
separately from OS high-water measurements.

MPO `bond-growth.jsonl` is flushed before and after each gate. It records all
19 bond dimensions, cumulative maximum, tensor bytes, elapsed wall time, and
OS peak RSS. A failed SVD therefore leaves the last completed gate and the next
attempt visible. Successful reports also contain the complete bond-growth array.
Runtime and memory include target contraction, while retained tensor bytes alone
are explicitly not treated as total memory. Completed checkpoints and campaign
claim directories prevent duplicate dispatch. Interrupted stages are not silently
resumed or restarted with changed numerical settings.

## Artifacts and inspection

The local deployment record is under `output/structured-q20-mpo-20261019-v2/`.
Remote root: `/Home/varioper/froehlich/yaqs-checks/structured-q20-mpo-20261019-v2`.
The source tree is frozen separately from `campaign/`, with a checksum inventory.
The existing tested `noiseless-growth-q12-20261015/env/bin/python` runtime is used
read-only; its package versions are captured before launch. No shared environment
is installed into or upgraded.

Inspect `campaign/status-leonhard-13.json` and `status-leonhard-16.json`, then
`campaign/jobs/TARGET/receipts/`, `supervision/`, and B/G `mpo-*/record.json`.
`parent/` is the single shared preparation endpoint; B/G `selection/` contains
returned parameters and authenticated references to every saved update. Each
target ends as complete or partial, retaining all failed and skipped stages.
A future replicated or larger campaign requires a fresh plan; this six-target,
one-seed pilot is a scaling diagnostic rather than a statistical comparison.

## Concurrent scheduling amendment

The optional `topdown_structured_q20_parallel` controller permits independent
targets around the unchanged frozen numerical source. Its separately hashed
controller and dispatch records live in `campaign/parallel-control/`. Per-target
exclusive claims prevent duplicate dispatch, and per-target status filenames
avoid concurrent writers. Original supervision logs remain in place.

To release a target from the original serial queue, a replacement controller
authenticates the old controller and its active leaf by PID, creation time and
complete command. It pauses and retires only the controller, keeping the optimizer
leaf running with its tensors, parameters and L-BFGS history. It carries forward
the existing sampled peak and elapsed wall allowance, then monitors the adopted
leaf until its final receipt. An adopted controller cannot obtain the original
OS wait status: the supervision record explicitly uses the authenticated leaf
completion receipt instead. Subsequent stages retain fresh-process supervision.

Random MPS bond 2 can then start concurrently with critical TFIM on Leonhard-13.
Each process retains the original 96-GiB guard, one numerical thread, 48-hour
wall allowance and frozen training settings. A 256-GiB available-memory admission
floor reserves two full process allowances plus 64 GiB for the node. Leonhard-16's
existing random-MPS-bond-3 controller continues independently. No failed
preparation is automatically retried by this scheduling change.

## Preserved first-launch failure

The initial v1 snapshot hit its 96-GiB guard during ferromagnetic target setup,
before training: a legacy `MPS.scalar_product` norm built a four-bond transfer
tensor. Its source, supervisor receipt (about 98.5 GiB sampled peak), and all
partial stages remain in the original `structured-q20-mpo-20261019` directory.
Only that new pilot's owned controllers/leaves were stopped for repair; older
campaigns were not changed. Version 2 uses the already tested streaming two-index
`mps_overlap` for target setup and MPO target norms, and explicitly limits the
Rayon pool as well as BLAS/OpenMP. The TT-SVD tail budget is 5e-13 while the
verified fixed-vector error limit remains 1e-12. The restart uses a fresh source
snapshot and namespace rather than overwriting or silently resuming v1 stages.
