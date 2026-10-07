# Strong stochastic controls for the mean-two objective

Status: implementation and engineering validation only. No scientific runs are
authorized by this document. The coordinator freezes a checksummed JSON protocol
after the complete lint batch passes and explicitly allocates the run.

## Scientific question

Do the strongest existing weighted Adam and adaptive online controls close the
gap to exact mean-two optimization when they receive the same objective and
structural opportunities? The objective is the equal mean of the unchanged
`depolarizing_1s_all` and `dephasing_1s_all` calibrated exact channels. The sampler
and its literature attribution are fixed in [MIXTURE_PROTOCOL.md](MIXTURE_PROTOCOL.md).
This driver changes the two stochastic controls' training objective and exact
deletion ranking to that mean. It introduces no new optimizer principle.

The four original q6 development targets are `gaussian_mu0p5_sigma0p1`,
`tfim_critical`, `haar_random_1` and `random_mps_bond2`, replica 11 only. Starts
come directly from the authenticated overnight `starts.json`: 78 one-qubit and
30 two-qubit physical gates, with the original scalar parameters and integer
gate identifiers. Each circuit prepares its target from the all-zero input. No
confirmation target or new construction is introduced.

## Frozen arms and budgets

| Arm                                | Parameter updates                                            | Structure                      |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------ |
| `fixed_mean2_weighted_adam`        | Weighted Adam, rate 0.02                                     | Original 108 gates             |
| `fixed_mean2_adaptive_online`      | First 25 online updates at 0.7, then adaptive online at 0.02 | Original 108 gates             |
| `structural_mean2_weighted_adam`   | Same weighted Adam                                           | Exact mean-two deletion/repair |
| `structural_mean2_adaptive_online` | Same online/adaptive schedule                                | Exact mean-two deletion/repair |

Every update uses **32 total paths per channel, 64 paths overall**, with one
deterministic empty-factor path plus 31 conditional paths per channel. Both
optimizers receive the same paired root seed and method-independent update
counter. There is no path-count or learning-rate screen in this driver.

Each arm receives 1500 nominal charged stochastic updates. Every arm executes
and pays for its own initial updates; no shared warmup is credited twice or
silently excluded from one arm. Fixed arms use all 1500 updates on the original
structure. Structural arms use 150 warmup updates, up to 15 search rounds, a
shortlist of two deletions, and 25 repair updates for each of the retained
circuit and both shortlisted deletions. The first 18 original gate origins are
protected. Search origins are the original physical gate positions, matching
the exact ensemble-structural comparator; separate sparse logical/native
construction IDs remain unchanged in the stored gate metadata. A round starts
only if all its branches and at least one final
polishing update fit. All remaining updates polish the selected incumbent.

All repair branches consume their full nominal allowance, including rejected
or numerically failed branches. Their actually attempted and completed work is
also recorded separately. Exact rankings, branch endpoint scores and discarded
repair work are additional charged work; they are not presented as stochastic
updates or hidden within a 1500-call equivalence claim.

## Structural decisions and preserved memory

The existing `ensemble_deletion.deletion_fidelities` computes the finite removal
of a complete gate and its entire noise window under both training channels.
It ranks all eligible deletions with one forward and one adjoint pass per
channel. Ranking is descending mean deleted fidelity, quantized at 1e-12 for
deterministic ties, then ascending original gate origin.

Each shortlisted deletion and the retained circuit receive equal repair update
caps and the same fresh epoch interval. Their exact mean-two endpoint scores
determine selection. A deletion must beat both the incoming endpoint and the
equally repaired retained endpoint by more than 1e-10. A retained endpoint is
adopted only if it improves the incoming endpoint by more than 1e-10; otherwise
the older incumbent and its older optimizer memory are preserved. The search
stops when no repaired deletion has an advantage. It never selects a favorable
training checkpoint within a repair stage.

The historical strong-baseline memory rule is preserved. Deletion compacts
surviving first moments, second moments and occurrence counters in original
parameter order, which can differ from physical gate order in the original
starts. Adam's selected-path update age continues across warmup,
accepted repairs and polishing. Adaptive online uses the existing backend's
second-moment scaling, beta2=0.999 and epsilon=1e-8. Its first 25 selected-path
updates are the historical plain online phase; the occurrence counters remain
zero during that phase. Rejected repairs consume compute but do not age the
selected branch's optimizer memory.

Sibling repairs deliberately use common random seeds for a paired comparison.
The epoch counter advances by 25 after the entire round, independent of which
branch is chosen, so future optimization does not recycle those repair streams.
Distinct physical architectures need not realize identical factor events from
the same integers. Every structural change rebuilds the two-channel sampler;
it never silently reuses maps for another gate list.

## Strong references and endpoint selection

The primary reference artifact is the coordinator's new completed
`ensemble_structural_experiment` panel with its own frozen 3000 individual
channel-call allowance: structural mean-two L-BFGS, fixed mean-two L-BFGS, and
structural mean-two exact Adam. Its original starts and protocol checksum must
match before comparison. These deterministic controls retain their own
preregistered stage-memory rules; the strong stochastic controls intentionally
retain their established surviving-memory rule. This difference is disclosed
and does not justify calling equal update counts equal work.

The complete seven-arm panel uses the minimum observed final training CPU as a
common cutoff. For each arm, select the chronologically last available endpoint
at or before that cutoff. Do not choose by fidelity, interpolate parameters,
or borrow a later iterate. During structural repair, only the incumbent remains
deployable until every branch in the round has been scored and selection is
complete. This is the same endpoint availability rule as the exact structural
reference. Failed arms or unknown physical CPU accounting block the complete
common-CPU panel instead of being dropped.

Every final and common-CPU endpoint is rescored independently under all twelve
unchanged benchmark conditions. Ballarin uses the existing native materializer,
its angle-dependent channel and pruning semantics. Logical and Ballarin-lowered
gate counts are reported separately. No stochastic final-evaluation randomness
is needed for these exact q6 density evaluations.

The unchanged original construction receives an exact twelve-condition score.
The existing `ensemble-original108` artifact supplies earlier fixed local Adam,
local L-BFGS, noiseless Adam, noiseless L-BFGS and mean-two L-BFGS controls without
new training. Their authenticated records are stored in `REFERENCES.json` and
shown as `historical_unmatched` in the scoreboard. They retain their earlier
1100-individual-channel-call caps and never enter this panel's common-CPU
selection. No noiseless historical reference is relabelled as objective-matched.

## CPU and work accounting

The primary CPU clock uses the same inclusion basis as the exact structural
reference: sum the stage optimization clocks, including numerical setup,
sampling, optimizer work and intermediate checkpoint writes, plus exact
ranking/branch-scoring operation clocks. Stage-final and attempt-ledger writes,
circuit-bank storage, outer orchestration, custody verification and final/common
rescoring are outside that clock. This inclusion basis is explicit in every
result. Evaluation CPU is separately retained. A controlled-clock test verifies
that intermediate checkpoint cost enters the stage clock and final-ledger cost
does not.

For each stochastic backend call, record attempted and completed updates, paths,
directional paths and local-operator applications from the verified mixture
sampler. Gradient calls require 128 directional paths per update and online
calls 192 at the default budget. Tangents, reevaluated gates, zero-weight paths
and cancelling Pauli factors are included in operator counts. If interrupted
inside a backend call, its complete path/operator budget is recorded as attempted
work; it is not asserted that every operator completed. Sampling attempts are
also recorded. Exact ranking and score ledgers record per-channel attempted and
completed calls and the corresponding density directions. All replayed work
after a lost checkpoint remains charged.

Curves store only theta, circuit digest, cumulative CPU and update count.
Circuits are stored once in a content-addressed bank with every gate field.
Current checkpoints contain full theta, first/second moments, occurrence
counters, update age and gate origins. Checkpoints are durable every ten stage
updates. The stage and stream identity include the complete starting memory,
target, configuration, root seed, epoch and source version. Resume restores the
last durable memory and deterministically replays lost work. A completed or
numerically failed repair decision is cached so a resumed search cannot quietly
retry a failed candidate. Every cached outcome authenticates its attempt ledgers.

A durable running marker exposes uncatchable process death. Such a marker
requires external resource accounting before that stage can resume; its missing
CPU is never treated as zero. Caught interruptions write measured attempt times
and actual completed counters. Numerical failures remain visible and the other
arms continue. Source or benchmark custody changes are rejected before scientific
execution and after each completed case.

## Falsification and interpretation

Before any scientific run, small q2 engineering tests must establish exact full
optimizer-memory resume, preserved deletion memory/metadata, matched initial
streams, the fixed phase schedule, explicit failed/replayed work, authenticated
reference import and final/common-CPU all-condition scoring. The separate
sampler tests already establish unbiased fixed-theta mean-two gradients by
complete factor enumeration.

The scientific report will retain every target and channel, including failures.
For promotion, compare mean noisy fidelity across all eleven noisy final
conditions, per-target signs, the weakest channels, gate counts and actual CPU.
The existing program's +0.5 percentage-point mean signal or a consistent
fidelity/resource advantage is the development criterion; no isolated best
channel or one favorable seed is sufficient. If the stochastic controls match
or outperform the candidate under the common CPU cutoff, that weakens any
claim that the candidate is the strongest repository method. If they fail, the
result distinguishes fair objective opportunity from estimator and update-rule
limitations. No setting is tuned on these outcomes or on confirmation targets.

The weighted sampler adapts known trajectory stratification; see
[Abdelhafez, Schuster and Koch (2019)](https://link.aps.org/accepted/10.1103/PhysRevA.99.052327)
and the precise adaptation in [MIXTURE_PROTOCOL.md](MIXTURE_PROTOCOL.md).
Weighted Adam uses [the Adam paper](https://arxiv.org/abs/1412.6980).
Adaptive online scaling and moment-preserving deletion are existing repository
controls, not new contributions. This is a preregistered fairness extension and
implementation study, with no state-of-the-art claim.
