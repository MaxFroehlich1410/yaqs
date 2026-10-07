# Next experiments: target-independent Krotov state preparation under noise

Protocol version: `yaqs.krotov_pruning_multitarget.v1.1`, revised 2026-09-05.
Status: experiment specification, **not an implemented runner or completed study**.

Revision v1.1 adds a separate multi-target hybrid/gradient ablation and a
theory/provenance review in Section 12. The supplied optimization-paper text
ends at its abstract; its detailed hyperparameters and theorem assumptions
have **not** been verified. The main ten-method pruning matrix is unchanged.
Version participates in seed derivation, so v1 manifests are not resumable as
v1.1 runs even where numerical settings agree.

## 1. Research question and scope

Determine which complete Krotov-based preparation procedure gives the best
fresh noisy fidelity, including the choice of circuit size. Test layerwise
training, circuit pruning, repair after pruning, and noise-aware refinement as
separate contributions. Do not require noisy refinement to win on an unpruned
circuit before investigating pruning.

The scientific targets, BMPD convention, and noise definitions come from the
[benchmark definition](../benchmarks/state_preparation_benchmarks.md). This is
an explicitly labelled noise-aware/architecture-changing extension, not a
change to its base train-once-without-noise benchmark. Different final
architectures are intentional here; comparisons at a fixed resource level
match both one- and two-qubit gate counts.

Evidence motivating the design is in the
[historical review](state_preparation_experiment_findings.md) and
[corrected-noise rerun](state_preparation_calibrated_rerun_findings.md).
The strongest archived pipeline was layerwise Krotov without pruning, while
pruning sometimes achieved useful compression. The recent fixed-architecture
runs did not test that compression contribution.

All numerical choices below are proposed protocol settings, not previously
validated optima. No experiments are launched by adding this document.

### Meaning of target independence

The algorithm may read the target amplitudes and optimize parameters and a
gate mask for that target. It must use the **same procedure, hyperparameters,
candidate resource levels, and selection rules for every target**. It must not
branch on the target identifier or family, use an analytic target-specific
preparation, or receive extra restarts for a difficult state.

Every scientific experiment below includes multiple target states. A single
target may be used in a unit test, but is not an experiment supporting the
method. Success on this finite suite supports cross-target robustness, not a
proof of optimality for arbitrary quantum states.

## 2. Targets, replicas, and sequence of experiments

Use the checked-in `benchmarks/state_preparation_target_states.json` without
regeneration or target substitution. Run its generator with `--check`; record
the fixture checksum and each normalized target-vector checksum. Preserve the
fixture's little-endian convention and global phase.

| Target identifier         | Definition                                                               | q6 role             |
| ------------------------- | ------------------------------------------------------------------------ | ------------------- |
| `gaussian_mu0p5_sigma0p1` | Endpoint-excluded quantics grid, square-root Gaussian, mu=0.5, sigma=0.1 | Development         |
| `tfim_ferro`              | Open uniform TFIM, J=1, h=0.5                                            | Reserved assessment |
| `tfim_critical`           | Open uniform TFIM, J=1, h=1.0                                            | Development         |
| `tfim_para`               | Open uniform TFIM, J=1, h=1.5                                            | Reserved assessment |
| `haar_random_1`           | Complex Gaussian vector, normalized; fixture seed 4001                   | Development         |
| `haar_random_2`           | Same construction; fixture seed 4002                                     | Reserved assessment |
| `haar_random_3`           | Same construction; fixture seed 4003                                     | Reserved assessment |
| `random_mps_bond2`        | Real Gaussian open-boundary MPS, bond 2; fixture seed 5002               | Development         |
| `random_mps_bond3`        | Real Gaussian open-boundary MPS, bond 3; fixture seed 5003               | Reserved assessment |

For the TFIM targets the eigensolver seeds are 1729, 2718, and 3141 plus
10000\*n, respectively; these are not disorder realizations. All nine targets
exist at both n=6 and n=12. The q12 vectors are separate larger-system tests,
not new unseen target families. No difficulty-based exclusions are allowed,
including low-fidelity q12 Haar cases where the ansatz may be insufficient.

| Experiment                   | Targets                                                       | Initialization-replica labels | Conditions and purpose                                                            |
| ---------------------------- | ------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------- |
| E0: implementation smoke     | All four development targets, n=6                             | 11                            | Reduced budgets in Section 11; execution checks only                              |
| E1: development comparison   | All four development targets, n=6                             | 11, 29, 47                    | All ten methods, all four resource levels, reference local depolarizing noise     |
| E1H: optimizer ablation      | All four development targets, n=6                             | 11, 29, 47                    | Separate fixed-b=2 batch/hybrid comparison, two starting points, Section 12       |
| E2: locked q6 comparison     | All nine targets, n=6                                         | 71, 101, 131, 173, 211        | Same methods/settings; primary new-target assessment on the five reserved targets |
| E3: locked size transfer     | All nine targets, n=12                                        | 71, 101, 131, 173, 211        | Same methods/settings, with n-dependent gate quotas only                          |
| E4: noise-strength transfer  | All nine targets at both sizes; reuse E2/E3 selected circuits | Same five labels              | Two additional calibrated local-depolarizing strength pairs                       |
| E5: noise-structure transfer | All nine targets at both sizes; reuse E2/E3 selected circuits | Same five labels              | Remaining agreed noise conditions at the reference calibration                    |

E1 is for debugging and assessing the proposed design, not for making final
claims. The main matrix has no hyperparameter search. E1H is a small declared
schedule/update ablation, not a reproduction of the paper's tuned sweep.
Complete E1 and E1H before E2. If either motivates changing the main settings,
publish a new version and rerun the whole development panel before
opening reserved E2 results. Never repair a losing target by changing only its
settings. Freeze a source revision/configuration manifest before E2; E3-E5
inherit it without retuning.

Reserved means not used to develop this protocol's implementation; the target
fixtures are public and may have appeared in earlier work. Report that history
rather than describing them as cryptographically blind data. The four
development targets' fresh-seed E2 results remain labelled exploratory.

## 3. Circuit and matched resource levels

Initial state: |0> tensor n. Use the canonical nearest-neighbor BMPD
state-preparation adjoint circuit with its initial trainable U3 layer enabled.
All gates are the existing scalar rotation primitives; no added routing,
native-device compilation, fusion, or automatic zero-angle removal in E0-E4.

Let b in {1,2,3,4} denote a **resource level**, not the depth of a sparse pruned
circuit. Its retained counts are

```text
G1(n,b) = 3*n + 6*b*(n-1)
G2(n,b) = 3*b*(n-1)
P(n,b)  = G1 + G2 = 3*n + 9*b*(n-1)
```

|   n | Resource b | One-qubit gates | Two-qubit gates | Parameters |
| --: | ---------: | --------------: | --------------: | ---------: |
|   6 |          4 |             138 |              60 |        198 |
|   6 |          3 |             108 |              45 |        153 |
|   6 |          2 |              78 |              30 |        108 |
|   6 |          1 |              48 |              15 |         63 |
|  12 |          4 |             300 |             132 |        432 |
|  12 |          3 |             234 |              99 |        333 |
|  12 |          2 |             168 |              66 |        234 |
|  12 |          1 |             102 |              33 |        135 |

A dense compact comparator uses BMPD depth b. A pruned candidate begins at
depth 4 but deletes individual gates until it has these same G1/G2 counts.
Its physical gate order and connectivity need not match a dense depth-b
circuit. Report its actual dependency depth, not "depth b". Parameter count
equals gate count only because this protocol uses one distinct scalar
parameter per primitive.

## 4. Methods: five construction routes, each with two refinements

All ten variants below use the repository's **batch Krotov-inspired discrete
adjoint optimizer**. None is ADAPT-VQE. Online/hybrid updates are tested
separately in E1H rather than silently mixed into this construction comparison.
Refresh-5 and per-target optimizer choices remain excluded. In the noiseless
fixed-generator, untruncated setting, batch Krotov is exact gradient descent;
it is not a numerically distinct competitor to identically configured GD.

| IDs        | Construction route                                                                                                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------- |
| `L0`, `LN` | Layerwise noiseless Krotov through depths 1..b; no pruning                                                          |
| `M0`, `MN` | Shared depth-4 layerwise checkpoint; iterative angle-magnitude pruning with noiseless repair                        |
| `I0`, `IN` | Same checkpoint and quotas; historical gradient-impact pruning with noiseless repair                                |
| `D0`, `DN` | Same checkpoint and quotas; actual single-gate-deletion noiseless-loss scoring with noiseless repair                |
| `C0`, `CN` | Direct fixed-depth-b noiseless training from random initialization; no inherited trained parameters or learned mask |

The suffix defines the **final 200-update refinement**:

- `0`: optimize noiseless loss L0 = 1 - |<target|psi>|^2.
- `N`: optimize the trajectory estimate of noisy loss
  LN = 1 - E[|<target|psi_noise>|^2] at the reference local-depolarizing channel.

For every route/resource/target/replica, fork suffixes 0 and N from exactly the
same pre-refinement parameters and gate list. This is the paired comparison
that isolates noise-aware refinement. Construction and repair are noiseless
for **both** suffixes. The same masks are therefore used before final
refinement; a noisy variant cannot secretly receive a different pruning path.

M is the inexpensive pruning baseline, I tests the manuscript's historical
first-order idea, and D checks actual finite gate-removal damage. D's score is
not a noisy importance score: noise enters its final N refinement and resource
selection. No claim that D is already superior is assumed.

At b=4, L, M, I, and D have the same starting construction and need no pruning.
Their identical numerical refinements must be reused, not counted as extra
independent observations. Keep a separate frozen, unrefined copy of every
pre-refinement candidate as a diagnostic; these frozen controls are not extra
algorithmic competitors.

## 5. Initialization, optimization, and repair parameters

The seed scheme is specified in Section 9. For each (n,target,replica), draw one
PCG64 standard-normal master vector z of length P(n,4). Direct depth-b
initialization is 0.1*z[:P(n,b)]. Layerwise depth 1 uses the same prefix;
subsequent depth increases preserve every previously trained parameter and
initialize only the added slice as 0.001*z[new_slice]. Never reseed from the
target's physical generation seed.

| Setting                                             | Exact value                                                                                                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Layerwise source depths                             | 1, 2, 3, 4                                                                                                               |
| Noiseless iterations per depth                      | 100                                                                                                                      |
| Noiseless source/compact/continuation learning rate | Constant 0.2                                                                                                             |
| Pruning repair                                      | 20 noiseless batch updates after each deletion round, including the last                                                 |
| Repair learning rate                                | Constant 0.1, same for all rounds                                                                                        |
| Final refinement                                    | 200 batch updates, no tolerance-based early termination                                                                  |
| Final refinement learning rate                      | eta(k)=0.1*exp[-0.01*(k-1)], k=1..200, for both suffixes                                                                 |
| Noisy training trajectories                         | 32 per update                                                                                                            |
| Noisy sampling policy                               | Resampled independently each update; reuse realized maps only within that update's forward/backward computation          |
| Trajectory update rule                              | `cross`, matching the recent experiment; record this explicitly rather than calling it a guaranteed exact noisy gradient |
| Jump-normalization differentiation                  | False                                                                                                                    |
| Circuit-TJM window                                  | dt=1 for every retained noisy gate                                                                                       |
| Standard-noise sampler                              | Exact commuting Pauli-Lindblad factor sampling, specified in Section 7; not the legacy one-jump approximation            |
| Tensor precision                                    | complex128 states/operators, float64 parameters                                                                          |
| Truncation                                          | No bond cap; SVD threshold 0; discarded-weight mode; minimum bond dimension 1                                            |
| Gradient clipping / angle penalty / gate penalty    | None                                                                                                                     |
| Final candidate update numbers                      | 0, 50, 100, 150, 200 only                                                                                                |
| Validation trajectories                             | 512 per candidate, one fixed validation stream per target/replica/condition                                              |
| Fresh final test trajectories                       | 2048 per selected candidate, batches of 128                                                                              |
| Restarts                                            | None beyond the listed initialization replicas                                                                           |

Cross updates remain a specified optimizer variant, not an assertion of
monotone descent of the exact noisy objective. Report actual objective traces;
do not drop bad steps from them. The primary scientific outcome is fresh
channel-averaged fidelity, not the training objective's maximum.

### Equal outer-update controls

There are R_b=3*(4-b) deletion rounds. At a fixed b, every route receives
400+20*R_b noiseless construction/repair updates before its final 200 updates:

- M/I/D: 400 updates of shared layerwise source preparation, then R_b repair
  blocks of 20 updates.
- L: 100*b layerwise updates to depth b; continue at that fixed depth for
  400-100*b more updates, then R_b blocks of 20 updates at the repair rate.
- C: 400 direct fixed-depth-b updates, then R_b blocks of 20 updates at the
  repair rate.

Noiseless L/C continuation uses the stated constant 0.2 rate, followed by the
constant 0.1 repair-control blocks. Every branch finishes its stated budget;
checkpoint selection does not change the executed training budget. L/C repair
blocks contain no deletions and make them stronger continued-training controls.

These are **update-matched, not compute-matched** comparisons: gradient
scoring, circuit sizes, and trajectory costs differ. Section 10 requires a
complete work ledger and forbids claiming equal computational cost from equal
iteration counts. The depth-16/1,032-parameter historical starting point is not
required in this first protocol; test the mechanics on depth 4 before adding
another depth axis.

## 6. Exact pruning rules

Assign every source primitive an immutable origin ID consisting of its initial
layer site/primitive index or its disentangling-layer/bond/primitive index.
Protect all 3\*n initial-layer rotations. All other one- and two-qubit scalar
rotations are eligible. Original relative gate order is always preserved.

Each round deletes exactly 2\*(n-1) eligible one-qubit gates and (n-1) eligible
two-qubit gates: 10+5 gates at n=6, 22+11 at n=12. Nine rounds reach b=1.
Retain repaired candidates after rounds 0, 3, 6, and 9 for b=4,3,2,1.

At the **start of each round**, compute scores on the current repaired
circuit, sort separately by arity, and delete the required lowest-scoring
gates simultaneously. Recompute all scores next round, not after each gate in
the same batch. This is batch deletion with repair, not sequential greedy
re-scoring after every removal.

Let alpha_g be the resolved physical rotation angle and wrap(alpha) lie in
[-pi,pi). Rank by the following scores, in ascending order:

1. **M: magnitude.** s_g=|wrap(alpha_g)|.
2. **I: gradient impact.** s_g=|wrap(alpha_g) \* dL0/dalpha_g|, using the exact
   noiseless adjoint gradient. Convert parameter gradients by the gate's angle
   scale; do not confuse theta with the resolved physical angle.
3. **D: actual deletion loss.** s_g=L0(C without gate g)-L0(C), with all other
   parameters fixed. Physically remove the gate. Negative scores are allowed
   and are preferred: deletion alone improved noiseless fidelity. Do not
   replace this finite change by a first-order gradient approximation.

For deterministic near-ties, use the lexicographic key
`(round(score/1e-12), |wrap(alpha_g)|, origin_id)`, with round-to-nearest-even.
The 1e-12 bin is only a tie rule, not a pruning threshold. Scores and the
resolved selection keys must be retained.

I can be uninformative near a stationary point, where first derivatives are
small; this is a reason to compare it with D, not to declare I invalid after
seeing results. An accelerated exact local-contraction evaluation of D is
allowed only after it agrees with explicit gate deletion on all four E0
targets to absolute fidelity tolerance 1e-10. Record every candidate score's
work, including gradients or deletion evaluations.

Deleting a gate means deleting its **unitary and attached noise window**, then
compacting parameter indices while retaining the origin-ID mapping. Setting
its angle to zero while leaving its noise is not pruning. Do not automatically
remove additional small-angle gates outside the stated quotas. No target may
receive a different deletion batch size or protected set.

Always preserve the original unpruned checkpoint and all four resource levels.
A bad prune does not trigger a larger training budget or a new random seed.
The unpruned candidate provides a performance fallback during resource
selection; forced-budget failures still remain in the fixed-budget results.

## 7. Noise calibration and transfer tests

### Primary training and evaluation condition

E0-E3 use local depolarizing noise after **every retained scalar gate**:
three local X/Y/Z generators on a one-qubit support, and six local generators
on a two-qubit support. Protected initial-layer gates are still noisy.
The complete-window average-infidelity targets are r1=0.00064 and r2=0.0051.
For dt=1, the per-operator Lindblad coefficients are

```text
a1 = -log(1-2*r1)/4
a2 = -log((4*sqrt(1-5*r2/4)-1)/3)/4
a1 = 0.00032020497493061084
a2 = 0.0010664702651014273
```

Use the exact-generator calibration, not the legacy per-operator values.
Unlike the recent rerun, this planned protocol samples the exact commuting
Pauli-Lindblad channel. For every active generator P_m with integrated
coefficient a_m, independently draw a Bernoulli variable with probability
q_m=(1-exp(-2\*a_m))/2 and apply P_m if that draw succeeds. Apply all selected
factors in the frozen operator order. This represents odd jump parity for
each factor and permits several factors in one window; it is not a
one-event approximation. It still evolves pure MPS trajectories in the
circuit-TJM framework, not a production density matrix.

This choice eliminates the known finite-step bias for the standard commuting
Pauli models and gives the intended 0.064% and 0.51% average gate errors. The
old one-jump rerun's approximately 0.06401025% and 0.51027142% errors belong
to a different execution version. Retain that version in old manifests;
do not silently substitute this sampler when resuming them. These are
homogeneous hardware-motivated abstract-gate models, not native executions or
exact replicas of a Google/IBM processor. Hardware sources and conversion
derivations are in the benchmark definition; do not invent new measurements.

### E4: two additional strengths, with no retraining or reselection

| Strength label                   |      r1 |     r2 | a1 per local Pauli | a2 per local Pauli |
| -------------------------------- | ------: | -----: | -----------------: | -----------------: |
| Reference, already used in E2/E3 | 0.00064 | 0.0051 |  0.000320204974931 |  0.001066470265101 |
| Central                          | 0.00035 | 0.0033 |  0.000175061278598 |  0.000689159459733 |
| Low                              | 0.00036 | 0.0014 |  0.000180064831121 |  0.000291964803153 |

Low/central describe selected hardware-motivated pairs, not an ordered sweep
of both coordinates and not device quantiles. Evaluate the **same adaptive
resource-selected circuit per method** from E2/E3 at the two additional pairs.
Keep 2048 test trajectories per pair, disjoint from all earlier tests. Do not
select a new mask, parameter checkpoint, or resource level for these tests.

### E5: the remaining benchmark structures, with no retraining or reselection

At reference r1/r2, add all nine standard profiles other than the already
tested `depolarizing_1s_all`:

```text
dephasing_1s_1q, dephasing_1s_2q, dephasing_1s_all,
dephasing_2s_2q, dephasing_1s2s_all,
depolarizing_1s_1q, depolarizing_1s_2q,
depolarizing_2s_2q, depolarizing_1s2s_all
```

Also add `ballarin_coupled`. Together with the noiseless and primary results,
this covers the benchmark's twelve noise conditions. E4 adds two more
strength conditions for local depolarizing noise. In E5, again test only each
method's reference-selected circuit, with 2048 trajectories per condition.
Report these explicitly as **noise-mismatch transfer**, not training optimized
separately for every channel. The 0 variants are likewise not reselected.

For every standard profile, use the exact operator list and gate placement in
the benchmark definition. For each support dimension D=2^k, solve for the
common nonnegative integrated coefficient a of its active generators P_m:

```text
eta_Q(a) = exp(-2*a * number_of_generators_anticommuting_with_Q)
r(a) = (D*D - sum_over_all_phase_free_Paulis_Q eta_Q(a)) / (D*(D+1))
```

Use a bracketing root solver on [0,1] for these reference errors, absolute
coefficient tolerance 1e-15 and relative tolerance 1e-14. Validate recovered
average infidelity by independent dense channel exponentiation to absolute
tolerance 1e-12. Calibrate r2 on the complete two-qubit window, not separately
on each qubit or each operator. The nine two-site-only Pauli operators are not
the fifteen-operator combined profile. An inactive window has no noise.

For Ballarin, keep its separate normative model: noiseless one-qubit gates,
exact RXX/RYY-to-RZZ lowering, canonical angles in [-pi,pi), removal of native
RZZ when |angle|<=1e-4, and exact independent local categorical Pauli sampling
with epsilon(a)=2.1e-4+1.43e-3*a and
p(a)=(1-sqrt(1-5*epsilon(a)/4))/3. The local probabilities are
(1-3\*p,p,p,p). Do not apply the standard channel on top of it. Its mandated
near-zero removal is separate from learned logical pruning and must be
reported separately, together with all resulting native counts.

### Numerical accuracy gate

Use no tensor truncation at either size; an exact n=12 pure-state MPS has a
maximum possible Schmidt rank of 64. Do not silently introduce a bond cap if
execution is expensive.

Check the factor sampler's enumerated channel against an independent dense
Lindblad exponential to maximum entrywise tolerance 1e-12 for every profile
and strength before it is used. For all four E0 targets, validate noiseless
MPS circuit fidelity against dense state evolution to 1e-10, and validate
the sampled channel against its density-matrix expectation statistically.
Use 8192 additional validation-only trajectories per E0 target for the latter
check, on a disjoint `numerical_validation` stream; require agreement within
five estimated standard errors plus 1e-4 absolute fidelity. In the zero
variance case use the 1e-4 tolerance alone. These draws never train or select
a method. A failure must be diagnosed, not cured by repeatedly changing seeds.

For E2/E3 monitor every trajectory's normalization and finite values; require
absolute norm-squared drift <=1e-10. Any failure is a numerical issue to report
and resolve globally, not a reason to introduce target-specific truncation.
All selected q6 circuits should additionally receive deterministic exact-
channel checks, reported separately from the fixed-budget Monte Carlo tests.
No timestep substeps or extra numerical noise approximation are part of v1.1.

The density-matrix calculation is a validation oracle, not the production
training method. Do not substitute it for TJM training on easier targets.

## 8. Checkpoint and resource selection without final-test leakage

Within each resource b and method, evaluate only the final-refinement updates
0,50,100,150,200 on the fixed 512-trajectory reference validation stream.
Choose the highest mean noisy fidelity; means equal within 1e-12 are tied,
and the earlier update wins. This selection rule is identical for suffixes
0 and N: noiseless-training methods may use noisy validation to choose the
delivered circuit, but do not use it for their parameter gradients.

Retain one selected checkpoint at **each** of the four resource levels. The
adaptive resource-selected output is chosen as follows:

1. Find the largest validation mean among the four selected checkpoints.
2. Keep all resource levels within 0.002 absolute fidelity of that mean.
3. Choose the one with the fewest two-qubit gates, then fewest one-qubit gates,
   then largest validation mean, then lexicographically smallest stable ID.

The 0.002 rule is a declared engineering tolerance, not a statistical
equivalence claim. Every method has access to the same four levels, including
the unpruned/dense b=4 fallback. This implements target-dependent circuit
selection through a target-independent rule.

E1-E3 final tests evaluate the four fixed-resource selections and their
unrefined, frozen starting candidates. The adaptive output is one of these
already tested selections, so reuse that evaluation rather than drawing an
extra test and choosing between the two. Report both fixed-resource results
and the adaptive output; never choose a method's headline result by its test
fidelity. E4/E5 transfer only the previously chosen adaptive output.

Intermediate pruning scores and noiseless repair losses are training data.
The 512 validation trajectories are selection data, not an untouched test.
The 2048 final trajectories must be generated only after all method outputs
for that target/replica have been frozen. Do not run fresh tests repeatedly
until a confidence interval becomes favourable.

## 9. Randomness, common random numbers, and reproducibility

Use root integers E0=2026090500, E1=2026090501, E2=2026090502,
E3=2026090503, E4=2026090504, E5=2026090505. The replica labels in Section 2
identify independent optimization trials; they are not raw target seeds.
The separate Section 12 ablation uses E0H=2026090510 and E1H=2026090511.

Derive each actual uint64 seed from the first eight SHA-256 digest bytes,
interpreted big-endian, of UTF-8 compact ASCII JSON of this ordered array:

```text
[protocol_version, experiment_root, role, n, target_id, replica_label,
 resource_b, stage, pruning_round, update, trajectory_index,
 gate_origin_id, draw_slot, noise_condition_id]
```

Use null for inapplicable fields, zero-based update/trajectory/draw indices,
and the exact stable strings in the manifest. Roles are `initialization`,
`optimizer`, `training`, `validation`, `numerical_validation`, and `final_test`. Initialization uses
only its experiment root, role, n, target and replica; other fields are null.
Use NumPy `Generator(PCG64(seed))`, never process-global mutable randomness or
Python's randomized `hash()`. Record the exact library versions.

Use stage strings `source_depth_d`, `compact_depth_b`, `repair_round_j`,
`continuation`, and `refinement`, substituting decimal d/b/j. Validation and
test stages are null. Only optimizer-role stages are prefixed by the method
ID. Represent origin IDs as JSON arrays `["initial",site,primitive_slot]` or
`["block",disentangling_layer,left_site,primitive_slot]`, with zero-based
indices. Pauli draw slots follow the benchmark's frozen generator order,
with local sites ascending and labels X,Y,Z. The condition ID includes the
noise profile, strength-pair label, and `exact_pauli_v1` or `ballarin_v1`.

For noisy training, include resource, update, trajectory, and origin-ID keys;
new updates get new draws. Different methods may share underlying uniforms
for paired comparisons, but training keys never overlap validation or test
keys. Optimizer randomness is method-specific through the stage string. The
noiseless batch routine must be deterministic given its initialization.

For validation/test, omit resource and method identity from the draw key so
that the same trajectory index couples methods and resource levels. Sample
with resource, stage, pruning round, and update fields all set to null;
candidate checkpoint numbers must not create different validation draws. Sample
by immutable gate origin and draw slot, not by the gate's current compacted
index. Removing a gate omits its noise window without shifting every later
gate's random numbers. A direct compact circuit uses the corresponding
canonical BMPD origin IDs. Additional native gates use origin-ID suffixes.

**Share random uniforms, not blindly frozen realized operator lists.** A
different topology has a different list length, and Ballarin's probabilities
depend on the current angle. Recompute each circuit's correct branch from
the common uniforms and its own probabilities. Keep stream keys independent
across target, replica, role, and condition. Replaying another circuit's
angle-dependent outcomes can bias the comparison and is forbidden.

Every output manifest must include resolved seeds or a lossless reproducible
seed specification, circuit/parameter/target/provider checksums, pruning
origin maps, source revision plus any dirty-source hashes, and configuration
version. Verify uniqueness within a stream and disjointness across roles.
Resume is allowed only under an identical scientific manifest. Worker count
and scheduling order must not change any result; do not overwrite older runs.

## 10. Reporting, work accounting, and conclusions

### Required output per target/replica/method/resource

- Noiseless fidelity, noisy validation fidelity, fresh noisy test mean,
  sample standard deviation, standard error, and raw trajectory fidelities.
- Paired per-trajectory differences for all relevant comparisons, and the
  selected iteration/resource plus all validation candidates considered.
- Before/after-prune noiseless fidelity for every round; removed origin IDs,
  scores, G1/G2/P and actual dependency depth; final and native counts.
- Gradient and update norms, all losses, trajectory event counts, maximum bond
  dimension, numerical drift, exceptions, and stopping reason.
- Wall/CPU time split into preparation, scoring, repair, refinement,
  validation/selection, and final testing; CPU model and worker/thread counts.
- Number of trajectories, forward/backward sweeps, candidate-deletion
  evaluations, gradient evaluations, one-/two-qubit operator applications,
  and cross-trajectory contraction counts per stage.

The work ledger must charge preparation, pruning search, all four candidate
levels, and selection to a standalone method. Sharing artifacts saves actual
study runtime but does not make pretraining free for a claimed deployed
algorithm. Report actual shared execution cost separately. Final-test cost
is separate from optimization cost. Show fidelity versus total optimization
work and wall time as well as fidelity versus final gate count. Do not claim
equal compute or asymptotic speedup from the outer-update matching alone.

### Multi-target statistics

First average over the five E2/E3 replicas **within each target**. Then give
every target equal weight. Primary new-target E2 statistics use its five
reserved targets; the full nine-target and four-family-balanced summaries are
also mandatory but are labelled appropriately. E3 reports all nine targets.
The four-family summary gives Gaussian, TFIM, Haar, and random MPS equal
weight after averaging within each family. Never pool trajectories to make
the effective number of target states appear larger.

Report every target's paired method differences, worst target, number of
positive target-level differences, and the full failure table. Aggregate
95% intervals use a paired hierarchical bootstrap: 10000 resamples, sample
targets with replacement within the stated assessment panel, then sample
replica indices with replacement within each sampled target; use the same
indices for both methods. Use PCG64 seed 2026090599 for this analysis and
percentile endpoints with NumPy's linear quantile convention. These intervals
describe this finite empirical suite, not every possible quantum state.
Within a target/replica, paired trajectory intervals use Student-t with
2047 degrees of freedom; marginal fidelity intervals may be clipped to [0,1],
but paired-effect intervals must not be clipped to [0,1].

Do not announce the largest observed held-out method mean as a validated
winner after searching ten methods. Before E2, nominate one pruned method
from M0/MN/I0/IN/D0/DN using E1's **family-balanced mean adaptive validation
fidelity**, averaging replicas within targets first. Within 0.001 of the
largest score, prefer the smaller mean two-qubit count, then fewer one-qubit
gates, then lexicographic method ID. Freeze this one global choice; do not
choose a different named algorithm per target. This nomination does not
discard the other methods' experiments or failures.

For that nominated method, the eight predeclared primary comparisons are
against L0, LN, C0, and CN at fixed resource b=2 and at adaptive resource
selection. For simultaneous uncertainty, report Bonferroni percentile
intervals using bootstrap quantiles 0.003125 and 0.996875 in addition to the
ordinary 95% intervals. All other pairwise comparisons are exploratory.
The fixed-resource comparisons support fidelity claims at equal gate counts;
the adaptive comparisons measure a fidelity/resource trade-off, not equal
circuit cost. Report paired N-minus-0 effects within each construction route
to distinguish noisy refinement from circuit construction.

A promising robust improvement should have a positive adjusted lower bound
against the strongest relevant control, positive mean differences on at least
4/5 reserved q6 targets and 7/9 q12 targets, and no target-level mean regression
larger than 0.01 absolute fidelity. Report violations rather than hiding them.
This is a conservative advancement criterion, not a definition of statistical
significance for arbitrary states. Compression at comparable fidelity is a
separate result: report the fidelity difference and gate saving explicitly.
Claim fixed-budget superiority only if all four predeclared control contrasts
at b=2 have positive adjusted lower bounds, not merely the easiest comparison.

No method is "state of the art" from this within-Krotov comparison alone.
That requires independently implemented literature competitors, comparable
resource accounting, and the broader evidence described in the benchmark.

### Failures

Do not drop a failed target, pruning level, or replica from an average.
Finite but poor fidelity is a scientific result, not a software failure.
If a refinement fails, its pre-refinement candidate remains available; if
a construction fails, preserve the valid candidates that actually exist and
mark the missing fixed-resource cells failed. Retain an all-zero-parameter
depth-4 circuit as the explicitly noisy last-resort output if no trained
candidate survives. It keeps its gates/noise; do not reinterpret it as an
empty circuit. Report fallback use prominently.

For algorithm-level averages, evaluate the defined fallback when available.
If an infrastructure failure prevents even that evaluation, publish the
missing row and do not claim a completed comparison. Do not invent a measured
zero fidelity or silently average only successful targets.

## 11. Execution gates, scale, and missing implementation

E0 uses the four development targets, replica 11, source depths 1..4 with two
updates each, the same nine pruning rounds with one repair update each,
four final-refinement updates with checkpoints 0,2,4, four training trajectories,
eight validation trajectories, and sixteen test trajectories. For its update
controls, replace 100 by 2, 400 by 8, and 20 by 1 in Section 5. All ten
methods and four resource levels still run. E0 is only an implementation and
resource-profile test; its numerical fidelities are not paper evidence.

E1-E3 use the full Section 5 budgets. Per experiment there are
`targets * replicas * 10 methods * 4 resource levels` labelled refinement
endpoints: E1=480, E2=1800, E3=1800. Deduplication of identical b=4 paths lowers
the number actually trained. Frozen controls, intermediate validation, and
transfer evaluations add work and must appear in the planning ledger. E4/E5
test only the one reference-selected output per method, giving respectively
1800 and 9000 noisy transfer cells across both sizes before deduplication.

This is a staged research program, not a small laptop smoke run. Measure E0
costs on **all four targets**, publish projected CPU-hours/peak memory/disk for
the next phase, and choose the execution allocation before launching it.
Never silently cut trajectories or skip Haar targets to fit a time limit.
Hardware allocation affects scheduling only; scientific budget changes require
a new protocol version. No wall-time early stopping is part of v1.1.

### What can be reused and what must be built

Execution allocation for the current laptop: 8 detected CPU cores and 16 GiB
RAM. Use up to eight spawned worker processes with one BLAS/OpenMP thread per
worker, scheduling independent target/replica/method jobs as their shared
checkpoint dependencies become ready. Do not launch nested eight-worker
trajectory pools or split gate-sequential online updates across workers.
There may be fewer than eight runnable jobs during source preparation or at
the end of a phase; do not invent extra replicas to fill cores.

The reusable
[execution helper](../benchmarks/state_preparation/execution.py) now provides
bounded, completion-driven process scheduling. The existing general benchmark
and causal-study CLIs accept `--workers auto` (their default), or an explicit
positive count. The general runner can fill the pool across targets and
initializations; the causal runner is still limited to its replica count.
Python entry points remain serial unless workers are requested. These are
execution capabilities, **not a claim that E0-E5/E1H are implemented**.

Only the coordinator writes shared result summaries; each job owns distinct
checkpoint paths. Restore canonical ordering before statistics, keep RNG keys
independent of process/completion order, and permit worker-count changes on
resume. Use E0/E0H peak-memory measurements on all four targets before allowing
eight simultaneous larger jobs: CPU-based `auto` is not a memory safety
estimate. Reduce workers if the projected concurrent resident memory plus OS
overhead does not fit 16 GiB; never change trajectory budgets, tensors, or
target coverage merely to keep all cores occupied. This scheduling choice
does not change the scientific protocol version.

| Component                                                            | Current status / required action                                                                                     |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| All 18 targets and scalar BMPD construction                          | Implemented; retain conventions and fixture checks                                                                   |
| Batch noiseless/noisy Krotov and fixed-map serialization             | Implemented; reuse focused primitives                                                                                |
| Bounded all-core execution and single-threaded numerical workers     | Implemented for existing runners; integrate protocol dependency stages without nested pools                          |
| Online/hybrid noiseless and noisy update primitives                  | Implemented; E0H/E1H orchestration, reference-gradient audits, and phase-boundary tests still required               |
| Reference local-depolarizing calibration                             | Implemented; other profiles still need exact-generator calibration                                                   |
| Exact standard Pauli-factor sampling                                 | Must be added as a separately versioned provider and validated; existing standard TJM profiles use one-jump sampling |
| Recent causal runner                                                 | q6 critical-TFIM-only; cannot execute this protocol unchanged                                                        |
| General origin-preserving M/I/D pruning and quota repair paths       | Must be implemented and tested; Ballarin threshold lowering is not a substitute                                      |
| Multi-target/method/resource runner and shared starting checkpoints  | Must be implemented with the explicit ten-method registry                                                            |
| Noisy checkpoint/resource selection and fallback rules               | Must be implemented with all candidate scores retained                                                               |
| Common uniforms across changed topologies / angle-dependent channels | Must be implemented or verified; same integer seed with sequential draws is insufficient                             |
| Family/target-aware statistics, work ledger, and transfer manifests  | Must be implemented; no single-target aggregate may stand in for them                                                |

Do not restore the entire removed experiment framework. A focused pruning
module plus a manifest-driven experiment script is sufficient; do not add
external optimizer libraries or target-specific preparation shortcuts.

Required implementation tests before E1:

1. Exact G1/G2/P quotas at both sizes; protected initial gates; stable origin
   IDs; deterministic ties and parameter remapping; no implicit extra pruning.
2. A deleted unitary loses its noise window; a retained zero-angle gate does
   not; dense reference evolution validates all four E0 targets.
3. D scores equal explicit deletion; I gradients equal finite differences;
   all scoring tests include nonzero offsets/scales or reject unsupported
   parameterizations explicitly rather than silently computing wrong scores.
4. Identical 0/N starting arrays and topologies; correct repair and total-update
   counts; continued-training controls are not accidentally frozen.
5. Common-uniform marginals reproduce each channel despite different masks,
   orders, and Ballarin angles; role streams are disjoint.
6. Selection ignores test data, ties/fallbacks are deterministic, and interrupted
   runs resume exactly; serial and parallel outputs agree across all four E0
   targets, excluding timing fields.
7. Dense calibration/discretization checks, paired statistics, family weights,
   finite-suite bootstrap, missing-cell handling, and honest cost attribution.
8. Unit versions of the Section 12 gradient normalization/reference checks,
   exact hybrid switches, shared starts, and E0H on all four targets. The full
   cold/warm checkpoint audits follow E1 and must pass before E1H.

Run the repository's required lint and relevant tests after implementation.
There is deliberately no runnable CLI example for E1-E5 yet: the recent
single-target scripts and legacy benchmark presets do not implement this
protocol. Record completion only after the required artifacts actually exist.

## 12. Optimization-paper review and controlled hybrid ablation

### What was supplied, and what it establishes

On 2026-09-05 the author supplied the title, preamble, authors, and abstract of
_Hybrid Krotov gradient descent Optimization for Quantum Machine Learning_.
No main text, algorithms, theorem statements, hyperparameter tables, or
supplement were included. The repository's `manuscript/main.tex` is a
different state-preparation manuscript, not the missing optimization paper.
Do not claim to have audited the full paper or to reproduce its tuned settings.

The supplied abstract identifies an online-to-batch hybrid as the principal
method; states exact-gradient equivalence for the batch direction under its
fixed-generator/terminal-loss assumptions; describes convergence of the
hybrid's batch tail after a finite switch; and reports tuned classification
comparisons with three seeds per configuration. It does not specify numerical
step sizes, phase lengths, initialization distributions, decay schedules,
line searches, or stopping thresholds. Classification accuracy does not
establish state-preparation fidelity or noisy-optimization performance.

These observations justify adding the missing optimizer ablation and
correctness checks, **not** changing calibrated noise rates, pruning quotas,
target coverage, or the main study's learning rates by inference.

### Parameter and update conventions that must be explicit

For the current primitives, write

```text
U_j(theta_j) = exp[-i * alpha_j(theta_j) * G_j / 2]
alpha_j(theta_j) = scale_j * theta_j + offset_j
L0(theta) = 1 - |<target|psi(theta)>|^2
Lhat_R(theta) = (1/R) * sum_r (1 - |<target|psi_r(theta)>|^2)
batch: theta_new = theta_old - eta * g(theta_old)
```

Use radians, include `scale_j` and the generator's factor 1/2 in the gradient,
and do not divide by the number of gates or parameters. Noiseless loss has
no additional factor 1/2. Average independent trajectory contributions once
by R; the existing cross double sum is normalized by R squared. Record these
conventions alongside learning rates. Transferring an MSE/BCE classifier's
learning rate without reconciling its loss and sample normalization is not
justified. Also, classification online epochs may update once per sample;
state preparation here uses one target and one gate sweep per outer update.

Batch evaluates all contributions at the same old parameter vector and then
changes all parameters together. Online means **gate-sequential stale-adjoint
updates**, not trajectory resampling or training on different target states:
form costates at the old vector; sweep gates in circuit order, using the
already updated prefix and the old costates; update each gate once. In the
noisy online implementation, average its 32 trajectories at each gate (or
use the specified cross double sum); do not perform 32 serial parameter
updates. Freeze the realized noise maps within that entire gate sweep.
Each target is optimized separately; no single parameter vector is trained
jointly across the four targets.

The [current optimizer](../src/mqt/yaqs/optimization/krotov.py) implements
these phase choices. `switch_iteration=s` means iterations 1..s are online
and s+1..T are batch. Schedule counters are phase-local in that API. Log the
actual step at every update, not just a base rate. A phase-local decaying
batch schedule would restart at its base rate after switching; do not hide
that restart in a comparison.

### What the noiseless theorem does not guarantee here

The abstract's convergence statement is conditional. For example, for an
exact gradient and an L-smooth objective the descent lemma gives

```text
L(theta - eta*g) <= L(theta) - eta*(1 - L*eta/2)*||g||^2.
```

Thus smoothness alone does not certify an arbitrary chosen constant rate;
a sufficient positive-descent condition is 0 < eta < 2/L. We have not
established that bound for this benchmark or inspected the paper's precise
step-size assumptions. A finite 200-update run with exponential decay is
not a convergence theorem either. Report nonmonotone steps rather than
asserting that "Krotov guarantees monotonicity" or adding unrecorded retries.

In this protocol's exact, state- and angle-independent Pauli channel, fixing
the random outcomes gives an ordinary unitary circuit for each trajectory.
Consequently the **independent** batch contribution can be checked against
the exact gradient of Lhat_R with those outcomes held fixed. Resampling at
the next update changes Lhat_R; deterministic monotonicity does not carry
over to the true expectation. This is a deduction for the specified channel,
not a theorem quoted from the missing paper. It does not extend automatically
to angle-dependent Ballarin probabilities, state-dependent nonunitary jumps,
or truncated tensors.

The current `cross` contribution is a different finite-ensemble double sum.
It is not generally the exact derivative of the _reported_ same-trajectory
Lhat_R. Do not label its finite-R trace ordinary empirical gradient descent,
or demand that it pass the independent-gradient identity as if that were its
definition. Test it separately. Likewise, freezing random numbers makes an
objective reproducible but does not establish a descent guarantee.

### E1H: small, explicitly provisional optimizer comparison

Run all four Section 2 development targets at n=6 with replicas 11,29,47.
Use the dense b=2 BMPD circuit (78 one-qubit gates, 30 two-qubit gates,
108 parameters) throughout. Do not prune, grow, or select a resource level
inside this ablation. Cross **both** starting contexts with every setting:

- `cold`: the Section 5 direct b=2 initial vector from the corresponding E1
  initialization master vector, before any optimization.
- `warm`: the corresponding E1 L-route b=2 pre-refinement checkpoint after
  all 520 noiseless construction/continuation/repair-control updates.

Reuse those arrays and circuit lists exactly, including their source hashes.
The warm context tests refinement; the cold context tests whether online
updates help earlier optimization that the previous short hybrid diagnostic
never investigated. Cold and warm are separate strata, not cost-matched
methods: charge the warm checkpoint's preparation cost.

| Setting ID | Total updates | Online updates | Batch updates | Online rate  | Batch rate                     |
| ---------- | ------------: | -------------: | ------------: | ------------ | ------------------------------ |
| `B-exp`    |           200 |              0 |           200 | Not used     | 0.1*exp[-0.01*(k-1)], k=1..200 |
| `B-flat`   |           200 |              0 |           200 | Not used     | Constant 0.1                   |
| `H20-flat` |           200 |             20 |           180 | Constant 0.1 | Constant 0.1                   |
| `H50-flat` |           200 |             50 |           150 | Constant 0.1 | Constant 0.1                   |

All flat schedules have decay 0 and no step-size change at the switch.
H20 versus B-flat and H50 versus B-flat isolate the update ordering/switch
at a common rate; B-exp versus B-flat isolates the existing decay choice.
Comparing hybrid only with B-exp would confound these effects. There is no
line search, momentum, clipping, early stopping, or extra restart in E1H.
The value 0.1 agrees with the current optimizer defaults; 20 is its default
switch and 50 is a proposed longer online phase. These are **our provisional
settings, not hyperparameters extracted from the supplied paper**. The
positive online rate is not evidence that its stale-adjoint sweep descends.

For every start/setting, run each of these three objective/update arms:

| Arm ID | Training objective                  | Noisy update        |
| ------ | ----------------------------------- | ------------------- |
| `Z`    | Noiseless L0                        | None                |
| `NI`   | Reference local-depolarizing Lhat_R | `independent`, R=32 |
| `NX`   | Same reference noisy objective      | `cross`, R=32       |

NI and NX use Section 7's exact factor sampler and per-update resampling;
neither freezes a training ensemble across iterations. All other Section 5
precision, normalization, and gate-window settings apply. NI versus NX at
the same setting/start isolates the trajectory update rather than calling
both "the noisy gradient". There are 4*3*2*4*3 = **288 training endpoints**,
each of 200 updates; warm-start preparation, validation and tests add cost.
Pure online to the very end is not included: the tested hypothesis
is a finite online phase followed by a substantial batch tail.

For each endpoint, select among updates 0,50,100,150,200 using the same 512
noisy validation trajectories as in Section 8, including for Z. Do not add
extra selection checkpoints at a hybrid switch. After all choices are frozen,
evaluate the selected circuits with 2048 fresh paired test trajectories and
also record exact noiseless fidelity at all five checkpoints. Preserve update
200 as a non-selected optimization diagnostic, so noisy checkpoint selection
does not conceal behavior of the noiseless training objective. Retain every
target, start, setting, arm, and failure; do not report only the best hybrid.

E1H uses root 2026090511 for its own training/validation/test draws, disjoint
from E1. Only the above initialization/checkpoint artifacts are inherited.
Use training-stage strings `optimizer_ablation_cold` and
`optimizer_ablation_warm`; do not add setting or NI/NX identity to training
draw keys, so matched updates share uniforms. Validation/test use Section 9's
null stage/resource/update fields, coupling every endpoint for a target and
replica. Both starts therefore share validation/test uniforms but have distinct
training streams. Never use an E1 test outcome to select the warm checkpoint.

Report every target and family-balanced aggregate, split by cold/warm and
Z/NI/NX. Report the three schedule contrasts above and NX-minus-NI for every
schedule, along with full cost accounting; use the Section 10 paired bootstrap
with three rather than five replicas and label intervals **exploratory**.
No paper-level win or rejection follows from searching these development
results. Do not silently replace the ten main methods with a target-specific
winning schedule. Any promotion of hybrid or NI into the locked main study
requires a new globally specified protocol revision, matched comparator
tuning rules, and evaluation on the reserved targets before making claims.

### Engineering and interpretation gates

E0H runs the same four development targets with replica 11 before E1H, using
root 2026090510. Reuse E0's b=2 cold vector and L pre-refinement checkpoint
(14 construction/control updates). Use four refinement updates, switch 1
for H20-flat and 2 for H50-flat, checkpoints 0,2,4, four training, eight
validation and sixteen test trajectories. B-exp retains its stated formula
for k=1..4. All other settings, arms, and two starts remain, giving **96
training endpoints**. These reduced switches are engineering-only; do not
substitute them into E1H or interpret their fidelities scientifically.

Before E1H, for all four development targets on b=2, at their cold and warm
checkpoints:

1. Compare noiseless batch contributions with an independently implemented
   dense statevector adjoint and central finite differences (epsilon=1e-6,
   componentwise absolute tolerance 2e-6). Include negative angle scales;
   retain the existing separate tests of offsets and nonunit scales.
2. Repeat for NI with R=4 fixed exact-Pauli trajectories and the exact same
   maps in positive/negative finite-difference evaluations. Duplicating every
   trajectory must not scale the gradient. Check NX against its own dense
   double-sum reference, not the NI identity; require both to reduce to the
   noiseless gradient when noise is disabled. Dense-reference contribution
   agreement uses absolute tolerance 1e-10.
3. Compare online one-sweep outputs with independent dense stale-adjoint
   sweeps, including NI and NX with fixed maps, to parameter tolerance 1e-10.
   Assert the exact phase counts, k=s and k=s+1 boundary labels, rates, gate
   order, loss/gradient averaging, and no extra update on each trajectory.
4. Verify identical starts across all settings/arms, stream disjointness,
   uninterrupted versus resumed phase schedules, test-blind selection, and
   the additional execution/cost counts. No separate scientific GD competitor
   is needed once its identical noiseless batch update has been verified.

The full paper is still needed to finish the detailed hyperparameter audit.
Request its main text and supplement, then explicitly check loss scaling,
online/batch rates and schedules, switch rule and phase budget, initialization,
sample-update normalization, stopping/line-search rules, and the matched
competitor search budgets. Keep current numerical choices provisional until
that review; do not describe E1H as a faithful paper reproduction meanwhile.
