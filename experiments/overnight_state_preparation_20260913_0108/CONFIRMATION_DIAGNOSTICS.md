# Post-confirmation history diagnosis

The frozen candidate failed confirmation. At the primary common CPU cutoff,
structural mean2 L-BFGS achieved **88.534223%**, versus **88.095145%** for the fixed
primary baseline, structural weighted Adam, and **88.209294%** for structural
adaptive online. Its gains were **+0.439079** and **+0.324929 percentage points**,
with **3/5 target wins and 5/10 case wins** against each. These miss the frozen
0.5-point and complete target-consistency requirements. This report explains saved
outcomes after confirmation; it does not confirm a new hypothesis, select a
different endpoint, tune the recipe, or change that failed gate.

## Coverage and custody

All ten cases, seven roles, and three existing comparison panels were retained.
Every case passed the frozen `best_candidate.read_case` validator, including its
complete physical child inventory. All 60 trained-arm histories were checked
against their parent receipt checksums and byte hashes. The reduction also binds
758 source files, checked unchanged before and after analysis. No target loader,
state evolution, gradient evaluation, new fidelity scoring, or compiler was run.

The new machine-readable artifact is
[`CONFIRMATION_HISTORY_REDUCTION.json`](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_HISTORY_REDUCTION.json).
Its byte SHA256 is
`514df43e9f7a8b1853f8e754d0ff97172c98cd49ebcdabfd9fa4d039cc6671e9`.
It cross-checks the previous
[`CONFIRMATION_CASE_DIAGNOSTICS.json`](../../output/overnight-state-preparation-20260913-0108/validation/CONFIRMATION_CASE_DIAGNOSTICS.json)
against the authenticated case scores. The parent content checksum is
`5a8a4980116a955ab5146c4e146c423d7f84ff92ce66a3c23167cfe394e0cb4e`.
The reduction consumed **19.257139 CPU seconds / 23.401408 wall seconds**. This is
read-only audit work; exploratory shell reads and prose engineering time remain
unmeasured and are not represented as zero.

Fidelity averages weight the eleven noisy profiles equally, then the two replicas
within each target, then the five targets equally. “Training2” means the exact
equal average of `depolarizing_1s_all` and `dephasing_1s_all`. All differences below
are percentage points. Gate counts refer to the physical circuit before the
separate Ballarin materialization.

## Every confirmation case

C, W, and A denote structural mean2 L-BFGS, weighted Adam, and adaptive online.
The primary cutoff is the minimum complete refinement CPU among the four
structural roles, applied to all seven roles. “Later C gain” compares C's saved
full-cap endpoint with its primary endpoint; it is not another primary result.

| Target     | Replica | C fidelity % | W fidelity % | A fidelity % |       C−W |       C−A | Later C gain | C/W/A two-qubit gates |
| ---------- | ------: | -----------: | -----------: | -----------: | --------: | --------: | -----------: | --------------------- |
| TFIM ferro |      11 |    94.137363 |    92.414211 |    94.259240 | +1.723152 | −0.121877 |    +0.223592 | 10/14/9               |
| TFIM ferro |      29 |    94.360955 |    92.421897 |    93.520147 | +1.939058 | +0.840808 |    +0.000000 | 10/14/11              |
| TFIM para  |      11 |    95.590295 |    95.875820 |    96.226949 | −0.285525 | −0.636654 |    +1.087745 | 10/9/10               |
| TFIM para  |      29 |    95.573427 |    96.897808 |    96.362368 | −1.324381 | −0.788941 |    +1.104613 | 10/9/7                |
| Haar 2     |      11 |    83.666996 |    82.160563 |    81.757602 | +1.506433 | +1.909393 |    +0.080354 | 24/22/27              |
| Haar 2     |      29 |    83.326881 |    82.606974 |    83.021787 | +0.719906 | +0.305093 |    +0.104222 | 26/25/27              |
| Haar 3     |      11 |    81.288750 |    80.008663 |    79.337691 | +1.280087 | +1.951059 |    +0.075329 | 23/27/24              |
| Haar 3     |      29 |    65.772773 |    66.092517 |    65.734770 | −0.319744 | +0.038003 |    +0.183772 | 20/22/23              |
| MPS bond 3 |      11 |    95.812369 |    96.107675 |    96.039252 | −0.295306 | −0.226883 |    +0.052440 | 10/8/8                |
| MPS bond 3 |      29 |    95.812425 |    96.365318 |    95.833137 | −0.552893 | −0.020712 |    +0.052384 | 10/6/9                |

For completeness, the primary all-eleven-channel means of the other roles are
87.689991% for fixed mean2 L-BFGS, 86.470996% for fixed noiseless Adam, 86.966933%
for structural exact Adam, and 77.633670% unchanged. These roles remain present in
every machine-readable panel.

## What the saved histories establish

**TFIM para: substantial polishing was not yet available at the primary cutoff.**
C's primary endpoints are the completed round-14 and round-13 checkpoints. The
cutoffs are 27.275538 and 26.390122 refinement CPU seconds, whereas C finishes at
about 31.15 and 31.23 seconds. Both runs eventually accept all 15 deletions and
reach the same numerical endpoint. The remaining deletions are one-qubit
rotations; the primary and full circuits both retain ten two-qubit gates.

There is a precise decomposition for the **training objective**. The last
structural round leaves training2 fidelity at 93.952963%. The saved L-BFGS final
polish state, with its loss bound to the same final parameter vector, reaches
95.291851%: **+1.338887 points**. Terminal compilation then fuses three one-qubit
rotations and yields 95.402953%: another **+0.111102 points**. The full eleven-channel
fidelity rises by 1.087745/1.104613 points from the primary endpoints. Its exact
split among the intervening steps is unavailable because no all-channel score of
the raw final circuit was saved. Thus the evidence locates unfinished useful work
after the cutoff; it does not justify replacing the cutoff or prove that another
allocation would win at the original cost.

**MPS bond 3: weighted Adam reaches better noisy endpoints with fewer two-qubit
gates.** C accepts ten deletions, including two two-qubit gates, and stops the
structural phase at round 11 in both replicas. In that round the retained repair
scores 93.488701% training2, above the repaired deletion alternatives at 93.380495%
and 93.260096%. Those alternatives remove `ry` on site 2 and `ryy` on sites 0–1.
The frozen decision is therefore consistent with its saved exact branch scores.
It is not a failed or nonfinite branch.

W instead accepts all 15 deletions, including four/six two-qubit gates, leaving
eight/six two-qubit gates versus C's ten/ten. Its primary noisy fidelities are
96.107675%/96.365318%, above C's approximately 95.8124%, while its noiseless
fidelities are **99.201889%/98.924223%**, below C's approximately **99.4333%**.
This is an observed noisy-versus-noiseless tradeoff, accompanied by different
structural paths. It does not establish that gate count alone caused the gain or
that changing C's shortlist or repair budget would recover it. Later polishing
and two terminal one-qubit fusions add only about 0.0524 points to C's primary
all-channel fidelity; C still loses to W at full cap.

**Haar 3 replica 29 is a shared difficult initialization, not an isolated C
failure.** The unchanged noisy fidelity is 47.192516% versus 58.039412% in replica
11; unchanged noiseless fidelity is 54.066884% versus 67.248415%. The rank-two probe
overlap is the same 0.415641 in both and triggers the legacy constructor fallback.
The two saved constructed endpoints differ. All six trained roles also finish
worse in replica 29 under both the primary and full-cap panels. C improves its
replica-29 noisy fidelity from 47.192516% to 65.772773% at the primary cutoff, yet
trails W there by 0.319744 points. This supports a shared starting-point or
optimization-path sensitivity. It does not isolate the constructor, final circuit
expressivity, or a particular local optimum as the sole cause.

**The structural and compiler records show completed, costly work.** Across all
ten cases C accepts 140 deletions, including 46 two-qubit deletions. W accepts 130,
including 43 two-qubit deletions. Their terminal compilers remove another 20 and
16 one-qubit gates respectively and no two-qubit gates. The 40 structural-arm
histories contain no explicitly failed repair branches. C spends 216.674856 of
434.242732 recorded training CPU seconds on unselected repaired branches and
their scores; W spends 163.517036 of 371.253972 seconds. These are necessary search
costs already charged in the saved clocks, not additional work to add to the
resource ledger. The totals do not measure branch usefulness by themselves.

## Surrogate mismatch and limits of the explanation

For C versus W and C versus A, the sign of the training2 difference agrees with
the sign of the eleven-channel difference in **all 60 case comparisons** across
the three saved panels. In particular, C already loses training2 on the primary
TFIM para and MPS bond-3 cases where it loses the final metric. The saved evidence
therefore does not support a reversal caused solely by evaluating the broader
noise family as the main explanation for these losses.

The two objectives still differ. For example, the target-averaged primary MPS
bond-3 difference C−W is −0.956699 points over the five depolarizing profiles,
+0.004744 over the five dephasing profiles, and +0.094680 for Ballarin. Pairwise
endpoint sign agreement cannot establish that training2 is globally optimal for
the eleven-channel objective, or describe unmeasured intermediate tradeoffs.

At full cap, C's advantage becomes +0.735640 points over W, with 4/5 target wins,
and +0.600137 over A, with 5/5. At the all-six-trained minimum CPU sensitivity,
the gains are +0.563770 and +0.117057 points. These pre-existing secondary panels
show the sensitivity of the comparison to available work. They neither replace
the failed primary panel nor supply a new confirmation opportunity. No method
change or new run is authorized by this posthoc explanation.

Any future change motivated by these observations would be a new development
recipe. Fresh optimizer or construction seeds could test repeatability, but they
would not make these already inspected q6 target identities untouched again.
