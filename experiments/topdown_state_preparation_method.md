# Top-down state preparation: B with online Krotov and L-BFGS

**Our main method and baseline for future development is pruning B with online
Krotov followed by L-BFGS batch refinement during noiseless construction.**
Use the following phase choices:

| Phase                  | Main method                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| Noiseless construction | 100 online Krotov sweeps at 0.7, then 100 safeguarded L-BFGS batch updates; restore the best parameters |
| Pruning                | B: score exact gate-removal damage once and simultaneously remove gates below 0.01                      |
| Noisy fine-tuning      | Existing 100 online + 100 batch cross-trajectory updates at 0.2, using 100 frozen weighted histories    |
| Assessment             | Independent validation for depth selection; exact density-matrix final fidelity                         |

**D is retained as a research reference, not a routine development arm.** Reuse
its implementation and completed results; rerun it when an experiment specifically
needs the more expensive deletion-and-repair comparison. Future method changes
should use the current B + online Krotov + L-BFGS recipe as their baseline.

**Fine-tuning every depth candidate remains optional** (section 5): it costs
substantially more than constructing and fine-tuning one circuit. The completed
L-BFGS comparison used this optional search; its improvements are established
for that setting. This choice of main method does not introduce a new stopping
rule or establish L-BFGS results for the archived first-90% pipeline. Each gate
has an independent scalar parameter; surviving parameters are retained after pruning.

**Noiseless construction returns the best observed parameter vector**, including
the initial vector, rather than a worse final iterate (section 1.1).

**Longer noiseless refinement can reduce the depth needed for a fidelity target.**
The twelve-qubit Haar check demonstrates this at 99%; optional fixed-depth
refinement and its limits are recorded in section 1.3. Default budgets remain unchanged.

**Keep $M=100$: the $M=400$ comparison is already completed and showed no
meaningful fidelity benefit for this setup (section 3.1).**

**Future competitor experiments focus on noise-aware training.** Compare this
method with NACL and RobustState, plus Fontana on real targets; fixed VQSP and
unpruned layerwise are internal controls. Exclude AQCE and BAA-LRSP from future
performance runs because they synthesize without noisy optimization, while
preserving their completed results. IQSP remains on hold until the Gaussian
pilot's local initialization failure is corrected. The
[active comparison protocol](state_preparation_competitors/NOISY_METHODS.md)
records the scope and target restrictions. These comparisons establish results
within the noise-aware subset, not superiority over all state-preparation methods.

## Numerical backends

The current pipeline now has an MPS implementation, including empirical weighted
cross-trajectory updates in both online and batch refinement. The shared
[backend runner and validation protocol](topdown_mps.md) preserve construction,
pruning B, optimizer budgets and logical noise. Exact MPS execution is the
initial comparison mode; bond caps, SVD thresholds and target compression are
explicit approximations requiring convergence checks. The backend implementation
alone does not establish a scalability advantage. The scalable runner uses
independent final trajectory batches; exact density assessment remains an
optional small-system check.

The MPS execution reuses mixed left/right overlap environments and batches
trajectory pairs, sharing unchanged private tape tensors. This accelerates the
same mathematical update; both cross weights, derivative SVDs, noise banks and
optimizer policies are preserved. Matching canonical centers alone is not used
as a shortcut for overlaps between different states. Performance comparisons
must record the implementation version and use fresh authenticated outputs.

The [matched truncation check](../output/topdown-mps-truncation-20260923/REPORT.md)
completed twenty full noisy refinements on twelve-qubit critical-TFIM circuits
at depths 1 and 3. Untruncated MPS matches statevector execution; threshold
1e-6 changes final held-out fidelity by at most 0.000213 percentage points.
Hard caps 2-16 substantially degrade fidelity in the present implementation.
An independent identity-gate check exposes gauge-dependent error from splitting
away from the orthogonality center. Keep exact execution as the reference;
center-aware truncation and further convergence checks are needed before
adopting aggressive caps. These measurements do not establish tolerances for
other targets.

## 1. Noiseless construction

For normalized target $|t\rangle$ and brickwall circuit
$C(\theta)=G_L(\theta_L)\cdots G_1(\theta_1)$, define

$$
F_0(\theta)=|\langle t|C(\theta)|0^n\rangle|^2,
\qquad \mathcal L_0=1-F_0.
$$

Grow one layer at a time, retaining trained parameters. Initialize new
parameters from $\mathcal N(0,\sigma^2)$: $\sigma=0.05$ initially and $0.001$
thereafter. The main construction optimizer performs **100 online Krotov sweeps
at $\eta=0.7$, then 100 L-BFGS batch updates** using section 1.2. Restore the best
actual parameters using section 1.1. The depth index counts complete even/odd
brickwall cycles. In the tested optional search, run this full budget at every
visited depth and use section 5 for stopping and selection.

The **archived cheaper single-circuit baseline** instead used up to 100 online
plus 100 fixed-step batch updates at $\eta=0.7$, stopping immediately at
$F_0\geq0.90$ or reporting failure at depth 16, then pruning and fine-tuning once.
That 90% cutoff is a computational heuristic. Its archived settings and results
remain a reference; applying the main L-BFGS recipe to that stopping mode has
not yet been measured.

### 1.1. Save and restore the best noiseless parameters

At each depth, initialize a separate checkpoint with the starting parameters:

$$
\theta_{\mathrm{best}}\gets\operatorname{copy}(\theta^{(0)}),
\qquad F_{\mathrm{best}}\gets F_0(\theta^{(0)}).
$$

After every complete online sweep or batch update $k$, evaluate $F_0(\theta^{(k)})$.
If it strictly exceeds $F_{\mathrm{best}}$, replace both the saved fidelity and
the **actual parameter vector** with copies of the current values. Exact ties
keep the earlier checkpoint. At the stopping threshold or update-budget limit,
restore and return

$$
\theta\gets\theta_{\mathrm{best}},\qquad
F_0(\theta_{\mathrm{best}})=\max_{0\leq k\leq K}F_0(\theta^{(k)}).
$$

Use this restored vector for pruning or as the unpruned parent for the next
layer extension. Reset the checkpoint when the circuit depth changes. The active
optimization trajectory continues normally between updates; restoration happens
at the end of each construction run, not after every decrease. This rule adds
no early stopping, step rejection or change to the learning rate.

The [completed stability comparison](topdown_optimizer_stability_findings.md)
tested this rule as `best_only`: the recorded TFIM checkpoint returns **99.542%**
instead of **7.037%**, with about **0.35%** extra optimizer CPU across the 124
matched construction inputs. That timing excludes downstream work. Both B and D
use the retained construction checkpoint.

The retention rule was tested in the [stability driver](topdown_optimizer_stability.py)
and is also used by the main L-BFGS construction in section 1.2. Archived
reference drivers retain their original behavior. The full comparisons tested
the optional depth search, so downstream gains for the cheaper first-90%
pipeline remain unmeasured.
Applying retention to D's repair or noisy fine-tuning is a separate change;
those stages still return their final iterates.

### 1.2. Main batch optimizer: L-BFGS

The [completed BB/L-BFGS comparison](topdown_batch_refinement_findings.md)
changes only noiseless batch construction. It reruns the then-current strong controls:
best retention plus backtracking for B, and best retention alone for D.

**Use L-BFGS with B as the main construction policy.** When D is needed as a
reference, use its best-only fixed-step construction setting. In the tested
optional every-depth search, L-BFGS with B improves mean final noisy
fidelity by 0.556/0.483 percentage points under the IBM/Google logical profiles,
at 21.2%/11.5% less measured search CPU. Compiled two-qubit depth rises slightly,
and some targets regress; this is an observed fidelity/classical-compute
tradeoff, not dominance in every metric. With D, L-BFGS costs more and its
fidelity changes are mixed (+0.078/-0.333 pp), so it does not replace D's
baseline. BB gives lower mean final noisy fidelity against both primary
baselines; its approximately 11% CPU saving with B is only an optional tradeoff.

After the unchanged online phase, L-BFGS uses the true noiseless loss gradient
and a ten-pair positive-curvature inverse-Hessian approximation:

$$
p_k=-H_k\nabla\mathcal L_0(\theta_k),\qquad
\mathcal L_0(\theta_k+\alpha p_k)
\leq\mathcal L_0(\theta_k)+10^{-4}\alpha\nabla\mathcal L_0(\theta_k)^Tp_k.
$$

Reset history at every depth, skip unsuitable curvature pairs, and reset to
steepest descent if the direction is invalid. Start the line search at 0.7 with
empty history and at 1 otherwise; halve up to 12 times, retaining the current
state on failure. Save and restore the best actual parameters as in section
1.1. The [frozen protocol](topdown_batch_refinement_protocol.md) specifies all
safeguards, and the [batch-refinement driver](topdown_batch_refinement.py)
implements the tested policies.

These results cover nine six-qubit targets and two optimizer seeds in the
optional depth search. They do not establish the gain for the cheaper first-90%
pipeline, whose archived settings are described above. D repair and noisy
cross-trajectory fine-tuning retain their existing updates and M=100; this
comparison does not apply L-BFGS to those phases. Reuse these completed results
for unchanged settings rather than treating BB/L-BFGS as untested options.

### 1.3. Optional: refine longer before adding circuit depth

A finite per-depth optimization budget can leave useful circuit capacity unused.
Before attributing low fidelity to insufficient depth, compare further noiseless
L-BFGS refinement of a saved unpruned checkpoint with another layer extension.
Retain the best parameters and record fidelity improvement, gate counts and CPU
cost. An update or depth cap is budget exhaustion, not evidence of convergence.
This is an optional budget-allocation experiment, not a new automatic stopping
rule or a change to the default 100 online + 100 batch construction updates.

The completed twelve-qubit `haar_random_1` check used exact statevectors, the
unchanged circuit architecture, and no noise or pruning. Each saved depth received
one continuous call of up to 2,000 additional safeguarded L-BFGS updates, with
curvature history reset once at the start and retained throughout that call.

| Fixed depth | Starting ideal fidelity | Final ideal fidelity | Additional training CPU | First 99% crossing |
| ----------- | ----------------------- | -------------------- | ----------------------- | ------------------ |
| 66          | 95.1165%                | 97.9955%             | 63.1 minutes            | Not reached        |
| 87          | 98.0503%                | 99.1999%             | 80.3 minutes            | Update 1,488       |

Earlier growth first recorded 99% at completed depth 107. Refinement at depth 87
achieves that threshold with **2,871 instead of 3,531 logical two-qubit rotations
(18.7% fewer)**. Including historical construction through depth 87, CPU to its
first 99% crossing was **8.42 hours versus 11.11 hours** for the earlier growth
checkpoint (24.2% less). These are retrospective preparation-path costs, excluding
the exploratory cost of choosing a depth; growth timing is at completed depths,
whereas refinement crossings are recorded per update.

**Neither refinement reached 99.9%, and both were still improving at the update
limit.** The result establishes a tradeoff between optimization effort and depth
for this target and initialization, not a minimum depth or a noisy-fidelity gain.
It motivates testing longer refinement before enlarging a still-improving circuit;
it does not establish a sufficient budget for 99.9%.

[Verified results and plots](../output/q12-haar-refinement-20261004/REPORT.md) ·
[Numerical summary](../output/q12-haar-refinement-20261004/validated-summary.json) ·
[Frozen refinement protocol](topdown_haar_refinement.md)

### 1.4. New initialization option: screen product states with a zero fallback

Before depth-one optimization, sample 1,000 independent Haar single-qubit
product states and include $|\phi_0\rangle=|0^n\rangle$. Select

$$
r_* = \arg\max_{0\leq r\leq1000}|\langle t|\phi_r\rangle|^2,
\qquad F_{\rm product}(\phi_{r_*})\geq|\langle t|0^n\rangle|^2.
$$

Encode the selected product state in the existing initial U3 angles, preserving
all other initial parameters and subsequent updates. Exact ties keep zero first.
Use independent reproducible candidate pools for different optimizer seeds and
count sampling/selection time in preparation cost. Save all sampled angles and
overlaps once, along with the selected parameters; a completed initialization is
not resampled on resume. The physical circuit still starts from zero.

The inequality protects product-layer starting overlap only. Initial entanglers
can change that score; record full depth-one fidelity separately. Improved final
noisy fidelity or runtime is **not yet established for this screened option**.
Use the [matched screening driver](topdown_product_screening.md) to compare it
against near-zero starts with the current 100+100 construction budget, pruning B,
100 weighted training histories and independent validation unchanged. Twelve-qubit
unscreened random-product results motivate this test; they do not validate
best-of-1,000 selection or justify a universal initialization advantage.

## 2. Exact noiseless pruning

Deleting a complete gate gives the signed sensitivity

$$
\Delta_i=F_0(C)-F_0(C\setminus G_i),\qquad \varepsilon=0.01.
$$

All candidates can share one forward/backward vector pass, with
$O(L2^n)$ work and storage. Deletion also removes that gate's future noise window.

| Method                     | Procedure                                                                                                                                                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B — main method**        | Score every gate once; simultaneously delete all gates with $\Delta_i<\varepsilon$.                                                                                                                                          |
| **D — research reference** | Choose $i=\arg\min_i\Delta_i$. If $\Delta_i<\varepsilon$, delete it, perform **5 online + 5 batch noiseless updates** at $\eta=0.7$, then recompute all scores. Repeat until no candidate qualifies or the circuit is empty. |

Ties follow gate order. D skips repair for an empty circuit. The threshold is
**per deletion**, with no cumulative fidelity bound or rollback. D's fixed-rate
repair returns its final iterate and can reduce fidelity; this occurred in the
tested TFIM case. Section 1.1 currently protects construction, not these repair
updates.

## 3. Frozen noise histories and exact duplicate compression

Sample $M=100$ complete, parameter-independent Pauli histories **before**
fine-tuning and freeze them throughout optimization (common random numbers).
For gate window $i$, with Pauli rates $\lambda_{i,a}$ and
$\kappa_i=\sum_a\lambda_{i,a}$, the implemented one-jump law is

$$
\Pr(N_i=I)=e^{-\kappa_i},\qquad
\Pr(N_i=P_a)=(1-e^{-\kappa_i})\frac{\lambda_{i,a}}{\kappa_i}.
$$

Each acted-on qubit has $X,Y,Z$ rates $\gamma_1/3$; a two-qubit gate also has
the listed correlated jumps, sharing total rate $\gamma_2$ equally.

| Historical logical profile | $\gamma_1$       | $\gamma_2$       | Correlated jumps |
| -------------------------- | ---------------- | ---------------- | ---------------- |
| IBM                        | $3\times10^{-4}$ | $3\times10^{-3}$ | $XX,ZZ$          |
| Google                     | $6\times10^{-4}$ | $5\times10^{-3}$ | $XX,YY,ZZ$       |

These are simulation coefficients, not calibrated average gate error rates.
If $m_0$ histories contain no jumps, keep one representative with weight
$w_{\mathrm{nj}}=m_0/M$ and every jump history with weight $1/M$:

$$
K=M-m_0+\mathbf1_{m_0>0},\qquad
\widehat F(\theta)=\sum_{r=1}^{K}w_r
|\langle t|\psi_r(\theta)\rangle|^2.
$$

Omit the representative when $m_0=0$. Recompute retained states after parameter
updates; compression preserves the full sampled objective and update signal.
For a nonempty circuit, a bank with zero jumps is a **coverage failure**:
do not force jumps or resample. Empty circuits remain explicit diagnostics.

### 3.1. Completed experiment: M=400 (2026-09-15)

Increasing the raw training bank from **100 to 400 histories** was tested on
all nine six-qubit targets, seeds 300/400, pruning B/D and both noise profiles.
Only $M$ changed; construction, pruning, optimization, validation and selection
stayed fixed. The original 100 histories formed the prefix of each larger bank.
Final fidelity was evaluated with exact density matrices after selection.
This historical comparison predates best-iterate retention; it was not rerun
with the construction rule in section 1.1.

| Pipeline                        | Mean fidelity change, percentage points | Total CPU ratio, M=400 / M=100 |
| ------------------------------- | --------------------------------------- | ------------------------------ |
| Default single-circuit pipeline | +0.0209                                 | 1.18x                          |
| Optional depth search           | -0.0016                                 | 1.40x                          |

Matched noisy fine-tuning alone cost **2.50–2.77x** as much CPU. The negligible
fidelity changes do not justify that cost; **$M=100$ remains the default**.
This conclusion applies to the tested setup, not every possible noise model or
system size.

**Do not repeat this isolated M=100 versus M=400 comparison with unchanged
settings; reuse the completed results.** Revisit it only for a documented new
scientific question, such as a substantive change to the optimizer, noise model
or system size. This is a completed experiment, not a pending method improvement.

[Findings and interpretation](topdown_m400_findings.md) ·
[Frozen comparison protocol](topdown_m400_protocol.md) ·
[All target results and audit](../output/topdown-depth-search-q6-m400-20260915/M_COMPARISON.md)

## 4. Hybrid updates

Write $A_i=(\partial_{\theta_i}G_i)G_i^\dagger$ and let $|\psi_{r,i}\rangle$
be the state just after gate $i$, before its noise. The legacy fixed-step rule uses
$\theta_i\leftarrow\theta_i-\eta s_i$:

- **Online:** compute backward states at the old iterate, hold them fixed,
  then update gates sequentially and propagate each updated forward prefix.
- **Batch:** evaluate all signals at the same iterate, then update simultaneously.

For noiseless training, backpropagate the terminal costate
$|b_L\rangle=-\langle t|\psi_L^{\mathrm{old}}\rangle|t\rangle$ and use

$$
s_i^{(0)}=2\operatorname{Re}\langle b_i|A_i|\psi_i\rangle.
$$

For batch evaluation, $s^{(0)}$ is the true noiseless loss gradient. Main B
construction uses this gradient in the L-BFGS update from section 1.2 instead
of the fixed-step batch rule; its online phase and all noisy updates are unchanged.

For noisy fine-tuning, let $|\chi_{r,i}\rangle$ be the **unscaled** target
backpropagated through history $r$ to the same pre-noise point. Use the
implemented cross-trajectory signal

$$
s_i^{(\times)}=-2\operatorname{Re}
\sum_{r,s=1}^{K}w_rw_s
\langle\chi_{r,i}|A_i|\psi_{s,i}\rangle
\langle\psi_{s,i}|\chi_{r,i}\rangle.
$$

Both trajectory indices carry weights. This signal is not assumed to equal
$\nabla(1-\widehat F)$. Fine-tune for **100 online + 100 batch** updates at
$\eta=0.2$, with no decay; noisy fine-tuning retains the final iterate.
Noiseless construction instead follows the checkpoint rule in section 1.1.

## 5. Optional: fine-tune every depth candidate

**Use this extension when the extra fidelity is worth the additional computation.**
Across nine six-qubit targets and two optimizer seeds, the average improvements
over the corresponding archived single-circuit B/D pipeline were modest relative to the CPU cost:

| Pruning | IBM fidelity gain       | IBM CPU ratio | Google fidelity gain    | Google CPU ratio |
| ------- | ----------------------- | ------------- | ----------------------- | ---------------- |
| B       | +3.56 percentage points | 6.26x         | +2.40 percentage points | 5.99x            |
| D       | +2.87 percentage points | 7.11x         | +3.03 percentage points | 6.19x            |

CPU includes all visited construction, pruning/repair, fine-tuning and validation
work, including discarded candidates. Gains are averages, not guarantees for
every target. The archived single-circuit pipeline is the cheaper reference.
These numbers are from the archived comparison before best-iterate retention;
they are not a new comparison of the two pipelines with section 1.1 enabled.

When enabled, remove the 90% cutoff and perform all **100 online + 100 batch**
noiseless updates at each depth, retaining the **best iterate using section 1.1**.
Choose the noiseless batch optimizer using section 1.2; the online phase is unchanged.
Save every depth's restored checkpoint; continue growth from this **unpruned
noiseless parent**. Apply sections 2–4 to a separate copy at every visited depth,
using the same **100 weighted training histories per candidate**. The validation
rule below stops the main B search for the chosen noise profile. Historical
comparisons applied it separately to every pruning/noise-profile branch.

For each fine-tuned depth $d$, independently draw $M_v=32768$ validation histories
from the same law, using a separate RNG stream. Compress no-jump duplicates with
their empirical weights. With trajectory fidelities $f_r$ and weights $v_r$,

$$
\widehat F_d=\sum_r v_r f_r,\qquad
\mathrm{SE}_d^2=\frac{\sum_r v_r(f_r-\widehat F_d)^2}{M_v-1}.
$$

The first valid candidate establishes a material record $(F_*,\mathrm{SE}_*)$.
Replace that record and reset patience only when

$$
\widehat F_d>F_*+0.001+2\sqrt{\mathrm{SE}_d^2+\mathrm{SE}_*^2}.
$$

Otherwise increment patience, including coverage failures. Stop after three
unsuccessful extensions or at depth 16. This uncertainty allowance is a heuristic,
not a formal sequential test. Select $d^*=\arg\max_d\widehat F_d$ over all valid
visited candidates; ties prefer earlier depth. The winner may differ from the
last material record. Freeze this choice before exact evaluation.

The reference D construction can still lose fidelity during fixed-rate batch updates;
restoring the best checkpoint prevents discarding a better circuit. B's Armijo
condition protects its noiseless batch steps. Neither condition guarantees
monotone online updates or an improvement in final noisy fidelity.

## 6. Independent final evaluation (both modes)

For the fixed final parameters, evaluate the exact channel average
(outcome $a$ includes the no-jump identity):

$$
\rho_0=|0^n\rangle\langle0^n|,\qquad
\rho_i=\sum_a p_{i,a}N_{i,a}G_i\rho_{i-1}G_i^\dagger N_{i,a}^\dagger,
\qquad F_{\mathrm{test}}=\langle t|\rho_L|t\rangle.
$$

This integrates all noise outcomes independently of training and any optional
validation samples. Retrospective exact scores never change the selected circuit.
At $n=6$, training uses 64-amplitude vectors and evaluation uses $64\times64$
density matrices. Report fidelity, optimization CPU time, logical one-/two-qubit
counts, and two-qubit count/depth under identical reference compilation.

[Main construction optimizer](topdown_batch_refinement_policies.py) ·
[Completed batch-comparison driver](topdown_batch_refinement.py) ·
[Batch-comparison findings](topdown_batch_refinement_findings.md) ·
[Archived depth-search implementation](topdown_depth_search.py) ·
[Archived search protocol](topdown_depth_search_protocol.md) ·
[Depth-search findings](topdown_depth_search_findings.md) ·
[Best-retention implementation](topdown_optimizer_policies.py) ·
[Stability comparison](topdown_optimizer_stability_findings.md) ·
[Pruning implementation](legacy_pruning_ablation.py) ·
[Earlier 90% comparison](legacy_pruning_ablation_findings.md)
