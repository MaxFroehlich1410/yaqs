# Matched logical-gate noise pilot

This is a separate comparison on the six-qubit **Gaussian, TFIM critical and
Haar 1** fixtures, with the historical synthetic **IBM** profile and optimizer
seed **300**. It aligns the placement of training and evaluation noise on each
method's original elementary gates. The earlier
[native-deployment protocol](README.md),
[noisy-method implementations](NOISY_METHODS.md) and their completed results
remain historical records; their scores must not be mixed with this pilot.

The external competitors are **NACL, RobustState and Fontana**. Fontana is
included only for Gaussian and TFIM critical because Haar 1 is complex.
**Fixed VQSP and unpruned layerwise are internal controls**, alongside our
archived B + online Krotov + L-BFGS construction and weighted noisy refinement.
AQCE and BAA-LRSP are excluded because their synthesis is noiseless. IQSP is
excluded while its initialization problem remains unresolved. No additional
pruning or optimization policy is added to an external method.

## Common channel and elementary-gate boundary

For each original elementary gate \(U_j\), apply

\[
\rho*{j+1}=\mathcal E*{a*j}(U_j\rho_j U_j^\dagger),\qquad
\mathcal E_a(\rho)=e^{-\Gamma_a}\rho+
(1-e^{-\Gamma_a})\sum*\mu\frac{\gamma*\mu}{\Gamma_a}
P*\mu\rho P*\mu,\quad \Gamma_a=\sum*\mu\gamma\_\mu.
\]

This is the existing at-most-one-Pauli-jump law per unit gate window, with
\(\gamma_X=\gamma_Y=\gamma_Z=10^{-4}\) on each acted-on qubit. For a
two-qubit gate, include those six local terms and correlated XX and ZZ terms
of strength \(1.5\times10^{-3}\) each. Thus \(\Gamma_1=0.0003\) and
\(\Gamma_2=0.0036\). Reuse `evaluation.pauli_distribution`; these coefficients
are historical simulation settings, not measured hardware gate infidelities.
There is no idle, measurement or readout noise. Final fidelity is
\(F=\langle t|\rho|t\rangle\).

The elementary-gate boundary is fixed before noisy optimization:

| Method                                   | Original elementary gates receiving individual noise windows                                                                            |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Our incumbent and both internal controls | Each scalar Pauli rotation in the stored circuit: RY/RZ and RXX/RYY/RZZ for these fixtures; RX is supported by the same representation. |
| NACL                                     | Each original RZ, SX and CX in the searched sequence. Its original and native schedules already coincide.                               |
| RobustState                              | Each CX, RZ, RX and RZ separately in its CNOT-first block; local rotations on both endpoints retain separate windows.                   |
| Fontana                                  | Each RY and each CX separately in the original ansatz.                                                                                  |

Do not replace an entire layer, a RobustState block, or an arbitrary multi-gate
unitary with one noise window. Do not compile, merge or cancel a trained
schedule during logical evaluation, including zero-angle rotations. NACL's
own published structural removal and simplification remain part of its search;
our pruning B remains part of our incumbent.

**Matching the channel by arity does not make the gate alphabets identical.**
For example, one RXX is one logical two-qubit window here, even if its native
decomposition contains multiple CX gates. This pilot assesses performance
under the declared original-gate model. It does not establish an advantage on
one common physical gate set. Report logical one- and two-qubit counts and
two-qubit depth; any separately compiled counts are deployment diagnostics and
do not define the fidelity used for selection or ranking in this pilot.

## Budgets and reusable training

Budgets, initializations and optimizer settings remain those of the completed
three-target pilot:

| Method             | Candidates and training                                                                                                                                                                                                                           | Reuse                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Our incumbent      | Previously selected pruning-B branch; noiseless construction uses 100 online Krotov plus 100 L-BFGS updates per visited depth; noisy refinement uses the unchanged 100 online plus 100 fixed-step batch updates with M=100 weighted trajectories. | Retain the authenticated archived circuit, parameters and original logical-noise depth selection. No retraining or depth reselection. |
| Fixed VQSP         | Depths 2 and 4; independent initialization per depth; 200 noiseless and 200 noisy L-BFGS-B iterations, M=100 weighted trajectories.                                                                                                               | Reuse both logical training endpoints; select afresh with logical validation.                                                         |
| Unpruned layerwise | Depths 1–4; 200 noiseless and 200 noisy updates per depth, with 100 online updates in each phase, M=100 weighted trajectories.                                                                                                                    | Reuse the entire trained grid; select afresh with logical validation.                                                                 |
| NACL               | Structural checkpoints 0, 10 and 50; one restart; 100 objective evaluations per inner OrthoMADS solve; 300-gate cap.                                                                                                                              | Reuse the original elementary-gate search unchanged; select afresh with logical validation.                                           |
| RobustState        | Depths 2 and 4; 500 noiseless plus 50 noisy Adam updates; learning rate 0.005; exact-density feedback and its original noiseless-Jacobian surrogate.                                                                                              | Fresh optimization with `DenseObjective(..., compile_native=False)` on the original template.                                         |
| Fontana            | Depths 2 and 4; 200 noiseless plus 200 noisy BFGS iterations.                                                                                                                                                                                     | Fresh optimization with `DenseObjective(..., compile_native=False)` on the original RY/CX template.                                   |

The existing training code preserves each method's endpoint policy; this
comparison does not add best-iterate recovery, pruning or another optimizer to
a competitor. Exact-density feedback for RobustState remains a simulator
adaptation, documented in [its provenance](ROBUSTSTATE_PROVENANCE.md).

Reusable full-budget construction bundles are located relative to the
repository root as follows. In the table, `method` is `fixed_vqsp`, `layerwise`
or `nacl`; each method root contains
`construction/target_id/method/300/IBM/{record.json,circuits.qpy,seal.json}`
and `histories.npz` when parameter histories exist.

| Target ID                 | Method root                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `gaussian_mu0p5_sigma0p1` | `output/gaussian-all-methods-pilot-20260920/method`              |
| `tfim_critical`           | `output/tfim-haar-noisy-ibm-pilot-20260920/tfim_critical/method` |
| `haar_random_1`           | `output/tfim-haar-noisy-ibm-pilot-20260920/haar_random_1/method` |

Our incumbent comes from
`output/topdown-batch-refinement-q6-20260919/pipeline/lbfgs/target_id__seed300`.
Use `reference.frozen_reference` to authenticate the archived B/IBM selection,
checkpoints, fixture and training-source hashes. Its selected depths are 1, 2
and 4 for Gaussian, TFIM critical and Haar 1, respectively.

Authenticate reused bundles and retain their source-manifest hashes. The
training dependencies for controls and NACL match their completed manifests;
changes to unrelated method files, documentation or runner defaults do not
justify pretending that an old whole-campaign manifest matches the new one.
Record the checked training dependency set and any excluded changed files in
the new reuse receipt. New optimization belongs to a new output campaign.

## Independent assessment and accounting

For every eligible candidate, sample **8,192 fresh independent logical-noise
validation trajectories**, separate from training and previous validation
banks. Collapse identical no-jump histories with their empirical weights and
check jump coverage for every nonempty circuit. Do not resample until jumps
appear: a missing-coverage case is explicit and ineligible. An empty circuit
has no noise windows and is the valid exception.

Use the declared logical validation rule to select within the reused control
and NACL grids and within the fresh RobustState/Fontana grids. The incumbent
retains its previously locked logical selection; its new validation is an
independent diagnostic. Write a new selection lock before evaluating exact
fidelity. The independent Qiskit density-matrix evaluator integrates the same
logical channel over all modeled outcomes for every candidate. These final
scores have no trajectory sampling error; their use must not change the
locked winner. Report validation standard errors and the distinction between
locked selections and any post hoc candidate tradeoffs.

The pilot has **17 supported method-target groups and 40 candidates**: each
target has one incumbent, two fixed VQSP, four layerwise, three NACL and two
RobustState candidates; Gaussian and TFIM additionally have two Fontana
candidates each. Preserve the Fontana/Haar exclusion explicitly.

Charge original construction and search CPU for every reused grid, including
discarded candidates, while reporting newly spent load, training, validation
and exact-evaluation time separately. Incumbent archive costs include its
original depth-validation work; exclude that component when comparing
construction-only costs. Do not sum NACL cumulative-prefix costs, count cached
training as newly executed, or report loading time as algorithm runtime. The
budgets are method-specific, so this is not an equal-compute comparison.
