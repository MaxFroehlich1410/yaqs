# Draft record

## Scientific story

1. Approximate preparation must balance circuit expressivity, gate noise, and classical construction cost.
2. Circuit pruning and noisy parameter optimization are established approaches with several existing variants.
3. Their usefulness depends on the target, noise placement, and complete search budget.
4. This paper specifies and evaluates noiseless online/L-BFGS construction, exact single-gate deletion scoring, simultaneous pruning B, and frozen-trajectory refinement.
5. Independent validation fixes the reported circuits before exact density-matrix assessment.
6. The tested workflow gives useful logical fidelity–CPU trade-offs, especially on the six real targets, while Haar and native results are less favorable.
7. One seed, development-used targets, unequal budgets, and six-qubit dense simulation limit generalization.

## Structural and scientific decisions

- Used the BUG manuscript's progression from setting to method, numerical evidence, and Discussion, with the Krotov manuscript's local adjoint exposition.
- Kept all six approved main sections and placed the two main tables in the protocol and native-results sections.
- Added mathematical appendices deriving shared deletion scoring and exact empirical multiplicity compression directly from the supplied method specification. These are explanations of the executed operations, not new convergence claims.
- Kept the noisy cross update explicit and separate from the true noiseless gradient. No proposed new counterexample or unreviewed theorem was promoted to a manuscript result.
- Included recent top-down and compilation-aware pruning literature to keep the contribution boundary accurate.
- Reported the native comparison in the main text and separated its same-schedule diagnostic from comparisons with older compiled deployment circuits.
- Retained archived depth selections, exact scores, full historical costs, and Fontana's paired six-target coverage.
- Did not reuse scientific claims or figures from the older repository manuscript, whose experiment description differs from the supplied handoff.

## Validation record

The 27 September discussion addition separates preliminary twelve-qubit IQSP
resource diagnostics from the ranked six-qubit comparison. The default depth-four
gradient completed on the MacBook; its cost is not a failed training run or a
measurement of comparative scaling. Joint Pauli noise requires additional
purification operations in the local implementation, but no controlled experiment
has isolated its contribution to runtime. The pure-state trajectory representation
suggests an advantage for parameter-independent Pauli noise, not arbitrary noise
models. IQSP's framework is not restricted to its H2 demonstration. Numerical
provenance is retained in `data/q12_iqsp_resource_diagnostic.json`; these values do
not enter any endpoint table, aggregate, or Pareto ranking.

Compilation, figure/table regression checks, bibliography checks, PDF visual review, archive-integrity checks, and the required repository-wide lint attempt are recorded in the delivered validation report. No optimization experiment was rerun for manuscript preparation.
