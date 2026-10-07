# Paper-drafting handoff: six-qubit noisy state preparation

This index identifies the completed experiments and source material for a first
paper draft. The writing guide is supplied material; adding it did not initiate
a manuscript rewrite or change the experiments.

Repository: `/Users/maximilianfrohlich/Desktop/GitHub/yaqs`.

## Read first

| Purpose                                                     | File                                                                                                                                 |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Supplied writing guide, copied unchanged                    | [PAPER_WRITING.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/PAPER_WRITING.md)                                                   |
| Mathematical specification of the current method            | [topdown_state_preparation_method.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/topdown_state_preparation_method.md) |
| Exact protocol for the latest completed comparison          | [LOGICAL_SUITE.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/LOGICAL_SUITE.md)         |
| Latest results, interpretation, runtime and resource tables | [REPORT.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/logical-noise-all-q6-ibm-20260920/REPORT.md)                        |
| Earlier matched native-noise results and limitations        | [native REPORT.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/native-topdown-ibm-pilot-20260920/REPORT.md)                 |

The latest study covers all nine fixed six-qubit targets, one optimizer seed
(300), and the synthetic IBM logical-gate noise profile. It contains 51 supported
method–target comparisons and 120 candidates. Seventeen completed comparisons
and six additional trained baseline endpoints were reused; only 28 missing
competitor/control groups were trained. The methods and budgets were unchanged.

## Latest result files

All files below are in
`/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/logical-noise-all-q6-ibm-20260920/`.

| File                           | Contents                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `REPORT.md`                    | Human-readable results and interpretation; start here.                                                                       |
| `summary.json`                 | Complete candidate records, selected indices, final exact scores, costs, exclusions and references to reused source outputs. |
| `analysis.json`                | Selected/candidate rows, method aggregates, paired comparisons, full-grid cost accounting and Pareto sets.                   |
| `report_data.json`             | Independently checked data used for report tables and figures.                                                               |
| `manifest.json`                | Frozen configuration, seed, package versions, noise convention and source fingerprints.                                      |
| `report_seal.json`             | SHA-256 fingerprints of final artifacts and verification evidence.                                                           |
| `fidelity.pdf`, `fidelity.png` | All-target fidelity figure.                                                                                                  |
| `pareto.pdf`, `pareto.png`     | Per-target fidelity versus full construction/training CPU.                                                                   |
| `construction/`, `evaluation/` | New construction bundles, validation-selection locks and exact final assessments.                                            |

Verification logs and independent audits are in
`/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/logical-suite-validation-20260920/`.
Read `independent-analysis-audit.json`, `training-audit.json` and
`tests-final.log`. All 879 targeted tests passed. Repository-wide lint has
separately documented failures; do not describe it as passing.

## Scripts and implementation

| Purpose                                                      | File                                                                                                                                           |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Latest experiment entry point, including authenticated reuse | [run_logical_suite.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/run_logical_suite.py)           |
| Aggregate results and calculate Pareto comparisons           | [logical_suite_analysis.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/logical_suite_analysis.py) |
| Earlier three-target logical comparison                      | [run_logical.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/run_logical.py)                       |
| Construct the archived B/L-BFGS baseline                     | [topdown_batch_refinement.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/topdown_batch_refinement.py)                           |
| Noiseless optimizer policies                                 | [topdown_batch_refinement_policies.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/topdown_batch_refinement_policies.py)         |
| Archived-baseline loading and provenance                     | [reference.py](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/reference.py)                           |
| Benchmark definition                                         | [state_preparation_benchmarks.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation_benchmarks.md)                    |
| Fixed target amplitudes and metadata                         | [state_preparation_target_states.json](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/benchmarks/state_preparation_target_states.json)          |
| Competitor scope and adaptation descriptions                 | [NOISY_METHODS.md](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/state_preparation_competitors/NOISY_METHODS.md)                   |

The comparator directory also contains `nacl.py`, `robuststate.py`,
`fontana.py`, `controls.py`, `NACL_PROVENANCE.md`, `ROBUSTSTATE_PROVENANCE.md`
and `SUBSET_PROVENANCE.md`. Use those provenance documents to locate and verify
the underlying publications. The general competitor documentation includes
native-noise procedures; `LOGICAL_SUITE.md` and its manifest define the noise
placement actually used for the latest results.
The general benchmark definition also describes other system sizes, training
contracts and calibrated noise families; it is background, not a substitute for
the executed logical-suite protocol.

The completed experiment command, from the repository root, was:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-noisy-competitors/bin/python -m experiments.state_preparation_competitors.run_logical_suite \
  --output output/logical-noise-all-q6-ibm-20260920 --workers 3
```

The run is complete. A paper-drafting agent can read the artifacts without
rerunning training. The runner relies on source archives and environment pins;
the entry point alone is not a standalone reproduction package.

## Interpretation that the draft must preserve

- The evaluated baseline is pruning B, online Krotov followed by L-BFGS during
  **noiseless construction**, and weighted M=100 noisy fine-tuning. Noisy
  fine-tuning retains 100 online plus 100 fixed-step cross-trajectory updates
  at rate 0.2. It does not use L-BFGS. Best-iterate restoration applies to
  noiseless construction; the historical noisy phase retains its final iterate.
  The cross-trajectory update is not established as the gradient of the
  sample-mean fidelity; preserve this distinction in the mathematical account.
- The baseline archive used the optional search that fine-tunes every depth
  candidate. Its results must not be attributed to the cheaper first-90%
  stopping procedure. Pruning D and M=400 are not arms of this latest comparison.
- These are dense state-vector/matrix experiments. “Random MPS” identifies
  target generation, not an MPS simulation backend.
- Independent validation is separate from training. Archived baseline depth
  selections remain fixed. Final fidelity is calculated by exact density-matrix
  propagation of the same channel and integrates all noise outcomes. Exact
  scores do not replace the validation-selected circuit.
- The profile includes single-site X/Y/Z jumps and correlated XX/ZZ jumps after
  original elementary gates. It is a synthetic single-jump-per-window law,
  not a fresh hardware calibration or the full repeated-jump Lindblad process.
- Our method has the highest selected-endpoint fidelity on 5/9 targets and is
  Pareto-efficient in fidelity versus construction/training CPU on 7/9.
  Mean fidelity is 92.260%, versus 91.019% for RobustState, with 11.17% more
  aggregate CPU. RobustState dominates ours on Haar 1 and Haar 3.
- Fontana covers only six real targets. Its paired mean fidelity is 3.875
  percentage points below ours, with substantially lower CPU cost. Do not
  compare its six-target mean directly with other methods' nine-target means.
- Fixed VQSP and layerwise are internal controls. NACL, RobustState and Fontana
  are documented publication-based simulator adaptations with limited budgets.
  The NACL result is not evidence of a converged search. AQCE/BAA-LRSP and the
  unresolved IQSP implementation are excluded from this study.
- Runtime comparisons charge all visited candidates and historical training
  work, including reused work. The latest campaign's 374.21-second wall time
  is not any individual method's complete construction cost.
- One seed, a development-used target fixture, differing gate alphabets and
  finite budgets limit generalization. Native-noise findings remain relevant
  and less favorable. These results do not establish universal superiority,
  asymptotic scaling, or a hardware speedup.

## Preserve archives when transferring to another machine

The latest output references earlier bundles instead of duplicating histories.
Keep the local repository and its existing `output/` archives available to the
drafting agent. In particular, retain the latest suite, the three-target logical
pilot and `output/topdown-batch-refinement-q6-20260919`, together with the source
outputs referenced recursively by their manifests and summaries. Do not move
only the latest result directory and assume its parameter histories are
self-contained. Some stored references are absolute local paths.

These files have been added locally; this handoff does not imply a commit,
push, manuscript draft or publication.
