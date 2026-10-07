# Twelve-qubit random-MPS initialization and longer depth refinement

This is a separate noiseless diagnostic on the hardcoded `random_mps_bond3`
target (12 qubits, target seed 5003). It investigates whether a better starting
overlap and more optimization at each depth can reach **99% ideal fidelity**.
The previous exact-statevector depth-ten control reached only
0.00003066282754%; its poor optimization persisted after the MPS SVD failure
was isolated. This experiment uses exact statevectors throughout.

The completed Haar refinement showed that an existing depth-87 circuit could
reach 99% with more L-BFGS updates, whereas the earlier short per-depth schedule
first recorded 99% at depth 107. Accordingly, this new random-MPS study allocates
**100 online Krotov sweeps followed by up to 2,000 safeguarded L-BFGS updates at
every depth**, rather than 100 + 100. Initialization and the batch budget are
the explicit experimental changes; the circuit family, online updates, initial
rate 0.7, line search, best-iterate retention and extension law are reused.

## Matched nine-run matrix

| Arm              | Initial U3 product-state layer                            | Remaining depth-one parameters                      | Seeds         |
| ---------------- | --------------------------------------------------------- | --------------------------------------------------- | ------------- |
| `near_zero`      | Historical Gaussian angles, standard deviation 0.05       | Historical Gaussian angles, standard deviation 0.05 | 300, 400, 500 |
| `random_product` | Independent Haar-distributed single-qubit pure states     | Identical to paired `near_zero` draw                | 300, 400, 500 |
| `largest_basis`  | Computational basis state with largest target probability | Identical to paired `near_zero` draw                | 300, 400, 500 |

All arms have the same trainable initial U3 layer and architecture. They change
only its starting parameters; all parameters remain trainable. The latter two
arms retain the original draw outside the first 36 angles. No extra gates or
fixed input-state oracle are introduced. Largest-basis selection scans the
available target vector, with ties going to the lowest amplitude index; this
target-informed setup cost is measured and counted.

For `random_product`, draw independent $u_i\sim U[0,1)$ and
$\phi_i\sim U[0,2\pi)$ from `SeedSequence([seed, 2381])`, and set the U3 angles
to $(\arccos(1-2u_i),\phi_i,0)$. The exact product state is

$$
|p\rangle=\bigotimes_i\left(\sqrt{1-u_i}|0\rangle
+e^{i\phi_i}\sqrt{u_i}|1\rangle\right).
$$

For `largest_basis`, choose $b=\arg\max_k|t_k|^2$ and set each polar angle to
$\pi$ times bit $i$ of $b$, with both phase angles zero. Vectors use little-endian
basis order. The random entangling layer is subsequently applied, so the actual
initial circuit fidelity can differ from the pure product-state overlap.
Both scores are saved explicitly.

## Growth, optimization and stopping

Grow from depth 1 through at most depth 16. New depths carry the previous
depth's best unpruned parameters and append the historical
`RandomState(seed + 1000 + depth)` perturbations: scale 0.05 at depth 1 and
0.001 thereafter. Different arms share the appended random draws within each
seed, although their trained parent parameters generally differ.

Each depth makes one continuous hybrid optimizer call. After at most 100 online
sweeps, L-BFGS retains ten curvature pairs throughout up to 2,000 batch updates.
Curvature resets once at each new depth. Its unchanged safeguards include
Armijo coefficient 1e-4, at most 12 backtracking reductions, invalid-curvature
skips and descent-direction resets. Failed line searches count against the
update budget. Restore the best actual parameters, including the starting point.

Stop at the first accepted iterate with $F_0\geq0.99$; if the starting circuit
already satisfies the threshold, perform no updates. Otherwise, finish the
depth and grow again. There is **no depth-patience stop in this bounded capacity
audit**, so tiny initial overlaps cannot prematurely terminate the study.
The depth cap and a six-hour CPU budget per arm are explicit limits. CPU is
checked after complete depths and can exceed the limit by one depth's work.
It includes setup, optimizer/progress work and endpoint replay; final checkpoint
seal serialization and standalone checkpoint authentication replays are excluded
from the stored CPU sum. Each final record declares this accounting scope. Neither limit
implies convergence or proves the target is unreachable.

Deployment should use one numerical thread per independent job, an eight-hour
wall guard and a 16-GiB memory guard through the existing `bounded_probe`
supervisor. A guard interruption is incomplete, and successful worker exit plus
sealed final records are required to report completion. Independent arms can
run in parallel; updates and depths within an arm remain sequential.

## Script and execution

The default action **only prepares** nine explicit commands, budgets, source and
target hashes, numerical thread settings and deployment guards:

```sh
python -m experiments.topdown_random_mps_initialization \
  --output output/q12-random-mps-initialization-20261005
```

Run one arm explicitly with a separate output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m experiments.topdown_random_mps_initialization \
  --action run \
  --output output/q12-random-mps-initialization-20261005/jobs/random_product__seed300 \
  --arm random_product --seed 300
```

All budget fields have explicit CLI options. For example, `--target-fidelity 0.999`
defines a different, authenticated 99.9% study. Changed settings require new
output directories. Commands use the interpreter and output root where the plan
was prepared; prepare a new plan in the staged remote workspace before server
dispatch. The matrix is a longer-budget initialization comparison;
it does not independently estimate the benefit of 100 versus 2,000 batch steps
for every initialization. Preserve the earlier short-budget results as historical
references rather than treating them as matched nine-run controls.

## Checkpoints, recovery and reporting

Each arm stores `manifest.json`, a sealed `initialization/` checkpoint,
`depth_0dd/construction/{record.json,arrays.npz,complete.json}`, and a sealed
`final/` summary. Each depth's parameters and fidelity history appear once in
its compact binary archive; the complete update/line-search diagnostics are
compressed UTF-8 JSON in `optimizer_records_utf8`. Selection and summary records
reference these archives. No training histories are duplicated in final JSON.

Each depth reports fresh statevector endpoint fidelity and gradient norm,
exact 99%/99.9% crossing costs, curvature skips, rejected trials, and optimizer
evaluation counts. Root `progress.json` identifies the active depth and links
its live progress file. Live parameter snapshots are observational and lack
L-BFGS curvature; they cannot resume an interrupted continuous call. A started
but unsealed depth causes an explicit failure on resume. Restart from the same
deterministic inputs in a new output directory, or reuse only sealed completed
depths. Original experiments and frozen sources remain separate.

Compare success counts across the three seeds, first successful depth and
within-depth update, achieved fidelity, total CPU including initialization,
and logical two-qubit rotations. Near-zero versus other arms isolates starting
conditions under the same enlarged optimizer budget. Report non-attainment,
continuing improvement and numerical failures separately. The depth index counts
entangling blocks per bond; it is not compiled two-qubit depth. These are noiseless
results and do not establish a noisy-fidelity benefit or globally minimal depth.

References: [Haar refinement findings](../output/q12-haar-refinement-20261004/REPORT.md) ·
[Original random-MPS failure diagnosis](../output/q12-noiseless-continuation-20260928/diagnosis/REPORT.md) ·
[Current method](topdown_state_preparation_method.md)
