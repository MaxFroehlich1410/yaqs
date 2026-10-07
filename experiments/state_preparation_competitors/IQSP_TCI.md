# TCI targets for the Ballarin benchmark

The compressed single-tooth IQSP pipeline now constructs each intermediate
target with the published two-site tensor cross interpolation algorithm in
TensorCrossInterpolation.jl 0.9.19. The existing benchmark targets and logical
noise profiles remain fixed. Ideal SU(4) training, warm starts, Adam updates,
the final noisy phase and the paper's small-angle postprocessing are unchanged.

This fills the target-construction gap in the previous dense-SVD adapter.
It does not establish which unpublished numerical settings Ballarin used.
TCI has several published variants, so an unspecified choice is not evidence
of a universal default. Our implementation and settings are recorded explicitly.

## Algorithm and scope

We reuse `crossinterpolate2`, rather than implement another interpolation
algorithm. It uses two-site sweeps and partially rank-revealing LU pivot
selection. The returned tensor train is exported directly, with no dense SVD
or subsequent target truncation. Normalization contracts its tensor-network
norm and rescales its first core.

The current benchmark circuit uses `CombLayout(n, 1)`, a single-tooth comb,
which is a tensor train. This adapter does not implement branching tree TCI.
The Gaussian is sampled directly as

\[
f*\lambda(b)=\exp[-\lambda(x(b)-0.5)^2/0.04],\qquad
x(b)=\sum*{j=0}^{n-1}b_j2^{-j-1}.
\]

Here `b_j` is Qiskit qubit `j`; its integer-vector index is
`sum(b_j * 2**j)`. This intentionally preserves the benchmark's bit ordering
and Gaussian width. No dense Gaussian amplitudes are sent to Julia.

For existing arbitrary complex fixtures, the adapter supplies an index oracle
over the stored vector. This supports those benchmark inputs but does not make
their loading scalable. Their existing geodesic interpolation remains an
explicit extension beyond Ballarin's demonstrated function paths. Merely using
TCI does not turn these into published-method reproduction cases.

## Frozen choices

| Setting                                   | Value                                                         | Basis                                                     |
| ----------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| Library                                   | TensorCrossInterpolation.jl 0.9.19                            | Published implementation; pinned environment              |
| Method                                    | `crossinterpolate2`, full pivot search, back-and-forth sweeps | Library algorithm and defaults                            |
| Maximum target bond dimension             | 16                                                            | Largest rank in Ballarin's target-accuracy study          |
| Relative interpolation tolerance          | `1e-12`                                                       | Explicit benchmark choice, not an author-reported setting |
| Maximum iterations                        | 20                                                            | Pinned library implementation default                     |
| Random seed                               | 0 plus target-bank position                                   | Explicit reproducibility choice                           |
| Gaussian initial pivot                    | All local indices equal to 1                                  | Library default; all bits zero                            |
| Generic initial pivot                     | Largest-magnitude fixture entry                               | Explicit nonzero initialization for sparse inputs         |
| Other pivot/convergence options           | Pinned library defaults                                       | No custom optimizer or rank-growth policy                 |
| Independent normalized-vector error limit | `1e-10`                                                       | Our small-system numerical validation criterion           |

At every interpolation coordinate, all normalized target amplitudes are checked
against the exact fixture up to a global phase. The audit is limited to two to
twelve qubits and is separately costed. Failure raises an error before circuit
optimization; there is no automatic increase of rank, tolerance adjustment,
or fallback to dense SVD. Generic high-entanglement targets may need a larger
declared cap to represent the input, rather than being admitted with poor
target fidelity.

Target rank 16 is distinct from circuit-state bond 64 and local purification
environment rank 16. The latter caps and tolerances `1e-9`/`1e-12` remain the
public Ballarin simulator defaults. They are not independently verified as the
exact settings of every published experiment. Tighter simulator runs remain
numerical audits, not a new competitor optimizer.

## Installation and use

Julia 1.12 is used in an isolated project. The lock file pins transitive package
versions. Installation is explicit; training never installs dependencies.
The Python bridge's child-process CPU accounting currently requires Unix.

```sh
julia --startup-file=no --project=experiments/state_preparation_competitors/tci_julia \
  -e 'using Pkg; Pkg.instantiate()'
```

```python
from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions
from experiments.state_preparation_competitors.iqsp_tci import TCIOptions
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_candidates

candidates = iqsp_tn_candidates(
    target_vector,
    target_id=target_id,
    depths=(1, 2, 3, 4),
    comb_options=CombOptions(checkpoint_interval=8),
    target_construction="tci",
    tci_options=TCIOptions(),
)
```

TCI is the default when `comb_options` is supplied. An explicit
`target_construction="dense_svd"` retains the historical diagnostic path.
`CombOptions.target_tolerance` must stay zero with TCI; it only controls the old
SVD converter. Omitting `comb_options` preserves the archived exact backend.
Use a new output directory, never relabel old dense-SVD results as TCI.

Each depth constructs its target bank once, before training; the final target
is reused for noisy optimization and postprocessing. Receipts include options,
Julia/library versions, source/lock hashes, seeds, pivot and rank histories,
function-call counts, tensor hashes, exact target errors, and setup timings.
Candidate CPU totals include the Julia child process, and wall totals include
startup, compilation, target validation and training. Setup cost is not hidden
or duplicated across the final training phases. Cores can be regenerated using
the pinned environment and recorded seeds.

For the startup diagnostic, pass `tci_options=TCIOptions()` to
`q12_readiness.audit_gaussian_initialization`. Its exact gradients remain audit
references only; no reference gradient replaces the author's training update.
TCI does not fix or conceal the known singular-SVD derivative at identity.

## Sources

- [Ballarin et al., Fig. 1 and Supplement I](https://arxiv.org/html/2511.15674v1):
  TCI target at each interpolation step and the target-rank study.
- [Núñez Fernández et al., two-site TCI algorithms and libraries](https://arxiv.org/abs/2407.02454):
  the algorithm reused here.
- [TensorCrossInterpolation.jl source](https://github.com/tensor4all/TensorCrossInterpolation.jl):
  upstream code installed through the Julia registry. The exact package tree
  is recorded in `tci_julia/Manifest.toml`.

These checks validate an implementation; they are not new competitive fidelity
results. No full comparison campaign is launched by this change.
