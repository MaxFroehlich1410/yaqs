# AQCE implementation and upstream provenance

The active implementation follows Algorithms 1–5 of the published AQCE paper.
Local `aqce_schedule.py` supplies their initialization, growth, and alternating
update schedule. It reuses the MNISQ project's verified, unchanged complex
`evaluate_F` and `evaluate_Rho` numerical routines and Qulacs state/gate
primitives. The original MNISQ `AQCE_python` driver is retained for provenance
and historical comparisons but is not called by the active adapter.

- Repository: <https://github.com/FujiiLabCollaboration/MNISQ-quantum-circuit-dataset>
- Pinned commit: `c0b6d860477c4dcfb2737e6e6c996a4854858d5d` (24 May 2024).
- Original source: `mnisq/internal/generator/aqce/AQCE_from_python.py`.
- Local resource: `vendor/mnisq_aqce/AQCE_from_python.py.txt`.
- Source SHA-256: `2ca009f6001ead4dd50ff5e875704329ce76384547ba1929cf538f457895a57b`.
- Original `LICENSE` is retained verbatim as `vendor/mnisq_aqce/LICENSE.txt`.
- License SHA-256: `2003829cf643d61b00a76088dd31bb11b8029546f24c56b39fd492adb9281483`.

The repository's root license is **Creative Commons Attribution–ShareAlike 4.0
International**, including its warranty disclaimer. Its packaging metadata says
MIT, inconsistently. This copy preserves and identifies the root license; it does
not assume the packaging metadata grants an MIT license. The original source and
license bytes are unchanged. A `.py.txt` resource keeps automated project Python
formatters from modifying upstream code. The adapter verifies both hashes before
loading that local resource; nothing is downloaded when running it. Reuse does
not imply endorsement by MNISQ or the AQCE paper's authors.

## Algorithm and published provenance

AQCE is described by Shirakawa, Ueda and Yunoki in
[Physical Review Research 6, 043008 (3 October 2024)](https://doi.org/10.1103/PhysRevResearch.6.043008),
with the [original preprint](https://arxiv.org/abs/2112.14524) submitted in 2021.
The source reused here belongs to the published follow-on
[MNISQ dataset project](https://www.nature.com/articles/s41597-026-07493-9),
not an implementation verified as supplied by the original AQCE authors.
The published paper, its linked resources, GitHub repository searches, and the
available MNISQ Python/C++ routines were checked; no reusable implementation
verified to implement all of the original paper's scheduling steps was found.
The local schedule corrects those differences without replacing the reused
tensor routines or adding a different optimizer.

For a gate between prefix state `phi` and suffix-adjusted target `psi`, AQCE
forms the reduced transition matrix `F = Tr_rest(|psi><phi|)`. For every allowed
pair it computes an SVD `F = X D Y†`; it chooses the pair maximizing `sum(D)` and
sets its unitary to `X Y†`. Pair placement therefore changes during optimization.
The active schedule implements the following published steps:

1. **Initialization (Algorithm 4):** build `M0` gates once by selecting the
   largest reduced-density eigenvalue over all pairs, placing the corresponding
   ordered eigenvectors in the new unitary, and updating the residual target.
2. **Sweeps (Algorithms 1–3):** perform `N` complete forward/backward sweeps.
   Each gate update uses the already updated environment on one side and the
   remaining old circuit on the other. Environment states are propagated between
   updates, as in the paper, rather than rebuilding the circuit for every gate.
3. **Growth (Algorithm 5):** insert `delta M` identity gates next to the input
   state. Apply a backward update only to this new block, using `C†|target>` as
   its target. Then perform `N` complete sweeps over the enlarged circuit.
   Repeat until the requested gate budget is reached.

The earlier local adapter called MNISQ's forward-only driver, which instead used
density-based insertion at every growth step. Results and timing estimates from
that earlier implementation describe a different schedule and must not be
reported as results of this corrected implementation.

## Exact scope of this adapter

- All inputs remain complex. The image-oriented upstream C++ implementation
  contains real-matrix restrictions and is deliberately not used.
- Each positive budget runs the published schedule with defaults `M0=1`,
  `delta M=1`, `Mmax=budget`, and `N=100`. The API exposes `initial_gates`,
  `growth_increment`, and `sweeps`; the corresponding CLI controls are
  `--aqce-initial-gates`, `--aqce-growth-increment`, and `--aqce-sweeps`.
  **A sweep means a forward pass followed by a backward pass.**
- The paper treats these as configurable algorithm controls. These benchmark
  defaults do not reproduce its particular many-body or image experiment
  settings. In particular, a small `M0` need not provide the nonzero initial
  overlap intended by the paper for an arbitrary sparse target. No special
  target-dependent initialization, perturbation, or restart remedy is added;
  any changed controls must be declared in the run manifest.
- Initialization, growth, and updates consider every two-qubit pair. Exact ties
  use the first pair in lexicographic order. The adapter accepts only all-to-all
  connectivity; common compilation accounts for required hardware routing.
- Every requested budget remains an independent restart. Metadata charges every
  restart; cumulative CPU includes source loading once, circuit conversion, and
  ideal-fidelity calculation. No warm-start reuse between budget calls is
  claimed. For budgets `0, 5, 10, 20, 40, 80` at the defaults, there are
  **868,000 sweep gate updates plus 150 growth gate updates**, or **868,150 SVD
  gate updates per target**, plus five density-initialization gate choices.
  Each choice scans all 15 pairs at six qubits. Budget 80 alone performs 648,079
  SVD gate updates plus one density-initialization choice. These are operation
  counts, not wall-time predictions.
- The implementation is deterministic: the common `seed` argument is recorded but unused.
  Repeated seeds must not be counted as independent optimizer replicas.
- Budget zero is a separate empty-circuit control. Positive budgets require at
  least two qubits, must be at least `M0`, and must be reachable exactly by
  adding integer multiples of `delta M`. There is no invented one-qubit AQCE
  fallback or silently shortened final growth block.
- Qulacs matrices and their ordered target lists are passed directly to Qiskit
  unitary instructions. Native gate decomposition and its gate/depth/noise costs
  belong to common compilation, not this adapter.
- The reported fidelity is `|<target|circuit|0>|²`, evaluated by Qiskit. Upstream's
  alternative `get_cost` helper is not used. No noisy optimization, pruning,
  selection by test data, or stopping at a chosen fidelity is added.

The adapter is tested with Qulacs 0.6.14. NumPy's SVD/eigensolver can choose
different bases in degenerate subspaces across library versions; exact bitwise
replay claims require a pinned numerical environment.

Use a fresh output directory for the corrected schedule. Source hashes and
explicit controls are part of the run manifest, so earlier completed results
cannot be resumed as if they had been generated by this implementation. Preserve
historical outputs and obtain new timings before estimating the corrected run.
