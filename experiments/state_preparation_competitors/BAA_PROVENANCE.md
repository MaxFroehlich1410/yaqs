# Official QCLib BAA-LRSP adapter

This adapter calls the original QCLib implementation of bounded approximate
amplitude preparation with low-rank state preparation (BAA-LRSP). It does not
reimplement the approximation search, Schmidt decomposition, rank selection,
isometry synthesis, or unitary synthesis.

## Source and paper

- Official repository: [qclib/qclib](https://github.com/qclib/qclib).
- Installed release: [qclib 0.1.15](https://pypi.org/project/qclib/0.1.15/).
- Audited official commit: [`b0bc339258532bd9aaecca3ddd2d710f0b92fa80`](https://github.com/qclib/qclib/tree/b0bc339258532bd9aaecca3ddd2d710f0b92fa80).
  This commit's `setup.py` identifies version 0.1.15. The three algorithm files
  below match the released wheel byte for byte.
- Algorithm entry point: [`BaaLowRankInitialize`](https://github.com/qclib/qclib/blob/b0bc339258532bd9aaecca3ddd2d710f0b92fa80/qclib/state_preparation/baa_lowrank.py).
- Paper: Israel F. Araujo, Carsten Blank, Ismael C. S. Araújo, and Adenilton J. da
  Silva, [_Low-rank quantum state preparation_](https://arxiv.org/abs/2111.03132),
  [DOI 10.1109/TCAD.2023.3297972](https://doi.org/10.1109/TCAD.2023.3297972).
- Upstream license: [Apache 2.0](https://github.com/qclib/qclib/blob/b0bc339258532bd9aaecca3ddd2d710f0b92fa80/LICENSE).
  The unchanged algorithm is imported from the installed package; no upstream
  algorithm source is copied into the adapter.

The wheel `qclib-0.1.15-py3-none-any.whl` has SHA-256
`773b41cefc7154d5466f55777c71ad5d3813e117f7f4637d957b982c48e4922a`.

At runtime the adapter verifies exact QCLib/Qiskit versions and the following
installed-source SHA-256 hashes. These authenticate the three audited core
algorithm files; they are not a claim that every transitive dependency is hashed.

| Official package file                    | SHA-256                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `qclib/state_preparation/baa_lowrank.py` | `87e44268d6278dd4531f8b2ed604dd8f7c62702477a221162ecb7dcb5bf60ab6` |
| `qclib/state_preparation/lowrank.py`     | `78ec56f3072685ad0d6b34026118074407e61766747ea4abab79a929c0d592d8` |
| `qclib/state_preparation/util/baa.py`    | `7aa209416caf4f4cef209ff76b56d13a37253045a1e9aa5c72f460ea758c1dbd` |

## Isolated installation and tests

Run from the repository root:

```sh
uv run -q --with-requirements experiments/state_preparation_competitors/requirements.txt \
  pytest -n 0 tests/optimization/test_state_preparation_competitors_baa.py
```

This leaves the project dependency declarations and lock unchanged. The optional
requirements pin QCLib 0.1.15, Qiskit 2.4.1, NumPy 2.4.6, SciPy 1.17.1, and the
direct supporting packages used by the competitor environment. QCLib imports
`qiskit_aer` through its utility module without declaring Aer in its package
requirements; the optional environment therefore explicitly includes
`qiskit-aer==0.17.2`. This resolves the import using the original package, with no
monkeypatch. Qulacs 0.6.14 is included for the separate AQCE adapter.

## Wrapper boundary and declared options

`baa_candidates(target, fidelity_losses=(...), strategy="brute_force",
use_low_rank=True)` validates a normalized finite complex q1–q6 state in Qiskit's
little-endian order. It preserves amplitudes and global phase without taking
magnitudes or renormalizing the input. It sorts/deduplicates the requested bounds
and always includes the exact zero-loss endpoint.

Each bound is passed directly as the official `max_fidelity_loss` option. The
adapter explicitly passes `strategy`, `use_low_rank`, `max_combination_size=0`
(all partition sizes), `iso_scheme="ccd"`, and `unitary_scheme="qsd"`. The adapter's
default `brute_force`/`True` choices differ from upstream's `greedy`/`False`
defaults; they select existing upstream features and do not alter the algorithm.
The alternative `greedy` strategy and `use_low_rank=False` are also supported.

The upstream gate's definition is expanded with Qiskit `decompose(reps=20)`.
This expands existing definitions without transpiler optimization, routing,
approximation, or qubit permutation. Any remaining gate wider than two qubits
causes an error. Ancillas, classical bits, unresolved parameters, nonunitary
operations, and incorrect output width also cause errors.

The independently simulated preparation is required to satisfy

\[
F=|\langle\psi\_{\rm target}|U|0\rangle|^2\ge 1-\epsilon-10^{-9}.
\]

Both this actual squared fidelity and the upstream approximation estimate are
recorded. The raw squared overlap is retained alongside a value clipped to
`[0, 1]` to handle numerical roundoff. Tests compare expanded circuits to the
original upstream definitions, check complex random q2/q3 inputs and a complex
q6 exact endpoint, and verify deterministic repetition and rejection of invalid
inputs, dependency versions, source hashes, and circuit outputs.

Each bound is a fresh original synthesis. CPU metadata includes setup, synthesis
(including lazy gate-definition expansion), and independent verification, with
these components also recorded separately. The cumulative cost charges all
requested bounds. The method is deterministic; optimizer seeds do not create
independent BAA repetitions. This wrapper does not select a noisy winner, apply
our noise fine-tuning, or implement a common compiler/noise model; those are
separate benchmark responsibilities.
