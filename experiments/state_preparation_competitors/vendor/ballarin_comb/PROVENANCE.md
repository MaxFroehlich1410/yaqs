# Compressed Ballarin comb kernels

Upstream: <https://github.com/mballarin97/torchquantum>

Pinned commit: `5bbd7353f63dfc868c3b0a0d6f29f5c86506f207`.

Paper: [Ballarin et al., arXiv:2511.15674v1](https://arxiv.org/html/2511.15674v1).

`source_hashes.json` records SHA256 hashes of the original source files.
The upstream MIT license is preserved in `LICENSE` and in the extracted files.

## Reused source

- `torchquantum/device/combdevice.py`: `CombTNDevice` initialization,
  indexing, overlap, local gate application and orthogonality-center movement.
- `torchquantum/device/noisy_combdevice.py`: the corresponding locally
  purified comb operations, including contractions against a pure comb target.
- `torchquantum/linalg/svd_gesdd.py`: `safe_inverse`, `safe_inverse_2`, and
  the active `SVDGESDD.forward` and regularized `backward` methods.
- `torchquantum/linalg/custom_svd.py`: `truncated_svd_gesdd` and its helper.
  The relative discarded-squared-norm rank rule, maximum rank, norm-restoring
  rescaling and regularized derivative are retained.

Unused methods, global gate registration, alternative decompositions and package
imports were removed. AST extraction removed comments; formatting was applied.
This is an attributed numerical extract, not an installation of TorchQuantum.

## Explicit changes

1. Use complex128/float64 throughout, matching our numerical benchmark precision.
2. Make the existing SVD callable replaceable per instance solely to count
   splits, record discarded weights and enforce local allocation limits.
3. Correct the pure device's leftward QR transfer from `jt @ R` to `jt @ R.T`
   through its contraction-axis choice. The noisy upstream code already has
   the corrected axis. Complex-state center-transport tests expose the error.
4. Correct the noisy root-root two-site contraction permutation: the second
   site's physical and environment axes were reversed when its purification
   rank exceeded one. An independently evaluated six-qubit noisy circuit
   tests this regression.
5. Remove implicit hardware-noise injection from the extracted two-site gate
   method. The wrapper applies the channel explicitly, then uses the same SVD
   rule for local purification compression. This keeps channel choice visible.

6. Associate the SVD backward projector contractions as
   `(I - U U^H) W = W - U (U^H W)` and
   `W (I - V V^H) = W - (W V) V^H`. This is the identical regularized derivative
   with skinny intermediates instead of explicit square projectors. The latter
   otherwise require gigabytes for rectangular purification splits. Independent
   exact-linear-loss and allocation-shape regressions cover both orientations.

7. Accumulate the backward terms into one owned buffer, in the original order
   `(u_term + sigma_term) + v_term + imag_term`, releasing terms after use.
   This reduces simultaneous full-matrix storage without changing the
   regularized first derivative, rank selection or forward SVD. Eight saved
   reference derivatives cover real/complex, rectangular/square, and
   repeated/zero singular-value inputs. The pre-change kernel and its hash
   are archived with the twelve-qubit diagnostic results. Higher derivatives
   are not used by the published Adam procedure and are not audited here.

No initialization jitter, optimizer replacement, pruning policy, straight-through
SVD derivative, gradient fallback, or automatic tolerance adjustment was added.
Vendored numerical files are excluded from first-party typing/style policies;
our wrapper and its tests are checked normally.

## Noise: paper versus public helper

The public `_zz_noise`/`apply_one_site_noise` helpers do not directly implement
all of the paper's stated channel settings: they can omit the constant error,
skip backbone roots, and use a different depolarization parameter convention.
They are therefore not called by this adapter.

`hardware="H2"` implements the paper's Eq. (8)-(9), with noiseless one-qubit
operations, the constant error retained, and independent local Pauli channels
on both RZZ endpoints. Angles use the benchmark's existing canonical signed
angle convention. Its angle derivative is included. Zero-angle gates retain
the constant error until explicitly deleted by postprocessing.

`hardware="IBM"` or `"Google"` is a declared common-noise adaptation using the
existing logical channel. Joint Kraus indices preserve correlated two-site
Pauli errors; replacing these with independent one-site noise is forbidden.

## Known derivative limitation

The upstream regularized SVD and state-dependent rank selection do not give
exact statevector gradients at singular degeneracies or rank changes. At the
all-identity SU(4) origin, the Bell-target test has the correct fidelity but
nearly zero compressed gradient, while the exact gradient is nonzero. This
behavior is preserved and explicitly tested; tiny numerical perturbations can
subsequently produce platform-dependent escape from that point.

Do not describe the compressed backend as exact-gradient equivalent at this
initialization, or treat a stalled run as evidence against the IQSP algorithm.
A campaign must check initialization/gradient behavior and convergence first.
