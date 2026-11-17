# Corrected matched L=16 BUG versus 2TDVP benchmark

All rows use `min_keep=2`, relative discarded weight `1e-12`, `chi_max=512`, a shared deterministically padded `chi=4` initial MPS, and the shared pure-NumPy matrix-free adaptive Lanczos implementation (maximum dimension 25, tolerance `1e-12`). Timings are medians of three warmed runs and exclude setup, diagnostics, and exact references.

The run used optimized Python YAQS on branch `codex/julia-equivalent-bug`
(`d2d525f`, plus the uncommitted BUG optimizations recorded in the worktree),
Python `3.12.11`, NumPy `2.4.6`, and SciPy `1.18.0`, on an 8-core Apple M1
Pro MacBook Pro with 16 GB RAM and macOS `26.6`.

## TFIM

| dt | BUG time | 2TDVP time | TDVP/BUG | BUG infid. | 2TDVP infid. | BUG max chi | 2TDVP max chi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.598 s | 2.286 s | 1.431 | 6.4194e-08 | 6.4854e-09 | 8 | 8 |
| 0.005 | 3.027 s | 4.153 s | 1.372 | 1.1643e-07 | 3.2007e-08 | 8 | 8 |
| 0.0025 | 5.479 s | 7.706 s | 1.407 | 5.2532e-07 | 1.5155e-07 | 8 | 8 |
| 0.00125 | 10.508 s | 14.214 s | 1.353 | 2.3687e-06 | 7.3689e-07 | 8 | 8 |

## periodic Haldane-Shastry

| dt | BUG time | 2TDVP time | TDVP/BUG | BUG infid. | 2TDVP infid. | BUG max chi | 2TDVP max chi |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 58.889 s | 63.991 s | 1.087 | 1.5991e-05 | 1.6802e-05 | 143 | 143 |
| 0.005 | 82.252 s | 89.370 s | 1.087 | 4.3753e-06 | 4.4227e-06 | 122 | 123 |
| 0.0025 | 100.732 s | 116.167 s | 1.153 | 3.0214e-06 | 2.1780e-06 | 92 | 103 |
| 0.00125 | 130.223 s | 151.073 s | 1.160 | 1.1402e-05 | 5.7623e-06 | 70 | 81 |

## Interpretation

- BUG was faster in every one of the 24 paired timing repetitions.  At the
  medians, 2TDVP took `1.353`--`1.431` times the BUG time for TFIM and
  `1.087`--`1.160` times the BUG time for Haldane--Shastry.  Equivalently, BUG
  saved `26.1`--`30.1%` of the 2TDVP runtime for TFIM and `8.0`--`13.8%` for
  Haldane--Shastry.
- The work counters explain the direction of the timing result without making
  it universal.  2TDVP made `57` local Krylov calls per step versus `32` for
  BUG and used `1.72`--`1.85` times as many Krylov operator applications.  The
  long-range MPO and larger ranks make BUG's non-Krylov sweep/compression work
  comparatively more important, so the wall-clock advantage is much smaller
  there.
- Accuracy is mixed.  For TFIM, 2TDVP had `3.2`--`9.9` times smaller
  infidelity at the four matched step sizes.  For Haldane--Shastry, BUG was
  slightly more accurate at `dt=0.01` and `0.005`, while 2TDVP was more
  accurate at `0.0025` and `0.00125`.  Neither method has a general accuracy
  advantage on this grid.
- Infidelity is nonmonotone at the smaller time steps because a fixed
  per-compression tolerance causes more truncations as the number of steps
  grows.  These points are a matched-parameter cost study, not an order test.
- The maximum bond dimensions were only `8` for TFIM and `70`--`143` for the
  long-range model, so the cap `512` was inactive.  The results probe the
  adaptive truncation rule rather than cap saturation.

## Validation and provenance

- Both methods started from clones of one deterministically perturbed and
  right-canonicalized MPS whose maximum initial bond dimension was `4`.
- BUG compressed after each half-sweep and retained at least two singular
  directions on every bond.  Complete first-step profiles were recorded after
  both half-sweeps and both compressions in `raw_results.json`.
- The direct TFIM MPO equals the analytic sparse Hamiltonian exactly.  The
  Haldane--Shastry MPO has relative Frobenius error `2.30e-16` against the
  analytic operator; all `1500` nonzero tensor entries also match Julia's
  bond-47 FSM MPO elementwise with maximum difference `0`.
- `VALIDATION.md` records `86/86` passing structural and numerical checks.

## Recommended short manuscript statement

> We impose a minimum retained bond dimension of two in all adaptive runs.
> This rank floor prevents early half-sweep compressions from repeatedly
> collapsing a nearly rank-one bond and discarding every newly generated
> low-weight direction before the opposite sweep can propagate it.  It is a
> numerical rank-growth safeguard, not a claim that rank-one states are
> intrinsically invalid.

The runtime claim should be restricted to the measured setting: corrected BUG
was faster at all matched step sizes in both `L=16` models, by a substantial
margin for TFIM and a modest margin for the exact bond-47 long-range MPO.  The
paper should not claim a universal BUG accuracy advantage; it should say that
2TDVP was more accurate for TFIM, whereas the long-range result depended on the
step size.
