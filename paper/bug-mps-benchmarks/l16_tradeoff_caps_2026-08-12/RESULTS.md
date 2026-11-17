# L=16 runtime-accuracy and active-cap results

All runs use the corrected optimized YAQS BUG schedule, a shared deterministic
initial MPS padded once to `chi=4`, direct Julia-gauge MPO tensors, relative
discarded-weight truncation, `min_keep=2`, and the same matrix-free adaptive
Lanczos implementation for BUG and 2TDVP.  The final time is `T=1`.

## Runtime-accuracy sweep

The tested grid is

- `dt in {0.01, 0.005, 0.0025}`;
- `epsilon in {1e-8, 1e-10, 1e-12, 1e-14}`;
- `chi_max=512`.

Each full-grid point has one diagnostic replay and at least one timed replay.
Every point on the final runtime-infidelity Pareto envelope has three timing
samples; the figure reports the median and uses the observed minimum/maximum as
horizontal error bars. Setup, exact references, diagnostics, warm-up, and file
output are excluded from the timed region.

For the transverse-field Ising model, BUG is faster at the matched parameter
points, whereas 2TDVP reaches substantially lower infidelity at the highest
accuracy end of this grid.  At `dt=0.01`, the stringent `epsilon=1e-14` point is
`1.258 s`, `3.903e-8` for BUG and `1.757 s`, `7.194e-11` for 2TDVP.  Hence this
model does not support a universal equal-accuracy speed advantage for BUG.

For Haldane-Shastry, the two Pareto envelopes are much closer in accuracy and
BUG remains faster throughout the observed frontier.  The highest-accuracy
point at `dt=0.0025`, `epsilon=1e-14` is `234.245 s`, `1.011e-6` for BUG and
`246.583 s`, `1.053e-6` for 2TDVP.  These are empirical results for this tested
grid, not a universal dominance claim.

## Active-cap Haldane-Shastry study

The cap study fixes `dt=0.005` and `epsilon=1e-12`.

| chi_max | BUG time (s) | 2TDVP time (s) | 2TDVP/BUG | BUG infidelity | 2TDVP infidelity | attained chi (BUG/2TDVP) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 23.650 | 24.373 | 1.031 | 1.287e-4 | 1.676e-4 | 32 / 32 |
| 64 | 54.320 | 55.344 | 1.019 | 5.850e-6 | 5.977e-6 | 64 / 64 |
| 96 | 64.274 | 68.549 | 1.067 | 4.403e-6 | 4.461e-6 | 96 / 96 |
| 512 | 82.252 | 89.370 | 1.087 | 4.375e-6 | 4.423e-6 | 122 / 123 |

All finite caps `32`, `64`, and `96` are active for both methods. BUG is faster
and slightly more accurate at each tested cap, but the runtime advantage is
small at the tightest caps. The cap-96 result is already close to the uncapped
accuracy at appreciably lower cost.

## Reproducibility

- `raw_results.json` is the canonical structured result file.
- `timing_samples.csv` retains every individual timing sample.
- `tradeoff_all_points.csv` and `tradeoff_pareto_points.csv` are flat plotting
  inputs.
- `cap_study.csv` is the active-cap table source.
- `bug_first_step_checkpoints.csv` stores every first-step BUG bond dimension at
  all four algorithmic checkpoints.
- `fixtures/` contains the exact initial states, references, and MPO tensors;
  `states/` contains final dense states for newly measured configurations.
- `provenance/` contains the YAQS commit, dirty patch, environment, dependency
  list, and imported matched-baseline source.
- `MANIFEST.json` records file hashes, and `VALIDATION.md` records 570 passing
  structural and numerical checks.
