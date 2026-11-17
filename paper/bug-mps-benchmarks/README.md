# BUG MPS paper benchmark data and scripts

This directory contains the scripts used to generate the uncompressed
dense-reference check and the 16-site BUG versus 2-TDVP comparisons in the
accompanying manuscript. The enclosing Git commit pins the benchmark scripts
and the exact YAQS implementation in one source tree. This branch is a paper
artifact; it is not intended to be merged into `main` or published as a YAQS
release.

The original numerical result files are included alongside the scripts. They
cover the six-site dense-reference study (Table I), the 16-site matched-parameter
comparison (Table II), the active bond-cap study (Table III), and the
runtime-accuracy plot (Figure 4).

## Access to the test data

Download this branch using Git, or choose **Code > Download ZIP** on GitHub:

```bash
git clone --single-branch --branch bug-paper-final https://github.com/MaxFroehlich1410/yaqs.git
cd yaqs/paper/bug-mps-benchmarks
```

The manuscript cites an immutable commit so readers can retrieve the exact
published version even if the branch changes later.

- `six_site_dense_reference_2026-08-17/raw_results.json` contains the dense
  refinement results and structural diagnostics for Table I.
- `l16_matched_optimized_2026-08-12/raw_results.json` and `summary.csv` contain
  the matched-parameter results for Table II.
- `l16_tradeoff_caps_2026-08-12/tradeoff_all_points.csv` contains all 48 points
  in Figure 4; `tradeoff_pareto_points.csv` contains its 21 Pareto points.
- `l16_tradeoff_caps_2026-08-12/timing_samples.csv` preserves individual timings;
  every Pareto point has three runs, while other points have one or three.
- `l16_tradeoff_caps_2026-08-12/cap_study.csv` supplies Table III.
- The trade-off directory also includes the canonical `raw_results.json`,
  initial-state/reference/MPO fixtures, final state vectors, the original figure,
  validation reports, and recorded environment/code provenance.
- `DATA_MANIFEST.json` records SHA-256 hashes of the recovered files. The original
  `MANIFEST.json` in the trade-off directory is historical: it predates figure
  formatting changes and references an optional spreadsheet outside this tree.
  Use `DATA_MANIFEST.json` to verify the files published here.

The published data were recovered from the authors' archived experiments.
Original data bytes are preserved. Validation of saved results does not rerun
simulations or reproduce machine-dependent wall-clock timings.

To recreate Figure 4 from the CSV data without running simulations, use the
existing `make_figure.py` script shown below. For a fresh simulation, run the
corresponding benchmark in a separate checkout so its outputs do not overwrite
the archived data.

## Contents

- `six_site_dense_reference_2026-08-17/run_benchmark.py` runs the asymmetric
  six-site, uncompressed dense-reference refinement study and records all
  ordering, reflection, endpoint, and input-preservation checks.
- `six_site_dense_reference_2026-08-17/validate_results.py` checks the complete
  saved output against the manuscript table and structural diagnostics.
- `l16_matched_optimized_2026-08-12/run_benchmark.py` runs the matched-parameter
  TFIM and Haldane--Shastry benchmarks.
- `l16_matched_optimized_2026-08-12/summarize.py` derives the compact summary
  table from `raw_results.json`.
- `l16_matched_optimized_2026-08-12/validate_results.py` validates the matched
  benchmark output.
- `l16_matched_optimized_2026-08-12/export_julia_hs_mpo.jl` exports the Julia
  Haldane--Shastry MPO for the optional cross-implementation tensor check.
- `l16_tradeoff_caps_2026-08-12/run_experiments.py` runs the runtime--accuracy
  and active-cap studies.
- `l16_tradeoff_caps_2026-08-12/validate_and_export.py` validates those runs and
  exports their derived tables.
- `l16_tradeoff_caps_2026-08-12/make_figure.py` creates the runtime--accuracy
  figure.
- `l16_tradeoff_caps_2026-08-12/build_workbook.mjs` creates the optional
  workbook export.

## Reproduction

Create the YAQS environment from the repository root:

```bash
uv sync
```

Run and validate the complete six-site dense-reference refinement table:

```bash
uv run python paper/bug-mps-benchmarks/six_site_dense_reference_2026-08-17/run_benchmark.py
uv run python paper/bug-mps-benchmarks/six_site_dense_reference_2026-08-17/validate_results.py
```

Run the complete matched-parameter benchmark and produce its table and
validation report:

```bash
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/run_benchmark.py
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/summarize.py
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/validate_results.py
```

Run the complete trade-off and bond-cap study from scratch, then validate it
and regenerate the figure:

```bash
uv run python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/run_experiments.py --stage all
uv run python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/validate_and_export.py
uv run --with matplotlib python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/make_figure.py
```

All runners expose smaller selectable grids through `--help`. The benchmark
configuration used Python 3.12.11, NumPy 2.4.6, and SciPy 1.18.0 on an
eight-core Apple M1 Pro. Construction, warm-up, padding, and reference-state
calculation are excluded from the reported timings by the runners.
