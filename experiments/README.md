# Noisy state-preparation experiments

## Published research source

This branch includes the top-down state-preparation experiments, legacy
pruning and refinement studies, trajectory and sampling experiments, the
[competitor comparison package](state_preparation_competitors/README.md), and
the [overnight experiment package](overnight_state_preparation_20260913_0108/HANDOFF.md).
Protocols and written findings accompany the scripts. See
[top-down method documentation](topdown_state_preparation_method.md) for the
current algorithm and [benchmark documentation](../benchmarks/state_preparation_benchmarks.md)
for the target and evaluation conventions.

Generated output, parameter checkpoints, numerical result files, plots, and
manuscript build artifacts stay local. Archive-driven replay and analysis
need the corresponding local inputs; a GitHub clone contains the source,
protocol configurations, and small automated-test reference fixtures.

## Canonical q6 causal study

`noisy_krotov_causal_study.py` tests whether noisy fine-tuning improves fresh
noisy fidelity, rather than merely fitting a small set of training
trajectories. It creates one layerwise noiseless checkpoint for the canonical
six-qubit `tfim_critical` target and starts every arm from the exact same
parameters. The five arms are:

- the frozen pretrained checkpoint;
- an equally long noiseless continuation;
- three-trajectory cross updates with fixed common random numbers (CRN);
- the same updates with CRN refreshed every five iterations; and
- the same updates with trajectories resampled every iteration.

A disjoint fixed ensemble is used only for checkpoint selection. Final
evaluation uses fresh held-out maps, and every map is replayed across all five
selected circuits for paired comparison. The standard profile is
`depolarizing_1s_all`; its Pauli jump probabilities are state independent, so
the paired replay remains unbiased for each circuit.

Run the registered defaults with:

```console
uv run python -m experiments.noisy_krotov_causal_study \
  --output-root /path/outside/the/repository/q6-causal-study
```

The defaults use 100 noiseless updates at each depth from one through four,
200 fine-tuning updates, 3 training trajectories, a refresh interval of 5,
12 validation trajectories, 500 final trajectories, and 3 independent noisy-
training replicates. Resume using the same arguments plus `--resume`.
Pretraining is checkpointed at depth boundaries, fine-tuning at validation
boundaries, and evaluation after every batch. The output includes every sampled
map, raw trajectory fidelity, paired difference, selected parameter vector,
training trace, seed audit, and a replicate-level Student-t summary.

For a quick end-to-end check, use:

```console
uv run python -m experiments.noisy_krotov_causal_study \
  --output-root /tmp/q6-causal-smoke \
  --pretraining-iterations-per-depth 1 \
  --finetune-updates 1 \
  --refresh-interval 1 \
  --validation-interval 1 \
  --training-trajectories 1 \
  --validation-trajectories 1 \
  --test-trajectories 2 \
  --test-batch-size 1 \
  --replicates 1
```

## Legacy q8 refresh experiment

This directory retains one exploratory experiment: fine-tune the saved noiseless q8,
depth-four circuit against three noisy trajectories, replace those trajectories every
five optimizer updates, and compare the result with both the noiseless checkpoint and
the historical circuit trained on three fixed trajectories.

The runner is deliberately direct. It uses the reusable Krotov primitives from
`mqt.yaqs.optimization` and has no benchmark matrix, competing-method, preregistration,
source-lock, custody, or work-package layer.

## Inputs

Point `--bundle-root` to a directory containing:

- `pretrained_depth4_checkpoint.npz`: the stored noiseless circuit with a
  `selected_theta` array.
- `historical_fixed3_post_checkpoint.npz`: the earlier fixed-three noisy circuit with
  a `selected_theta` array.

The exact disordered-TFIM q8 target is reconstructed from seed 100 and checked against
the retained vector checksum.

## Full run

```console
uv run python -m experiments.noisy_krotov_refresh_minimal \
  --bundle-root /path/to/pretrained-bundle \
  --output-root /path/to/refresh5-run
```

The defaults are the agreed experiment: 200 updates, three training trajectories,
refresh every five updates, training seed `17580007066253930890`, and 2,000 fresh
paired evaluation trajectories in batches of 250. To use four cores without BLAS
oversubscription, launch four independent repetitions with different training and
evaluation seeds rather than splitting one optimizer trajectory set across processes.

Resume an interrupted run with the same scientific arguments:

```console
uv run python -m experiments.noisy_krotov_refresh_minimal \
  --output-root /path/to/refresh5-run \
  --resume
```

Training is checkpointed after each five-update refresh window. Evaluation is saved
after every batch. The final summary is `refresh_study_result.json`; the detailed
per-update trace, sampled training ensembles, per-trajectory paired evaluation rows,
and final circuit checkpoint remain beside it.

For a quick end-to-end check, add `--updates 1 --trajectories 1
--refresh-interval 1 --evaluation-trajectories 2 --evaluation-batch-size 1`.
