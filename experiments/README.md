# Minimal noisy-Krotov refresh experiment

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
uv run python experiments/noisy_krotov_refresh_minimal.py \
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
uv run python experiments/noisy_krotov_refresh_minimal.py \
  --output-root /path/to/refresh5-run \
  --resume
```

Training is checkpointed after each five-update refresh window. Evaluation is saved
after every batch. The final summary is `refresh_study_result.json`; the detailed
per-update trace, sampled training ensembles, per-trajectory paired evaluation rows,
and final circuit checkpoint remain beside it.

For a quick end-to-end check, add `--updates 1 --trajectories 1
--refresh-interval 1 --evaluation-trajectories 2 --evaluation-batch-size 1`.
