# Legacy weighted-trajectory comparison on all q6 targets

This extension uses every six-qubit vector from the normative
[`state_preparation_target_states.json`](../benchmarks/state_preparation_target_states.json)
fixture. It changes the targets and register size of the
[previous 3/100-history experiment](legacy_topdown_weighted_findings.md), retaining
the construction, pruning, optimizer, and noise settings.

The nine targets are Gaussian `gaussian_mu0p5_sigma0p1`; uniform TFIM ground
states `tfim_ferro`, `tfim_critical`, and `tfim_para`; complex Haar states
`haar_random_1`, `haar_random_2`, and `haar_random_3`; and `random_mps_bond2`
and `random_mps_bond3`. The last two are loaded as their exact dense vectors;
all training uses 64-amplitude statevectors and final evaluation uses independent
64-by-64 Qiskit density matrices. No twelve-qubit cases are included.

## Frozen experiment

Optimizer seeds 300 and 400 are applied to every target. Target-generation
seeds remain the fixture's seeds. For each target/optimizer-seed pair:

1. Construct the archived depth-four VQA reference with 300 three-trajectory
   noisy batch steps, rate 0.2 and exponential decay 0.01. This is a reference
   without a separate fine-tuning stage.
2. Construct the archived layerwise depth-one/depth-two circuit with 100
   noiseless batch steps per depth, rate 1.0. Fine-tune for 100 batch steps at
   rate 0.1, retaining its original per-step noise refresh, under the IBM profile.
3. Construct archived top-down depths 4, 8, 12, and 16, each with 150 noiseless
   batch steps at rates 5, 2, 0.5, and 0.2, with decay 0.01. Iteratively rank-prune
   `abs(theta * dF/dtheta)`, deleting at most 150 parameters per pass toward a
   final 150; between nonfinal passes repair for 40 steps at rate 0.2. Fine-tune
   for 200 batch steps at rate 0.1 with frozen CRN under the IBM profile.
4. Independently run literal noiseless growth at depths 1 through 16, with up
   to 200 updates per depth (first 100 online, remaining batch), rate 0.7, stopping
   at the first fidelity of at least 90%. Apply the unchanged one-shot absolute
   `0.01` impact cutoff. Fine-tune the pruned circuit and an unpruned control
   under IBM and Google separately, for 200 hybrid steps (100 online, 100 batch)
   at rate 0.2 with frozen CRN.
5. Pair every fine-tuning stage: start from identical gates and parameters and
   compare the original three histories with 100 sampled histories compressed
   by empirical no-jump multiplicities. Apply weights to both cross indices.
   Preserve counters, rates, phase schedules, and the first three sampled paths.
6. Score fixed final parameters under both archived noise profiles using exact
   density matrices. Store both the exact sampler law (the main comparison) and
   the full Lindblad exponential, plus noiseless fidelity. Exact test scores do
   not select parameters, learning rates, depth, seeds, or checkpoints.

This gives 18 target/seed jobs, 108 paired fine-tuning conditions, and 18 VQA
references. Completed conditions include empty cutoff circuits. A nonempty
circuit whose 100-path bank has no jumps is recorded as a coverage failure;
it is not resampled using another seed or artificially assigned a jump. If the
literal construction never reaches 90%, its four fine-tuning conditions are
recorded as construction failures. Report these cases in the full denominator.

For six qubits, the unchanged depth-four VQA has 198 gates and depth-two
layerwise circuit has 108 gates. The archived rank-pruning floor remains
150 gates. Therefore the two archived methods no longer have equal final gate
counts at six qubits; changing that floor would be another algorithm change.
No inference of a matched-gate or matched-compute advantage is warranted.

The benchmark definition is used for its target vectors. The existing logical
IBM/Google Pauli coefficients and at-most-one-jump gate windows remain unchanged.
This is a separately labelled noise-aware extension, not a run of the base
benchmark's Ballarin/dephasing/depolarizing condition matrix.

## Execution and provenance

```sh
uv run python -m experiments.legacy_topdown_q6 \
  --output output/legacy-topdown-q6-weighted100-20260914 --workers 4
```

Independent target/seed jobs use up to four processes with one BLAS thread each.
Each job owns its files; the parent writes the aggregate in canonical target/seed
order. Seeds do not depend on worker count or completion order. Report these
wall times as concurrent execution measurements, separate from previous serial
q8 timing results.

The manifest hashes the normative fixture, six source files, protocol, and package
versions. Source snapshots and the fixture accompany it. Every circuit checkpoint
has JSON and parameter hashes, and completed-job records authenticate all job
artifacts. `--resume` requires the same source, fixture and environment. Existing
q8 experiment scripts, outputs, and fixture amplitudes remain untouched.
