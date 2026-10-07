# Ballarin memory attribution diagnostic

This diagnostic changes no target, noise, compression or update rule. It is
not a new training experiment or an accuracy certification. Start with the
same twelve-qubit Gaussian depth-three frozen checkpoint used in the failed
tighter-gradient run. Compare default and tighter forward-only execution;
reuse the saved gradient process receipts rather than repeat a failed full
gradient calculation. Expand to depth four only if the results warrant it.

The forward worker calls the existing `fidelity` method under `no_grad`.
Instrumentation wraps the original `torch.linalg.svd` without changing the
decomposition or retaining input/output tensors. It records input shape,
strides, logical/backing bytes, factor bytes, time and singular-value hashes.
The same TCI target, logical circuit, parameters and numerical settings are
authenticated against the earlier results. A default forward result must
match its previous fidelity to absolute tolerance 1e-10.

An independent supervisor samples RSS and physical footprint every 0.25 s,
attributing samples to the active SVD call or other execution. These are
sampled process peaks, not exact per-operation allocations; fast calls can be
missed and allocator retention affects apparent incremental costs. The 8 GiB
process guard and 5,400-second deadline remain. A 512 MiB free-disk floor
protects the nearly full host; no unrelated data may be deleted.

After the forward trace, select the largest observed SVD input, reporting also
the call with the highest sampled memory. Reconstruct the circuit prefix and
capture that exact matrix immediately before its SVD. No decomposition is
performed at the capture point. Capture requires the file size plus a one-GiB
disk reserve. Replay the same thin SVD on that input in a fresh worker with no
circuit, target or gradient tape. Preserve dtype and values, record input
layout, and verify the matrix file hash. Capture only this matrix rather than
an archive of all intermediate tensors. On a nearly full host, remove this
reproducible scratch file after the probes, preserving its hash, capture
receipt and prefix-regeneration command.

Also replay the existing author-derived regularized SVD backward in a fresh worker, supplying
dense unit cotangents for U, S and V. This exercises the derivative buffers
without any circuit tape. These cotangents are synthetic: the resulting VJP
is not a noisy circuit gradient and is never used for training. The measured
workspace is evidence about this execution, not a bound for every cotangent
or numerical input. The separate entry point is `ballarin_memory_backward`;
the same supervisor and captured matrix are reused.
The kernel includes the previously audited projector reassociation and buffer
reuse documented in `vendor/ballarin_comb/PROVENANCE.md`; this diagnostic adds
no numerical changes to that implementation.

Compare forward-only, isolated-decomposition and prior gradient peaks. If a
single decomposition reaches the guard, checkpoint offloading alone cannot
address that local workspace demand. If only the gradient run is large,
retained/recomputed autograd data and backward workspace remain candidates;
the measurements do not by themselves distinguish all of those contributions.
Neither outcome justifies lowering tolerances or claiming a competitor failed
to prepare the state.

Entry point: `python -m experiments.state_preparation_competitors.ballarin_memory`.
Modes are `forward`, `isolated`, and `supervise`; `forward --capture-index I`
replays only the prefix and writes the selected matrix. Use fresh output
directories and one large worker at a time. Source hashes, phase traces,
input receipts, result receipts and supervisor logs accompany every run.

The [2026-09-27 report](../../output/q12-ballarin-memory-20260927/REPORT.md)
records a completed default forward pass and isolated SVD replay. Tighter
forward-only propagation exceeded 8 GiB. The backward probe stopped at the
disk guard during its forward SVD, so it supplies no backward-memory result.
