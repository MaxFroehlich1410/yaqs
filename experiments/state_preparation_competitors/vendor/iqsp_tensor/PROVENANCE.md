# Ballarin numerical extract

- Upstream: <https://github.com/mballarin97/torchquantum>
- Commit: `5bbd7353f63dfc868c3b0a0d6f29f5c86506f207`
- Source: `torchquantum/functional/sun.py`
- Source SHA256: `5816b597b488e4652fddee1904a146e089e445b2312fd24baca8ed3e37c94f2a`
- License: MIT, preserved verbatim in `LICENSE`.

`sun.py` extracts the authors' Pauli-generator SU(4) exponential, preserving
its ordering and sign. It replaces global dtype and batching conventions with
explicit complex128 tensors and removes unrelated gate registration/imports.

The experiment backend implements exact contraction of pure circuit tensor
networks and locally purified noisy circuit tensor networks with opt_einsum
and PyTorch autograd. It is **not a vendored compressed CombTNDevice**. Each
logical noise window has a local Kraus purification leg. The gate and channel
network is contracted against a target MPS without materializing a global
state vector or density matrix as an explicit simulation representation.

This is the one-dimensional, untruncated circuit-network specialization. The
pinned authors' `NoisyCombTNDevice` and `truncated_svd_gesdd` were inspected but
are not used: the original noise model differs from this benchmark, and SVD
rank selection at the identity SU(4) origin can remove first-order entangling
directions. Exact contractions retain those directions and avoid degenerate
SVD derivatives. No SVD lies on a trainable parameter's computational path.

The only SVD is a one-time exact TT-SVD of each supplied target fixture. There
is no claim that this exact contraction backend reproduces the authors' large
system compression, memory scaling, or reported runtime. Contraction cost and
largest intermediate are recorded explicitly; exact contraction can still be
exponential. The original common logical noise model replaces the authors'
hardware-specific channel for a matched benchmark, and is disclosed as an
adaptation.
