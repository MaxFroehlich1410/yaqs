# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Batched mixed MPS environments with bounded, tensor-identity-checked reuse.

Canonical forms are not assumed: the two states can have different bases,
bond dimensions and norms. Private backend snapshots replace tensor arrays
on every local mutation. That contract makes identity-based invalidation safe.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mps import MPS


def advance_left(environment: NDArray, bra: NDArray, ket: NDArray) -> NDArray:
    """Extend batched left environments using two matrix multiplications.

    Args:
        environment: Pair, bra-left, ket-left axes.
        bra: Pair, physical, left, right axes; not conjugated yet.
        ket: Pair, physical, left, right axes.

    Returns:
        Pair, bra-right, ket-right environment, without path planning.
    """
    count, physical, _, right = bra.shape
    intermediate = bra.conj().transpose(0, 1, 3, 2) @ environment[:, None, :, :]
    return intermediate.transpose(0, 2, 1, 3).reshape(count, right, -1) @ ket.reshape(
        count, physical * ket.shape[2], ket.shape[3]
    )


def advance_right(environment: NDArray, bra: NDArray, ket: NDArray) -> NDArray:
    """Extend batched right environments without assuming common Schmidt bases.

    Returns:
        Pair, bra-left, ket-left environment.
    """
    count, physical, left, _ = bra.shape
    intermediate = bra.conj() @ environment[:, None, :, :]
    return intermediate.transpose(0, 2, 1, 3).reshape(count, left, -1) @ ket.transpose(0, 1, 3, 2).reshape(
        count, physical * ket.shape[3], ket.shape[2]
    )


def overlap(left: MPS, right: MPS) -> np.complex128:
    """Contract a scalar mixed overlap with no per-site contraction planner.

    Returns:
        The complex overlap, retaining arbitrary phase and normalization.

    Raises:
        ValueError: If register lengths differ.
    """
    if left.length != right.length:
        msg = "MPS overlap requires equal register sizes."
        raise ValueError(msg)
    environment = np.ones((1, 1, 1), dtype=np.complex128)
    for bra, ket in zip(left.tensors, right.tensors, strict=True):
        environment = advance_left(environment, bra[None], ket[None])
    return np.complex128(environment[0, 0, 0])


def _stack(tensors: list[NDArray]) -> NDArray:
    """Stack tensors, zero-padding unequal virtual dimensions exactly.

    Returns:
        A dense batch of local tensors, never a dense full-register state.
    """
    shapes = [tensor.shape for tensor in tensors]
    if all(shape == shapes[0] for shape in shapes):
        return np.stack(tensors)
    shape = tuple(max(s[axis] for s in shapes) for axis in range(3))
    result = np.zeros((len(tensors), *shape), dtype=np.complex128)
    for index, tensor in enumerate(tensors):
        result[index, : tensor.shape[0], : tensor.shape[1], : tensor.shape[2]] = tensor
    return result


class MixedEnvironments:
    """Reuse bounded left/right environments across local, replace-only changes.

    The cache is scoped to one optimizer call. Tensor references, not just ids,
    are retained to prevent id recycling. In-place writes to those arrays are
    forbidden by the private backend snapshot contract. Local tangent states
    may differ inside the supplied interval, including swap-chain intermediates.
    """

    def __init__(self, *, max_cache_bytes: int = 32 * 1024**2, max_pairs: int = 64) -> None:
        """Set independent cache and batch working-size limits.

        Raises:
            ValueError: If limits are invalid.
        """
        if max_cache_bytes < 0 or max_pairs < 1:
            msg = "Overlap cache limits must be nonnegative with at least one pair."
            raise ValueError(msg)
        self.max_cache_bytes = max_cache_bytes
        self.max_pairs = max_pairs
        self.environments: OrderedDict[tuple[int, str, int], NDArray] = OrderedDict()
        self.snapshots: dict[int, tuple[list[list[NDArray]], list[list[NDArray]]]] = {}
        self.cache_bytes = 0
        self.peak_cache_bytes = 0
        self.site_contractions = 0
        self.environment_reuses = 0
        self._layout: tuple[int, int, int, bool, int] | None = None

    def clear(self) -> None:
        """Release all tensor references and environments, retaining work counters."""
        self.environments.clear()
        self.snapshots.clear()
        self.cache_bytes = 0
        self._layout = None

    def _forget(self, key: tuple[int, str, int]) -> None:
        """Remove a cached environment and update byte accounting."""
        self.cache_bytes -= self.environments.pop(key).nbytes

    def _remember(self, key: tuple[int, str, int], value: NDArray) -> None:
        """Cache an environment if it fits, evicting least-recently used entries."""
        if key in self.environments:
            self._forget(key)
        if value.nbytes > self.max_cache_bytes:
            return
        while self.environments and self.cache_bytes + value.nbytes > self.max_cache_bytes:
            self._forget(next(iter(self.environments)))
        self.environments[key] = value
        self.cache_bytes += value.nbytes
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)

    def _prepare(self, batch: int, bras: list[MPS], kets: list[MPS]) -> None:
        """Invalidate only environments crossing changed physical-site tensors."""
        left = [list(state.tensors) for state in bras]
        right = [list(state.tensors) for state in kets]
        previous = self.snapshots.get(batch)
        if previous is not None:
            changed = [
                site
                for site in range(bras[0].length)
                if any(
                    before[site] is not after[site]
                    for old, new in zip(previous, (left, right), strict=True)
                    for before, after in zip(old, new, strict=True)
                )
            ]
            if changed:
                first, last = min(changed), max(changed)
                for key in list(self.environments):
                    index, direction, cut = key
                    if index == batch and (
                        (direction == "left" and cut > first) or (direction == "right" and cut <= last)
                    ):
                        self._forget(key)
        self.snapshots[batch] = left, right

    def _boundary(self, batch: int, cut: int, *, right: bool) -> NDArray:
        """Build or reuse a prefix/suffix up to a requested cut.

        Returns:
            A batch of mixed environments at the cut.
        """
        bras, kets = self.snapshots[batch]
        length, count = len(bras[0]), len(bras)
        direction = "right" if right else "left"
        available = [
            key[2]
            for key in self.environments
            if key[0] == batch and key[1] == direction and (key[2] >= cut if right else key[2] <= cut)
        ]
        if available:
            position = min(available) if right else max(available)
            key = batch, direction, position
            environment = self.environments[key]
            self.environments.move_to_end(key)
            self.environment_reuses += 1
        else:
            position = length if right else 0
            environment = np.ones((count, 1, 1), dtype=np.complex128)
        sites = range(position - 1, cut - 1, -1) if right else range(position, cut)
        for site in sites:
            a, b = _stack([state[site] for state in bras]), _stack([state[site] for state in kets])
            environment = (advance_right if right else advance_left)(environment, a, b)
            self.site_contractions += 1
            self._remember((batch, direction, site if right else site + 1), environment)
        return environment

    def local_overlaps(
        self,
        bras: list[MPS],
        kets: list[MPS],
        tangents: list[MPS],
        interval: tuple[int, int],
        *,
        diagonal: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Evaluate ordinary and locally modified overlaps with shared environments.

        Args:
            bras: Arbitrary backward states.
            kets: Arbitrary forward states, with replace-only tensors.
            tangents: Ket snapshots differing only within ``interval``.
            interval: Inclusive first/last changed sites (including swap chains).
            diagonal: Return matched pairs only and skip unused ordinary overlaps.

        Returns:
            Ordinary and tangent overlaps; matrices for cross pairs or vectors
            for matched pairs. The unused ordinary diagonal vector is zero.

        Raises:
            ValueError: If lengths, pair counts or the interval are invalid.
        """
        if not bras or not kets or len(tangents) != len(kets) or (diagonal and len(bras) != len(kets)):
            msg = "Invalid overlap ensembles."
            raise ValueError(msg)
        length = bras[0].length
        first, last = interval
        if any(state.length != length for state in (*bras, *kets, *tangents)) or not 0 <= first <= last < length:
            msg = "Invalid overlap register or local interval."
            raise ValueError(msg)
        # Bound batch operands and intermediates even when one MPS is much larger.
        largest = max(tensor.size for state in (*bras, *kets, *tangents) for tensor in state.tensors)
        pairs_per_batch = min(self.max_pairs, max(1, (16 * 1024**2) // (8 * 16 * largest)))
        layout = len(bras), len(kets), length, diagonal, pairs_per_batch
        if self._layout != layout:
            self.clear()
            self._layout = layout
        pairs = (
            [(i, i) for i in range(len(bras))]
            if diagonal
            else [(i, j) for i in range(len(bras)) for j in range(len(kets))]
        )
        plain, modified = np.zeros(len(pairs), dtype=np.complex128), np.empty(len(pairs), dtype=np.complex128)
        for batch, start in enumerate(range(0, len(pairs), pairs_per_batch)):
            group = pairs[start : start + pairs_per_batch]
            a, b = [bras[i] for i, _ in group], [kets[j] for _, j in group]
            self._prepare(batch, a, b)
            left, right = self._boundary(batch, first, right=False), self._boundary(batch, last + 1, right=True)
            derivative, ordinary = left, left
            for site in range(first, last + 1):
                bra = _stack([state.tensors[site] for state in a])
                tangent = _stack([tangents[j].tensors[site] for _, j in group])
                derivative = advance_left(derivative, bra, tangent)
                self.site_contractions += 1
                if not diagonal:
                    ordinary = advance_left(ordinary, bra, _stack([state.tensors[site] for state in b]))
                    self.site_contractions += 1
                    self._remember((batch, "left", site + 1), ordinary)
            stop = start + len(group)
            modified[start:stop] = np.sum(derivative * right, axis=(1, 2))
            if not diagonal:
                plain[start:stop] = np.sum(ordinary * right, axis=(1, 2))
        shape = (len(bras),) if diagonal else (len(bras), len(kets))
        return plain.reshape(shape), modified.reshape(shape)
