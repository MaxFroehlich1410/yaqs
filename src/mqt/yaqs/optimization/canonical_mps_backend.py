# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional canonical MPS sweeps with a configurable relative propagation cutoff.

Only numerical representation changes: trajectory weights, gate order and
optimizer signals are inherited from MPSBackend. This is an approximation,
with observed SVD tails, rather than the default untruncated implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..core import linalg
from ..core.data_structures.mps import MPS
from ..core.linalg.svd_utils import observe_truncations
from ..core.methods.decompositions import left_qr, right_qr
from .mps_backend import MPSBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.methods.decompositions import TruncMode
    from .parameterized_circuit import ParameterizedCircuit

_EPSILON = float(np.finfo(np.float64).eps)


class _CanonicalState(MPS):
    """Private replace-only tensors and an explicitly tracked orthogonality center."""

    def __init__(self, state: MPS) -> None:
        """Canonicalize private input tensors without normalizing or changing phase."""
        super().__init__(state.length, tensors=list(state.tensors))
        self.flipped = state.flipped
        self.center = self.length - 1
        self.move_center(0)
        for site in range(self.length - 1):
            tensor = self.tensors[site]
            physical, left, right = tensor.shape
            u, singular, vh = linalg.svd(tensor.reshape(physical * left, right), full_matrices=False)
            keep = linalg.truncate(singular, mode="relative", threshold=_EPSILON, min_keep=1)
            self.tensors[site] = u[:, :keep].reshape(physical, left, keep)
            self.tensors[site + 1] = np.einsum("ij,ajk->aik", singular[:keep, None] * vh[:keep], self.tensors[site + 1])
            self.center = site + 1

    def move_center(self, site: int) -> None:
        """Shift the known canonical center by QR, replacing only crossed tensors."""
        while self.center < site:
            index = self.center
            q, r = right_qr(self.tensors[index])
            self.tensors[index] = q
            self.tensors[index + 1] = np.einsum("ij,ajk->aik", r, self.tensors[index + 1])
            self.center += 1
        while self.center > site:
            index = self.center
            q, r = left_qr(self.tensors[index])
            self.tensors[index] = q
            self.tensors[index - 1] = np.einsum("aij,jk->aik", self.tensors[index - 1], r)
            self.center -= 1

    def _apply_adjacent_two_site_operator(
        self,
        matrix: NDArray[np.complex128],
        left_site: int,
        *,
        threshold: float,
        max_bond_dim: int | None,
        min_bond_dim: int,
        trunc_mode: TruncMode,
    ) -> None:
        """Split a canonical two-site center, including every long-range SWAP."""
        self.move_center(left_site)
        super()._apply_adjacent_two_site_operator(
            matrix,
            left_site,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_bond_dim=min_bond_dim,
            trunc_mode=trunc_mode,
        )
        self.center = left_site + 1


class CanonicalMPSBackend(MPSBackend):
    """Opt-in canonical compression, without a hard cap or renormalization.

    Each propagated state has a known canonical center. QR moves that center
    before one-site operations and before each two-site SVD. Only singular
    values smaller than the propagation cutoff times the largest are discarded.
    Inputs retain double-precision epsilon canonicalization independently of the
    propagation cutoff. Local tangent contractions keep the output gauge outside
    the derivative's support.
    """

    def __init__(
        self,
        circuit: ParameterizedCircuit,
        target: MPS | NDArray[np.complex128],
        initial_state: MPS | NDArray[np.complex128] | None = None,
        *,
        relative_cutoff: float = _EPSILON,
    ) -> None:
        """Keep input precision while selecting a private propagation tolerance.

        Args:
            circuit: Parameterized circuit to propagate.
            target: Original target state, canonicalized at machine epsilon.
            initial_state: Optional initial state, canonicalized at machine epsilon.
            relative_cutoff: Relative singular-value threshold for propagation only.

        Raises:
            ValueError: If the propagation cutoff is not finite and between zero and one.
        """
        if (
            isinstance(relative_cutoff, (bool, np.bool_))
            or not np.isfinite(relative_cutoff)
            or not 0 < relative_cutoff < 1
        ):
            msg = "relative_cutoff must be finite and strictly between zero and one."
            raise ValueError(msg)
        self.relative_cutoff = float(relative_cutoff)
        super().__init__(circuit, target, initial_state)
        if initial_state is None:
            self.initial = _CanonicalState(self.initial)

    def _state(self, value: MPS | NDArray[np.complex128]) -> MPS:
        """Copy, validate and canonically compress an input without rescaling it.

        Returns:
            A private canonical MPS with its global phase preserved.
        """
        state = super()._state(value)
        with observe_truncations(self.diagnostics.split):
            return _CanonicalState(state)

    def _apply(self, state: MPS, matrix: NDArray, sites: tuple[int, ...], role: str) -> None:
        """Apply a canonical propagation, or a gauge-local untruncated derivative."""
        if role == "derivative":
            # Canonical gauge moves outside this interval would invalidate the
            # mixed-environment tangent locality contract. This temporary state
            # is used only for contractions, never for subsequent propagation.
            plain = MPS(state.length, tensors=list(state.tensors))
            super()._apply(plain, matrix, sites, role)
            state.tensors = plain.tensors
            return
        canonical = cast("_CanonicalState", state)
        if len(sites) == 2 and sites[0] > sites[1]:
            sites = sites[::-1]
            matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        if len(sites) == 1:
            canonical.move_center(sites[0])
        with observe_truncations(self.diagnostics.split):
            canonical.apply_local_operator(matrix, sites, threshold=self.relative_cutoff, trunc_mode="relative")
        self.diagnostics.operator_applications += 1
        self.diagnostics.observe(canonical, role)

    def _apply_all(self, states: list[MPS], matrix: NDArray, sites: tuple[int, ...], role: str) -> None:
        """Share only identical full canonical inputs, including center metadata."""
        if role == "derivative":
            super()._apply_all(states, matrix, sites, role)
            return
        evaluated: dict[tuple, tuple] = {}
        for state in states:
            canonical = cast("_CanonicalState", state)
            inputs = tuple(canonical.tensors)
            key = canonical.center, tuple(id(tensor) for tensor in inputs)
            previous = evaluated.get(key)
            if previous is None:
                self._apply(canonical, matrix, sites, role)
                evaluated[key] = inputs, tuple(canonical.tensors), canonical.center
            else:
                canonical.tensors = list(previous[1])
                canonical.center = previous[2]
                self.diagnostics.reused_local_applications += 1
                self.diagnostics.observe(canonical, role)
