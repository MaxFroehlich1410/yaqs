# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Published AQCE scheduling around unchanged MNISQ tensor/gate primitives.

Algorithms 1-5 of Shirakawa, Ueda and Yunoki, Phys. Rev. Research 6,
043008 (2024), https://doi.org/10.1103/PhysRevResearch.6.043008.
Gate lists use execution order (the rightmost matrix acts first). The upstream
Pauli expansion orders a pair (i, j) as Qulacs targets [j, i]. See
AQCE_PROVENANCE.md for the independently licensed upstream resource.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


class StateProtocol(Protocol):
    """Minimal Qulacs state interface without requiring the optional dependency."""

    def copy(self) -> StateProtocol:
        """Return an independent state copy."""
        ...

    def get_qubit_count(self) -> int:
        """Return the state register size."""
        ...

    def set_zero_state(self) -> None:
        """Set all amplitudes to the computational zero state."""
        ...

    def get_vector(self) -> NDArray[np.complex128]:
        """Return the amplitudes for independent verification."""
        ...


class GateProtocol(Protocol):
    """Minimal interface of the reused Qulacs unitary gates."""

    def get_inverse(self) -> GateProtocol:
        """Return the unitary inverse gate."""
        ...

    def update_quantum_state(self, state: StateProtocol) -> None:
        """Apply this gate to the supplied state in place."""
        ...

    def get_matrix(self) -> NDArray[np.complex128]:
        """Return the local unitary for circuit conversion and verification."""
        ...

    def get_target_index_list(self) -> list[int]:
        """Return the ordered target qubits for conversion and verification."""
        ...


def initialize(upstream: ModuleType, target: StateProtocol, count: int) -> list[GateProtocol]:
    """Build Algorithm 4's density-eigenvector initialization in execution order.

    Args:
        upstream: Verified MNISQ module providing the tensor and gate primitives.
        target: Qulacs target state, never modified.
        count: Number of initial two-qubit unitaries.

    Returns:
        Gates whose matrix product is V1 V2 ... Vcount, as in Algorithm 4.
    """
    residual = target.copy()
    gates = []
    for _ in range(count):
        best_value = -1.0
        best_pair = (0, 1)
        best_density = np.eye(4, dtype=complex)
        for pair in itertools.combinations(range(target.get_qubit_count()), 2):
            density = upstream.evaluate_Rho(target, residual, pair)
            value = float(np.linalg.eigvalsh(density)[-1])
            if value > best_value:
                best_value, best_pair, best_density = value, pair, density
        _, eigenvectors = np.linalg.eigh(best_density)
        gate = upstream.DenseMatrix(list(reversed(best_pair)), eigenvectors[:, ::-1])
        gates.insert(0, gate)
        gate.get_inverse().update_quantum_state(residual)
    return gates


def _optimal_gate(upstream: ModuleType, target: StateProtocol, phi: StateProtocol, psi: StateProtocol) -> GateProtocol:
    """Return the all-pair SVD maximizer using the unchanged upstream tensor."""
    best_value = -1.0
    best_pair = (0, 1)
    best_tensor = np.eye(4, dtype=complex)
    for pair in itertools.combinations(range(target.get_qubit_count()), 2):
        tensor = upstream.evaluate_F(target, phi, psi, pair)
        value = float(np.linalg.svd(tensor, compute_uv=False).sum())
        if value > best_value:
            best_value, best_pair, best_tensor = value, pair, tensor
    left, _, right = np.linalg.svd(best_tensor)
    return upstream.DenseMatrix(list(reversed(best_pair)), left @ right)


def optimize_pass(upstream: ModuleType, target: StateProtocol, gates: list[GateProtocol], *, backward: bool) -> None:
    """Apply Algorithm 1 or 2 in place, propagating the two environment states.

    Args:
        upstream: Verified MNISQ module.
        target: Qulacs target state, never modified.
        gates: Two-qubit gates in execution order; each matrix and pair may change.
        backward: Whether to update from the output end towards the input end.
    """
    phi = upstream.QuantumState(target.get_qubit_count())
    phi.set_zero_state()
    psi = target.copy()
    if backward:
        for gate in gates[:-1]:
            gate.update_quantum_state(phi)
        indices = range(len(gates) - 1, -1, -1)
    else:
        for gate in reversed(gates[1:]):
            gate.get_inverse().update_quantum_state(psi)
        indices = range(len(gates))
    for index in indices:
        updated = _optimal_gate(upstream, target, phi, psi)
        gates[index] = updated
        if backward and index > 0:
            gates[index - 1].get_inverse().update_quantum_state(phi)
            updated.get_inverse().update_quantum_state(psi)
        elif not backward and index + 1 < len(gates):
            gates[index + 1].update_quantum_state(psi)
            updated.update_quantum_state(phi)


def sweep(upstream: ModuleType, target: StateProtocol, gates: list[GateProtocol], count: int) -> None:
    """Apply Algorithm 3: each sweep is one forward pass then one backward pass.

    Args:
        upstream: Verified MNISQ module.
        target: Qulacs target state, never modified.
        gates: Gate list updated in place.
        count: Number of complete sweeps.
    """
    for _ in range(count):
        optimize_pass(upstream, target, gates, backward=False)
        optimize_pass(upstream, target, gates, backward=True)


def grow(upstream: ModuleType, target: StateProtocol, gates: list[GateProtocol], increment: int) -> None:
    """Apply Algorithm 5's growth step, updating only the new identity block.

    Args:
        upstream: Verified MNISQ module.
        target: Qulacs target state, never modified.
        gates: Existing execution-order list; new gates are prepended in place.
        increment: Number of new identity gates to optimize against C†target.
    """
    residual = target.copy()
    for gate in reversed(gates):
        gate.get_inverse().update_quantum_state(residual)
    block = [upstream.DenseMatrix([1, 0], np.eye(4, dtype=complex)) for _ in range(increment)]
    optimize_pass(upstream, residual, block, backward=True)
    gates[:0] = block


def encode(
    upstream: ModuleType,
    target: StateProtocol,
    *,
    initial_gates: int,
    growth_increment: int,
    max_gates: int,
    sweeps: int,
) -> list[GateProtocol]:
    """Run published Algorithm 5 using explicitly specified integer controls.

    Args:
        upstream: Verified MNISQ module.
        target: Qulacs state with amplitudes already validated by the adapter.
        initial_gates: M0, used once for density-based initialization.
        growth_increment: Positive delta M for subsequent identity-block growth.
        max_gates: Mmax, reachable exactly from M0 by the growth increment.
        sweeps: N complete forward/backward sweeps after initialization and growth.

    Returns:
        Optimized full two-qubit gates in execution order.

    Raises:
        ValueError: If integer controls are invalid or two-qubit gates cannot act.
    """
    for value in (initial_gates, growth_increment, max_gates, sweeps):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = "AQCE controls must be nonnegative integers, with a positive growth increment."
            raise ValueError(msg)
    if growth_increment == 0 or initial_gates > max_gates:
        msg = "AQCE needs a positive growth increment and initial_gates <= max_gates."
        raise ValueError(msg)
    if (max_gates - initial_gates) % growth_increment:
        msg = "AQCE max_gates must be reachable from initial_gates by the growth increment."
        raise ValueError(msg)
    if max_gates and target.get_qubit_count() < 2:
        msg = "AQCE requires at least two qubits for a positive gate budget."
        raise ValueError(msg)
    gates = initialize(upstream, target, initial_gates)
    sweep(upstream, target, gates, sweeps)
    for _ in range(initial_gates, max_gates, growth_increment):
        grow(upstream, target, gates, growth_increment)
        sweep(upstream, target, gates, sweeps)
    return gates
