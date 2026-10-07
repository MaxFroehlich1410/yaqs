# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Weighted MPS state-preparation adjoints and sequential cross-trajectory sweeps.

Exact mode changes only the representation. With truncation, adjoints do not
include derivatives of SVD rank decisions and are approximate update signals.
Noisy cross signals retain the historical estimator, not the gradient of the
finite-sample mean fidelity. Both trajectory indices carry their probability.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.data_structures.mps import MPS
from ..core.linalg.svd_utils import TruncationEvent, observe_truncations
from .krotov import KrotovTruncation
from .mps_contractions import MixedEnvironments, overlap
from .trajectories import Estimator, NoiseMaps, TrajectorySampler

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from .parameterized_circuit import ParameterizedCircuit, ParameterizedGate


def mps_overlap(left: MPS, right: MPS) -> np.complex128:
    """Contract an overlap with a two-index environment, avoiding chi**4 transfers.

    Returns:
        The complex inner product without copies of complete states.

    """
    return overlap(left, right)


def _snapshot(state: MPS) -> MPS:
    """Share untouched arrays in a private, replace-only local-operator tape.

    All backend gate kernels replace affected tensors rather than writing into
    existing arrays. Public input states are still copied in ``_state``. Never
    apply an in-place tensor operation to these private snapshots.

    Returns:
        A new MPS container and tensor list sharing their immutable arrays.
    """
    result = copy.copy(state)
    result.tensors = list(state.tensors)
    return result


@dataclass
class MPSDiagnostics:
    """Cumulative measured work; tensor snapshots exclude contraction temporaries."""

    operator_applications: int = 0
    reused_local_applications: int = 0
    splits: int = 0
    discarded_weight_sum: float = 0.0
    discarded_weight_max: float = 0.0
    cap_reductions: int = 0
    maximum_bonds: dict[str, int] = field(default_factory=dict)
    maximum_retained_tensor_bytes: int = 0
    maximum_forward_norm_error: float = 0.0
    overlap_site_contractions: int = 0
    overlap_environment_reuses: int = 0
    maximum_overlap_cache_bytes: int = 0

    def split(self, event: TruncationEvent) -> None:
        """Accumulate actual SVD tails without keeping an unbounded event history."""
        self.splits += 1
        self.discarded_weight_sum += event.discarded_weight
        self.discarded_weight_max = max(self.discarded_weight_max, event.discarded_weight)
        self.cap_reductions += int(event.capped)

    def observe(self, state: MPS, role: str) -> None:
        """Observe bond axes only, excluding the physical dimension."""
        bond = max((t.shape[2] for t in state.tensors[:-1]), default=1)
        self.maximum_bonds[role] = max(self.maximum_bonds.get(role, 1), bond)

    def storage(self, states: Iterable[MPS]) -> None:
        """Deduplicate NumPy backing allocations of simultaneously retained states."""
        owners = {}
        for state in states:
            for tensor in state.tensors:
                owner = tensor
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                owners[id(owner)] = owner.nbytes
        self.maximum_retained_tensor_bytes = max(self.maximum_retained_tensor_bytes, sum(owners.values()))

    def record(self) -> dict:
        """Return scalar diagnostics with explicit limitations."""
        return {
            **asdict(self),
            "tensor_memory_scope": "observed simultaneous forward/adjoint/current/target tensor backing allocations; "
            "excludes contraction temporaries, Python objects and allocator overhead",
            "discarded_weight_scope": "sum of local SVD tails, not a global fidelity bound",
        }


class MPSBackend(TrajectorySampler):
    """MPS implementation of the dense backend's fixed-map optimizer interface."""

    def __init__(
        self,
        circuit: ParameterizedCircuit,
        target: MPS | NDArray[np.complex128],
        initial_state: MPS | NDArray[np.complex128] | None = None,
        *,
        truncation: KrotovTruncation | None = None,
        compress_target: bool = False,
    ) -> None:
        """Cache private target/input tensors and explicit truncation settings.

        Target compression is optional and its error is recorded separately.
        The default exact conversion keeps all SVD values. No trajectory is
        silently renormalized after a truncating gate application.

        Raises:
            ValueError: If a state or truncation setting is invalid.
        """
        super().__init__(circuit)
        self.truncation = copy.deepcopy(truncation or KrotovTruncation())
        t = self.truncation
        if (
            not np.isfinite(t.svd_threshold)
            or t.svd_threshold < 0
            or type(t.min_bond_dim) is not int
            or t.min_bond_dim < 1
            or (t.max_bond_dim is not None and (type(t.max_bond_dim) is not int or t.max_bond_dim < t.min_bond_dim))
            or t.trunc_mode not in {"discarded_weight", "relative"}
        ):
            msg = "Invalid MPS truncation settings."
            raise ValueError(msg)
        self.diagnostics = MPSDiagnostics()
        self.target = self._state(target)
        self.initial = MPS(circuit.num_qubits) if initial_state is None else self._state(initial_state)
        self.target_compression_infidelity = 0.0
        if compress_target:
            original = copy.deepcopy(self.target)
            with observe_truncations(self.diagnostics.split):
                self.target.compress(
                    t.svd_threshold, max_bond_dim=t.max_bond_dim, min_bond_dim=t.min_bond_dim, trunc_mode=t.trunc_mode
                )
            norm = float(mps_overlap(self.target, self.target).real)
            if norm <= 0 or not np.isfinite(norm):
                msg = "Target compression produced a zero or invalid state."
                raise ValueError(msg)
            self.target.tensors[0] /= np.sqrt(norm)
            self.target_compression_infidelity = max(0.0, 1 - float(abs(mps_overlap(original, self.target)) ** 2))
        self.diagnostics.observe(self.target, "target")
        self.diagnostics.observe(self.initial, "initial")
        self._live: list[MPS] = []
        self._overlaps = MixedEnvironments()

    @property
    def retained_state_count(self) -> int:
        """Number of explicitly retained tape states; zero between public calls."""
        return len(self._live)

    @contextmanager
    def _tape(self) -> Iterator[None]:
        """Release all saved states on successful and exceptional exits.

        Yields:
            None while a single optimizer operation owns its tape.
        """
        try:
            yield
        finally:
            self._live.clear()
            self.diagnostics.overlap_site_contractions += self._overlaps.site_contractions
            self.diagnostics.overlap_environment_reuses += self._overlaps.environment_reuses
            self.diagnostics.maximum_overlap_cache_bytes = max(
                self.diagnostics.maximum_overlap_cache_bytes, self._overlaps.peak_cache_bytes
            )
            self._overlaps.clear()
            self._overlaps.site_contractions = self._overlaps.environment_reuses = 0
            self._overlaps.peak_cache_bytes = 0

    def _state(self, value: MPS | NDArray[np.complex128]) -> MPS:
        """Validate normalization before copying or factoring a supplied state.

        Returns:
            A private MPS in little-endian site order.

        Raises:
            ValueError: If the state has invalid dimensions, norm or entries.
        """
        if isinstance(value, MPS):
            if value.length != self.circuit.num_qubits or any(
                t.shape[0] != 2 or not np.all(np.isfinite(t)) for t in value.tensors
            ):
                msg = "Expected a finite qubit MPS matching the circuit."
                raise ValueError(msg)
            norm = mps_overlap(value, value)
            result = copy.deepcopy(value)
        else:
            vector = np.asarray(value, dtype=np.complex128)
            if vector.shape != (2**self.circuit.num_qubits,) or not np.all(np.isfinite(vector)):
                msg = "Expected a finite statevector matching the circuit."
                raise ValueError(msg)
            norm = np.vdot(vector, vector)
            if abs(norm - 1) > 1e-10:
                msg = "Expected a normalized state."
                raise ValueError(msg)
            result = MPS.from_statevector(vector)
        if not np.isfinite(norm) or abs(norm - 1) > 1e-10:
            msg = "Expected a normalized state."
            raise ValueError(msg)
        return result

    def _apply(self, state: MPS, matrix: NDArray, sites: tuple[int, ...], role: str) -> None:
        """Apply an existing local MPS kernel while observing actual SVD decisions."""
        if len(sites) == 2 and sites[0] > sites[1]:
            sites = sites[::-1]
            matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        t = self.truncation
        with observe_truncations(self.diagnostics.split):
            state.apply_local_operator(
                matrix,
                sites,
                threshold=t.svd_threshold,
                max_bond_dim=t.max_bond_dim,
                min_bond_dim=t.min_bond_dim,
                trunc_mode=t.trunc_mode,
            )
        self.diagnostics.operator_applications += 1
        self.diagnostics.observe(state, role)

    def _storage(self, *extra: MPS) -> None:
        """Observe simultaneously retained tensors without retaining new references."""
        self.diagnostics.storage([self.initial, self.target, *self._live, *extra])

    def _apply_all(self, states: list[MPS], matrix: NDArray, sites: tuple[int, ...], role: str) -> None:
        """Evaluate identical local inputs once, retaining all trajectory weights.

        Only object-identical immutable input tensors are shared, never merely
        close states. The existing kernel depends only on tensors in this span,
        including intermediate sites of a long-range swap chain. References to
        the inputs are kept until the loop ends to prevent identity recycling.
        """
        first, stop = min(sites), max(sites) + 1
        evaluated = {}
        for state in states:
            inputs = tuple(state.tensors[first:stop])
            key = tuple(id(tensor) for tensor in inputs)
            previous = evaluated.get(key)
            if previous is None:
                self._apply(state, matrix, sites, role)
                evaluated[key] = inputs, tuple(state.tensors[first:stop])
            else:
                state.tensors[first:stop] = previous[1]
                self.diagnostics.reused_local_applications += 1
                self.diagnostics.observe(state, role)

    def _noise(self, state: MPS, row: NoiseMaps, path: int, gate: int, *, adjoint: bool = False) -> None:
        """Replay a unitary history in forward or reverse operator order."""
        operators = row[path][gate].operators
        for matrix, sites in reversed(operators) if adjoint else operators:
            self._apply(state, matrix.conj().T if adjoint else matrix, sites, "adjoint" if adjoint else "forward")

    def _forward(self, theta: NDArray, maps: NoiseMaps, *, retain: bool) -> tuple[list[list[MPS]], list[MPS]]:
        """Propagate paths, optionally retaining pre-noise outputs for adjoints.

        Returns:
            Gate-major outputs and final states.
        """
        states = [_snapshot(self.initial) for _ in maps]
        outputs: list[list[MPS]] = []
        for k, gate in enumerate(self.circuit.gates):
            matrix, sites = self.circuit.gate_matrix(gate, theta)
            self._apply_all(states, matrix, sites, "forward")
            if retain:
                output = [_snapshot(state) for state in states]
                outputs.append(output)
                self._live.extend(output)
            for r, state in enumerate(states):
                self._noise(state, maps, r, k)
        for state in states:
            self.diagnostics.maximum_forward_norm_error = max(
                self.diagnostics.maximum_forward_norm_error, abs(float(mps_overlap(state, state).real) - 1)
            )
        self._storage(*states)
        return outputs, states

    def _costates(
        self, theta: NDArray, maps: NoiseMaps, final: list[MPS], estimator: Estimator, weights: NDArray
    ) -> list[list[MPS]]:
        """Construct stale adjoints, placing independent weights exactly once.

        Returns:
            Gate-major adjoints after pulling back each gate's noise.
        """
        states = [_snapshot(self.target) for _ in maps]
        if estimator == "independent":
            for state, output, weight in zip(states, final, weights, strict=True):
                state.tensors[0] = np.multiply(state.tensors[0], -weight * mps_overlap(self.target, output))
        backwards = []
        for k in reversed(range(len(self.circuit.gates))):
            for r, state in enumerate(states):
                self._noise(state, maps, r, k, adjoint=True)
            frozen = [_snapshot(state) for state in states]
            backwards.append(frozen)
            self._live.extend(frozen)
            matrix, sites = self.circuit.gate_matrix(self.circuit.gates[k], theta)
            self._apply_all(states, matrix.conj().T, sites, "adjoint")
        self._storage(*states, *final)
        return list(reversed(backwards))

    def _signal(
        self, gate: ParameterizedGate, outputs: list[MPS], costates: list[MPS], estimator: Estimator, weights: NDArray
    ) -> float:
        """Evaluate the unchanged estimator, weighting BOTH cross indices.

        Returns:
            One scalar gate signal in the original parameter coordinates.
        """
        matrix, sites = self.circuit.derivative_operator(gate)
        tangents = [_snapshot(output) for output in outputs]
        self._apply_all(tangents, matrix, sites, "derivative")
        plain, derivative = self._overlaps.local_overlaps(
            costates, outputs, tangents, (min(sites), max(sites)), diagonal=estimator == "independent"
        )
        if estimator == "independent":
            signal = 2 * float(derivative.sum().real)
        else:
            signal = -2 * float((weights[:, None] * weights[None, :] * derivative * plain.conj()).sum().real)
        return gate.angle_scale * signal

    def _inputs(
        self, theta: NDArray, maps: NoiseMaps | None, weights: NDArray | None, estimator: Estimator = "independent"
    ) -> tuple:
        """Validate all inputs before propagation.

        Returns:
            Private parameters, replay maps and normalized probability weights.

        Raises:
            ValueError: If the estimator is unknown.
        """
        if estimator not in {"independent", "cross"}:
            msg = "Unknown trajectory estimator."
            raise ValueError(msg)
        theta, maps = self._parameters(theta), self._maps(maps)
        checked = self._weights(weights, len(maps))
        return theta, maps, np.full(len(maps), 1 / len(maps)) if checked is None else checked

    def metrics(
        self, theta: NDArray, maps: NoiseMaps | None = None, *, weights: NDArray | None = None
    ) -> tuple[float, float]:
        """Evaluate paths sequentially without retaining gradient tapes.

        Returns:
            Weighted infidelity and fidelity.
        """
        theta, maps, mass = self._inputs(theta, maps, weights)
        fidelity = 0.0
        for row, weight in zip(maps, mass, strict=True):
            _, final = self._forward(theta, [row], retain=False)
            fidelity += weight * float(abs(mps_overlap(self.target, final[0])) ** 2)
        return 1 - fidelity, float(fidelity)

    def gradient(
        self,
        theta: NDArray,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray | None = None,
    ) -> tuple[NDArray, float, float]:
        """Evaluate a weighted adjoint signal without changing caller-owned inputs.

        Returns:
            Signal, infidelity and fidelity at the supplied parameters.
        """
        theta, maps, mass = self._inputs(theta, maps, weights, estimator)
        with self._tape():
            outputs, final = self._forward(theta, maps, retain=True)
            costates = self._costates(theta, maps, final, estimator, mass)
            signal = np.zeros_like(theta)
            for gate, output, chi in zip(self.circuit.gates, outputs, costates, strict=True):
                if gate.param_index is not None:
                    signal[gate.param_index] += self._signal(gate, output, chi, estimator, mass)
            fidelity = float(sum(w * abs(mps_overlap(self.target, s)) ** 2 for w, s in zip(mass, final, strict=True)))
            return signal, 1 - fidelity, fidelity

    def online_update(
        self,
        theta: NDArray,
        step_size: float,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Update sequential gates against stale adjoints with joint weighted signals.

        Returns:
            New parameters and accumulated signals, including shared parameters.

        Raises:
            ValueError: If the step size is invalid.
        """
        if isinstance(step_size, bool) or not np.isfinite(step_size) or step_size < 0:
            msg = "Step size must be finite and nonnegative."
            raise ValueError(msg)
        theta, maps, mass = self._inputs(theta, maps, weights, estimator)
        with self._tape():
            # Cross costates do not depend on final forward overlaps. Independent
            # costates do, and retain the original preliminary propagation.
            final = self._forward(theta, maps, retain=False)[1] if estimator == "independent" else []
            costates = self._costates(theta, maps, final, estimator, mass)
            states = [_snapshot(self.initial) for _ in maps]
            signal = np.zeros_like(theta)
            for k, gate in enumerate(self.circuit.gates):
                matrix, sites = self.circuit.gate_matrix(gate, theta)
                outputs = [_snapshot(s) for s in states]
                self._apply_all(outputs, matrix, sites, "forward")
                if gate.param_index is not None:
                    local = self._signal(gate, outputs, costates[k], estimator, mass)
                    signal[gate.param_index] += local
                    theta[gate.param_index] -= step_size * local
                    matrix, sites = self.circuit.gate_matrix(gate, theta)
                    outputs = [_snapshot(s) for s in states]
                    self._apply_all(outputs, matrix, sites, "forward")
                for r, output in enumerate(outputs):
                    self._noise(output, maps, r, k)
                states = outputs
            for state in states:
                self.diagnostics.maximum_forward_norm_error = max(
                    self.diagnostics.maximum_forward_norm_error, abs(float(mps_overlap(state, state).real) - 1)
                )
            self._storage(*states)
            return theta, signal

    def deletion_fidelities(self, theta: NDArray) -> tuple[NDArray, float]:
        """Score all complete noiseless gate removals using MPS prefix/suffix overlaps.

        Returns:
            Fidelity after each individual deletion and the original fidelity.
            Exact without truncation; with truncation these are approximate
            contractions, not exact derivatives or exhaustive truncated replays.
        """
        theta = self._parameters(theta)
        state = _snapshot(self.initial)
        with self._tape():
            for gate in self.circuit.gates:
                self._live.append(_snapshot(state))
                matrix, sites = self.circuit.gate_matrix(gate, theta)
                self._apply(state, matrix, sites, "forward")
            fidelity = float(abs(mps_overlap(self.target, state)) ** 2)
            suffix = _snapshot(self.target)
            removed = np.empty(len(self.circuit.gates))
            for k in reversed(range(len(self.circuit.gates))):
                removed[k] = abs(mps_overlap(suffix, self._live[k])) ** 2
                matrix, sites = self.circuit.gate_matrix(self.circuit.gates[k], theta)
                self._apply(suffix, matrix.conj().T, sites, "adjoint")
            self._storage(state, suffix)
            return removed, fidelity
