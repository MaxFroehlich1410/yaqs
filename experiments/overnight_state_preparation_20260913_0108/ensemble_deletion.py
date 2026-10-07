# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact finite gate-and-noise deletion for constant benchmark channel ensembles.

For layer C_j = E_j o U_j, rho_j = C_j(rho_{j-1}) and the suffix
observable B_j = C_{j+1}^* ... C_L^*(|target><target|), removing the whole
layer gives F_{-j} = Tr(B_j rho_{j-1}). Every fixed Pauli window is self-adjoint,
so its adjoint uses the same probabilities before the inverse unitary. This
adapts the existing local-channel shared deletion adjoint to an ensemble mean.
It is a finite counterfactual, not a small-angle or gradient pruning heuristic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# Reuse the validated local conjugation kernel, without a full-system unitary.
from benchmarks.state_preparation.exact_density import _conjugate  # noqa: PLC2701
from experiments.overnight_state_preparation_20260913_0108 import channels

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

MEAN2 = ("depolarizing_1s_all", "dephasing_1s_all")
FIXED_PROFILES = tuple(noise for noise in channels.NOISE_IDS if noise != "ballarin_coupled")


@dataclass(frozen=True)
class DeletionResult:
    """Return exact component/mean scores with complete ranking work counts."""

    mean_scores: NDArray[np.float64]
    mean_fidelity: float
    profile_scores: dict[str, NDArray[np.float64]]
    profile_fidelities: dict[str, float]
    individual_channel_calls: int
    density_direction_passes: int
    cpu_seconds: float
    wall_seconds: float


def _validate(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_ids: Sequence[str],
) -> None:
    """Enforce the exact, parameter-independent q1-q6 benchmark domain.

    Raises:
        ValueError: If data, profiles or gate supports are outside that domain.
    """
    if (
        type(circuit.num_qubits) is not int
        or not 1 <= circuit.num_qubits <= 6
        or theta.shape != (circuit.num_params,)
        or not np.isrealobj(theta)
        or not np.all(np.isfinite(theta))
        or target.shape != (2**circuit.num_qubits,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
        or not noise_ids
        or len(set(noise_ids)) != len(noise_ids)
        or any(noise not in FIXED_PROFILES for noise in noise_ids)
        or any(
            gate.data_map is not None
            or len(gate.sites) not in {1, 2}
            or not np.isrealobj(gate.angle_scale)
            or not np.isrealobj(gate.angle_offset)
            or not np.isrealobj(gate.fixed_params)
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
            or not np.all(np.isfinite(gate.fixed_params))
            for gate in circuit.gates
        )
    ):
        msg = (
            "Ensemble deletion requires q1-q6, finite scalar gates/theta, normalized target and unique fixed profiles."
        )
        raise ValueError(msg)


def _physical_fidelity(rho: NDArray[np.complex128], target: NDArray[np.complex128]) -> float:
    """Return a fidelity after strict density physicality checks.

    Raises:
        FloatingPointError: If the propagated state is unphysical or nonfinite.
    """
    fidelity = float(np.vdot(target, rho @ target).real)
    if (
        not np.all(np.isfinite(rho))
        or abs(np.trace(rho) - 1) > 1e-10
        or np.max(np.abs(rho - rho.conj().T)) > 1e-10
        or np.linalg.eigvalsh(rho).min() < -1e-10
        or not -1e-10 <= fidelity <= 1 + 1e-10
    ):
        msg = "Ensemble deletion forward density failed physicality checks."
        raise FloatingPointError(msg)
    return fidelity


def _component(
    circuit: ParameterizedCircuit,
    matrices: list[tuple[NDArray[np.complex128], tuple[int, ...]]],
    target: NDArray[np.complex128],
    noise_id: str,
) -> tuple[NDArray[np.float64], float]:
    """Return all finite deletions in one forward and one adjoint profile pass.

    Raises:
        FloatingPointError: If a finite-deletion overlap is nonphysical.
    """
    n = circuit.num_qubits
    rho = np.zeros((2**n, 2**n), dtype=complex)
    rho[0, 0] = 1
    prefixes: list[NDArray[np.complex128]] = []
    windows: list[tuple[float, ...] | None] = []
    for gate, (unitary, sites) in zip(circuit.gates, matrices, strict=True):
        probabilities = (
            tuple(channels.window(noise_id, len(sites))["probabilities"])
            if noise_id != "noiseless" and gate.noise_enabled
            else None
        )
        prefixes.append(rho)
        windows.append(probabilities)
        rho = _conjugate(rho, unitary, sites, n)
        if probabilities is not None:
            rho = channels.apply_channel(rho, sites, probabilities, n)
    fidelity = _physical_fidelity(rho, target)
    suffix = np.outer(target, target.conj())
    scores = np.empty(len(circuit.gates), dtype=float)
    for index in range(len(circuit.gates) - 1, -1, -1):
        overlap = np.vdot(suffix, prefixes[index])
        if not np.isfinite(overlap) or abs(overlap.imag) > 1e-10 or not -1e-10 <= overlap.real <= 1 + 1e-10:
            msg = "Ensemble finite-deletion overlap failed reality/fidelity checks."
            raise FloatingPointError(msg)
        scores[index] = overlap.real
        unitary, sites = matrices[index]
        probabilities = windows[index]
        if probabilities is not None:
            suffix = channels.apply_channel(suffix, sites, probabilities, n)
        suffix = _conjugate(suffix, unitary.conj().T, sites, n)
    scores.setflags(write=False)
    return scores, fidelity


def deletion_fidelities(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    noise_ids: Sequence[str] = MEAN2,
    *,
    progress: Callable[[str, str], None] | None = None,
) -> DeletionResult:
    """Rank exact removal of each gate and its complete noise window.

    Args:
        circuit: Supported physical circuit from the all-zero input.
        theta: Current unmodified parameters, including shared coordinates.
        target: Fixed normalized pure target in little-endian ordering.
        noise_ids: Unique fixed profiles, equally weighted; default is mean2.
        progress: Optional per-profile start/complete events for attempt accounting.

    Returns:
        Per-profile and mean deletion/original fidelities. A ranking uses one
        complete forward and adjoint pass per channel. Channels are processed
        sequentially, with peak O(L*4**n) tape storage plus O(K*L) scores. The
        shared local gate matrices are constructed only once. These are ranking
        calls, not loss-gradient calls, although both use two density directions.

    Raises:
        ValueError: If a requested channel is parameter dependent or invalid.
    """
    _validate(circuit, theta, target, noise_ids)
    cpu, wall = time.process_time(), time.perf_counter()
    matrices = [circuit.gate_matrix(gate, theta) for gate in circuit.gates]
    scores, fidelities = {}, {}
    for noise in noise_ids:
        if progress is not None:
            progress(noise, "start")
        scores[noise], fidelities[noise] = _component(circuit, matrices, target, noise)
        if progress is not None:
            progress(noise, "complete")
    mean = np.mean(list(scores.values()), axis=0)
    mean.setflags(write=False)
    return DeletionResult(
        mean_scores=mean,
        mean_fidelity=float(np.mean(list(fidelities.values()))),
        profile_scores=scores,
        profile_fidelities=fidelities,
        individual_channel_calls=len(noise_ids),
        density_direction_passes=2 * len(noise_ids),
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.perf_counter() - wall,
    )


def ranked_indices(
    result: DeletionResult,
    origins: Sequence[int],
    *,
    protected_origins: frozenset[int] = frozenset(range(18)),
    ranking_quantum: float = 1e-12,
) -> list[int]:
    """Return eligible local indices, ordered by mean score then original index.

    Origin IDs are the original physical gate positions and survive every prior
    deletion; they do not depend on mutable local positions or parameter indices.

    Raises:
        ValueError: If origin identity, scores or ranking resolution is invalid.
    """
    if (
        len(origins) != len(result.mean_scores)
        or len(set(origins)) != len(origins)
        or any(type(origin) is not int or origin < 0 for origin in origins)
        or not np.all(np.isfinite(result.mean_scores))
        or isinstance(ranking_quantum, bool)
        or not np.isfinite(ranking_quantum)
        or ranking_quantum <= 0
    ):
        msg = "Invalid original gate identities or finite-deletion ranking resolution."
        raise ValueError(msg)
    eligible = [index for index, origin in enumerate(origins) if origin not in protected_origins]
    return sorted(eligible, key=lambda index: (-round(result.mean_scores[index] / ranking_quantum), origins[index]))
