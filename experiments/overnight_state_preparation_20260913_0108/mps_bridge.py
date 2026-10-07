# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Weighted fixed-map MPS gradient bridge with passive truncation accounting.

This is an overlap/scaling support component, not authorization for a larger
scientific run. It replays the exact same state-independent unitary maps as the
dense backend and computes a weighted sum of public single-trajectory adjoints.
"""

from __future__ import annotations

import copy
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np

from mqt.yaqs.core import linalg
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTJMOptions, KrotovTruncation, noisy_state_preparation_contribution

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.statevector import NoiseMaps

VERSION = "overnight.weighted_mps_bridge.v1"


def bonds(state: MPS) -> list[int]:
    """Return actual internal bond dimensions, excluding the physical qubit axis."""
    return [int(tensor.shape[2]) for tensor in state.tensors[:-1]]


@dataclass
class BondAudit:
    """Passively observe actual operator and SVD truncation work in one process."""

    operator_applications: int = 0
    maximum_internal_bond: int = 1
    maximum_state_tensor_bytes: int = 0
    truncation_calls: int = 0
    discarded_squared_singular_values: float = 0.0
    maximum_discarded_squared_singular_values: float = 0.0
    hard_cap_reductions: int = 0
    splits: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, state: MPS) -> None:
        """Record actual tensor storage and bonds without converting to dense form."""
        self.maximum_internal_bond = max(self.maximum_internal_bond, max(bonds(state), default=1))
        self.maximum_state_tensor_bytes = max(
            self.maximum_state_tensor_bytes, sum(tensor.nbytes for tensor in state.tensors)
        )

    def instrument(self, stack: ExitStack) -> None:
        """Install reversible passive wrappers without changing a numerical choice."""
        raw_apply, raw_truncate = MPS.apply_local_operator, linalg.truncate

        def apply(
            state: MPS,
            matrix: NDArray,
            sites: int | list[int] | tuple[int, ...],
            *,
            threshold: float = 0.0,
            max_bond_dim: int | None = None,
            min_bond_dim: int = 1,
            trunc_mode: Literal["discarded_weight", "relative"] = "discarded_weight",
            allow_long_range: bool = True,
        ) -> None:
            self.observe(state)
            raw_apply(
                state,
                matrix,
                sites,
                threshold=threshold,
                max_bond_dim=max_bond_dim,
                min_bond_dim=min_bond_dim,
                trunc_mode=trunc_mode,
                allow_long_range=allow_long_range,
            )
            self.operator_applications += 1
            self.observe(state)

        def truncate(
            values: NDArray,
            *,
            mode: Literal["discarded_weight", "relative", "hard_cutoff"],
            threshold: float,
            max_bond_dim: int | None = None,
            min_keep: int = 1,
        ) -> int:
            keep = raw_truncate(values, mode=mode, threshold=threshold, max_bond_dim=max_bond_dim, min_keep=min_keep)
            discarded = float(np.sum(np.abs(values[keep:]) ** 2))
            cap = max_bond_dim
            self.truncation_calls += 1
            self.discarded_squared_singular_values += discarded
            self.maximum_discarded_squared_singular_values = max(
                self.maximum_discarded_squared_singular_values, discarded
            )
            self.hard_cap_reductions += int(cap is not None and len(values) > cap and keep == cap)
            self.splits.append({
                "available": len(values),
                "kept": keep,
                "discarded_squared_singular_values": discarded,
                "mode": mode,
                "threshold": threshold,
                "max_bond_dim": cap,
            })
            return keep

        stack.enter_context(patch.object(MPS, "apply_local_operator", apply))
        stack.enter_context(patch.object(linalg, "truncate", truncate))


def validate_maps(circuit: ParameterizedCircuit, maps: NoiseMaps, weights: NDArray) -> None:
    """Require a normalized nonnegative mixture of fixed unitary physical maps.

    Raises:
        ValueError: If weights, arity or map normalization would change the task.
    """
    if (
        not maps
        or weights.shape != (len(maps),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        msg = "MPS bridge requires finite nonnegative normalized path weights."
        raise ValueError(msg)
    for row in maps:
        if len(row) != len(circuit.gates):
            msg = "MPS bridge maps must cover every physical circuit gate."
            raise ValueError(msg)
        for noise in row:
            if noise.normalized or noise.normalization_checkpoints:
                msg = "State-dependent normalized maps are outside this unitary MPS bridge."
                raise ValueError(msg)
            for operator, sites in noise.operators:
                if (
                    len(sites) not in {1, 2}
                    or len(set(sites)) != len(sites)
                    or any(site < 0 or site >= circuit.num_qubits for site in sites)
                    or operator.shape != (2 ** len(sites),) * 2
                    or not np.allclose(operator.conj().T @ operator, np.eye(operator.shape[0]), rtol=0, atol=1e-12)
                ):
                    msg = "MPS bridge requires supported finite unitary local replay operators."
                    raise ValueError(msg)


@dataclass
class GradientResult:
    """Retain the weighted objective, per-path scores and measured tensor work."""

    gradient: NDArray[np.float64]
    loss: float
    fidelity: float
    trajectory_fidelities: list[float]
    audit: dict[str, Any]
    cpu_seconds: float
    wall_seconds: float


def gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS | NDArray[np.complex128],
    maps: NoiseMaps,
    weights: NDArray[np.float64],
    *,
    truncation: KrotovTruncation | None = None,
) -> GradientResult:
    """Return the weighted public MPS adjoint and actual truncation measurements.

    The physical input is always the all-zero MPS. Dense target factorization,
    when requested, is inside the cost interval. Instrumentation measures each
    split's discarded squared singular values; their sum is diagnostic and is
    not asserted to be a global fidelity-error bound.
    """
    validate_maps(circuit, maps, np.asarray(weights))
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (circuit.num_params,) or not np.all(np.isfinite(theta)):
        msg = "MPS bridge parameters must be a finite vector of the circuit dimension."
        raise ValueError(msg)
    truncation = KrotovTruncation() if truncation is None else truncation
    cpu, wall = time.process_time(), time.perf_counter()
    audit = BondAudit()
    signal = np.zeros_like(theta)
    fidelities = []
    forward_peak, forward_norm_error = 1, 0.0
    with ExitStack() as stack:
        audit.instrument(stack)
        target_mps = copy.deepcopy(target) if isinstance(target, MPS) else MPS.from_statevector(target)
        audit.observe(target_mps)
        target_bonds = bonds(target_mps)
        for row, weight in zip(maps, weights, strict=True):
            contribution, _, fidelity, trajectories = noisy_state_preparation_contribution(
                circuit,
                theta,
                target_mps,
                None,
                KrotovTJMOptions(num_trajectories=1),
                MPS(circuit.num_qubits, state="zeros"),
                truncation,
                fixed_noise_maps=[list(row)],
            )
            signal += weight * contribution
            fidelities.append(fidelity)
            for state in trajectories[0].states:
                forward_peak = max(forward_peak, max(bonds(state), default=1))
            final = trajectories[0].states[-1]
            forward_norm_error = max(forward_norm_error, abs(float(final.scalar_product(final).real) - 1))
    fidelity = float(np.asarray(weights) @ np.asarray(fidelities))
    record = {
        **asdict(audit),
        "target_bonds": target_bonds,
        "forward_peak_internal_bond": forward_peak,
        "forward_final_max_norm_error": forward_norm_error,
        "all_work_peak_scope": "all instrumented forward, adjoint, derivative and target MPS work",
        "truncation": asdict(truncation),
        "trajectory_paths": len(maps),
        "directional_paths": 2 * len(maps),
        "memory_scope": "maximum individual MPS tensor storage; excludes complete tapes and process overhead",
        "gradient_semantics": "fixed-map adjoint; truncated mode does not differentiate SVD truncation decisions",
    }
    return GradientResult(
        signal, 1.0 - fidelity, fidelity, fidelities, record, time.process_time() - cpu, time.perf_counter() - wall
    )
