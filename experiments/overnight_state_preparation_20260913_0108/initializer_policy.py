# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic target-overlap policy with full physical synthesis and costs."""

from __future__ import annotations

import copy
import time
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import compilation, runner, synthesis

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "overnight.conditional_rank_two_initializer.v1"
OVERLAP_THRESHOLD = 0.99


def use_rank_two(overlap: float) -> bool:
    """Return the single fixed overlap decision without target labels or tolerances."""
    if not np.isfinite(overlap) or not 0 <= overlap <= 1 + 1e-10:
        msg = "Initializer overlap must be a finite physical fidelity."
        raise ValueError(msg)
    return overlap >= OVERLAP_THRESHOLD


def state_vector(record: dict[str, Any], theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return an independent complete scalar physical preparation from the zero state."""
    circuit = pilot.restore_circuit(record)
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        getattr(reference, gate.name)(circuit.angle(gate, theta, None), *gate.sites)
    return Statevector.from_instruction(reference).data


def compile_source(source: dict[str, Any]) -> dict[str, Any]:
    """Return the unchanged compiler result with an independent full-state audit."""
    cpu = time.process_time()
    if any(g.get("data_map") is not None or g.get("fixed_params") for g in source["circuit"]["gates"]):
        msg = "Initializer compilation requires scalar gates without data maps."
        raise ValueError(msg)
    circuit, theta = pilot.restore_circuit(source["circuit"]), np.asarray(source["theta"], dtype=float)
    for gate, original in zip(circuit.gates, source["circuit"]["gates"], strict=True):
        gate.native_gate_id = original.get("native_gate_id")
    result = compilation.compile_endpoint(circuit, theta)
    before = state_vector(source["circuit"], theta)
    after = state_vector(result["circuit"], np.asarray(result["theta"]))
    error = float(np.max(abs(before - np.exp(1j * result["omitted_global_phase"]) * after)))
    if not np.isfinite(error) or error > max(1e-10, result["ideal_operator_error_bound"]):
        msg = "Initializer compilation changed its ideal physical preparation."
        raise FloatingPointError(msg)
    result["state_audit_error"] = error
    result["compiler_and_audit_cpu_seconds"] = time.process_time() - cpu
    return result


def build_initializers(target: NDArray[np.complex128], legacy: dict[str, Any]) -> dict[str, Any]:
    """Return both compiled initial sources and the target-only rank-two decision.

    The supplied legacy construction cost must be measured and complete. This
    kernel has no target identifiers, family labels, noisy scores or optimizers.
    """
    legacy_cpu = legacy.get("construction_cpu_seconds")
    if legacy_cpu is None or not np.isfinite(legacy_cpu) or legacy_cpu < 0 or not legacy.get("timing_complete"):
        msg = "Conditional initialization requires complete measured legacy construction CPU."
        raise ValueError(msg)
    target = np.asarray(target, dtype=np.complex128)
    if target.shape != (2 ** legacy["circuit"]["num_qubits"],):
        msg = "Legacy initialization and supplied target dimensions differ."
        raise ValueError(msg)
    cpu, wall = time.process_time(), time.perf_counter()
    rank_two = synthesis.synthesize(target)
    overlap_cpu = time.process_time()
    approximation = np.asarray(list(starmap(complex, rank_two["approximation"])))
    overlap = float(abs(np.vdot(target, approximation)) ** 2)
    physical = state_vector(rank_two["circuit"], np.asarray(rank_two["theta"]))
    lowering_overlap = float(abs(np.vdot(approximation, physical)) ** 2)
    if (
        abs(overlap - rank_two["compression"]["approximation_fidelity"]) > 1e-10
        or not np.isfinite(lowering_overlap)
        or abs(lowering_overlap - 1) > 1e-10
    ):
        msg = "Rank-two ideal overlap or physical lowering audit disagrees."
        raise FloatingPointError(msg)
    chosen = use_rank_two(overlap)
    overlap_cpu = time.process_time() - overlap_cpu
    probe = compile_source(rank_two)
    original = compile_source(legacy)
    probe_cpu = rank_two["construction_cpu_seconds"] + overlap_cpu + probe["compiler_and_audit_cpu_seconds"]
    legacy_total = legacy_cpu + original["compiler_and_audit_cpu_seconds"]

    def endpoint(compiled: dict[str, Any], total: float, name: str) -> dict[str, Any]:
        return {
            "circuit": copy.deepcopy(compiled["circuit"]),
            "theta": list(compiled["theta"]),
            "selected_source": name,
            "initializer_cpu_seconds": total,
            "physical_counts": runner.counts(pilot.restore_circuit(compiled["circuit"])),
            "compilation_checksum": pilot.digest(compiled),
        }

    return {
        "version": VERSION,
        "threshold": OVERLAP_THRESHOLD,
        "ideal_tt_overlap": overlap,
        "rank_two_selected": chosen,
        "rank_two_physical_overlap": lowering_overlap,
        "legacy": endpoint(original, legacy_total, "legacy"),
        "conditional": endpoint(
            probe if chosen else original,
            probe_cpu + (0.0 if chosen else legacy_total),
            "rank_two" if chosen else "legacy",
        ),
        "raw_legacy_counts": runner.counts(pilot.restore_circuit(legacy["circuit"])),
        "raw_rank_two_counts": runner.counts(pilot.restore_circuit(rank_two["circuit"])),
        "legacy_source_checksum": pilot.digest(legacy),
        "rank_two_synthesis": rank_two,
        "rank_two_compilation": probe,
        "legacy_compilation": original,
        "costs": {
            "legacy_construction_cpu_seconds": legacy_cpu,
            "rank_two_synthesis_cpu_seconds": rank_two["construction_cpu_seconds"],
            "overlap_and_lowering_audit_cpu_seconds": overlap_cpu,
            "rank_two_compilation_and_audit_cpu_seconds": probe["compiler_and_audit_cpu_seconds"],
            "legacy_compilation_and_audit_cpu_seconds": original["compiler_and_audit_cpu_seconds"],
            "rank_two_probe_cpu_seconds": probe_cpu,
            "discarded_probe_cpu_seconds": 0.0 if chosen else probe_cpu,
        },
        "physical_build_cpu_seconds": time.process_time() - cpu,
        "physical_build_wall_seconds": time.perf_counter() - wall,
    }
