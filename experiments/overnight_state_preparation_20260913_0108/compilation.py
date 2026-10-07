# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic endpoint-only physical simplification of Pauli rotations."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.state_preparation_pilot import circuit_record, digest
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "overnight.physical_endpoint_compilation.v1"
IDENTITY_ATOL = 1e-12
PRIMITIVES = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})


def reduced_angle(angle: float) -> tuple[float, int, float]:
    """Return a principal angle, discarded periods and argument-reduction error.

    Raises:
        ValueError: If nonfinite input or argument reduction exceeds tolerance.
    """
    if not np.isfinite(angle):
        msg = "Endpoint angles must be finite."
        raise ValueError(msg)
    reduced = math.remainder(angle, math.tau)
    turns = round((angle - reduced) / math.tau)
    sign = -1 if turns % 2 else 1
    error = abs(math.cos(angle / 2) - sign * math.cos(reduced / 2)) + abs(
        math.sin(angle / 2) - sign * math.sin(reduced / 2)
    )
    if error > IDENTITY_ATOL:
        msg = "Endpoint angle is too large for reliable modulo reduction."
        raise ValueError(msg)
    return reduced, turns, error


def compile_endpoint(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> dict[str, Any]:
    """Return a physical compact circuit equivalent at the supplied endpoint only.

    Equal generators fuse across disjoint gates only, without crossing a
    retained noise-flag boundary. Every source gate maps to one output gate or
    a recorded identity deletion. Shared and fixed inputs become independent
    output scalar parameters; no original parameter-family identity is claimed.

    Raises:
        ValueError: If a source gate, angle map or parameter vector is unsupported.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    values = np.asarray(theta)
    if (
        type(circuit.num_qubits) is not int
        or circuit.num_qubits < 1
        or values.shape != (circuit.num_params,)
        or not np.isrealobj(values)
        or not np.all(np.isfinite(values))
    ):
        msg = "Expected finite real endpoint parameters matching a physical circuit."
        raise ValueError(msg)
    source_angles = []
    for gate in circuit.gates:
        arity = 1 if gate.name in {"rx", "ry", "rz"} else 2
        if (
            gate.name not in PRIMITIVES
            or gate.data_map is not None
            or gate.fixed_params
            or len(gate.sites) != arity
            or len(set(gate.sites)) != arity
            or any(type(site) is not int or not 0 <= site < circuit.num_qubits for site in gate.sites)
            or type(gate.noise_enabled) is not bool
            or not np.isfinite(gate.angle_scale)
            or not np.isfinite(gate.angle_offset)
            or (
                gate.param_index is not None
                and (type(gate.param_index) is not int or not 0 <= gate.param_index < circuit.num_params)
            )
        ):
            msg = "Expected supported Pauli rotations with valid fixed/affine endpoint angles and no data maps."
            raise ValueError(msg)
        source_angles.append(float(circuit.angle(gate, values, None)))
    nodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    periods = 0
    error_bound = 0.0

    def normalize(node: dict[str, Any]) -> bool:
        nonlocal periods, error_bound
        angle, turns, error = reduced_angle(node["angle"])
        periods += turns
        error_bound += error
        node["angle"] = angle
        if turns:
            events.append({
                "kind": "period_reduction",
                "sources": list(node["sources"]),
                "turns": turns,
                "reduced_angle": angle,
            })
        if abs(angle) <= IDENTITY_ATOL:
            error_bound += abs(angle) / 2
            events.append({
                "kind": "remove_identity",
                "sources": list(node["sources"]),
                "residual_angle": angle,
                "noise_enabled": node["noise_enabled"],
            })
            return False
        return True

    for index, (gate, angle) in enumerate(zip(circuit.gates, source_angles, strict=True)):
        incoming = {
            "name": gate.name,
            "sites": gate.sites,
            "angle": angle,
            "noise_enabled": gate.noise_enabled,
            "sources": [index],
        }
        if not normalize(incoming):
            continue
        support = set(gate.sites)
        crossed = []
        merged = False
        for position in range(len(nodes) - 1, -1, -1):
            prior = nodes[position]
            if prior["noise_enabled"] != incoming["noise_enabled"]:
                break
            if support.isdisjoint(prior["sites"]):
                crossed.extend(prior["sources"])
                continue
            if prior["name"] == incoming["name"] and prior["sites"] == incoming["sites"]:
                events.append({
                    "kind": "fuse",
                    "left_sources": list(prior["sources"]),
                    "right_sources": [index],
                    "commuted_past_sources": list(reversed(crossed)),
                })
                prior["angle"] = math.fsum((prior["angle"], incoming["angle"]))
                prior["sources"].append(index)
                if not normalize(prior):
                    nodes.pop(position)
                merged = True
            break
        if not merged:
            nodes.append(incoming)
    output = []
    output_angles = []
    source_to_output: list[int | None] = [None] * len(circuit.gates)
    mapping = []
    for index, node in enumerate(nodes):
        output.append(
            ParameterizedGate(
                node["name"],
                node["sites"],
                index,
                logical_gate_id=index,
                native_gate_id=index,
                noise_enabled=node["noise_enabled"],
            )
        )
        output_angles.append(node["angle"])
        for source in node["sources"]:
            source_to_output[source] = index
        mapping.append({
            "output_index": index,
            "source_gate_indices": node["sources"],
            "source_parameter_indices": [circuit.gates[i].param_index for i in node["sources"]],
            "source_logical_gate_ids": [circuit.gates[i].logical_gate_id for i in node["sources"]],
            "source_native_gate_ids": [circuit.gates[i].native_gate_id for i in node["sources"]],
            "noise_enabled": node["noise_enabled"],
        })
    compiled = ParameterizedCircuit(circuit.num_qubits, output, num_params=len(output_angles))
    input_windows = sum(gate.noise_enabled for gate in circuit.gates)
    output_windows = sum(gate.noise_enabled for gate in output)
    return {
        "version": VERSION,
        "source_checksum": digest({"circuit": circuit_record(circuit), "theta": values.tolist()}),
        "circuit": circuit_record(compiled),
        "theta": output_angles,
        "source_resolved_angles": source_angles,
        "source_to_output": source_to_output,
        "output_mapping": mapping,
        "events": events,
        "identity_tolerance": IDENTITY_ATOL,
        "discarded_periods": periods,
        "omitted_global_phase": math.pi * (periods % 2),
        "ideal_operator_error_bound": error_bound + 32 * np.finfo(float).eps * max(1, len(circuit.gates)),
        "input_gates": len(circuit.gates),
        "output_gates": len(output),
        "input_noise_windows": input_windows,
        "output_noise_windows": output_windows,
        "removed_noise_windows": input_windows - output_windows,
        "reparameterization": "fixed_endpoint_only_independent_output_angles",
        "compilation_cpu_seconds": time.process_time() - cpu,
        "compilation_wall_seconds": time.perf_counter() - wall,
    }
