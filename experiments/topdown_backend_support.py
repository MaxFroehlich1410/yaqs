# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Streamed logical assessment, pruning B and sealed backend checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from experiments import legacy_pruning_ablation as pruning
from experiments import legacy_topdown_dense as legacy
from experiments import topdown_depth_search as depth
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovNoiseMap, ParameterizedCircuit, ParameterizedGate
from mqt.yaqs.optimization.mps_backend import mps_overlap

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from qiskit.circuit import CircuitInstruction

    from mqt.yaqs.optimization.backends import PreparationBackend


def digest(path: Path) -> str:
    """Hash a file used by a sealed checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON file only after finite serialization succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def save_checkpoint(folder: Path, record: dict, arrays: dict[str, NDArray]) -> dict:
    """Save each array once; commit the checksum seal last.

    Returns:
        The record containing binary array descriptors.

    Raises:
        ValueError: If an array name, dtype or value cannot be safely replayed.
    """
    for name, value in arrays.items():
        if not name.isidentifier() or value.dtype.kind not in "biufc" or not np.all(np.isfinite(value)):
            msg = "Checkpoint arrays require identifier names and finite numeric values."
            raise ValueError(msg)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "arrays.tmp"
    # Write NPY members directly: NumPy 1.x interprets savez's allow_pickle
    # keyword as another array. Fixed metadata also makes receipts reproducible.
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            with archive.open(member, "w") as stream:
                np.lib.format.write_array(stream, np.asarray(arrays[name], order="C"), allow_pickle=False)
    temporary.replace(folder / "arrays.npz")
    result = {
        **record,
        "arrays": {
            k: {"file": "arrays.npz", "key": k, "shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()
        },
    }
    write_json(folder / "record.json", result)
    write_json(folder / "complete.json", {name: digest(folder / name) for name in ("record.json", "arrays.npz")})
    return result


def load_checkpoint(folder: Path) -> tuple[dict, dict[str, NDArray]] | None:
    """Read only complete checkpoints and authenticate every referenced payload.

    Returns:
        Record and private arrays, or None for an uncommitted stage.

    Raises:
        ValueError: If a seal or binary descriptor is inconsistent.
    """
    seal = folder / "complete.json"
    if not seal.exists():
        return None
    hashes = json.loads(seal.read_text())
    if set(hashes) != {"record.json", "arrays.npz"} or any(
        not (folder / k).exists() or digest(folder / k) != v for k, v in hashes.items()
    ):
        msg = "Checkpoint checksum mismatch."
        raise ValueError(msg)
    record = json.loads((folder / "record.json").read_text())
    with np.load(folder / "arrays.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if set(arrays) != set(record["arrays"]):
        msg = "Checkpoint array names mismatch."
        raise ValueError(msg)
    for key, value in arrays.items():
        descriptor = record["arrays"][key]
        if descriptor != {"file": "arrays.npz", "key": key, "shape": list(value.shape), "dtype": value.dtype.str}:
            msg = "Checkpoint array descriptor mismatch."
            raise ValueError(msg)
    return record, arrays


def circuit_record(circuit: ParameterizedCircuit) -> dict:
    """Serialize scalar-gate circuits without embedding large matrices."""
    return {
        "num_qubits": circuit.num_qubits,
        "num_params": circuit.num_params,
        "gates": [{**asdict(g), "sites": list(g.sites)} for g in circuit.gates],
    }


def restore_circuit(record: dict) -> ParameterizedCircuit:
    """Restore the exact retained logical schedule and parameter coordinates."""
    gates = []
    for gate in record["gates"]:
        fields: dict[str, Any] = dict(gate)
        fields["sites"] = tuple(gate["sites"])
        gates.append(ParameterizedGate(**fields))
    return ParameterizedCircuit(record["num_qubits"], gates, record["num_params"])


def prune_b(engine: PreparationBackend, theta: NDArray) -> tuple[ParameterizedCircuit, NDArray, dict]:
    """Apply unchanged simultaneous pruning B with backend-specific contractions.

    Returns:
        Physically pruned circuit, surviving parameters and all signed scores.

    Raises:
        ValueError: If parameters are not independent scalar gates as required by B.
    """
    circuit = engine.circuit
    indices = [g.param_index for g in circuit.gates]
    if any(i is None for i in indices) or sorted(indices) != list(range(circuit.num_params)):
        msg = "Pruning B requires one independent parameter per gate."
        raise ValueError(msg)
    removed, baseline = engine.deletion_fidelities(theta)
    scores = baseline - removed
    # legacy.prune takes parameter-indexed scores; deletion scores are gate-indexed.
    parameter_scores = np.empty(circuit.num_params)
    for gate, score in zip(circuit.gates, scores, strict=True):
        parameter_scores[gate.param_index] = score
    pruned, values = legacy.prune(circuit, theta, parameter_scores)
    return (
        pruned,
        values,
        {
            "variant": "B",
            "cutoff": 0.01,
            "initial_fidelity": baseline,
            "removed_gate_indices": np.flatnonzero(scores < 0.01).tolist(),
            "scores": scores.tolist(),
        },
    )


def score_codes(engine: PreparationBackend, theta: NDArray, hardware: str, codes: NDArray) -> dict:
    """Stream unconditional validation histories with exact no-jump compression.

    Variance uses raw multiplicities and the original sample count, not the
    number of retained paths. Only one replay path is propagated at a time.

    Returns:
        Fidelity, raw-sample standard error, approximate 95% interval and coverage.

    Raises:
        ValueError: If codes have invalid shapes or refer to impossible outcomes.
    """
    circuit = engine.circuit
    codes = np.asarray(codes)
    if (
        codes.ndim != 2
        or codes.shape[1] != len(circuit.gates)
        or len(codes) < 2
        or codes.dtype.kind not in "iu"
        or np.any(codes < 0)
    ):
        msg = "Invalid assessment codes."
        raise ValueError(msg)
    outcomes = [depth.noise_outcomes(hardware, g.sites) for g in circuit.gates]
    for k, (gate, (probabilities, _)) in enumerate(zip(circuit.gates, outcomes, strict=True)):
        if np.any(codes[:, k] >= len(probabilities)) or (not gate.noise_enabled and np.any(codes[:, k])):
            msg = "Invalid assessment jump code."
            raise ValueError(msg)
    jumped = np.any(codes != 0, axis=1)
    zero_count = int((~jumped).sum())
    indices = np.flatnonzero(jumped).tolist()
    masses = [1] * len(indices)
    if zero_count:
        indices.insert(0, int(np.flatnonzero(~jumped)[0]))
        masses.insert(0, zero_count)
    total, mean, moment = 0, 0.0, 0.0
    for index, multiplicity in zip(indices, masses, strict=True):
        row = tuple(
            KrotovNoiseMap(operators=() if code == 0 else (outcomes[k][1][int(code) - 1],))
            for k, code in enumerate(codes[index])
        )
        fidelity = engine.metrics(theta, [row])[1]
        if not np.isfinite(fidelity) or not -1e-8 <= fidelity <= 1 + 1e-8:
            msg = "Nonphysical assessment fidelity."
            raise ValueError(msg)
        combined = total + multiplicity
        delta = fidelity - mean
        moment += delta * delta * total * multiplicity / combined
        mean += delta * multiplicity / combined
        total = combined
    se = float(np.sqrt(max(0.0, moment) / (total * (total - 1))))
    return {
        "fidelity": mean,
        "standard_error": se,
        "ci95_normal": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)],
        "sampled_paths": len(codes),
        "no_jump_paths": zero_count,
        "jump_paths": int(jumped.sum()),
        "retained_paths": len(indices),
        "bank_sha256": hashlib.sha256(codes.astype(np.uint8).tobytes()).hexdigest(),
        "status": "no_jump_bank" if circuit.gates and not np.any(jumped) else "complete",
    }


def compile_resources(circuit: ParameterizedCircuit, theta: NDArray) -> dict:
    """Count native gates/depth without full-system unitaries or density matrices.

    On the current adjacent brickwall schedule the fixed compiler requires no
    routing. A changed final layout is rejected. Verification compares prepared
    MPS states, which establishes state-preparation equivalence, not full-unitary
    equivalence. Verification is untruncated regardless of training settings.

    Returns:
        Logical/native resources and prepared-state equivalence diagnostics.

    Raises:
        ValueError: If routing changes the final layout or state equivalence fails.
    """
    source = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        angle = gate.angle_offset + (gate.angle_scale * theta[gate.param_index] if gate.param_index is not None else 0)
        source.append(legacy.PRIMITIVES[gate.name](angle), list(gate.sites))
    compiled = transpile(
        source,
        coupling_map=CouplingMap.from_line(circuit.num_qubits),
        initial_layout=list(range(circuit.num_qubits)),
        **pruning.PROTOCOL["compiler"],
    )
    if compiled.layout is not None and compiled.layout.final_index_layout() != list(range(circuit.num_qubits)):
        msg = "Resource verification requires an unchanged output layout."
        raise ValueError(msg)

    def propagate(qc: QuantumCircuit) -> MPS:
        """Replay local Qiskit matrices in ascending MPS tensor-factor order."""
        state = MPS(qc.num_qubits)
        for item in qc.data:
            sites = tuple(qc.find_bit(q).index for q in item.qubits)
            matrix = Operator(item.operation).data
            if len(sites) == 2:
                if sites[0] < sites[1]:
                    matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
                sites = tuple(sorted(sites))
            state.apply_local_operator(matrix, sites)
        return state

    a, b = propagate(source), propagate(compiled)
    overlap = mps_overlap(a, b)
    residual = max(0.0, float(mps_overlap(a, a).real + mps_overlap(b, b).real - 2 * abs(overlap)))
    if residual > 1e-9:
        msg = "Compiled preparation-state equivalence failed."
        raise ValueError(msg)

    def two(item: CircuitInstruction) -> bool:
        """Count two-qubit operations along dependency-aware depth paths."""
        return len(item.qubits) == 2

    return {
        "logical_one_qubit": sum(len(g.sites) == 1 for g in circuit.gates),
        "logical_two_qubit": sum(len(g.sites) == 2 for g in circuit.gates),
        "logical_two_qubit_depth": source.depth(two),
        "compiled_one_qubit": sum(len(i.qubits) == 1 for i in compiled.data),
        "compiled_two_qubit": sum(two(i) for i in compiled.data),
        "compiled_two_qubit_depth": compiled.depth(two),
        "compiled_total_depth": compiled.depth(),
        "verification": "untruncated MPS prepared-state equivalence up to global phase",
        "phase_aligned_squared_state_distance": residual,
    }
