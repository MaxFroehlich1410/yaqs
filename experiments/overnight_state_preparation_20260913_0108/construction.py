# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Development-only, resumable reproduction of the original H25 scalar starts."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.pilot_components import delete_gates, pruning_scores, select_deletions, with_origins
from experiments import state_preparation_hybrid_pilot as hybrid
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import runner
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovTruncation,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.legacy_h25_construction.v1"
ROOT_SEED = 2026090520
TARGETS = tuple(pilot.TARGETS)
TRUNCATION = {"max_bond_dim": None, "svd_threshold": 0.0, "min_bond_dim": 1, "trunc_mode": "discarded_weight"}


@dataclass(frozen=True)
class Config:
    """Exact legacy scientific schedule; smaller values require engineering mode."""

    mode: str = "development"
    depths: int = 4
    source_updates: int = 100
    online_updates: int = 25
    pruning_rounds: int = 6
    repair_updates: int = 20
    final_updates: int = 50
    checkpoint_every: int = 10
    backend: str = "mps"

    def __post_init__(self) -> None:
        """Reject changed scientific recipes and unsupported optimizer backends."""
        counts = (
            self.depths,
            self.source_updates,
            self.online_updates,
            self.pruning_rounds,
            self.repair_updates,
            self.final_updates,
            self.checkpoint_every,
        )
        if (
            self.mode not in {"development", "engineering"}
            or self.backend != "mps"
            or any(type(value) is not int or value < 1 for value in counts)
            or self.online_updates >= self.source_updates
            or self.pruning_rounds > 3 * self.depths
            or (self.mode == "development" and counts != (4, 100, 25, 6, 20, 50, 10))
        ):
            msg = "Construction requires the frozen legacy MPS schedule or explicit small engineering settings."
            raise ValueError(msg)


def sources() -> dict[str, str]:
    """Return hashes covering MPS contractions, SVDs, gate kernels and this adapter."""
    paths = [
        Path(__file__),
        Path(__file__).with_name("CONSTRUCTION_PROTOCOL.md"),
        Path(pilot.__file__),
        Path(hybrid.__file__),
        Path(runner.__file__),
    ]
    for directory in (runner.REPO / "src/mqt/yaqs", runner.REPO / "benchmarks/state_preparation"):
        paths.extend(directory.rglob("*.py"))
    return {str(p.relative_to(runner.REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment() -> dict[str, Any]:
    """Return numerical dependency and thread settings required for exact resume."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "qiskit": version("qiskit"),
        "opt_einsum": version("opt_einsum"),
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }


def seed_for(role: str, target_id: str, replica: int, block: str | None = None) -> int:
    """Return exactly the old pilot seed domain, without the newer left shift."""
    identity = (role, target_id, replica) if block is None else (role, target_id, replica, block)
    return pilot.pilot_seed(pilot.PilotConfig(root_seed=ROOT_SEED), *identity)


def circuit_at_depth(num_qubits: int, depth: int) -> ParameterizedCircuit:
    """Return physical scalar BMPD preparation with immutable parameter-origin IDs."""
    return with_origins(
        create_brickwall_matrix_product_disentangler_parameterized_circuit(
            num_qubits, depth, initial_single_qubit_layer=True
        )
    )


def validate_target(
    target: NDArray[np.complex128], target_id: str, replica: int, config: Config
) -> NDArray[np.complex128]:
    """Return a copied normalized finite input, rejecting confirmation identities."""
    vector = np.asarray(target, dtype=np.complex128)
    size = vector.size
    qubits = size.bit_length() - 1
    if (
        vector.ndim != 1
        or size < 2
        or 2**qubits != size
        or not np.all(np.isfinite(vector))
        or abs(np.linalg.norm(vector) - 1) > 1e-12
        or type(replica) is not int
        or replica < 0
        or (config.mode == "development" and (qubits != 6 or target_id not in TARGETS or replica not in {11, 29}))
        or (config.mode == "engineering" and (qubits > 3 or not target_id.startswith("synthetic_")))
    ):
        msg = "Supply a normalized development q6 target or explicitly named synthetic q1-q3 engineering vector."
        raise ValueError(msg)
    return vector.copy()


def archive_reference(root: Path, target_id: str, replica: int) -> dict[str, Any]:
    """Return an authenticated archived development endpoint and its unknown CPU cost."""
    if target_id not in TARGETS or replica not in {11, 29}:
        msg = "Archive import is limited to the unchanged development panel."
        raise ValueError(msg)
    manifest_path = root / "manifest.json"
    path = root / "H25_lr0p7" / target_id / f"seed_{replica}/result.json"
    manifest = cast("dict[str, Any]", pilot.read_record(manifest_path))
    record = cast("dict[str, Any]", pilot.read_record(path))
    if (
        manifest["version"] != hybrid.CONSTRUCTION_VERSION
        or manifest["config"]["root_seed"] != ROOT_SEED
        or manifest["config"]["source_updates"] != 100
        or manifest["config"]["repair_updates"] != 20
        or manifest["config"]["refinement_updates"] != 50
        or record["target_id"] != target_id
        or record["replica"] != replica
        or record["setting"] != "H25_lr0p7"
        or record["schedule"] != {"online_updates": 25, "online_rate": 0.7}
        or record["source_construction_checksum"]
        != manifest["source_records"][f"{target_id}/seed_{replica}/construction.json"]
        or record["source_refinement_checksum"]
        != manifest["source_records"][f"{target_id}/seed_{replica}/refinement.json"]
    ):
        msg = "Archived construction identity differs from the selected legacy recipe."
        raise ValueError(msg)
    candidate = next(c for c in record["candidates"] if c["route"] == "H25_lr0p7:cold_final")
    circuit = pilot.restore_circuit(candidate["circuit"])
    physical = runner.counts(circuit)
    if (
        circuit.num_qubits != 6
        or circuit.num_params != 108
        or physical["one_qubit"] != 78
        or physical["two_qubit"] != 30
        or len(candidate["theta"]) != 108
        or not np.all(np.isfinite(candidate["theta"]))
        or any(not gate.noise_enabled for gate in circuit.gates)
    ):
        msg = "Archived construction endpoint has incompatible physical gates or parameters."
        raise ValueError(msg)
    return {
        "target_id": target_id,
        "replica": replica,
        "path": str(path),
        "checksum": pilot.digest(record),
        "manifest_path": str(manifest_path),
        "manifest_checksum": pilot.digest(manifest),
        "source_fingerprint": manifest["source_fingerprint"],
        "circuit": candidate["circuit"],
        "theta": candidate["theta"],
        "historical_cpu_seconds": None,
        "historical_wall_seconds": record["wall_seconds"],
    }


def verify_manifest(manifest: dict[str, Any]) -> None:
    """Reject changed numerical sources, target record, environment or archive."""
    if (
        manifest["version"] != VERSION
        or manifest["sources"] != sources()
        or manifest["environment"] != environment()
        or manifest["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or manifest["truncation"] != asdict(KrotovTruncation())
        or manifest["truncation"] != TRUNCATION
    ):
        msg = "Construction numerical or benchmark custody changed."
        raise ValueError(msg)
    reference = manifest["archive_reference"]
    if reference is not None and (
        pilot.digest(pilot.read_record(Path(reference["path"]))) != reference["checksum"]
        or pilot.digest(pilot.read_record(Path(reference["manifest_path"]))) != reference["manifest_checksum"]
    ):
        msg = "Construction archive reference changed."
        raise ValueError(msg)


def prepare(
    root: Path,
    target: NDArray[np.complex128],
    target_id: str,
    replica: int,
    *,
    config: Config | None = None,
    archive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return and freeze construction inputs without running any circuit or optimizer."""
    config = Config() if config is None else config
    vector = validate_target(target, target_id, replica, config)
    if archive is not None and (archive["target_id"] != target_id or archive["replica"] != replica):
        msg = "Construction reference target or replica differs from the supplied input."
        raise ValueError(msg)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "config": asdict(config),
        "target_id": target_id,
        "replica": replica,
        "target": [[float(z.real), float(z.imag)] for z in vector],
        "target_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "initialization_seed": seed_for("initialization", target_id, replica),
        "root_seed": ROOT_SEED,
        "sources": sources(),
        "environment": environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "truncation": asdict(KrotovTruncation()),
        "archive_reference": archive,
    }
    verify_manifest(manifest)
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(manifest):
        msg = "Frozen construction inputs differ; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, manifest)
    return manifest


def mps_diagnostics(
    circuit: ParameterizedCircuit, theta: NDArray[np.float64], vector: NDArray[np.complex128]
) -> dict[str, Any]:
    """Return observed final-forward internal bonds, norm and full physical overlap."""
    state, target = MPS(circuit.num_qubits, state="zeros"), MPS.from_statevector(vector)
    peaks = []
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta, None)
        state.apply_local_operator(
            matrix, sites, threshold=0.0, max_bond_dim=None, min_bond_dim=1, trunc_mode="discarded_weight"
        )
        peaks.append(max((max(t.shape[1:]) for t in state.tensors), default=1))
    overlap = target.scalar_product(state)
    return {
        "fidelity": float(abs(overlap) ** 2),
        "norm_squared": float(state.scalar_product(state).real),
        "target_internal_bonds": [int(t.shape[2]) for t in target.tensors[:-1]],
        "final_internal_bonds": [int(t.shape[2]) for t in state.tensors[:-1]],
        "maximum_internal_bond_after_each_gate": peaks,
        "observed_final_forward_peak_bond": max(peaks, default=1),
        "training_peak_bond": None,
        "truncation": asdict(KrotovTruncation()),
    }


def run_pipeline(
    root: Path, manifest: dict[str, Any], vector: NDArray[np.complex128], config: Config
) -> dict[str, Any]:
    """Return the legacy final endpoint after resumable deterministic block training."""
    qubits = vector.size.bit_length() - 1
    signature = pilot.digest(manifest)
    master = np.random.default_rng(manifest["initialization_seed"]).normal(
        size=circuit_at_depth(qubits, config.depths).num_params
    )
    theta = 0.1 * master[: circuit_at_depth(qubits, 1).num_params]
    blocks: dict[str, Any] = {}

    def train(name: str, updates: int, rate: float, schedule: hybrid.Schedule) -> NDArray[np.float64]:
        block = hybrid.optimize_block(
            root / "blocks" / f"{name}.json",
            circuit,
            theta,
            vector,
            updates=updates,
            batch_rate=rate,
            schedule=schedule,
            seed=seed_for("training", manifest["target_id"], manifest["replica"], name),
            trajectories=0,
            checkpoint_every=config.checkpoint_every,
            backend="mps",
        )
        values = np.asarray(block["theta"], dtype=np.float64)
        if values.shape != (circuit.num_params,) or not np.all(np.isfinite(values)):
            msg = "Construction checkpoint parameters are invalid."
            raise ValueError(msg)
        blocks[name] = {
            "checksum": pilot.digest(block),
            "completed_updates": block["completed_updates"],
            "executed_chunks": block["executed_chunks"],
            "wall_seconds": block["wall_seconds"],
            "physical_counts": runner.counts(circuit),
            "initial_parameter_checksum": pilot.digest(theta.tolist()),
            "final_parameter_checksum": pilot.digest(values.tolist()),
        }
        return values

    for depth in range(1, config.depths + 1):
        circuit = circuit_at_depth(qubits, depth)
        if theta.size < circuit.num_params:
            theta = np.concatenate((theta, 0.001 * master[theta.size : circuit.num_params]))
        theta = train(f"source_{depth}", config.source_updates, 0.2, hybrid.Schedule(config.online_updates, 0.7))
    events = []
    for index in range(config.pruning_rounds):
        scores = pruning_scores(circuit, theta, vector, "M")
        removed = select_deletions(scores, qubits)
        events.append({
            "round": index + 1,
            "scores": scores,
            "removed_origins": [circuit.gates[i].logical_gate_id for i in removed],
        })
        circuit, theta = delete_gates(circuit, theta, removed)
        theta = train(f"M_repair_{index}", config.repair_updates, 0.1, hybrid.Schedule())
    constructed_theta = theta.tolist()
    theta = train("refine_M_0", config.final_updates, 0.1, hybrid.Schedule())
    diagnostic_cpu = time.process_time()
    diagnostics = mps_diagnostics(circuit, theta, vector)
    diagnostic_cpu = time.process_time() - diagnostic_cpu
    result = {
        "status": "complete",
        "signature": signature,
        "target_id": manifest["target_id"],
        "replica": manifest["replica"],
        "route": "H25_lr0p7:cold_final" if config.mode == "development" else "engineering:final",
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "cold_construct_theta": constructed_theta,
        "physical_counts": runner.counts(circuit),
        "parameters": circuit.num_params,
        "updates": sum(row["completed_updates"] for row in blocks.values()),
        "blocks": blocks,
        "pruning_events": events,
        "diagnostics": diagnostics,
        "diagnostic_cpu_seconds": diagnostic_cpu,
        "archive_reproduction": None,
    }
    if config.mode == "development" and (
        result["physical_counts"]["one_qubit"] != 78
        or result["physical_counts"]["two_qubit"] != 30
        or circuit.num_params != 108
        or result["updates"] != 570
    ):
        msg = "Legacy construction budget or full physical gate count changed."
        raise ValueError(msg)
    reference = manifest["archive_reference"]
    if reference is not None:
        circuit_equal = pilot.digest(result["circuit"]) == pilot.digest(reference["circuit"])
        parameter_error = (
            float(np.max(abs(theta - np.asarray(reference["theta"]))))
            if len(reference["theta"]) == len(theta)
            else None
        )
        result["archive_reproduction"] = {
            "circuit_equal": circuit_equal,
            "parameter_max_abs_error": parameter_error,
            "matches": circuit_equal and parameter_error is not None and parameter_error <= 1e-11,
            "historical_cpu_seconds": None,
            "reference_checksum": reference["checksum"],
        }
    return result


def reproduce(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Recreate or resume the exact layer growth, magnitude deletion and final tail."""
    verify_manifest(manifest)
    if pilot.digest(pilot.read_record(root / "PROTOCOL.json")) != pilot.digest(manifest):
        msg = "Construction job differs from its stored protocol."
        raise ValueError(msg)
    signature = pilot.digest(manifest)
    result_path = root / "RESULT.json"
    if result_path.exists():
        result = cast("dict[str, Any]", pilot.read_record(result_path))
        if result["signature"] != signature:
            msg = "Completed construction identity changed."
            raise ValueError(msg)
        if pilot.digest(pilot.read_record(root / "ATTEMPTS.json")) != result["attempts_checksum"]:
            msg = "Completed construction timing ledger changed."
            raise ValueError(msg)
        for name, block in result["blocks"].items():
            if pilot.digest(pilot.read_record(root / "blocks" / f"{name}.json")) != block["checksum"]:
                msg = "Completed construction block changed."
                raise ValueError(msg)
        return result
    config = Config(**manifest["config"])
    vector = validate_target(
        np.asarray(list(starmap(complex, manifest["target"]))), manifest["target_id"], manifest["replica"], config
    )
    if hashlib.sha256(vector.tobytes()).hexdigest() != manifest["target_vector_sha256"]:
        msg = "Construction target bytes differ from the frozen vector."
        raise ValueError(msg)
    attempts_path = root / "ATTEMPTS.json"
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature:
        msg = "Construction attempt identity changed."
        raise ValueError(msg)
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
    attempts["attempts"].append(attempt)
    pilot.write_record(attempts_path, attempts)
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = run_pipeline(root, manifest, vector, config)
        attempt["status"] = "complete"
    except BaseException as error:
        attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt.update({"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall})
        pilot.write_record(attempts_path, attempts)
    timed = all(a["cpu_seconds"] is not None and a["wall_seconds"] is not None for a in attempts["attempts"])
    result["construction_cpu_seconds"] = sum(a["cpu_seconds"] for a in attempts["attempts"]) if timed else None
    result["construction_wall_seconds"] = sum(a["wall_seconds"] for a in attempts["attempts"]) if timed else None
    result["timing_complete"] = timed
    result["attempts_checksum"] = pilot.digest(attempts)
    verify_manifest(manifest)
    pilot.write_record(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare or explicitly execute one development construction from a supplied NPY."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--target-id", choices=TARGETS, required=True)
    parser.add_argument("--replica", type=int, choices=(11, 29), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.run and not (args.output / "PROTOCOL.json").exists():
        parser.error("Prepare and freeze the development construction before execution.")
    target = np.asarray(np.load(args.target_file, allow_pickle=False), dtype=np.complex128)
    archive = archive_reference(args.archive_root, args.target_id, args.replica) if args.archive_root else None
    manifest = prepare(args.output, target, args.target_id, args.replica, archive=archive)
    if args.prepare_only:
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    result = reproduce(args.output, manifest)
    matched = result["archive_reproduction"] is None or result["archive_reproduction"]["matches"]
    print(result["target_id"], result["replica"], result["status"], "archive_match", matched, flush=True)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
