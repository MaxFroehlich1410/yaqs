# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Authenticated four-source endpoint compilation screen without optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import operator
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, compilation, runner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.physical_compilation_panel.v1"
TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")
LABELS = ("rank2_unchanged", "rank2_exact_adam_600", "original108", "structural_lbfgs25_final")


def file_hash(path: Path) -> str:
    """Return a streaming artifact digest without loading optimizer histories."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def numerical_sources() -> dict[str, str]:
    """Bind the compiler, driver and preregistration to frozen numerical helpers."""
    result = runner.source_hashes()
    for name in ("compilation.py", "compilation_experiment.py", "COMPILATION_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = file_hash(path)
    return result


def restore_source(record: dict[str, Any]) -> ParameterizedCircuit:
    """Restore scalar metadata without silently discarding unsupported data maps."""
    if any(g.get("data_map") is not None or g.get("fixed_params") for g in record["gates"]):
        msg = "Compilation sources require scalar endpoint gates with no data maps."
        raise ValueError(msg)
    circuit = pilot.restore_circuit(record)
    for gate, source in zip(circuit.gates, record["gates"], strict=True):
        gate.native_gate_id = source.get("native_gate_id")
    return circuit


def load_sources() -> list[dict[str, Any]]:
    """Authenticate and extract small endpoint snapshots, never compilation scores."""
    starts_path = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [s for s in starts if s["replica"] == 11 and s["target_id"] in TARGETS]
    if len(selected) != len(TARGETS) or {s["target_id"] for s in selected} != set(TARGETS):
        msg = "Compilation requires every fixed development construction at replica 11."
        raise ValueError(msg)
    synthesis_root, structural_root = runner.OUTPUT / "synthesis-wave1", runner.OUTPUT / "structural_lbfgs25"
    synthesis_protocol_path, structural_protocol_path = (
        synthesis_root / "PROTOCOL.json",
        structural_root / "PROTOCOL.json",
    )
    synthesis_protocol = cast("dict[str, Any]", pilot.read_record(synthesis_protocol_path))
    structural_protocol = cast("dict[str, Any]", pilot.read_record(structural_protocol_path))
    benchmark_hashes = runner.custody()["benchmark_hashes"]
    if (
        synthesis_protocol["spec"]["phase"] != "development"
        or structural_protocol["phase"] != "development"
        or synthesis_protocol["benchmark_hashes"] != benchmark_hashes
        or structural_protocol["benchmark_hashes"] != benchmark_hashes
        or structural_protocol["config"]["full_evaluations"] != 1500
        or structural_protocol["config"]["repair_evaluations"] != 25
    ):
        msg = "Only development endpoint sources with unchanged benchmark and budgets are permitted."
        raise ValueError(msg)
    sources = []
    for original in selected:
        target = original["target_id"]
        candidate = original["candidate"]
        circuit = restore_source(candidate["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Expected the authenticated original 108-gate q6 source."
            raise ValueError(msg)
        sources.append({
            "target_id": target,
            "replica": 11,
            "label": "original108",
            "circuit": candidate["circuit"],
            "theta": candidate["theta"],
            "origin_checksum": pilot.digest(original),
            "source_costs": {"construction_cpu_seconds": None, "refinement_cpu_seconds": 0.0},
            "file_hashes": {str(starts_path): file_hash(starts_path)},
        })
        synthesis_case = synthesis_root / f"{target}__11"
        construction_path = synthesis_case / "CONSTRUCTION.json"
        construction = cast("dict[str, Any]", pilot.read_record(construction_path))
        signature = pilot.digest({"protocol": synthesis_protocol, "target_id": target, "replica": 11})
        if construction["status"] != "complete" or construction["signature"] != signature:
            msg = "Rank-two construction identity changed."
            raise ValueError(msg)
        for label in ("rank2_unchanged", "rank2_exact_adam_600"):
            path = synthesis_case / f"{label}.json"
            record = cast("dict[str, Any]", pilot.read_record(path))
            job = record["job"]
            if (
                record["status"] != "complete"
                or job["target_id"] != target
                or job["replica"] != 11
                or job["label"] != label
                or job["phase"] != "development"
                or job["protocol_checksum"] != pilot.digest(synthesis_protocol)
                or job["config"] != synthesis_protocol["spec"]["methods"][label]
                or record["signature"] != pilot.digest({k: v for k, v in job.items() if k != "path"})
                or job["construction_checksum"] != pilot.digest(construction)
                or pilot.digest(job["circuit"]) != pilot.digest(construction["candidate"]["circuit"])
                or job["initial"] != construction["candidate"]["theta"]
            ):
                msg = "Rank-two endpoint identity changed."
                raise ValueError(msg)
            sources.append({
                "target_id": target,
                "replica": 11,
                "label": label,
                "circuit": job["circuit"],
                "theta": record["state"]["theta"],
                "origin_checksum": pilot.digest(record),
                "source_costs": {
                    "construction_cpu_seconds": construction["candidate"]["construction_cpu_seconds"],
                    "refinement_cpu_seconds": record["state"]["training_cpu_seconds"],
                },
                "file_hashes": {str(p): file_hash(p) for p in (path, construction_path, synthesis_protocol_path)},
            })
        structural_source = next(
            (s for s in structural_protocol["starts"] if s["target_id"] == target and s["replica"] == 11), None
        )
        if pilot.digest(structural_source) != pilot.digest(original):
            msg = "Structural and original construction source identities differ."
            raise ValueError(msg)
        path = structural_root / f"{target}__11/RESULT.json"
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["status"] != "complete" or record["signature"] != pilot.digest({
            "protocol": structural_protocol,
            "source": original,
        }):
            msg = "Structural L-BFGS final endpoint identity changed."
            raise ValueError(msg)
        arm = record["arms"]["structural_lbfgs"]
        sources.append({
            "target_id": target,
            "replica": 11,
            "label": "structural_lbfgs25_final",
            "circuit": arm["circuit"],
            "theta": arm["theta"],
            "origin_checksum": pilot.digest(record),
            "source_costs": {"construction_cpu_seconds": None, "refinement_cpu_seconds": arm["cpu_seconds"]},
            "file_hashes": {str(p): file_hash(p) for p in (path, structural_protocol_path)},
        })
    return sorted(sources, key=operator.itemgetter("target_id", "label"))


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Freeze all sixteen sources without compiling, training or evaluating any."""
    manifest = runner.custody()
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "labels": list(LABELS),
        "compiler_version": compilation.VERSION,
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "sources": load_sources(),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Frozen compilation protocol differs; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any], source: dict[str, Any] | None = None) -> None:
    """Recheck numerical custody and the relevant streamed source artifact hashes."""
    manifest = runner.custody()
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["labels"] != list(LABELS)
        or protocol["compiler_version"] != compilation.VERSION
        or protocol["benchmark_hashes"] != manifest["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or len(protocol["sources"]) != len(TARGETS) * len(LABELS)
        or {(s["target_id"], s["label"], s["replica"]) for s in protocol["sources"]}
        != {(t, label, 11) for t in TARGETS for label in LABELS}
    ):
        msg = "Compilation protocol custody or numerical identity changed."
        raise ValueError(msg)
    if source is not None:
        if pilot.digest(source) not in {pilot.digest(s) for s in protocol["sources"]}:
            msg = "Compilation source is absent from the frozen panel."
            raise ValueError(msg)
        if any(file_hash(Path(path)) != expected for path, expected in source["file_hashes"].items()):
            msg = "Compilation input artifact changed after preparation."
            raise ValueError(msg)


def ideal_state(circuit: ParameterizedCircuit, theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Prepare the full physical all-zero input with independent Qiskit gates."""
    reference = QuantumCircuit(circuit.num_qubits)
    for gate in circuit.gates:
        getattr(reference, gate.name)(circuit.angle(gate, theta, None), *gate.sites)
    return Statevector.from_instruction(reference).data


def score(circuit: ParameterizedCircuit, theta: NDArray[np.float64], target: NDArray[np.complex128]) -> dict[str, Any]:
    """Audit all twelve profiles plus an independent local-noise density score."""
    cpu = time.process_time()
    fidelities = {noise: channels.evaluate(circuit, theta, target, noise) for noise in channels.NOISE_IDS}
    if any(not np.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10 for value in fidelities.values()):
        msg = "Compilation endpoint produced a nonfinite or invalid fidelity."
        raise FloatingPointError(msg)
    independent = exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider())
    if not np.isfinite(independent) or abs(independent - fidelities["depolarizing_1s_all"]) > 1e-10:
        msg = "Independent compilation endpoint noisy audit disagrees."
        raise FloatingPointError(msg)
    return {
        "status": "complete",
        "fidelities": fidelities,
        "physical_counts": runner.counts(circuit),
        "ballarin_counts": runner.counts(materialize_ballarin_circuit(circuit, theta).circuit),
        "parameters": circuit.num_params,
        "evaluation_cpu_seconds": time.process_time() - cpu,
    }


def compare_endpoint(source: dict[str, Any], target: NDArray[np.complex128]) -> dict[str, Any]:
    """Compare one compiled endpoint with its own untouched source and no training."""
    circuit, theta = restore_source(source["circuit"]), np.asarray(source["theta"])
    result = compilation.compile_endpoint(circuit, theta)
    compiled, values = restore_source(result["circuit"]), np.asarray(result["theta"])
    cpu = time.process_time()
    original_state, compiled_state = ideal_state(circuit, theta), ideal_state(compiled, values)
    overlap = float(abs(np.vdot(original_state, compiled_state)) ** 2)
    error = float(np.max(abs(original_state - np.exp(1j * result["omitted_global_phase"]) * compiled_state)))
    audit_cpu = time.process_time() - cpu
    if abs(overlap - 1) > 1e-10 or error > max(1e-10, result["ideal_operator_error_bound"]):
        msg = "Compilation changed the independently prepared ideal state."
        raise FloatingPointError(msg)
    return {
        "status": "complete",
        "compilation": result,
        "ideal_overlap": overlap,
        "ideal_state_error": error,
        "ideal_audit_cpu_seconds": audit_cpu,
        "arms": {"uncompiled": score(circuit, theta, target), "compiled": score(compiled, values, target)},
    }


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Run or read one authenticated source-paired compilation result."""
    protocol, source, path = job["protocol"], job["source"], Path(job["path"])
    verify_protocol(protocol, source)
    signature = pilot.digest([pilot.digest(protocol), source])
    result: dict[str, Any]
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or result["status"] not in {"complete", "failed"}:
            msg = "Completed compilation case identity changed."
            raise ValueError(msg)
        return result
    target = load_target(6, source["target_id"]).state_vector_copy()
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = compare_endpoint(source, target)
    except (ArithmeticError, RuntimeError, ValueError) as error:
        result = {
            "status": "failed",
            "failure": f"{type(error).__name__}: {error}",
            "arms": {name: {"status": "failed", "fidelities": {}} for name in ("uncompiled", "compiled")},
        }
    result.update({
        "signature": signature,
        "target_id": source["target_id"],
        "replica": 11,
        "source_label": source["label"],
        "source_checksum": pilot.digest(source),
        "source_endpoint": {"circuit": source["circuit"], "theta": source["theta"]},
        "source_costs": source["source_costs"],
        "invocation_cpu_seconds": time.process_time() - cpu,
        "invocation_wall_seconds": time.perf_counter() - wall,
        "comparison": "paired_compilation_of_same_final_endpoint_no_refinement",
    })
    verify_protocol(protocol, source)
    pilot.write_record(path, result)
    return result


def write_scoreboard(root: Path, records: list[dict[str, Any]]) -> None:
    """Retain every paired source, target, condition and failed outcome."""
    rows = []
    for record in sorted(records, key=operator.itemgetter("target_id", "source_label")):
        for name, arm in record["arms"].items():
            for noise in channels.NOISE_IDS:
                physical = arm.get("physical_counts", {})
                lowered = arm.get("ballarin_counts", {}) if noise == "ballarin_coupled" else physical
                rows.append({
                    "target_id": record["target_id"],
                    "replica": 11,
                    "source": record["source_label"],
                    "arm": name,
                    "noise_id": noise,
                    "status": arm["status"],
                    "fidelity": arm["fidelities"].get(noise),
                    "paired_fidelity_change": arm["fidelities"].get(noise, 0)
                    - record["arms"]["uncompiled"]["fidelities"].get(noise, 0)
                    if record["status"] == "complete"
                    else None,
                    "physical_one_qubit": physical.get("one_qubit"),
                    "physical_two_qubit": physical.get("two_qubit"),
                    "physical_depth": physical.get("depth"),
                    "lowered_one_qubit": lowered.get("one_qubit"),
                    "lowered_two_qubit": lowered.get("two_qubit"),
                    "parameters": arm.get("parameters"),
                    "source_construction_cpu_seconds": record["source_costs"]["construction_cpu_seconds"],
                    "source_refinement_cpu_seconds": record["source_costs"]["refinement_cpu_seconds"],
                    "compilation_cpu_seconds": record.get("compilation", {}).get("compilation_cpu_seconds")
                    if name == "compiled"
                    else 0.0,
                    "evaluation_cpu_seconds": arm.get("evaluation_cpu_seconds"),
                    "ideal_overlap": record.get("ideal_overlap"),
                    "failure": record.get("failure"),
                    "comparison": record["comparison"],
                })
    if rows:
        with (root / "SCOREBOARD.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a source-frozen panel or explicitly execute its fixed postprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "compilation-wave1")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four workers.")
    if args.prepare_only:
        freeze_protocol(args.output)
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {
            "protocol": protocol,
            "source": source,
            "path": str(args.output / f"{source['target_id']}__11__{source['label']}.json"),
        }
        for source in protocol["sources"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute_case, jobs):
            records.append(result)
            write_scoreboard(args.output, records)
            print(result["target_id"], result["source_label"], result["status"], flush=True)
    complete = all(record["status"] == "complete" for record in records)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete" if complete else "partial_failure",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
        },
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
