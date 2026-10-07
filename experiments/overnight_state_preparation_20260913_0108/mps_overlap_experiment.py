# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded development-only dense/MPS replay and three-update agreement audit.

The frozen MPS_BRIDGE_PROTOCOL.md defines this implementation gate. Nothing in
this driver authorizes scaling, estimates an optimization advantage, or loads a
confirmation target. Both backends physically start at zero and receive the
same eight weighted unitary paths at each independently seeded update.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compilation_experiment,
    construction,
    mps_bridge,
    runner,
    trajectory_mixture,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import KrotovTruncation, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray

VERSION = "overnight.mps_overlap_development.v1"
TARGETS = tuple(construction.TARGETS)
ENDPOINTS = ("original108", "structural_mean2_lbfgs")
UPDATES = 3
PATHS_PER_CHANNEL = 4
TOLERANCE = 1e-10
ADAM = adam.AdamConfig(updates=UPDATES, learning_rate=0.02)
T = TypeVar("T")


def read(path: Path) -> dict[str, Any]:
    """Read a checked JSON artifact with heterogeneous research metadata."""
    return cast("dict[str, Any]", pilot.read_record(path))


def file_hash(path: Path) -> str:
    """Hash a raw artifact, including compressed audits and persisted tensors."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    """Freeze all numerical MPS, sampler, dense, optimizer and custody dependencies."""
    paths = [
        Path(__file__),
        Path(mps_bridge.__file__),
        Path(trajectory_mixture.__file__),
        Path(compilation_experiment.__file__),
        Path(construction.__file__),
        Path(adam.__file__),
        Path(pilot.__file__),
        Path(runner.__file__),
        Path(__file__).with_name("channels.py"),
        Path(__file__).with_name("MPS_BRIDGE_PROTOCOL.md"),
    ]
    for directory in (runner.REPO / "src/mqt/yaqs", runner.REPO / "benchmarks/state_preparation"):
        paths.extend(directory.rglob("*.py"))
    return {str(path.relative_to(runner.REPO)): file_hash(path) for path in sorted(paths)}


def endpoints(structural_root: Path) -> list[dict[str, Any]]:
    """Authenticate original starts and final mean-two structural L-BFGS endpoints."""
    starts_path = runner.OUTPUT / "starts.json"
    starts = read(starts_path)["starts"]
    protocol_path, final_path = structural_root / "PROTOCOL.json", structural_root / "RESULT.json"
    protocol, final = read(protocol_path), read(final_path)
    selected = [s for s in starts if s["target_id"] in TARGETS and s["replica"] == 11]
    if (
        len(selected) != 4
        or {s["target_id"] for s in selected} != set(TARGETS)
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replica"] != 11
        or protocol["starts"] != selected
        or protocol["source_starts_sha256"] != file_hash(starts_path)
        or protocol["noise_ids"] != list(trajectory_mixture.NOISE_IDS)
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["config"]["repair_evaluations"] != 25
        or protocol["config"]["channel_evaluations"] != 3000
        or final["status"] != "complete"
        or final["protocol_checksum"] != pilot.digest(protocol)
    ):
        msg = "MPS overlap requires the complete frozen development structural25 panel."
        raise ValueError(msg)
    cases = []
    for start in selected:
        target = start["target_id"]
        case_root = structural_root / f"{target}__11"
        result_path = case_root / "RESULT.json"
        result = read(result_path)
        arm = result["arms"][ENDPOINTS[1]]
        circuit_path = case_root / "circuits" / f"{arm['circuit_hash']}.json"
        circuit_record = read(circuit_path)
        if (
            result["status"] != "complete"
            or result["target_id"] != target
            or result["replica"] != 11
            or result["protocol_checksum"] != pilot.digest(protocol)
            or pilot.digest(result) not in final["results"]
            or arm["status"] != "complete"
            or pilot.digest(circuit_record) != arm["circuit_hash"]
            or result["arms"][ENDPOINTS[1]]["curve"][-1]["theta"] != arm["theta"]
        ):
            msg = "Structural final endpoint or original source identity changed."
            raise ValueError(msg)
        pair = {
            "original108": {"circuit": start["candidate"]["circuit"], "theta": start["candidate"]["theta"]},
            ENDPOINTS[1]: {"circuit": circuit_record, "theta": arm["theta"]},
        }
        for label, endpoint in pair.items():
            circuit = compilation_experiment.restore_source(endpoint["circuit"])
            theta = np.asarray(endpoint["theta"])
            if (
                circuit.num_qubits != 6
                or theta.shape != (circuit.num_params,)
                or not np.all(np.isfinite(theta))
                or (label == "original108" and (len(circuit.gates) != 108 or circuit.num_params != 108))
            ):
                msg = "Overlap sources must be lowered finite q6 physical endpoints."
                raise ValueError(msg)
        cases.append({
            "target_id": target,
            "replica": 11,
            "endpoints": pair,
            "endpoints_checksum": pilot.digest(pair),
            "files": {
                str(path): file_hash(path)
                for path in (starts_path, protocol_path, final_path, result_path, circuit_path)
            },
        })
    return cases


def settings() -> dict[str, Any]:
    """Return the fixed small overlap protocol, without configurable scientific knobs."""
    return {
        "updates": UPDATES,
        "paths_per_channel": PATHS_PER_CHANNEL,
        "noise_ids": list(trajectory_mixture.NOISE_IDS),
        "adam": asdict(ADAM),
        "truncation": asdict(KrotovTruncation()),
        "tolerance": TOLERANCE,
        "seed_namespace": "mps_overlap",
        "endpoint": "three fresh updates; final iterate; no selection by fidelity",
        "comparison_points": (
            "path fidelities, loss and gradient before updates 1-3; parameters and full moments after each update"
        ),
        "acceptance": "all cases complete, every error <=1e-10, exact zero discarded tail, q6 bond <=8",
        "scope": "backend agreement only; no confirmation or scaling authorization",
    }


def prepare(root: Path, structural_root: Path) -> dict[str, Any]:
    """Freeze all eight endpoint checksums without target loading or simulation."""
    protocol = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replicas": [11],
        "endpoint_labels": list(ENDPOINTS),
        "settings": settings(),
        "sources": sources(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "cases": endpoints(structural_root),
    }
    verify(protocol)
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(read(path)) != pilot.digest(protocol):
        msg = "MPS overlap protocol changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return read(path)


def verify(protocol: dict[str, Any]) -> None:
    """Enforce benchmark, source, endpoint and complete development panel custody."""
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["endpoint_labels"] != list(ENDPOINTS)
        or protocol["settings"] != settings()
        or protocol["sources"] != sources()
        or protocol["environment"] != construction.environment()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or len(protocol["cases"]) != 4
        or {(c["target_id"], c["replica"]) for c in protocol["cases"]} != {(t, 11) for t in TARGETS}
    ):
        msg = "MPS overlap protocol or numerical custody changed."
        raise ValueError(msg)
    for case in protocol["cases"]:
        if (
            set(case["endpoints"]) != set(ENDPOINTS)
            or pilot.digest(case["endpoints"]) != case["endpoints_checksum"]
            or any(file_hash(Path(path)) != expected for path, expected in case["files"].items())
        ):
            msg = "MPS overlap endpoint source changed."
            raise ValueError(msg)


class Attempt:
    """Durably mark attempted stages and measure failed work without zero imputation."""

    def __init__(self, root: Path, signature: str) -> None:
        """Open a fresh attempt after authenticating the existing independent ledger."""
        self.path = root / "ATTEMPTS.json"
        self.record: dict[str, Any] = (
            read(self.path) if self.path.exists() else {"signature": signature, "attempts": []}
        )
        if self.record["signature"] != signature:
            msg = "Overlap attempt identity changed."
            raise ValueError(msg)
        self.current: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None, "stages": []}
        self.context: dict[str, Any] = {"endpoint": None, "update_index": None}
        self.record["attempts"].append(self.current)
        pilot.write_record(self.path, self.record)
        self.cpu, self.wall = time.process_time(), time.perf_counter()

    def call(self, phase: str, function: Callable[[], T]) -> T:
        """Persist the stage before execution and retain its cost even when it raises."""
        stage: dict[str, Any] = {
            **self.context,
            "phase": phase,
            "status": "running",
            "cpu_seconds": None,
            "wall_seconds": None,
        }
        self.current["stages"].append(stage)
        pilot.write_record(self.path, self.record)
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            value = function()
            stage["status"] = "complete"
            return value
        finally:
            stage.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            if stage["status"] == "running":
                stage["status"] = "failed_or_interrupted"
            pilot.write_record(self.path, self.record)

    def finish(self, error: BaseException | None) -> None:
        """Close a catchable attempt; a hard kill deliberately leaves its null marker."""
        self.current.update(
            status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            cpu_seconds=time.process_time() - self.cpu,
            wall_seconds=time.perf_counter() - self.wall,
        )
        if error is not None:
            self.current["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_record(self.path, self.record)


def stage_costs(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate disjoint measured stages, retaining unknown failure costs separately."""
    return {
        phase: {
            "attempted": sum(s["phase"] == phase for s in stages),
            "completed": sum(s["phase"] == phase and s["status"] == "complete" for s in stages),
            "known_cpu_seconds": sum(s["cpu_seconds"] or 0.0 for s in stages if s["phase"] == phase),
            "known_wall_seconds": sum(s["wall_seconds"] or 0.0 for s in stages if s["phase"] == phase),
            "timing_complete": all(
                s["cpu_seconds"] is not None and s["wall_seconds"] is not None for s in stages if s["phase"] == phase
            ),
        }
        for phase in sorted({s["phase"] for s in stages})
    }


def costs(record: dict[str, Any]) -> dict[str, Any]:
    """Report known costs and explicit unknown totals after uncatchable termination."""
    attempts = record["attempts"]
    known = all(a["cpu_seconds"] is not None and a["wall_seconds"] is not None for a in attempts)
    stages = [s for a in attempts for s in a["stages"]]
    measured = sum(a["cpu_seconds"] or 0.0 for a in attempts)
    return {
        "timing_complete": known,
        "cpu_seconds": measured if known else None,
        "known_cpu_seconds": measured,
        "wall_seconds": sum(a["wall_seconds"] for a in attempts) if known else None,
        "unknown_attempts": sum(a["cpu_seconds"] is None for a in attempts),
        "phases": stage_costs(stages),
        "endpoints": {label: stage_costs([s for s in stages if s.get("endpoint") == label]) for label in ENDPOINTS},
        "total_scope": (
            "all attempted target input, setup, replay, diagnostics, updates and checkpoint I/O; "
            "excludes preflight and final report serialization"
        ),
    }


def target_mps(root: Path, target: NDArray[np.complex128], attempt: Attempt) -> MPS:
    """Factorize once per case and persist exact tensors for resume and both circuits."""
    path = root / "TARGET_MPS.json"
    signature = hashlib.sha256(target.tobytes()).hexdigest()
    if path.exists():
        saved = attempt.call("target_mps_input", partial(read, path))
        if saved["target_sha256"] != signature:
            msg = "Overlap persisted target MPS differs from its dense input."
            raise ValueError(msg)
        tensors = [np.asarray(t["real"]) + 1j * np.asarray(t["imag"]) for t in saved["tensors"]]
        return MPS(len(tensors), tensors=tensors)
    state = attempt.call("target_factorization", lambda: MPS.from_statevector(target))
    error = attempt.call("target_factorization_audit", lambda: float(np.max(np.abs(state.to_vec() - target))))
    if error > TOLERANCE:
        msg = "Exact target factorization failed the overlap tolerance."
        raise ArithmeticError(msg)
    pilot.write_record(
        path,
        {
            "target_sha256": signature,
            "tensors": [{"real": t.real.tolist(), "imag": t.imag.tolist()} for t in state.tensors],
            "bonds": mps_bridge.bonds(state),
            "max_abs_error": error,
            "discarded_squared_singular_values": 0.0,
            "factorization_rule": (
                "MPS.from_statevector retains all SVD singular values; dense exponential input charged separately"
            ),
        },
    )
    return state


def maps_hash(batch: trajectory_mixture.MixtureBatch) -> str:
    """Hash the exact ordered complex operators, sites, normalization flags and weights."""
    record = [
        [
            {
                "normalized": noise.normalized,
                "normalization_checkpoints": noise.normalization_checkpoints,
                "operators": [
                    {"sites": list(sites), "real": operator.real.tolist(), "imag": operator.imag.tolist()}
                    for operator, sites in noise.operators
                ],
            }
            for noise in row
        ]
        for row in batch.maps
    ]
    return pilot.digest({"maps": record, "weights": batch.weights.tolist()})


def initial_state(theta: list[float]) -> dict[str, Any]:
    """Start fresh Adam moments at the supplied frozen physical endpoint."""
    return {
        "theta": theta.copy(),
        "first_moment": [0.0] * len(theta),
        "second_moment": [0.0] * len(theta),
        "iterations": 0,
        "evaluations": 0,
    }


def update(state: dict[str, Any], gradient: NDArray[np.float64]) -> dict[str, Any]:
    """Apply the unchanged repository Adam update and preserve both complete moments."""
    step = state["iterations"] + 1
    theta, first, second = adam.adam_step(
        np.asarray(state["theta"]),
        gradient,
        np.asarray(state["first_moment"]),
        np.asarray(state["second_moment"]),
        step,
        ADAM,
    )
    return {
        "theta": theta.tolist(),
        "first_moment": first.tolist(),
        "second_moment": second.tolist(),
        "iterations": step,
        "evaluations": step,
    }


def difference(left: ArrayLike, right: ArrayLike) -> float:
    """Return the finite maximum absolute discrepancy, including empty vectors."""
    delta = np.abs(np.asarray(left) - np.asarray(right))
    if not np.all(np.isfinite(delta)):
        msg = "Nonfinite dense/MPS overlap discrepancy."
        raise ArithmeticError(msg)
    return float(np.max(delta, initial=0.0))


def path_fidelities(
    engine: StatevectorBackend, theta: NDArray[np.float64], batch: trajectory_mixture.MixtureBatch
) -> list[float]:
    """Use public dense metrics for independently timed individual replay scores."""
    return [engine.metrics(theta, [row])[1] for row in batch.maps]


def train_pair(
    root: Path,
    endpoint: dict[str, Any],
    target: NDArray[np.complex128],
    tensor_target: MPS,
    target_id: str,
    label: str,
    signature: str,
    attempt: Attempt,
) -> dict[str, Any]:
    """Resume three fresh paired updates, saving full moments and each raw bridge audit."""
    path = root / "STATE.json"
    signature = pilot.digest([signature, endpoint, target_id, label, settings()])
    saved: dict[str, Any] = (
        read(path)
        if path.exists()
        else {
            "signature": signature,
            "dense": initial_state(endpoint["theta"]),
            "mps": initial_state(endpoint["theta"]),
            "steps": [],
        }
    )
    if (
        saved["signature"] != signature
        or not 0 <= len(saved["steps"]) <= UPDATES
        or saved["dense"]["iterations"] != len(saved["steps"])
        or saved["mps"]["iterations"] != len(saved["steps"])
    ):
        msg = "Paired Adam checkpoint identity or counters changed."
        raise ValueError(msg)
    for backend in ("dense", "mps"):
        state = saved[backend]
        if (
            type(state["iterations"]) is not int
            or type(state["evaluations"]) is not int
            or state["iterations"] != state["evaluations"]
            or any(
                np.asarray(state[key]).shape != (len(endpoint["theta"]),) or not np.all(np.isfinite(state[key]))
                for key in ("theta", "first_moment", "second_moment")
            )
            or np.any(np.asarray(state["second_moment"]) < 0)
        ):
            msg = "Paired Adam checkpoint has malformed moments, parameters or counters."
            raise ValueError(msg)
    for row in saved["steps"]:
        if file_hash(root / row["raw_audit_file"]) != row["raw_audit_sha256"]:
            msg = "Saved MPS raw audit changed."
            raise ValueError(msg)
    if len(saved["steps"]) == UPDATES:
        return saved
    attempt.context.update(endpoint=label, update_index=None)
    circuit = compilation_experiment.restore_source(endpoint["circuit"])
    engine = attempt.call("dense_setup", lambda: StatevectorBackend(circuit, target))
    sampler = attempt.call("sampler_setup", lambda: trajectory_mixture.MixtureSampler(circuit))
    seed = runner.seed_for("mps_overlap", target_id, 11)
    for step in range(len(saved["steps"]), UPDATES):
        attempt.context["update_index"] = step
        batch = attempt.call(
            "sampling",
            partial(
                sampler.ensemble, PATHS_PER_CHANNEL, seed=seed, update_index=step, namespace="development_diagnostics"
            ),
        )
        replay_hash = maps_hash(batch)
        dense_theta, mps_theta = np.asarray(saved["dense"]["theta"]), np.asarray(saved["mps"]["theta"])
        dense_gradient, dense_loss, dense_fidelity = attempt.call(
            "dense_gradient", partial(batch.gradient, engine, dense_theta)
        )
        dense_paths = attempt.call("dense_path_metrics", partial(path_fidelities, engine, dense_theta, batch))
        actual = attempt.call(
            "mps_gradient",
            partial(
                mps_bridge.gradient,
                circuit,
                mps_theta,
                tensor_target,
                batch.maps,
                batch.weights,
                truncation=KrotovTruncation(),
            ),
        )
        audit = actual.audit
        audit_name = f"step{step + 1}.audit.json.gz"
        raw = json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f".{audit_name}.tmp"
        temporary.write_bytes(gzip.compress(raw, mtime=0))
        temporary.replace(root / audit_name)
        if maps_hash(batch) != replay_hash:
            msg = "A backend mutated the fixed paired trajectory maps."
            raise ValueError(msg)
        next_dense = attempt.call("dense_adam_update", partial(update, saved["dense"], dense_gradient))
        next_mps = attempt.call("mps_adam_update", partial(update, saved["mps"], actual.gradient))
        errors = {
            "loss": difference(dense_loss, actual.loss),
            "fidelity": difference(dense_fidelity, actual.fidelity),
            "path_fidelity": difference(dense_paths, actual.trajectory_fidelities),
            "gradient": difference(dense_gradient, actual.gradient),
            **{key: difference(next_dense[key], next_mps[key]) for key in ("theta", "first_moment", "second_moment")},
        }
        accepted = (
            all(value <= TOLERANCE for value in errors.values())
            # Exact zero is the frozen no-truncation requirement, not an approximate comparison.
            and audit["discarded_squared_singular_values"] == 0.0  # noqa: RUF069
            and audit["maximum_internal_bond"] <= 2 ** (circuit.num_qubits // 2)
            and audit["forward_final_max_norm_error"] <= TOLERANCE
        )
        compact_audit = {key: value for key, value in audit.items() if key != "splits"}
        saved["steps"].append({
            "update_index": step,
            "accepted": accepted,
            "errors": errors,
            "map_sha256": replay_hash,
            "weights": batch.weights.tolist(),
            "sampling": batch.record,
            "dense": {
                "gradient": dense_gradient.tolist(),
                "loss": dense_loss,
                "fidelity": dense_fidelity,
                "trajectory_fidelities": dense_paths,
            },
            "mps": {
                "gradient": actual.gradient.tolist(),
                "loss": actual.loss,
                "fidelity": actual.fidelity,
                "trajectory_fidelities": actual.trajectory_fidelities,
            },
            "audit": compact_audit,
            "raw_audit_file": audit_name,
            "raw_audit_sha256": file_hash(root / audit_name),
            "dense_gradient_resources": batch.resource_counts("gradient"),
            "dense_diagnostic_paths": len(batch.maps),
            "mps_kernel_cpu_seconds": actual.cpu_seconds,
            "mps_kernel_wall_seconds": actual.wall_seconds,
        })
        saved.update(dense=next_dense, mps=next_mps)
        pilot.write_record(path, saved)
    return saved


def run_case(
    root: Path, case: dict[str, Any], signature: str, target_loader: Callable[[], NDArray[np.complex128]]
) -> dict[str, Any]:
    """Execute one target pair with durable failure costs and exact full-state resume."""
    signature = pilot.digest([signature, case, VERSION, settings()])
    path = root / "RESULT.json"
    if path.exists():
        result = read(path)
        if result["signature"] != signature or (
            result["status"] != "failed"
            and any(file_hash(root / name) != value for name, value in result["dependencies"].items())
        ):
            msg = "Completed MPS overlap case or dependency changed."
            raise ValueError(msg)
        if result["status"] != "failed":
            return result
    attempt = Attempt(root, signature)
    error: BaseException | None = None
    pairs = {}
    try:
        target = attempt.call("target_input", target_loader)
        tensor_target = target_mps(root, target, attempt)
        for label in ENDPOINTS:
            pairs[label] = train_pair(
                root / label,
                case["endpoints"][label],
                target,
                tensor_target,
                case["target_id"],
                label,
                signature,
                attempt,
            )
    except BaseException as caught:
        error = caught
    finally:
        attempt.finish(error)
    if error is not None and not isinstance(error, (ArithmeticError, RuntimeError)):
        raise error
    result = {
        "signature": signature,
        "target_id": case["target_id"],
        "replica": 11,
        "status": "failed" if error is not None else "complete",
        "failure": None if error is None else f"{type(error).__name__}: {error}",
        "accepted": error is None and all(row["accepted"] for pair in pairs.values() for row in pair["steps"]),
        "endpoints": {
            label: {
                "accepted": all(row["accepted"] for row in pair["steps"]),
                "state_checksum": pilot.digest(pair),
                "steps": len(pair["steps"]),
                "physical_counts": runner.counts(
                    compilation_experiment.restore_source(case["endpoints"][label]["circuit"])
                ),
                "parameters": len(case["endpoints"][label]["theta"]),
                "maximum_errors": {
                    key: max(row["errors"][key] for row in pair["steps"]) for key in pair["steps"][0]["errors"]
                },
            }
            for label, pair in pairs.items()
        },
        "costs": costs(attempt.record),
        "dependencies": {
            str(p.relative_to(root)): file_hash(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p != path and not p.name.startswith(".")
        },
    }
    pilot.write_record(path, result)
    return result


def execute(job: dict[str, Any]) -> dict[str, Any]:
    """Load only a frozen development vector after full custody verification."""
    protocol, case = job["protocol"], job["case"]
    verify(protocol)
    if case not in protocol["cases"]:
        msg = "Overlap case is outside the fixed development panel."
        raise ValueError(msg)
    result = run_case(
        Path(job["root"]), case, pilot.digest(protocol), lambda: load_target(6, case["target_id"]).state_vector_copy()
    )
    verify(protocol)
    return result


def summarize(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Retain every case and endpoint, including pending work and numerical failures."""
    records, rows = {}, []
    for case in protocol["cases"]:
        target = case["target_id"]
        path = root / f"{target}__11" / "RESULT.json"
        record: dict[str, Any] = (
            read(path) if path.exists() else {"status": "pending", "accepted": False, "endpoints": {}}
        )
        records[target] = {
            "status": record["status"],
            "accepted": record["accepted"],
            "checksum": pilot.digest(record) if path.exists() else None,
            "timing_complete": record.get("costs", {}).get("timing_complete", False),
        }
        for label in ENDPOINTS:
            arm = record["endpoints"].get(label, {})
            rows.append({
                "target_id": target,
                "replica": 11,
                "endpoint": label,
                "status": record["status"],
                "accepted": arm.get("accepted", False),
                "failure": record.get("failure"),
                **arm.get("maximum_errors", {}),
            })
    summary: dict[str, Any] = {
        "version": VERSION,
        "protocol_checksum": pilot.digest(protocol),
        "accepted": all(r["status"] == "complete" and r["accepted"] for r in records.values()),
        "resource_accounting_complete": all(
            r["status"] == "complete" and r["timing_complete"] for r in records.values()
        ),
        "cases": records,
        "scope": "q6 implementation agreement; no scaling authorization",
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    with (root / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare without simulation, or run an existing protocol after coordinator allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "mps-overlap")
    parser.add_argument("--structural-root", type=Path, default=runner.OUTPUT / "ensemble-structural25")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four single-thread numerical workers.")
    if args.prepare_only:
        prepare(args.output, args.structural_root)
        return 0
    protocol = read(args.output / "PROTOCOL.json")
    verify(protocol)
    summarize(args.output, protocol)
    jobs = [
        {"protocol": protocol, "case": case, "root": str(args.output / f"{case['target_id']}__11")}
        for case in protocol["cases"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute, jobs):
            summarize(args.output, protocol)
            print(result["target_id"], result["status"], result["accepted"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
