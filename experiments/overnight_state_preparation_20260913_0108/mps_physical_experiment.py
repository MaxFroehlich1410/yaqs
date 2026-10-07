# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Separate same-input and physical-state audit of the failed MPS overlap gate."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    compilation_experiment,
    construction,
    mps_bridge,
    runner,
    trajectory_mixture,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    mps_overlap_experiment as old,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.mps_physical_followup.v1"
TOLERANCE = old.TOLERANCE
NUMERICAL_ALLOWANCE = 2 * TOLERANCE


def settings() -> dict[str, Any]:
    """Return fixed follow-up criteria without changing the prior failed criteria."""
    return {
        "prior_settings": old.settings(),
        "same_theta_tolerance": TOLERANCE,
        "different_theta_numerical_allowance": NUMERICAL_ALLOWANCE,
        "roundoff_multiplier": 16,
        "different_theta_gate": "phase-aligned distance <= Btheta + numerical allowance",
        "prior_acceptance": "preserved failed; separate engineering audit",
        "physical_dense_backend": "qiskit.quantum_info.Statevector; reversed YAQS sites",
        "scope": "same four development targets, two endpoints, three saved steps, eight original paths",
    }


def sources() -> dict[str, str]:
    """Freeze this implementation, its protocol and every prior numerical dependency."""
    paths = (Path(__file__), Path(__file__).with_name("MPS_PHYSICAL_PROTOCOL.md"))
    return {**old.sources(), **{str(p.relative_to(runner.REPO)): old.file_hash(p) for p in paths}}


def input_panel(root: Path) -> dict[str, Any]:
    """Authenticate completed failed prior records without invoking any old computation."""
    protocol_path, summary_path = root / "PROTOCOL.json", root / "SUMMARY.json"
    protocol, summary = old.read(protocol_path), old.read(summary_path)
    old.verify(protocol)
    if (
        summary["version"] != old.VERSION
        or summary["protocol_checksum"] != pilot.digest(protocol)
        or summary["accepted"] is not False
        or set(summary["cases"]) != set(old.TARGETS)
        or not summary["resource_accounting_complete"]
    ):
        msg = "Physical follow-up requires the complete authenticated failed prior panel."
        raise ValueError(msg)
    files = {str(p): old.file_hash(p) for p in (protocol_path, summary_path)}
    for case in protocol["cases"]:
        case_root = root / f"{case['target_id']}__11"
        result_path = case_root / "RESULT.json"
        result = old.read(result_path)
        signature = pilot.digest([pilot.digest(protocol), case, old.VERSION, old.settings()])
        if (
            result["signature"] != signature
            or result["status"] != "complete"
            or result["accepted"] is not False
            or summary["cases"][case["target_id"]]["checksum"] != pilot.digest(result)
            or set(result["endpoints"]) != set(old.ENDPOINTS)
        ):
            msg = "Prior case identity, completion or failed acceptance changed."
            raise ValueError(msg)
        files[str(result_path)] = old.file_hash(result_path)
        expected = {"TARGET_MPS.json", "ATTEMPTS.json"}
        for label in old.ENDPOINTS:
            expected.add(f"{label}/STATE.json")
            saved = old.read(case_root / label / "STATE.json")
            if result["endpoints"][label]["accepted"] is not False or result["endpoints"][label][
                "state_checksum"
            ] != pilot.digest(saved):
                msg = "Prior saved optimizer state or endpoint acceptance changed."
                raise ValueError(msg)
            replay(case["endpoints"][label], saved)
            for row in saved["steps"]:
                name = f"{label}/{row['raw_audit_file']}"
                expected.add(name)
                if old.file_hash(case_root / name) != row["raw_audit_sha256"]:
                    msg = "Prior compressed split audit changed."
                    raise ValueError(msg)
        if set(result["dependencies"]) != expected:
            msg = "Prior dependency set is incomplete or unknown."
            raise ValueError(msg)
        for name, checksum in result["dependencies"].items():
            path = case_root / name
            if old.file_hash(path) != checksum:
                msg = "Prior numerical artifact failed authentication."
                raise ValueError(msg)
            files[str(path)] = checksum
    return {"root": str(root), "protocol_checksum": pilot.digest(protocol), "cases": protocol["cases"], "files": files}


def prepare(root: Path, prior_root: Path) -> dict[str, Any]:
    """Freeze read-only input custody and new criteria without physical replay."""
    protocol: dict[str, Any] = {
        "version": VERSION,
        "settings": settings(),
        "sources": sources(),
        "environment": construction.environment(),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "input_panel": input_panel(prior_root.resolve()),
    }
    if root.resolve() == prior_root.resolve():
        msg = "The follow-up must use a separate output root."
        raise ValueError(msg)
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(old.read(path)) != pilot.digest(protocol):
        msg = "Follow-up protocol changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return old.read(path)


def verify(protocol: dict[str, Any]) -> None:
    """Reconstruct the required input manifest instead of trusting arbitrary hash lists."""
    if (
        protocol["version"] != VERSION
        or protocol["settings"] != settings()
        or protocol["sources"] != sources()
        or protocol["environment"] != construction.environment()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["input_panel"] != input_panel(Path(protocol["input_panel"]["root"]))
    ):
        msg = "Physical follow-up protocol or input custody changed."
        raise ValueError(msg)


def replay(endpoint: dict[str, Any], saved: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct both complete old Adam states and coordinatewise sensitivity bounds."""
    states = {backend: old.initial_state(endpoint["theta"]) for backend in ("dense", "mps")}
    bound = np.zeros(len(endpoint["theta"]))
    rows = []
    if len(saved["steps"]) != old.UPDATES:
        msg = "The prior overlap trajectory must have exactly three saved updates."
        raise ValueError(msg)
    for index, row in enumerate(saved["steps"]):
        if row["update_index"] != index:
            msg = "Prior step order changed."
            raise ValueError(msg)
        after = {b: old.update(states[b], np.asarray(row[b]["gradient"])) for b in states}
        theta = {b: np.asarray(after[b]["theta"]) for b in states}
        before = {b: np.asarray(states[b]["theta"]) for b in states}
        m = {b: np.asarray(after[b]["first_moment"]) / (1 - old.ADAM.beta1 ** (index + 1)) for b in states}
        s = {b: np.sqrt(np.asarray(after[b]["second_moment"]) / (1 - old.ADAM.beta2 ** (index + 1))) for b in states}
        epsilon = old.ADAM.epsilon
        increment = old.ADAM.learning_rate * (
            np.abs(m["dense"] - m["mps"]) / (s["dense"] + epsilon)
            + np.abs(m["mps"]) * np.abs(s["dense"] - s["mps"]) / ((s["dense"] + epsilon) * (s["mps"] + epsilon))
        )
        rounding = (
            16
            * np.finfo(float).eps
            * (1 + np.abs(before["dense"]) + np.abs(before["mps"]) + np.abs(theta["dense"]) + np.abs(theta["mps"]))
        )
        bound += increment + rounding
        delta = np.abs(theta["dense"] - theta["mps"])
        first = (
            old.ADAM.learning_rate
            / epsilon
            * np.abs(np.asarray(row["dense"]["gradient"]) - np.asarray(row["mps"]["gradient"]))
            + rounding
        )
        rows.append({
            "before": states,
            "after": after,
            "theta_delta": (theta["dense"] - theta["mps"]).tolist(),
            "adam_bound": bound.tolist(),
            "adam_bound_accepted": bool(np.all(delta <= bound)),
            "first_step_bound": first.tolist() if index == 0 else None,
            "first_step_bound_accepted": bool(np.all(delta <= first)) if index == 0 else None,
        })
        states = after
    if any(states[b] != saved[b] for b in states):
        msg = "Saved gradients do not reproduce both full final Adam states bit-for-bit."
        raise ValueError(msg)
    return rows


def authenticate_batch(batch: trajectory_mixture.MixtureBatch, old_row: dict[str, Any]) -> None:
    """Require exact ordered maps and weights, rather than merely matching distributions."""
    if old.maps_hash(batch) != old_row["map_sha256"] or batch.weights.tolist() != old_row["weights"]:
        msg = "Regenerated path order, maps or weights differ from the saved step."
        raise ValueError(msg)


def parameter_bound(
    circuit: ParameterizedCircuit, left: NDArray, right: NDArray, batch: trajectory_mixture.MixtureBatch
) -> dict[str, Any]:
    """Bound phase-aligned unitary state drift, excluding only proven zero-input phases."""
    touched: set[int] = set()
    terms = []
    for position, gate in enumerate(circuit.gates):
        phase = gate.name == "rz" and not touched.intersection(gate.sites)
        if gate.param_index is not None:
            if gate.name not in {"rx", "ry", "rz", "rxx", "ryy", "rzz"}:
                msg = "Sensitivity bounds require scalar Pauli rotations."
                raise ValueError(msg)
            value = abs(float(gate.angle_scale * (left[gate.param_index] - right[gate.param_index]))) / 2
            terms.append({"position": position, "parameter": gate.param_index, "phase": phase, "bound": value})
        touched.update(gate.sites)
        for row in batch.maps:
            for _, sites in row[position].operators:
                touched.update(sites)
    return {"bound": sum(t["bound"] for t in terms if not t["phase"]), "terms": terms}


def state_comparison(left: NDArray, right: NDArray, target: NDArray) -> dict[str, float]:
    """Compare raw vectors after only a global phase alignment, without hiding norm drift."""
    if (
        left.shape != right.shape
        or left.shape != target.shape
        or not all(np.all(np.isfinite(v)) for v in (left, right, target))
    ):
        msg = "Physical state comparison requires finite equal-sized vectors."
        raise ValueError(msg)
    overlap = np.vdot(left, right)
    phase = np.exp(-1j * np.angle(overlap)) if abs(overlap) > 0 else 1.0
    norms = [float(np.vdot(v, v).real) for v in (left, right)]
    if min(norms) <= 0:
        msg = "Physical output cannot have zero norm."
        raise ValueError(msg)
    return {
        "phase_aligned_distance": float(np.linalg.norm(left - phase * right)),
        "unaligned_distance": float(np.linalg.norm(left - right)),
        "left_norm_squared": norms[0],
        "right_norm_squared": norms[1],
        "maximum_norm_error": max(abs(v - 1) for v in norms),
        "normalized_infidelity": max(0.0, 1 - float(abs(overlap) ** 2 / np.prod(norms))),
        "target_fidelity_difference": abs(float(abs(np.vdot(target, left)) ** 2 - abs(np.vdot(target, right)) ** 2)),
    }


def forward_states(
    circuit: ParameterizedCircuit, theta: NDArray, batch: trajectory_mixture.MixtureBatch, *, tensor: bool
) -> tuple[list[NDArray[np.complex128]], dict[str, Any]]:
    """Propagate every ordered path using independent Qiskit or public MPS local updates."""
    mps_bridge.validate_maps(circuit, batch.maps, batch.weights)
    states = []
    path_audits = []
    audit = mps_bridge.BondAudit()
    with ExitStack() as stack:
        if tensor:
            audit.instrument(stack)
        for row in batch.maps:
            split_start = len(audit.splits)
            state = MPS(circuit.num_qubits, state="zeros") if tensor else Statevector.from_int(0, 2**circuit.num_qubits)
            for gate, noise in zip(circuit.gates, row, strict=True):
                operators = (circuit.gate_matrix(gate, theta), *noise.operators)
                for matrix, sites in operators:
                    if isinstance(state, MPS):
                        state.apply_local_operator(matrix, sites, threshold=0, max_bond_dim=None)
                    else:
                        state = state.evolve(Operator(matrix), qargs=list(reversed(sites)))
            vector = np.asarray(state.to_vec() if isinstance(state, MPS) else state.data, dtype=np.complex128)
            states.append(vector)
            path_audits.append({
                "path_index": len(states) - 1,
                "final_bonds": mps_bridge.bonds(state) if isinstance(state, MPS) else None,
                "norm_squared": float(np.vdot(vector, vector).real),
                "split_start": split_start,
                "split_count": len(audit.splits) - split_start,
            })
    return states, {
        **asdict(audit),
        "paths": path_audits,
        "trajectory_paths": len(batch.maps),
        "directional_paths": len(batch.maps),
        "maximum_norm_error": max(abs(float(np.vdot(v, v).real) - 1) for v in states),
        "memory_scope": "individual tensor storage only; process peak unavailable",
    }


def audit_ok(audit: dict[str, Any], qubits: int) -> bool:
    """Retain the old exact no-truncation and finite-dimensional bond requirements."""
    return (
        audit["discarded_squared_singular_values"] == 0.0  # noqa: RUF069
        and audit["hard_cap_reductions"] == 0
        and audit["maximum_internal_bond"] <= 2 ** (qubits // 2)
        and audit.get("maximum_norm_error", audit.get("forward_final_max_norm_error", float("inf"))) <= TOLERANCE
    )


def physical_comparisons(vectors: dict[str, list[NDArray]], target: NDArray, bound: float) -> dict[str, Any]:
    """Apply separately declared same-theta and different-theta physical tests."""
    pairs = {
        "same_dense_theta": ("dense_dense", "mps_dense"),
        "same_mps_theta": ("dense_mps", "mps_mps"),
        "different_dense": ("dense_dense", "dense_mps"),
        "different_mps": ("mps_dense", "mps_mps"),
        "different_cross": ("dense_dense", "mps_mps"),
        "different_reverse_cross": ("mps_dense", "dense_mps"),
    }
    result = {}
    for label, (left, right) in pairs.items():
        rows = [state_comparison(a, b, target) for a, b in zip(vectors[left], vectors[right], strict=True)]
        same = label.startswith("same_")
        distance_limit = TOLERANCE if same else bound + NUMERICAL_ALLOWANCE
        fidelity_limit = TOLERANCE if same else 2 * bound + 2 * NUMERICAL_ALLOWANCE
        result[label] = {
            "paths": rows,
            "distance_limit": distance_limit,
            "fidelity_limit": fidelity_limit,
            "accepted": all(
                row["phase_aligned_distance"] <= distance_limit
                and row["target_fidelity_difference"] <= fidelity_limit
                and row["maximum_norm_error"] <= TOLERANCE
                for row in rows
            ),
        }
    return result


def audit_step(
    circuit: ParameterizedCircuit,
    target: NDArray,
    tensor_target: MPS,
    batch: trajectory_mixture.MixtureBatch,
    previous: dict[str, Any],
    replayed: dict[str, Any],
    attempt: old.Attempt,
) -> dict[str, Any]:
    """Evaluate one new diagnostic unit, retaining full arrays and passive tensor audits."""
    authenticate_batch(batch, previous)
    engine = attempt.call("dense_setup", lambda: StatevectorBackend(circuit, target))
    theta = np.asarray(replayed["before"]["dense"]["theta"])
    dense, loss, fidelity = attempt.call("dense_gradient", partial(batch.gradient, engine, theta))
    paths = attempt.call("dense_preupdate_path_metrics", partial(old.path_fidelities, engine, theta, batch))
    actual = attempt.call(
        "mps_same_theta_gradient",
        partial(mps_bridge.gradient, circuit, theta, tensor_target, batch.maps, batch.weights),
    )
    errors = {
        "gradient": old.difference(dense, actual.gradient),
        "loss": old.difference(loss, actual.loss),
        "fidelity": old.difference(fidelity, actual.fidelity),
        "path_fidelity": old.difference(paths, actual.trajectory_fidelities),
    }
    thetas = {b: np.asarray(replayed["after"][b]["theta"]) for b in ("dense", "mps")}
    sensitivity = parameter_bound(circuit, thetas["dense"], thetas["mps"], batch)
    vectors, audits = {}, {}
    for backend in ("dense", "mps"):
        for source, post in thetas.items():
            key = f"{backend}_{source}"
            vectors[key], audits[key] = attempt.call(
                f"physical_{key}", partial(forward_states, circuit, post, batch, tensor=backend == "mps")
            )
    comparisons = physical_comparisons(vectors, target, sensitivity["bound"])
    authenticate_batch(batch, previous)
    accepted = (
        all(value <= TOLERANCE for value in errors.values())
        and all(row["accepted"] for row in comparisons.values())
        and audit_ok(actual.audit, circuit.num_qubits)
        and all(audit_ok(audits[key], circuit.num_qubits) for key in ("mps_dense", "mps_mps"))
        and replayed["adam_bound_accepted"]
        and replayed["first_step_bound_accepted"] is not False
    )
    return {
        "accepted": accepted,
        "prior_step_accepted": previous["accepted"],
        "map_sha256": old.maps_hash(batch),
        "weights": batch.weights.tolist(),
        "sampling": batch.record,
        "replay": replayed,
        "same_theta_errors": errors,
        "same_theta_dense": {"gradient": dense.tolist(), "loss": loss, "fidelity": fidelity, "paths": paths},
        "same_theta_mps": {
            "gradient": actual.gradient.tolist(),
            "loss": actual.loss,
            "fidelity": actual.fidelity,
            "paths": actual.trajectory_fidelities,
        },
        "parameter_sensitivity": sensitivity,
        "physical_comparisons": comparisons,
        "gradient_audit": actual.audit,
        "physical_audits": audits,
        "states": {
            key: [
                {
                    "path_index": j,
                    "weight": float(batch.weights[j]),
                    "path_sha256": old.maps_hash(
                        replace(batch, maps=(batch.maps[j],), weights=batch.weights[j : j + 1])
                    ),
                    "real": v.real.tolist(),
                    "imag": v.imag.tolist(),
                }
                for j, v in enumerate(values)
            ]
            for key, values in vectors.items()
        },
        "work_counts": {
            "weighted_gradient_calls": 2,
            "gradient_path_calls": 2 * len(batch.maps),
            "forward_path_calls": 5 * len(batch.maps),
            "directional_paths": 9 * len(batch.maps),
            "dense_gradient_implementation": batch.resource_counts("gradient"),
        },
        "nested_mps_gradient_cpu_seconds": actual.cpu_seconds,
        "nested_mps_gradient_wall_seconds": actual.wall_seconds,
    }


def open_attempt(root: Path, signature: str) -> old.Attempt:
    """Block unmeasured hard-kill work instead of making resume costs appear complete."""
    path = root / "ATTEMPTS.json"
    if path.exists():
        saved = old.read(path)
        if saved["signature"] != signature or any(
            a["status"] == "running" or a["cpu_seconds"] is None or a["wall_seconds"] is None for a in saved["attempts"]
        ):
            msg = "Unknown interrupted attempt cost or changed identity blocks automatic resume."
            raise ValueError(msg)
    return old.Attempt(root, signature)


def persist_step(path: Path, signature: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Reuse an authenticated complete unit, otherwise publish compressed raw output atomically."""
    raw_path = path.with_suffix(".raw.json.gz")
    if path.exists():
        record = old.read(path)
        if record["signature"] != signature or record["raw_sha256"] != old.file_hash(raw_path):
            msg = "Completed physical step or raw data changed."
            raise ValueError(msg)
        return record
    raw = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_bytes(
        gzip.compress(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
    )
    temporary.replace(raw_path)
    record: dict[str, Any] = {
        "signature": signature,
        "accepted": raw["accepted"],
        "prior_step_accepted": raw["prior_step_accepted"],
        "same_theta_errors": raw["same_theta_errors"],
        "parameter_bound": raw["parameter_sensitivity"]["bound"],
        "maximum_theta_difference": max(map(abs, raw["replay"]["theta_delta"]), default=0),
        "physical_maxima": {
            key: {metric: max(r[metric] for r in value["paths"]) for metric in value["paths"][0]}
            for key, value in raw["physical_comparisons"].items()
        },
        "work_counts": raw["work_counts"],
        "raw_file": raw_path.name,
        "raw_sha256": old.file_hash(raw_path),
    }
    pilot.write_record(path, record)
    return old.read(path)


def run_endpoint(
    root: Path,
    prior_root: Path,
    case: dict[str, Any],
    label: str,
    signature: str,
    target: NDArray,
    tensor_target: MPS,
    attempt: old.Attempt,
) -> dict[str, Any]:
    """Audit one frozen endpoint using durable complete step units."""
    attempt.context.update(endpoint=label, update_index=None)
    endpoint = case["endpoints"][label]
    saved = attempt.call("prior_state_input", partial(old.read, prior_root / label / "STATE.json"))
    replayed = attempt.call("scalar_adam_replay", partial(replay, endpoint, saved))
    circuit = attempt.call("circuit_restore", partial(compilation_experiment.restore_source, endpoint["circuit"]))
    sampler = attempt.call("sampler_setup", partial(trajectory_mixture.MixtureSampler, circuit))
    rows = []
    for index, (previous, step) in enumerate(zip(saved["steps"], replayed, strict=True)):
        attempt.context["update_index"] = index
        unit_signature = pilot.digest([signature, label, index, previous, step])

        def compute(
            previous: dict[str, Any] = previous,
            step: dict[str, Any] = step,
            index: int = index,
        ) -> dict[str, Any]:
            batch = attempt.call(
                "sampling",
                partial(
                    sampler.ensemble,
                    old.PATHS_PER_CHANNEL,
                    seed=runner.seed_for("mps_overlap", case["target_id"], 11),
                    update_index=index,
                    namespace="development_diagnostics",
                ),
            )
            return audit_step(circuit, target, tensor_target, batch, previous, step, attempt)

        rows.append(persist_step(root / label / f"step{index + 1}.json", unit_signature, compute))
    return {"accepted": all(row["accepted"] for row in rows), "steps": rows}


def run_case(
    root: Path,
    prior_root: Path,
    case: dict[str, Any],
    protocol_signature: str,
    target_loader: Callable[[], NDArray[np.complex128]],
) -> dict[str, Any]:
    """Resume independently durable step units and charge every physical attempt once."""
    signature = pilot.digest([protocol_signature, case, VERSION])
    path = root / "RESULT.json"
    if path.exists():
        result = old.read(path)
        if result["signature"] != signature:
            msg = "Physical case identity changed."
            raise ValueError(msg)
        if result["status"] == "complete":
            if any(old.file_hash(root / p) != h for p, h in result["dependencies"].items()):
                msg = "Physical case dependency changed."
                raise ValueError(msg)
            return result
    attempt = open_attempt(root, signature)
    error: BaseException | None = None
    endpoints: dict[str, Any] = {}
    try:
        target = attempt.call("target_input", target_loader)
        tensor_target = old.target_mps(root, target, attempt)
        for label in old.ENDPOINTS:
            endpoints[label] = run_endpoint(root, prior_root, case, label, signature, target, tensor_target, attempt)
    except BaseException as caught:
        error = caught
    finally:
        attempt.finish(error)
    result: dict[str, Any] = {
        "signature": signature,
        "target_id": case["target_id"],
        "replica": 11,
        "status": "complete" if error is None else "failed",
        "failure": None if error is None else f"{type(error).__name__}: {error}",
        "accepted": error is None and all(row["accepted"] for row in endpoints.values()),
        "prior_gate": "failed; unchanged",
        "endpoints": endpoints,
        "costs": old.costs(attempt.record),
        "dependencies": {
            str(p.relative_to(root)): old.file_hash(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p != path and not p.name.startswith(".") and p.suffix != ".tmp"
        },
    }
    pilot.write_record(path, result)
    if error is not None:
        raise error
    return old.read(path)


def summarize(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Report complete panel coverage and separate new acceptance from old failure."""
    cases: dict[str, Any] = {}
    for case in protocol["input_panel"]["cases"]:
        target = case["target_id"]
        path = root / f"{target}__11" / "RESULT.json"
        result = old.read(path) if path.exists() else None
        cases[target] = {
            "status": "pending" if result is None else result["status"],
            "accepted": result is not None and result["accepted"],
            "checksum": None if result is None else pilot.digest(result),
            "cpu_seconds": None if result is None else result["costs"]["cpu_seconds"],
        }
    summary: dict[str, Any] = {
        "version": VERSION,
        "protocol_checksum": pilot.digest(protocol),
        "cases": cases,
        "accepted": all(c["status"] == "complete" and c["accepted"] for c in cases.values()),
        "prior_gate": "failed; unchanged",
        "scope": "bounded q6 engineering diagnosis; no scaling authorization",
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare without replay, or execute sequentially only after coordinator allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "mps-physical-followup")
    parser.add_argument("--prior-root", type=Path, default=runner.OUTPUT / "mps-overlap")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare(args.output, args.prior_root)
        return 0
    if any(
        os.environ.get(name) != "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    ):
        parser.error("Scientific follow-up requires one explicitly single-thread worker.")
    protocol = old.read(args.output / "PROTOCOL.json")
    verify(protocol)
    for case in protocol["input_panel"]["cases"]:
        target = case["target_id"]
        result = run_case(
            args.output / f"{target}__11",
            Path(protocol["input_panel"]["root"]) / f"{target}__11",
            case,
            pilot.digest(protocol),
            lambda target=target: load_target(6, target).state_vector_copy(),
        )
        verify(protocol)
        summarize(args.output, protocol)
        print(target, result["status"], result["accepted"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
