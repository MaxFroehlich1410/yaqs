# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independently replay every sealed confirmation endpoint without reconstructing benchmark targets."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import best_candidate as parent_api
from experiments.overnight_state_preparation_20260913_0108 import independent_qiskit_oracle as oracle

if TYPE_CHECKING:
    from collections.abc import Callable

VERSION = "overnight.independent_sealed_confirmation_validation.v1"
PANELS = ("primary", "full_cap", "all_trained_sensitivity")
PROTOCOL_PATH = Path(__file__).with_name("INDEPENDENT_CONFIRMATION_VALIDATION_PROTOCOL.md")
TOLERANCE = 1e-10
CPU_CAP = 600.0
FORECAST_PATH = Path(
    "output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_ORACLE_ENGINEERING_TIMING_V2.json"
)


def read(path: Path) -> dict[str, Any]:
    """Read a checksummed plain dictionary."""
    value = pilot.read_record(path)
    if not isinstance(value, dict):
        msg = "Expected a checksummed dictionary."
        raise TypeError(msg)
    return value


def file_hash(path: Path) -> str:
    """Hash source or artifact bytes without decoding numerical targets."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry(root: Path) -> dict[str, str]:
    """Bind every descendant except the two terminal top-level records."""
    excluded = {root / "RESULT.json", root / "COMPLETED.json"}
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in excluded
    }


def sources(parent: dict[str, Any]) -> dict[str, str]:
    """Freeze original dependencies and every new verifier implementation input."""
    paths = [Path(name) for name in parent["source_hashes"]]
    paths.extend((Path(__file__), Path(oracle.__file__), PROTOCOL_PATH, FORECAST_PATH))
    return {str(path.resolve()): file_hash(path) for path in paths}


def close_numeric(value: object, name: str) -> float:
    """Require an actual finite real scalar for physical evidence and clocks."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = f"Invalid finite scalar: {name}."
        raise ValueError(msg)
    return float(value)


def costs(attempts: dict[str, Any]) -> dict[str, float]:
    """Sum observed attempts; unknown process death never becomes zero work."""
    cpu, wall = 0.0, 0.0
    for row in attempts["attempts"]:
        if row["status"] == "running":
            msg = "Unknown hard-kill work blocks resume; an external cost receipt is required."
            raise ValueError(msg)
        if row["status"] not in {"complete", "failed", "interrupted"}:
            msg_0 = "Unknown attempt status."
            raise ValueError(msg_0)
        c = close_numeric(row.get("cpu_seconds"), "CPU")
        w = close_numeric(row.get("wall_seconds"), "wall")
        if c < 0 or w < 0:
            msg = "Negative measured work is invalid."
            raise ValueError(msg)
        cpu, wall = cpu + c, wall + w
    return {"cpu_seconds": cpu, "wall_seconds": wall}


class Attempt:
    """Persist one inclusive invocation before any possibly failing work."""

    def __init__(self, path: Path, identity: object) -> None:
        """Open one durable measured attempt after authenticating all older work."""
        self.path = path
        self.signature = pilot.digest(identity)
        self.record: dict[str, Any] = read(path) if path.exists() else {"signature": self.signature, "attempts": []}
        if self.record["signature"] != self.signature:
            msg = "Attempt identity changed."
            raise ValueError(msg)
        self.previous = costs(self.record)
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        self.row: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
        self.record["attempts"].append(self.row)
        pilot.write_record(path, self.record)

    def finish(self, error: BaseException | None) -> None:
        """Retain complete, failed or caught-interrupted elapsed work."""
        self.row.update(
            status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            cpu_seconds=time.process_time() - self.cpu,
            wall_seconds=time.perf_counter() - self.wall,
        )
        if error is not None:
            self.row["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_record(self.path, self.record)


def input_panel(parent_root: Path, *, engineering: bool = False) -> dict[str, Any]:
    """Authenticate complete sealed cases, saved vectors and every physical endpoint."""
    parent = read(parent_root / "PROTOCOL.json")
    parent_api.verify_parent(parent_root, parent)
    cases = parent["cases"]
    expected = {(target, replica) for target in parent_api.TARGETS for replica in parent_api.REPLICAS}
    actual = {(case["target_id"], case["replica"]) for case in cases}
    if len(actual) != len(cases) or not cases:
        msg = "The sealed case product is empty or duplicated."
        raise ValueError(msg)
    if engineering:
        if parent["phase"] != "engineering" or any(case["num_qubits"] not in {1, 2, 3} for case in cases):
            msg = "Engineering verification is limited to sealed synthetic q1-q3 parents."
            raise ValueError(msg)
    elif parent["phase"] != "confirmation" or actual != expected or any(c["num_qubits"] != 6 for c in cases):
        msg = "Scientific verification requires all ten fixed confirmation cases."
        raise ValueError(msg)
    summary = read(parent_root / "SUMMARY.json")
    summaries = {(row["target_id"], row["replica"]): row for row in summary["cases"]}
    if summary["status"] != "complete" or set(summaries) != actual or len(summary["cases"]) != len(cases):
        msg = "A missing or failed sealed parent cannot define a smaller verification panel."
        raise ValueError(msg)
    files = {
        str((parent_root / name).resolve()): file_hash(parent_root / name)
        for name in ("PROTOCOL.json", "SUMMARY.json", "DESIGN.json")
    }
    endpoints: dict[str, Any] = {}
    roles: list[dict[str, Any]] = []
    for case in cases:
        record = parent_api.read_case(parent_root, parent, case)
        key = (case["target_id"], case["replica"])
        if record["status"] != "complete" or summaries[key]["checksum"] != pilot.digest(record):
            msg = "Sealed case and parent summary disagree."
            raise ValueError(msg)
        child = parent_root / f"{key[0]}__{key[1]}"
        for path in (child / "RESULT.json", child / "execution/COMPLETED.json", child / "execution/input/RESULT.json"):
            files[str(path.resolve())] = file_hash(path)
        loaded = parent_api.read_phase(child / "execution/input", [pilot.digest(parent), case, "input"])
        pairs = loaded["value"]["target"]
        target = np.asarray(list(starmap(complex, pairs)), dtype=np.complex128)
        target_hash = hashlib.sha256(target.tobytes()).hexdigest()
        if (
            target.shape != (2 ** case["num_qubits"],)
            or target_hash != case["target_vector_sha256"]
            or not np.all(np.isfinite(target))
            or abs(np.linalg.norm(target) - 1) > TOLERANCE
        ):
            msg = "Saved target bytes differ from the complete frozen parent."
            raise ValueError(msg)
        if set(record["panels"]) != set(PANELS):
            msg = "Every sealed comparison panel is required before scoring."
            raise ValueError(msg)
        for panel in PANELS:
            arms = record["panels"][panel]["arms"]
            if set(arms) != set(parent_api.METHODS):
                msg = "Every role must exist before independent scoring."
                raise ValueError(msg)
            for method in parent_api.METHODS:
                arm = arms[method]
                point = arm["point"]
                path = child / "execution/training/work/circuits" / f"{point['circuit_hash']}.json"
                circuit = read(path)
                files[str(path.resolve())] = file_hash(path)
                if pilot.digest(circuit) != point["circuit_hash"] or circuit["num_qubits"] != case["num_qubits"]:
                    msg = "A saved physical circuit changed."
                    raise ValueError(msg)
                theta = [close_numeric(v, "theta") for v in point["theta"]]
                if len(theta) != circuit["num_params"]:
                    msg = "Saved parameters do not match their physical circuit."
                    raise ValueError(msg)
                fidelities = arm["scores"]["fidelities"]
                if set(fidelities) != set(oracle.PROFILE_IDS):
                    msg = "Every one of the twelve stored profiles is required."
                    raise ValueError(msg)
                values = {name: close_numeric(fidelities[name], "fidelity") for name in oracle.PROFILE_IDS}
                if any(v < -TOLERANCE or v > 1 + TOLERANCE for v in values.values()):
                    msg = "Stored fidelity is outside its physical range."
                    raise ValueError(msg)
                identity = {"target_vector_sha256": target_hash, "circuit": circuit, "theta": theta}
                endpoint_id = pilot.digest(identity)
                endpoints[endpoint_id] = {**identity, "target": pairs}
                roles.append({
                    "target_id": key[0],
                    "replica": key[1],
                    "panel": panel,
                    "method": method,
                    "endpoint_id": endpoint_id,
                    "expected_fidelities": values,
                    "case_checksum": pilot.digest(record),
                    "point_checksum": pilot.digest(point),
                })
    if len(roles) != len(cases) * len(PANELS) * len(parent_api.METHODS):
        msg = "The full role product must be frozen before any oracle call."
        raise ValueError(msg)
    return {
        "parent_root": str(parent_root.resolve()),
        "parent_checksum": pilot.digest(parent),
        "source_hashes": sources(parent),
        "input_files": files,
        "engineering": engineering,
        "roles": roles,
        "endpoints": endpoints,
    }


def prepare(root: Path, parent_root: Path, *, engineering: bool = False) -> dict[str, Any]:
    """Freeze every expected row and all source identities before numerical verification."""
    parent_api.require_single_threads()
    if (root / "PROTOCOL.json").exists():
        protocol = read(root / "PROTOCOL.json")
        verify(root, protocol)
        return protocol
    attempt = Attempt(root / "PREPARATION_ATTEMPTS.json", [VERSION, str(parent_root.resolve()), engineering])
    try:
        inputs = input_panel(parent_root, engineering=engineering)
        calibrations = {
            name: {str(arity): oracle.calibration_profile(name, arity) for arity in (1, 2)}
            for name in oracle.STANDARD_PROFILES
        }
        pilot.write_record(root / "INPUTS.json", inputs)
        environment = parent_api.construction.environment()
        inputs_checksum, input_file_sha256 = pilot.digest(inputs), file_hash(root / "INPUTS.json")
    except BaseException as error:
        attempt.finish(error)
        raise
    attempt.finish(None)
    protocol: dict[str, Any] = {
        "version": VERSION,
        "status": "frozen",
        "inputs_checksum": inputs_checksum,
        "input_file_sha256": input_file_sha256,
        "engineering": engineering,
        "profile_ids": list(oracle.PROFILE_IDS),
        "role_count": len(inputs["roles"]),
        "row_count": len(inputs["roles"]) * len(oracle.PROFILE_IDS),
        "unique_endpoints": len(inputs["endpoints"]),
        "unique_calls": len(inputs["endpoints"]) * len(oracle.PROFILE_IDS),
        "fidelity_tolerance": TOLERANCE,
        "physicality_tolerance": TOLERANCE,
        "cpu_admission_cap_seconds": CPU_CAP,
        "workers": 1,
        "preparation_attempts_checksum": pilot.digest(attempt.record),
        "preparation_costs": costs(attempt.record),
        "source_hashes": inputs["source_hashes"],
        "environment": environment,
        "calibrations": calibrations,
        "engineering_forecast": {
            "path": str(FORECAST_PATH.resolve()),
            "byte_sha256": inputs["source_hashes"][str(FORECAST_PATH.resolve())],
        },
        "scope": (
            "Independent numerical verification of sealed endpoints; no training, new targets or method selection."
        ),
    }
    pilot.write_record(root / "PROTOCOL.json", protocol)
    return protocol


def verify(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Authenticate complete frozen input and source identities without scoring."""
    if read(root / "PROTOCOL.json") != protocol or protocol["version"] != VERSION:
        msg = "Frozen independent verification protocol changed."
        raise ValueError(msg)
    if (
        protocol["cpu_admission_cap_seconds"] != CPU_CAP
        or protocol["fidelity_tolerance"] != TOLERANCE
        or protocol["physicality_tolerance"] != TOLERANCE
        or protocol["profile_ids"] != list(oracle.PROFILE_IDS)
        or protocol["workers"] != 1
    ):
        msg = "A numerical tolerance, budget or profile cannot be retuned."
        raise ValueError(msg)
    if file_hash(root / "INPUTS.json") != protocol["input_file_sha256"]:
        msg = "Frozen input arrays or endpoint identities changed."
        raise ValueError(msg)
    inputs = read(root / "INPUTS.json")
    if pilot.digest(inputs) != protocol["inputs_checksum"]:
        msg = "Frozen complete input record changed."
        raise ValueError(msg)
    preparation = read(root / "PREPARATION_ATTEMPTS.json")
    if preparation["signature"] != pilot.digest([VERSION, inputs["parent_root"], inputs["engineering"]]):
        msg_0 = "Preparation attempt identity changed."
        raise ValueError(msg_0)
    if (
        pilot.digest(preparation) != protocol["preparation_attempts_checksum"]
        or costs(preparation) != protocol["preparation_costs"]
    ):
        msg = "Preparation cost provenance changed."
        raise ValueError(msg)
    for name, checksum in {**protocol["source_hashes"], **inputs["input_files"]}.items():
        if file_hash(Path(name)) != checksum:
            msg = f"Frozen source or sealed input changed: {name}"
            raise ValueError(msg)
    calibrations = {
        name: {str(arity): oracle.calibration_profile(name, arity) for arity in (1, 2)}
        for name in oracle.STANDARD_PROFILES
    }
    if protocol["calibrations"] != calibrations:
        msg_0 = "Independent calibration metadata changed."
        raise ValueError(msg_0)
    if protocol["environment"] != parent_api.construction.environment():
        msg = "Independent numerical environment changed."
        raise ValueError(msg)
    if (
        protocol["source_hashes"] != inputs["source_hashes"]
        or protocol["engineering"] != inputs["engineering"]
        or protocol["unique_endpoints"] != len(inputs["endpoints"])
    ):
        msg_0 = "Complete frozen source inventory or scope changed."
        raise ValueError(msg_0)
    reconstructed = input_panel(Path(inputs["parent_root"]), engineering=inputs["engineering"])
    if (
        reconstructed != inputs
        or protocol["role_count"] != len(inputs["roles"])
        or protocol["row_count"] != 12 * len(inputs["roles"])
        or protocol["unique_calls"] != 12 * len(inputs["endpoints"])
    ):
        msg = "Frozen role/profile rows differ from the complete sealed parent."
        raise ValueError(msg)
    return inputs


def seal(root: Path, result: dict[str, Any]) -> None:
    """Seal complete child bytes and one reconstructed public result."""
    pilot.write_record(root / "RESULT.json", result)
    pilot.write_record(root / "COMPLETED.json", {"result_checksum": pilot.digest(result), "files": registry(root)})


def cached(root: Path) -> dict[str, Any]:
    """Read a terminal result only when its full descendant inventory still matches."""
    result, completion = read(root / "RESULT.json"), read(root / "COMPLETED.json")
    if completion != {"result_checksum": pilot.digest(result), "files": registry(root)}:
        msg = "Sealed result or complete child inventory changed."
        raise ValueError(msg)
    return result


def call_result(root: Path, identity: list[Any]) -> dict[str, Any]:
    """Reconstruct one complete or failed profile result from raw evidence and paid work."""
    measurement = read(root / "MEASUREMENT.json")
    attempts = read(root / "ATTEMPTS.json")
    if attempts["signature"] != pilot.digest(identity):
        msg = "Independent call identity changed."
        raise ValueError(msg)
    status = measurement.get("status")
    expected_keys = {"status", "oracle"} if status == "complete" else {"status", "failure"}
    if (
        status not in {"complete", "failed"}
        or set(measurement) != expected_keys
        or not attempts["attempts"]
        or attempts["attempts"][-1]["status"] != status
    ):
        msg_0 = "Raw oracle result and terminal attempt status disagree."
        raise ValueError(msg_0)
    if status == "complete":
        validate_oracle(measurement["oracle"], identity[-1])
    elif not isinstance(measurement["failure"], str):
        msg_0 = "A failed oracle call requires explicit failure text."
        raise TypeError(msg_0)
    return {
        "signature": pilot.digest(identity),
        **measurement,
        "costs": costs(attempts),
        "attempts_checksum": pilot.digest(attempts),
    }


def validate_oracle(value: dict[str, Any], noise: str) -> dict[str, Any]:
    """Require finite physical evidence from exactly the requested oracle profile."""
    if value["noise_id"] != noise or value["version"] != oracle.VERSION:
        msg = "Independent oracle returned a different profile or implementation version."
        raise ValueError(msg)
    close_numeric(value["fidelity"], "oracle fidelity")
    physicality(value, TOLERANCE)
    return value


def profile_call(root: Path, identity: list[Any], function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run once, retain terminal failures, and resume only observed interrupted work."""
    if (root / "COMPLETED.json").exists():
        result = cached(root)
        if result != call_result(root, identity):
            msg = "Public call fields differ from raw oracle evidence."
            raise ValueError(msg)
        return result
    attempt = Attempt(root / "ATTEMPTS.json", identity)
    try:
        measured = validate_oracle(function(), identity[-1])
        measurement: dict[str, Any] = {"status": "complete", "oracle": measured}
    except KeyboardInterrupt as error:
        attempt.finish(error)
        raise
    except (ArithmeticError, RuntimeError, ValueError, MemoryError, OSError) as error:
        attempt.finish(error)
        measurement = {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
    except BaseException as error:
        attempt.finish(error)
        raise
    else:
        attempt.finish(None)
    pilot.write_record(root / "MEASUREMENT.json", measurement)
    result = call_result(root, identity)
    seal(root, result)
    return result


def physicality(value: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Apply independent density physicality gates without normalizing or clipping."""
    raw = value["physicality"]
    trace_error = abs(
        complex(close_numeric(raw["trace_real"], "trace real"), close_numeric(raw["trace_imag"], "trace imaginary")) - 1
    )
    hermitian = close_numeric(raw["hermitian_max_abs"], "Hermitian error")
    eigenvalue = close_numeric(raw["minimum_eigenvalue"], "minimum eigenvalue")
    return {
        "trace_error": trace_error,
        "hermitian_max_abs": hermitian,
        "minimum_eigenvalue": eigenvalue,
        "accepted": trace_error <= tolerance and 0 <= hermitian <= tolerance and eigenvalue >= -tolerance,
    }


def endpoint_result(root: Path, identity: object) -> dict[str, Any]:
    """Bind endpoint costs to every observed invocation, including interrupted ones."""
    attempts = read(root / "ATTEMPTS.json")
    if attempts["signature"] != pilot.digest(identity):
        msg = "Endpoint invocation identity changed."
        raise ValueError(msg)
    return {
        "signature": pilot.digest(identity),
        "status": "complete",
        "costs": costs(attempts),
        "attempts_checksum": pilot.digest(attempts),
    }


def endpoint_run(
    root: Path, protocol: dict[str, Any], key: str, endpoint: dict[str, Any], admitted: Callable[[], bool]
) -> None:
    """Record one endpoint invocation while reusing only identical completed profile calls."""
    identity = [pilot.digest(protocol), key]
    if (root / "COMPLETED.json").exists():
        result = cached(root)
        if result != endpoint_result(root, identity):
            msg = "Endpoint invocation identity or cost changed."
            raise ValueError(msg)
        return
    attempt = Attempt(root / "ATTEMPTS.json", identity)
    try:
        target = np.asarray(list(starmap(complex, endpoint["target"])), dtype=np.complex128)
        for noise in protocol["profile_ids"]:
            path = root / "calls" / noise
            if not (path / "COMPLETED.json").exists() and not admitted():
                break
            profile_call(
                path,
                [*identity, noise],
                lambda noise=noise: oracle.evaluate_profile(endpoint["circuit"], endpoint["theta"], target, noise),
            )
    except BaseException as error:
        attempt.finish(error)
        raise
    attempt.finish(None)
    complete = all((root / "calls" / name / "COMPLETED.json").exists() for name in protocol["profile_ids"])
    if complete:
        seal(
            root,
            {
                "signature": pilot.digest(identity),
                "status": "complete",
                "costs": costs(attempt.record),
                "attempts_checksum": pilot.digest(attempt.record),
            },
        )


def summarize(
    root: Path, protocol: dict[str, Any], inputs: dict[str, Any], *, run_costs: dict[str, float] | None = None
) -> dict[str, Any]:
    """Keep every expected role/profile row, including failures and budget-missing work."""
    calls: dict[tuple[str, str], dict[str, Any] | None] = {}
    for key in inputs["endpoints"]:
        endpoint_root = root / "endpoints" / key
        if (endpoint_root / "COMPLETED.json").exists() and cached(endpoint_root) != endpoint_result(
            endpoint_root, [pilot.digest(protocol), key]
        ):
            msg = "Endpoint public costs differ from observed attempts."
            raise ValueError(msg)
        for noise in protocol["profile_ids"]:
            path = root / "endpoints" / key / "calls" / noise
            value = None
            if (path / "COMPLETED.json").exists():
                value = cached(path)
                if value != call_result(path, [pilot.digest(protocol), key, noise]):
                    msg = "Cached oracle evidence changed."
                    raise ValueError(msg)
            calls[key, noise] = value
    rows = []
    for role in inputs["roles"]:
        for noise in protocol["profile_ids"]:
            call = calls[role["endpoint_id"], noise]
            row: dict[str, Any] = {
                **{k: v for k, v in role.items() if k != "expected_fidelities"},
                "noise_id": noise,
                "expected_fidelity": role["expected_fidelities"][noise],
                "status": "missing" if call is None else call["status"],
                "accepted": False,
            }
            if call is not None:
                row["call_checksum"] = pilot.digest(call)
                if call["status"] == "complete":
                    actual = close_numeric(call["oracle"]["fidelity"], "independent fidelity")
                    checks = physicality(call["oracle"], protocol["physicality_tolerance"])
                    error = abs(actual - row["expected_fidelity"])
                    row.update(
                        independent_fidelity=actual,
                        absolute_error=error,
                        physicality=checks,
                        accepted=checks["accepted"] and error <= protocol["fidelity_tolerance"],
                    )
                else:
                    row["failure"] = call["failure"]
            rows.append(row)
    run_attempts = read(root / "RUN_ATTEMPTS.json")
    if run_attempts["signature"] != pilot.digest([VERSION, pilot.digest(protocol)]):
        msg_0 = "Run attempt identity changed."
        raise ValueError(msg_0)
    total = costs(run_attempts) if run_costs is None else run_costs
    combined = {name: total[name] + protocol["preparation_costs"][name] for name in total}
    completed = sum(value is not None for value in calls.values())
    return {
        "version": VERSION,
        "protocol_checksum": pilot.digest(protocol),
        "status": "complete" if completed == protocol["unique_calls"] else "incomplete_budget",
        "accepted": all(row["accepted"] for row in rows),
        "rows": rows,
        "expected_rows": protocol["row_count"],
        "unique_calls_terminal": completed,
        "unique_calls_failed": sum(value is not None and value["status"] != "complete" for value in calls.values()),
        "unique_calls_expected": protocol["unique_calls"],
        "costs": combined,
        "run_costs": total,
        "cpu_admission_cap_seconds": protocol["cpu_admission_cap_seconds"],
        "cpu_overrun_seconds": max(0.0, combined["cpu_seconds"] - protocol["cpu_admission_cap_seconds"]),
        "cost_scope": (
            "Preparation plus inclusive run attempts. Endpoint, oracle and per-call timers are "
            "nested breakdowns, never additional CPU. Final receipt publication and later "
            "cache-only authentication are excluded."
        ),
    }


def run(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Execute the frozen one-worker verification with a cumulative CPU admission cap."""
    parent_api.require_single_threads()
    if (root / "COMPLETED.json").exists():
        inputs = verify(root, protocol)
        result = cached(root)
        if result != summarize(root, protocol, inputs):
            msg = "Public verification rows differ from sealed call evidence."
            raise ValueError(msg)
        return result
    attempt = Attempt(root / "RUN_ATTEMPTS.json", [VERSION, pilot.digest(protocol)])

    def admitted() -> bool:
        return (
            protocol["preparation_costs"]["cpu_seconds"]
            + attempt.previous["cpu_seconds"]
            + time.process_time()
            - attempt.cpu
            < protocol["cpu_admission_cap_seconds"]
        )

    try:
        inputs = verify(root, protocol)
        for key, endpoint in inputs["endpoints"].items():
            if not admitted():
                break
            endpoint_run(root / "endpoints" / key, protocol, key, endpoint, admitted)
        result = summarize(root, protocol, inputs, run_costs=attempt.previous)
    except BaseException as error:
        attempt.finish(error)
        raise
    attempt.finish(None)
    paid = costs(attempt.record)
    combined = {name: paid[name] + protocol["preparation_costs"][name] for name in paid}
    result.update(
        run_costs=paid,
        costs=combined,
        cpu_overrun_seconds=max(0.0, combined["cpu_seconds"] - protocol["cpu_admission_cap_seconds"]),
    )
    seal(root, result)
    return result


def main() -> None:
    """Freeze or execute one explicitly single-thread independent validation panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    parent_api.require_single_threads()
    if args.prepare_only:
        if args.parent is None:
            parser.error("--parent is required for preparation")
        protocol = prepare(args.output, args.parent)
        print({"status": "frozen", "rows": protocol["row_count"], "unique_calls": protocol["unique_calls"]})
    else:
        result = run(args.output, read(args.output / "PROTOCOL.json"))
        print({key: result[key] for key in ("status", "accepted", "unique_calls_terminal", "costs")})


if __name__ == "__main__":
    main()
