# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Develop the complete conditional-initializer and structural ensemble recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    ensemble_experiment,
    initializer_experiment,
    initializer_policy,
    runner,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    ensemble_structural_experiment as exact,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    mixture_baseline_experiment as stochastic,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

VERSION = "overnight.conditional_structural_composition.v1"
TARGETS = tuple(exact.TARGETS)
PRIMARY = (
    "structural_mean2_lbfgs",
    "structural_mean2_adam",
    "structural_mean2_weighted_adam",
    "structural_mean2_adaptive_online",
)
TRAINED = (*PRIMARY, "fixed_mean2_lbfgs", "fixed_noiseless_adam")
METHODS = (*TRAINED, "unchanged")


def restore_physical(record: dict[str, Any]) -> ParameterizedCircuit:
    """Restore every gate field, including the compiler's physical native identities."""
    gates = []
    for gate in record["gates"]:
        fields: dict[str, Any] = {**gate, "sites": tuple(gate["sites"]), "fixed_params": tuple(gate["fixed_params"])}
        gates.append(ParameterizedGate(**fields))
    return ParameterizedCircuit(record["num_qubits"], gates, record["num_params"])


class CompleteStore(exact.Store):
    """Keep all physical gate metadata through exact structural circuit-bank reloads."""

    def restore(self, key: str) -> ParameterizedCircuit:
        """Authenticate and reconstruct complete compiled gate records."""
        record = cast("dict[str, Any]", pilot.read_record(self.root / "circuits" / f"{key}.json"))
        if pilot.digest(record) != key:
            msg = "Composition circuit bank differs from its content identity."
            raise ValueError(msg)
        return restore_physical(record)


def tracked_call(path: Path, identity: object, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Cache a complete operation, retaining interrupted CPU and rejecting unknown deaths."""
    signature = pilot.digest(identity)
    attempts_path = path.with_name(path.stem + ".attempts.json")
    ledger: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if ledger["signature"] != signature or any(a["status"] == "running" for a in ledger["attempts"]):
        msg = "Composition operation changed or unmeasured killed work requires external accounting."
        raise ValueError(msg)
    if path.exists():
        saved = cast("dict[str, Any]", pilot.read_record(path))
        if saved["signature"] != signature or saved["attempts_checksum"] != pilot.digest(ledger):
            msg = "Composition operation or its attempt ledger changed."
            raise ValueError(msg)
        return saved
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
    ledger["attempts"].append(attempt)
    pilot.write_record(attempts_path, ledger)
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        value = function()
        attempt["status"] = "complete"
    except BaseException as error:
        attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        pilot.write_record(attempts_path, ledger)
    result = {
        "signature": signature,
        "value": value,
        "cpu_seconds": sum(a["cpu_seconds"] for a in ledger["attempts"]),
        "wall_seconds": sum(a["wall_seconds"] for a in ledger["attempts"]),
        "attempts_checksum": pilot.digest(ledger),
    }
    pilot.write_record(path, result)
    return result


def compile_final(store: exact.Store, point: dict[str, Any]) -> dict[str, Any]:
    """Make the final compiled copy available only after its complete measured cost."""
    source = {"circuit": pilot.circuit_record(store.restore(point["circuit_hash"])), "theta": point["theta"]}
    result = tracked_call(
        store.root / "compilations" / f"{pilot.digest(source)}.json",
        [VERSION, source],
        lambda: initializer_policy.compile_source(source),
    )
    deployed = result["value"]
    key = store.circuit(restore_physical(deployed["circuit"]))
    return {
        "theta": deployed["theta"],
        "circuit_hash": key,
        "cpu_seconds": point["cpu_seconds"] + result["cpu_seconds"],
        "compiled_final": True,
        "compiler_cpu_seconds": result["cpu_seconds"],
        "compiler_checksum": pilot.digest(result),
    }


def score(store: exact.Store, point: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a physical endpoint exactly under all twelve unchanged conditions."""
    identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
    circuit, theta = store.restore(point["circuit_hash"]), np.asarray(point["theta"])
    scored = store.operation("audit_" + pilot.digest(identity), circuit, theta, profiles=channels.NOISE_IDS)
    return {
        "fidelities": scored["value"]["fidelities"],
        "mean_noisy_fidelity": float(
            np.mean([scored["value"]["fidelities"][k] for k in channels.NOISE_IDS if k != "noiseless"])
        ),
        "physical_counts": runner.counts(circuit),
        "parameters": circuit.num_params,
        "ballarin_counts": runner.counts(exact.materialize_ballarin_circuit(circuit, theta).circuit),
        "evaluation": scored,
    }


def fixed_noiseless(store: exact.Store, source: dict[str, Any]) -> dict[str, Any]:
    """Grant noiseless Adam a full 1500-update refinement ablation and exact resume."""
    root = store.root / "stages" / "fixed_noiseless_adam"
    circuit, theta = restore_physical(source["circuit"]), np.asarray(source["theta"])
    spec = {"optimizer": "adam", "noise_ids": ["noiseless"], "joint_cap": store.config.channel_evaluations // 2}
    call = tracked_call(
        root / "INVOCATION_RESULT.json",
        [VERSION, source, store.target_hash, asdict(store.config), spec],
        lambda: ensemble_experiment.train_arm(root, circuit, theta, store.target, spec, store.config),
    )
    trained = call["value"]
    if trained["attempts_checksum"] != pilot.digest(pilot.read_record(root / "ATTEMPTS.json")):
        msg = "Noiseless training attempts changed."
        raise ValueError(msg)
    key = store.circuit(circuit)
    return {
        "status": "complete",
        "theta": trained["theta"],
        "circuit_hash": key,
        "cpu_seconds": trained["cpu_seconds"],
        "curve": [{**point, "circuit_hash": key} for point in trained["curve"]],
        "individual_channel_calls": trained["individual_channel_gradient_calls"],
        "training_checksum": pilot.digest(trained),
        "invocation_checksum": pilot.digest(call),
    }


def common_panel(
    store: exact.Store, arms: dict[str, Any], methods: Sequence[str], initializer_cpu: float
) -> dict[str, Any]:
    """Select chronological deployable endpoints on the complete initialization clock."""
    if any(arms[method]["status"] != "complete" for method in methods):
        return {"status": "unavailable_failed_arm", "methods": list(methods), "arms": {}}
    cutoff = min(arms[method]["refinement_cpu_seconds"] for method in methods)
    selected = {}
    for method in methods:
        eligible = [point for point in arms[method]["curve"] if point["cpu_seconds"] <= cutoff]
        if not eligible:
            return {"status": "unavailable_no_checkpoint", "methods": list(methods), "arms": {}}
        point = eligible[-1]
        selected[method] = {"point": point, "scores": score(store, point)}
    return {
        "status": "complete",
        "methods": list(methods),
        "refinement_cpu_seconds": cutoff,
        "total_cpu_seconds": initializer_cpu + cutoff,
        "arms": selected,
    }


def train_deploy(
    store: CompleteStore,
    sampled: stochastic.Store,
    source: dict[str, Any],
    method: str,
    signature: str,
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Train one independent arm and audit its charged final compiled endpoint."""
    circuit, theta = restore_physical(source["circuit"]), np.asarray(source["theta"])
    if method in exact.METHODS:
        raw = exact.train_method(store, circuit, theta, method)
    elif method == "fixed_noiseless_adam":
        raw = fixed_noiseless(store, source)
    else:
        raw = stochastic.train_method(sampled, circuit, theta, method)
    last = raw.get("state", raw)
    raw_point = {"theta": last["theta"], "circuit_hash": last["circuit_hash"], "cpu_seconds": raw["cpu_seconds"]}
    final = compile_final(store, raw_point)
    return {
        "status": "complete",
        "signature": signature,
        "raw_training": raw,
        "final": final,
        "curve": [initial, *raw["curve"], raw_point, final],
        "scores": score(store, final),
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "training_cpu_seconds": raw["cpu_seconds"],
        "refinement_cpu_seconds": final["cpu_seconds"],
        "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
    }


def execute(
    root: Path,
    source: dict[str, Any],
    target: NDArray[np.complex128],
    exact_config: exact.Config,
    stochastic_config: stochastic.Config,
    seed: int,
    *,
    order: Sequence[str] = TRAINED,
) -> dict[str, Any]:
    """Run all compatible strong controls on the same complete conditional initializer."""
    if set(order) != set(TRAINED) or len(order) != len(TRAINED):
        msg = "Composition execution must retain every unique trained arm."
        raise ValueError(msg)
    if exact_config.protected_initial_gates or stochastic_config.protected:
        msg = "Composition freezes an empty protected set for every physical initializer."
        raise ValueError(msg)
    circuit, theta = restore_physical(source["circuit"]), np.asarray(source["theta"])
    if len(circuit.gates) <= max(exact_config.max_rounds, stochastic_config.rounds):
        msg = "Composition inputs must retain at least one scalar gate through every permitted deletion."
        raise ValueError(msg)
    store = CompleteStore(root, target, exact_config)
    sampled = stochastic.Store(root, target, stochastic_config, seed)
    initial = {"circuit_hash": store.circuit(circuit), "theta": theta.tolist(), "cpu_seconds": 0.0, "initial": True}
    arms: dict[str, Any] = {}
    for method in order:
        arm: dict[str, Any]
        path = root / "arms" / f"{method}.json"
        signature = pilot.digest([
            VERSION,
            source,
            store.target_hash,
            asdict(exact_config),
            asdict(stochastic_config),
            seed,
            method,
        ])
        if path.exists():
            arm = cast("dict[str, Any]", pilot.read_record(path))
            if arm["signature"] != signature or any(
                hashlib.sha256((root / p).read_bytes()).hexdigest() != h for p, h in arm["dependencies"].items()
            ):
                msg = "Completed composition arm identity changed."
                raise ValueError(msg)
            arms[method] = arm
            continue
        try:
            arm = train_deploy(store, sampled, source, method, signature, initial)
        except (ArithmeticError, RuntimeError) as error:
            arm = {"status": "failed", "signature": signature, "failure": f"{type(error).__name__}: {error}"}
        arm["dependencies"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.json")
            if p != path and p.name not in {"CASE_ATTEMPTS.json", "RESULT.json"}
        }
        pilot.write_record(path, arm)
        arms[method] = arm
    arms["unchanged"] = {
        "status": "complete",
        "scores": score(store, initial),
        "final": initial,
        "total_cpu_seconds": source["initializer_cpu_seconds"],
        "refinement_cpu_seconds": 0.0,
    }
    return {
        "status": "complete" if all(arm["status"] == "complete" for arm in arms.values()) else "partial_failure",
        "arms": arms,
        "common_primary": common_panel(store, arms, PRIMARY, source["initializer_cpu_seconds"]),
        "common_all_trained": common_panel(store, arms, TRAINED, source["initializer_cpu_seconds"]),
        "source": source,
    }


def sources() -> dict[str, str]:
    """Freeze every previously validated primitive and this composition protocol."""
    result = initializer_experiment.sources()
    result.update(stochastic.numerical_sources())
    for name in ("composed_experiment.py", "COMPOSITION_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def input_panel(initializer_root: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Return a complete source-bound initializer panel without invoking scientific work."""
    original = cast("dict[str, Any]", pilot.read_record(initializer_root / "PROTOCOL.json"))
    initializer_experiment.verify(original)
    summary = cast("dict[str, Any]", pilot.read_record(initializer_root / "SUMMARY.json"))
    if summary["status"] != "complete" or summary["protocol_checksum"] != pilot.digest(original):
        msg = "Composition requires every completed initializer development case."
        raise ValueError(msg)
    starts, files = [], {}
    for source in original["starts"]:
        case_root = initializer_root / f"{source['target_id']}__11"
        case = cast("dict[str, Any]", pilot.read_record(case_root / "RESULT.json"))
        if case["signature"] != pilot.digest({"protocol": original, "source": source}) or any(
            hashlib.sha256((case_root / p).read_bytes()).hexdigest() != h for p, h in case["dependencies"].items()
        ):
            msg = "Composition initializer source or durable dependency changed."
            raise ValueError(msg)
        if case["status"] != "complete" or summary["case_checksums"][source["target_id"]] != pilot.digest(case):
            msg = "Composition initializer case is incomplete or changed."
            raise ValueError(msg)
        built = cast("dict[str, Any]", pilot.read_record(case_root / "INITIALIZERS.json"))
        if case["initializer_checksum"] != pilot.digest(built):
            msg = "Composition initializer endpoint differs from its authenticated case."
            raise ValueError(msg)
        starts.append({"target_id": source["target_id"], "replica": 11, "endpoint": built["conditional"]})
        for path in (case_root / "RESULT.json", case_root / "INITIALIZERS.json"):
            files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("PROTOCOL.json", "SUMMARY.json"):
        path = initializer_root / name
        files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return starts, files, original


def prepare(output: Path, initializer_root: Path) -> dict[str, Any]:
    """Freeze a complete authenticated development initializer panel before training."""
    starts, files, original = input_panel(initializer_root)
    protocol = {
        "version": VERSION,
        "phase": "development",
        "methods": list(METHODS),
        "primary": list(PRIMARY),
        "targets": list(TARGETS),
        "replicas": [11],
        "starts": starts,
        "sources": sources(),
        "input_files": files,
        "initializer_root": str(initializer_root.resolve()),
        "exact_config": asdict(exact.Config(protected_initial_gates=0)),
        "stochastic_config": asdict(stochastic.Config(protected=0)),
        "benchmark_hashes": runner.custody()["benchmark_hashes"],
        "environment": original["environment"],
    }
    verify(protocol)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.digest(pilot.read_record(path)) != pilot.digest(protocol):
        msg = "Composition protocol changed; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify(protocol: dict[str, Any]) -> None:
    """Reject configuration, dependency, benchmark and target-panel custody changes."""
    starts, files, _ = input_panel(Path(protocol["initializer_root"]))
    if (
        pilot.digest(protocol["starts"]) != pilot.digest(starts)
        or protocol["input_files"] != files
        or protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["sources"] != sources()
        or protocol["methods"] != list(METHODS)
        or protocol["primary"] != list(PRIMARY)
        or protocol["targets"] != list(TARGETS)
        or protocol["replicas"] != [11]
        or protocol["exact_config"] != asdict(exact.Config(protected_initial_gates=0))
        or protocol["stochastic_config"] != asdict(stochastic.Config(protected=0))
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["environment"] != initializer_experiment.construction.environment()
        or {(s["target_id"], s["replica"]) for s in protocol["starts"]} != {(t, 11) for t in TARGETS}
        or len(protocol["starts"]) != 4
        or any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in protocol["input_files"].items())
    ):
        msg = "Composition frozen inputs or numerical protocol changed."
        raise ValueError(msg)


def measured_execute(root: Path, signature: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Measure the whole scientific invocation, including bookkeeping and all replay."""
    path = root / "CASE_ATTEMPTS.json"
    ledger: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(path)) if path.exists() else {"signature": signature, "attempts": []}
    )
    if ledger["signature"] != signature or any(a["status"] == "running" for a in ledger["attempts"]):
        msg = "Composition case changed or killed invocation requires external accounting."
        raise ValueError(msg)
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
    ledger["attempts"].append(attempt)
    pilot.write_record(path, ledger)
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        result = function()
        attempt["status"] = "complete"
    except BaseException as error:
        attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
        pilot.write_record(path, ledger)
    result["physical_invocation"] = {
        "cpu_seconds": sum(a["cpu_seconds"] for a in ledger["attempts"]),
        "wall_seconds": sum(a["wall_seconds"] for a in ledger["attempts"]),
        "attempts_checksum": pilot.digest(ledger),
        "scope": "inclusive scientific case; excludes input/source verification and final case/attempt writes",
    }
    return result


def run_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute or authenticate one complete development case with full artifact coverage."""
    protocol, source, root = job["protocol"], job["source"], Path(job["root"])
    verify(protocol)
    if pilot.digest(source) not in {pilot.digest(s) for s in protocol["starts"]}:
        msg = "Composition source is outside the frozen development panel."
        raise ValueError(msg)
    signature = pilot.digest([protocol, source])
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or any(
            hashlib.sha256((root / p).read_bytes()).hexdigest() != h for p, h in result["dependencies"].items()
        ):
            msg = "Composition result or durable artifacts changed."
            raise ValueError(msg)
        return result
    offset = TARGETS.index(source["target_id"]) % len(TRAINED)
    result = measured_execute(
        root,
        signature,
        lambda: execute(
            root,
            source["endpoint"],
            load_target(6, source["target_id"]).state_vector_copy(),
            exact.Config(**protocol["exact_config"]),
            stochastic.Config(**protocol["stochastic_config"]),
            runner.seed_for("composition_training", source["target_id"], 11),
            order=(*TRAINED[offset:], *TRAINED[:offset]),
        ),
    )
    verify(protocol)
    result.update(signature=signature, target_id=source["target_id"], replica=11)
    result["dependencies"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json") if p != path
    }
    pilot.write_record(path, result)
    return result


def write_summary(root: Path, protocol: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write separate full-budget and chronological common-CPU scoreboard strata."""
    rows = []
    for case in records:
        panels = {"full_cap": case["arms"]}
        panels.update({name: case[name]["arms"] for name in ("common_primary", "common_all_trained")})
        for panel, arms in panels.items():
            for method in METHODS:
                if panel != "full_cap" and method not in case[panel]["methods"]:
                    continue
                arm = arms.get(method, {"status": "missing"})
                scores = arm.get("scores", {})
                for noise in channels.NOISE_IDS:
                    counts = scores.get("ballarin_counts" if noise == "ballarin_coupled" else "physical_counts", {})
                    rows.append({
                        "target_id": case["target_id"],
                        "replica": 11,
                        "panel": panel,
                        "method": method,
                        "noise_id": noise,
                        "status": arm.get("status", "complete"),
                        "fidelity": scores.get("fidelities", {}).get(noise),
                        "one_qubit": counts.get("one_qubit"),
                        "two_qubit": counts.get("two_qubit"),
                        "depth": counts.get("depth"),
                        "parameters": scores.get("parameters"),
                        "total_cpu_seconds": arm.get("total_cpu_seconds")
                        if panel == "full_cap"
                        else case[panel].get("total_cpu_seconds"),
                    })
    with (root / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pilot.write_record(
        root / "SUMMARY.json",
        {
            "status": "running"
            if len(records) != 4
            else "complete"
            if all(r["status"] == "complete" for r in records)
            else "partial_failure",
            "protocol_checksum": pilot.digest(protocol),
            "case_checksums": {r["target_id"]: pilot.digest(r) for r in records},
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze or explicitly run the complete seven-arm development composition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.OUTPUT / "conditional-composition")
    parser.add_argument("--initializer-root", type=Path, default=runner.OUTPUT / "initializer-policy")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Allocate one to four single-threaded workers.")
    if args.prepare_only:
        prepare(args.output, args.initializer_root)
        print("Prepared only:", args.output / "PROTOCOL.json", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify(protocol)
    jobs = [
        {"protocol": protocol, "source": s, "root": str(args.output / f"{s['target_id']}__11")}
        for s in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_case, jobs):
            records.append(record)
            print(record["target_id"], record["status"], flush=True)
            write_summary(args.output, protocol, records)
    return 0 if all(r["status"] == "complete" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
