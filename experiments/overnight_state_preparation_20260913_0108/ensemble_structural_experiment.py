# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Composed mean2 exact deletion/repair with matched deterministic controls.

The frozen ensemble trainer supplies resumable Adam/L-BFGS stages. Circuits are
stored once by digest, while curves retain only parameters and circuit digests.
This is a development mechanism screen; strong stochastic baselines still need
the same objective opportunity before claiming overall method superiority.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import operator
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.pilot_components import delete_gates
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    ensemble_deletion,
    ensemble_experiment,
    runner,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.ensemble_structural.v1"
TARGETS = ensemble_experiment.TARGETS
METHODS = ("structural_mean2_lbfgs", "fixed_mean2_lbfgs", "structural_mean2_adam")


@dataclass(frozen=True)
class Config(ensemble_experiment.Config):
    """Adapt the frozen trainer's budget validation to two-profile structural work."""

    channel_evaluations: int = 3000
    warmup_evaluations: int = 150
    max_rounds: int = 15
    shortlist: int = 2
    repair_evaluations: int = 25
    protected_initial_gates: int = 18
    acceptance_tolerance: float = 1e-10
    ranking_quantum: float = 1e-12

    def __post_init__(self) -> None:
        """Reject budgets that cannot fund warmup, one symmetric round and polish.

        Raises:
            ValueError: If any frozen budget or numerical constant is invalid.
        """
        values = (
            self.channel_evaluations,
            self.warmup_evaluations,
            self.max_rounds,
            self.shortlist,
            self.repair_evaluations,
            self.history,
            self.checkpoint_every,
        )
        reals = (self.adam_learning_rate, self.acceptance_tolerance, self.ranking_quantum)
        minimum = self.warmup_evaluations + 1 + (self.shortlist + 1) * (self.repair_evaluations + 1) + 1
        if (
            any(type(value) is not int or value < 1 for value in values)
            or self.channel_evaluations % 2
            or type(self.protected_initial_gates) is not int
            or self.protected_initial_gates < 0
            or any(isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in reals)
            or self.channel_evaluations // 2 < minimum
        ):
            msg = "Invalid mean2 structural budgets or numerical constants."
            raise ValueError(msg)


class Store:
    """Cache stages and exact operations with compact circuit references and costs."""

    def __init__(self, root: Path, target: NDArray[np.complex128], config: Config) -> None:
        """Bind one target and create the shared circuit bank."""
        self.root, self.target, self.config = root, target.copy(), config
        self.target_hash = hashlib.sha256(target.tobytes()).hexdigest()
        (root / "circuits").mkdir(parents=True, exist_ok=True)

    def circuit(self, circuit: ParameterizedCircuit) -> str:
        """Store a physical circuit once and return its stable content identity."""
        payload = cast("dict[str, Any]", pilot.circuit_record(circuit))
        key = pilot.digest(payload)
        path = self.root / "circuits" / f"{key}.json"
        if path.exists():
            if pilot.digest(pilot.read_record(path)) != key:
                msg = "Stored physical circuit differs from its content identity."
                raise ValueError(msg)
        else:
            pilot.write_record(path, payload)
        return key

    def restore(self, key: str) -> ParameterizedCircuit:
        """Restore a circuit after validating its content-addressed identity."""
        payload = pilot.read_record(self.root / "circuits" / f"{key}.json")
        if pilot.digest(payload) != key:
            msg = "Circuit bank checksum differs from its filename."
            raise ValueError(msg)
        return pilot.restore_circuit(payload)

    def train(
        self, name: str, circuit: ParameterizedCircuit, initial: NDArray[np.float64], cap: int, optimizer: str
    ) -> dict[str, Any]:
        """Run a fresh-memory stage or resume its complete moments/secants."""
        self.circuit(circuit)
        root = self.root / "stages" / name
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "INVOCATIONS.json"
        signature = pilot.digest({
            "circuit": self.circuit(circuit),
            "theta": initial.tolist(),
            "cap": cap,
            "optimizer": optimizer,
            "config": asdict(self.config),
            "target": self.target_hash,
        })
        ledger: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(marker))
            if marker.exists()
            else {"signature": signature, "attempts": []}
        )
        if ledger["signature"] != signature or (ledger["attempts"] and ledger["attempts"][-1]["status"] == "running"):
            msg = "Stage identity changed or killed work requires an external resource ledger."
            raise ValueError(msg)
        final = root / "TRAINING.json"
        if final.exists():
            result = cast("dict[str, Any]", pilot.read_record(final))
            if result["attempts_checksum"] != pilot.digest(pilot.read_record(root / "ATTEMPTS.json")):
                msg = "Cached training physical work ledger changed."
                raise ValueError(msg)
            return result
        entry: dict[str, Any] = {"status": "running"}
        ledger["attempts"].append(entry)
        pilot.write_record(marker, ledger)
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            result = ensemble_experiment.train_arm(
                root,
                circuit,
                initial,
                self.target,
                {"optimizer": optimizer, "noise_ids": list(ensemble_deletion.MEAN2), "joint_cap": cap},
                self.config,
            )
            entry["status"] = "complete"
        except BaseException as error:
            entry["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            pilot.write_record(marker, ledger)
        return result

    def operation(
        self,
        name: str,
        circuit: ParameterizedCircuit,
        theta: NDArray[np.float64],
        *,
        ranking: bool = False,
        profiles: Sequence[str] = ensemble_deletion.MEAN2,
    ) -> dict[str, Any]:
        """Cache exact ranking/scoring with per-channel attempted/completed work."""
        identity = {
            "version": VERSION,
            "circuit": self.circuit(circuit),
            "theta": theta.tolist(),
            "target": self.target_hash,
            "ranking": ranking,
            "profiles": list(profiles),
        }
        signature = pilot.digest(identity)
        root = self.root / "operations"
        root.mkdir(parents=True, exist_ok=True)
        path, attempts_path = root / f"{name}.json", root / f"{name}.attempts.json"
        if path.exists():
            result = cast("dict[str, Any]", pilot.read_record(path))
            if result["signature"] != signature or result["attempts_checksum"] != pilot.digest(
                pilot.read_record(attempts_path)
            ):
                msg = "Cached exact operation or its physical work ledger changed."
                raise ValueError(msg)
            return result
        attempts: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(attempts_path))
            if attempts_path.exists()
            else {"signature": signature, "attempts": []}
        )
        if attempts["signature"] != signature or (
            attempts["attempts"] and attempts["attempts"][-1]["status"] == "running"
        ):
            msg = "Operation identity changed or killed work requires an external resource ledger."
            raise ValueError(msg)
        entry: dict[str, Any] = {"status": "running", "attempted_channel_calls": 0, "completed_channel_calls": 0}
        attempts["attempts"].append(entry)
        pilot.write_record(attempts_path, attempts)

        def progress(_noise: str, phase: str) -> None:
            entry["attempted_channel_calls" if phase == "start" else "completed_channel_calls"] += 1

        def evaluate() -> dict[str, Any]:
            if ranking:
                ranked = ensemble_deletion.deletion_fidelities(circuit, theta, self.target, profiles, progress=progress)
                return {
                    "mean_scores": ranked.mean_scores.tolist(),
                    "mean_fidelity": ranked.mean_fidelity,
                    "fidelities": ranked.profile_fidelities,
                }
            fidelities = {}
            for noise in profiles:
                progress(noise, "start")
                fidelities[noise] = channels.evaluate(circuit, theta, self.target, noise)
                progress(noise, "complete")
            return {"fidelities": fidelities, "mean_fidelity": float(np.mean(list(fidelities.values())))}

        cpu, wall = time.process_time(), time.perf_counter()
        try:
            value = evaluate()
            entry["status"] = "complete"
        except BaseException as error:
            entry["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            pilot.write_record(attempts_path, attempts)
        result: dict[str, Any] = {
            "signature": signature,
            "kind": "ranking" if ranking else "score",
            "value": value,
            "individual_channel_calls": len(profiles),
            "density_direction_passes": len(profiles) * (2 if ranking else 1),
            "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in attempts["attempts"]),
            "physical_completed_channel_calls": sum(row["completed_channel_calls"] for row in attempts["attempts"]),
            "physical_completed_direction_passes": sum(row["completed_channel_calls"] for row in attempts["attempts"])
            * (2 if ranking else 1),
            "cpu_seconds": sum(row["cpu_seconds"] for row in attempts["attempts"]),
            "wall_seconds": sum(row["wall_seconds"] for row in attempts["attempts"]),
            "attempts_checksum": pilot.digest(attempts),
        }
        pilot.write_record(path, result)
        return result


def _stage_entry(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return compact complete stage cost and provenance without copying curves."""
    return {
        "role": name,
        "kind": "gradient",
        "artifact_checksum": pilot.digest(result),
        "individual_channel_calls": result["individual_channel_gradient_calls"],
        "density_direction_passes": result["density_direction_passes"],
        "physical_completed_channel_calls": result["physical_completed_channel_calls"],
        "physical_attempted_channel_calls": result["physical_attempted_channel_calls"],
        "physical_completed_direction_passes": result["physical_completed_direction_passes"],
        "cpu_seconds": result["cpu_seconds"],
    }


def _totals(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Return separate gradient, ranking, forward and physical training costs."""
    return {
        "individual_channel_calls": sum(row["individual_channel_calls"] for row in ledger),
        "gradient_channel_calls": sum(row["individual_channel_calls"] for row in ledger if row["kind"] == "gradient"),
        "ranking_channel_calls": sum(row["individual_channel_calls"] for row in ledger if row["kind"] == "ranking"),
        "forward_channel_calls": sum(row["individual_channel_calls"] for row in ledger if row["kind"] == "score"),
        "density_direction_passes": sum(row["density_direction_passes"] for row in ledger),
        "physical_completed_channel_calls": sum(row["physical_completed_channel_calls"] for row in ledger),
        "physical_attempted_channel_calls": sum(row["physical_attempted_channel_calls"] for row in ledger),
        "physical_completed_direction_passes": sum(row["physical_completed_direction_passes"] for row in ledger),
        "cpu_seconds": sum(row["cpu_seconds"] for row in ledger),
    }


def _curve(result: dict[str, Any], circuit_hash: str, prefix: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return theta-only accepted-iterate points with physical CPU offsets."""
    prefix = prefix or {"cpu_seconds": 0.0, "individual_channel_calls": 0}
    return [
        {
            "circuit_hash": circuit_hash,
            "theta": point["theta"],
            "cpu_seconds": prefix["cpu_seconds"] + point["cpu_seconds"],
            "individual_channel_calls": prefix["individual_channel_calls"] + point["individual_channel_evaluations"],
        }
        for point in result["curve"]
    ]


def physical_work(root: Path, method: str) -> dict[str, Any]:
    """Aggregate raw failed/interrupted attempts without treating missing costs as zero.

    Completed direction counts exclude unfinished individual calls. Invocation
    CPU additionally measures wrapper bookkeeping and is kept separate from the
    numerical/intermediate-checkpoint CPU used for paired trajectory selection.
    """
    result: dict[str, Any] = {
        "accounting_status": "known",
        "training": {"attempted_channel_calls": 0, "completed_channel_calls": 0, "cpu_seconds": 0.0},
        "evaluation": {"attempted_channel_calls": 0, "completed_channel_calls": 0, "cpu_seconds": 0.0},
        "completed_training_direction_passes": 0,
        "invocation_cpu_seconds": 0.0,
        "raw_ledgers": {},
    }

    def accumulate(path: Path, category: str, directions: int) -> None:
        payload = cast("dict[str, Any]", pilot.read_record(path))
        result["raw_ledgers"][str(path.relative_to(root))] = pilot.digest(payload)
        for attempt in payload["attempts"]:
            if attempt.get("status") == "running":
                result["accounting_status"] = "unknown_requires_external_ledger"
                continue
            for key in ("attempted_channel_calls", "completed_channel_calls", "cpu_seconds"):
                result[category][key] += attempt[key]
            if category == "training":
                result["completed_training_direction_passes"] += directions * attempt["completed_channel_calls"]

    for stage in sorted((root / "stages").glob(f"{method}*")):
        if not (stage / "INVOCATIONS.json").exists():
            result["accounting_status"] = "unknown_requires_external_ledger"
            continue
        marker = cast("dict[str, Any]", pilot.read_record(stage / "INVOCATIONS.json"))
        result["raw_ledgers"][str((stage / "INVOCATIONS.json").relative_to(root))] = pilot.digest(marker)
        for entry in marker["attempts"]:
            if entry["status"] == "running":
                result["accounting_status"] = "unknown_requires_external_ledger"
            else:
                result["invocation_cpu_seconds"] += entry["cpu_seconds"]
        if (stage / "ATTEMPTS.json").exists():
            accumulate(stage / "ATTEMPTS.json", "training", 2)
        else:
            result["accounting_status"] = "unknown_requires_external_ledger"
    for path in sorted((root / "operations").glob("*.attempts.json")):
        name = path.name.removesuffix(".attempts.json")
        if name.startswith(method):
            accumulate(path, "training", 2 if name.endswith("_ranking") else 1)
        elif name in {f"final_{method}", f"common_{method}"}:
            accumulate(path, "evaluation", 1)
    return result


def train_method(
    store: Store,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    method: str,
) -> dict[str, Any]:
    """Train one final-iterate method with a hard complete channel-call cap."""
    if method not in METHODS:
        msg = "Unsupported composed structural method."
        raise ValueError(msg)
    config = store.config
    optimizer = "adam" if method.endswith("adam") else "lbfgs"
    if method == "fixed_mean2_lbfgs":
        result = store.train(method, circuit, initial, config.channel_evaluations // 2, optimizer)
        ledger = [_stage_entry(method, result)]
        return {
            "status": "complete",
            "theta": result["theta"],
            "circuit_hash": store.circuit(circuit),
            "curve": _curve(result, store.circuit(circuit)),
            "ledger": ledger,
            "rounds": [],
            **_totals(ledger),
        }
    current = copy.deepcopy(circuit)
    origins = list(range(len(current.gates)))
    warm = store.train(f"{method}_warm", current, initial, config.warmup_evaluations, optimizer)
    theta = np.asarray(warm["theta"])
    ledger = [_stage_entry(f"{method}_warm", warm)]
    curve = _curve(warm, store.circuit(current))
    rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds"
    incoming: float | None = None
    for number in range(config.max_rounds):
        eligible = [index for index, origin in enumerate(origins) if origin >= config.protected_initial_gates]
        count = min(config.shortlist, len(eligible))
        reserved = 1 + (count + 1) * (config.repair_evaluations + 1)
        remaining = (config.channel_evaluations - _totals(ledger)["individual_channel_calls"]) // 2
        if not count or remaining < reserved + 1:
            stop_reason = "no_eligible_gates" if not count else "complete_round_budget"
            break
        prefix = f"{method}_round{number}"
        ranked = store.operation(prefix + "_ranking", current, theta, ranking=True)
        ledger.append({"role": prefix + "_ranking", **{key: value for key, value in ranked.items() if key != "value"}})
        reference = ranked["value"]["mean_fidelity"]
        if incoming is not None and abs(reference - incoming) > 1e-10:
            msg = "Independent mean score and shared ensemble deletion oracle disagree."
            raise FloatingPointError(msg)
        values = ranked["value"]["mean_scores"]
        ranking = sorted(eligible, key=lambda index: (-round(values[index] / config.ranking_quantum), origins[index]))
        ranked_origins = [origins[index] for index in ranking]
        branches: list[dict[str, Any]] = []
        for index in (None, *ranking[:count]):
            candidate, start = (
                (copy.deepcopy(current), theta.copy()) if index is None else delete_gates(current, theta, [index])
            )
            name = prefix + ("_retain" if index is None else f"_delete{origins[index]}")
            repaired = store.train(name, candidate, start, config.repair_evaluations, optimizer)
            ledger.append(_stage_entry(name, repaired))
            scored = store.operation(name + "_score", candidate, np.asarray(repaired["theta"]))
            ledger.append({"role": name + "_score", **{key: value for key, value in scored.items() if key != "value"}})
            branches.append({
                "index": index,
                "origin": None if index is None else origins[index],
                "theta": repaired["theta"],
                "circuit_hash": store.circuit(candidate),
                "fidelity": scored["value"]["mean_fidelity"],
                "training_checksum": pilot.digest(repaired),
            })
        selected = 0 if branches[0]["fidelity"] > reference + config.acceptance_tolerance else None
        best = max(reference, branches[0]["fidelity"])
        for index in range(1, len(branches)):
            if branches[index]["fidelity"] > best + config.acceptance_tolerance:
                selected, best = index, branches[index]["fidelity"]
        if selected is not None:
            choice = branches[selected]
            current, theta = store.restore(choice["circuit_hash"]), np.asarray(choice["theta"])
            if choice["index"] is not None:
                origins.pop(choice["index"])
        incoming = reference if selected is None else branches[selected]["fidelity"]
        rounds.append({
            "round": number + 1,
            "branches": branches,
            "selected_branch": selected,
            "ranking_origins": ranked_origins,
            "ranking_scores": [values[index] for index in ranking],
            "incoming_fidelity": reference,
        })
        curve.append({
            "theta": theta.tolist(),
            "circuit_hash": store.circuit(current),
            **{
                key: value
                for key, value in _totals(ledger).items()
                if key in {"cpu_seconds", "individual_channel_calls"}
            },
        })
        if selected in {None, 0}:
            stop_reason = "no_repaired_deletion_advantage"
            break
    prefix_cost = _totals(ledger)
    remaining = (config.channel_evaluations - prefix_cost["individual_channel_calls"]) // 2
    if remaining < 1:
        msg = "Complete symmetric search rounds must reserve final refinement."
        raise RuntimeError(msg)
    polished = store.train(f"{method}_polish", current, theta, remaining, optimizer)
    curve.extend(_curve(polished, store.circuit(current), prefix_cost))
    ledger.append(_stage_entry(f"{method}_polish", polished))
    return {
        "status": "complete",
        "theta": polished["theta"],
        "circuit_hash": store.circuit(current),
        "curve": curve,
        "ledger": ledger,
        "rounds": rounds,
        "stop_reason": stop_reason,
        "origins": origins,
        **_totals(ledger),
    }


def run_case(
    root: Path,
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
    *,
    target_id: str = "engineering",
    replica: int = 11,
    protocol_checksum: str | None = None,
) -> dict[str, Any]:
    """Run three methods, independently score final and CPU-only selected iterates."""
    signature = pilot.digest({
        "version": VERSION,
        "circuit": pilot.circuit_record(circuit),
        "theta": initial.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "config": asdict(config),
        "target_id": target_id,
        "replica": replica,
        "protocol_checksum": protocol_checksum,
    })
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature:
            msg = "Completed composed structural case identity changed."
            raise ValueError(msg)
        return result
    store = Store(root, target, config)
    arms: dict[str, Any] = {}
    for method in METHODS:
        arm: dict[str, Any] = {}
        try:
            arm = train_method(store, circuit, initial, method)
            final_circuit = store.restore(arm["circuit_hash"])
            scored = store.operation(
                f"final_{method}", final_circuit, np.asarray(arm["theta"]), profiles=channels.NOISE_IDS
            )
            arm.update(
                fidelities=scored["value"]["fidelities"],
                final_evaluation=scored,
                logical_counts=runner.counts(final_circuit),
                ballarin_counts=runner.counts(
                    materialize_ballarin_circuit(final_circuit, np.asarray(arm["theta"])).circuit
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arm.update(status="failed", failure=f"{type(error).__name__}: {error}", partial_artifacts=str(root))
        arms[method] = arm
    common = None
    if all(arm["status"] == "complete" for arm in arms.values()):
        cutoff = min(arm["cpu_seconds"] for arm in arms.values())
        points = {}
        for method, arm in arms.items():
            selected = [point for point in arm["curve"] if point["cpu_seconds"] <= cutoff][-1]
            scored = store.operation(
                f"common_{method}",
                store.restore(selected["circuit_hash"]),
                np.asarray(selected["theta"]),
                profiles=channels.NOISE_IDS,
            )
            points[method] = {"point": selected, "fidelities": scored["value"]["fidelities"], "evaluation": scored}
        common = {"cpu_seconds": cutoff, "methods": points, "selection": "last available iterate by CPU only"}
    for method, arm in arms.items():
        arm["physical_work"] = physical_work(root, method)
    known = all(arm["physical_work"]["accounting_status"] == "known" for arm in arms.values())
    measured_cpu = sum(arm["physical_work"]["training"]["cpu_seconds"] for arm in arms.values())
    result: dict[str, Any] = {
        "signature": signature,
        "status": "complete" if common is not None else "complete_with_failed_arms",
        "target_id": target_id,
        "replica": replica,
        "protocol_checksum": protocol_checksum,
        "arms": arms,
        "common_cpu": common,
        "physical_training_cpu_seconds": measured_cpu if known else None,
        "known_training_cpu_seconds": measured_cpu,
        "physical_accounting_status": "known" if known else "unknown_requires_external_ledger",
        "warmup_rule": "executed and charged separately in every arm",
    }
    pilot.write_record(path, result)
    return result


def numerical_sources() -> dict[str, str]:
    """Freeze the composed modules and all reused numerical dependencies."""
    result = ensemble_experiment.numerical_sources()
    for name in ("ensemble_deletion.py", "ensemble_structural_experiment.py", "ENSEMBLE_STRUCTURAL_PROTOCOL.md"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def freeze_protocol(output: Path) -> dict[str, Any]:
    """Prepare authenticated original starts and one fixed setting without simulation."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [source for source in starts if source["target_id"] in TARGETS and source["replica"] == 11]
    if len(selected) != len(TARGETS) or {source["target_id"] for source in selected} != set(TARGETS):
        msg = "The complete four-target original construction panel is required."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Composed structural science requires the original q6 108-gate starts."
            raise ValueError(msg)
    protocol: dict[str, Any] = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "methods": list(METHODS),
        "noise_ids": list(ensemble_deletion.MEAN2),
        "config": asdict(Config()),
        "starts": selected,
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "source_hashes": numerical_sources(),
        "benchmark_hashes": manifest["benchmark_hashes"],
        "comparison_scope": "deterministic mechanism screen; strong stochastic same-objective controls remain required",
        "endpoint": "final iterate",
        "storage": "circuit once by digest; theta-only curves; complete current moments/secants",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen ensemble-structural protocol differs."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Reject changed benchmark, original starts, sources or scientific settings."""
    if (
        protocol["version"] != VERSION
        or protocol["targets"] != list(TARGETS)
        or protocol["replica"] != 11
        or protocol["methods"] != list(METHODS)
        or protocol["noise_ids"] != list(ensemble_deletion.MEAN2)
        or protocol["config"] != asdict(Config())
        or protocol["source_hashes"] != numerical_sources()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["source_starts_sha256"] != hashlib.sha256((runner.OUTPUT / "starts.json").read_bytes()).hexdigest()
    ):
        msg = "Composed structural benchmark, source or configuration custody mismatch."
        raise ValueError(msg)
    original = cast("list[dict[str, Any]]", pilot.read_record(runner.OUTPUT / "starts.json")["starts"])
    expected = [source for source in original if source["target_id"] in TARGETS and source["replica"] == 11]
    if protocol["starts"] != expected:
        msg = "Composed structural original starts changed inside protocol."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one frozen target with separate exact final evaluation."""
    protocol, source = job["protocol"], job["source"]
    verify_protocol(protocol)
    result = run_case(
        Path(job["root"]),
        pilot.restore_circuit(source["candidate"]["circuit"]),
        np.asarray(source["candidate"]["theta"]),
        load_target(6, source["target_id"]).state_vector_copy(),
        Config(**protocol["config"]),
        target_id=source["target_id"],
        replica=source["replica"],
        protocol_checksum=pilot.digest(protocol),
    )
    verify_protocol(protocol)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Preserve all method/condition endpoints and explicit failures."""
    rows = []
    for record in sorted(records, key=operator.itemgetter("target_id", "replica")):
        for method in METHODS:
            arm = record["arms"][method]
            for noise in channels.NOISE_IDS:
                row = {
                    "target_id": record["target_id"],
                    "replica": record["replica"],
                    "method": method,
                    "noise_id": noise,
                    "status": arm["status"],
                    "failure": arm.get("failure"),
                    "convention": "calibrated_exact",
                    "exactness": "exact_calibrated_channel",
                    "case_checksum": pilot.digest(record),
                }
                if arm["status"] == "complete":
                    row.update(
                        fidelity=arm["fidelities"][noise],
                        **{
                            key: arm[key]
                            for key in (
                                "individual_channel_calls",
                                "gradient_channel_calls",
                                "ranking_channel_calls",
                                "forward_channel_calls",
                                "density_direction_passes",
                                "physical_completed_channel_calls",
                                "cpu_seconds",
                            )
                        },
                        logical_one_qubit=arm["logical_counts"]["one_qubit"],
                        logical_two_qubit=arm["logical_counts"]["two_qubit"],
                        logical_depth=arm["logical_counts"]["depth"],
                        ballarin_one_qubit=arm["ballarin_counts"]["one_qubit"],
                        ballarin_two_qubit=arm["ballarin_counts"]["two_qubit"],
                        ballarin_depth=arm["ballarin_counts"]["depth"],
                        parameters=len(arm["theta"]),
                        evaluation_cpu_seconds=arm["final_evaluation"]["cpu_seconds"],
                    )
                rows.append(row)
    with (output / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only or execute an existing immutable protocol after allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four single-thread numerical workers.")
    if args.prepare_only:
        freeze_protocol(args.output)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    records: list[dict[str, Any]] = []
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(execute_case, jobs):
            records.append(record)
            write_scoreboard(args.output, records)
            print(record["target_id"], record["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete"
            if all(record["status"] == "complete" for record in records)
            else "complete_with_failed_arms",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(record) for record in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
