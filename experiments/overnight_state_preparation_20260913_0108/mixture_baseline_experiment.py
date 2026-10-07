# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Strong objective-matched weighted Adam and adaptive online development controls."""

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
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_diagnostics as diagnostics
from experiments.overnight_state_preparation_20260913_0108 import (
    channels,
    ensemble_deletion,
    runner,
    trajectory_mixture,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    ensemble_structural_experiment as reference,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "overnight.mean2_stochastic_baselines.v1"
TARGETS = reference.TARGETS
METHODS = (
    "fixed_mean2_weighted_adam",
    "fixed_mean2_adaptive_online",
    "structural_mean2_weighted_adam",
    "structural_mean2_adaptive_online",
)
CPU_SCOPE = (
    "sum of stage optimization clocks including setup/sampling/intermediate checkpoints, plus exact ranking/score "
    "operation clocks; excludes stage-final/attempt writes, circuit bank, outer orchestration, "
    "custody and final scoring"
)
COUNT_KEYS = (
    "attempted_updates",
    "completed_updates",
    "attempted_trajectory_paths",
    "completed_trajectory_paths",
    "attempted_trajectory_direction_passes",
    "completed_trajectory_direction_passes",
    "attempted_local_operator_applications",
    "completed_local_operator_applications",
    "sampling_attempts",
)


@dataclass(frozen=True)
class Config:
    """Frozen strong-baseline settings with smaller positive engineering budgets."""

    total_updates: int = 1500
    warmup: int = 150
    rounds: int = 15
    shortlist: int = 2
    repair: int = 25
    checkpoint_every: int = 10
    paths_per_channel: int = 32
    protected: int = 18
    learning_rate: float = 0.02
    online_warmup: int = 25
    online_rate: float = 0.7
    acceptance_tolerance: float = 1e-10
    ranking_quantum: float = 1e-12

    def __post_init__(self) -> None:
        """Reject settings that cannot fund a complete balanced round and polishing.

        Raises:
            ValueError: If counts, rates, tolerances or the aggregate budget are invalid.
        """
        integers = (
            self.total_updates,
            self.warmup,
            self.rounds,
            self.shortlist,
            self.repair,
            self.checkpoint_every,
            self.paths_per_channel,
            self.online_warmup,
        )
        reals = (self.learning_rate, self.online_rate, self.acceptance_tolerance, self.ranking_quantum)
        if (
            any(type(value) is not int or value < 1 for value in integers)
            or type(self.protected) is not int
            or self.protected < 0
            or self.paths_per_channel < 2
            or self.total_updates <= self.warmup + (self.shortlist + 1) * self.repair
            or any(isinstance(value, bool) or not np.isfinite(value) or value <= 0 for value in reals)
        ):
            msg = "Invalid complete-round stochastic baseline budget or numerical setting."
            raise ValueError(msg)


class Store:
    """Keep compact current optimizer memory and authenticated exact-operation caches."""

    def __init__(self, root: Path, target: NDArray, config: Config, seed: int) -> None:
        """Bind one target and reuse only the public exact scoring/circuit-bank helpers."""
        self.root, self.target, self.config, self.seed = root, target.copy(), config, seed
        self.exact = reference.Store(root, target, reference.Config())

    def circuit(self, circuit: ParameterizedCircuit) -> str:
        """Return the verified content identity of a stored complete physical circuit."""
        return self.exact.circuit(circuit)

    def restore(self, key: str) -> ParameterizedCircuit:
        """Restore all gate fields, including fixed parameters and native identifiers.

        Raises:
            ValueError: If the circuit-bank payload disagrees with its filename.
        """
        payload = pilot.read_record(self.root / "circuits" / f"{key}.json")
        if pilot.digest(payload) != key:
            msg = "Mixture circuit bank identity changed."
            raise ValueError(msg)
        gates = []
        for value in cast("list[dict[str, Any]]", payload["gates"]):
            fields: dict[str, Any] = {
                **value,
                "sites": tuple(value["sites"]),
                "fixed_params": tuple(value["fixed_params"]),
            }
            gates.append(ParameterizedGate(**fields))
        return ParameterizedCircuit(cast("int", payload["num_qubits"]), gates, cast("int", payload["num_params"]))

    def initial(self, circuit: ParameterizedCircuit, theta: NDArray) -> dict[str, Any]:
        """Initialize the historical independent moment memory for unique scalar gates.

        Raises:
            ValueError: If unique parameters or stable integer origins are absent.
        """
        indices = [gate.param_index for gate in circuit.gates]
        origins = [gate.logical_gate_id for gate in circuit.gates]
        if (
            any(type(value) is not int for value in (*indices, *origins))
            or sorted(indices) != list(range(circuit.num_params))
            or len(set(origins)) != len(origins)
            or theta.shape != (circuit.num_params,)
            or not np.isrealobj(theta)
            or not np.all(np.isfinite(theta))
        ):
            msg = "Strong structural controls require unique scalar parameters and stable integer origins."
            raise ValueError(msg)
        return {
            "circuit_hash": self.circuit(circuit),
            "theta": theta.tolist(),
            # Search identities are initial physical positions, matching the exact
            # ensemble-structural comparator. Logical/native IDs remain in the circuit.
            "origins": list(range(len(circuit.gates))),
            "first": np.zeros_like(theta).tolist(),
            "second": np.zeros_like(theta).tolist(),
            "counts": np.zeros(len(theta), dtype=int).tolist(),
            "updates": 0,
        }

    def deleted(self, state: dict[str, Any], index: int) -> dict[str, Any]:
        """Delete one gate/window and preserve surviving moments, counts and Adam age."""
        circuit = self.restore(state["circuit_hash"])
        compact, theta = delete_gates(circuit, np.asarray(state["theta"]), [index])
        retained = sorted(gate.param_index for position, gate in enumerate(circuit.gates) if position != index)
        result = copy.deepcopy(state)
        for key in ("first", "second", "counts"):
            result[key] = [state[key][position] for position in retained]
        result.update(circuit_hash=self.circuit(compact), theta=theta.tolist())
        result["origins"].pop(index)
        return result

    def operation(
        self,
        name: str,
        state: dict[str, Any],
        *,
        ranking: bool = False,
        profiles: Sequence[str] = ensemble_deletion.MEAN2,
    ) -> dict[str, Any]:
        """Return a cached exact operation and its attempted/completed component work."""
        return self.exact.operation(
            name, self.restore(state["circuit_hash"]), np.asarray(state["theta"]), ranking=ranking, profiles=profiles
        )

    def train(self, name: str, start: dict[str, Any], cap: int, epoch: int, method: str) -> dict[str, Any]:
        """Run or resume a deterministic stream of weighted updates with full memory.

        Caught failures charge actual attempted/completed updates and full reserved
        per-call trajectory/operator budgets when a backend call is interrupted.
        A durable running marker rejects unobserved process death on resume.

        Raises:
            ValueError: If method, budget, cache identity or unfinished-death custody is invalid.
            FloatingPointError: If an update produces nonfinite memory or parameters.
        """
        if (
            method not in {"weighted_adam", "adaptive_online"}
            or type(cap) is not int
            or cap < 1
            or type(epoch) is not int
            or epoch < 0
            or epoch + cap > trajectory_mixture.COUNTER_LIMIT
        ):
            msg = "Invalid mixture stage method or budget."
            raise ValueError(msg)
        signature = pilot.digest([
            VERSION,
            start,
            cap,
            epoch,
            method,
            self.seed,
            asdict(self.config),
            hashlib.sha256(self.target.tobytes()).hexdigest(),
        ])
        root = self.root / "stages" / name
        root.mkdir(parents=True, exist_ok=True)
        final, partial, attempts_path = (
            root / filename for filename in ("TRAINING.json", "PARTIAL.json", "ATTEMPTS.json")
        )
        attempts: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(attempts_path))
            if attempts_path.exists()
            else {"signature": signature, "attempts": []}
        )
        if attempts["signature"] != signature or any(entry["status"] == "running" for entry in attempts["attempts"]):
            msg = "Stage identity changed or killed work requires an external resource ledger."
            raise ValueError(msg)
        if final.exists():
            result = cast("dict[str, Any]", pilot.read_record(final))
            if result["signature"] != signature or result["attempts_checksum"] != pilot.digest(attempts):
                msg = "Cached mixture stage or physical attempt ledger changed."
                raise ValueError(msg)
            return result
        progress: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(partial))
            if partial.exists()
            else {
                "signature": signature,
                "state": copy.deepcopy(start),
                "completed": 0,
                "trace": [],
                "curve": [
                    {"circuit_hash": start["circuit_hash"], "theta": start["theta"], "cpu_seconds": 0.0, "updates": 0}
                ],
            }
        )
        if progress["signature"] != signature:
            msg = "Partial mixture stage identity changed."
            raise ValueError(msg)
        prior_cpu = sum(entry["cpu_seconds"] for entry in attempts["attempts"])
        entry: dict[str, Any] = {"status": "running", **dict.fromkeys(COUNT_KEYS, 0)}
        attempts["attempts"].append(entry)
        pilot.write_record(attempts_path, attempts)
        cpu, wall = time.process_time(), time.perf_counter()

        def advance() -> None:
            """Advance from the last durable memory and preserve every completed update."""
            engine = StatevectorBackend(self.restore(start["circuit_hash"]), self.target)
            sampler = trajectory_mixture.MixtureSampler(engine.circuit)
            state = progress["state"]
            theta, first, second = (np.asarray(state[key], dtype=float) for key in ("theta", "first", "second"))
            counts = np.asarray(state["counts"], dtype=np.int64)
            while progress["completed"] < cap:
                stream_index = epoch + progress["completed"]
                step = state["updates"] + 1
                mode = (
                    "gradient"
                    if method == "weighted_adam"
                    else "online"
                    if step <= self.config.online_warmup
                    else "adaptive_online"
                )
                entry["sampling_attempts"] += 1
                batch = sampler.ensemble(self.config.paths_per_channel, seed=self.seed, update_index=stream_index)
                work = batch.resource_counts(mode)
                entry["attempted_updates"] += 1
                for suffix, key in (
                    ("trajectory_paths", "paths"),
                    ("trajectory_direction_passes", "directional_passes"),
                    ("local_operator_applications", "total_local_operator_applications"),
                ):
                    entry[f"attempted_{suffix}"] += work[key]
                if method == "weighted_adam":
                    signal = batch.gradient(engine, theta)[0]
                    theta, first, second = adam.adam_step(
                        theta, signal, first, second, step, adam.AdamConfig(learning_rate=self.config.learning_rate)
                    )
                elif mode == "online":
                    theta, signal = batch.online_update(engine, theta, self.config.online_rate)
                else:
                    theta, signal, second, counts = batch.adaptive_online_update(
                        engine, theta, self.config.learning_rate, second, counts
                    )
                if not all(np.all(np.isfinite(value)) for value in (theta, first, second, signal)):
                    msg = "Nonfinite objective-matched stochastic update."
                    raise FloatingPointError(msg)
                entry["completed_updates"] += 1
                for suffix, key in (
                    ("trajectory_paths", "paths"),
                    ("trajectory_direction_passes", "directional_passes"),
                    ("local_operator_applications", "total_local_operator_applications"),
                ):
                    entry[f"completed_{suffix}"] += work[key]
                state.update(
                    theta=theta.tolist(),
                    first=first.tolist(),
                    second=second.tolist(),
                    counts=counts.tolist(),
                    updates=step,
                )
                progress["completed"] += 1
                progress["trace"].append({
                    "step": step,
                    "stream_index": stream_index,
                    "phase": mode,
                    "ensemble_digest": diagnostics.ensemble_digest(batch.maps, batch.weights),
                    "signal_norm": float(np.linalg.norm(signal)),
                    "directional_passes": work["directional_passes"],
                })
                progress["curve"].append({
                    "circuit_hash": state["circuit_hash"],
                    "theta": state["theta"],
                    "cpu_seconds": prior_cpu + time.process_time() - cpu,
                    "updates": progress["completed"],
                })
                if progress["completed"] % self.config.checkpoint_every == 0:
                    pilot.write_record(partial, progress)

        try:
            advance()
            entry["status"] = "complete"
        except BaseException as error:
            entry.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                failure=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            entry.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
            pilot.write_record(attempts_path, attempts)
        total_cpu = sum(row["cpu_seconds"] for row in attempts["attempts"])
        progress["curve"].append({**progress["curve"][-1], "cpu_seconds": total_cpu})
        result = {
            **progress,
            "status": "complete",
            "epoch_end": epoch + cap,
            "cpu_seconds": total_cpu,
            "wall_seconds": sum(row["wall_seconds"] for row in attempts["attempts"]),
            **{key: sum(row[key] for row in attempts["attempts"]) for key in COUNT_KEYS},
            "replayed_completed_updates": sum(row["completed_updates"] for row in attempts["attempts"]) - cap,
            "attempts_checksum": pilot.digest(attempts),
            "cpu_scope": CPU_SCOPE,
        }
        pilot.write_record(final, result)
        return result


def _stage_cost(name: str, result: dict[str, Any], reserved: int) -> dict[str, Any]:
    """Return compact stage provenance and physically charged work without curves."""
    return {
        "role": name,
        "kind": "stochastic",
        "reserved_updates": reserved,
        "artifact_checksum": pilot.digest(result),
        "cpu_seconds": result["cpu_seconds"],
        **{key: result[key] for key in COUNT_KEYS},
    }


def _operation_cost(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep exact-ranking/score costs separate from stochastic updates."""
    return {"role": name, **{key: value for key, value in result.items() if key != "value"}}


def totals(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Return separate update, trajectory, density and physical-call accounting."""
    return {
        "cpu_seconds": sum(row["cpu_seconds"] for row in ledger),
        "nominal_charged_updates": sum(row.get("reserved_updates", 0) for row in ledger),
        **{key: sum(row.get(key, 0) for row in ledger) for key in COUNT_KEYS},
        "exact_ranking_channel_calls": sum(
            row.get("individual_channel_calls", 0) for row in ledger if row["kind"] == "ranking"
        ),
        "exact_forward_channel_calls": sum(
            row.get("individual_channel_calls", 0) for row in ledger if row["kind"] == "score"
        ),
        "exact_gradient_channel_calls": 0,
        "density_direction_passes": sum(row.get("density_direction_passes", 0) for row in ledger),
        "physical_attempted_density_channel_calls": sum(
            row.get("physical_attempted_channel_calls", 0) for row in ledger
        ),
        "physical_completed_density_channel_calls": sum(
            row.get("physical_completed_channel_calls", 0) for row in ledger
        ),
        "physical_completed_density_direction_passes": sum(
            row.get("physical_completed_direction_passes", 0) for row in ledger
        ),
    }


def _curve(stage: dict[str, Any], offset: float) -> list[dict[str, Any]]:
    """Return only deployable theta/circuit identities on the cumulative CPU clock."""
    return [{**point, "cpu_seconds": offset + point["cpu_seconds"]} for point in stage["curve"]]


def repair_branch(
    store: Store, candidate: dict[str, Any], name: str, epoch: int, method: str, index: int | None, origin: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cache complete or failed branch decisions so resume cannot retry a failed candidate."""
    signature = pilot.digest([
        VERSION,
        candidate,
        name,
        epoch,
        method,
        asdict(store.config),
        store.seed,
        hashlib.sha256(store.target.tobytes()).hexdigest(),
    ])
    path = store.root / "branches" / f"{name}.json"
    if path.exists():
        saved = cast("dict[str, Any]", pilot.read_record(path))
        if saved["signature"] != signature or any(
            hashlib.sha256((store.root / key).read_bytes()).hexdigest() != value
            for key, value in saved["dependencies"].items()
        ):
            msg = "Cached repair branch or its physical work changed."
            raise ValueError(msg)
        return saved["branch"], saved["ledger"]
    branch: dict[str, Any] = {"index": index, "origin": origin, "name": name}
    ledger: list[dict[str, Any]] = []
    try:
        repaired = store.train(name, candidate, store.config.repair, epoch, method)
        ledger.append(_stage_cost(name, repaired, store.config.repair))
        scored = store.operation(name + "_score", repaired["state"])
        ledger.append(_operation_cost(name + "_score", scored))
        branch.update(
            status="complete",
            state=repaired["state"],
            fidelity=scored["value"]["mean_fidelity"],
            training_checksum=pilot.digest(repaired),
        )
    except (ArithmeticError, RuntimeError) as error:
        rows = cast(
            "list[dict[str, Any]]", pilot.read_record(store.root / "stages" / name / "ATTEMPTS.json")["attempts"]
        )
        if not ledger:
            failed = {
                "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
                **{key: sum(row[key] for row in rows) for key in COUNT_KEYS},
            }
            ledger.append(_stage_cost(name, failed, store.config.repair))
        score_attempts = store.root / "operations" / f"{name}_score.attempts.json"
        if score_attempts.exists():
            rows = cast("list[dict[str, Any]]", pilot.read_record(score_attempts)["attempts"])
            ledger.append({
                "role": name + "_score",
                "kind": "score",
                "status": "failed",
                "individual_channel_calls": 2,
                "density_direction_passes": 2,
                "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
                "physical_attempted_channel_calls": sum(row["attempted_channel_calls"] for row in rows),
                "physical_completed_channel_calls": sum(row["completed_channel_calls"] for row in rows),
                "physical_completed_direction_passes": sum(row["completed_channel_calls"] for row in rows),
            })
        branch.update(status="failed", failure=f"{type(error).__name__}: {error}", reserved_updates=store.config.repair)
    dependencies = {}
    for child in (
        store.root / "stages" / name / "ATTEMPTS.json",
        store.root / "operations" / f"{name}_score.attempts.json",
    ):
        if child.exists():
            dependencies[str(child.relative_to(store.root))] = hashlib.sha256(child.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    pilot.write_record(path, {"signature": signature, "branch": branch, "ledger": ledger, "dependencies": dependencies})
    return branch, ledger


def train_method(store: Store, circuit: ParameterizedCircuit, theta: NDArray, label: str) -> dict[str, Any]:
    """Spend the full update allowance, charging all repairs before final polishing.

    Raises:
        ValueError: If the method label is unsupported.
    """
    if label not in METHODS:
        msg = "Unknown objective-matched baseline label."
        raise ValueError(msg)
    config = store.config
    method = "weighted_adam" if label.endswith("weighted_adam") else "adaptive_online"
    start = store.initial(circuit, theta)
    if label.startswith("fixed"):
        stage = store.train(label, start, config.total_updates, 0, method)
        ledger = [_stage_cost(label, stage, config.total_updates)]
        return {
            "status": "complete",
            "state": stage["state"],
            "curve": _curve(stage, 0),
            "rounds": [],
            "ledger": ledger,
            "stop_reason": "fixed_full_cap",
            **totals(ledger),
        }
    warm = store.train(label + "_warm", start, config.warmup, 0, method)
    ledger = [_stage_cost(label + "_warm", warm, config.warmup)]
    curve, rounds = _curve(warm, 0), []
    current, epoch, stop_reason = warm["state"], config.warmup, "max_rounds"
    for number in range(config.rounds):
        eligible = [index for index, origin in enumerate(current["origins"]) if origin >= config.protected]
        count = min(config.shortlist, len(eligible))
        if not count or config.total_updates - totals(ledger)["nominal_charged_updates"] <= (count + 1) * config.repair:
            stop_reason = "no_eligible_gates" if not count else "complete_round_budget"
            break
        prefix = f"{label}_round{number}"
        ranked = store.operation(prefix + "_ranking", current, ranking=True)
        ledger.append(_operation_cost(prefix + "_ranking", ranked))
        incoming = ranked["value"]["mean_fidelity"]
        scores = ranked["value"]["mean_scores"]
        ranking = sorted(
            eligible, key=lambda index: (-round(scores[index] / config.ranking_quantum), current["origins"][index])
        )
        branches: list[dict[str, Any]] = []
        for index in (None, *ranking[:count]):
            candidate = copy.deepcopy(current) if index is None else store.deleted(current, index)
            name = prefix + ("_retain" if index is None else f"_delete{current['origins'][index]}")
            branch, costs = repair_branch(
                store, candidate, name, epoch, method, index, None if index is None else current["origins"][index]
            )
            ledger.extend(costs)
            branches.append(branch)
        selected = select_branch(branches, incoming, config.acceptance_tolerance)
        if selected is not None:
            current = branches[selected]["state"]
        epoch += config.repair
        rounds.append({
            "round": number + 1,
            "incoming_fidelity": incoming,
            "branches": branches,
            "selected_branch": selected,
            "ranking": ranking,
            "epoch_end": epoch,
        })
        curve.append({
            "circuit_hash": current["circuit_hash"],
            "theta": current["theta"],
            "cpu_seconds": totals(ledger)["cpu_seconds"],
            "updates": current["updates"],
        })
        if selected in {None, 0}:
            stop_reason = "no_repaired_deletion_advantage"
            break
    remaining = config.total_updates - totals(ledger)["nominal_charged_updates"]
    polished = store.train(label + "_polish", current, remaining, epoch, method)
    curve.extend(_curve(polished, totals(ledger)["cpu_seconds"]))
    ledger.append(_stage_cost(label + "_polish", polished, remaining))
    return {
        "status": "complete",
        "state": polished["state"],
        "curve": curve,
        "ledger": ledger,
        "rounds": rounds,
        "stop_reason": stop_reason,
        **totals(ledger),
    }


def select_branch(branches: list[dict[str, Any]], incoming: float, tolerance: float) -> int | None:
    """Retain older memory on ties and require deletion to beat the repaired retain score."""
    retained = branches[0]["fidelity"] if branches[0]["status"] == "complete" else -np.inf
    selected, best = (0 if retained > incoming + tolerance else None), max(incoming, retained)
    for index in range(1, len(branches)):
        branch = branches[index]
        if branch["status"] == "complete" and branch["fidelity"] > best + tolerance:
            selected, best = index, branch["fidelity"]
    return selected


def physical_work(root: Path, label: str) -> dict[str, Any]:
    """Collect all completed, failed and replayed work even when no method result exists."""
    result: dict[str, Any] = {
        "accounting_status": "known",
        "training_cpu_seconds": 0.0,
        "evaluation_cpu_seconds": 0.0,
        "attempted_density_channel_calls": 0,
        "completed_density_channel_calls": 0,
        "completed_density_direction_passes": 0,
        **dict.fromkeys(COUNT_KEYS, 0),
        "attempt_checksums": {},
    }
    paths = [(path, "stochastic") for path in sorted((root / "stages").glob(f"{label}*/ATTEMPTS.json"))]
    paths.extend(
        (path, "ranking" if path.name.endswith("_ranking.attempts.json") else "score")
        for path in sorted((root / "operations").glob(f"{label}*.attempts.json"))
    )
    paths.extend((path, "evaluation") for path in sorted((root / "operations").glob(f"*_{label}.attempts.json")))
    for path, kind in paths:
        record = pilot.read_record(path)
        result["attempt_checksums"][str(path.relative_to(root))] = pilot.digest(record)
        for row in cast("list[dict[str, Any]]", record["attempts"]):
            if row["status"] == "running":
                result["accounting_status"] = "unknown_requires_external_ledger"
                continue
            result["evaluation_cpu_seconds" if kind == "evaluation" else "training_cpu_seconds"] += row["cpu_seconds"]
            if kind == "stochastic":
                for key in COUNT_KEYS:
                    result[key] += row[key]
            elif kind != "evaluation":
                result["attempted_density_channel_calls"] += row["attempted_channel_calls"]
                result["completed_density_channel_calls"] += row["completed_channel_calls"]
                result["completed_density_direction_passes"] += row["completed_channel_calls"] * (
                    2 if kind == "ranking" else 1
                )
    return result


def score_endpoint(store: Store, name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Audit all unchanged final conditions and physical/native resource counts."""
    scored = store.operation(name, state, profiles=channels.NOISE_IDS)
    circuit, theta = store.restore(state["circuit_hash"]), np.asarray(state["theta"])
    return {
        "fidelities": scored["value"]["fidelities"],
        "evaluation": scored,
        "logical_counts": runner.counts(circuit),
        "ballarin_counts": runner.counts(materialize_ballarin_circuit(circuit, theta).circuit),
    }


def common_cpu_panel(store: Store, arms: dict[str, Any]) -> dict[str, Any]:
    """Rescore last available iterates on one shared physical CPU basis, never best checkpoints."""
    if any(arm["status"] != "complete" for arm in arms.values()):
        return {"status": "unavailable_incomplete_panel", "cpu_seconds": None, "arms": {}}
    if any(arm.get("physical_work", {}).get("accounting_status", "known") != "known" for arm in arms.values()):
        return {"status": "unavailable_unknown_physical_cpu", "cpu_seconds": None, "arms": {}}
    cutoff = min(arm["cpu_seconds"] for arm in arms.values())
    selected = {}
    for label, arm in arms.items():
        points = [point for point in arm["curve"] if point["cpu_seconds"] <= cutoff]
        if not points:
            return {"status": "unavailable_no_checkpoint", "cpu_seconds": cutoff, "arms": {}}
        point = points[-1]
        selected[label] = {
            "status": "complete",
            "point": point,
            **score_endpoint(store, "common_" + label, point),
            "comparison": "common_cpu_last_available_final_iterate",
        }
    return {"status": "complete", "cpu_seconds": cutoff, "arms": selected, "cpu_scope": CPU_SCOPE}


def run_case(
    root: Path,
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: NDArray,
    config: Config,
    *,
    target_id: str = "engineering",
    replica: int = 11,
    protocol_checksum: str | None = None,
    reference_case: dict[str, Any] | None = None,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    """Run four independent arms and score final/common-CPU iterates with durable failures."""
    signature = pilot.digest([
        VERSION,
        pilot.circuit_record(circuit),
        theta.tolist(),
        hashlib.sha256(target.tobytes()).hexdigest(),
        asdict(config),
        target_id,
        replica,
        protocol_checksum,
        None if reference_case is None else pilot.digest(reference_case),
    ])
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature or any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != value
            for name, value in result["artifact_hashes"].items()
        ):
            msg = "Completed mixture case or its durable artifacts changed."
            raise ValueError(msg)
        return result
    seed = runner.seed_for("mean2_stochastic_training", target_id, replica)
    store = Store(root, target, config, seed)
    arms: dict[str, Any] = {}
    # Rotate order by target to reduce systematic first/last position effects.
    offset = TARGETS.index(target_id) % len(METHODS) if target_id in TARGETS else 0
    for label in (*METHODS[offset:], *METHODS[:offset]):
        arm: dict[str, Any] = {}
        try:
            arm = train_method(store, circuit, theta, label)
            arm.update(score_endpoint(store, "final_" + label, arm["state"]))
            arm.update(
                comparison="full_update_cap_heterogeneous_compute",
                cpu_scope=CPU_SCOPE,
                selected_optimizer_updates=arm["state"]["updates"],
                memory_rule="preserved_surviving_moments_occurrence_counts_and_Adam_age",
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            arm.update(status="failed", failure=f"{type(error).__name__}: {error}", partial_artifacts=str(root))
        arms[label] = arm
    if reference_case is not None:
        if reference_root is None:
            msg = "Reference circuits require the matching reference case root."
            raise ValueError(msg)
        bank = Store(reference_root, target, config, seed)
        for label in reference.METHODS:
            arm = copy.deepcopy(reference_case["arms"][label])
            if arm["status"] == "complete":
                for key in {point["circuit_hash"] for point in arm["curve"]} | {arm["circuit_hash"]}:
                    store.circuit(bank.restore(key))
                arm["comparison"] = "objective_matched_deterministic_reference_heterogeneous_call_cap"
                arm["cpu_scope"] = CPU_SCOPE
            arms[label] = arm
    unchanged = score_endpoint(store, "unchanged_start", store.initial(circuit, theta))
    unchanged["comparison"] = "unchanged_construction_reference_no_additional_training"
    panel = common_cpu_panel(store, arms)
    for label in METHODS:
        arms[label]["physical_work"] = physical_work(root, label)
    result: dict[str, Any] = {
        "signature": signature,
        "target_id": target_id,
        "replica": replica,
        "status": "complete" if panel["status"] == "complete" else "complete_with_failed_arms",
        "protocol_checksum": protocol_checksum,
        "arms": arms,
        "common_cpu": panel,
        "unchanged_start": unchanged,
        "warmup_rule": "executed_and_charged_independently_in_all_four_arms",
        "reference_case_checksum": None if reference_case is None else pilot.digest(reference_case),
        "artifact_hashes": {
            str(child.relative_to(root)): hashlib.sha256(child.read_bytes()).hexdigest()
            for child in sorted(root.rglob("*.json"))
            if child != path
        },
    }
    pilot.write_record(path, result)
    return result


def numerical_sources() -> dict[str, str]:
    """Freeze both reused numerical helpers and the new mixture driver/protocol."""
    result = reference.numerical_sources()
    for name in (
        "trajectory_mixture.py",
        "MIXTURE_PROTOCOL.md",
        "mixture_baseline_experiment.py",
        "MIXTURE_BASELINE_PROTOCOL.md",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = runner.REPO / "experiments/state_preparation_trajectory_diagnostics.py"
    result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def freeze_protocol(output: Path, reference_root: Path, unmatched_root: Path | None = None) -> dict[str, Any]:
    """Authenticate original starts, objective-matched references and historical controls."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    selected = [
        source
        for source in cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
        if source["target_id"] in TARGETS and source["replica"] == 11
    ]
    ref = cast("dict[str, Any]", pilot.read_record(reference_root / "PROTOCOL.json"))
    if (
        len(selected) != len(TARGETS)
        or {source["target_id"] for source in selected} != set(TARGETS)
        or ref["version"] != reference.VERSION
        or ref["config"] != asdict(reference.Config())
        or ref["phase"] != "development"
        or ref["noise_ids"] != list(ensemble_deletion.MEAN2)
        or pilot.digest(selected) != pilot.digest(ref["starts"])
    ):
        msg = "Expected complete original starts and the frozen mean-two structural reference protocol."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if (
            circuit.num_qubits != 6
            or runner.counts(circuit)["one_qubit"] != 78
            or runner.counts(circuit)["two_qubit"] != 30
            or sorted(gate.param_index for gate in circuit.gates) != list(range(108))
        ):
            msg = "Mixture science requires the original q6 78+30 gate construction."
            raise ValueError(msg)
    unmatched = {}
    if unmatched_root is not None:
        for source in selected:
            child = unmatched_root / f"{source['target_id']}__11" / "RESULT.json"
            unmatched[str(child)] = hashlib.sha256(child.read_bytes()).hexdigest()
    protocol: dict[str, Any] = {
        "version": VERSION,
        "phase": "development",
        "targets": list(TARGETS),
        "replica": 11,
        "methods": list(METHODS),
        "config": asdict(Config()),
        "noise_ids": list(ensemble_deletion.MEAN2),
        "starts": selected,
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "reference_root": str(reference_root),
        "reference_protocol_checksum": pilot.digest(ref),
        "unmatched_reference_files": unmatched,
        "cpu_scope": CPU_SCOPE,
        "comparison": "common CPU final available iterates; 1500 stochastic updates are not 3000 density calls",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen mixture protocol differs; use a new output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Reject changed sources, benchmarks, settings, historical files or original starts."""
    ref = cast("dict[str, Any]", pilot.read_record(Path(protocol["reference_root"]) / "PROTOCOL.json"))
    original = pilot.read_record(runner.OUTPUT / "starts.json")
    expected = [
        source
        for source in cast("list[dict[str, Any]]", original["starts"])
        if source["target_id"] in TARGETS and source["replica"] == 11
    ]
    if (
        protocol["version"] != VERSION
        or protocol["phase"] != "development"
        or protocol["targets"] != list(TARGETS)
        or protocol["replica"] != 11
        or protocol["methods"] != list(METHODS)
        or protocol["config"] != asdict(Config())
        or protocol["noise_ids"] != list(ensemble_deletion.MEAN2)
        or protocol["source_hashes"] != numerical_sources()
        or protocol["benchmark_hashes"] != runner.custody()["benchmark_hashes"]
        or protocol["source_starts_sha256"] != hashlib.sha256((runner.OUTPUT / "starts.json").read_bytes()).hexdigest()
        or protocol["starts"] != expected
        or protocol["reference_protocol_checksum"] != pilot.digest(ref)
        or any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() != value
            for path, value in protocol["unmatched_reference_files"].items()
        )
    ):
        msg = "Mixture baseline scientific custody changed."
        raise ValueError(msg)
    return ref


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one authenticated q6 development case and retain unmatched historical references."""
    protocol, source = job["protocol"], job["source"]
    ref_protocol = verify_protocol(protocol)
    if source not in protocol["starts"]:
        msg = "Mixture case is absent from the frozen original panel."
        raise ValueError(msg)
    target = load_target(6, source["target_id"]).state_vector_copy()
    circuit, theta = pilot.restore_circuit(source["candidate"]["circuit"]), np.asarray(source["candidate"]["theta"])
    ref_root = Path(protocol["reference_root"]) / f"{source['target_id']}__11"
    ref = cast("dict[str, Any]", pilot.read_record(ref_root / "RESULT.json"))
    expected = pilot.digest({
        "version": reference.VERSION,
        "circuit": pilot.circuit_record(circuit),
        "theta": theta.tolist(),
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "config": ref_protocol["config"],
        "target_id": source["target_id"],
        "replica": 11,
        "protocol_checksum": pilot.digest(ref_protocol),
    })
    if ref["signature"] != expected or ref["status"] != "complete":
        msg = "Mixture comparison requires the matching complete mean-two reference case."
        raise ValueError(msg)
    result = run_case(
        Path(job["root"]),
        circuit,
        theta,
        target,
        Config(**protocol["config"]),
        target_id=source["target_id"],
        replica=11,
        protocol_checksum=pilot.digest(protocol),
        reference_case=ref,
        reference_root=ref_root,
    )
    historical = {}
    for filename, checksum in protocol["unmatched_reference_files"].items():
        if Path(filename).parent.name != f"{source['target_id']}__11":
            continue
        value = cast("dict[str, Any]", pilot.read_record(Path(filename)))
        if value["target_id"] != source["target_id"] or value["replica"] != 11:
            msg = "Historical reference target identity differs."
            raise ValueError(msg)
        for label in ("local_adam", "local_lbfgs", "noiseless_adam", "noiseless_lbfgs", "mean2_lbfgs"):
            historical[label] = {
                "comparison": "unmatched_historical_fixed_1100_individual_channel_call_cap",
                "source_sha256": checksum,
                "source_file": filename,
                "record": value["arms"][label],
            }
    verify_protocol(protocol)
    # Keep context separate from the authenticated numerical case and its byte-preserving resume.
    context: dict[str, Any] = {"case_checksum": pilot.digest(result), "historical_unmatched": historical}
    context_path = Path(job["root"]) / "REFERENCES.json"
    if context_path.exists():
        if pilot.read_record(context_path) != context:
            msg = "Retained historical context changed."
            raise ValueError(msg)
    else:
        pilot.write_record(context_path, context)
    return result


def write_scoreboard(output: Path, records: list[dict[str, Any]]) -> None:
    """Report every condition for full-cap and common-CPU panels, including failed arms."""
    rows = []
    for record in sorted(records, key=operator.itemgetter("target_id")):
        panels = [
            ("full_cap", record["arms"]),
            ("common_cpu", record["common_cpu"]["arms"]),
            ("unchanged", {"unchanged_start": record["unchanged_start"]}),
        ]
        context_path = output / f"{record['target_id']}__{record['replica']}" / "REFERENCES.json"
        if context_path.exists():
            context = cast("dict[str, Any]", pilot.read_record(context_path))
            if context["case_checksum"] != pilot.digest(record):
                msg = "Historical scoreboard context belongs to another numerical case."
                raise ValueError(msg)
            historical = {
                label: {
                    **value["record"].get("final_score", {}),
                    "status": value["record"].get("status", "complete"),
                    "comparison": value["comparison"],
                }
                for label, value in context["historical_unmatched"].items()
            }
            panels.append(("historical_unmatched", historical))
        for panel, arms in panels:
            for label, arm in arms.items():
                rows.extend(
                    {
                        "target_id": record["target_id"],
                        "replica": record["replica"],
                        "panel": panel,
                        "method": label,
                        "noise_id": noise,
                        "status": arm.get("status", "complete"),
                        "fidelity": arm.get("fidelities", {}).get(noise),
                        "failure": arm.get("failure"),
                        "comparison": arm.get("comparison"),
                        "cpu_seconds": arm.get("cpu_seconds"),
                        "common_cpu_seconds": record["common_cpu"]["cpu_seconds"] if panel == "common_cpu" else None,
                        "logical_one_qubit": arm.get("logical_counts", {}).get("one_qubit"),
                        "logical_two_qubit": arm.get("logical_counts", {}).get("two_qubit"),
                        "ballarin_two_qubit": arm.get("ballarin_counts", {}).get("two_qubit"),
                        **{
                            key: arm.get(key)
                            for key in (
                                *COUNT_KEYS,
                                "nominal_charged_updates",
                                "density_direction_passes",
                                "exact_ranking_channel_calls",
                                "exact_forward_channel_calls",
                            )
                        },
                    }
                    for noise in channels.NOISE_IDS
                )
    with (output / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare only or run an already frozen objective-matched four-target panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--unmatched-root", type=Path, default=runner.OUTPUT / "ensemble-original108")
    parser.add_argument("--workers", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four single-thread numerical workers.")
    if args.prepare_only:
        if args.reference_root is None:
            parser.error("Preparation requires the frozen ensemble-structural --reference-root.")
        freeze_protocol(args.output, args.reference_root, args.unmatched_root)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {"protocol": protocol, "source": source, "root": str(args.output / f"{source['target_id']}__11")}
        for source in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute_case, jobs):
            records.append(result)
            write_scoreboard(args.output, records)
            print(result["target_id"], result["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete"
            if all(record["status"] == "complete" for record in records)
            else "complete_with_failed_arms",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(record) for record in records],
        },
    )
    return 0 if all(record["status"] == "complete" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
