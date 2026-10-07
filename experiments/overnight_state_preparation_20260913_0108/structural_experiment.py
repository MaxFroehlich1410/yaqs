# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen development driver for exact deletion/repair and matched L-BFGS.

Preparation only writes an authenticated protocol. An explicit --run executes
that already frozen protocol. Every branch receives fresh curvature memory;
the fixed control retains its warmup memory. All twelve benchmark conditions
are evaluated exactly on final circuits, outside the training-call budget.
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
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, optimizers, runner, structural

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

VERSION = "overnight.exact_structural_lbfgs.v1"
DEVELOPMENT_TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1", "random_mps_bond2")


@dataclass(frozen=True)
class Config:
    """Predeclare every numerical decision for one paired experiment."""

    full_evaluations: int = 1500
    warmup_evaluations: int = 150
    max_rounds: int = 15
    shortlist: int = 2
    repair_evaluations: int = 25
    history: int = 10
    checkpoint_every: int = 10
    protected_initial_gates: int = 18

    def __post_init__(self) -> None:
        """Reject budgets that cannot reserve even one balanced search round."""
        values = asdict(self)
        if any(type(v) is not int or v < int(k != "protected_initial_gates") for k, v in values.items()):
            msg = "Structural experiment configuration must contain valid integer budgets."
            raise ValueError(msg)
        if self.search_refinement_cap < (self.shortlist + 1) * self.repair_evaluations:
            msg = "Full budget cannot cover warmup, reserved ranking/scoring and a balanced repair round."
            raise ValueError(msg)

    @property
    def search_refinement_cap(self) -> int:
        """Reserve every possible shared ranking and branch endpoint score."""
        scoring_cap = 1 + self.max_rounds * (self.shortlist + 2)
        return self.full_evaluations - self.warmup_evaluations - scoring_cap


def numerical_sources() -> dict[str, str]:
    """Fingerprint this driver and every imported numerical implementation."""
    result = runner.source_hashes()
    for name in ("structural", "structural_experiment"):
        path = Path(__file__).with_name(f"{name}.py")
        result[str(path.relative_to(runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class BranchStore:
    """Persist complete per-branch L-BFGS states with exact accepted-boundary resume."""

    def __init__(self, root: Path, target: NDArray[np.complex128], config: Config) -> None:
        """Create a private artifact directory and fixed exact objective context."""
        self.root, self.target, self.config = root, target.copy(), config
        root.mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        name: str,
        circuit: ParameterizedCircuit,
        initial: NDArray[np.float64],
        cap: int,
        *,
        continuation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Optimize from fresh memory unless an explicit warmup continuation is given.

        A retained/deleted repair never passes continuation. Completed artifacts
        are read-only; partial checkpoints retain all secant pairs and counters.
        Process CPU includes optimization and intermediate checkpoint I/O.
        """
        signature = pilot.digest({
            "version": VERSION,
            "circuit": pilot.circuit_record(circuit),
            "initial": initial.tolist(),
            "target_sha256": hashlib.sha256(self.target.tobytes()).hexdigest(),
            "cap": cap,
            "history": self.config.history,
            "checkpoint_every": self.config.checkpoint_every,
            "continuation": None if continuation is None else continuation["optimizer"],
        })
        path, checkpoint_path = self.root / f"{name}.json", self.root / f"{name}.partial.json"
        if path.exists():
            record = cast("dict[str, Any]", pilot.read_record(path))
            if record["signature"] != signature:
                msg = "Completed refinement branch signature changed."
                raise ValueError(msg)
            return record
        prior = cast("dict[str, Any]", pilot.read_record(checkpoint_path)) if checkpoint_path.exists() else None
        if prior is not None and prior["signature"] != signature:
            msg = "Partial refinement branch signature changed."
            raise ValueError(msg)
        base = prior if prior is not None else continuation
        curve = copy.deepcopy(base["curve"]) if base is not None else []
        attempts_path = self.root / f"{name}.attempts.json"
        attempts: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(attempts_path)) if attempts_path.exists() else {"attempts": []}
        )
        prior_cpu = sum(a["cpu_seconds"] for a in attempts["attempts"])
        prior_wall = sum(a["wall_seconds"] for a in attempts["attempts"])
        if continuation is not None:
            prior_cpu += continuation["cpu_seconds"]
            prior_wall += continuation["wall_seconds"]
        cpu, wall = time.process_time(), time.perf_counter()
        new_calls = 0

        def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            nonlocal new_calls
            new_calls += 1
            gradient, loss, _ = density_loss_gradient(
                circuit, theta, self.target, CalibratedLocalDepolarizingProvider()
            )
            return loss, gradient

        def save(state: dict[str, Any]) -> None:
            curve.append({
                "theta": state["theta"],
                "fidelity": 1 - state["loss"],
                "iterations": state["iterations"],
                "evaluations": state["evaluations"],
                "cpu_seconds": prior_cpu + time.process_time() - cpu,
            })
            if state["iterations"] % self.config.checkpoint_every != 0:
                return
            pilot.write_record(
                checkpoint_path,
                {
                    "signature": signature,
                    "optimizer": state,
                    "curve": curve,
                    "cpu_seconds": prior_cpu + time.process_time() - cpu,
                    "wall_seconds": prior_wall + time.perf_counter() - wall,
                },
            )

        completed = False
        try:
            state = optimizers.minimize(
                objective,
                initial,
                max_evaluations=cap,
                max_iterations=cap,
                history=self.config.history,
                state=None if base is None else base["optimizer"],
                callback=save,
            )
            completed = True
        finally:
            attempts["attempts"].append({
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
                "objective_calls": new_calls,
                "completed": completed,
            })
            pilot.write_record(attempts_path, attempts)
        curve.append({
            "theta": state["theta"],
            "fidelity": 1 - state["loss"],
            "iterations": state["iterations"],
            "evaluations": state["evaluations"],
            "cpu_seconds": prior_cpu + time.process_time() - cpu,
        })
        record = {
            "signature": signature,
            "optimizer": state,
            "curve": curve,
            "cpu_seconds": prior_cpu + time.process_time() - cpu,
            "wall_seconds": prior_wall + time.perf_counter() - wall,
            "memory_rule": "fresh_curvature" if continuation is None else "continue_shared_warmup_curvature",
            "physical_joint_gradient_calls": sum(a["objective_calls"] for a in attempts["attempts"])
            + (continuation["physical_joint_gradient_calls"] if continuation is not None else 0),
            "interruption_accounting": (
                "all caught-interruption attempt CPU/calls retained; unsaved calls replayed on resume"
            ),
        }
        record["interruption_replayed_joint_calls"] = record["physical_joint_gradient_calls"] - state["evaluations"]
        pilot.write_record(path, record)
        return record


def _curve_point(
    circuit: dict[str, Any],
    theta: list[float],
    fidelity: float,
    cpu: float,
    calls: int,
) -> dict[str, Any]:
    """Construct an available final-iterate curve point; never choose the best."""
    return {"circuit": circuit, "theta": theta, "fidelity": fidelity, "cpu_seconds": cpu, "evaluations": calls}


def run_pair(
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: Config,
    root: Path,
) -> dict[str, Any]:
    """Execute a paired structural/fixed comparison from one shared warmup.

    Small systems are accepted for automated tests. Scientific jobs separately
    enforce authenticated fixed q6 starts. All scoring calls and directions are
    charged; unlike joint gradients, postrepair scoring uses one direction.
    """
    store = BranchStore(root / "branches", target, config)
    warm = store.optimize("warmup", circuit, initial, config.warmup_evaluations)
    warm_theta = np.asarray(warm["optimizer"]["theta"])
    fixed = store.optimize("fixed", circuit, initial, config.full_evaluations, continuation=warm)
    warm_calls = warm["optimizer"]["evaluations"]
    search_path = root / "SEARCH.json"
    branch_index = 0

    def repair(candidate: ParameterizedCircuit, theta: NDArray[np.float64], cap: int) -> structural.RefinementResult:
        nonlocal branch_index
        branch_index += 1
        record = store.optimize(f"repair_{branch_index:03d}", candidate, theta, cap)
        state = record["optimizer"]
        return structural.RefinementResult(
            np.asarray(state["theta"]),
            state["evaluations"],
            state["iterations"],
            {
                "branch_artifact": f"branches/repair_{branch_index:03d}.json",
                "optimizer": state,
                "cpu_seconds": record["cpu_seconds"],
                "wall_seconds": record["wall_seconds"],
                "memory_rule": record["memory_rule"],
                "physical_joint_gradient_calls": record["physical_joint_gradient_calls"],
            },
        )

    if search_path.exists():
        search = cast("dict[str, Any]", pilot.read_record(search_path))
    else:
        attempts_path = root / "SEARCH_ATTEMPTS.json"
        attempts: dict[str, Any] = (
            cast("dict[str, Any]", pilot.read_record(attempts_path)) if attempts_path.exists() else {"attempts": []}
        )
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            search = structural.adaptive_delete_refine(
                circuit,
                warm_theta,
                target,
                CalibratedLocalDepolarizingProvider(),
                repair,
                config=structural.StructuralConfig(
                    max_rounds=config.max_rounds,
                    shortlist=config.shortlist,
                    repair_evaluations=config.repair_evaluations,
                    max_refinement_evaluations=config.search_refinement_cap,
                ),
                protected_origins=frozenset(
                    g.logical_gate_id
                    for g in circuit.gates[: config.protected_initial_gates]
                    if g.logical_gate_id is not None
                ),
            )
        finally:
            attempts["attempts"].append({
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
            })
            pilot.write_record(attempts_path, attempts)
        # Replayed cached branch CPU must retain its original measured work.
        branches = iter(b for r in search["rounds"] for b in r["branches"])
        for entry in search["ledger"]:
            if entry["role"] in {"retain_repair", "delete_repair"}:
                branch = next(branches)
                if branch["status"] == "complete":
                    entry["cpu_seconds"] = branch["diagnostics"]["cpu_seconds"]
                    entry["wall_seconds"] = branch["diagnostics"]["wall_seconds"]
        search["cpu_seconds"] = sum(x["cpu_seconds"] for x in search["ledger"])
        search["physical_attempt_cpu_seconds"] = sum(x["cpu_seconds"] for x in attempts["attempts"])
        search["resume_replay_cpu_seconds"] = max(0.0, search["physical_attempt_cpu_seconds"] - search["cpu_seconds"])
        pilot.write_record(search_path, search)
        search = cast("dict[str, Any]", pilot.read_record(search_path))
    search_scores = sum(row["exact_channel_evaluations"] for row in search["ledger"])
    search_calls = search["refinement_evaluations"] + search_scores
    remaining = config.full_evaluations - warm_calls - search_calls
    if remaining < 1:
        msg = "Structural search overran its reserved full training-call budget."
        raise ValueError(msg)
    selected_circuit = pilot.restore_circuit(search["circuit"])
    polish = store.optimize("polish", selected_circuit, np.asarray(search["theta"]), remaining)
    original_record = cast("dict[str, Any]", pilot.circuit_record(circuit))
    curve = [
        _curve_point(original_record, p["theta"], p["fidelity"], p["cpu_seconds"], p["evaluations"])
        for p in warm["curve"]
    ]
    elapsed = warm["cpu_seconds"] + search["ledger"][0]["cpu_seconds"] + search["resume_replay_cpu_seconds"]
    calls, position = warm_calls + 1, 1
    current_circuit, current_theta, current_fidelity = (
        original_record,
        warm_theta.tolist(),
        1 - warm["optimizer"]["loss"],
    )
    for row in search["rounds"]:
        block = search["ledger"][position : position + 1 + 2 * len(row["branches"])]
        position += len(block)
        elapsed += sum(entry["cpu_seconds"] for entry in block)
        calls += sum(entry["exact_channel_evaluations"] + entry["refinement_evaluations"] for entry in block)
        if row["selected_branch"] is not None:
            branch = row["branches"][row["selected_branch"]]
            current_circuit, current_theta, current_fidelity = branch["circuit"], branch["theta"], branch["fidelity"]
        curve.append(_curve_point(current_circuit, current_theta, current_fidelity, elapsed, calls))
    prefix_cpu = warm["cpu_seconds"] + search["cpu_seconds"] + search["resume_replay_cpu_seconds"]
    curve.extend(
        _curve_point(
            search["circuit"],
            point["theta"],
            point["fidelity"],
            prefix_cpu + point["cpu_seconds"],
            warm_calls + search_calls + point["evaluations"],
        )
        for point in polish["curve"]
    )
    gradient_calls = warm_calls + search["refinement_evaluations"] + polish["optimizer"]["evaluations"]
    rankings = sum(entry["role"] in {"initial_score_validation", "deletion_ranking"} for entry in search["ledger"])
    structure: dict[str, Any] = {
        "circuit": search["circuit"],
        "theta": polish["optimizer"]["theta"],
        "curve": curve,
        "fidelity": 1 - polish["optimizer"]["loss"],
        "joint_gradient_calls": gradient_calls,
        "forward_score_calls": search_scores - rankings,
        "deletion_ranking_calls": rankings,
        "total_channel_calls": gradient_calls + search_scores,
        "density_direction_passes": 2 * gradient_calls + 2 * rankings + search_scores - rankings,
        "cpu_seconds": prefix_cpu + polish["cpu_seconds"],
        "search_stop_reason": search["stop_reason"],
        "optimizer_stop_reason": polish["optimizer"]["status"],
        "failed_branches": sum(b["status"] == "failed" for r in search["rounds"] for b in r["branches"]),
        "physical_joint_gradient_calls": warm["physical_joint_gradient_calls"]
        + polish["physical_joint_gradient_calls"]
        + sum(
            b.get("diagnostics", {}).get("physical_joint_gradient_calls", b["evaluations"])
            for r in search["rounds"]
            for b in r["branches"]
        ),
    }
    fixed_result: dict[str, Any] = {
        "circuit": original_record,
        "theta": fixed["optimizer"]["theta"],
        "fidelity": 1 - fixed["optimizer"]["loss"],
        "joint_gradient_calls": fixed["optimizer"]["evaluations"],
        "forward_score_calls": 0,
        "deletion_ranking_calls": 0,
        "total_channel_calls": fixed["optimizer"]["evaluations"],
        "density_direction_passes": 2 * fixed["optimizer"]["evaluations"],
        "cpu_seconds": fixed["cpu_seconds"],
        "curve": [
            _curve_point(original_record, p["theta"], p["fidelity"], p["cpu_seconds"], p["evaluations"])
            for p in fixed["curve"]
        ],
        "optimizer_stop_reason": fixed["optimizer"]["status"],
        "failed_branches": 0,
        "physical_joint_gradient_calls": fixed["physical_joint_gradient_calls"],
    }
    common_cpu = min(structure["cpu_seconds"], fixed_result["cpu_seconds"])
    paired = {}
    for name, result in (("structural_lbfgs", structure), ("fixed_lbfgs", fixed_result)):
        available = [p for p in result["curve"] if p["cpu_seconds"] <= common_cpu]
        paired[name] = available[-1] if available else None
    return {
        "status": "complete",
        "config": asdict(config),
        "structural_lbfgs": structure,
        "fixed_lbfgs": fixed_result,
        "common_cpu_seconds": common_cpu,
        "common_cpu_last_available": paired,
        "shared_warmup_cpu_seconds": warm["cpu_seconds"],
        "shared_warmup_calls": warm_calls,
        "physical_training_cpu_seconds": structure["cpu_seconds"] + fixed_result["cpu_seconds"] - warm["cpu_seconds"],
        "shared_work_rule": "warmup physically executed once and charged fully to both arms",
    }


def freeze_protocol(output: Path, *, config: Config, replicas: tuple[int, ...] = (11,)) -> dict[str, Any]:
    """Write a prepare-only immutable development protocol with source custody."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    starts = cast("list[dict[str, Any]]", pilot.read_record(starts_path)["starts"])
    selected = [s for s in starts if s["target_id"] in DEVELOPMENT_TARGETS and s["replica"] in replicas]
    if {(s["target_id"], s["replica"]) for s in selected} != {(t, r) for t in DEVELOPMENT_TARGETS for r in replicas}:
        msg = "Authenticated development source panel is incomplete."
        raise ValueError(msg)
    for source in selected:
        circuit = pilot.restore_circuit(source["candidate"]["circuit"])
        if circuit.num_qubits != 6 or len(circuit.gates) != 108:
            msg = "Scientific structural starts must be authenticated q6 108-gate circuits."
            raise ValueError(msg)
    historical_path = runner.REPO / "output/statevector-adjoint-restructuring-20260912/result.json"
    historical = pilot.read_record(historical_path)
    protocol = {
        "version": VERSION,
        "phase": "development",
        "config": asdict(config),
        "replicas": list(replicas),
        "targets": list(DEVELOPMENT_TARGETS),
        "benchmark_hashes": manifest["benchmark_hashes"],
        "source_hashes": numerical_sources(),
        "source_starts_sha256": hashlib.sha256(starts_path.read_bytes()).hexdigest(),
        "starts": selected,
        "historical_root": str(historical_path.parent),
        "historical_result_checksum": pilot.digest(historical),
        "endpoint_rule": "final accepted optimizer iterate; structure selected only by frozen repaired-deletion policy",
        "fairness": (
            "same full channel-call cap; joint gradient/ranking versus forward-only "
            "counts and measured CPU distinguished"
        ),
        "primary": "equal target mean over all eleven unchanged noisy conditions; local depolarization secondary",
        "references": "unchanged start and historical best-checkpoint structural endpoints are labeled unmatched",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "PROTOCOL.json"
    if path.exists() and pilot.read_record(path) != protocol:
        msg = "Frozen structural protocol differs; use a new output directory."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def verify_protocol(protocol: dict[str, Any]) -> None:
    """Abort on changed benchmark, implementation or authenticated starts."""
    manifest = runner.custody()
    starts_path = runner.OUTPUT / "starts.json"
    if (
        protocol["version"] != VERSION
        or protocol["benchmark_hashes"] != manifest["benchmark_hashes"]
        or protocol["source_hashes"] != numerical_sources()
        or protocol["source_starts_sha256"] != hashlib.sha256(starts_path.read_bytes()).hexdigest()
    ):
        msg = "Structural protocol custody or source identity changed after preparation."
        raise ValueError(msg)


def execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Preserve a failed case explicitly instead of dropping it from the panel."""
    verify_protocol(job["protocol"])
    path = Path(job["root"]) / "FAILED.json"
    signature = pilot.digest({"protocol": job["protocol"], "source": job["source"]})
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature:
            msg = "Failed structural case signature changed."
            raise ValueError(msg)
        return record
    try:
        return _execute_case(job)
    except (ArithmeticError, RuntimeError, ValueError) as error:
        source = job["source"]
        record = {
            "status": "failed",
            "signature": signature,
            "target_id": source["target_id"],
            "replica": source["replica"],
            "failure": f"{type(error).__name__}: {error}",
            "arms": {},
            "partial_artifacts": str(path.parent),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        pilot.write_record(path, record)
        return record


def historical_endpoint(refined: dict[str, Any], budget_seconds: float) -> dict[str, Any]:
    """Select the last historical incumbent inside its declared CPU budget."""
    eligible = [point for point in refined["curve"] if point["cpu_seconds"] <= budget_seconds]
    if not eligible:
        msg = "Historical CPU budget precedes the first scored checkpoint."
        raise ValueError(msg)
    selected = eligible[-1]
    return {
        "circuit": selected["state"]["circuit"],
        "theta": selected["state"]["theta"],
        "comparison": "unmatched_historical_best_checkpoint_reference",
        "archived_local_fidelity": selected["fidelity"],
        "historical_cpu_budget_seconds": budget_seconds,
        "historical_checkpoint_cpu_seconds": selected["cpu_seconds"],
    }


def _execute_case(job: dict[str, Any]) -> dict[str, Any]:
    """Run one authenticated paired development case and all final noise scores."""
    protocol = job["protocol"]
    verify_protocol(protocol)
    source, root = job["source"], Path(job["root"])
    signature = pilot.digest({"protocol": protocol, "source": source})
    path = root / "RESULT.json"
    if path.exists():
        result = cast("dict[str, Any]", pilot.read_record(path))
        if result["signature"] != signature:
            msg = "Completed structural case signature changed."
            raise ValueError(msg)
        return result
    root.mkdir(parents=True, exist_ok=True)
    target = load_target(6, source["target_id"]).state_vector_copy()
    circuit = pilot.restore_circuit(source["candidate"]["circuit"])
    initial = np.asarray(source["candidate"]["theta"])
    pair = run_pair(circuit, initial, target, Config(**protocol["config"]), root)
    arms = {name: pair[name] for name in ("structural_lbfgs", "fixed_lbfgs")}
    arms["unchanged_reference"] = {
        "circuit": source["candidate"]["circuit"],
        "theta": initial.tolist(),
        "comparison": "unmatched_no_refinement",
    }
    historical_root = Path(protocol["historical_root"])
    history = cast("dict[str, Any]", pilot.read_record(historical_root / "result.json"))
    if pilot.digest(history) != protocol["historical_result_checksum"]:
        msg = "Historical reference aggregate changed."
        raise ValueError(msg)
    row = next(
        j for j in history["jobs"] if j["target_id"] == source["target_id"] and j["replica"] == source["replica"]
    )
    archived = cast("dict[str, Any]", pilot.read_record(historical_root / row["path"]))
    if pilot.digest(archived) != row["checksum"]:
        msg = "Historical structural reference checksum mismatch."
        raise ValueError(msg)
    for name, refined in archived["refined"].items():
        arms[f"historical_structural_{name}"] = historical_endpoint(refined, archived["budget_seconds"])
    for arm in arms.values():
        candidate, theta = pilot.restore_circuit(arm["circuit"]), np.asarray(arm["theta"])
        cpu = time.process_time()
        arm["fidelities"] = {noise: channels.evaluate(candidate, theta, target, noise) for noise in channels.NOISE_IDS}
        arm["evaluation_cpu_seconds"] = time.process_time() - cpu
        arm["logical_counts"] = runner.counts(candidate)
        arm["ballarin_counts"] = runner.counts(materialize_ballarin_circuit(candidate, theta).circuit)
        arm.setdefault("comparison", "matched_full_call_cap_with_measured_cpu_comparison")
        expected = arm.get("archived_local_fidelity", arm.get("fidelity"))
        if expected is not None and abs(expected - arm["fidelities"]["depolarizing_1s_all"]) > 1e-10:
            msg = "Independent fanout and saved endpoint local fidelity disagree."
            raise ValueError(msg)
    result = {
        "status": "complete",
        "signature": signature,
        "target_id": source["target_id"],
        "replica": source["replica"],
        "pair": pair,
        "arms": arms,
    }
    verify_protocol(protocol)
    pilot.write_record(path, result)
    return result


def write_scoreboard(root: Path, records: list[dict[str, Any]]) -> None:
    """Keep every endpoint and condition in a machine-readable comparison."""
    rows = []
    for result in sorted(records, key=operator.itemgetter("target_id", "replica")):
        if result["status"] == "failed":
            rows.extend(
                {
                    "target_id": result["target_id"],
                    "replica": result["replica"],
                    "method": method,
                    "noise_id": noise,
                    "status": "failed",
                    "failure": result["failure"],
                    "comparison": "failed_no_final_endpoint",
                }
                for method in ("fixed_lbfgs", "structural_lbfgs")
                for noise in channels.NOISE_IDS
            )
            continue
        for method in sorted(result["arms"]):
            arm = result["arms"][method]
            for noise in channels.NOISE_IDS:
                fidelity = arm["fidelities"][noise]
                logical = arm["logical_counts"]
                lowered = arm["ballarin_counts"] if noise == "ballarin_coupled" else logical
                rows.append({
                    "target_id": result["target_id"],
                    "replica": result["replica"],
                    "status": result["status"],
                    "failure": None,
                    "method": method,
                    "noise_id": noise,
                    "fidelity": fidelity,
                    "comparison": arm["comparison"],
                    "total_channel_calls": arm.get("total_channel_calls"),
                    "joint_gradient_calls": arm.get("joint_gradient_calls"),
                    "physical_joint_gradient_calls": arm.get("physical_joint_gradient_calls"),
                    "forward_score_calls": arm.get("forward_score_calls"),
                    "deletion_ranking_calls": arm.get("deletion_ranking_calls"),
                    "density_direction_passes": arm.get("density_direction_passes"),
                    "training_cpu_seconds": arm.get("cpu_seconds"),
                    "logical_one_qubit": logical["one_qubit"],
                    "logical_two_qubit": logical["two_qubit"],
                    "lowered_one_qubit": lowered["one_qubit"],
                    "lowered_two_qubit": lowered["two_qubit"],
                    "parameters": len(arm["theta"]),
                    "logical_depth": logical["depth"],
                    "lowered_depth": lowered["depth"],
                    "evaluation_cpu_seconds": arm["evaluation_cpu_seconds"],
                    "exactness": "exact_calibrated_channel",
                })
    if rows:
        with (root / "SCOREBOARD.csv").open("w") as stream:
            fieldnames = ("target_id", "replica", "method", "noise_id", "status", "failure")
            remaining = sorted({key for row in rows for key in row} - set(fieldnames))
            writer = csv.DictWriter(stream, fieldnames=(*fieldnames, *remaining))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a frozen protocol or explicitly execute it with bounded workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--repair-evaluations", type=int, choices=(25, 50), default=25)
    parser.add_argument("--replicas", type=int, nargs="+", default=[11])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4 or not args.replicas or len(set(args.replicas)) != len(args.replicas):
        parser.error("Use one to four workers and distinct replica identifiers.")
    if args.prepare_only:
        freeze_protocol(
            args.output, config=Config(repair_evaluations=args.repair_evaluations), replicas=tuple(args.replicas)
        )
        print(f"Prepared only: {args.output / 'PROTOCOL.json'}", flush=True)
        return 0
    protocol = cast("dict[str, Any]", pilot.read_record(args.output / "PROTOCOL.json"))
    verify_protocol(protocol)
    jobs = [
        {
            "protocol": protocol,
            "source": source,
            "root": str(args.output / f"{source['target_id']}__{source['replica']}"),
        }
        for source in protocol["starts"]
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(execute_case, jobs):
            records.append(record)
            write_scoreboard(args.output, records)
            print(record["target_id"], record["replica"], record["status"], flush=True)
    pilot.write_record(
        args.output / "RESULT.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(protocol),
            "results": [pilot.digest(r) for r in records],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
