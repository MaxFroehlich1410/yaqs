# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reproducible custodied refinement screening and complete channel fan-out."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.ballarin import materialize_ballarin_circuit
from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider as Calibration
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider, exact_density_fidelity
from benchmarks.state_preparation.targets import load_target
from benchmarks.state_preparation.weighted_sampling import FactorSampler
from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, coordinate, optimizers
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
OUTPUT = REPO / "output/overnight-state-preparation-20260913-0108"
FIELDS = (
    "phase",
    "target_id",
    "replica",
    "method",
    "noise_id",
    "convention",
    "status",
    "fidelity",
    "noiseless_fidelity",
    "logical_one_qubit",
    "logical_two_qubit",
    "lowered_one_qubit",
    "lowered_two_qubit",
    "parameters",
    "logical_depth",
    "lowered_depth",
    "iterations",
    "exact_training_evaluations",
    "trajectory_paths",
    "direction_passes",
    "training_cpu_seconds",
    "training_wall_seconds",
    "diagnostic_cpu_seconds",
    "evaluation_cpu_seconds",
    "peak_rss_bytes",
    "seed",
    "run_checksum",
)


def custody() -> dict[str, Any]:
    """Abort if either immutable benchmark file changed since startup."""
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["benchmark_hashes"].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            msg = f"Benchmark custody mismatch: {name}."
            raise ValueError(msg)
    return manifest


def seed_for(phase: str, target_id: str, replica: int, epoch: int = 0, trajectory: int = 0) -> int:
    """Derive independent domains; paired methods deliberately share a stream."""
    payload = [202609130108, phase, target_id, replica, epoch, trajectory]
    return int.from_bytes(hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest()[:8], "little")


def source_hashes() -> dict[str, str]:
    """Fingerprint actual experiment and imported numerical sources."""
    paths = [ROOT / f"{name}.py" for name in ("__init__", "runner", "channels", "coordinate", "optimizers")]
    for prefix in ("benchmarks/state_preparation", "src/mqt/yaqs/optimization"):
        paths.extend((REPO / prefix).glob("*.py"))
    paths.extend(
        REPO / f"experiments/{name}.py" for name in ("state_preparation_pilot", "state_preparation_exact_adam")
    )
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def counts(circuit: ParameterizedCircuit) -> dict[str, int]:
    """Count already lowered physical gates and dependency-scheduled depth."""
    layers = [0] * circuit.num_qubits
    one = two = 0
    for gate in circuit.gates:
        if len(gate.sites) not in {1, 2}:
            msg = "Every operation must be lowered to one or two qubits."
            raise ValueError(msg)
        one += len(gate.sites) == 1
        two += len(gate.sites) == 2
        level = max(layers[q] for q in gate.sites) + 1
        for q in gate.sites:
            layers[q] = level
    return {"one_qubit": one, "two_qubit": two, "depth": max(layers, default=0)}


def train(
    circuit: ParameterizedCircuit,
    initial: NDArray[np.float64],
    target: NDArray[np.complex128],
    config: dict[str, Any],
    training_seed: int,
    *,
    state: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train with a final-iterate rule and checkpoint complete optimizer state.

    All exact scoring used for curves is charged separately. Adam moments,
    coordinate progress and L-BFGS histories resume without algorithm resets.
    """
    method = config["method"]
    if method not in {
        "adam",
        "noiseless_adam",
        "lbfgs",
        "homotopy",
        "coordinate",
        "weighted_adam",
        "adaptive_online",
        "unchanged",
    }:
        msg = "Unknown refinement method."
        raise ValueError(msg)
    if (
        type(config.get("iterations")) is not int
        or config["iterations"] < 0
        or type(config.get("checkpoint_every", 10)) is not int
        or config.get("checkpoint_every", 10) < 1
        or initial.shape != (circuit.num_params,)
        or not np.isrealobj(initial)
        or not np.all(np.isfinite(initial))
    ):
        msg = "Invalid refinement budget or initial parameters."
        raise ValueError(msg)
    if method == "unchanged" and config["iterations"] != 0:
        msg = "Unchanged control must have zero updates."
        raise ValueError(msg)
    state = copy.deepcopy(state)
    theta = np.array(initial, dtype=float, copy=True)
    state = state or {
        "theta": theta.tolist(),
        "first": np.zeros_like(theta).tolist(),
        "second": np.zeros_like(theta).tolist(),
        "counts": np.zeros_like(theta, dtype=int).tolist(),
        "iterations": 0,
        "exact_training_evaluations": 0,
        "trajectory_paths": 0,
        "direction_passes": 0,
        "training_cpu_seconds": 0.0,
        "training_wall_seconds": 0.0,
        "diagnostic_cpu_seconds": 0.0,
        "curve": [],
    }
    theta = np.asarray(state["theta"], dtype=float)
    first, second = (np.asarray(state[k], dtype=float) for k in ("first", "second"))
    raw_counters = state["counts"]
    if any(type(value) is not int or value < 0 for value in raw_counters):
        msg = "Saved occurrence counters must be nonnegative integers."
        raise ValueError(msg)
    counters = np.asarray(raw_counters, dtype=int)
    if (
        type(state["iterations"]) is not int
        or not 0 <= state["iterations"] <= config["iterations"]
        or any(a.shape != initial.shape or not np.all(np.isfinite(a)) for a in (theta, first, second, counters))
        or np.any(second < 0)
        or np.any(counters < 0)
    ):
        msg = "Invalid saved optimizer state."
        raise ValueError(msg)
    calibration = Calibration(0, 0) if method == "noiseless_adam" else Calibration()
    stochastic = method in {"weighted_adam", "adaptive_online"}
    engine = StatevectorBackend(circuit, target) if stochastic else None
    sampler = FactorSampler(circuit, Calibration()) if stochastic else None

    def point(fidelity: float | None = None) -> None:
        begin = time.process_time()
        if fidelity is None:
            fidelity = density_loss_gradient(circuit, theta, target, Calibration())[2]
        state["diagnostic_cpu_seconds"] += time.process_time() - begin
        state["curve"].append({
            "iteration": state["iterations"],
            "fidelity": fidelity,
            "theta": theta.tolist(),
            "training_cpu_seconds": state["training_cpu_seconds"],
            "charged_cpu_seconds": state["training_cpu_seconds"] + state["diagnostic_cpu_seconds"],
            "exact_training_evaluations": state["exact_training_evaluations"],
            "trajectory_paths": state["trajectory_paths"],
        })

    if not state["curve"]:
        point()
    if method == "lbfgs":

        def objective(values: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            gradient, loss, _ = density_loss_gradient(circuit, values, target, calibration)
            return loss, gradient

        cpu0, wall0 = time.process_time(), time.perf_counter()
        previous_cpu, previous_wall = state["training_cpu_seconds"], state["training_wall_seconds"]

        excluded_cpu = excluded_wall = 0.0

        def save_optimizer(optimizer_state: dict[str, Any]) -> None:
            nonlocal theta, excluded_cpu, excluded_wall
            callback_cpu, callback_wall = time.process_time(), time.perf_counter()
            theta = np.asarray(optimizer_state["theta"])
            state.update({
                "optimizer": optimizer_state,
                "theta": theta.tolist(),
                "iterations": optimizer_state["iterations"],
                "exact_training_evaluations": optimizer_state["evaluations"],
                "direction_passes": 2 * optimizer_state["evaluations"],
                "training_cpu_seconds": previous_cpu + callback_cpu - cpu0 - excluded_cpu,
                "training_wall_seconds": previous_wall + callback_wall - wall0 - excluded_wall,
            })
            point(1 - optimizer_state["loss"])
            if checkpoint:
                checkpoint(state)

            excluded_cpu += time.process_time() - callback_cpu
            excluded_wall += time.perf_counter() - callback_wall

        result = optimizers.minimize(
            objective,
            initial,
            max_evaluations=config["evaluations"],
            max_iterations=config["iterations"],
            history=config.get("history", 10),
            state=state.get("optimizer"),
            callback=save_optimizer,
        )
        save_optimizer(result)
        state["stop_reason"] = result["status"]
        return state
    for step in range(state["iterations"] + 1, config["iterations"] + 1):
        cpu0, wall0 = time.process_time(), time.perf_counter()
        measured_fidelity = None
        if method == "coordinate":
            theta, detail = coordinate.sweep(circuit, theta, target, calibration)
            measured_fidelity = cast("float", detail["final_fidelity"])
            state["direction_passes"] += 2
            state["coordinate_conjugations"] = state.get("coordinate_conjugations", 0) + detail["density_conjugations"]
        elif stochastic:
            assert engine is not None
            assert sampler is not None
            maps, weights = sampler.ensemble(
                config.get("paths", 32), seed_for("development_training", str(training_seed), 0, step)
            )
            if method == "weighted_adam":
                gradient = engine.gradient(theta, maps, weights=weights)[0]
                theta, first, second = adam.adam_step(
                    theta, gradient, first, second, step, adam.AdamConfig(learning_rate=config.get("rate", 0.02))
                )
            elif step <= 25:
                theta, _ = engine.online_update(theta, 0.7, maps, weights=weights)
            else:
                theta, _, second, counters = engine.adaptive_online_update(
                    theta, 0.02, second, counters, maps, weights=weights
                )
            state["trajectory_paths"] += config.get("paths", 32)
            state["direction_passes"] += config.get("paths", 32) * (2 if method == "weighted_adam" else 3)
        else:
            if method == "homotopy":
                scale = 0 if step <= config["iterations"] // 4 else (0.5 if step <= config["iterations"] // 2 else 1)
                calibration = Calibration(0.00064 * scale, 0.0051 * scale)
            gradient, _, _ = density_loss_gradient(circuit, theta, target, calibration)
            theta, first, second = adam.adam_step(
                theta, gradient, first, second, step, adam.AdamConfig(learning_rate=config.get("rate", 0.02))
            )
            state["exact_training_evaluations"] += 1
            state["direction_passes"] += 2
        state["training_cpu_seconds"] += time.process_time() - cpu0
        state["training_wall_seconds"] += time.perf_counter() - wall0
        state.update({
            "theta": theta.tolist(),
            "first": first.tolist(),
            "second": second.tolist(),
            "counts": counters.tolist(),
            "iterations": step,
        })
        if step % config.get("checkpoint_every", 10) == 0 or step == config["iterations"]:
            point(measured_fidelity)
            if checkpoint:
                checkpoint(state)
    state["stop_reason"] = "iteration_budget"
    return state


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Authenticate, resume, execute, independently score and checksum one run."""
    custody()
    if job.get("source_hash") is not None and job["source_hash"] != pilot.digest(source_hashes()):
        msg = "Numerical source fingerprint changed after protocol freeze."
        raise ValueError(msg)
    path = Path(job["path"])
    signature = pilot.digest({k: v for k, v in job.items() if k != "path"})
    state_path = path.with_suffix(".checkpoint.json")
    if path.exists():
        record = pilot.read_record(path)
        if record["signature"] != signature or record.get("status") != "complete":
            msg = "Completed run configuration mismatch."
            raise ValueError(msg)
        return record
    target = load_target(6, job["target_id"]).state_vector_copy()
    circuit = pilot.restore_circuit(job["circuit"])
    initial = np.asarray(job["initial"], dtype=float)
    config = job["config"]
    prior = None
    if state_path.exists():
        saved = pilot.read_record(state_path)
        if saved["signature"] != signature:
            msg = "Checkpoint configuration mismatch."
            raise ValueError(msg)
        prior = cast("dict[str, Any]", saved["state"])

    def checkpoint(state: dict[str, Any]) -> None:
        custody()
        pilot.write_record(state_path, {"signature": signature, "state": state})

    begin = time.perf_counter()
    state = train(circuit, initial, target, config, job["seed"], state=prior, checkpoint=checkpoint)
    theta = np.asarray(state["theta"])
    cpu0 = time.process_time()
    fidelities = {name: channels.evaluate(circuit, theta, target, name) for name in channels.NOISE_IDS}
    independent = exact_density_fidelity(circuit, theta, target, ExactLocalPauliProvider())
    if abs(fidelities["depolarizing_1s_all"] - independent) > 1e-10:
        msg = "Independent Qiskit final score mismatch."
        raise FloatingPointError(msg)
    evaluation_cpu = time.process_time() - cpu0
    native = materialize_ballarin_circuit(circuit, theta).circuit
    record: dict[str, Any] = {
        "signature": signature,
        "status": "complete",
        "job": job,
        "state": state,
        "fidelities": fidelities,
        "exactness": "deterministic exact calibrated channel expectations",
        "logical_counts": counts(circuit),
        "ballarin_counts": counts(native),
        "evaluation_cpu_seconds": evaluation_cpu,
        "invocation_wall_seconds": time.perf_counter() - begin,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    custody()
    pilot.write_record(path, record)
    return record


def scoreboard(records: list[dict[str, Any]], output: Path) -> None:
    """Write all target/noise results without discarding unfavorable cases."""
    rows = []
    for result in records:
        job, state = result["job"], result["state"]
        for noise_id, fidelity in result["fidelities"].items():
            logical = result["logical_counts"]
            lowered = result["ballarin_counts"] if noise_id == "ballarin_coupled" else logical
            rows.append({
                "phase": job["phase"],
                "target_id": job["target_id"],
                "replica": job["replica"],
                "method": job["label"],
                "noise_id": noise_id,
                "convention": "calibrated_exact",
                "status": result["status"],
                "fidelity": fidelity,
                "noiseless_fidelity": result["fidelities"]["noiseless"],
                "logical_one_qubit": logical["one_qubit"],
                "logical_two_qubit": logical["two_qubit"],
                "lowered_one_qubit": lowered["one_qubit"],
                "lowered_two_qubit": lowered["two_qubit"],
                "parameters": len(state["theta"]),
                "logical_depth": logical["depth"],
                "lowered_depth": lowered["depth"],
                **{
                    k: state[k]
                    for k in (
                        "iterations",
                        "exact_training_evaluations",
                        "trajectory_paths",
                        "direction_passes",
                        "training_cpu_seconds",
                        "training_wall_seconds",
                        "diagnostic_cpu_seconds",
                    )
                },
                "evaluation_cpu_seconds": result["evaluation_cpu_seconds"],
                "peak_rss_bytes": result["peak_rss_bytes"],
                "seed": job["seed"],
                "run_checksum": pilot.digest(result),
            })
    with (output / "SCOREBOARD.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Development scoreboard",
        "",
        "Exact calibrated channels; equal target and noisy-condition weights.",
        "",
        "| Method | Local depolarizing mean (%) | All 11 noisy conditions mean (%) | Mean training CPU (s) |",
        "|---|---:|---:|---:|",
    ]
    for label in sorted({r["job"]["label"] for r in records}):
        matched = [r for r in records if r["job"]["label"] == label]
        local = np.mean([r["fidelities"]["depolarizing_1s_all"] for r in matched])
        overall = np.mean([v for r in matched for k, v in r["fidelities"].items() if k != "noiseless"])
        cpu = np.mean([r["state"]["training_cpu_seconds"] for r in matched])
        lines.append(f"| {label} | {100 * local:.6f} | {100 * overall:.6f} | {cpu:.4f} |")
    (output / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a predeclared JSON panel, with authenticated per-run exact resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between one and four")
    manifest = custody()
    protocol = json.loads(args.protocol.read_text())
    if protocol["phase"] != "development":
        parser.error("Confirmation requires a separate frozen construction protocol")
    root = OUTPUT / protocol["name"]
    root.mkdir(exist_ok=True)
    starts = cast("list[dict[str, Any]]", pilot.read_record(OUTPUT / "starts.json")["starts"])
    runtime = {"protocol": protocol, "source_hashes": source_hashes(), "benchmark_hashes": manifest["benchmark_hashes"]}
    frozen_path = root / "PROTOCOL.json"
    if frozen_path.exists() and pilot.read_record(frozen_path) != runtime:
        parser.error("Retained protocol/source hash mismatch")
    if not frozen_path.exists():
        pilot.write_record(frozen_path, runtime)
    jobs = []
    for source in starts:
        if source["replica"] not in protocol["replicas"] or source["target_id"] not in protocol["targets"]:
            continue
        for label, config in protocol["methods"].items():
            candidate = source["candidate"]
            jobs.append({
                "phase": "development",
                "label": label,
                "config": config,
                "target_id": source["target_id"],
                "replica": source["replica"],
                "circuit": candidate["circuit"],
                "initial": candidate["theta"],
                "source_checksum": pilot.digest(source),
                "runtime_checksum": pilot.digest(runtime),
                "source_hash": pilot.digest(runtime["source_hashes"]),
                "seed": seed_for("development_training", source["target_id"], source["replica"]),
                "path": str(root / f"{source['target_id']}__{source['replica']}__{label}.json"),
            })
    np.random.default_rng(seed_for("development_diagnostics", protocol["name"], 0)).shuffle(jobs)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_job, jobs):
            records.append(record)
            scoreboard(records, root)
            print(
                record["job"]["label"],
                record["job"]["target_id"],
                record["fidelities"]["depolarizing_1s_all"],
                flush=True,
            )
    pilot.write_record(
        root / "RESULT.json",
        {"runtime_checksum": pilot.digest(runtime), "runs": [pilot.digest(r) for r in records], "status": "complete"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
