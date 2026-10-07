# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare matched product-state starts with longer noiseless depth optimization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target
from experiments import topdown_haar_refinement as refinement
from mqt.yaqs.optimization import ParameterizedCircuit, StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.random_mps_initialization.v1"
ARMS = ("near_zero", "random_product", "largest_basis")
SEEDS = (300, 400, 500)
TARGET_ID = "random_mps_bond3"
support = refinement.support
capacity = refinement.capacity


@dataclass(frozen=True)
class Budget:
    """Allocate more batch optimization at each depth, with explicit audit limits."""

    target_fidelity: float = 0.99
    online_steps: int = 100
    batch_steps: int = 2000
    max_depth: int = 16
    max_cpu_seconds: float = 21600.0
    progress_every: int = 50

    def validate(self, seed: int) -> None:
        """Reject invalid resource limits, random seeds and stopping thresholds.

        Raises:
            ValueError: If any budget field or seed is invalid.
        """
        if (
            type(self.online_steps) is not int
            or self.online_steps < 0
            or any(type(v) is not int or v < 1 for v in (self.batch_steps, self.max_depth, self.progress_every))
            or type(seed) is not int
            or not 0 <= seed < 2**32 - 1000 - self.max_depth
            or isinstance(self.target_fidelity, bool)
            or not np.isfinite(self.target_fidelity)
            or not 0 < self.target_fidelity <= 1
            or isinstance(self.max_cpu_seconds, bool)
            or not np.isfinite(self.max_cpu_seconds)
            or self.max_cpu_seconds <= 0
        ):
            msg = "Invalid initialization-study budget or seed."
            raise ValueError(msg)


def sources() -> dict[str, str]:
    """Fingerprint the reused numerical implementation and this experiment."""
    return {
        **refinement.sources(),
        str(Path(__file__).relative_to(capacity.topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def validate_target(target: NDArray) -> tuple[NDArray, int]:
    """Copy a finite normalized qubit vector and infer its register size.

    Returns:
        A complex128 target and its number of qubits.

    Raises:
        ValueError: If the input is not a normalized finite qubit statevector.
    """
    vector = np.asarray(target, dtype=np.complex128)
    size = vector.size
    if (
        vector.ndim != 1
        or size < 2
        or size & (size - 1)
        or not np.all(np.isfinite(vector))
        or abs(np.vdot(vector, vector) - 1) > 1e-10
    ):
        msg = "Require a finite normalized qubit target vector."
        raise ValueError(msg)
    return vector.copy(), size.bit_length() - 1


def initial_parameters(target: NDArray, arm: str, seed: int) -> tuple[NDArray, dict]:
    """Change only the initial U3 layer, preserving paired entangling parameters.

    Returns:
        Depth-one angles and exact product-state/complete-circuit starting scores.

    Raises:
        ValueError: If the target, seed or initialization arm is invalid.
    """
    Budget().validate(seed)
    if arm not in ARMS:
        msg = "Unknown initialization arm."
        raise ValueError(msg)
    target, qubits = validate_target(target)
    circuit = capacity.continuation.brickwall(qubits, 1, initial_single_qubit_layer=True)
    theta = capacity.continuation.legacy.extend(None, circuit.num_params, seed + 1001, 0.05)
    largest = int(np.argmax(np.abs(target) ** 2))
    if arm == "random_product":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2381]))
        angles = np.zeros((qubits, 3))
        angles[:, 0] = np.arccos(1 - 2 * rng.random(qubits))
        angles[:, 1] = rng.uniform(0, 2 * np.pi, qubits)
        theta[: 3 * qubits] = angles.ravel()
    elif arm == "largest_basis":
        angles = np.zeros((qubits, 3))
        angles[:, 0] = [np.pi * ((largest >> site) & 1) for site in range(qubits)]
        theta[: 3 * qubits] = angles.ravel()
    product = ParameterizedCircuit(qubits, circuit.gates[: 3 * qubits], num_params=3 * qubits)
    return theta, {
        "arm": arm,
        "seed": seed,
        "parameter_seed": seed + 1001,
        "random_product_seed": [seed, 2381] if arm == "random_product" else None,
        "largest_basis_index": largest,
        "largest_basis_probability": float(abs(target[largest]) ** 2),
        "zero_state_fidelity": float(abs(target[0]) ** 2),
        "product_state_fidelity": StatevectorBackend(product, target).metrics(theta[: 3 * qubits])[1],
        "initial_circuit_fidelity": StatevectorBackend(circuit, target).metrics(theta)[1],
        "basis_order": "little endian: site i is bit i of amplitude index",
        "paired_parameters": "identical depth-one parameters outside the initial U3 layer",
    }


def check_depth(folder: Path, target: NDArray, initial: NDArray, depth: int) -> tuple | None:
    """Authenticate complete training, its starting parameters and exact replay.

    Returns:
        The sealed record and arrays, or None for an uncommitted optimizer call.

    Raises:
        ValueError: If stored input, topology, history or fidelity disagrees.
    """
    saved = support.load_checkpoint(folder / "construction")
    if saved is None:
        return None
    record, arrays = saved
    qubits = validate_target(target)[1]
    circuit = capacity.continuation.brickwall(qubits, depth, initial_single_qubit_layer=True)
    trace = record["training"]
    expected = json.loads(json.dumps(support.circuit_record(circuit)))
    replay = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    records = json.loads(arrays["optimizer_records_utf8"].tobytes())
    if (
        trace["depth"] != depth
        or trace["circuit"] != expected
        or not np.array_equal(arrays[trace["initial_theta"]["key"]], initial)
        or len(records) != trace["steps"]
        or [r["iteration"] for r in records] != list(range(1, len(records) + 1))
        or not np.array_equal(arrays["fidelity"], [r["fidelity"] for r in records])
        or not np.isfinite(replay)
        or max(abs(replay - record["fidelity"]), abs(replay - trace["best_fidelity"])) > 1e-9
    ):
        msg = "Stored depth input, history, topology or exact replay differs."
        raise ValueError(msg)
    return saved


def train_depth(folder: Path, target: NDArray, initial: NDArray, depth: int, budget: Budget) -> tuple:
    """Run one continuous extended hybrid call; reuse only sealed complete calls.

    Returns:
        Authenticated training record and private arrays.

    Raises:
        ValueError: If an interrupted call would restart silently or replay fails.
    """
    saved = check_depth(folder, target, initial, depth)
    if saved is not None:
        return saved
    marker = folder / "started.json"
    if marker.exists():
        msg = "Incomplete continuous call: use a new output directory for an explicit restart."
        raise ValueError(msg)
    support.write_json(marker, {"depth": depth, "resume_scope": "completed stages only; live lacks L-BFGS history"})
    start, wall = time.process_time(), time.perf_counter()
    qubits = validate_target(target)[1]
    circuit = capacity.continuation.brickwall(qubits, depth, initial_single_qubit_layer=True)
    engine = capacity.ProgressBackend(circuit, target, folder, budget.progress_every)
    already_reached = engine.metrics(initial)[1] >= budget.target_fidelity
    theta, trace = capacity.policies.train_noiseless(
        circuit,
        initial,
        target,
        policy="lbfgs",
        steps=0 if already_reached else budget.online_steps + budget.batch_steps,
        online=0 if already_reached else budget.online_steps,
        rate=0.7,
        stop=budget.target_fidelity,
        engine=engine,
    )
    verification_start = time.process_time()
    gradient, _, replay = StatevectorBackend(circuit, target).gradient(theta)
    if not np.all(np.isfinite(gradient)) or abs(replay - trace["best_fidelity"]) > 1e-9:
        msg = "Optimized depth fails independent endpoint replay."
        raise ValueError(msg)
    trace.update({
        "depth": depth,
        "circuit": support.circuit_record(circuit),
        "backend": "statevector",
        "gradient_norm": float(np.linalg.norm(gradient)),
        "statevector_replay_error": abs(replay - trace["best_fidelity"]),
        "endpoint_verification_cpu_seconds": time.process_time() - verification_start,
        "threshold_crossings": {
            str(t): refinement.threshold_crossing(trace["initial_fidelity"], trace["records"], t)
            for t in sorted({0.99, 0.999, budget.target_fidelity})
        },
        "stage_cpu_seconds": time.process_time() - start,
        "stage_wall_seconds": time.perf_counter() - wall,
        "stage_cost_scope": "includes training/progress/replay; excludes final seal serialization",
    })
    capacity.save_training(folder / "construction", circuit, target, theta, trace)
    saved = check_depth(folder, target, initial, depth)
    assert saved is not None
    return saved


def run_case(output: Path, target: NDArray, target_id: str, arm: str, seed: int, budget: Budget) -> dict:
    """Grow one matched arm with longer per-depth refinement and sealed recovery.

    Returns:
        A sealed best-circuit result and complete depth/cost references.

    Raises:
        ValueError: If the manifest, initial checkpoint or numerical sources differ.
    """
    budget.validate(seed)
    target, qubits = validate_target(target)
    start, wall = time.process_time(), time.perf_counter()
    initial, metadata = initial_parameters(target, arm, seed)
    setup_cpu, setup_wall = time.process_time() - start, time.perf_counter() - wall
    output = output.resolve()
    manifest = {
        "version": VERSION,
        "target_id": target_id,
        "qubits": qubits,
        "target_sha256": capacity.topdown_mps.target_digest(target),
        "fixture_sha256": support.digest(DEFAULT_TARGET_PATH),
        "arm": arm,
        "seed": seed,
        "budget": asdict(budget),
        "sources": sources(),
        "optimizer": "100 online by default; safeguarded L-BFGS; rate=0.7; continuous call per depth",
        "extension": "RandomState(seed+1000+depth); scale=0.001 after depth one",
        "patience": "disabled for this bounded initialization/capacity audit",
        "noise_or_pruning": False,
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Initialization-study manifest differs on resume."
        raise ValueError(msg)
    support.write_json(path, manifest)
    prepared = support.load_checkpoint(output / "initialization")
    if prepared is None:
        support.save_checkpoint(
            output / "initialization",
            {**metadata, "setup_cpu_seconds": setup_cpu, "setup_wall_seconds": setup_wall},
            {"theta": initial},
        )
        prepared = support.load_checkpoint(output / "initialization")
    assert prepared is not None
    if not np.array_equal(prepared[1]["theta"], initial) or any(prepared[0][k] != v for k, v in metadata.items()):
        msg = "Stored initialization differs from its matched deterministic input."
        raise ValueError(msg)
    rows, previous = [], initial
    elapsed = prepared[0]["setup_cpu_seconds"]
    best, best_depth, selected = -1.0, 0, None
    stop = "depth_budget"
    for depth in range(1, budget.max_depth + 1):
        folder = output / f"depth_{depth:03d}"
        if depth > 1:
            circuit = capacity.continuation.brickwall(qubits, depth, initial_single_qubit_layer=True)
            previous = capacity.continuation.legacy.extend(previous, circuit.num_params, seed + 1000 + depth, 0.001)
        support.write_json(
            output / "progress.json",
            {
                "status": "running",
                "arm": arm,
                "seed": seed,
                "depth": depth,
                "depth_progress": str((folder / "progress.json").relative_to(output)),
                "completed_depths": rows,
            },
        )
        record, arrays = train_depth(folder, target, previous, depth, budget)
        trace = record["training"]
        previous = arrays["theta"]
        elapsed += trace["stage_cpu_seconds"]
        if record["fidelity"] > best:
            best, best_depth, selected = record["fidelity"], depth, str((folder / "construction").relative_to(output))
        rows.append({
            "depth": depth,
            "fidelity": record["fidelity"],
            "updates": trace["steps"],
            "online_sweeps": trace["online_sweeps"],
            "batch_updates": trace["steps"] - trace["online_sweeps"],
            "gradient_norm": trace["gradient_norm"],
            "stage_cpu_seconds": trace["stage_cpu_seconds"],
            "stage_wall_seconds": trace["stage_wall_seconds"],
            "threshold_crossings": trace["threshold_crossings"],
            "checkpoint": str((folder / "construction").relative_to(output)),
            "checkpoint_seal_sha256": support.digest(folder / "construction/complete.json"),
        })
        print(json.dumps(rows[-1]), flush=True)
        if sources() != manifest["sources"]:
            msg = "Numerical sources changed during the study."
            raise ValueError(msg)
        if best >= budget.target_fidelity:
            stop = "target_reached"
            break
        if elapsed >= budget.max_cpu_seconds:
            stop = "cpu_budget"
            break
    result = {
        "target_id": target_id,
        "arm": arm,
        "seed": seed,
        "best_fidelity": best,
        "best_depth": best_depth,
        "selected_checkpoint": selected,
        "stop_reason": stop,
        "status": "target_reached" if stop == "target_reached" else "budget_exhausted",
        "target_fidelity": budget.target_fidelity,
        "total_cpu_seconds": elapsed,
        "cpu_accounting_scope": (
            "initialization plus measured training/progress/endpoint replay; "
            "excludes final seal serialization and standalone authentication replays"
        ),
        "setup_cpu_seconds": prepared[0]["setup_cpu_seconds"],
        "num_parameters": 3 * qubits + 9 * (qubits - 1) * best_depth,
        "logical_two_qubit_gates": 3 * (qubits - 1) * best_depth,
        "depth_note": "entangling blocks per bond; not compiled two-qubit depth",
        "completed_depths": rows,
        "noise_or_pruning_performed": False,
        "optimizer_history": "reset once per depth; retained throughout each call; live is observational",
    }
    existing = support.load_checkpoint(output / "final")
    if existing is not None:
        if {k: v for k, v in existing[0].items() if k != "arrays"} != result:
            msg = "Sealed final result disagrees with authenticated completed stages."
            raise ValueError(msg)
        support.write_json(output / "progress.json", existing[0])
        return existing[0]
    support.write_json(output / "progress.json", result)
    return support.save_checkpoint(output / "final", result, {})


def prepare_plan(output: Path, budget: Budget) -> dict:
    """Write nine explicit independent job commands without starting training.

    Returns:
        The matched matrix, declared numerical budget and single-thread environment.
    """
    budget.validate(SEEDS[-1])
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            command = [
                sys.executable,
                "-m",
                "experiments.topdown_random_mps_initialization",
                "--action",
                "run",
                "--output",
                str(output.resolve() / "jobs" / f"{arm}__seed{seed}"),
                "--arm",
                arm,
                "--seed",
                str(seed),
            ]
            for name, value in asdict(budget).items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            jobs.append({"arm": arm, "seed": seed, "command": command})
    target = load_target(12, TARGET_ID).state_vector_copy()
    plan = {
        "version": VERSION,
        "target_id": TARGET_ID,
        "qubits": 12,
        "target_sha256": capacity.topdown_mps.target_digest(target),
        "sources": sources(),
        "budget": asdict(budget),
        "jobs": jobs,
        "environment": dict.fromkeys(
            (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ),
            "1",
        ),
        "deployment_guards": {"wall_seconds_per_job": 28800, "memory_gib_per_job": 16},
        "launched": False,
    }
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        msg = "Existing matrix plan differs; use a new output directory."
        raise ValueError(msg)
    support.write_json(path, plan)
    return plan


def main(argv: list[str] | None = None) -> None:
    """Prepare the agreed matrix by default, or explicitly run one independent arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    for name, value in asdict(Budget()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    budget = Budget(**{name: getattr(args, name) for name in asdict(Budget())})
    with threadpool_limits(limits=1):
        if args.action == "plan":
            result = prepare_plan(args.output, budget)
            print(json.dumps({"plan": str(args.output / "plan.json"), "jobs": len(result["jobs"]), "launched": False}))
        else:
            if args.arm is None or args.seed is None:
                parser.error("run requires --arm and --seed")
            target = load_target(12, TARGET_ID).state_vector_copy()
            print(json.dumps(run_case(args.output, target, TARGET_ID, args.arm, args.seed, budget)), flush=True)


if __name__ == "__main__":
    main()
