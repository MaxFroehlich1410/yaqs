# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Separate additional optimization from additional depth at a frozen endpoint."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target
from experiments import topdown_backend_support as support
from experiments import topdown_batch_refinement_policies as policies
from experiments import topdown_mps
from experiments import topdown_noiseless_continuation as continuation
from mqt.yaqs.optimization import StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.trajectories import Estimator, NoiseMaps

VERSION = "topdown.haar_capacity.v1"


@dataclass(frozen=True)
class Budget:
    """Declare a target threshold and finite, separately reported resource limits."""

    target_fidelity: float = 0.999
    fixed_steps: int = 10000
    max_depth: int = 128
    growth_cpu_seconds: float = 86400.0
    progress_every: int = 100

    def validate(self, start_depth: int) -> None:
        """Reject invalid limits before writing any artifacts.

        Raises:
            ValueError: If a threshold, interval, or limit is invalid.
        """
        if (
            isinstance(self.target_fidelity, bool)
            or not np.isfinite(self.target_fidelity)
            or not 0 < self.target_fidelity <= 1
            or any(type(v) is not int or v < 1 for v in (self.fixed_steps, self.max_depth, self.progress_every))
            or self.max_depth <= start_depth
            or isinstance(self.growth_cpu_seconds, bool)
            or not np.isfinite(self.growth_cpu_seconds)
            or self.growth_cpu_seconds <= 0
        ):
            msg = "Invalid capacity-study budget."
            raise ValueError(msg)


def sources() -> dict[str, str]:
    """Fingerprint the unchanged numerical code and this new driver."""
    return {
        **continuation.sources(),
        str(Path(__file__).relative_to(topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def authenticate(parent: Path, target: NDArray, target_id: str) -> tuple:
    """Load the selected statevector endpoint and verify its target and provenance.

    Returns:
        Construction configuration, selected record, arrays, and parent references.

    Raises:
        ValueError: If provenance, topology, or an independent replay disagrees.
    """
    manifest = json.loads((parent / "manifest.json").read_text())
    config = topdown_mps.Config(**manifest["construction_config"])
    config.validate()
    if (
        manifest["version"] != continuation.VERSION
        or manifest["execution_backend"] != "statevector"
        or config.backend != "statevector"
        or manifest["target_id"] != target_id
        or manifest["target_sha256"] != topdown_mps.target_digest(target)
        or manifest["sources"] != continuation.sources()
    ):
        msg = "Parent provenance, target, or backend differs."
        raise ValueError(msg)
    final = support.load_checkpoint(parent / "final")
    if final is None:
        msg = "Parent final checkpoint is incomplete."
        raise ValueError(msg)
    depth = final[0]["best_depth"]
    folder = parent / f"depth_{depth:02d}" / "construction"
    saved = support.load_checkpoint(folder)
    if saved is None:
        msg = "Parent selected checkpoint is incomplete."
        raise ValueError(msg)
    record, arrays = saved
    circuit = continuation.brickwall(config.qubits, depth, initial_single_qubit_layer=True)
    expected = json.loads(json.dumps(support.circuit_record(circuit)))
    if record["circuit"] != expected or record["backend"] != "statevector":
        msg = "Parent selected circuit or backend differs."
        raise ValueError(msg)
    fidelity = StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    if max(abs(fidelity - record["fidelity"]), abs(fidelity - final[0]["best_ideal_fidelity"])) > 1e-9:
        msg = "Parent fidelity fails exact replay."
        raise ValueError(msg)
    references = {
        "parent": str(parent),
        "manifest_sha256": support.digest(parent / "manifest.json"),
        "final_seal_sha256": support.digest(parent / "final" / "complete.json"),
        "checkpoint": str(folder),
        "checkpoint_seal_sha256": support.digest(folder / "complete.json"),
        "initial_fidelity": float(fidelity),
        "historical_continuation_cpu_seconds": final[0]["new_cpu_seconds"],
        "historical_construction_cpu_seconds": final[0]["historical_construction_cpu_seconds"],
    }
    return config, record, arrays, references


class ProgressBackend(StatevectorBackend):
    """Observe accepted batch iterates without altering the optimizer arithmetic."""

    def __init__(self, circuit: ParameterizedCircuit, target: NDArray, output: Path, every: int) -> None:
        """Keep progress outside the unchanged optimizer's numerical state."""
        super().__init__(circuit, target)
        self.output, self.every, self.calls = output, every, 0
        self.started = time.process_time()

    def gradient(
        self,
        theta: NDArray,
        maps: NoiseMaps | None = None,
        *,
        estimator: Estimator = "independent",
        weights: NDArray | None = None,
    ) -> tuple:
        """Checkpoint the incoming accepted iterate at regular gradient boundaries.

        Returns:
            The original gradient, loss, and fidelity without numerical changes.
        """
        result = super().gradient(theta, maps, estimator=estimator, weights=weights)
        self.calls += 1
        if self.calls == 1 or self.calls % self.every == 0:
            record = {
                "status": "running",
                "arm": "fixed",
                "completed_updates": self.calls - 1,
                "fidelity": float(result[2]),
                "gradient_norm": float(np.linalg.norm(result[0])),
                "cpu_seconds": time.process_time() - self.started,
                "resume_scope": "observational snapshot; continuous L-BFGS history is in worker memory",
            }
            support.save_checkpoint(self.output / "live", record, {"theta": theta.copy()})
            support.write_json(self.output / "progress.json", record)
            print(json.dumps(record), flush=True)
        return result


def save_training(folder: Path, circuit: ParameterizedCircuit, target: NDArray, theta: NDArray, trace: dict) -> dict:
    """Store parameters and the full optimizer trace once in authenticated binaries.

    Returns:
        Compact receipt with fresh exact replay and binary trace references.

    Raises:
        ValueError: If the optimizer's claimed best fidelity fails replay.
    """
    fidelity = StatevectorBackend(circuit, target).metrics(theta)[1]
    if not np.isfinite(fidelity) or abs(fidelity - trace["best_fidelity"]) > 1e-9:
        msg = "Optimized fidelity fails exact replay."
        raise ValueError(msg)
    arrays = {"theta": theta, "fidelity": np.asarray(trace.pop("fidelity_trace"))}
    for key in ("initial_theta", "online_final_theta", "best_theta", "final_theta", "returned_theta"):
        value = np.asarray(trace.pop(key))
        destination = next((name for name, existing in arrays.items() if np.array_equal(value, existing)), key)
        arrays[destination] = value
        trace[key] = {"file": "arrays.npz", "key": destination}
    # Exact JSON bytes preserve optional fields and ragged line-search diagnostics.
    # They are compressed inside the NPZ, not duplicated in result/selection JSON.
    records = json.dumps(trace.pop("records"), allow_nan=False, separators=(",", ":")).encode()
    arrays["optimizer_records_utf8"] = np.frombuffer(records, dtype=np.uint8).copy()
    trace["records"] = {"file": "arrays.npz", "key": "optimizer_records_utf8", "encoding": "utf-8 JSON"}
    trace["fidelity_curve"] = {"file": "arrays.npz", "key": "fidelity"}
    return support.save_checkpoint(folder, {"fidelity": fidelity, "training": trace}, arrays)


def fixed(output: Path, target: NDArray, record: dict, arrays: dict, budget: Budget) -> dict:
    """Continue the saved depth with one uninterrupted, otherwise unchanged L-BFGS call.

    Returns:
        Observed threshold attainment or update-budget exhaustion.
    """
    circuit = support.restore_circuit(record["circuit"])
    saved = support.load_checkpoint(output / "training")
    if saved is None:
        engine = ProgressBackend(circuit, target, output, budget.progress_every)
        theta, trace = policies.train_noiseless(
            circuit,
            arrays["theta"],
            target,
            policy="lbfgs",
            steps=budget.fixed_steps,
            online=0,
            stop=budget.target_fidelity,
            engine=engine,
        )
        save_training(output / "training", circuit, target, theta, trace)
        saved = support.load_checkpoint(output / "training")
    assert saved is not None
    trained, history = saved
    hits = np.flatnonzero(history["fidelity"] >= budget.target_fidelity)
    return {
        "best_depth": record["depth"],
        "best_fidelity": trained["fidelity"],
        "completed_updates": trained["training"]["steps"],
        "first_target_update": int(hits[0]) + 1 if hits.size else None,
        "stop_reason": "target_reached" if trained["fidelity"] >= budget.target_fidelity else "update_budget",
        "selected_checkpoint": "training",
        "new_cpu_seconds": trained["training"]["cpu_seconds"],
        "optimizer_history": "reset once at branch start; preserved throughout this call",
    }


def grow(output: Path, target: NDArray, record: dict, arrays: dict, config: topdown_mps.Config, budget: Budget) -> dict:
    """Extend the independent parent with the original per-depth training procedure.

    Returns:
        Every-depth measurements and the first successful depth, if observed.
    """
    previous = arrays["theta"]
    best, best_depth, selected = record["fidelity"], record["depth"], "parent"
    rows, elapsed, stop = [], 0.0, "depth_budget"
    for depth in range(record["depth"] + 1, budget.max_depth + 1):
        folder = output / f"depth_{depth:03d}" / "construction"
        support.write_json(
            output / "progress.json",
            {
                "status": "running",
                "arm": "grow",
                "active_depth": depth,
                "best_fidelity": best,
                "new_cpu_seconds": elapsed,
                "completed_depths": rows,
            },
        )
        saved = support.load_checkpoint(folder)
        if saved is None:
            continuation.construct(folder, target, previous, config, depth)
            saved = support.load_checkpoint(folder)
        assert saved is not None
        result, values = saved
        previous = values["theta"]
        fidelity = result["fidelity"]
        elapsed += result["cpu_seconds"]
        if fidelity > best:
            best, best_depth, selected = fidelity, depth, str(folder.relative_to(output))
        hits = np.flatnonzero(values["fidelity"] >= budget.target_fidelity)
        rows.append({
            "depth": depth,
            "fidelity": fidelity,
            "cpu_seconds": result["cpu_seconds"],
            "wall_seconds": result["wall_seconds"],
            "first_target_update": int(hits[0]) + 1 if hits.size else None,
            "checkpoint": str(folder.relative_to(output)),
        })
        print(json.dumps(rows[-1]), flush=True)
        if best >= budget.target_fidelity:
            stop = "target_reached"
            break
        if elapsed >= budget.growth_cpu_seconds:
            stop = "cpu_budget"
            break
    return {
        "best_depth": best_depth,
        "best_fidelity": best,
        "selected_checkpoint": selected,
        "stop_reason": stop,
        "new_cpu_seconds": elapsed,
        "completed_depths": rows,
        "first_target_depth": rows[-1]["depth"] if stop == "target_reached" else None,
    }


def run(output: Path, parent: Path, target: NDArray, target_id: str, arm: str, budget: Budget) -> dict:
    """Authenticate a separate branch and save success separately from budget exhaustion.

    Returns:
        Sealed final result, without claiming a global minimum depth or convergence.

    Raises:
        ValueError: If the branch, output separation, or resume settings differ.
    """
    output, parent = output.resolve(), parent.resolve()
    if (
        arm not in {"fixed", "grow"}
        or output == parent
        or output.is_relative_to(parent)
        or parent.is_relative_to(output)
    ):
        msg = "Require a valid arm and an output separate from the parent."
        raise ValueError(msg)
    config, record, arrays, references = authenticate(parent, target, target_id)
    budget.validate(record["depth"])
    manifest = {
        "version": VERSION,
        "arm": arm,
        "budget": asdict(budget),
        "parent": references,
        "target_id": target_id,
        "target_sha256": topdown_mps.target_digest(target),
        "construction_config": asdict(config),
        "sources": sources(),
        "growth_stopping": "target or explicit limits; no production depth-patience stop in this capacity audit",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Capacity-study manifest differs on resume."
        raise ValueError(msg)
    support.write_json(path, manifest)
    completed = support.load_checkpoint(output / "final")
    if completed is not None:
        selected = completed[0]["selected_checkpoint"]
        if selected != "parent":
            saved = support.load_checkpoint(output / selected)
            if saved is None:
                msg = "Selected result checkpoint is incomplete."
                raise ValueError(msg)
            circuit = continuation.brickwall(config.qubits, completed[0]["best_depth"], initial_single_qubit_layer=True)
            score = StatevectorBackend(circuit, target).metrics(saved[1]["theta"])[1]
            if abs(score - completed[0]["best_fidelity"]) > 1e-9:
                msg = "Selected result fails exact replay."
                raise ValueError(msg)
        return completed[0]
    result: dict
    with threadpool_limits(limits=1):
        if record["fidelity"] >= budget.target_fidelity:
            result = {
                "best_depth": record["depth"],
                "best_fidelity": record["fidelity"],
                "stop_reason": "target_reached_at_parent",
                "selected_checkpoint": "parent",
                "new_cpu_seconds": 0.0,
            }
        else:
            result = (
                fixed(output, target, record, arrays, budget)
                if arm == "fixed"
                else grow(output, target, record, arrays, config, budget)
            )
    if sources() != manifest["sources"]:
        msg = "Capacity-study sources changed during execution."
        raise ValueError(msg)
    result.update({
        "arm": arm,
        "target": target_id,
        "target_fidelity": budget.target_fidelity,
        "status": "target_reached" if result["best_fidelity"] >= budget.target_fidelity else "budget_exhausted",
        "noise_or_pruning_performed": False,
        "initial_fidelity": references["initial_fidelity"],
    })
    support.write_json(output / "progress.json", result)
    return support.save_checkpoint(output / "final", result, {})


def main(argv: list[str] | None = None) -> None:
    """Launch exactly one independent Haar branch from the selected parent endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("fixed", "grow"), required=True)
    parser.add_argument("--fixed-steps", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--growth-cpu-seconds", type=float, default=86400)
    args = parser.parse_args(argv)
    target = load_target(12, "haar_random_1").state_vector_copy()
    budget = Budget(fixed_steps=args.fixed_steps, max_depth=args.max_depth, growth_cpu_seconds=args.growth_cpu_seconds)
    print(json.dumps(run(args.output, args.parent, target, "haar_random_1", args.arm, budget)), flush=True)


if __name__ == "__main__":
    main()
