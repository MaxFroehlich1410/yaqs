# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Refine authenticated intermediate Haar depths with unchanged continuous L-BFGS."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target
from experiments import topdown_haar_capacity as capacity

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERSION = "topdown.haar_refinement.v1"
support = capacity.support


def sources() -> dict[str, str]:
    """Fingerprint this driver together with the unchanged parent implementation.

    Returns:
        Repository-relative paths and SHA-256 digests.
    """
    return {
        **capacity.sources(),
        str(Path(__file__).relative_to(capacity.topdown_mps.ROOT)): support.digest(Path(__file__)),
    }


def authenticate(parent: Path, target: NDArray, target_id: str, depth: int) -> tuple:
    """Verify one completed growth checkpoint, its provenance and exact fidelity.

    Returns:
        Selected record, arrays, and source/cost references.

    Raises:
        ValueError: If the parent, target, depth, topology or replay differs.
    """
    manifest = json.loads((parent / "manifest.json").read_text())
    config = capacity.topdown_mps.Config(**manifest["construction_config"])
    config.validate()
    if (
        type(depth) is not int
        or depth < 1
        or manifest["version"] != capacity.VERSION
        or manifest["arm"] != "grow"
        or config.backend != "statevector"
        or manifest["target_id"] != target_id
        or manifest["target_sha256"] != capacity.topdown_mps.target_digest(target)
        or manifest["sources"] != capacity.sources()
    ):
        msg = "Growth provenance, target, backend or requested depth differs."
        raise ValueError(msg)
    final = support.load_checkpoint(parent / "final")
    folder = parent / f"depth_{depth:03d}" / "construction"
    selected = support.load_checkpoint(folder)
    if final is None or selected is None:
        msg = "Require sealed growth final and selected depth checkpoints."
        raise ValueError(msg)
    record, arrays = selected
    rows = final[0]["completed_depths"]
    depths = [row["depth"] for row in rows]
    if depth not in depths or depths != sorted(set(depths)):
        msg = "Requested depth is absent or duplicated in the growth receipt."
        raise ValueError(msg)
    row = rows[depths.index(depth)]
    circuit = capacity.continuation.brickwall(config.qubits, depth, initial_single_qubit_layer=True)
    expected = json.loads(json.dumps(support.circuit_record(circuit)))
    fidelity = capacity.StatevectorBackend(circuit, target).metrics(arrays["theta"])[1]
    if (
        record["depth"] != depth
        or record["backend"] != "statevector"
        or record["circuit"] != expected
        or row["checkpoint"] != str(folder.relative_to(parent))
        or not np.isfinite(fidelity)
        or max(abs(fidelity - record["fidelity"]), abs(fidelity - row["fidelity"])) > 1e-9
    ):
        msg = "Selected topology or fidelity fails exact replay."
        raise ValueError(msg)
    shared = manifest["parent"]
    common_cpu = shared["historical_construction_cpu_seconds"] + shared["historical_continuation_cpu_seconds"]
    growth_cpu = sum(row["cpu_seconds"] for row in rows if row["depth"] <= depth)
    references = {
        "parent": str(parent),
        "manifest_sha256": support.digest(parent / "manifest.json"),
        "final_seal_sha256": support.digest(parent / "final" / "complete.json"),
        "checkpoint": str(folder),
        "checkpoint_seal_sha256": support.digest(folder / "complete.json"),
        "initial_fidelity": float(fidelity),
        "shared_depth16_cpu_seconds": common_cpu,
        "growth_to_selected_cpu_seconds": growth_cpu,
        "historical_cpu_seconds": common_cpu + growth_cpu,
    }
    return record, arrays, references


def threshold_crossing(initial: float, records: list[dict], threshold: float) -> dict | None:
    """Locate a threshold crossing without rounding or counting rejected trials.

    Returns:
        The first observed crossing and cumulative refinement costs, or None.
    """
    if initial >= threshold:
        return {"update": 0, "fidelity": initial, "cpu_seconds": 0.0, "wall_seconds": 0.0}
    for record in records:
        if record["fidelity"] >= threshold:
            return {
                "update": record["iteration"],
                "fidelity": record["fidelity"],
                "cpu_seconds": record["cumulative_cpu_seconds"],
                "wall_seconds": record["cumulative_wall_seconds"],
            }
    return None


def run(output: Path, parent: Path, target: NDArray, target_id: str, depth: int, budget: capacity.Budget) -> dict:
    """Refine one saved depth in a single continuous, isolated optimizer call.

    Returns:
        Authenticated final result with threshold crossings and separate costs.

    Raises:
        ValueError: If paths, provenance, resume state or final replay disagree.
    """
    output, parent = output.resolve(), parent.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        msg = "Refinement output must be separate from the historical parent."
        raise ValueError(msg)
    budget.validate(depth)
    with threadpool_limits(limits=1):
        record, arrays, references = authenticate(parent, target, target_id, depth)
    manifest = {
        "version": VERSION,
        "depth": depth,
        "target_id": target_id,
        "target_sha256": capacity.topdown_mps.target_digest(target),
        "sources": sources(),
        "parent": references,
        "budget": asdict(budget),
        "optimizer": "unchanged safeguarded L-BFGS; online=0; rate=0.7; one continuous call",
        "optimizer_history": "reset once at start; retained throughout call; live snapshots are observational",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        msg = "Refinement manifest differs on resume."
        raise ValueError(msg)
    if path.exists() and support.load_checkpoint(output / "training") is None:
        msg = "Incomplete continuous call: use a new output directory to replay from the original parent."
        raise ValueError(msg)
    support.write_json(path, manifest)
    completed = support.load_checkpoint(output / "final")
    with threadpool_limits(limits=1):
        result = capacity.fixed(output, target, record, arrays, budget)
        saved = support.load_checkpoint(output / "training")
        assert saved is not None
        trained, history = saved
        circuit = support.restore_circuit(record["circuit"])
        started = time.process_time()
        gradient, _, replay = capacity.StatevectorBackend(circuit, target).gradient(history["theta"])
        verification_cpu = time.process_time() - started
    if (
        not np.isfinite(replay)
        or not np.all(np.isfinite(gradient))
        or abs(replay - trained["fidelity"]) > 1e-9
        or sources() != manifest["sources"]
    ):
        msg = "Endpoint replay, gradient or numerical sources failed verification."
        raise ValueError(msg)
    if completed is not None:
        if abs(completed[0]["best_fidelity"] - replay) > 1e-9:
            msg = "Final result disagrees with saved training."
            raise ValueError(msg)
        return completed[0]
    updates = json.loads(history["optimizer_records_utf8"].tobytes())
    result.update({
        "status": "target_reached" if replay >= budget.target_fidelity else "budget_exhausted",
        "target_id": target_id,
        "target_fidelity": budget.target_fidelity,
        "initial_fidelity": references["initial_fidelity"],
        "threshold_crossings": {
            str(threshold): threshold_crossing(references["initial_fidelity"], updates, threshold)
            for threshold in sorted({0.99, budget.target_fidelity})
        },
        "gradient_norm": float(np.linalg.norm(gradient)),
        "statevector_replay_error": abs(replay - trained["fidelity"]),
        "endpoint_verification_cpu_seconds": verification_cpu,
        "training_wall_seconds": trained["training"]["wall_seconds"],
        "historical_cpu_seconds": references["historical_cpu_seconds"],
        "total_training_cpu_seconds": references["historical_cpu_seconds"] + result["new_cpu_seconds"],
        "num_parameters": circuit.num_params,
        "logical_two_qubit_gates": sum(len(gate.sites) == 2 for gate in circuit.gates),
        "noise_or_pruning_performed": False,
        "fidelity_curve": {"file": "training/arrays.npz", "key": "fidelity"},
    })
    support.write_json(output / "progress.json", result)
    return support.save_checkpoint(output / "final", result, {})


def main(argv: list[str] | None = None) -> None:
    """Run one of the two agreed Haar refinements; external scheduling is separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(66, 87), required=True)
    args = parser.parse_args(argv)
    target = load_target(12, "haar_random_1").state_vector_copy()
    budget = capacity.Budget(fixed_steps=2000, target_fidelity=0.999, progress_every=50)
    print(json.dumps(run(args.output, args.parent, target, "haar_random_1", args.depth, budget)), flush=True)


if __name__ == "__main__":
    main()
