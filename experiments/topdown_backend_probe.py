# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fixed-input dense/MPS parity and explicit bond-cap convergence diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target_collection
from experiments import topdown_backend_support as support
from experiments.topdown_mps import diagnostics, measure_resources
from mqt.yaqs.optimization import KrotovNoiseMap, KrotovTruncation, MPSBackend, StatevectorBackend, TrajectorySampler

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.trajectories import NoiseMaps


def compare_backends(
    circuit: ParameterizedCircuit,
    theta: NDArray,
    target: NDArray,
    maps: NoiseMaps,
    weights: NDArray,
    *,
    bond_caps: tuple[int, ...] = (4, 8, 16),
) -> dict:
    """Compare identical frozen inputs before optimizing any new scientific case.

    Returns:
        Exact parity and each capped run's fidelity/signal/update discrepancies,
        independent dense assessment of updated parameters, CPU and memory.
        Cross signals are not labeled finite-sample objective gradients.

    Raises:
        ValueError: If the dense diagnostic is too large or caps are invalid.
    """
    if circuit.num_qubits > 12 or any(type(cap) is not int or cap < 1 for cap in bond_caps):
        msg = "Backend probes require at most twelve qubits and positive bond caps."
        raise ValueError(msg)
    reference = StatevectorBackend(circuit, target)
    signal, loss, fidelity = reference.gradient(theta, maps, estimator="cross", weights=weights)
    updated, _ = reference.online_update(theta, 0.2, maps, estimator="cross", weights=weights)
    deletion, ideal = reference.deletion_fidelities(theta)
    runs: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for cap in (None, *bond_caps):
            start = time.process_time()
            with measure_resources() as resources:
                engine = MPSBackend(circuit, target, truncation=KrotovTruncation(max_bond_dim=cap))
                actual, actual_loss, actual_fidelity = engine.gradient(theta, maps, estimator="cross", weights=weights)
                candidate, _ = engine.online_update(theta, 0.2, maps, estimator="cross", weights=weights)
                removed, original = engine.deletion_fidelities(theta)
            runs.append({
                "max_bond_dim": cap,
                "cross_signal_max_error": float(np.max(abs(actual - signal), initial=0)),
                "fidelity_error": abs(actual_fidelity - fidelity),
                "loss_error": abs(actual_loss - loss),
                "online_parameter_max_error": float(np.max(abs(candidate - updated), initial=0)),
                "deletion_fidelity_max_error": float(np.max(abs(removed - deletion), initial=0)),
                "ideal_fidelity_error": abs(original - ideal),
                "exact_fidelity_after_online": reference.metrics(candidate, maps, weights=weights)[1],
                "cpu_seconds": time.process_time() - start,
                "resources": resources,
                "diagnostics": diagnostics(engine),
            })
    exact = runs[0]
    passed = all(
        exact[key] <= 1e-10
        for key in (
            "cross_signal_max_error",
            "fidelity_error",
            "loss_error",
            "online_parameter_max_error",
            "deletion_fidelity_max_error",
            "ideal_fidelity_error",
        )
    )
    return {
        "exact_parity_passed": passed,
        "tolerance": 1e-10,
        "reference_fidelity": fidelity,
        "reference_fidelity_after_online": reference.metrics(updated, maps, weights=weights)[1],
        "runs": runs,
    }


def replay_training_bank(circuit: ParameterizedCircuit, bank: dict) -> tuple:
    """Reconstruct the exact recorded histories, never sample a replacement bank.

    Returns:
        Retained maps and their empirical no-jump multiplicity weights.

    Raises:
        ValueError: If the recorded map digest or multiplicities disagree.
    """
    events = bank["events"]
    digest = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
    if digest != bank["raw_sha256"]:
        msg = "Recorded training history checksum mismatch."
        raise ValueError(msg)
    raw = []
    for path in events:
        row = [KrotovNoiseMap() for _ in circuit.gates]
        for event in path:
            row[event["gate"]] = KrotovNoiseMap(
                operators=tuple(
                    (np.array(op["real"]) + 1j * np.array(op["imag"]), tuple(op["sites"])) for op in event["operators"]
                )
            )
        raw.append(tuple(row))
    maps, weights = TrajectorySampler(circuit).compress_no_jump_maps(raw)
    if len(raw) != bank["sampled_paths"] or not np.array_equal(
        np.rint(weights * len(raw)).astype(int), bank["multiplicities"]
    ):
        msg = "Recorded training multiplicities disagree."
        raise ValueError(msg)
    return maps, weights


def main(argv: list[str] | None = None) -> None:
    """Probe a sealed selected checkpoint and save an independent diagnostic file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bond-caps", nargs="+", type=int, default=[4, 8, 16])
    args = parser.parse_args(argv)
    manifest = json.loads((args.case_root / "manifest.json").read_text())
    final = support.load_checkpoint(args.case_root / "final")
    if final is None or final[0]["selected_depth"] is None:
        parser.error("A complete selected checkpoint is required.")
    base = args.case_root / f"depth_{final[0]['selected_depth']:02d}"
    pruning, training = support.load_checkpoint(base / "pruning"), support.load_checkpoint(base / "refinement")
    assert pruning is not None
    assert training is not None
    circuit = support.restore_circuit(pruning[0]["circuit"])
    target = load_target_collection().load(manifest["config"]["qubits"], manifest["target_id"]).state_vector_copy()
    maps, weights = replay_training_bank(circuit, training[0]["training"]["noise_banks"][0])
    report = compare_backends(circuit, training[1]["theta"], target, maps, weights, bond_caps=tuple(args.bond_caps))
    report["source_checkpoint_sha256"] = support.digest(base / "refinement/complete.json")
    support.write_json(args.output, report)


if __name__ == "__main__":
    main()
