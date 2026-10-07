# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Paired rerun of saved q8 fine-tuning with 100 compressed noisy histories.

Construction, pruning, learning rates, step counts, seeds, and CRN schedules
are read from frozen checkpoints. Only the sampled trajectory count and exact
duplicate-history compression change. Direct noisy-from-scratch VQA is retained
as a historical reference, outside this fine-tuning intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.legacy_topdown_dense import LegacyNoise, Recorder, train
from mqt.yaqs.optimization import KrotovTJMOptions, ParameterizedCircuit, ParameterizedGate, StatevectorBackend

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "output/legacy-topdown-dense-20260914",
    "output/legacy-topdown-dense-20260914-seed400",
    "output/legacy-topdown-dense-20260914-baselines400",
)


def file_hash(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_circuit(record: dict) -> ParameterizedCircuit:
    """Reconstruct the exact stored gate sequence without pruning or synthesis."""
    gates = []
    for saved in record["circuit"]["gates"]:
        gate = dict(saved)
        gate["sites"] = tuple(gate["sites"])
        gates.append(ParameterizedGate(**gate))
    return ParameterizedCircuit(record["circuit"]["num_qubits"], gates, record["circuit"]["num_params"])


def discover_cases() -> list[tuple[Path, Path]]:
    """Find all 13 saved fine-tuning endpoints and authenticate their source JSON."""
    audit = json.loads((REPOSITORY / SOURCE_ROOTS[0] / "verification.json").read_text())
    expected = {entry["file"]: entry["sha256"] for entry in audit["checkpoints"]}
    cases = []
    for root in SOURCE_ROOTS:
        for path in sorted((REPOSITORY / root).glob("*/*.json")):
            record = json.loads(path.read_text())
            trace = record["training"]
            if not isinstance(trace, dict) or not trace.get("noisy") or path.stem == "fixed_vqa":
                continue
            if path.stem == "topdown_final":
                name = "topdown_before_noise"
            elif path.stem == "layerwise_final":
                name = "layerwise_before_noise"
            elif path.stem.startswith("cutoff_"):
                name = "cutoff_before_noise"
            elif path.stem.startswith("unpruned_control_"):
                name = "before_pruning"
            else:
                msg = f"Unexpected fine-tuning endpoint: {path}"
                raise ValueError(msg)
            initial = path.with_name(name + ".json")
            for source in (path, initial):
                if file_hash(source) != expected[str(source.relative_to(REPOSITORY))]:
                    msg = f"Historical checkpoint authentication failed: {source}"
                    raise ValueError(msg)
            cases.append((initial, path))
    if len(cases) != 13:
        msg = f"Expected 13 fixed fine-tuning comparisons, found {len(cases)}."
        raise ValueError(msg)
    return cases


def compression_diagnostic(circuit: ParameterizedCircuit, theta: np.ndarray, target: np.ndarray, seed: int) -> dict:
    """Compare full and compressed 100-path cross kernels on identical histories."""
    engine = StatevectorBackend(circuit, target)
    sampling_start = time.perf_counter()
    raw = engine.sample_noise_maps(theta, KrotovTJMOptions(num_trajectories=100), LegacyNoise(), iteration=seed)
    sampling_seconds = time.perf_counter() - sampling_start
    compression_start = time.perf_counter()
    compressed, weights = engine.compress_no_jump_maps(raw)
    compression_seconds = time.perf_counter() - compression_start
    times = {"full": [], "compressed": []}
    values = {}
    for repeat in range(3):
        order = ("full", "compressed") if repeat % 2 == 0 else ("compressed", "full")
        for name in order:
            start = time.perf_counter()
            values[name] = engine.gradient(
                theta,
                raw if name == "full" else compressed,
                estimator="cross",
                weights=None if name == "full" else weights,
            )
            times[name].append(time.perf_counter() - start)
    full_theta, full_signal = engine.online_update(theta, 0.2, raw, estimator="cross")
    short_theta, short_signal = engine.online_update(theta, 0.2, compressed, estimator="cross", weights=weights)
    errors = {
        "gradient_max": float(np.max(np.abs(values["full"][0] - values["compressed"][0]))),
        "fidelity": abs(values["full"][2] - values["compressed"][2]),
        "online_theta_max": float(np.max(np.abs(full_theta - short_theta))),
        "online_signal_max": float(np.max(np.abs(full_signal - short_signal))),
    }
    assert max(errors.values()) < 1e-11
    return {
        "sampled": len(raw),
        "retained": len(compressed),
        "weights": weights.tolist(),
        "sampling_seconds": sampling_seconds,
        "compression_seconds": compression_seconds,
        "gradient_seconds": times,
        "errors": errors,
        "median_gradient_speedup": statistics.median(times["full"]) / statistics.median(times["compressed"]),
    }


def run_case(initial_path: Path, old_final_path: Path, output: Path) -> dict:
    """Rerun the original and 100-path fine-tuning from one identical checkpoint."""
    before = json.loads(initial_path.read_text(encoding="utf-8"))
    previous = json.loads(old_final_path.read_text(encoding="utf-8"))
    assert before["circuit"] == previous["circuit"]
    target = np.load(initial_path.parent / "target.npy")
    theta = np.load(initial_path.with_suffix(".theta.npy"))
    expected_theta = np.load(old_final_path.with_suffix(".theta.npy"))
    circuit = load_circuit(before)
    case_id = old_final_path.parent.name + "__" + old_final_path.stem
    root = output / case_id
    if (root / "comparison.json").exists():
        return json.loads((root / "comparison.json").read_text())
    recorder = Recorder(root, target, resume=True)
    np.save(root / "initial.theta.npy", theta)
    (root / "initial.json").write_text(json.dumps(before, indent=2) + "\n")
    old = previous["training"]
    settings: dict[str, Any] = {
        "steps": old["steps"],
        "rate": old["rate"],
        "seed": old["seed"],
        "decay": old["decay"],
        "noisy": True,
        "crn": old["crn"],
        "online": old["online_steps"],
        "hardware": old["hardware"],
        "record_noise_banks": True,
    }
    if (root / "baseline3.json").exists():
        baseline = json.loads((root / "baseline3.json").read_text())["training"]
    else:
        baseline_theta, baseline = train(circuit, theta, target, **settings)
        error = float(np.max(np.abs(baseline_theta - expected_theta))) if len(theta) else 0.0
        assert error < 1e-10, (case_id, error)
        np.save(root / "baseline3.theta.npy", baseline_theta)
        copied = {**previous, "training": baseline, "historical_parameter_max_error": error}
        (root / "baseline3.json").write_text(json.dumps(copied, indent=2) + "\n")
    print("fine-tuning 100:", case_id, "gates", len(circuit.gates), flush=True)
    updated, trace = train(
        circuit, theta, target, **settings, num_trajectories=100, compress_no_jump=True, require_jump_coverage=True
    )
    assert [bank["first_three_sha256"] for bank in baseline["noise_banks"]] == [
        bank["first_three_sha256"] for bank in trace["noise_banks"]
    ]
    recorder.save("weighted100", circuit, updated, trace)
    result = json.loads((root / "weighted100.json").read_text())
    comparison = {
        "case": case_id,
        "source_initial": str(initial_path),
        "source_previous": str(old_final_path),
        "gate_counts": result["gate_counts"],
        "hardware_trained": old["hardware"],
        "fidelity_before": before["noise"][old["hardware"]]["tjm"]["fidelity"],
        "fidelity_three": previous["noise"][old["hardware"]]["tjm"]["fidelity"],
        "fidelity_hundred": result["noise"][old["hardware"]]["tjm"]["fidelity"],
        "training_fidelity_three": baseline["fidelity_trace"][-1],
        "training_fidelity_hundred": trace["fidelity_trace"][-1],
        "training_wall_three": baseline["wall_seconds"],
        "training_wall_hundred": trace["wall_seconds"],
        "banks": len(trace["noise_banks"]),
        "retained_paths": [bank["retained_paths"] for bank in trace["noise_banks"]],
        "jump_paths": [bank["jump_paths"] for bank in trace["noise_banks"]],
        "first_three_histories_preserved": True,
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def main() -> None:
    """Run the fixed thirteen-case matrix with authenticated resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = discover_cases()
    code = [
        Path(__file__),
        REPOSITORY / "experiments/legacy_topdown_dense.py",
        REPOSITORY / "src/mqt/yaqs/optimization/statevector.py",
    ]
    inputs = set()
    for initial, final in cases:
        inputs.update((
            initial,
            final,
            initial.with_suffix(".theta.npy"),
            final.with_suffix(".theta.npy"),
            initial.parent / "target.npy",
        ))
    manifest = {
        "version": 1,
        "sampled_trajectories": 100,
        "compression": "empty operator histories only; empirical multiplicities",
        "code": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in code},
        "inputs": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sorted(inputs)},
        "scope": (
            "Only noisy fine-tuning; all construction and pruning frozen; "
            "retain original CRN schedules; no rejected/resampled banks."
        ),
    }
    if args.output.exists():
        if not args.resume or json.loads((args.output / "manifest.json").read_text()) != manifest:
            msg = "Use a fresh output directory, or --resume with identical source and inputs."
            raise ValueError(msg)
    else:
        args.output.mkdir(parents=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in code:
            (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    with threadpool_limits(limits=1):
        probe = next(
            pair for pair in cases if pair[1].parent.name == "archived_tfim_400" and pair[1].stem == "topdown_final"
        )
        if not (args.output / "compression_diagnostic.json").exists():
            initial = json.loads(probe[0].read_text())
            diagnostic = compression_diagnostic(
                load_circuit(initial),
                np.load(probe[0].with_suffix(".theta.npy")),
                np.load(probe[0].parent / "target.npy"),
                5400,
            )
            (args.output / "compression_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        results = []
        for initial, final in cases:
            results.append(run_case(initial, final, args.output))
            (args.output / "comparisons.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
