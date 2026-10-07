# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matched full-refinement truncation study with exact paired held-out assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation.targets import load_target_collection
from experiments import topdown_backend_support as support
from experiments.topdown_backend_probe import replay_training_bank
from experiments.topdown_depth_search import noise_outcomes, sample_validation
from experiments.topdown_mps import diagnostics, measure_resources, source_hashes, target_digest
from mqt.yaqs.optimization import KrotovNoiseMap, KrotovTruncation, MPSBackend, StatevectorBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization.backends import PreparationBackend
    from mqt.yaqs.optimization.trajectories import NoiseMaps

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "statevector": None,
    "mps_exact": {},
    **{f"cap_{cap}": {"max_bond_dim": cap} for cap in (2, 4, 8, 16, 32)},
    **{f"threshold_{power}": {"svd_threshold": 10.0**-power} for power in (8, 6, 4)},
}


def fine_tune_fixed(
    engine: PreparationBackend,
    initial: NDArray,
    maps: NoiseMaps,
    weights: NDArray,
    *,
    steps: int = 200,
    online: int = 100,
    rate: float = 0.2,
) -> dict[str, NDArray]:
    """Replay the current noisy policy; retain final iterate and every parameter row.

    Both phases use the unchanged weighted cross-trajectory signal. This is
    the frozen noisy stage, not the noiseless L-BFGS construction stage.

    Returns:
        Parameters after each update and backend-reported training fidelities.

    Raises:
        ValueError: If update budgets or the rate are invalid.
        FloatingPointError: If an update produces nonfinite parameters or metrics.
    """
    if steps < 0 or not 0 <= online <= steps or not np.isfinite(rate) or rate <= 0:
        msg = "Invalid refinement budget or rate."
        raise ValueError(msg)
    theta = initial.copy()
    parameters = np.empty((steps + 1, len(theta)))
    parameters[0] = theta
    fidelity = np.empty(steps)
    for k in range(steps):
        if k < online:
            theta, _ = engine.online_update(theta, rate, maps, estimator="cross", weights=weights)
        else:
            signal, _, _ = engine.gradient(theta, maps, estimator="cross", weights=weights)
            theta -= rate * signal
        fidelity[k] = engine.metrics(theta, maps, weights=weights)[1]
        parameters[k + 1] = theta
        if not np.all(np.isfinite(theta)) or not np.isfinite(fidelity[k]):
            msg = "Nonfinite truncated refinement."
            raise FloatingPointError(msg)
    return {"parameters": parameters, "reported_fidelity": fidelity}


def weighted_statistics(values: NDArray, counts: NDArray) -> dict:
    """Compute raw-sample mean/SE, also valid for paired signed differences.

    Returns:
        Mean, standard error and an unclipped approximate normal 95% interval.

    Raises:
        ValueError: If counts are not positive integer multiplicities or values fail.
    """
    if (
        values.ndim != 1
        or values.shape != counts.shape
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or np.sum(counts) < 2
        or not np.all(np.isfinite(values))
    ):
        msg = "Invalid weighted observations."
        raise ValueError(msg)
    total = int(counts.sum())
    mean = float(np.dot(values, counts) / total)
    se = float(np.sqrt(np.dot(counts, (values - mean) ** 2) / (total * (total - 1))))
    return {"mean": mean, "standard_error": se, "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se]}


def assessment_values(engine: PreparationBackend, theta: NDArray, codes: NDArray, *, hardware: str = "IBM") -> NDArray:
    """Evaluate each distinct unconditional validation history exactly once.

    The caller retains raw multiplicities for paired uncertainty calculations.
    This extra deduplication is confined to assessment, never optimizer updates.

    Returns:
        One fidelity per supplied code row, in the original row order.

    Raises:
        ValueError: If codes or evaluated fidelities are invalid.
    """
    circuit = engine.circuit
    if codes.ndim != 2 or codes.shape[1] != len(circuit.gates) or codes.dtype.kind not in "iu" or np.any(codes < 0):
        msg = "Invalid assessment codes."
        raise ValueError(msg)
    outcomes = [noise_outcomes(hardware, g.sites) for g in circuit.gates]
    for k, (gate, (probabilities, _)) in enumerate(zip(circuit.gates, outcomes, strict=True)):
        if np.any(codes[:, k] >= len(probabilities)) or (not gate.noise_enabled and np.any(codes[:, k])):
            msg = "Invalid assessment jump code."
            raise ValueError(msg)
    result = []
    for codes_row in codes:
        row = tuple(
            KrotovNoiseMap(operators=() if code == 0 else (outcomes[k][1][int(code) - 1],))
            for k, code in enumerate(codes_row)
        )
        result.append(engine.metrics(theta, [row])[1])
    values = np.asarray(result)
    if not np.all(np.isfinite(values)) or np.any(values < -1e-8) or np.any(values > 1 + 1e-8):
        msg = "Nonphysical assessment fidelity."
        raise ValueError(msg)
    return values


def prepare(root: Path, case_root: Path, depths: list[int]) -> None:
    """Freeze source receipts, initial circuits, raw maps and fresh held-out banks.

    Raises:
        FileExistsError: If the output protocol already exists.
        ValueError: If input checkpoints or recorded training settings disagree.
    """
    if (root / "protocol.json").exists():
        msg = "Use a fresh output directory."
        raise FileExistsError(msg)
    manifest = json.loads((case_root / "manifest.json").read_text())
    qubits, target_id = manifest["config"]["qubits"], manifest["target_id"]
    target = load_target_collection().load(qubits, target_id).state_vector_copy()
    if (
        target_digest(target) != manifest["target_sha256"]
        or support.digest(ROOT / "benchmarks/state_preparation_target_states.json") != manifest["fixture_sha256"]
    ):
        msg = "Source target fixture changed."
        raise ValueError(msg)
    for depth in depths:
        base = case_root / f"depth_{depth:02d}"
        pruned, trained = support.load_checkpoint(base / "pruning"), support.load_checkpoint(base / "refinement")
        if pruned is None or trained is None:
            msg = "Complete pruning and refinement checkpoints are required."
            raise ValueError(msg)
        training = trained[0]["training"]
        expected = {
            "steps": 200,
            "online_steps": 100,
            "rate": 0.2,
            "decay": 0.0,
            "num_trajectories": 100,
            "compress_no_jump": True,
            "crn": True,
            "noisy": True,
            "hardware": "IBM",
        }
        if any(training[k] != v for k, v in expected.items()) or len(training["noise_banks"]) != 1:
            msg = "Stored training settings disagree with this matched study."
            raise ValueError(msg)
        circuit = support.restore_circuit(pruned[0]["circuit"])
        bank = training["noise_banks"][0]
        replay_training_bank(circuit, bank)
        entropy = [20260923, 92173, qubits, depth]
        codes = sample_validation(circuit, "IBM", 32768, entropy)
        support.save_checkpoint(
            root / f"input_depth{depth}",
            {
                "circuit": pruned[0]["circuit"],
                "training_bank": bank,
                "source_pruning": str(base / "pruning"),
                "source_refinement": str(base / "refinement"),
                "source_seals": {
                    phase: support.digest(base / phase / "complete.json") for phase in ("pruning", "refinement")
                },
                "target_id": target_id,
                "depth": depth,
                "qubits": qubits,
                "validation_entropy": entropy,
                "validation_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                "validation_jump_paths": int(np.count_nonzero(np.any(codes, axis=1))),
            },
            {"initial": pruned[1]["theta"], "target": target, "validation_codes": codes},
        )
    sources = source_hashes()
    for name in ("topdown_mps_truncation", "topdown_backend_probe"):
        relative = f"experiments/{name}.py"
        sources[relative] = support.digest(ROOT / relative)
    for relative in sources:
        destination = root / "source" / (relative + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    support.write_json(
        root / "protocol.json",
        {
            "depths": depths,
            "variants": VARIANTS,
            "sources": sources,
            "source_case": str(case_root.resolve()),
            "steps": 200,
            "online": 100,
            "rate": 0.2,
            "noisy_batch_policy": "unchanged weighted cross Krotov",
            "returned_iterate": "final",
            "training_M": 100,
            "validation_M": 32768,
            "hardware": "IBM logical",
            "target_compression": False,
            "trunc_mode": "discarded_weight",
            "min_bond_dim": 1,
            "fixture_sha256": support.digest(ROOT / "benchmarks/state_preparation_target_states.json"),
            "thread_limit": 1,
            "sequential_fresh_processes": True,
            "allocation_tracing": False,
        },
    )


def run_worker(root: Path, depth: int, variant: str) -> dict:
    """Train one fresh-process variant, then assess using exact statevectors.

    Returns:
        A sealed record referencing compact parameter, curve and assessment arrays.

    Raises:
        ValueError: If a source or input protocol changed.
    """
    protocol = json.loads((root / "protocol.json").read_text())
    if any(support.digest(ROOT / path) != sha for path, sha in protocol["sources"].items()):
        msg = "Study numerical sources changed."
        raise ValueError(msg)
    if depth not in protocol["depths"] or variant not in protocol["variants"]:
        msg = "Case is outside the frozen study."
        raise ValueError(msg)
    loaded = support.load_checkpoint(root / f"input_depth{depth}")
    assert loaded is not None
    inputs, arrays = loaded
    circuit = support.restore_circuit(inputs["circuit"])
    maps, weights = replay_training_bank(circuit, inputs["training_bank"])
    settings = protocol["variants"][variant]
    with threadpool_limits(limits=1):
        start, wall = time.process_time(), time.perf_counter()
        with measure_resources(trace_allocations=False) as resources:
            engine = (
                StatevectorBackend(circuit, arrays["target"])
                if settings is None
                else MPSBackend(
                    circuit, arrays["target"], truncation=KrotovTruncation(**settings), compress_target=False
                )
            )
            fit = fine_tune_fixed(engine, arrays["initial"], maps, weights)
        cpu_seconds, wall_seconds = time.process_time() - start, time.perf_counter() - wall
        train_diagnostics = diagnostics(engine)
        start = time.process_time()
        exact = StatevectorBackend(circuit, arrays["target"])
        fit["exact_training_fidelity"] = np.array([
            exact.metrics(theta, maps, weights=weights)[1] for theta in fit["parameters"]
        ])
        unique, counts = np.unique(arrays["validation_codes"], axis=0, return_counts=True)
        fit["assessment_values"] = assessment_values(exact, fit["parameters"][-1], unique)
        evaluation_cpu = time.process_time() - start
        # Fixed-input signal probes occur after all training/evaluation timings.
        signal, _, _ = engine.gradient(arrays["initial"], maps, estimator="cross", weights=weights)
        reference, _, _ = exact.gradient(arrays["initial"], maps, estimator="cross", weights=weights)
        error = float(np.linalg.norm(signal - reference))
    record = {
        "variant": variant,
        "depth": depth,
        "training_cpu_seconds": cpu_seconds,
        "training_wall_seconds": wall_seconds,
        "evaluation_cpu_seconds": evaluation_cpu,
        "resources": resources,
        "diagnostics": train_diagnostics,
        "heldout": weighted_statistics(fit["assessment_values"], counts),
        "training_report_bias_final": float(fit["reported_fidelity"][-1] - fit["exact_training_fidelity"][-1]),
        "signal_max_error_initial": float(np.max(abs(signal - reference), initial=0)),
        "signal_relative_l2_error_initial": error / max(float(np.linalg.norm(reference)), 1e-30),
        "training_bank_sha256": inputs["training_bank"]["raw_sha256"],
        "input_seal_sha256": support.digest(root / f"input_depth{depth}/complete.json"),
        "validation_unique_paths": len(unique),
        "validation_jump_paths": inputs["validation_jump_paths"],
        "validation_sha256": inputs["validation_sha256"],
        "python": sys.version,
        "numpy": np.__version__,
    }
    return support.save_checkpoint(root / f"depth{depth}/{variant}", record, fit)


def summarize(root: Path) -> dict:
    """Compare exact held-out fidelities with paired trajectory uncertainty.

    Returns:
        All completed variant results, parameter/curve errors and paired intervals.
    """
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for depth in protocol["depths"]:
        inputs = support.load_checkpoint(root / f"input_depth{depth}")
        baseline = support.load_checkpoint(root / f"depth{depth}/statevector")
        assert inputs is not None
        assert baseline is not None
        _, counts = np.unique(inputs[1]["validation_codes"], axis=0, return_counts=True)
        for variant in protocol["variants"]:
            loaded = support.load_checkpoint(root / f"depth{depth}/{variant}")
            if loaded is None:
                continue
            record, arrays = loaded
            assert record["input_seal_sha256"] == baseline[0]["input_seal_sha256"]
            record["paired_difference"] = weighted_statistics(
                arrays["assessment_values"] - baseline[1]["assessment_values"], counts
            )
            record["final_parameter_max_error"] = float(
                np.max(abs(arrays["parameters"][-1] - baseline[1]["parameters"][-1]), initial=0)
            )
            record["exact_training_curve_max_difference"] = float(
                np.max(abs(arrays["exact_training_fidelity"] - baseline[1]["exact_training_fidelity"]), initial=0)
            )
            record["speedup_vs_statevector"] = baseline[0]["training_cpu_seconds"] / record["training_cpu_seconds"]
            rows.append(record)
    result = {"rows": rows, "complete": len(rows) == len(protocol["variants"]) * len(protocol["depths"])}
    support.write_json(root / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Prepare once, execute isolated sequential workers, or summarize saved arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path, metavar="SOURCE_CASE")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--variant", choices=list(VARIANTS))
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args(argv)
    if args.prepare:
        prepare(args.root, args.prepare, args.depths)
    elif args.variant:
        record = run_worker(args.root, args.depth, args.variant)
        print(json.dumps({k: record[k] for k in ("variant", "depth", "training_cpu_seconds", "heldout")}), flush=True)
    elif args.summarize:
        summarize(args.root)
    else:
        protocol = json.loads((args.root / "protocol.json").read_text())
        environment = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "QISKIT_NUM_PROCS",
        ):
            environment[name] = "1"
        for depth in protocol["depths"]:
            for variant in ("statevector", *(v for v in protocol["variants"] if v != "statevector")):
                if support.load_checkpoint(args.root / f"depth{depth}/{variant}") is not None:
                    continue
                print(f"Starting depth {depth}: {variant}", flush=True)
                # Fixed Python module, structured arguments, and no shell execution.
                subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "experiments.topdown_mps_truncation",
                        "--root",
                        str(args.root),
                        "--depth",
                        str(depth),
                        "--variant",
                        variant,
                    ],
                    check=True,
                    env=environment,
                    cwd=ROOT,
                )
        summarize(args.root)


if __name__ == "__main__":
    main()
