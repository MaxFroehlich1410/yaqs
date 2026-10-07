# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Repeat the small hybrid diagnostic with hardware-error-calibrated noise."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import t as student_t

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.targets import load_target
from experiments.noisy_krotov_causal_study import create_circuit, derive_study_seed
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovTJMOptions,
    KrotovTruncation,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
    train_krotov_noisy_state_preparation_hybrid,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _digest(value: object) -> str:
    """Return a stable configuration checksum."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _summary(values: NDArray[np.float64]) -> dict[str, object]:
    """Summarize independent trajectories or their paired differences."""
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1))
    error = deviation / np.sqrt(len(values))
    half_width = float(student_t.ppf(0.975, len(values) - 1) * error)
    return {
        "count": len(values),
        "mean": mean,
        "sample_standard_deviation": deviation,
        "standard_error": float(error),
        "confidence_interval_95_student_t": [mean - half_width, mean + half_width],
    }


def run_hybrid_smoke(
    output_root: Path,
    source_root: Path,
    *,
    max_iterations: int = 20,
    switch_iteration: int = 10,
    training_trajectories: int = 3,
    test_trajectories: int = 200,
) -> dict[str, object]:
    """Rerun the historical hybrid settings from a verified source checkpoint.

    Returns:
        Configuration, training trace, paired evaluation, and artifact checksums.

    Raises:
        ValueError: If inputs, checkpoint identity, or output location are invalid.
    """
    if (
        any(
            type(value) is not int
            for value in (max_iterations, switch_iteration, training_trajectories, test_trajectories)
        )
        or max_iterations < 1
        or not 0 < switch_iteration <= max_iterations
        or training_trajectories < 1
        or test_trajectories < 2
    ):
        msg = "Invalid hybrid iteration or trajectory budget."
        raise ValueError(msg)
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        msg = "Hybrid output must be missing or empty; existing results are never overwritten."
        raise ValueError(msg)
    source_config = json.loads((source_root / "config.json").read_text())
    if (source_config["qubits"], source_config["target_id"], source_config["depths"]) != (
        6,
        "tfim_critical",
        [1, 2, 3, 4],
    ):
        msg = "Hybrid source must be the canonical q6 depth-four TFIM study."
        raise ValueError(msg)
    source_result = json.loads((source_root / "causal_study_result.json").read_text())
    with np.load(source_root / "pretraining" / "pretrained_checkpoint.npz", allow_pickle=False) as archive:
        initial = np.asarray(archive["theta"], dtype=np.float64).copy()
    initial_checksum = f"sha256:{hashlib.sha256(np.ascontiguousarray(initial, dtype='<f8').tobytes()).hexdigest()}"
    if initial_checksum != source_result["pretraining"]["theta_checksum"] or not np.all(np.isfinite(initial)):
        msg = "The source pretrained checkpoint checksum is invalid."
        raise ValueError(msg)
    circuit = create_circuit(6, 4)
    if initial.shape != (circuit.num_params,):
        msg = "The source checkpoint has an incompatible parameter shape."
        raise ValueError(msg)
    target = load_target(6, "tfim_critical").state_vector_copy()
    truncation = KrotovTruncation()
    provider = CalibratedLocalDepolarizingProvider()
    circuit_checksum = _digest({
        "qubits": circuit.num_qubits,
        "parameters": circuit.num_params,
        "gates": [
            {
                "name": gate.name,
                "sites": list(gate.sites),
                "parameter": gate.param_index,
                "scale": gate.angle_scale,
                "offset": gate.angle_offset,
                "fixed": list(gate.fixed_params),
                "noise_enabled": gate.noise_enabled,
            }
            for gate in circuit.gates
        ],
    })
    configuration = {
        "schema_version": "yaqs.noisy_krotov.hybrid_smoke.calibrated.v1",
        "source_root": str(source_root.resolve()),
        "initial_theta_checksum": initial_checksum,
        "circuit_checksum": circuit_checksum,
        "noise_calibration": provider.scientific_dict(),
        "noise_provider_checksum": provider.content_checksum,
        "max_iterations": max_iterations,
        "switch_iteration": switch_iteration,
        "online_step_size": 0.1,
        "batch_step_size": 0.1,
        "online_schedule": "constant",
        "batch_schedule": "constant",
        "training_trajectories": training_trajectories,
        "test_trajectories": test_trajectories,
        "training_seed": 314159,
        "optimizer_seed": 271828,
        "test_root_seed": 20260905,
        "sampling_policy": "resampled_each_update",
        "trajectory_update": "cross",
        "qubits": 6,
        "target_id": "tfim_critical",
        "depth": 4,
    }
    checksum = _digest(configuration)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    started = time.perf_counter()
    trained = train_krotov_noisy_state_preparation_hybrid(
        circuit,
        target,
        None,
        KrotovTJMOptions(
            num_trajectories=training_trajectories, random_seed=314159, trajectory_update="cross", use_crn=False
        ),
        initial_theta=initial,
        options=KrotovOptions(
            max_iterations=max_iterations, switch_iteration=switch_iteration, seed=271828, truncation=truncation
        ),
        noise_provider=provider,
    )
    training_seconds = time.perf_counter() - started
    np.savez_compressed(output_root / "hybrid_checkpoint.npz", theta=trained.theta, initial_theta=initial)
    print(f"Hybrid training completed in {training_seconds:.1f}s; evaluating paired held-out trajectories.", flush=True)
    test_options = KrotovTJMOptions(num_trajectories=test_trajectories)
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        initial,
        None,
        truncation,
        provider,
        test_options,
        role="pilot_evaluation",
        resolved_seed=derive_study_seed(20260905, "pilot_evaluation", 0),
        stage_index=0,
        stage_id="hybrid_smoke_test",
        stage_configuration_checksum=checksum,
        circuit_checksum=circuit_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
    )
    (output_root / "test_ensemble.json").write_text(ensemble.to_json() + "\n")
    fidelities = {}
    for name, theta in (("frozen", initial), ("hybrid", trained.theta)):
        values = noisy_state_preparation_metrics(
            circuit,
            theta,
            target,
            None,
            test_options,
            truncation=truncation,
            noise_provider=provider,
            fixed_noise_maps=ensemble.replay_maps(),
        )[2]
        fidelities[name] = np.asarray(values, dtype=np.float64)
    differences = fidelities["hybrid"] - fidelities["frozen"]
    result: dict[str, object] = {
        "status": "success",
        "scope": "small untuned diagnostic",
        "configuration": configuration,
        "configuration_checksum": checksum,
        "training": {**trained.trace, "wall_time_seconds": training_seconds},
        "evaluation": {
            **{name: _summary(values) for name, values in fidelities.items()},
            "hybrid_minus_frozen": _summary(differences),
            "frozen_noiseless_fidelity": state_preparation_metrics(circuit, initial, target, truncation=truncation)[1],
            "hybrid_noiseless_fidelity": state_preparation_metrics(
                circuit, trained.theta, target, truncation=truncation
            )[1],
            "same_maps_used_for_both_circuits": True,
            "test_ensemble_checksum": ensemble.content_checksum,
            "rows": [
                {
                    "trajectory_index": index,
                    "frozen": float(frozen),
                    "hybrid": float(hybrid),
                    "hybrid_minus_frozen": float(hybrid - frozen),
                }
                for index, (frozen, hybrid) in enumerate(zip(fidelities["frozen"], fidelities["hybrid"], strict=True))
            ],
        },
        "process_wall_time_seconds": time.perf_counter() - started,
    }
    result["content_checksum"] = _digest(result)
    (output_root / "hybrid_smoke_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Run the calibrated diagnostic from a canonical pretraining source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    run_hybrid_smoke(args.output_root, args.source_root)
    print(args.output_root / "hybrid_smoke_result.json")


if __name__ == "__main__":
    main()
