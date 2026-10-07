# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen-input compression and sampled-feedback diagnostics, not new optimizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.stats import binomtest, t

from benchmarks.state_preparation.targets import load_target
from experiments.state_preparation_competitors.checkpointed_dense import CheckpointedDenseObjective
from experiments.state_preparation_competitors.iqsp_comb_backend import CombOptions, CombTensorNetworkObjective
from experiments.state_preparation_competitors.iqsp_tci import tci_target_bank
from experiments.state_preparation_competitors.iqsp_tn import iqsp_tn_ansatz, logical_parameterization
from experiments.state_preparation_competitors.iqsp_tn_backend import CircuitTensorGates
from experiments.state_preparation_competitors.noisy_core import nonnegative_integer, real_parameters
from experiments.state_preparation_competitors.robuststate import (
    classical_shadow_feedback,
    robuststate_loss_gradient,
    topology_aware_ansatz,
)
from experiments.state_preparation_exact_adam import AdamConfig, adam_step

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def pure_hilbert_schmidt_gradient(
    engine: CheckpointedDenseObjective, theta: ArrayLike
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate exactly the published ideal Frobenius loss using pure states.

    For normalized pure states L=sqrt(2*(1-F)), hence dL=d(1-F)/L.
    This changes no objective, learning rate, moment, or retention policy.
    """
    if engine.hardware is not None:
        msg = "The pure Hilbert-Schmidt identity requires a noiseless engine."
        raise ValueError(msg)
    infidelity, gradient = engine.loss_gradient(theta)
    if not -1e-12 <= infidelity <= 1 + 1e-12:
        msg = "Invalid pure-state infidelity."
        raise ValueError(msg)
    loss = math.sqrt(2 * max(0.0, infidelity))
    return loss, gradient / loss if loss else np.zeros_like(gradient)


def warm_robuststate(
    engine: CheckpointedDenseObjective, initial: ArrayLike, *, steps: int = 500
) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """Run the original ideal Adam updates, preserving moments for the phase switch."""
    nonnegative_integer(steps, "steps")
    theta = real_parameters(initial, len(engine.parameters))
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    config = AdamConfig(updates=max(1, steps), learning_rate=0.005)
    records, parameters = [], []
    wall, cpu = time.perf_counter(), time.process_time()
    for step in range(steps + 1):
        loss, gradient = pure_hilbert_schmidt_gradient(engine, theta)
        records.append([step, loss, 1 - loss**2 / 2, float(np.linalg.norm(gradient)), time.process_time() - cpu])
        parameters.append(theta.copy())
        if step < steps:
            theta, first, second = adam_step(theta, gradient, first, second, step + 1, config)
    return {
        "theta": theta,
        "first": first,
        "second": second,
        "curve": np.asarray(records),
        "parameter_history": np.asarray(parameters),
    }, {
        "steps": steps,
        "retention": "final_iterate",
        "adam": asdict(config),
        "moment_policy": "preserved for next update",
        "loss": "Hilbert-Schmidt norm",
        "initial_fidelity": records[0][2],
        "final_fidelity": records[-1][2],
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def gradient_comparison(reference: ArrayLike, actual: ArrayLike) -> dict[str, float | None]:
    """Compare finite real gradients without treating zero norms as agreement."""
    ref, value = np.asarray(reference, dtype=float), np.asarray(actual, dtype=float)
    if ref.ndim != 1 or ref.shape != value.shape or not np.isfinite(ref).all() or not np.isfinite(value).all():
        msg = "Require finite, equally sized gradient vectors."
        raise ValueError(msg)
    ref_norm, norm = float(np.linalg.norm(ref)), float(np.linalg.norm(value))
    error = float(np.linalg.norm(value - ref))
    return {
        "reference_norm": ref_norm,
        "norm": norm,
        "absolute_error": error,
        "relative_error": error / ref_norm if ref_norm else None,
        "cosine": float(np.dot(ref, value) / (ref_norm * norm)) if ref_norm and norm else None,
    }


def mean_interval(values: ArrayLike) -> dict[str, Any]:
    """Report a two-sided Student-t interval for independent repeated samples."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        msg = "Intervals require at least two finite independent observations."
        raise ValueError(msg)
    mean = float(data.mean())
    sem = float(data.std(ddof=1) / math.sqrt(data.size))
    half = float(t.ppf(0.975, data.size - 1)) * sem
    positive = binomtest(int(np.count_nonzero(data > 0)), data.size).proportion_ci()
    return {
        "n": data.size,
        "mean": mean,
        "std": float(data.std(ddof=1)),
        "sem": sem,
        "mean_ci95": [mean - half, mean + half],
        "positive_count": int(np.count_nonzero(data > 0)),
        "positive_fraction_ci95": [positive.low, positive.high],
    }


def _write(path: Path, value: dict) -> None:
    """Write strict JSON atomically so interrupted probes retain complete receipts."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _event(name: str, **values: object) -> None:
    """Expose progress from long-running numerical workers."""
    print(json.dumps({"event": name, **values}, allow_nan=False), flush=True)


class CompactCheckpointObjective(CombTensorNetworkObjective):
    """Retain compact checkpoint arrays instead of views into full SVD factors."""

    def _checkpoint_segment(
        self,
        start: int,
        end: int,
        center: int,
        environment: tuple[int, ...],
        theta: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Copy oversized backing views; preserve sharing of already compact tensors."""
        output = super()._checkpoint_segment(start, end, center, environment, theta, *tensors)
        return tuple(
            tensor.clone(memory_format=torch.contiguous_format)
            if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size()
            else tensor
            for tensor in output
        )


def ballarin_probe(
    output: Path,
    *,
    depth: int,
    compression: str,
    checkpoint: Path,
    num_qubits: int = 12,
    max_tensor_elements: int = 2**24,
    checkpoint_interval: int = 1,
    compact_checkpoints: bool = False,
) -> dict:
    """Evaluate one compression setting on a saved, identical logical circuit."""
    if compression not in {"default", "tighter"}:
        msg = "compression must be default or tighter."
        raise ValueError(msg)
    if type(compact_checkpoints) is not bool:
        msg = "compact_checkpoints must be a boolean."
        raise TypeError(msg)
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    wall, cpu = time.perf_counter(), time.process_time()
    source = json.loads(checkpoint.read_text(encoding="utf-8"))
    if source["depth"] != depth or source["num_qubits"] != num_qubits:
        msg = "Checkpoint depth/register does not match the requested circuit."
        raise ValueError(msg)
    target = load_target(num_qubits, "gaussian_mu0p5_sigma0p1").state_vector
    targets, construction = tci_target_bank([target], gaussian_coordinates=[1.0])
    circuit = CircuitTensorGates(iqsp_tn_ansatz(num_qubits, depth)).bind(source["final_theta"])
    template, theta, lowering = logical_parameterization(circuit)
    options = (
        CombOptions(max_tensor_elements=max_tensor_elements, checkpoint_interval=checkpoint_interval)
        if compression == "default"
        else CombOptions(
            closed_bond=128,
            environment_bond=32,
            closed_tolerance=1e-11,
            environment_tolerance=1e-14,
            max_tensor_elements=max_tensor_elements,
            checkpoint_interval=checkpoint_interval,
        )
    )
    backend = CompactCheckpointObjective if compact_checkpoints else CombTensorNetworkObjective
    objective = backend(template, targets[0], "IBM", options=options)
    stem = output / f"ballarin_depth{depth}_{compression}"
    _event("gradient_started", depth=depth, compression=compression)
    started = time.perf_counter()
    loss, gradient = objective.loss_gradient(theta)
    np.savez_compressed(stem.with_suffix(".npz"), theta=theta, gradient=gradient)
    result = {
        "depth": depth,
        "compression": asdict(options),
        "compact_checkpoints": compact_checkpoints,
        "fidelity": 1 - loss,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu + construction["child_cpu_seconds"],
        "wall_seconds": time.perf_counter() - wall,
        "hardware": "IBM",
        "noise_basis": "logical",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
        "target_construction": construction,
        "logical_lowering": lowering,
        "diagnostics": objective.last_diagnostics,
        "diagnostic_source_sha256": source_sha256,
    }
    _write(stem.with_suffix(".json"), result)
    _event("gradient_complete", depth=depth, compression=compression, fidelity=1 - loss)
    return result


def robuststate_probe(
    output: Path,
    *,
    target_id: str,
    trials: int = 32,
    num_qubits: int = 12,
    num_bases: int = 40,
    trial_start: int = 0,
) -> dict:
    """Audit a declared basis budget after 500 unchanged ideal updates.

    The first two predetermined sampled updates and the exact-feedback update
    receive independent exact noisy scoring. These are one-step diagnostics,
    not a shortened replacement for the paper's 50 noisy training steps.
    """
    if type(trials) is not int or trials < 2:
        msg = "Use at least two independent feedback trials."
        raise ValueError(msg)
    if type(num_bases) is not int or not 1 <= num_bases <= 3**num_qubits:
        msg = "num_bases must lie between one and 3**num_qubits."
        raise ValueError(msg)
    nonnegative_integer(trial_start, "trial_start")
    wall, cpu = time.perf_counter(), time.process_time()
    target = load_target(num_qubits, target_id).state_vector
    circuit = topology_aware_ansatz(num_qubits, 1)
    initialization_seed, measurement_seed = map(int, np.random.SeedSequence([300, 1]).generate_state(2))
    initial = np.random.default_rng(initialization_seed).uniform(-np.pi, np.pi, circuit.num_parameters)
    engine = CheckpointedDenseObjective(circuit, target, None, compile_native=False)
    arrays, warm = warm_robuststate(engine, initial)
    stem = output / f"robuststate_{target_id}"
    np.savez_compressed(stem.with_name(stem.name + "_warm.npz"), allow_pickle=False, **arrays)
    _write(
        stem.with_name(stem.name + "_warm.json"),
        {
            **warm,
            "initialization_seed": initialization_seed,
            "target_id": target_id,
            "num_qubits": num_qubits,
            "depth": 1,
            "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        },
    )
    _event("ideal_training_complete", target_id=target_id, fidelity=warm["final_fidelity"])
    theta, first, second = (arrays[name] for name in ("theta", "first", "second"))
    noisy = CheckpointedDenseObjective(circuit, target, "IBM", compile_native=False)
    rho = noisy.density(theta)
    base_fidelity = float(np.vdot(target, rho @ target).real)
    loss, exact_gradient = robuststate_loss_gradient(noisy, theta, feedback_density=rho)
    config = AdamConfig(updates=550, learning_rate=0.005)
    exact_theta, _, _ = adam_step(theta, exact_gradient, first.copy(), second.copy(), 501, config)
    exact_delta = exact_theta - theta
    # Fix the reference score before looking at any sampled update.
    exact_updated_fidelity = noisy.fidelity(exact_theta)
    _event("exact_feedback_complete", target_id=target_id, fidelity=base_fidelity)
    records, gradients, updates = [], [], []
    for trial in range(trial_start, trial_start + trials):
        trial_wall_start = time.perf_counter()
        seed = int(np.random.SeedSequence([measurement_seed, 120000 + num_bases, trial]).generate_state(1)[0])
        feedback = classical_shadow_feedback(
            rho,
            rng=np.random.default_rng(seed),
            num_bases=num_bases,
            shots_per_basis=1024,
            trusted_cptp_output=True,
        )
        sampled_loss, gradient = robuststate_loss_gradient(noisy, theta, feedback_density=feedback)
        del feedback
        candidate, _, _ = adam_step(theta, gradient, first.copy(), second.copy(), 501, config)
        delta = candidate - theta
        row = {
            "trial": trial,
            "seed": seed,
            "sampled_loss": sampled_loss,
            "gradient": gradient_comparison(exact_gradient, gradient),
            "update": gradient_comparison(exact_delta, delta),
            "surrogate_descent": float(-np.dot(exact_gradient, delta)),
            "sampling_wall_seconds": time.perf_counter() - trial_wall_start,
            "independent_noisy_fidelity_after": noisy.fidelity(candidate) if trial < 2 else None,
        }
        records.append(row)
        gradients.append(gradient)
        updates.append(delta)
        np.savez_compressed(
            stem.with_name(stem.name + "_feedback.npz"),
            exact_gradient=exact_gradient,
            exact_update=exact_delta,
            gradients=np.asarray(gradients),
            updates=np.asarray(updates),
        )
        _write(stem.with_name(stem.name + "_progress.json"), {"records": records, "warm": warm})
        _event("sample_complete", target_id=target_id, trial=trial, cosine=row["gradient"]["cosine"])
    result = {
        "target_id": target_id,
        "num_qubits": num_qubits,
        "depth": 1,
        "hardware": "IBM",
        "noise_basis": "logical",
        "warm": warm,
        "measurement_seed": measurement_seed,
        "initialization_seed": initialization_seed,
        "theta_sha256": hashlib.sha256(theta.tobytes()).hexdigest(),
        "first_moment_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "second_moment_sha256": hashlib.sha256(second.tobytes()).hexdigest(),
        "trials": trials,
        "trial_start": trial_start,
        "bases": num_bases,
        "shots_per_basis": 1024,
        "shots_per_update": num_bases * 1024,
        "exact_noisy_fidelity_before": base_fidelity,
        "exact_feedback_loss": loss,
        "exact_feedback_fidelity_after": exact_updated_fidelity,
        "exact_feedback_surrogate_descent": float(-np.dot(exact_gradient, exact_delta)),
        "mean_gradient_comparison": gradient_comparison(exact_gradient, np.mean(gradients, axis=0)),
        "mean_update_comparison": gradient_comparison(exact_delta, np.mean(updates, axis=0)),
        "surrogate_descent": mean_interval([row["surrogate_descent"] for row in records]),
        "gradient_cosine": mean_interval([row["gradient"]["cosine"] for row in records]),
        "records": records,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }
    _write(stem.with_suffix(".json"), result)
    _event("robuststate_complete", target_id=target_id)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run one independently bounded probe; never launch a comparison matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ballarin", "robuststate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--compression", choices=("default", "tighter"), default="default")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-id", default="haar_random_1")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--num-bases", type=int, default=40)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--max-tensor-elements", type=int, default=2**24)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--compact-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "ballarin":
        if args.checkpoint is None:
            parser.error("Ballarin requires a frozen checkpoint")
        ballarin_probe(
            args.output,
            depth=args.depth,
            compression=args.compression,
            checkpoint=args.checkpoint,
            max_tensor_elements=args.max_tensor_elements,
            checkpoint_interval=args.checkpoint_interval,
            compact_checkpoints=args.compact_checkpoints,
        )
    else:
        robuststate_probe(
            args.output,
            target_id=args.target_id,
            trials=args.trials,
            num_bases=args.num_bases,
            trial_start=args.trial_start,
        )


if __name__ == "__main__":
    main()
