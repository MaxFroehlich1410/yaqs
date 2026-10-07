# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact mean-two refinement with independently compiled deployable checkpoints."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from experiments import state_preparation_exact_adam as adam
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import ensemble_experiment, initializer_policy, optimizers

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

VERSION = "overnight.initializer_mean_two_refinement.v1"
PROFILES = ("depolarizing_1s_all", "dephasing_1s_all")


@dataclass(frozen=True)
class Config:
    """Fixed scientific defaults; smaller caps support independent engineering tests."""

    joint_cap: int = 550
    history: int = 10
    checkpoint_every: int = 10
    adam_rate: float = 0.02

    def __post_init__(self) -> None:
        """Reject invalid work caps or optimizer constants."""
        if (
            any(type(v) is not int or v < 1 for v in (self.joint_cap, self.history, self.checkpoint_every))
            or isinstance(self.adam_rate, bool)
            or not np.isfinite(self.adam_rate)
            or self.adam_rate <= 0
        ):
            msg = "Initializer refinement requires positive integral caps and a finite positive Adam rate."
            raise ValueError(msg)


def deployment(root: Path, circuit: dict[str, Any], theta: list[float], *, initial: bool = False) -> str:
    """Return a content-addressed physical endpoint, compiling every trained checkpoint."""
    signature = pilot.digest({"circuit": circuit, "theta": theta, "initial": initial})
    path = root / "deployments" / f"{signature}.json"
    if path.exists():
        if pilot.read_record(path)["signature"] != signature:
            msg = "Deployable checkpoint identity changed."
            raise ValueError(msg)
        return signature
    result = (
        {"circuit": circuit, "theta": theta, "initial": True}
        if initial
        else initializer_policy.compile_source({"circuit": circuit, "theta": theta})
    )
    pilot.write_record(path, {"signature": signature, "endpoint": result})
    return signature


def train(
    root: Path, source: dict[str, Any], target: NDArray[np.complex128], method: str, config: Config
) -> dict[str, Any]:
    """Return or resume full optimizer state and sparse compiled-checkpoint clocks."""
    if method not in {"adam", "lbfgs"}:
        msg = "Initializer refinement supports exact Adam or L-BFGS only."
        raise ValueError(msg)
    signature = pilot.digest({
        "version": VERSION,
        "source": source,
        "target": hashlib.sha256(target.tobytes()).hexdigest(),
        "method": method,
        "config": asdict(config),
    })
    final_path, state_path, attempts_path = (root / name for name in ("TRAINING.json", "STATE.json", "ATTEMPTS.json"))
    if final_path.exists():
        result = cast("dict[str, Any]", pilot.read_record(final_path))
        if result["signature"] != signature or result["attempts_checksum"] != pilot.digest(
            pilot.read_record(attempts_path)
        ):
            msg = "Initializer training endpoint or attempt identity changed."
            raise ValueError(msg)
        for point in result["curve"]:
            pilot.read_record(root / "deployments" / f"{point['deployment']}.json")
        return result
    prior = cast("dict[str, Any]", pilot.read_record(state_path)) if state_path.exists() else None
    attempts: dict[str, Any] = (
        cast("dict[str, Any]", pilot.read_record(attempts_path))
        if attempts_path.exists()
        else {"signature": signature, "attempts": []}
    )
    if attempts["signature"] != signature or (prior is not None and prior["signature"] != signature):
        msg = "Initializer optimizer checkpoint or attempt identity changed."
        raise ValueError(msg)
    initial = np.asarray(source["theta"], dtype=float)
    circuit = pilot.restore_circuit(source["circuit"])
    start_deployment = deployment(root, source["circuit"], source["theta"], initial=True)
    curve = (
        list(prior["curve"])
        if prior is not None
        else [
            {
                "iterations": 0,
                "joint_calls": 0,
                "deployment": start_deployment,
                "training_cpu_seconds": 0.0,
                "deployment_cpu_seconds": 0.0,
                "refinement_cpu_seconds": 0.0,
            }
        ]
    )
    prior_training = sum(a["training_cpu_seconds"] or 0.0 for a in attempts["attempts"])
    prior_deployment = sum(a["deployment_cpu_seconds"] or 0.0 for a in attempts["attempts"])
    attempt: dict[str, Any] = {
        "status": "running",
        "training_cpu_seconds": None,
        "deployment_cpu_seconds": None,
        "wall_seconds": None,
        "attempted_channel_calls": 0,
        "completed_channel_calls": 0,
    }
    attempts["attempts"].append(attempt)
    pilot.write_record(attempts_path, attempts)
    cpu, wall = time.process_time(), time.perf_counter()
    deployed_cpu = 0.0

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        loss, gradient = 0.0, np.zeros_like(theta)
        for profile in PROFILES:
            attempt["attempted_channel_calls"] += 1
            local_gradient, local_loss, _ = ensemble_experiment.component_gradient(
                circuit, theta, target, profile, fast=False
            )
            attempt["completed_channel_calls"] += 1
            loss += local_loss
            gradient += local_gradient
        return loss / 2, gradient / 2

    def save(state: dict[str, Any], *, force: bool = False) -> None:
        nonlocal deployed_cpu
        if not force and state["iterations"] % config.checkpoint_every:
            return
        started = time.process_time()
        try:
            identity = deployment(root, source["circuit"], state["theta"])
            elapsed, spent = time.process_time() - cpu, time.process_time() - started
            training_cpu, compile_cpu = (
                prior_training + elapsed - deployed_cpu - spent,
                prior_deployment + deployed_cpu + spent,
            )
            curve.append({
                "iterations": state["iterations"],
                "joint_calls": state["evaluations"],
                "deployment": identity,
                "training_cpu_seconds": training_cpu,
                "deployment_cpu_seconds": compile_cpu,
                "refinement_cpu_seconds": training_cpu + compile_cpu,
            })
            pilot.write_record(state_path, {"signature": signature, "optimizer": state, "curve": curve})
        finally:
            deployed_cpu += time.process_time() - started

    def optimize() -> dict[str, Any]:
        state = None if prior is None else prior["optimizer"]
        if method == "lbfgs":
            return optimizers.minimize(
                objective,
                initial,
                max_evaluations=config.joint_cap,
                max_iterations=config.joint_cap,
                history=config.history,
                state=state,
                callback=save,
            )
        return ensemble_experiment.minimize_adam(
            objective,
            initial,
            adam.AdamConfig(updates=config.joint_cap, learning_rate=config.adam_rate),
            state=state,
            callback=save,
        )

    try:
        state = optimize()
        save(state, force=True)
        attempt["status"] = "complete"
    except BaseException as error:
        attempt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt.update({
            "training_cpu_seconds": time.process_time() - cpu - deployed_cpu,
            "deployment_cpu_seconds": deployed_cpu,
            "wall_seconds": time.perf_counter() - wall,
        })
        pilot.write_record(attempts_path, attempts)
    timed = all(
        a["training_cpu_seconds"] is not None and a["deployment_cpu_seconds"] is not None for a in attempts["attempts"]
    )
    training_cpu = sum(a["training_cpu_seconds"] for a in attempts["attempts"]) if timed else None
    compile_cpu = sum(a["deployment_cpu_seconds"] for a in attempts["attempts"]) if timed else None
    result = {
        "status": "complete",
        "signature": signature,
        "optimizer": state,
        "curve": curve,
        "final_deployment": curve[-1]["deployment"],
        "method": method,
        "joint_calls": state["evaluations"],
        "individual_channel_gradient_calls": 2 * state["evaluations"],
        "physical_completed_channel_calls": sum(a["completed_channel_calls"] for a in attempts["attempts"]),
        "physical_attempted_channel_calls": sum(a["attempted_channel_calls"] for a in attempts["attempts"]),
        "training_cpu_seconds": training_cpu,
        "deployment_cpu_seconds": compile_cpu,
        "refinement_cpu_seconds": sum(
            a["training_cpu_seconds"] + a["deployment_cpu_seconds"] for a in attempts["attempts"]
        )
        if timed
        else None,
        "physical_counts_complete": all(a["status"] != "running" for a in attempts["attempts"]),
        "timing_complete": timed,
        "attempts_checksum": pilot.digest(attempts),
    }
    pilot.write_record(final_path, result)
    return result
