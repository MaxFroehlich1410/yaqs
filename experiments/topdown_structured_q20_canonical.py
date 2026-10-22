# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Isolated, explicitly sealed canonical-epsilon continuation of a q20 pilot."""

from __future__ import annotations

import argparse
import json
import operator
import os
import socket
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments import topdown_backend_support as storage
from experiments import topdown_structured_q20 as pilot
from mqt.yaqs.optimization import KrotovNoiseMap
from mqt.yaqs.optimization.canonical_mps_backend import CanonicalMPSBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.optimization import ParameterizedCircuit
    from mqt.yaqs.optimization.trajectories import NoiseMaps

VERSION = "topdown.canonical_epsilon_continuation.v1"
TRUNCATED_VERSION = "topdown.canonical_relative_continuation.v2"
INPUT_CUTOFF = float(np.finfo(float).eps)
MODULE = "experiments.topdown_structured_q20_canonical"
STAGES = tuple(stage for stage in pilot.STAGES if stage != "prepare")


class CanonicalProgressBackend(CanonicalMPSBackend):
    """Write the original progress schema around explicitly approximate propagation."""

    def __init__(
        self, circuit: ParameterizedCircuit, target: MPS, path: Path, *, relative_cutoff: float = INPUT_CUTOFF
    ) -> None:
        """Bind a private directory without changing the fixed-map numerical policy."""
        super().__init__(circuit, target, relative_cutoff=relative_cutoff)
        self.path = path
        self.calls = 0

    def metrics(
        self, theta: NDArray, maps: NoiseMaps | None = None, *, weights: NDArray | None = None
    ) -> tuple[float, float]:
        """Persist objective/resource observations after each complete replay.

        Returns:
            Weighted loss and fidelity.
        """
        result = super().metrics(theta, maps, weights=weights)
        self.calls += 1
        storage.write_json(
            self.path / "progress.json",
            {
                "calls": self.calls,
                "fidelity": result[1],
                **pilot.resources(),
                **representation(self.relative_cutoff),
                "diagnostics": self.diagnostics.record(),
            },
        )
        return result


def replay_bank(branch: Path, backend: CanonicalMPSBackend, count: int) -> tuple:
    """Restore every saved physical path, checking duplicate compression weights.

    Returns:
        Compressed replay maps and exactly the recorded weights.

    Raises:
        ValueError: If path counts, gates or stored probabilities disagree.
    """
    record = json.loads((branch / "noise-bank.json").read_text())
    raw = [
        tuple(
            KrotovNoiseMap(
                operators=tuple(
                    (np.asarray(event["real"]) + 1j * np.asarray(event["imag"]), tuple(event["sites"]))
                    for event in events
                )
            )
            for events in row
        )
        for row in record["raw_events"]
    ]
    maps, weights = backend.compress_no_jump_maps(raw)
    if (
        len(raw) != count
        or len(maps) != record["retained_paths"]
        or any(len(row) != len(backend.circuit.gates) for row in raw)
        or not np.array_equal(weights, record["weights"])
    ):
        msg = "Frozen replay bank differs from the original protocol."
        raise ValueError(msg)
    return maps, weights


def representation(relative_cutoff: float) -> dict:
    """Return separately recorded propagation and input precision policies.

    Returns:
        Explicit representation metadata for progress and checkpoint records.
    """
    return {
        "representation": VERSION if relative_cutoff == INPUT_CUTOFF else TRUNCATED_VERSION,
        "relative_cutoff": relative_cutoff,
        "input_relative_cutoff": INPUT_CUTOFF,
    }


def resume_b(folder: Path, settings: pilot.Settings, *, relative_cutoff: float = INPUT_CUTOFF) -> None:
    """Continue the remaining B updates, preserving historical scores and CPU costs.

    Raises:
        ValueError: If saved history is incomplete or a completed selection exists.
    """
    branch = folder / "B"
    if (branch / "selection/complete.json").exists():
        msg = "B selection already exists; refusing duplicate continuation."
        raise ValueError(msg)
    target, _ = pilot.get_target(folder)
    pruning = storage.load_checkpoint(branch / "pruning")
    if pruning is None:
        msg = "Sealed original B pruning is missing."
        raise ValueError(msg)
    circuit = storage.restore_circuit(pruning[0]["circuit"])
    history = []
    parameters = None
    for index, path in enumerate(sorted((branch / "iterates").glob("step_*"))):
        saved = storage.load_checkpoint(path)
        if saved is None or path.name != f"step_{index:05d}" or saved[0]["step"] != index:
            msg = "Original B history is unsealed or non-contiguous."
            raise ValueError(msg)
        history.append({key: saved[0][key] for key in ("step", "fidelity", "kernel_cpu_seconds")})
        parameters = saved[1]["theta"]
    if not history or parameters is None or history[-1]["step"] > settings.noisy_online + settings.b_batch:
        msg = "Original B history does not fit the fixed update budget."
        raise ValueError(msg)
    backend = CanonicalProgressBackend(circuit, target, branch, relative_cutoff=relative_cutoff)
    maps, weights = replay_bank(branch, backend, settings.training_paths)
    step = history[-1]["step"]
    resume_step = step
    kernel_cpu = history[-1]["kernel_cpu_seconds"]
    historical_cpu = kernel_cpu
    initial_cpu = history[min(step, settings.noisy_online)]["kernel_cpu_seconds"]
    extra_cpu = kernel_cpu - initial_cpu
    best = max(history, key=operator.itemgetter("fidelity"))
    best_step, best_fidelity = best["step"], best["fidelity"]
    storage.write_json(
        branch / "continuation-started.json",
        {
            "resume_step": resume_step,
            "historical_kernel_cpu_seconds": historical_cpu,
            "remaining_updates": settings.noisy_online + settings.b_batch - step,
            "raw_training_paths": settings.training_paths,
            "retained_paths": len(maps),
            **representation(relative_cutoff),
            "pid": os.getpid(),
            **pilot.resources(),
        },
    )
    while step < settings.noisy_online + settings.b_batch:
        started = time.process_time()
        if step < settings.noisy_online:
            parameters, _ = backend.online_update(parameters, settings.rate, maps, estimator="cross", weights=weights)
        else:
            signal, _, _ = backend.gradient(parameters, maps, estimator="cross", weights=weights)
            parameters -= settings.rate * signal
        fidelity = float(backend.metrics(parameters, maps, weights=weights)[1])
        elapsed = time.process_time() - started
        kernel_cpu += elapsed
        if step < settings.noisy_online:
            initial_cpu += elapsed
        else:
            extra_cpu += elapsed
        step += 1
        entry = {
            "step": step,
            "fidelity": fidelity,
            "kernel_cpu_seconds": kernel_cpu,
            **representation(relative_cutoff),
        }
        storage.save_checkpoint(branch / "iterates" / f"step_{step:05d}", entry, {"theta": parameters})
        history.append(entry)
        if fidelity > best_fidelity:
            best_step, best_fidelity = step, fidelity
        storage.write_json(
            branch / "training-progress.json",
            {
                "step": step,
                "fidelity": fidelity,
                "best_step": best_step,
                "best_fidelity": best_fidelity,
                "initial_online_cpu_seconds": initial_cpu,
                "extra_online_cpu_seconds": extra_cpu,
                "historical_kernel_cpu_seconds": historical_cpu,
                "resume_step": resume_step,
                **representation(relative_cutoff),
                **pilot.resources(),
            },
        )
    selected = storage.load_checkpoint(branch / "iterates" / f"step_{best_step:05d}")
    assert selected is not None
    storage.save_checkpoint(
        branch / "selection",
        {
            "policy": "B",
            "selected_step": best_step,
            "training_fidelity": best_fidelity,
            "kernel_cpu_seconds": kernel_cpu,
            "initial_online_cpu_seconds": initial_cpu,
            "extra_online_cpu_seconds": extra_cpu,
            "historical_kernel_cpu_seconds": historical_cpu,
            "resume_step": resume_step,
            "stop": "fixed_budget",
            "history": history,
            "training_bank_sha256": storage.digest(branch / "noise-bank.json"),
            "pruning_seal": storage.digest(branch / "pruning/complete.json"),
            **representation(relative_cutoff),
        },
        {"theta": selected[1]["theta"]},
    )


def verify(root: Path) -> dict:
    """Require both the frozen source closure and an explicit continuation tag.

    Returns:
        Verified campaign plan.

    Raises:
        ValueError: If this is an original root or has an unexpected compression rule.
    """
    plan = pilot.verify(root)
    rule = plan.get("canonical_continuation", {})
    cutoff = rule.get("relative_cutoff")
    legacy = rule.get("version") == VERSION and np.array_equal(cutoff, INPUT_CUTOFF)
    explicit = (
        rule.get("version") == TRUNCATED_VERSION
        and isinstance(cutoff, (float, int))
        and not isinstance(cutoff, bool)
        and np.isfinite(cutoff)
        and 0 < cutoff < 1
        and rule.get("input_relative_cutoff") == INPUT_CUTOFF
    )
    if not (legacy or explicit):
        msg = "This root is not sealed for canonical continuation with preserved input precision."
        raise ValueError(msg)
    inputs = rule.get("inputs", {})
    if not inputs:
        msg = "Sealed continuation inputs are missing."
        raise ValueError(msg)
    for name, checksum in inputs.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or storage.digest(root / path) != checksum:
            msg = f"Continuation input changed: {name}."
            raise ValueError(msg)
    return plan


def execute_stage(
    folder: Path, target: str, stage: str, settings: pilot.Settings, *, relative_cutoff: float = INPUT_CUTOFF
) -> None:
    """Dispatch unchanged assessment or explicitly compressed training.

    Raises:
        ValueError: If preparation or an unknown stage is requested.
    """
    with threadpool_limits(limits=1):
        if stage == "B":
            resume_b(folder, settings, relative_cutoff=relative_cutoff)
        elif stage == "G":
            pilot.train_branch(
                folder,
                "G",
                settings,
                backend_factory=partial(CanonicalProgressBackend, relative_cutoff=relative_cutoff),
            )
        elif stage in STAGES:
            pilot.execute_stage(folder, target, stage, settings)
        else:
            msg = "Preparation and unknown stages are not continuation work."
            raise ValueError(msg)


def leaf(root: Path, target: str, stage: str) -> None:
    """Run one explicitly selected stage with durable success/failure evidence."""
    plan = verify(root)
    if target not in plan["canonical_continuation"]["targets"]:
        msg = "Target is outside the sealed continuation assignment."
        raise ValueError(msg)
    settings = pilot.Settings(**plan["settings"])
    folder = root / "jobs" / target
    started = time.perf_counter()
    cutoff = plan["canonical_continuation"]["relative_cutoff"]
    receipt = {
        "target_id": target,
        "stage": stage,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **representation(cutoff),
    }
    try:
        execute_stage(folder, target, stage, settings, relative_cutoff=cutoff)
        receipt["status"] = "complete"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        storage.write_json(
            folder / "receipts" / f"{stage}.json",
            {**receipt, **pilot.resources(), "stage_wall_seconds": time.perf_counter() - started},
        )


def supervise(root: Path, target: str, host: str) -> None:
    """Own one replacement seat, preserving the original stage guards and dependencies."""
    plan = verify(root)
    rule = plan["canonical_continuation"]
    if socket.gethostname().split(".")[0] != host or rule["host"] != host or target not in rule["targets"]:
        msg = "Continuation host or target differs from the sealed assignment."
        raise ValueError(msg)
    claim = root / "claims" / target
    claim.mkdir(parents=True, exist_ok=False)
    storage.write_json(
        claim / "owner.json", {"pid": os.getpid(), "host": host, "created": psutil.Process().create_time()}
    )
    failed: set[str] = set()
    for stage in STAGES:
        dependency = stage[0] if stage not in {"B", "G"} else None
        if dependency in failed:
            failed.add(stage)
            storage.write_json(
                root / "jobs" / target / "supervision" / f"{stage}.json",
                {"status": "skipped", "dependency": dependency},
            )
            continue
        while psutil.virtual_memory().available < rule["admission_available_gib"] * pilot.GIB:
            storage.write_json(root / f"status-{host}.json", {"status": "waiting_for_memory", "stage": stage})
            time.sleep(30)
        code, _ = pilot.run_stage(root, target, stage, host, plan, leaf_module=MODULE)
        if code:
            failed.add(stage)
    storage.write_json(
        root / "jobs" / target / "finished.json",
        {"status": "partial" if failed else "complete", "failed_or_skipped_stages": sorted(failed)},
    )


def main() -> None:
    """Dispatch only an explicitly sealed continuation root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "leaf", "supervise"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--host")
    args = parser.parse_args()
    if args.action == "check":
        verify(args.root)
    elif args.action == "leaf":
        if args.stage is None:
            parser.error("leaf requires --stage")
        leaf(args.root, args.target_id, args.stage)
    else:
        if args.host is None:
            parser.error("supervise requires --host")
        supervise(args.root, args.target_id, args.host)


if __name__ == "__main__":
    main()
