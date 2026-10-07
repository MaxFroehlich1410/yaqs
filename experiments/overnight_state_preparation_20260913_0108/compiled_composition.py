# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compose known-input compilation with unchanged symmetric structural training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    common_control_audit,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    composed_experiment as base,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    state_preparation_compilation as compiler,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

VERSION = "overnight.compiled_conditional_composition_kernel.v1"
PRIMARY = base.PRIMARY
TRAINED = base.TRAINED
METHODS = base.METHODS
BASELINE = "structural_mean2_weighted_adam"
CANDIDATE = "structural_mean2_lbfgs"


def read(path: Path) -> dict[str, Any]:
    """Return an authenticated heterogeneous research record."""
    return cast("dict[str, Any]", pilot.read_record(path))


def file_hash(path: Path) -> str:
    """Hash durable numerical bytes without changing a source artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts(root: Path) -> dict[str, str]:
    """Snapshot raw execution artifacts, excluding compact outer publication records."""
    excluded = {root / "RESULT.json", root / "EXECUTION.json"}
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.json")) if p not in excluded}


def authenticate(root: Path, dependencies: dict[str, str]) -> None:
    """Reject modification of a completed arm or execution's durable sources."""
    if any(file_hash(root / name) != expected for name, expected in dependencies.items()):
        msg = "Compiled composition artifact custody changed."
        raise ValueError(msg)


def state_compile(
    store: base.CompleteStore,
    point: dict[str, Any],
    role: str,
    *,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Apply the fixed state-preparation compiler and publish only after its charged audit."""
    source = {"circuit": pilot.circuit_record(store.restore(point["circuit_hash"])), "theta": point["theta"]}
    path = store.root / "state_compilations" / f"{pilot.digest(source)}.json"
    if cache_only and not path.exists():
        msg = "Missing cached state-preparation compiler receipt."
        raise ValueError(msg)
    result = base.tracked_call(
        path,
        [VERSION, compiler.VERSION, source],
        lambda: compiler.compile_endpoint(base.restore_physical(source["circuit"]), np.asarray(source["theta"])),
    )
    value = result["value"]
    if (
        value["source_checksum"] != pilot.digest(source)
        or value["version"] != compiler.VERSION
        or value["input_state"] != "all_zero_only"
        or not value["independent_audit"]["passed"]
    ):
        msg = "State-preparation compiler receipt does not authenticate its input or audit."
        raise ValueError(msg)
    return {
        "theta": value["theta"],
        "circuit_hash": store.circuit(base.restore_physical(value["circuit"])),
        "cpu_seconds": point["cpu_seconds"] + result["cpu_seconds"],
        "stage": role,
        "state_compiler_cpu_seconds": result["cpu_seconds"],
        "state_compiler_receipt": str(path.relative_to(store.root)),
        "state_compiler_checksum": pilot.digest(result),
        "removed_gates": value["input_gates"] - value["output_gates"],
    }


def deploy(
    store: base.CompleteStore,
    raw: dict[str, Any],
    *,
    cache_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose the generic compiler and state-preparation compiler as successive endpoints."""
    source = {"circuit": pilot.circuit_record(store.restore(raw["circuit_hash"])), "theta": raw["theta"]}
    if cache_only and not (store.root / "compilations" / f"{pilot.digest(source)}.json").exists():
        msg = "Missing cached generic compiler receipt."
        raise ValueError(msg)
    generic = {**base.compile_final(store, raw), "stage": "generic_final"}
    final = state_compile(store, generic, "state_preparation_final", cache_only=cache_only)
    return generic, final


def compact_arm(arm: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    """Return scores, charged clocks and authenticated arm references without copying curves."""
    keys = (
        "status",
        "failure",
        "final",
        "scores",
        "initializer_cpu_seconds",
        "initial_state_compiler_cpu_seconds",
        "training_cpu_seconds",
        "generic_final_compiler_cpu_seconds",
        "state_final_compiler_cpu_seconds",
        "refinement_cpu_seconds",
        "total_cpu_seconds",
    )
    return {
        **{key: arm[key] for key in keys if key in arm},
        "arm_file": str(path.relative_to(root)),
        "arm_checksum": pilot.digest(arm),
    }


def train_arm(
    store: base.CompleteStore,
    sampled: base.stochastic.Store,
    source: dict[str, Any],
    initial: dict[str, Any],
    original: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """Run the existing optimizer from the common compiled initializer, then both final compilers."""
    circuit = store.restore(initial["circuit_hash"])
    theta = np.asarray(initial["theta"])
    modified = {**source, "circuit": pilot.circuit_record(circuit), "theta": theta.tolist()}
    if method in base.exact.METHODS:
        trained = base.exact.train_method(store, circuit, theta, method)
    elif method == "fixed_noiseless_adam":
        trained = base.fixed_noiseless(store, modified)
    elif method in base.stochastic.METHODS:
        trained = base.stochastic.train_method(sampled, circuit, theta, method)
    else:
        msg = "Unknown compiled-composition training method."
        raise ValueError(msg)
    last = trained.get("state", trained)
    raw = {
        "theta": last["theta"],
        "circuit_hash": last["circuit_hash"],
        "cpu_seconds": initial["cpu_seconds"] + trained["cpu_seconds"],
        "stage": "raw_final",
    }
    generic, final = deploy(store, raw)
    shifted = [{**point, "cpu_seconds": initial["cpu_seconds"] + point["cpu_seconds"]} for point in trained["curve"]]
    receipt: dict[str, Any] = {
        "training": {key: value for key, value in trained.items() if key != "curve"},
        "shifted_curve_checksum": pilot.digest(shifted),
        "raw_final": raw,
    }
    receipt_path = store.root / "training_results" / f"{method}.json"
    if receipt_path.exists() and pilot.digest(read(receipt_path)) != pilot.digest(receipt):
        msg = "Completed training receipt differs from cached primitive results."
        raise ValueError(msg)
    if not receipt_path.exists():
        pilot.write_record(receipt_path, receipt)
    curve = [original, initial, *shifted, raw, generic, final]
    return {
        "status": "complete",
        "final": final,
        "curve": curve,
        "training_receipt": str(receipt_path.relative_to(store.root)),
        "training_checksum": pilot.digest(receipt),
        "scores": base.score(store, final),
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "initial_state_compiler_cpu_seconds": initial["state_compiler_cpu_seconds"],
        "training_cpu_seconds": trained["cpu_seconds"],
        "generic_final_compiler_cpu_seconds": generic["compiler_cpu_seconds"],
        "state_final_compiler_cpu_seconds": final["state_compiler_cpu_seconds"],
        "refinement_cpu_seconds": final["cpu_seconds"],
        "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
    }


def unchanged_arm(
    store: base.CompleteStore,
    source: dict[str, Any],
    initial: dict[str, Any],
    original: dict[str, Any],
) -> dict[str, Any]:
    """Grant the unchanged role the same initial and two final compiler opportunities."""
    generic, final = deploy(store, initial)
    return {
        "status": "complete",
        "final": final,
        "curve": [original, initial, generic, final],
        "scores": base.score(store, final),
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "initial_state_compiler_cpu_seconds": initial["state_compiler_cpu_seconds"],
        "training_cpu_seconds": 0.0,
        "generic_final_compiler_cpu_seconds": generic["compiler_cpu_seconds"],
        "state_final_compiler_cpu_seconds": final["state_compiler_cpu_seconds"],
        "refinement_cpu_seconds": final["cpu_seconds"],
        "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
    }


def common_panel(
    store: base.CompleteStore,
    arms: dict[str, Any],
    cutoff_methods: Sequence[str],
    initializer_cpu: float,
    *,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Score all seven roles at one fixed cutoff with chronological raw/compiler availability."""
    if any(arms[name]["status"] != "complete" for name in METHODS):
        return {"status": "unavailable_failed_arm", "cutoff_methods": list(cutoff_methods), "arms": {}}
    cutoff = min(arms[name]["refinement_cpu_seconds"] for name in cutoff_methods)
    selected = common_control_audit.select_points(arms, METHODS, cutoff)
    return {
        "status": "complete",
        "cutoff_methods": list(cutoff_methods),
        "methods": list(METHODS),
        "refinement_cpu_seconds": cutoff,
        "total_cpu_seconds": initializer_cpu + cutoff,
        "arms": {
            name: {"point": point, "scores": cached_score(store, point) if cache_only else base.score(store, point)}
            for name, point in selected.items()
        },
    }


def validate_arm(
    store: base.CompleteStore,
    arm: dict[str, Any],
    source: dict[str, Any],
    initial: dict[str, Any],
    original: dict[str, Any],
    method: str,
) -> None:
    """Bind public arm theta, costs, curve and scores to independent durable child receipts."""
    if arm["status"] == "failed":
        failure = read(store.root / "failures" / f"{method}.json")
        if arm["failure"] != failure["failure"] or arm["signature"] != failure["signature"]:
            msg = "Cached failure differs from its durable failure receipt."
            raise ValueError(msg)
        return
    training_cpu = 0.0
    if method == "unchanged":
        raw = initial
        curve = arm["curve"]
        valid_curve = len(curve) == 4 and curve[:2] == [original, initial]
    else:
        receipt = read(store.root / "training_results" / f"{method}.json")
        if (
            arm["training_checksum"] != pilot.digest(receipt)
            or arm["training_receipt"] != f"training_results/{method}.json"
        ):
            msg = "Arm training receipt changed."
            raise ValueError(msg)
        training = receipt["training"]
        last = training.get("state", training)
        training_cpu = training["cpu_seconds"]
        raw = {
            "theta": last["theta"],
            "circuit_hash": last["circuit_hash"],
            "cpu_seconds": initial["cpu_seconds"] + training_cpu,
            "stage": "raw_final",
        }
        curve = arm["curve"]
        valid_curve = (
            len(curve) >= 5
            and curve[:2] == [original, initial]
            and curve[-3] == raw
            and pilot.digest(curve[2:-3]) == receipt["shifted_curve_checksum"]
            and receipt["raw_final"] == raw
        )
    generic, final = deploy(store, raw, cache_only=True)
    expected = {
        "final": final,
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "initial_state_compiler_cpu_seconds": initial["state_compiler_cpu_seconds"],
        "training_cpu_seconds": training_cpu,
        "generic_final_compiler_cpu_seconds": generic["compiler_cpu_seconds"],
        "state_final_compiler_cpu_seconds": final["state_compiler_cpu_seconds"],
        "refinement_cpu_seconds": final["cpu_seconds"],
        "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
    }
    if not valid_curve or curve[-2:] != [generic, final] or any(arm.get(k) != v for k, v in expected.items()):
        msg = "Cached arm theta, curve or charged costs differ from child receipts."
        raise ValueError(msg)
    if arm["scores"] != cached_score(store, final):
        msg = "Cached arm score differs from its exact child receipt."
        raise ValueError(msg)


def cached_score(store: base.CompleteStore, point: dict[str, Any]) -> dict[str, Any]:
    """Validate an existing exact score receipt without permitting any fresh evaluation."""
    identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
    if not (store.root / "operations" / f"audit_{pilot.digest(identity)}.json").exists():
        msg = "Selected physical point lacks its cached all12 score receipt."
        raise ValueError(msg)
    return base.score(store, point)


def validate(
    source: dict[str, Any],
    target: NDArray,
    exact_config: base.exact.Config,
    stochastic_config: base.stochastic.Config,
    order: Sequence[str],
) -> None:
    """Validate shared physical input, method coverage and identical structural opportunities."""
    if len(order) != len(TRAINED) or set(order) != set(TRAINED):
        msg = "Every distinct trained role must be present exactly once."
        raise ValueError(msg)
    if (
        exact_config.max_rounds,
        exact_config.shortlist,
        exact_config.repair_evaluations,
        exact_config.warmup_evaluations,
    ) != (
        stochastic_config.rounds,
        stochastic_config.shortlist,
        stochastic_config.repair,
        stochastic_config.warmup,
    ):
        msg = "Every structural role needs the same warmup, rounds, shortlist and repair opportunity."
        raise ValueError(msg)
    if exact_config.protected_initial_gates or stochastic_config.protected:
        msg = "Compiled composition requires the same empty protected set."
        raise ValueError(msg)
    value = source["initializer_cpu_seconds"]
    circuit = base.restore_physical(source["circuit"])
    compiler.validate(circuit, np.asarray(source["theta"]))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or source.get("timing_complete", True) is not True
        or target.shape != (2**circuit.num_qubits,)
        or not np.all(np.isfinite(target))
        or abs(np.vdot(target, target) - 1) > 1e-10
    ):
        msg = "Shared initialization needs finite known costs and a normalized physical target."
        raise ValueError(msg)


def execute_arms(
    root: Path,
    source: dict[str, Any],
    target: NDArray,
    exact_config: base.exact.Config,
    stochastic_config: base.stochastic.Config,
    seed: int,
    order: Sequence[str],
    signature: str,
) -> dict[str, Any]:
    """Execute durable arm units and publish compact common-cutoff and full-cap records."""
    store = base.CompleteStore(root, target, exact_config)
    sampled = base.stochastic.Store(root, target, stochastic_config, seed)
    original = {
        "circuit_hash": store.circuit(base.restore_physical(source["circuit"])),
        "theta": source["theta"],
        "cpu_seconds": 0.0,
        "stage": "generic_initializer",
    }
    initial = state_compile(store, original, "state_preparation_initializer")
    if len(store.restore(initial["circuit_hash"]).gates) <= max(exact_config.max_rounds, stochastic_config.rounds):
        msg = "Compiled initialization must retain at least one gate through all allowed deletions."
        raise ValueError(msg)
    arms: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for method in (*order, "unchanged"):
        arm: dict[str, Any]
        path = root / "arms" / f"{method}.json"
        identity = pilot.digest([signature, method, initial])
        if path.exists():
            arm = read(path)
            if arm["signature"] != identity:
                msg = "Compiled composition arm identity changed."
                raise ValueError(msg)
            authenticate(root, arm["dependencies"])
        else:
            try:
                arm = (
                    unchanged_arm(store, source, initial, original)
                    if method == "unchanged"
                    else train_arm(store, sampled, source, initial, original, method)
                )
            except (ArithmeticError, RuntimeError) as error:
                arm = {"status": "failed", "failure": f"{type(error).__name__}: {error}"}
                failure: dict[str, Any] = {**arm, "signature": identity}
                pilot.write_record(root / "failures" / f"{method}.json", failure)
            arm["signature"] = identity
            arm["dependencies"] = {
                name: value
                for name, value in artifacts(root).items()
                if name != "CASE_ATTEMPTS.json" and not name.startswith("arms/")
            }
            pilot.write_record(path, arm)
            arm = read(path)
        validate_arm(store, arm, source, initial, original, method)
        arms[method], compact[method] = arm, compact_arm(arm, path, root)
    return {
        "status": "complete" if all(a["status"] == "complete" for a in arms.values()) else "partial_failure",
        "source_checksum": pilot.digest(source),
        "initial": initial,
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "physical_initial_compiler_receipt": initial["state_compiler_receipt"],
        "initial_compiler_scope": "executed once physically, charged to every declared procedure",
        "arms": compact,
        "common_primary": common_panel(store, arms, PRIMARY, source["initializer_cpu_seconds"]),
        "common_all_trained": common_panel(store, arms, TRAINED, source["initializer_cpu_seconds"]),
    }


def validate_execution(
    root: Path,
    result: dict[str, Any],
    source: dict[str, Any],
    target: NDArray,
    config: base.exact.Config,
    signature: str,
) -> None:
    """Reconstruct compact endpoint/cost/selection views from immutable child receipts."""
    store = base.CompleteStore(root, target, config)
    original = {
        "circuit_hash": pilot.digest(source["circuit"]),
        "theta": source["theta"],
        "cpu_seconds": 0.0,
        "stage": "generic_initializer",
    }
    initial = state_compile(store, original, "state_preparation_initializer", cache_only=True)
    arms: dict[str, Any] = {}
    for method in METHODS:
        path = root / "arms" / f"{method}.json"
        arm = read(path)
        if arm["signature"] != pilot.digest([signature, method, initial]):
            msg = "Cached arm identity differs from the compiled execution."
            raise ValueError(msg)
        authenticate(root, arm["dependencies"])
        validate_arm(store, arm, source, initial, original, method)
        if result["arms"].get(method) != compact_arm(arm, path, root):
            msg = "Cached compact arm differs from its complete arm receipt."
            raise ValueError(msg)
        arms[method] = arm
    if set(result["arms"]) != set(METHODS):
        msg = "The compiled execution must preserve all seven roles."
        raise ValueError(msg)
    expected = {
        "status": "complete" if all(a["status"] == "complete" for a in arms.values()) else "partial_failure",
        "source_checksum": pilot.digest(source),
        "initial": initial,
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "physical_initial_compiler_receipt": initial["state_compiler_receipt"],
        "common_primary": common_panel(store, arms, PRIMARY, source["initializer_cpu_seconds"], cache_only=True),
        "common_all_trained": common_panel(store, arms, TRAINED, source["initializer_cpu_seconds"], cache_only=True),
    }
    attempts = read(root / "CASE_ATTEMPTS.json")
    invocation = result["physical_invocation"]
    if (
        attempts["signature"] != signature
        or invocation["attempts_checksum"] != pilot.digest(attempts)
        or any(a["status"] == "running" for a in attempts["attempts"])
        or any(invocation[key] != sum(a[key] for a in attempts["attempts"]) for key in ("cpu_seconds", "wall_seconds"))
        or any(result.get(key) != value for key, value in expected.items())
    ):
        msg = "Cached selected points, scores or clocks differ from their child receipts."
        raise ValueError(msg)


def execute(
    root: Path,
    source: dict[str, Any],
    target: NDArray[np.complex128],
    exact_config: base.exact.Config,
    stochastic_config: base.stochastic.Config,
    seed: int,
    *,
    order: Sequence[str] = TRAINED,
) -> dict[str, Any]:
    """Run the complete new pipeline from supplied authenticated arrays, with exact durable resume.

    Full curves reside in one arm record each. The compact return value exposes
    references, scores and point-selection clocks for a development or future
    separately authorized parent driver. It never loads a benchmark target.
    """
    validate(source, target, exact_config, stochastic_config, order)
    signature = pilot.digest({
        "version": VERSION,
        "source": source,
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "exact_config": asdict(exact_config),
        "stochastic_config": asdict(stochastic_config),
        "seed": seed,
        "order": list(order),
    })
    path = root / "EXECUTION.json"
    if path.exists():
        result = read(path)
        if result["signature"] != signature or artifacts(root) != result["dependencies"]:
            msg = "Compiled composition execution or complete artifact index changed."
            raise ValueError(msg)
        validate_execution(root, result, source, target, exact_config, signature)
        return result
    result = base.measured_execute(
        root,
        signature,
        lambda: execute_arms(root, source, target, exact_config, stochastic_config, seed, order, signature),
    )
    result.update(signature=signature, version=VERSION, dependencies=artifacts(root))
    pilot.write_record(path, result)
    return read(path)
