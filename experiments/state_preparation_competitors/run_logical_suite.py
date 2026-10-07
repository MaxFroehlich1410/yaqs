# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Extend the completed logical-noise pilot to all nine six-qubit fixtures.

Completed pilot selections and assessments are authenticated and referenced in
place. Only missing method-target groups run; archived incumbent endpoints need
assessment but no training. Target workers own disjoint bundles and preserve the
pilot's fixed optimizer budgets, seed, elementary-gate noise and selection rule.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation import execution
from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.state_preparation_competitors import controls, fontana, nacl, reference
from experiments.state_preparation_competitors import run as storage
from experiments.state_preparation_competitors import run_logical as pilot
from experiments.state_preparation_competitors.run_noisy import PINNED

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray
    from qiskit import QuantumCircuit

    from experiments.state_preparation_competitors.common import Candidate

SCHEMA = "yaqs.logical_noise_suite.v1"
METHODS = pilot.METHODS
PILOT_ROOT = storage.ROOT / "output/logical-noise-three-target-ibm-pilot-20260920"
GRID_SIZES = dict(zip(METHODS, (1, 3, 2, 2, 2, 4), strict=True))


def make_parser() -> argparse.ArgumentParser:
    """Return the fixed-budget all-target suite interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=reference.DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--validation-paths", type=int, default=8192)
    parser.add_argument(
        "--workers", type=int, default=3, help="One to three isolated target processes, one BLAS thread each."
    )
    return parser


def validate_config(args: argparse.Namespace) -> None:
    """Reject changed assessment budgets and unsupported or duplicate jobs.

    Raises:
        ValueError: If targets, methods, workers or the fixed path count are invalid.
    """
    for name, choices in (("targets", TARGET_IDS), ("methods", METHODS)):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values) or any(value not in choices for value in values):
            msg = f"{name} must contain unique supported identifiers."
            raise ValueError(msg)
    if type(args.validation_paths) is not int or args.validation_paths != 8192:
        msg = "This unchanged-budget suite requires exactly 8192 validation trajectories."
        raise ValueError(msg)
    if type(args.workers) is not int or not 1 <= args.workers <= 3:
        msg = "workers must be an integer from one through three."
        raise ValueError(msg)
    if args.output.resolve() in {args.pilot_root.resolve(), args.reference_root.resolve()}:
        msg = "Suite output must be separate from immutable pilot and incumbent archives."
        raise ValueError(msg)


def check_sources(manifest: dict[str, Any], *, allowed: set[str] | None = None, python_only: bool = False) -> None:
    """Authenticate stored source hashes with narrowly recorded historical exceptions.

    Args:
        manifest: Original manifest containing a nonempty source fingerprint map.
        allowed: Previously documented unrelated Python edits in a nested source.
        python_only: Nested native campaigns permit later documentation edits.

    Raises:
        ValueError: If fingerprints are absent, unsafe, changed or unrecorded.
    """
    exceptions = set() if allowed is None else allowed
    if not exceptions <= pilot.REUSE_UNRELATED_CHANGES or not manifest.get("sources"):
        msg = "Unsupported source exceptions or missing source fingerprints."
        raise ValueError(msg)
    changed = set()
    for name, expected in manifest["sources"].items():
        if python_only and not name.endswith(".py") and name != str(DEFAULT_TARGET_PATH.relative_to(storage.ROOT)):
            continue
        source = reference.archive_member(storage.ROOT, name)
        if not source.is_file() or storage.digest(source) != expected:
            if name not in exceptions:
                msg = f"Authenticated source changed: {name}"
                raise ValueError(msg)
            changed.add(name)
    if changed != exceptions:
        msg = "Recorded unrelated source changes no longer match their receipt."
        raise ValueError(msg)


def authenticate_source_candidate(
    metadata: dict[str, Any], circuit: QuantumCircuit, *, seen: frozenset[tuple[str, str, int]] = frozenset()
) -> None:
    """Recursively verify immutable construction references without copying histories.

    Args:
        metadata: One compact candidate, possibly referring to an earlier campaign.
        circuit: Its preserved operation schedule.
        seen: Recursion ancestry used to reject circular references.

    Raises:
        ValueError: If a source manifest, sealed bundle, index or identity differs.
    """
    pointer = metadata.get("source_candidate")
    if pointer is None:
        return
    root = Path(pointer["output"]).resolve()
    folder = reference.archive_member(root, pointer["construction_path"])
    index = pointer["candidate_index"]
    if type(index) is not int or index < 0:
        msg = "Source candidate index must be a nonnegative integer."
        raise ValueError(msg)
    identity = (str(root), str(folder), index)
    if identity in seen:
        msg = "Circular source candidate reference."
        raise ValueError(msg)
    if (
        storage.digest(root / "manifest.json") != pointer["manifest_sha256"]
        or storage.digest(folder / "seal.json") != pointer["construction_seal_sha256"]
    ):
        msg = "Source candidate manifest or construction seal changed."
        raise ValueError(msg)
    manifest = json.loads((root / "manifest.json").read_text())
    check_sources(manifest, allowed=set(pointer.get("unrelated_changed_sources", [])), python_only=True)
    record, circuits = storage.load_bundle(folder)
    if index >= len(circuits) or len(circuits) != len(record["candidates"]):
        msg = "Source candidate index or bundle size differs."
        raise ValueError(msg)
    source = record["candidates"][index]
    if circuits[index] != circuit or any(source[key] != metadata[key] for key in ("method", "candidate_id", "status")):
        msg = "Source candidate identity or original circuit differs."
        raise ValueError(msg)
    authenticate_source_candidate(source, circuit, seen=seen | {identity})


def _pilot_args(manifest: dict[str, Any]) -> argparse.Namespace:
    """Restore only the fixed original configuration needed for authentication.

    Returns:
        Original pilot arguments with source-directory strings converted to paths.
    """
    values = dict(manifest["config"])
    for name in ("gaussian_source", "other_source", "reference_root"):
        values[name] = Path(values[name])
    return argparse.Namespace(**values)


def _authenticate_group(
    root: Path, group: dict[str, Any], args: argparse.Namespace, target: NDArray[np.complex128]
) -> dict[str, Any]:
    """Verify one completed pilot group and return a reference-only suite view.

    Returns:
        Original results plus their effective source directory and reuse receipt.

    Raises:
        ValueError: If locks, circuits, budgets, costs or fixture identities disagree.
    """
    identity = group["context"]["identity"]
    target_id, method = identity["target_id"], identity["method"]
    relative = Path(target_id) / method / "300/IBM"
    construction_path = str(Path("construction") / relative)
    construction_folder = reference.archive_member(root, construction_path)
    construction, circuits = storage.load_bundle(construction_folder)
    selection_folder = root / "evaluation" / relative / "selection_locked"
    final_folder = root / "evaluation" / relative / "final"
    locked, selected_circuits = storage.load_bundle(selection_folder)
    final, final_circuits = storage.load_bundle(final_folder)
    context = {
        "identity": {"target_id": target_id, "method": method, "seed": 300, "hardware": "IBM"},
        "noise_basis": "logical",
        "validation_paths": 8192,
        "construction_receipt_sha256": storage.digest(construction_folder / "seal.json"),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    if (
        group["construction_cache"] != construction_path
        or locked["context"] != context
        or final["selection_lock_sha256"] != storage.digest(selection_folder / "seal.json")
        or any(final.get(key) != value for key, value in locked.items())
        or any(group.get(key) != value for key, value in final.items())
        or final["status"] != "complete"
        or circuits != selected_circuits
        or circuits != final_circuits
        or len(circuits) != GRID_SIZES[method]
        or len(construction["candidates"]) != len(circuits)
        or len(final["records"]) != len(circuits)
        or len(final["final_scores"]) != len(circuits)
    ):
        msg = "Pilot selection, final results, construction grid or fixture identity changed."
        raise ValueError(msg)
    metadata = construction["candidates"]
    for index, (candidate, circuit, row, score) in enumerate(
        zip(metadata, circuits, final["records"], final["final_scores"], strict=True)
    ):
        if (
            candidate["method"] != method
            or row["method"]["candidate_index"] != index
            or row["method"]["candidate_id"] != candidate["candidate_id"]
            or score["candidate_id"] != candidate["candidate_id"]
            or row["method"]["construction_reference"]
            != {"path": construction_path, "seal_sha256": context["construction_receipt_sha256"]}
        ):
            msg = "Pilot candidate references no longer match construction."
            raise ValueError(msg)
        authenticate_source_candidate(candidate, circuit)
    if method in pilot.REUSED:
        authenticated = pilot.reuse_candidates(args, target_id, method, target)
        if len(authenticated) != len(circuits) or any(
            item.circuit != circuit or item.metadata != saved
            for item, circuit, saved in zip(authenticated, circuits, metadata, strict=True)
        ):
            msg = "Reused original training no longer matches the completed pilot."
            raise ValueError(msg)
    else:
        if [item["depth"] for item in metadata] != [2, 4] or any(
            item.get("noise_basis") != "logical" for item in metadata
        ):
            msg = "Pilot logical training depths or noise placement differ."
            raise ValueError(msg)
        for item in metadata:
            if method == "robuststate":
                trace = item["training"]
                matching = (
                    trace["noiseless_steps"] == 500
                    and trace["noisy_steps"] == 50
                    and trace["feedback"] == "exact_density"
                )
            else:
                matching = (
                    item["construction"]["requested_steps"] == 200 and item["noisy_training"]["requested_steps"] == 200
                )
            if not matching:
                msg = "Pilot optimizer budgets differ from the fixed suite."
                raise ValueError(msg)
    expected_cost = metadata[0].get("historical_build_train_cpu_seconds", construction["construction_cpu_seconds"])
    if not math.isclose(group["build_train_cpu_seconds"], expected_cost, rel_tol=0, abs_tol=1e-9):
        msg = "Pilot build/train cost differs from its authenticated full-grid receipt."
        raise ValueError(msg)
    return {
        **copy.deepcopy(group),
        "source_output": str(root.resolve()),
        "reused_evaluation": True,
        "training_reused": True,
        "new_build_train_cpu_seconds": 0.0,
        "reused_result": {
            "manifest_sha256": storage.digest(root / "manifest.json"),
            "summary_sha256": storage.digest(root / "summary.json"),
            "final_seal_sha256": storage.digest(final_folder / "seal.json"),
            "selection_seal_sha256": storage.digest(selection_folder / "seal.json"),
        },
    }


def authenticate_pilot(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Any]]:
    """Authenticate all 17 prior groups without reevaluation or reselection.

    Returns:
        Immutable-source group views keyed by target and method.

    Raises:
        ValueError: If campaign configuration, sources, coverage or receipts differ.
    """
    root = args.pilot_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    config = manifest["config"]
    required = {
        str(Path(pilot.__file__).relative_to(storage.ROOT)),
        "experiments/state_preparation_competitors/logical_evaluation.py",
        str(DEFAULT_TARGET_PATH.relative_to(storage.ROOT)),
    }
    if (
        manifest["schema"] != pilot.SCHEMA
        or summary["schema"] != pilot.SCHEMA
        or summary["manifest_sha256"] != storage.digest(root / "manifest.json")
        or manifest["hardware"] != "IBM"
        or manifest["optimizer_seed"] != 300
        or manifest["packages"] != PINNED
        or config["targets"] != list(pilot.TARGETS)
        or config["methods"] != list(METHODS)
        or config["validation_paths"] != 8192
        or Path(config["reference_root"]).resolve() != args.reference_root.resolve()
        or not required <= set(manifest["sources"])
    ):
        msg = "Original pilot manifest or fixed study configuration differs."
        raise ValueError(msg)
    check_sources(manifest)
    old_args = _pilot_args(manifest)
    current = pilot.build_manifest(old_args)
    if any(manifest[key] != current[key] for key in ("noise", "compiler_role", "budgets")):
        msg = "Original pilot channel, assessment role or budgets differ."
        raise ValueError(msg)
    expected = {
        (target, method)
        for target in pilot.TARGETS
        for method in METHODS
        if not (target == "haar_random_1" and method == "fontana")
    }
    identities = [
        (item["context"]["identity"]["target_id"], item["context"]["identity"]["method"]) for item in summary["groups"]
    ]
    excluded = [(item["target"], item["method"]) for item in summary["exclusions"]]
    if set(identities) != expected or len(identities) != len(expected) or excluded != [("haar_random_1", "fontana")]:
        msg = "Completed pilot coverage or exclusions differ from its 17 declared groups."
        raise ValueError(msg)
    collection = load_target_collection()
    return {
        identity: _authenticate_group(root, group, old_args, collection.load(6, identity[0]).state_vector_copy())
        for identity, group in zip(identities, summary["groups"], strict=True)
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint the new suite, original study and actual numerical dependencies.

    Returns:
        New immutable configuration and source map, separate from the old pilot.

    Raises:
        ValueError: If a selected method's installed numerical dependency differs.
    """
    required = {**PINNED, **({"PyNomadBBO": nacl.NOMAD_VERSION} if "nacl" in args.methods else {})}
    versions = {name: importlib.metadata.version(name) for name in required}
    if versions != required:
        msg = "Logical suite numerical dependencies differ from their pins."
        raise ValueError(msg)
    paths = set((storage.ROOT / "experiments").rglob("*.py"))
    paths |= set((storage.ROOT / "benchmarks/state_preparation").glob("*.py"))
    paths |= set((storage.ROOT / "src/mqt/yaqs").rglob("*.py"))
    paths.add(DEFAULT_TARGET_PATH)
    paths.add(Path(__file__).with_name("LOGICAL_SUITE.md"))
    return {
        "schema": SCHEMA,
        "config": {
            name: str(value.resolve()) if isinstance(value, Path) else value
            for name, value in vars(args).items()
            if name != "output"
        },
        "hardware": "IBM",
        "optimizer_seed": 300,
        "numeric_threads_per_worker": 1,
        "packages": versions,
        "pilot_manifest_sha256": storage.digest(args.pilot_root / "manifest.json"),
        "pilot_summary_sha256": storage.digest(args.pilot_root / "summary.json"),
        "sources": {str(path.relative_to(storage.ROOT)): storage.digest(path) for path in sorted(paths)},
        "budget_policy": "unchanged run_logical pilot grids and iteration budgets; no equal-compute claim",
        "noise": "existing IBM Pauli channel after every original elementary gate; compiled resources only",
        "reuse": "existing 17 completed groups retain their original validation selections and exact scores",
    }


def construct(args: argparse.Namespace, target_id: str, method: str, target: NDArray[np.complex128]) -> list[Candidate]:
    """Invoke unchanged pilot factories only for missing method-target groups.

    Returns:
        Full candidate grid, or one authenticated incumbent needing no training.

    Raises:
        ValueError: If a method is unsupported.
    """
    if method == "current_b_lbfgs_frozen":
        item = reference.frozen_reference(args.reference_root, target_id, 300, "IBM", target)
        cost = item.metadata["archived_search_cost"]
        item.metadata.update({
            "noise_basis": "logical",
            "training_reused": True,
            "historical_build_train_cpu_seconds": sum(
                cost[key] for key in ("construction_cpu_seconds", "pruning_cpu_seconds", "fine_tuning_cpu_seconds")
            ),
            "historical_selection_cpu_seconds": cost["validation_cpu_seconds"],
        })
        return [item]
    if method == "nacl":
        return nacl.nacl_candidates(
            target,
            hardware="IBM",
            seed=300,
            search_steps=(0, 10, 50),
            inner_evaluations=100,
            restarts=1,
            max_native_gates=300,
        )
    if method in {"robuststate", "fontana"}:
        return pilot.construct(args, target_id, method, target)
    if method == "fixed_vqsp":
        return controls.fixed_vqsp_candidates(
            target, hardware="IBM", seed=300, depths=(2, 4), construction_steps=200, noisy_steps=200, training_paths=100
        )
    if method == "layerwise":
        return controls.layerwise_candidates(
            target,
            hardware="IBM",
            seed=300,
            depths=(1, 2, 3, 4),
            construction_steps=200,
            online_steps=100,
            noisy_steps=200,
            noisy_online_steps=100,
            training_paths=100,
        )
    msg = f"Unsupported logical-suite method: {method}"
    raise ValueError(msg)


def run_target(args: argparse.Namespace, target_id: str) -> dict[str, Any]:
    """Run missing groups for one target in an isolated, single-BLAS-thread worker.

    Returns:
        Full target groups, exclusions, and actual worker CPU/wall time.

    Raises:
        ValueError: If a resumed incumbent differs from its authenticated archive.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    groups: list[dict[str, Any]] = []
    result: dict[str, Any]
    exclusions = []
    with execution.single_threaded_numerics():
        target = load_target_collection().load(6, target_id).state_vector_copy()
        for method in args.methods:
            if method == "fontana":
                try:
                    fontana.real_target(target)
                except ValueError as error:
                    exclusions.append({"target": target_id, "method": method, "reason": str(error)})
                    continue
            relative = Path(target_id) / method / "300/IBM"
            construction_path = str(Path("construction") / relative)
            folder = args.output / construction_path
            candidates, receipt = storage.prepare_candidates(
                folder, partial(construct, args, target_id, method, target)
            )
            first = candidates[0].metadata
            reused = method == "current_b_lbfgs_frozen"
            if reused:
                # Reauthenticate the archive even when our local load-only bundle was resumed.
                fresh = construct(args, target_id, method, target)[0]
                if fresh.circuit != candidates[0].circuit or any(
                    fresh.metadata[key] != first[key]
                    for key in ("theta_sha256", "archive_checkpoint_sha256", "archive_selection_sha256")
                ):
                    msg = "Resumed incumbent source differs from its authenticated archive."
                    raise ValueError(msg)
            result = pilot.evaluate_group(
                args.output / "evaluation" / relative,
                candidates,
                target,
                identity={"target_id": target_id, "method": method, "seed": 300, "hardware": "IBM"},
                validation_paths=args.validation_paths,
                construction=receipt,
                construction_path=construction_path,
                construction_hash=storage.digest(folder / "seal.json"),
            )
            result = {
                **result,
                "construction_cache": construction_path,
                "source_output": str(args.output.resolve()),
                "reused_evaluation": False,
                "training_reused": reused,
                "build_train_cpu_seconds": first["historical_build_train_cpu_seconds"]
                if reused
                else receipt["construction_cpu_seconds"],
                "new_build_train_cpu_seconds": 0.0 if reused else receipt["construction_cpu_seconds"],
                "historical_selection_cpu_seconds": first.get("historical_selection_cpu_seconds", 0.0),
            }
            groups.append(result)
            storage.write_json(
                args.output / "progress" / f"{target_id}.json", {"groups": groups, "exclusions": exclusions}
            )
            index = result["selected_index"]
            fidelity = result["final_scores"][index]["exact"]["fidelity"] if index is not None else None
            print(target_id, method, "logical fidelity", fidelity, "training reused", reused, flush=True)
    return {
        "groups": groups,
        "exclusions": exclusions,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the suite while preserving every previously completed pilot result.

    Returns:
        Full group views, exclusions, original-grid costs and new invocation timing.

    Raises:
        ValueError: If existing output or authenticated source configurations differ.
    """
    validate_config(args)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    old = authenticate_pilot(args)
    manifest = build_manifest(args)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            msg = "Logical-suite output manifest changed; choose a separate output directory."
            raise ValueError(msg)
    elif args.output.exists() and any(args.output.iterdir()):
        msg = "Refusing nonempty logical-suite output without a manifest."
        raise ValueError(msg)
    else:
        storage.write_json(manifest_path, manifest)
    previous_invocations = []
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["schema"] != SCHEMA or previous["manifest_sha256"] != storage.digest(manifest_path):
            msg = "Previous logical-suite summary does not match its manifest."
            raise ValueError(msg)
        previous_invocations = previous.get("invocation_history", [])
        if not previous_invocations and "timing" in previous:
            previous_invocations = [{"status": previous.get("status", "unknown"), "timing": previous["timing"]}]
    invocation_started = time.time()
    groups = [group for (target, method), group in old.items() if target in args.targets and method in args.methods]
    exclusions = []
    if "haar_random_1" in args.targets and "fontana" in args.methods:
        exclusions.append({
            "target": "haar_random_1",
            "method": "fontana",
            "reason": "Fontana RY/CX ansatz requires a real target up to one global phase.",
        })
    pending = [target for target in args.targets if target not in pilot.TARGETS]
    worker_cpu = 0.0
    effective_workers = execution.resolve_workers(args.workers, len(pending))
    invocation_mode = "sequential" if effective_workers < 2 else "spawn_processes_by_target"

    def persist(status: str = "running") -> dict[str, Any]:
        ordered = sorted(
            groups,
            key=lambda group: (
                args.targets.index(group["context"]["identity"]["target_id"]),
                args.methods.index(group["context"]["identity"]["method"]),
            ),
        )
        timing = {
            "mode": invocation_mode,
            "workers": args.workers,
            "effective_workers": effective_workers,
            "invocation_wall_seconds": time.perf_counter() - start_wall,
            "parent_cpu_seconds": time.process_time() - start_cpu,
            "worker_cpu_seconds": worker_cpu,
            "full_grid_build_train_cpu_seconds": sum(group["build_train_cpu_seconds"] for group in groups),
            "new_campaign_build_train_cpu_seconds": sum(group["new_build_train_cpu_seconds"] for group in groups),
            "scope": "campaign work charged once per grid; invocation CPU includes authentication and resume overhead",
        }
        total = {
            "schema": SCHEMA,
            "manifest_sha256": storage.digest(manifest_path),
            "status": status,
            "groups": ordered,
            "exclusions": sorted(exclusions, key=lambda item: args.targets.index(item["target"])),
            "timing": timing,
            "invocation_history": [
                *previous_invocations,
                {"started_at_unix": invocation_started, "status": status, "timing": timing},
            ],
        }
        storage.write_json(summary_path, total)
        return total

    persist()
    for _index, completed in execution.completed_jobs(partial(run_target, args), pending, workers=args.workers):
        if effective_workers > 1:
            worker_cpu += completed["cpu_seconds"]
        groups.extend(completed["groups"])
        exclusions.extend(completed["exclusions"])
        persist()
    check_sources(manifest)
    return persist("complete")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the suite from a command line."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
