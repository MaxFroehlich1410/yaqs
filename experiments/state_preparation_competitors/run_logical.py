# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare the three existing IBM pilot targets with noise on original gates.

Reuse authenticated training that already used this channel; train RobustState
and Fontana on their original elementary gates. Validation and locked final
assessment use that same representation. Native compilation is resource-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.state_preparation_competitors import evaluation, fontana, logical_evaluation, reference, robuststate
from experiments.state_preparation_competitors import run as storage
from experiments.state_preparation_competitors.common import Candidate
from experiments.state_preparation_competitors.run_noisy import PINNED

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

TARGETS = ("gaussian_mu0p5_sigma0p1", "tfim_critical", "haar_random_1")
METHODS = ("current_b_lbfgs_frozen", "nacl", "robuststate", "fontana", "fixed_vqsp", "layerwise")
REUSED = {"current_b_lbfgs_frozen", "nacl", "fixed_vqsp", "layerwise"}
SCHEMA = "yaqs.logical_noise_pilot.v1"
# These modules are not training dependencies of the imported controls/NACL.
# run.py changed only its default method list since the Gaussian pilot.
REUSE_UNRELATED_CHANGES = {
    "experiments/state_preparation_competitors/run.py",
    "experiments/state_preparation_competitors/fontana.py",
    "experiments/state_preparation_competitors/robuststate.py",
}


def make_parser() -> argparse.ArgumentParser:
    """Return the fixed-budget, IBM seed-300 pilot interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--validation-paths", type=int, default=8192)
    parser.add_argument(
        "--gaussian-source", type=Path, default=storage.ROOT / "output/gaussian-all-methods-pilot-20260920"
    )
    parser.add_argument("--other-source", type=Path, default=storage.ROOT / "output/tfim-haar-noisy-ibm-pilot-20260920")
    parser.add_argument("--reference-root", type=Path, default=reference.DEFAULT_REFERENCE_ROOT)
    return parser


def validate_config(args: argparse.Namespace) -> None:
    """Reject unsupported jobs before creating output or calling a solver.

    Raises:
        ValueError: If targets, methods or trajectory counts are invalid.
    """
    for name, choices in (("targets", TARGETS), ("methods", METHODS)):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values) or any(value not in choices for value in values):
            msg = f"{name} must contain unique supported identifiers."
            raise ValueError(msg)
    if type(args.validation_paths) is not int or args.validation_paths < 2:
        msg = "Validation requires at least two raw trajectories."
        raise ValueError(msg)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Record fixed budgets, channel placement, source hashes and dependency pins.

    Raises:
        ValueError: If a numerical dependency differs from its required version.
    """
    packages = {name: importlib.metadata.version(name) for name in PINNED}
    if packages != PINNED:
        msg = "Logical pilot requires the same pinned NumPy, SciPy and Qiskit versions as the native pilot."
        raise ValueError(msg)
    paths = set((storage.ROOT / "experiments").rglob("*.py"))
    paths |= set((storage.ROOT / "benchmarks/state_preparation").glob("*.py"))
    paths |= set((storage.ROOT / "src/mqt/yaqs").rglob("*.py"))
    paths |= {DEFAULT_TARGET_PATH, Path(__file__).with_name("LOGICAL_NOISE.md")}
    return {
        "schema": SCHEMA,
        "config": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "hardware": "IBM",
        "optimizer_seed": 300,
        "packages": packages,
        "python": platform.python_version(),
        "sources": {str(path.relative_to(storage.ROOT)): storage.digest(path) for path in sorted(paths)},
        "noise": "same arity-dependent Pauli channel after every original elementary gate in training/validation/test",
        "compiler_role": "resource accounting only; compiled circuits never determine noisy fidelity",
        "budgets": {
            "current_b_lbfgs_frozen": "original logical-noise selected B/lbfgs endpoint; unchanged M=100 training",
            "nacl": "reuse original RZ/SX/CX search, checkpoints 0/10/50, 100 inner evaluations, one restart",
            "robuststate": "depths 2/4, 500 ideal and 50 noisy Adam updates, exact-density feedback",
            "fontana": "real targets, depths 2/4, 200 ideal and 200 noisy BFGS iterations",
            "fixed_vqsp": "reuse logical-trained depths 2/4, 200 ideal and 200 noisy updates, M=100",
            "layerwise": "reuse logical-trained depths 1/2/3/4, 200+200 updates with 100 online each, M=100",
        },
    }


def reuse_candidates(
    args: argparse.Namespace, target_id: str, method: str, target: NDArray[np.complex128]
) -> list[Candidate]:
    """Authenticate matching old training and reference its histories in place.

    Returns:
        Unchanged source circuits with compact immutable provenance references.

    Raises:
        ValueError: If sources, budgets, fixture, metadata or archive identity differ.
    """
    folder = (args.gaussian_source if target_id == TARGETS[0] else args.other_source / target_id) / method
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if storage.digest(manifest_path) != summary["manifest_sha256"]:
        msg = "Reused manifest checksum differs from its original summary."
        raise ValueError(msg)
    changed = []
    for name, expected in manifest["sources"].items():
        if name.endswith(".py") and storage.digest(storage.ROOT / name) != expected:
            if name not in REUSE_UNRELATED_CHANGES:
                msg = f"Reused training source changed: {name}"
                raise ValueError(msg)
            changed.append(name)
    config = manifest["config"]
    expected_config: dict[str, Any] = {"targets": [target_id], "methods": [method], "hardware": ["IBM"], "seeds": [300]}
    if method in {"fixed_vqsp", "layerwise"}:
        expected_config.update({
            "depths": [2, 4] if method == "fixed_vqsp" else [1, 2, 3, 4],
            "construction_steps": 200,
            "noisy_steps": 200,
            "training_paths": 100,
            "online_steps": 100,
            "noisy_online_steps": 100,
        })
    elif method == "nacl":
        expected_config.update({
            "nacl_steps": [0, 10, 50],
            "nacl_evaluations": 100,
            "nacl_restarts": 1,
            "nacl_max_gates": 300,
        })
    if method not in REUSED or any(config.get(key) != value for key, value in expected_config.items()):
        msg = "Reused construction does not match the declared logical pilot budgets."
        raise ValueError(msg)
    relative = Path("construction") / target_id / method / "300/IBM"
    record, circuits = storage.load_bundle(folder / relative)
    expected_grid = {
        "fixed_vqsp": ("depth", [2, 4]),
        "layerwise": ("depth", [1, 2, 3, 4]),
        "nacl": ("search_steps_per_restart", [0, 10, 50]),
    }
    if method in expected_grid:
        field, values = expected_grid[method]
        if [item.get(field) for item in record["candidates"]] != values or len(circuits) != len(values):
            msg = "Reused candidate grid differs from its declared budget."
            raise ValueError(msg)
    old_final, _ = storage.load_bundle(folder / "evaluation" / target_id / method / "300/IBM/final")
    receipt_hash = storage.digest(folder / relative / "seal.json")
    expected_identity = {"target_id": target_id, "method": method, "seed": 300, "hardware": "IBM"}
    context = old_final["context"]
    if (
        context["target_sha256"] != hashlib.sha256(target.tobytes()).hexdigest()
        or context["identity"] != expected_identity
        or context["construction_receipt_sha256"] != receipt_hash
    ):
        msg = "Reused construction target, identity or receipt differs from its original evaluation."
        raise ValueError(msg)
    cost = record["construction_cpu_seconds"]
    historical_validation = 0.0
    if method == "current_b_lbfgs_frozen":
        fresh = reference.frozen_reference(args.reference_root, target_id, 300, "IBM", target)
        old = record["candidates"][0]
        if (
            len(circuits) != 1
            or fresh.circuit != circuits[0]
            or any(
                fresh.metadata[key] != old[key]
                for key in ("theta_sha256", "archive_checkpoint_sha256", "archive_selection_sha256")
            )
        ):
            msg = "Archived baseline differs from the saved pilot endpoint."
            raise ValueError(msg)
        search = fresh.metadata["archived_search_cost"]
        cost = sum(
            search[key] for key in ("construction_cpu_seconds", "pruning_cpu_seconds", "fine_tuning_cpu_seconds")
        )
        historical_validation = search["validation_cpu_seconds"]
    candidates = []
    for index, (metadata, circuit) in enumerate(zip(record["candidates"], circuits, strict=True)):
        if metadata["method"] != method or metadata["seed"] != 300 or metadata["hardware"] != "IBM":
            msg = "Reused candidate identity differs from its requested method or seed."
            raise ValueError(msg)
        if method == "nacl":
            evaluation.validate_circuit(circuit, native=True)
        candidates.append(
            Candidate(
                circuit,
                {
                    **{key: metadata[key] for key in ("method", "candidate_id", "status")},
                    "seed": 300,
                    "hardware": "IBM",
                    "noise_basis": "logical",
                    "training_reused": True,
                    "historical_build_train_cpu_seconds": cost,
                    "historical_selection_cpu_seconds": historical_validation,
                    "source_candidate": {
                        "output": str(folder.resolve()),
                        "construction_path": str(relative),
                        "construction_seal_sha256": receipt_hash,
                        "candidate_index": index,
                        "manifest_sha256": storage.digest(manifest_path),
                        "unrelated_changed_sources": changed,
                    },
                    "reuse_reason": "original elementary schedule already uses the requested logical noise channel",
                },
            )
        )
    return candidates


def construct(args: argparse.Namespace, target_id: str, method: str, target: NDArray[np.complex128]) -> list[Candidate]:
    """Train only missing logical-noise methods with unchanged paper policies.

    Raises:
        ValueError: If an unsupported method is requested.
    """
    if method in REUSED:
        return reuse_candidates(args, target_id, method, target)
    if method == "robuststate":
        return robuststate.robuststate_candidates(
            target, hardware="IBM", seed=300, depths=(2, 4), noiseless_steps=500, noisy_steps=50, noise_basis="logical"
        )
    if method == "fontana":
        return fontana.fontana_candidates(
            target,
            hardware="IBM",
            seed=300,
            depths=(2, 4),
            construction_steps=200,
            noisy_steps=200,
            noise_basis="logical",
        )
    msg = f"Unsupported logical pilot method: {method}"
    raise ValueError(msg)


def evaluate_group(
    folder: Path,
    candidates: list[Candidate],
    target: NDArray[np.complex128],
    *,
    identity: dict[str, Any],
    validation_paths: int,
    construction: dict[str, Any],
    construction_path: str,
    construction_hash: str,
) -> dict[str, Any]:
    """Validate original gates, seal selection, then compute exact logical scores.

    Returns:
        Complete candidate diagnostics and the validation-selected winner.

    Raises:
        ValueError: If resumed receipts disagree with the requested context.
    """
    locked: dict[str, Any]
    context = {
        "identity": identity,
        "noise_basis": "logical",
        "validation_paths": validation_paths,
        "construction_receipt_sha256": construction_hash,
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    locked_folder, final_folder = folder / "selection_locked", folder / "final"
    if (locked_folder / "seal.json").exists():
        locked, circuits = storage.load_bundle(locked_folder)
        if locked["context"] != context:
            msg = "Logical selection context changed."
            raise ValueError(msg)
    else:
        records, circuits = [], []
        for index, candidate in enumerate(candidates):
            logical = logical_evaluation.logical_resources(candidate.circuit, target)
            # Compile a private copy only to report comparable physical resources.
            compiled = evaluation.compile_candidate(candidate, target)
            entropy = storage.validation_entropy(
                {**identity, "noise_basis": "logical"}, candidate.metadata["candidate_id"]
            )
            cpu, wall = time.process_time(), time.perf_counter()
            validation = logical_evaluation.score_validation(
                candidate.circuit, target, "IBM", validation_paths, entropy
            )
            validation.update({"cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - wall})
            coverage = validation["jump_paths"] > 0 or not candidate.circuit.data
            records.append({
                "method": {
                    **{key: candidate.metadata[key] for key in ("method", "candidate_id", "status")},
                    "candidate_index": index,
                    "construction_reference": {"path": construction_path, "seal_sha256": construction_hash},
                },
                "logical_resources": logical,
                "compilation": {**compiled.metadata, "role": "resources only; no native noise evaluation"},
                "validation": validation,
                "eligible": candidate.metadata["status"] == "complete" and coverage,
                "validation_coverage": "complete" if coverage else "no_jump_coverage",
            })
            circuits.append(candidate.circuit)
        locked = {
            "context": context,
            "records": records,
            "selected_index": storage.select_candidate(records),
            "selection_rule": "maximum logical validation fidelity; ties: native CX, CX depth, candidate ID",
            "selection_scope": "frozen logical-noise archive winner"
            if identity["method"] == "current_b_lbfgs_frozen"
            else "declared candidate grid",
            "construction": {key: value for key, value in construction.items() if key != "candidates"},
        }
        storage.save_bundle(locked_folder, locked, circuits)
    lock_hash = storage.digest(locked_folder / "seal.json")
    if (final_folder / "seal.json").exists():
        result, _ = storage.load_bundle(final_folder)
        if (
            result["selection_lock_sha256"] != lock_hash
            or any(result.get(key) != value for key, value in locked.items())
            or [score["candidate_id"] for score in result["final_scores"]]
            != [record["method"]["candidate_id"] for record in locked["records"]]
        ):
            msg = "Logical final result differs from its selection lock."
            raise ValueError(msg)
        return result
    scores = []
    for record, circuit in zip(locked["records"], circuits, strict=True):
        cpu, wall = time.process_time(), time.perf_counter()
        exact = logical_evaluation.score_exact(circuit, target, "IBM")
        scores.append({
            "candidate_id": record["method"]["candidate_id"],
            "exact": exact,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        })
    result = {
        **locked,
        "selection_lock_sha256": lock_hash,
        "final_scores": scores,
        "status": "complete" if locked["selected_index"] is not None else "no_eligible_candidate",
        "assessment": "exact logical-gate density matrix; no test trajectory sampling error",
    }
    storage.save_bundle(final_folder, result, circuits)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute or authenticate-resume the fixed matched-logical pilot.

    Returns:
        Completed groups and explicit unsupported Fontana target exclusions.

    Raises:
        ValueError: If an existing output belongs to a different configuration.
    """
    validate_config(args)
    result: dict[str, Any]
    manifest = build_manifest(args)
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            msg = "Logical output manifest changed; choose a new directory."
            raise ValueError(msg)
    elif args.output.exists() and any(args.output.iterdir()):
        msg = "Refusing nonempty logical output without a manifest."
        raise ValueError(msg)
    else:
        storage.write_json(path, manifest)
    groups, exclusions = [], []
    collection = load_target_collection()
    for target_id in args.targets:
        target = collection.load(6, target_id).state_vector_copy()
        for method in args.methods:
            if method == "fontana":
                try:
                    fontana.real_target(target)
                except ValueError as error:
                    exclusions.append({"target": target_id, "method": method, "reason": str(error)})
                    continue
            relative = Path(target_id) / method / "300/IBM"
            construction_path = str(Path("construction") / relative)
            construction_folder = args.output / construction_path
            candidates, construction = storage.prepare_candidates(
                construction_folder, partial(construct, args, target_id, method, target)
            )
            result = evaluate_group(
                args.output / "evaluation" / relative,
                candidates,
                target,
                identity={"target_id": target_id, "method": method, "seed": 300, "hardware": "IBM"},
                validation_paths=args.validation_paths,
                construction=construction,
                construction_path=construction_path,
                construction_hash=storage.digest(construction_folder / "seal.json"),
            )
            first = candidates[0].metadata
            reused = bool(first.get("training_reused", False))
            result = {
                **result,
                "construction_cache": construction_path,
                "training_reused": reused,
                "build_train_cpu_seconds": first["historical_build_train_cpu_seconds"]
                if reused
                else construction["construction_cpu_seconds"],
                "new_build_train_cpu_seconds": 0.0 if reused else construction["construction_cpu_seconds"],
                "historical_selection_cpu_seconds": first.get("historical_selection_cpu_seconds", 0.0),
            }
            groups.append(result)
            storage.write_json(
                args.output / "summary.json",
                {"schema": SCHEMA, "manifest_sha256": storage.digest(path), "groups": groups, "exclusions": exclusions},
            )
            index = result["selected_index"]
            fidelity = result["final_scores"][index]["exact"]["fidelity"] if index is not None else None
            print(target_id, method, "logical fidelity", fidelity, "training reused", reused, flush=True)
    summary = {"schema": SCHEMA, "manifest_sha256": storage.digest(path), "groups": groups, "exclusions": exclusions}
    storage.write_json(args.output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fixed pilot from command-line arguments."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
