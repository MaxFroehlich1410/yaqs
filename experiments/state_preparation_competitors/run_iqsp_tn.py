# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run IQSP tensor-network training on the frozen nine six-qubit target states.

The Gaussian uses the published function interpolation. The other targets use
an explicitly generalized state interpolation, not a claim of paper reproduction.
Training and independent validation use the same IBM logical-gate channel. Final
dense assessment is independent and occurs only after validation locks selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation import execution
from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.state_preparation_competitors import iqsp_tn, logical_evaluation
from experiments.state_preparation_competitors import run as storage
from experiments.state_preparation_competitors import run_logical as pilot

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from experiments.state_preparation_competitors.common import Candidate

METHOD = "iqsp_tn"
SCHEMA = "yaqs.iqsp_tensor_network_logical_suite.v1"
PINNED = {"numpy": "2.4.6", "scipy": "1.17.1", "qiskit": "2.4.1", "torch": "2.14.0", "opt_einsum": "3.4.0"}


def make_parser() -> argparse.ArgumentParser:
    """Return the all-target logical-noise IQSP tensor-network interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--interpolation-steps", type=int, default=20)
    parser.add_argument("--construction-steps", type=int, default=200, help="Adam updates per interpolation stage.")
    parser.add_argument("--noisy-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-paths", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="Write and authenticate the plan without training.")
    return parser


def validate_config(args: argparse.Namespace) -> None:
    """Reject invalid target grids and numerical budgets before touching output.

    Raises:
        ValueError: If identifiers, integer budgets, or the learning rate are invalid.
    """
    if (
        not args.targets
        or len(set(args.targets)) != len(args.targets)
        or any(t not in TARGET_IDS for t in args.targets)
    ):
        msg = "targets must contain unique supported identifiers."
        raise ValueError(msg)
    if (
        not args.depths
        or len(set(args.depths)) != len(args.depths)
        or any(type(depth) is not int or depth < 1 for depth in args.depths)
    ):
        msg = "depths must contain unique positive integers."
        raise ValueError(msg)
    for field, minimum in (
        ("interpolation_steps", 1),
        ("construction_steps", 0),
        ("noisy_steps", 0),
        ("validation_paths", 2),
    ):
        value = getattr(args, field)
        if type(value) is not int or value < minimum:
            msg = f"{field} must be an integer of at least {minimum}."
            raise ValueError(msg)
    if (
        isinstance(args.learning_rate, bool)
        or not isinstance(args.learning_rate, (int, float))
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        msg = "learning_rate must be finite and positive."
        raise ValueError(msg)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze configuration, fixture, source fingerprints and numerical versions.

    Raises:
        ValueError: If a required numerical dependency differs from its pinned version.
    """
    packages = {name: importlib.metadata.version(name) for name in PINNED}
    if packages != PINNED:
        msg = "IQSP tensor-network runner requires the pinned numerical dependency versions."
        raise ValueError(msg)
    paths = set((storage.ROOT / "experiments").rglob("*.py"))
    paths |= set((storage.ROOT / "benchmarks/state_preparation").glob("*.py"))
    paths |= set((storage.ROOT / "src/mqt/yaqs").rglob("*.py"))
    paths.add(DEFAULT_TARGET_PATH)
    paths |= {Path(__file__).with_name(name) for name in ("requirements-iqsp-tn.txt", "IQSP_TN.md")}
    paths |= {Path(__file__).parent / "vendor/iqsp_tensor" / name for name in ("LICENSE", "PROVENANCE.md")}
    return {
        "schema": SCHEMA,
        "config": {key: value for key, value in vars(args).items() if key not in {"output", "dry_run"}},
        "method": METHOD,
        "hardware": "IBM",
        "num_qubits": 6,
        "validation_seed": 300,
        "initialization": "deterministic zero SU(4) exponential parameters; no optimizer RNG",
        "packages": packages,
        "python": platform.python_version(),
        "sources": {str(path.relative_to(storage.ROOT)): storage.digest(path) for path in sorted(paths)},
        "training_representation": "tensor-network pure-state construction and mixed-state noisy contraction",
        "representation_scope": (
            "untruncated circuit TN with local Kraus legs; not the authors' compressed CombTN runtime"
        ),
        "noise_basis": "logical",
        "noise": (
            "same IBM arity-dependent Pauli channel after every original elementary gate in training/validation/test"
        ),
        "noise_placement": logical_evaluation.NOISE_PLACEMENT,
        "phase_schedules": {
            "construction": "Hadamards followed by exponential SU(4) blocks, without noise",
            "noisy_training": "fixed exact Weyl/ZYZ elementary schedule including zero-angle rotations",
            "final_assessment": "post-cutoff elementary schedule; deleted gates have no noise windows",
        },
        "two_qubit_cutoff": {
            "absolute_rotation_angle_at_most": iqsp_tn.TWO_QUBIT_CUTOFF,
            "placement": "once after noisy refinement; placement is a declared adapter choice",
            "additional_optimization_steps": 0,
        },
        "compiler_role": "resource accounting only; compiled circuits never determine noisy fidelity",
        "assessment": "independent validation trajectories; selection lock precedes exact-density final assessment",
        "target_protocols": {
            target_id: "published Gaussian exponential path"
            if target_id == TARGET_IDS[0]
            else "generalized phase-aligned geodesic path; not a published non-Gaussian benchmark"
            for target_id in args.targets
        },
        "runtime_policy": "charge all interpolation stages, decomposition and noisy refinement across every depth",
    }


def check_sources(manifest: dict[str, Any]) -> None:
    """Reject mid-run source or fixture changes before sealing a final summary.

    Raises:
        ValueError: If any recorded source fingerprint no longer matches.
    """
    for name, expected in manifest["sources"].items():
        path = storage.ROOT / name
        if not path.is_file() or storage.digest(path) != expected:
            msg = f"IQSP tensor-network source changed during the run: {name}"
            raise ValueError(msg)


def require_complete_bundle(folder: Path) -> None:
    """Refuse to overwrite a partially written nonempty stage directory.

    Raises:
        ValueError: If an existing nonempty bundle has no authentication seal.
    """
    if folder.exists() and any(folder.iterdir()) and not (folder / "seal.json").is_file():
        msg = f"Refusing partial nonempty IQSP tensor-network bundle: {folder}"
        raise ValueError(msg)


def candidate_context(args: argparse.Namespace, target_id: str, target: NDArray[np.complex128]) -> dict[str, Any]:
    """Return the immutable fixture and training configuration attached to each endpoint."""
    return {
        "target_id": target_id,
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "depths": list(args.depths),
        "interpolation_steps": args.interpolation_steps,
        "construction_steps_per_stage": args.construction_steps,
        "noisy_steps": args.noisy_steps,
        "learning_rate": args.learning_rate,
        "hardware": "IBM",
        "noise_basis": "logical",
    }


def construct(args: argparse.Namespace, target_id: str, target: NDArray[np.complex128]) -> list[Candidate]:
    """Run the complete tensor-network grid and attach its immutable training context."""
    candidates = iqsp_tn.iqsp_tn_candidates(
        target,
        target_id=target_id,
        depths=tuple(args.depths),
        interpolation_steps=args.interpolation_steps,
        construction_steps=args.construction_steps,
        noisy_steps=args.noisy_steps,
        hardware="IBM",
        learning_rate=args.learning_rate,
    )
    context = candidate_context(args, target_id, target)
    for candidate in candidates:
        candidate.metadata["runner_context"] = context
    authenticate_candidates(args, target_id, target, candidates)
    return candidates


def authenticate_candidates(
    args: argparse.Namespace, target_id: str, target: NDArray[np.complex128], candidates: list[Candidate]
) -> None:
    """Check the original or resumed endpoint grid before independent assessment.

    Raises:
        ValueError: If identity, training context, depth grid or gate representation differs.
    """
    expected = candidate_context(args, target_id, target)
    if len(candidates) != len(args.depths):
        msg = "IQSP tensor-network candidate grid size differs from the requested depths."
        raise ValueError(msg)
    for depth, candidate in zip(args.depths, candidates, strict=True):
        metadata = candidate.metadata
        fields = {"method": METHOD, "candidate_id": f"depth_{depth:02d}", "status": "complete", "depth": depth}
        if (
            any(metadata.get(key) != value for key, value in fields.items())
            or metadata.get("noise_basis") != "logical"
            or metadata.get("hardware") != "IBM"
            or metadata.get("runner_context") != expected
            or 2**candidate.circuit.num_qubits != target.size
        ):
            msg = "IQSP tensor-network candidate identity, logical noise or training context changed."
            raise ValueError(msg)
        logical_evaluation.validate_logical_circuit(candidate.circuit)


def authenticate_assessment(folder: Path, candidates: list[Candidate]) -> None:
    """Require a cached assessment to preserve the authenticated original circuits.

    Raises:
        ValueError: If candidate identities, indices or stored circuits differ.
    """
    if not (folder / "seal.json").is_file():
        return
    record, circuits = storage.load_bundle(folder)
    if circuits != [candidate.circuit for candidate in candidates] or len(record["records"]) != len(candidates):
        msg = "IQSP tensor-network assessment circuits or grid size differ from construction."
        raise ValueError(msg)
    for index, (item, candidate) in enumerate(zip(record["records"], candidates, strict=True)):
        method = item["method"]
        if method["candidate_index"] != index or any(
            method[key] != candidate.metadata[key] for key in ("method", "candidate_id", "status")
        ):
            msg = "IQSP tensor-network assessment candidate identity differs from construction."
            raise ValueError(msg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or authenticate-resume the tensor-network comparison in a separate archive.

    Returns:
        A dry-run plan or completed groups with independent assessment and full costs.

    Raises:
        ValueError: If output provenance, partial bundles, sources or candidates disagree.
    """
    validate_config(args)
    manifest = build_manifest(args)
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            msg = "IQSP tensor-network output manifest changed; choose a new directory."
            raise ValueError(msg)
    elif args.output.exists() and any(args.output.iterdir()):
        msg = "Refusing nonempty IQSP tensor-network output without a manifest."
        raise ValueError(msg)
    else:
        storage.write_json(path, manifest)
    plan = {
        "schema": SCHEMA,
        "manifest_sha256": storage.digest(path),
        "groups": len(args.targets),
        "candidates": len(args.targets) * len(args.depths),
        "status": "planned",
    }
    if args.dry_run:
        storage.write_json(args.output / "plan.json", plan)
        return plan
    cpu, wall = time.process_time(), time.perf_counter()
    groups: list[dict[str, Any]] = []
    collection = load_target_collection()
    with execution.single_threaded_numerics():
        for target_id in args.targets:
            target = collection.load(6, target_id).state_vector_copy()
            relative = Path(target_id) / METHOD / "300/IBM"
            construction_path = str(Path("construction") / relative)
            folder = args.output / construction_path
            evaluation_folder = args.output / "evaluation" / relative
            for stage in (folder, evaluation_folder / "selection_locked", evaluation_folder / "final"):
                require_complete_bundle(stage)
            resumed = (folder / "seal.json").is_file()
            candidates, construction = storage.prepare_candidates(folder, partial(construct, args, target_id, target))
            authenticate_candidates(args, target_id, target, candidates)
            for stage in ("selection_locked", "final"):
                authenticate_assessment(evaluation_folder / stage, candidates)
            result = pilot.evaluate_group(
                evaluation_folder,
                candidates,
                target,
                identity={"target_id": target_id, "method": METHOD, "seed": 300, "hardware": "IBM"},
                validation_paths=args.validation_paths,
                construction=construction,
                construction_path=construction_path,
                construction_hash=storage.digest(folder / "seal.json"),
            )
            check_sources(manifest)
            groups.append({
                **result,
                "construction_cache": construction_path,
                "training_reused": resumed,
                "build_train_cpu_seconds": construction["construction_cpu_seconds"],
                "new_build_train_cpu_seconds": 0.0 if resumed else construction["construction_cpu_seconds"],
                "target_protocol": manifest["target_protocols"][target_id],
            })
            storage.write_json(
                args.output / "summary.json",
                {**plan, "status": "running", "groups": groups, "exclusions": []},
            )
            print(target_id, METHOD, "logical assessment", result["status"], "training reused", resumed, flush=True)
    summary = {
        **plan,
        "status": "complete" if all(group["status"] == "complete" for group in groups) else "incomplete",
        "groups": groups,
        "exclusions": [],
        "timing": {
            "invocation_cpu_seconds": time.process_time() - cpu,
            "invocation_wall_seconds": time.perf_counter() - wall,
            "build_train_cpu_seconds": sum(group["build_train_cpu_seconds"] for group in groups),
            "new_build_train_cpu_seconds": sum(group["new_build_train_cpu_seconds"] for group in groups),
        },
    }
    storage.write_json(args.output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run or prepare the IQSP tensor-network comparison from command-line arguments."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
