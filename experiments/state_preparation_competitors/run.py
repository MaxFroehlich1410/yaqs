# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run pinned competitor implementations on the nine existing q6 targets.

This is a compiled-deployment comparison. The archived incumbent retains its
original selection; new candidate grids use independent native-noise validation.
No exact final score is available to construction or selection.
Defaults include the noise-trained incumbent and controls. Noiseless synthesis
methods remain available through explicit selection for historical reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import time
from functools import partial
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qiskit import qpy

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.state_preparation_competitors import aqce, baa, controls, evaluation, histories
from experiments.state_preparation_competitors.common import Candidate
from experiments.state_preparation_competitors.reference import frozen_reference

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray
    from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("aqce", "baa_lrsp", "fixed_vqsp", "layerwise", "current_b_lbfgs_frozen")
DEFAULT_METHODS = ("current_b_lbfgs_frozen", "fixed_vqsp", "layerwise")
DETERMINISTIC = ("aqce", "baa_lrsp")
SCHEMA = "yaqs.compiled_state_preparation_competitors.v2"


def digest(path: Path) -> str:
    """Return a file's SHA-256 fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically persist finite JSON, rejecting NaN and infinity.

    Raises:
        ValueError: If a nonfinite number is present.
    """
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def save_bundle(folder: Path, record: dict[str, Any], circuits: list[QuantumCircuit]) -> dict[str, Any]:
    """Persist scalar metadata, circuits and one binary copy of parameter histories.

    Returns:
        Compact metadata with binary-history descriptors instead of iterate arrays.
    """
    compact, arrays = histories.pack_histories(record)
    descriptors = histories.history_descriptors(compact)
    json.dumps(compact, allow_nan=False)  # Validate before replacing any artifact.
    stream = io.BytesIO()
    qpy.dump(circuits, stream)
    folder.mkdir(parents=True, exist_ok=True)
    if arrays:
        histories.write_history_archive(folder / "histories.npz", arrays)
    names = ["record.json", "circuits.qpy"]
    if descriptors:
        for descriptor in descriptors:
            histories.load_history_array(folder / "histories.npz", descriptor["key"], descriptor["shape"])
        names.append("histories.npz")
    write_json(folder / "record.json", compact)
    temporary = folder / "circuits.qpy.tmp"
    temporary.write_bytes(stream.getvalue())
    temporary.replace(folder / "circuits.qpy")
    write_json(folder / "seal.json", {name: digest(folder / name) for name in names})
    if not descriptors:
        (folder / "histories.npz").unlink(missing_ok=True)
    return compact


def load_bundle(folder: Path) -> tuple[dict[str, Any], list[QuantumCircuit]]:
    """Authenticate metadata and circuits before resuming a completed stage.

    Raises:
        ValueError: If the seal is incomplete or an artifact has changed.
    """
    seal = json.loads((folder / "seal.json").read_text())
    if set(seal) not in ({"record.json", "circuits.qpy"}, {"record.json", "circuits.qpy", "histories.npz"}) or any(
        not (folder / name).is_file() or digest(folder / name) != expected for name, expected in seal.items()
    ):
        msg = f"Artifact integrity failure in {folder}."
        raise ValueError(msg)
    record = json.loads((folder / "record.json").read_text())
    descriptors = histories.history_descriptors(record)
    if bool(descriptors) != ("histories.npz" in seal):
        msg = f"History integrity failure in {folder}: descriptors and seal disagree."
        raise ValueError(msg)
    for descriptor in descriptors:
        histories.load_history_array(folder / "histories.npz", descriptor["key"], descriptor["shape"])
    return record, qpy.load(io.BytesIO((folder / "circuits.qpy").read_bytes()))


def load_candidate_metadata(output: Path, method: dict[str, Any]) -> dict[str, Any]:
    """Resolve a result's candidate reference against its authenticated construction.

    Args:
        output: Campaign root, which may have been moved together with its files.
        method: A selection/final/summary record's compact ``method`` object.

    Returns:
        Complete construction metadata, including scalar curves and history descriptors.

    Raises:
        ValueError: If the path, construction seal, index or candidate identity differs.
    """
    reference = method["construction_reference"]
    path = Path(reference["path"])
    folder = (output / path).resolve()
    if path.is_absolute() or ".." in path.parts or not folder.is_relative_to(output.resolve()):
        msg = "Construction reference must stay within the campaign directory."
        raise ValueError(msg)
    if not (folder / "seal.json").is_file() or digest(folder / "seal.json") != reference["seal_sha256"]:
        msg = "Construction reference integrity failure."
        raise ValueError(msg)
    record, circuits = load_bundle(folder)
    if len(record["candidates"]) != len(circuits):
        msg = "Candidate receipt and circuit count differ."
        raise ValueError(msg)
    index = method["candidate_index"]
    if type(index) is not int or not 0 <= index < len(record["candidates"]):
        msg = "Construction reference has an invalid candidate index."
        raise ValueError(msg)
    metadata = record["candidates"][index]
    if any(metadata[key] != method[key] for key in ("candidate_id", "method", "status")):
        msg = "Construction reference candidate identity does not match."
        raise ValueError(msg)
    return metadata


def load_parameter_history(folder: Path, descriptor: dict[str, Any]) -> NDArray[np.float64]:
    """Read one lossless parameter history after authenticating its construction bundle.

    Args:
        folder: Construction bundle containing ``histories.npz`` and its seal.
        descriptor: ``parameter_history`` from a resolved optimizer phase.

    Returns:
        Float64 array, with rows aligned to that phase's scalar ``records``.

    Raises:
        ValueError: If the archive or descriptor does not match the sealed bundle.
    """
    record, _ = load_bundle(folder)
    if descriptor not in histories.history_descriptors(record):
        msg = "Parameter history descriptor is not part of the authenticated bundle."
        raise ValueError(msg)
    return histories.load_history_array(folder / "histories.npz", descriptor["key"], descriptor["shape"])


def validation_entropy(identity: dict[str, Any], candidate_id: str) -> list[int]:
    """Return a stable, role-separated validation seed independent of training."""
    payload = json.dumps([identity, candidate_id], sort_keys=True).encode()
    fingerprint = hashlib.sha256(payload).digest()
    return [20260920, 0x56414C, *(int.from_bytes(fingerprint[i : i + 4], "little") for i in range(0, 32, 4))]


def select_candidate(records: list[dict[str, Any]]) -> int | None:
    """Select by validation alone, with native two-qubit resources as tie breakers.

    Returns:
        The selected index, or None when all candidates failed coverage/training.
    """
    eligible = [index for index, record in enumerate(records) if record["eligible"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda index: (
            -records[index]["validation"]["fidelity"],
            records[index]["compilation"]["compiled_two_qubit"],
            records[index]["compilation"]["compiled_two_qubit_depth"],
            records[index]["method"]["candidate_id"],
        ),
    )


def prepare_candidates(folder: Path, factory: Callable[[], list[Candidate]]) -> tuple[list[Candidate], dict[str, Any]]:
    """Build a grid once or reload its authenticated circuits without retraining.

    Returns:
        Candidates and actual construction cost, including every grid endpoint.

    Raises:
        ValueError: If identifiers are missing, repeated, or circuits are absent.
    """
    record: dict[str, Any]
    if (folder / "seal.json").exists():
        record, circuits = load_bundle(folder)
    else:
        cpu, wall = time.process_time(), time.perf_counter()
        candidates = factory()
        identifiers = [item.metadata.get("candidate_id") for item in candidates]
        if (
            not candidates
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            msg = "Candidate grids must be nonempty with unique string identifiers."
            raise ValueError(msg)
        record = {
            "candidates": [item.metadata for item in candidates],
            "construction_cpu_seconds": time.process_time() - cpu,
            "construction_wall_seconds": time.perf_counter() - wall,
        }
        circuits = [item.circuit for item in candidates]
        record = save_bundle(folder, record, circuits)
    if len(circuits) != len(record["candidates"]):
        msg = "Candidate receipt and circuit count differ."
        raise ValueError(msg)
    return list(starmap(Candidate, zip(circuits, record["candidates"], strict=True))), record


def evaluate_group(
    folder: Path,
    candidates: list[Candidate],
    target: NDArray[np.complex128],
    *,
    hardware: str,
    validation_paths: int,
    identity: dict[str, Any],
    construction: dict[str, Any],
    construction_receipt: str,
    construction_cache: str,
) -> dict[str, Any]:
    """Compile and validate a grid, lock its winner, then evaluate exact noise.

    The validation bundle is the immutable selection lock. A resumed final stage
    reads its authenticated circuits and never reruns selection with test scores.

    Returns:
        Every candidate's resources, validation and exact score plus the winner.

    Raises:
        ValueError: If a resumed identity, construction receipt, or register differs.
    """
    locked: dict[str, Any]
    context = {
        "identity": identity,
        "hardware": hardware,
        "validation_paths": validation_paths,
        "construction_receipt_sha256": construction_receipt,
        "construction_cache": construction_cache,
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    locked_folder, final_folder = folder / "selection_locked", folder / "final"
    if (locked_folder / "seal.json").exists():
        locked, circuits = load_bundle(locked_folder)
        if locked["context"] != context:
            msg = "Selection lock does not match this evaluation context."
            raise ValueError(msg)
    else:
        records, circuits = [], []
        reference = {"path": construction_cache, "seal_sha256": construction_receipt}
        for index, candidate in enumerate(candidates):
            compiled = (
                evaluation.preserve_native_candidate(candidate, target)
                if candidate.metadata.get("preserve_native_schedule") is True
                else evaluation.compile_candidate(candidate, target)
            )
            cpu, wall = time.process_time(), time.perf_counter()
            entropy = validation_entropy(identity, candidate.metadata["candidate_id"])
            bank = evaluation.sample_validation(compiled.circuit, hardware, validation_paths, entropy)
            validation = evaluation.validation_score(compiled, hardware, bank)
            validation.update({
                "entropy": entropy,
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
            })
            coverage = validation["jump_paths"] > 0 or len(compiled.circuit.data) == 0
            records.append({
                "method": {
                    **{key: candidate.metadata[key] for key in ("candidate_id", "method", "status")},
                    "candidate_index": index,
                    "construction_reference": reference,
                },
                "compilation": compiled.metadata,
                "validation": validation,
                "validation_coverage": "complete" if coverage else "no_jump_coverage",
                "eligible": candidate.metadata["status"] == "complete" and coverage,
            })
            circuits.append(compiled.circuit)
        locked = {
            "context": context,
            "records": records,
            "selected_index": select_candidate(records),
            "selection_rule": (
                "maximum independent validation fidelity; ties use fewer native CX, CX depth, candidate ID"
            ),
            "construction": {
                "reference": reference,
                **{key: value for key, value in construction.items() if key != "candidates"},
            },
            "selection_scope": "frozen archived winner"
            if identity["method"] == "current_b_lbfgs_frozen"
            else "declared candidate grid",
        }
        save_bundle(locked_folder, locked, circuits)
    selection_hash = digest(locked_folder / "seal.json")
    if (final_folder / "seal.json").exists():
        result, _ = load_bundle(final_folder)
        if result["selection_lock_sha256"] != selection_hash:
            msg = "Final evaluation does not match its selection lock."
            raise ValueError(msg)
        return result
    scores = []
    for record, circuit in zip(locked["records"], circuits, strict=True):
        target_physical = evaluation.physical_target(target, record["compilation"]["logical_to_physical_output"])
        compiled = evaluation.CompiledCandidate(circuit, target_physical, record["compilation"])
        cpu, wall = time.process_time(), time.perf_counter()
        exact = evaluation.exact_score(compiled, hardware)
        scores.append({
            "candidate_id": record["method"]["candidate_id"],
            "exact": exact,
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.perf_counter() - wall,
        })
    result = {
        **locked,
        "selection_lock_sha256": selection_hash,
        "final_scores": scores,
        "status": "complete" if locked["selected_index"] is not None else "no_eligible_candidate",
        "assessment": "exact density matrix; all modeled noise outcomes, no test-trajectory sampling error",
    }
    save_bundle(final_folder, result, circuits)
    return result


def make_parser() -> argparse.ArgumentParser:
    """Build the CLI with noise-trained methods by default and explicit candidate grids."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
        help=(
            "Methods to evaluate; defaults to the noise-trained incumbent and controls: "
            "%(default)s. AQCE and BAA-LRSP require explicit selection for historical reproduction."
        ),
    )
    parser.add_argument("--hardware", nargs="+", choices=evaluation.HARDWARE, default=list(evaluation.HARDWARE))
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 400])
    parser.add_argument("--depths", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--aqce-gates", nargs="+", type=int, default=[0, 5, 10, 20, 40, 80])
    parser.add_argument(
        "--aqce-sweeps", type=int, default=100, help="Full forward/backward sweeps (paper Algorithm 3)."
    )
    parser.add_argument("--aqce-initial-gates", type=int, default=1, help="M0 density-based initialization gates.")
    parser.add_argument("--aqce-growth-increment", type=int, default=1, help="Delta M identity gates per enlargement.")
    parser.add_argument("--baa-losses", nargs="+", type=float, default=[0, 0.001, 0.01, 0.03, 0.1, 0.2])
    parser.add_argument("--construction-steps", type=int, default=200)
    parser.add_argument("--online-steps", type=int, default=100)
    parser.add_argument("--noisy-steps", type=int, default=200)
    parser.add_argument("--noisy-online-steps", type=int, default=100)
    parser.add_argument("--training-paths", type=int, default=100)
    parser.add_argument("--validation-paths", type=int, default=32768)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "output/topdown-batch-refinement-q6-20260919")
    return parser


def validate_config(args: argparse.Namespace) -> None:
    """Reject invalid budgets and duplicate jobs before writing output.

    Raises:
        ValueError: If lists or integer budgets are invalid.
    """
    for name in ("targets", "methods", "hardware", "seeds", "depths", "aqce_gates", "baa_losses"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            msg = f"{name} must be nonempty without duplicates."
            raise ValueError(msg)
    integers = (
        "aqce_sweeps",
        "aqce_initial_gates",
        "construction_steps",
        "online_steps",
        "noisy_steps",
        "noisy_online_steps",
    )
    if any(getattr(args, name) < 0 for name in integers) or any(seed < 0 for seed in args.seeds):
        msg = "Seeds and update budgets must be nonnegative."
        raise ValueError(msg)
    if args.training_paths < 1 or args.validation_paths < 2:
        msg = "Training needs at least one path and validation at least two."
        raise ValueError(msg)
    if args.online_steps > args.construction_steps or args.noisy_online_steps > args.noisy_steps:
        msg = "Online budgets cannot exceed their phase totals."
        raise ValueError(msg)
    if (
        args.depths != sorted(args.depths)
        or args.depths[0] < 0
        or args.aqce_gates != sorted(args.aqce_gates)
        or args.aqce_gates[0] < 0
    ):
        msg = "Depths and AQCE budgets must be sorted nonnegative integers."
        raise ValueError(msg)
    if args.aqce_growth_increment < 1 or any(
        budget and (budget < args.aqce_initial_gates or (budget - args.aqce_initial_gates) % args.aqce_growth_increment)
        for budget in args.aqce_gates
    ):
        msg = "AQCE budgets must be reachable from initial gates with a positive growth increment."
        raise ValueError(msg)
    if "layerwise" in args.methods and args.depths != list(range(1, args.depths[-1] + 1)):
        msg = "Layerwise depths must be contiguous from one."
        raise ValueError(msg)
    if any(not 0 <= loss < 1 for loss in args.baa_losses):
        msg = "BAA infidelity budgets must lie in [0, 1)."
        raise ValueError(msg)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint pinned runtime, fixture, controls and all numerical dependencies.

    Raises:
        ValueError: If an installed optional dependency differs from the pin.
    """
    requirements = Path(__file__).with_name("requirements.txt")
    packages = {}
    for line in requirements.read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, expected = line.split("==")
            packages[package] = importlib.metadata.version(package)
            if packages[package] != expected:
                msg = f"Pinned {package}=={expected} required; use the competitor requirements file."
                raise ValueError(msg)
    paths = set((ROOT / "experiments").rglob("*.py")) | set(Path(__file__).parent.rglob("*.txt"))
    paths |= set(Path(__file__).parent.glob("*.md"))
    paths |= set((ROOT / "src/mqt/yaqs").rglob("*.py"))
    paths |= set((ROOT / "benchmarks/state_preparation").glob("*.py"))
    paths |= {ROOT / "experiments/topdown_state_preparation_method.md", DEFAULT_TARGET_PATH}
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "output"
    }
    return {
        "schema": SCHEMA,
        "config": config,
        "python": platform.python_version(),
        "packages": packages,
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)},
        "compiler": evaluation.COMPILER,
        "noise": "legacy synthetic single-jump coefficients after every compiled operation including RZ",
        "reference_scope": "archived B+LBFGS endpoint; no retraining or native-noise depth reselection",
        "budget_scope": "predeclared method-specific grids; actual CPU reported, not compute-matched",
        "cost_scope": "sealed completed stages including ineligible candidates; excludes interrupted unsealed attempts",
        "reference_manifest_sha256": digest(args.reference_root / "manifest.json")
        if "current_b_lbfgs_frozen" in args.methods
        else None,
    }


def construct(
    args: argparse.Namespace, method: str, target_id: str, target: NDArray[np.complex128], seed: int, hardware: str
) -> list[Candidate]:
    """Dispatch to upstream libraries or existing controls without changing them.

    Returns:
        The method's complete declared candidate grid.

    Raises:
        ValueError: If the method name is unknown.
    """
    if method == "aqce":
        return aqce.aqce_candidates(
            target,
            gate_budgets=tuple(args.aqce_gates),
            sweeps=args.aqce_sweeps,
            initial_gates=args.aqce_initial_gates,
            growth_increment=args.aqce_growth_increment,
        )
    if method == "baa_lrsp":
        return baa.baa_candidates(target, fidelity_losses=tuple(args.baa_losses))
    if method == "current_b_lbfgs_frozen":
        return [frozen_reference(args.reference_root, target_id, seed, hardware, target)]
    options: dict[str, Any] = {
        "depths": tuple(args.depths),
        "seed": seed,
        "hardware": hardware,
        "construction_steps": args.construction_steps,
        "noisy_steps": args.noisy_steps,
        "training_paths": args.training_paths,
    }
    if method == "fixed_vqsp":
        return controls.fixed_vqsp_candidates(target, **options)
    if method == "layerwise":
        return controls.layerwise_candidates(
            target, online_steps=args.online_steps, noisy_online_steps=args.noisy_online_steps, **options
        )
    msg = f"Unknown competitor method {method}."
    raise ValueError(msg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume the declared sweep, persisting per-candidate exact frontiers.

    Returns:
        Authenticated group results. Deterministic synthesis is shared by profiles.

    Raises:
        ValueError: If an existing output directory belongs to another protocol.
    """
    validate_config(args)
    manifest = build_manifest(args)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            msg = "Output manifest changed; use a new directory for a different configuration or source."
            raise ValueError(msg)
    elif args.output.exists() and any(args.output.iterdir()):
        msg = "Refusing a nonempty output directory without a matching manifest."
        raise ValueError(msg)
    else:
        write_json(manifest_path, manifest)
    collection, results = load_target_collection(), []
    for target_id in args.targets:
        target = collection.load(6, target_id).state_vector_copy()
        for method in args.methods:
            seeds = [0] if method in DETERMINISTIC else args.seeds
            for seed in seeds:
                for hardware in args.hardware:
                    profile = "shared" if method in DETERMINISTIC else hardware
                    relative = Path(target_id) / method / str(seed) / profile
                    source_folder = args.output / "construction" / relative
                    candidates, construction = prepare_candidates(
                        source_folder, partial(construct, args, method, target_id, target, seed, hardware)
                    )
                    identity = {
                        "target_id": target_id,
                        "method": method,
                        "seed": None if method in DETERMINISTIC else seed,
                        "hardware": hardware,
                    }
                    result = evaluate_group(
                        args.output / "evaluation" / target_id / method / str(seed) / hardware,
                        candidates,
                        target,
                        hardware=hardware,
                        validation_paths=args.validation_paths,
                        identity=identity,
                        construction=construction,
                        construction_receipt=digest(source_folder / "seal.json"),
                        construction_cache=str(Path("construction") / relative),
                    )
                    result["construction_cache"] = str(Path("construction") / relative)
                    result["cost_note"] = (
                        "Charge each distinct construction_cache once for campaign CPU; "
                        "its full cost applies to a standalone hardware run."
                    )
                    results.append(result)
                    print(
                        json.dumps({
                            **identity,
                            "status": result["status"],
                            "selected_index": result["selected_index"],
                        }),
                        flush=True,
                    )
    summary = {"schema": SCHEMA, "manifest_sha256": digest(manifest_path), "groups": results}
    write_json(args.output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the CLI; a completed run writes summary.json and replay artifacts."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
