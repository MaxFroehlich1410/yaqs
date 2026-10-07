# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run paper-based noisy competitors with authenticated, independent assessment.

The default primary comparison includes NACL and RobustState. Explicit optional
Fontana/IQSP requests record unsupported targets as excluded, never as failures
or zero fidelities. This driver reuses the existing compact storage, independent
validation, selection locks, and exact final evaluator.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.state_preparation.constants import TARGET_IDS
from benchmarks.state_preparation.targets import DEFAULT_TARGET_PATH, load_target_collection
from experiments.state_preparation_competitors import evaluation, fontana, iqsp, nacl, robuststate
from experiments.state_preparation_competitors import run as storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from experiments.state_preparation_competitors.common import Candidate

METHODS = ("nacl", "robuststate", "fontana", "iqsp")
SCHEMA = "yaqs.noisy_paper_competitors.v1"
PINNED = {"numpy": "2.4.6", "scipy": "1.17.1", "qiskit": "2.4.1"}


def make_parser() -> argparse.ArgumentParser:
    """Return a CLI separating method defaults from explicit smoke overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGET_IDS, default=list(TARGET_IDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["nacl", "robuststate"])
    parser.add_argument("--hardware", nargs="+", choices=evaluation.HARDWARE, default=list(evaluation.HARDWARE))
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 400])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument(
        "--construction-steps",
        type=int,
        default=None,
        help="Override warm-start updates; IQSP counts updates per interpolation stage. NACL uses its own budgets.",
    )
    parser.add_argument(
        "--noisy-steps",
        type=int,
        default=None,
        help="Override noisy updates (RobustState 50, Fontana 200, IQSP 10000).",
    )
    parser.add_argument("--nacl-steps", nargs="+", type=int, default=[0, 10, 50])
    parser.add_argument("--nacl-evaluations", type=int, default=100)
    parser.add_argument("--nacl-restarts", type=int, default=1)
    parser.add_argument("--nacl-max-gates", type=int, default=300)
    parser.add_argument("--robust-feedback", choices=["exact_density", "pauli_shadows"], default="exact_density")
    parser.add_argument("--shots-per-basis", type=int, default=1024)
    parser.add_argument(
        "--shadow-bases", type=int, default=None, help="All 3**6 bases when omitted; a fresh random subset otherwise."
    )
    parser.add_argument("--iqsp-interpolation-steps", type=int, default=20)
    parser.add_argument("--validation-paths", type=int, default=32768)
    return parser


def validate_config(args: argparse.Namespace) -> None:
    """Validate all budgets before creating files or invoking optional solvers.

    Raises:
        ValueError: If a requested grid, identifier or count is invalid.
    """
    for key in ("targets", "methods", "hardware", "seeds", "depths", "nacl_steps"):
        values = getattr(args, key)
        if not values or len(set(values)) != len(values):
            msg = f"{key} must be nonempty and unique."
            raise ValueError(msg)
    for values, choices in ((args.targets, TARGET_IDS), (args.methods, METHODS), (args.hardware, evaluation.HARDWARE)):
        if any(value not in choices for value in values):
            msg = "Unknown target, method or hardware profile."
            raise ValueError(msg)
    for key in ("seeds", "depths", "nacl_steps"):
        values = getattr(args, key)
        if any(type(value) is not int or value < (1 if key == "depths" else 0) for value in values):
            msg = f"{key} contains invalid integer values."
            raise ValueError(msg)
        if key != "seeds" and values != sorted(values):
            msg = f"{key} must be sorted."
            raise ValueError(msg)
    for key in ("construction_steps", "noisy_steps"):
        value = getattr(args, key)
        if value is not None and (type(value) is not int or value < 0):
            msg = f"{key} must be nonnegative or omitted."
            raise ValueError(msg)
    for key in (
        "nacl_evaluations",
        "nacl_restarts",
        "nacl_max_gates",
        "shots_per_basis",
        "iqsp_interpolation_steps",
        "validation_paths",
    ):
        value = getattr(args, key)
        if type(value) is not int or value < (2 if key == "validation_paths" else 1):
            msg = f"{key} must be a positive integer (validation needs at least two)."
            raise ValueError(msg)
    if args.shadow_bases is not None and (type(args.shadow_bases) is not int or not 1 <= args.shadow_bases <= 3**6):
        msg = "shadow_bases must lie in [1,729] or be omitted."
        raise ValueError(msg)
    if args.robust_feedback not in {"exact_density", "pauli_shadows"}:
        msg = "Unsupported RobustState feedback."
        raise ValueError(msg)
    if "nacl" in args.methods and (args.nacl_max_gates < 5 or any(seed > 2**32 - 1 for seed in args.seeds)):
        msg = "NACL requires at least five native gates in its cap and seeds no greater than 2**32-1."
        raise ValueError(msg)


def eligibility(method: str, target: NDArray[np.complex128]) -> str | None:
    """Return an explicit unsupported-target reason, or None for an eligible pair.

    Raises:
        ValueError: If the method identifier is unknown.
    """
    if method not in METHODS:
        msg = f"Unknown noisy method {method}."
        raise ValueError(msg)
    try:
        if method == "fontana":
            fontana.real_target(target)
        elif method == "iqsp":
            iqsp.validate_gaussian_target(target)
    except ValueError as error:
        return str(error)
    return None


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint actual numerical code, fixtures, package pins and configuration.

    Raises:
        ValueError: If an installed numerical dependency differs from its pin.
        PackageNotFoundError: If a required selected-method dependency is missing.
    """
    required = {**PINNED, **({"PyNomadBBO": "4.6.0"} if "nacl" in args.methods else {})}
    packages = {}
    for package, expected in required.items():
        packages[package] = importlib.metadata.version(package)
        if packages[package] != expected:
            msg = f"Require {package}=={expected} for an authenticated comparison."
            raise ValueError(msg)
    paths = set((storage.ROOT / "experiments").rglob("*.py"))
    paths |= set((storage.ROOT / "benchmarks/state_preparation").glob("*.py"))
    paths |= set((storage.ROOT / "src/mqt/yaqs").rglob("*.py"))
    paths |= set(Path(__file__).parent.glob("*.md")) | set(Path(__file__).parent.glob("*.txt"))
    paths.add(DEFAULT_TARGET_PATH)
    return {
        "schema": SCHEMA,
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "packages": packages,
        "python": platform.python_version(),
        "sources": {str(path.relative_to(storage.ROOT)): storage.digest(path) for path in sorted(paths)},
        "compiler": evaluation.COMPILER,
        "noise": "existing synthetic IBM/Google Pauli mixture after every RZ/SX/X/CX; no idle/readout noise",
        "assessment": "exact density final evaluator; role-separated sampled validation selects the declared grid",
        "fidelity_scope": "simulator benchmark adaptations, not reproductions of published hardware results",
        "cost_scope": "all completed construction/grid/validation work; excludes interrupted unsealed attempts",
        "budget_scope": "method-specific grids and actual costs; iteration counts do not match compute",
    }


def construct(
    args: argparse.Namespace, method: str, target: NDArray[np.complex128], seed: int, hardware: str
) -> list[Candidate]:
    """Return one method's complete grid with its own published update policy.

    Raises:
        ValueError: If the method or its target family is unsupported.
    """
    reason = eligibility(method, target)
    if reason is not None:
        raise ValueError(reason)
    if method == "nacl":
        return nacl.nacl_candidates(
            target,
            hardware=hardware,
            seed=seed,
            search_steps=tuple(args.nacl_steps),
            inner_evaluations=args.nacl_evaluations,
            restarts=args.nacl_restarts,
            max_native_gates=args.nacl_max_gates,
        )
    construction_default = 500 if method == "robuststate" else 200
    noisy_default = {"robuststate": 50, "fontana": 200, "iqsp": 10000}[method]
    warm = construction_default if args.construction_steps is None else args.construction_steps
    noisy = noisy_default if args.noisy_steps is None else args.noisy_steps
    common: dict[str, Any] = {"hardware": hardware, "seed": seed, "depths": tuple(args.depths)}
    if method == "robuststate":
        return robuststate.robuststate_candidates(
            target,
            **common,
            noiseless_steps=warm,
            noisy_steps=noisy,
            feedback=args.robust_feedback,
            shots_per_basis=args.shots_per_basis,
            num_bases=args.shadow_bases,
        )
    if method == "fontana":
        return fontana.fontana_candidates(target, **common, construction_steps=warm, noisy_steps=noisy)
    return iqsp.iqsp_candidates(
        target, **common, construction_steps=warm, noisy_steps=noisy, interpolation_steps=args.iqsp_interpolation_steps
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run or authenticate-resume the sweep, explicitly recording subset exclusions.

    Returns:
        Compact completed groups and excluded method/target pairs.

    Raises:
        ValueError: If an output directory belongs to a different configuration.
    """
    validate_config(args)
    manifest = build_manifest(args)
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            msg = "Output manifest changed; choose a new directory."
            raise ValueError(msg)
    elif args.output.exists() and any(args.output.iterdir()):
        msg = "Refusing a nonempty output directory without a matching manifest."
        raise ValueError(msg)
    else:
        storage.write_json(path, manifest)
    collection = load_target_collection()
    results, exclusions = [], []
    for target_id in args.targets:
        target = collection.load(6, target_id).state_vector_copy()
        for method in args.methods:
            reason = eligibility(method, target)
            if reason is not None:
                exclusions.append({
                    "target_id": target_id,
                    "method": method,
                    "status": "unsupported_target",
                    "reason": reason,
                })
                continue
            # IQSP's prescribed uniform starting state is deterministic. Its
            # noisy stage still depends on the hardware and cannot be shared.
            for seed in [0] if method == "iqsp" else args.seeds:
                for hardware in args.hardware:
                    relative = Path(target_id) / method / str(seed) / hardware
                    folder = args.output / "construction" / relative
                    candidates, receipt = storage.prepare_candidates(
                        folder, partial(construct, args, method, target, seed, hardware)
                    )
                    identity = {
                        "target_id": target_id,
                        "method": method,
                        "seed": None if method == "iqsp" else seed,
                        "hardware": hardware,
                    }
                    result = storage.evaluate_group(
                        args.output / "evaluation" / relative,
                        candidates,
                        target,
                        hardware=hardware,
                        validation_paths=args.validation_paths,
                        identity=identity,
                        construction=receipt,
                        construction_receipt=storage.digest(folder / "seal.json"),
                        construction_cache=str(Path("construction") / relative),
                    )
                    result["construction_cache"] = str(Path("construction") / relative)
                    results.append(result)
                    print(
                        json.dumps({
                            **identity,
                            "status": result["status"],
                            "selected_index": result["selected_index"],
                        }),
                        flush=True,
                    )
    summary = {"schema": SCHEMA, "manifest_sha256": storage.digest(path), "groups": results, "exclusions": exclusions}
    storage.write_json(args.output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the explicit noisy-paper CLI without launching any implicit full study."""
    run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
