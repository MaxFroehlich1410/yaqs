# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Opt-in native-MPS scaling pilot with fixed 1e-8 canonical propagation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.state_preparation import mps_targets
from experiments import topdown_backend_support as storage
from experiments import topdown_product_screening as screening
from experiments import topdown_structured_q20 as pilot
from experiments.topdown_structured_q20_canonical import CanonicalProgressBackend
from mqt.yaqs.optimization.canonical_mps_backend import CanonicalMPSBackend

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.optimization import ParameterizedCircuit

PROPAGATION_CUTOFF = 1e-8
VERSION = "topdown.native_mps_scaling.v1"
REPO = Path(__file__).resolve().parents[1]


def sources() -> dict[str, str]:
    """Fingerprint the native workflow and its reused preparation/propagation kernels."""
    paths = {
        Path(__file__),
        Path(mps_targets.__file__),
        Path(screening.__file__),
        Path(pilot.__file__),
        Path(storage.__file__),
        REPO / "experiments/topdown_structured_q20_canonical.py",
        REPO / "experiments/legacy_topdown_dense.py",
        REPO / "experiments/topdown_batch_refinement_policies.py",
        REPO / "experiments/topdown_guarded_pruning.py",
    }
    for folder in ("optimization", "core/data_structures", "core/methods", "core/linalg"):
        paths.update((REPO / "src/mqt/yaqs" / folder).glob("*.py"))
    return {str(p.relative_to(REPO)): storage.digest(p) for p in sorted(paths)}


def precision() -> dict:
    """Keep target/input precision separate from the explicit propagation approximation."""
    return {"relative_cutoff": PROPAGATION_CUTOFF, "input_relative_cutoff": float(np.finfo(float).eps)}


def backend(circuit: ParameterizedCircuit, target: MPS, path: Path) -> CanonicalProgressBackend:
    """Construct the existing tested backend with the required 1e-8 cutoff."""
    return CanonicalProgressBackend(circuit, target, path, relative_cutoff=PROPAGATION_CUTOFF)


def create(root: Path, settings: pilot.Settings | None = None, *, bond: int = 2, target_seed: int = 5002) -> dict:
    """Seal a new tensor-only target and pilot plan without starting optimization.

    Returns:
        The saved plan, with separate target and optimizer seeds.
    """
    settings = settings or replace(pilot.Settings(), qubits=30, max_depth=4)
    root.mkdir(parents=True, exist_ok=False)
    mps_targets.write_mps_target(root / "target.json", settings.qubits, bond, target_seed)
    plan = {
        "version": VERSION,
        "settings": asdict(settings),
        "precision": precision(),
        "target_manifest_sha256": storage.digest(root / "target.json"),
        "sources": sources(),
        "final_assessment": "untruncated forward MPO; training cutoff does not set final MPO precision",
        "pruning": "original B/G deletion policies with explicitly approximate canonical 1e-8 replay",
    }
    storage.write_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(storage.digest(root / "plan.json") + "\n")
    return plan


def verify(root: Path) -> tuple[dict, pilot.Settings, MPS]:
    """Reject changed sources, target bytes or precision before any stage executes.

    Returns:
        Authenticated plan, settings and a private target MPS.

    Raises:
        ValueError: If the scientific identity or native target width changed.
    """
    plan = json.loads((root / "plan.json").read_text())
    if (
        (root / "plan.sha256").read_text().strip() != storage.digest(root / "plan.json")
        or plan["version"] != VERSION
        or plan["precision"] != precision()
        or plan["sources"] != sources()
        or plan["target_manifest_sha256"] != storage.digest(root / "target.json")
    ):
        msg = "Native pilot source, precision or target identity changed."
        raise ValueError(msg)
    settings = pilot.Settings(**plan["settings"])
    target, _ = mps_targets.load_mps_target(root / "target.json")
    if target.length != settings.qubits:
        msg = "Native target width differs from the optimizer plan."
        raise ValueError(msg)
    return plan, settings, target


def run_stage(root: Path, stage: str) -> None:
    """Run one explicitly claimed original stage, with no dense target conversion.

    Args:
        root: New pilot root created by this module.
        stage: Existing preparation, B/G refinement or final MPO assessment stage.

    Raises:
        ValueError: If the stage is unsupported or prerequisites are absent.
    """
    if stage not in pilot.STAGES:
        msg = "Unknown native-MPS pilot stage."
        raise ValueError(msg)
    plan, settings, target = verify(root)
    folder = root / "job"
    claim = root / "claims" / stage
    claim.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    storage.write_json(
        claim / "started.json", {"stage": stage, "plan_sha256": storage.digest(root / "plan.json"), **precision()}
    )
    if stage == "prepare":
        storage.save_checkpoint(
            folder / "target",
            {
                "qubits": settings.qubits,
                "approximation": {
                    "representation": "native MPS tensor definition",
                    "pure_target_fidelity_error_bound": 0.0,
                    "scope": "stored tensor target; excludes floating-point roundoff",
                },
            },
            {f"site_{i}": t for i, t in enumerate(target.tensors)},
        )
        record, arrays = screening.initial_parameters_mps(
            target,
            settings.seed,
            settings.product_candidates,
            settings.product_batch_size,
        )
        storage.save_checkpoint(folder / "initialization", record, arrays)
        pilot.optimize_preparation(
            folder, target, arrays["theta"], settings, backend_factory=backend, started_cpu=started
        )
    elif stage in {"B", "G"}:
        if storage.load_checkpoint(folder / "parent") is None:
            msg = "Native pilot preparation is incomplete."
            raise ValueError(msg)
        pilot.train_branch(
            folder,
            stage,
            settings,
            backend_factory=backend,
            pruning_backend_factory=partial(CanonicalMPSBackend, relative_cutoff=PROPAGATION_CUTOFF),
        )
    else:
        policy, endpoint, _ = stage.split("_")
        pilot.assess(folder, policy, endpoint.lower(), settings)
    verify(root)
    storage.write_json(
        claim / "complete.json",
        {
            "stage": stage,
            "precision": plan["precision"],
            "cpu_seconds": time.process_time() - started,
            "resources": pilot.resources(),
        },
    )


def main(argv: list[str] | None = None) -> None:
    """Create a fresh pilot or execute one stage with one numerical thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qubits", type=int, default=30)
    parser.add_argument("--bond", type=int, default=2)
    parser.add_argument("--target-seed", type=int, default=5002)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--stage", choices=(*pilot.STAGES, "all"))
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        if args.action == "create":
            create(
                args.root,
                replace(pilot.Settings(), qubits=args.qubits, seed=args.seed, max_depth=args.max_depth),
                bond=args.bond,
                target_seed=args.target_seed,
            )
        else:
            if args.stage is None:
                parser.error("run requires an explicit --stage")
            for stage in pilot.STAGES if args.stage == "all" else (args.stage,):
                run_stage(args.root, stage)


if __name__ == "__main__":
    main()
