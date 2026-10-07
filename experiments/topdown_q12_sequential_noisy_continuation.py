# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matched sequential-pruning continuation of the six structured q12 targets."""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

from experiments import legacy_pruning_ablation as pruning
from experiments import topdown_q12_noisy_continuation as base

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import StatevectorBackend

VERSION = "topdown.q12_sequential_noisy_continuation.v1"
TARGET_IDS = tuple(target for target in base.TARGET_IDS if not target.startswith("haar_random_"))
PROTOCOL = {
    **base.PROTOCOL,
    "version": VERSION,
    "pruning": "sequential C, minimum signed ideal deletion score, rescore after each deletion, cutoff 0.01",
    "pruning_repairs": 0,
    "pruning_stop": "no score strictly below cutoff or empty circuit; no cumulative fidelity guard",
    "pruning_ties": "stable original gate order",
    "pruning_cutoff": 0.01,
}
_base_sources = base.sources


def sources() -> dict[str, str]:
    """Fingerprint the unchanged continuation and this sequential adapter.

    Returns:
        Relative source paths and SHA-256 digests.
    """
    return {
        **_base_sources(),
        str(Path(__file__).relative_to(base.topdown_mps.ROOT)): base.support.digest(Path(__file__)),
    }


def sequential_prune(engine: StatevectorBackend, theta: NDArray) -> tuple:
    """Use the archived C rule without its optional repair or density evaluation.

    Returns:
        Reduced circuit, surviving angles and complete sequential decision trace.

    Raises:
        ValueError: If the archived cutoff differs from the matched experiment.
    """
    if pruning.PROTOCOL["cutoff"] != PROTOCOL["pruning_cutoff"]:
        msg = "Sequential comparison requires the original cutoff 0.01."
        raise ValueError(msg)
    return pruning.prune_variant(engine.circuit, theta, engine.target, "C")


@contextmanager
def sequential_policy() -> Generator[None, None, None]:
    """Select C inside one worker and restore all imported defaults on exit.

    Yields:
        The unchanged refinement, checkpoint and assessment workflow using C.
    """
    bindings = [
        (base, "VERSION", VERSION),
        (base, "PROTOCOL", PROTOCOL),
        (base, "sources", sources),
        (base.support, "prune_b", sequential_prune),
    ]
    original = [(owner, name, getattr(owner, name)) for owner, name, _ in bindings]
    for owner, name, value in bindings:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in original:
            setattr(owner, name, value)


def prepare(references: Path, output: Path) -> dict:
    """Authenticate and freeze the same eighteen structured noiseless endpoints.

    Returns:
        Protocol, source fingerprints, job identities and sealed input hashes.

    Raises:
        ValueError: If the panel is incomplete, duplicated or includes Haar targets.
    """
    rows = json.loads(references.read_text(encoding="utf-8"))
    expected = {(target, seed) for target in TARGET_IDS for seed in (300, 400, 500)}
    if len(rows) != 18 or {(r["target_id"], r["seed"]) for r in rows} != expected or output.exists():
        msg = "Require a fresh output and all six structured targets with three seeds."
        raise ValueError(msg)
    jobs = []
    with sequential_policy():
        for row in rows:
            provenance, stage, arrays = base.authenticate_parent(Path(row["parent"]), row["target_id"], row["seed"])
            job_id = f"{row['target_id']}__seed{row['seed']}"
            folder = output / "inputs" / job_id
            base.support.save_checkpoint(folder / "construction", stage, arrays)
            receipt = {
                "version": VERSION,
                **provenance,
                "construction_seal_sha256": base.support.digest(folder / "construction/complete.json"),
            }
            base.support.write_json(folder / "input.json", receipt)
            jobs.append({"job_id": job_id, "target_id": row["target_id"], "seed": row["seed"], "host": row["host"]})
        plan = {
            "protocol": PROTOCOL,
            "sources": sources(),
            "jobs": jobs,
            "inputs": {j["job_id"]: base.support.digest(output / "inputs" / j["job_id"] / "input.json") for j in jobs},
            "references_sha256": base.support.digest(references),
        }
        shutil.copyfile(references, output / "parent-references.json")
        base.support.write_json(output / "plan.json", plan)
        for job in jobs:
            base.authenticate_input(output / "inputs" / job["job_id"])
    return plan


def run_case(folder: Path, input_folder: Path) -> dict:
    """Reuse the original q12 phases with sequential pruning and a distinct identity.

    Returns:
        Final fidelities, resources, CPU costs and authenticated checkpoint seals.
    """
    with sequential_policy():
        return base.run_case(folder, input_folder)


def main(argv: list[str] | None = None) -> None:
    """Prepare the structured panel or run one explicit independent continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            if args.references is None:
                parser.error("Preparation requires references.")
            result = prepare(args.references, args.output)
        else:
            if args.input is None:
                parser.error("Run requires a sealed input.")
            result = run_case(args.output, args.input)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
