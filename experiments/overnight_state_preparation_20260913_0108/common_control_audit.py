# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Score every developed control at the frozen structural-panel CPU cutoff."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import composed_experiment as composed

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = "overnight.common_control_audit.v1"


def read(path: Path) -> dict[str, Any]:
    """Authenticate a heterogeneous saved research record."""
    return cast("dict[str, Any]", pilot.read_record(path))


def select_points(arms: dict[str, Any], methods: Sequence[str], cutoff: float) -> dict[str, Any]:
    """Select last chronological available points, including completed shorter arms.

    Raises:
        ValueError: If any requested arm fails or its timing/availability is invalid.
    """
    if not methods or len(set(methods)) != len(methods) or not math.isfinite(cutoff) or cutoff < 0:
        msg = "A finite nonnegative cutoff and distinct complete methods are required."
        raise ValueError(msg)
    selected = {}
    for method in methods:
        arm = arms.get(method, {})
        curve = arm.get("curve", [])
        times = [point["cpu_seconds"] for point in curve]
        if (
            arm.get("status") != "complete"
            or not times
            or any(not math.isfinite(value) or value < 0 for value in times)
            or times != sorted(times)
        ):
            msg = "Every requested control needs a complete chronological curve."
            raise ValueError(msg)
        eligible = [point for point in curve if point["cpu_seconds"] <= cutoff]
        if not eligible:
            msg = "A control has no deployable point at the fixed cutoff."
            raise ValueError(msg)
        selected[method] = eligible[-1]
    return selected


def sources() -> dict[str, str]:
    """Bind this diagnostic and its protocol without editing frozen scientific code."""
    result = composed.sources()
    for path in (Path(__file__), Path(__file__).with_name("COMMON_CONTROL_PROTOCOL.md")):
        result[str(path.relative_to(composed.runner.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def input_panel(root: Path) -> dict[str, Any]:
    """Require the completed original development composition and all its durable bytes."""
    protocol, summary = read(root / "PROTOCOL.json"), read(root / "SUMMARY.json")
    composed.verify(protocol)
    if (
        summary["status"] != "complete"
        or summary["protocol_checksum"] != pilot.digest(protocol)
        or set(summary["case_checksums"]) != set(composed.TARGETS)
    ):
        msg = "The audit requires the complete four-target development panel."
        raise ValueError(msg)
    cases = []
    for source in protocol["starts"]:
        case_root = root / f"{source['target_id']}__11"
        case = read(case_root / "RESULT.json")
        if (
            case["status"] != "complete"
            or case["signature"] != pilot.digest([protocol, source])
            or summary["case_checksums"][source["target_id"]] != pilot.digest(case)
            or any(
                hashlib.sha256((case_root / name).read_bytes()).hexdigest() != value
                for name, value in case["dependencies"].items()
            )
        ):
            msg = "A source case or its complete durable dependencies changed."
            raise ValueError(msg)
        cutoff = case["common_primary"]["refinement_cpu_seconds"]
        if cutoff != min(case["arms"][name]["refinement_cpu_seconds"] for name in composed.PRIMARY):
            msg = "The structural cutoff differs from its frozen definition."
            raise ValueError(msg)
        selected = select_points(case["arms"], composed.TRAINED, cutoff)
        if any(selected[name] != case["common_primary"]["arms"][name]["point"] for name in composed.PRIMARY):
            msg = "The original structural selection must remain unchanged."
            raise ValueError(msg)
        cases.append({"target_id": source["target_id"], "checksum": pilot.digest(case), "points": selected})
    return {
        "root": str(root.resolve()),
        "protocol_checksum": pilot.digest(protocol),
        "summary_checksum": pilot.digest(summary),
        "cases": cases,
    }


def prepare(output: Path, source_root: Path) -> dict[str, Any]:
    """Freeze all selected control points before any additional fidelity evaluation."""
    protocol = {
        "version": VERSION,
        "phase": "development_diagnostic",
        "sources": sources(),
        "input_panel": input_panel(source_root.resolve()),
        "environment": composed.initializer_experiment.construction.environment(),
        "benchmark_hashes": composed.runner.custody()["benchmark_hashes"],
    }
    path = output / "PROTOCOL.json"
    if path.exists() and read(path) != protocol:
        msg = "Frozen control audit changed; use a fresh output root."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, protocol)
    return protocol


def score_case(output: Path, source_root: Path, entry: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate extra controls in a separate circuit bank, preserving all old artifacts."""
    source_case = source_root / f"{entry['target_id']}__11"
    case = read(source_case / "RESULT.json")
    if pilot.digest(case) != entry["checksum"]:
        msg = "Control source case changed after freeze."
        raise ValueError(msg)
    config = composed.exact.Config(**read(source_root / "PROTOCOL.json")["exact_config"])
    target = load_target(6, entry["target_id"]).state_vector_copy()
    store = composed.CompleteStore(output, target, config)
    signature = pilot.digest([protocol, entry])
    metadata = {
        "status": "complete",
        "target_id": entry["target_id"],
        "refinement_cpu_seconds": case["common_primary"]["refinement_cpu_seconds"],
        "total_cpu_seconds": case["common_primary"]["total_cpu_seconds"],
        "points": entry["points"],
        "source_case_checksum": entry["checksum"],
        "source_unchanged_fidelities": case["arms"]["unchanged"]["scores"]["fidelities"],
    }

    def compute() -> dict[str, Any]:
        scores = {name: case["common_primary"]["arms"][name]["scores"] for name in composed.PRIMARY}
        for name in composed.TRAINED:
            point = entry["points"][name]
            if name in scores:
                continue
            raw = read(source_case / "circuits" / f"{point['circuit_hash']}.json")
            if store.circuit(composed.restore_physical(raw)) != point["circuit_hash"]:
                msg = "The selected physical circuit changed."
                raise ValueError(msg)
            scores[name] = composed.score(store, point)
        return {
            **metadata,
            "scores": scores,
            "dependencies": {
                str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*.json")
                if path.parent != output
            },
        }

    result = composed.tracked_call(output / "RESULT.json", signature, compute)
    current = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*.json")
        if path.parent != output
    }
    if current != result["value"]["dependencies"]:
        msg = "Completed control audit dependency changed."
        raise ValueError(msg)
    if any(result["value"].get(key) != value for key, value in metadata.items()):
        msg = "Cached control points, cutoff or source identity changed."
        raise ValueError(msg)
    expected_scores = {name: case["common_primary"]["arms"][name]["scores"] for name in composed.PRIMARY}
    for name in composed.TRAINED:
        if name in expected_scores:
            continue
        point = entry["points"][name]
        identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
        if not (output / "operations" / f"audit_{pilot.digest(identity)}.json").is_file():
            msg = "A completed control is missing its physical score receipt."
            raise ValueError(msg)
        expected_scores[name] = composed.score(store, point)
    if result["value"]["scores"] != expected_scores:
        msg = "Cached control scores differ from their authenticated receipts."
        raise ValueError(msg)
    return result


def summarize(records: Sequence[dict[str, Any]], targets: Sequence[str]) -> dict[str, Any]:
    """Report every control and the fixed gate against the strongest aggregate control."""
    cases = {row["value"]["target_id"]: row["value"] for row in records}
    if len(cases) != len(records) or set(cases) != set(targets) or not targets:
        msg = "A strongest-control aggregate requires the entire distinct target panel."
        raise ValueError(msg)
    methods: dict[str, Any] = {}
    for name in composed.METHODS:
        values = {}
        for target in targets:
            case = cases[target]
            scores = case["source_unchanged_fidelities"] if name == "unchanged" else case["scores"][name]["fidelities"]
            if set(scores) != set(composed.channels.NOISE_IDS) or any(not math.isfinite(v) for v in scores.values()):
                msg = "Every control needs all twelve finite endpoint scores."
                raise ValueError(msg)
            values[target] = float(np.mean([scores[k] for k in composed.channels.NOISE_IDS if k != "noiseless"]))
        methods[name] = {"mean": float(np.mean(list(values.values()))), "by_target": values}
    candidate = "structural_mean2_lbfgs"
    strongest = max((name for name in composed.METHODS if name != candidate), key=lambda name: methods[name]["mean"])
    comparisons: dict[str, Any] = {}
    for name in composed.METHODS:
        if name == candidate:
            continue
        deltas = {
            target: 100 * (methods[candidate]["by_target"][target] - methods[name]["by_target"][target])
            for target in targets
        }
        comparisons[name] = {
            "mean_delta_pp": float(np.mean(list(deltas.values()))),
            "by_target_delta_pp": deltas,
            "target_wins": sum(value > 1e-8 for value in deltas.values()),
        }
    comparison = comparisons[strongest]
    return {
        "methods": methods,
        "candidate": candidate,
        "strongest_control": strongest,
        "paired_comparisons": comparisons,
        "same_rule_passes_against_strongest": comparison["mean_delta_pp"] >= 0.5
        and comparison["target_wins"] == len(targets),
        "comparison": "all controls at the original four-structural common CPU cutoff; development diagnostic",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare first, then score only frozen development controls without new training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=composed.runner.OUTPUT / "common-control-audit")
    parser.add_argument("--source-root", type=Path, default=composed.runner.OUTPUT / "conditional-composition")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        prepare(args.output, args.source_root)
        return 0
    protocol = read(args.output / "PROTOCOL.json")
    expected = prepare(args.output, Path(protocol["input_panel"]["root"]))
    if protocol != expected or protocol["sources"] != sources():
        msg = "Control audit source freeze changed."
        raise ValueError(msg)
    records = [
        score_case(args.output / f"{entry['target_id']}__11", Path(protocol["input_panel"]["root"]), entry, protocol)
        for entry in protocol["input_panel"]["cases"]
    ]
    analysis = summarize(records, composed.TARGETS)
    pilot.write_record(
        args.output / "SUMMARY.json",
        {
            "status": "complete",
            "protocol_checksum": pilot.digest(protocol),
            "cases": [pilot.digest(r) for r in records],
            **analysis,
        },
    )
    print(analysis, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
