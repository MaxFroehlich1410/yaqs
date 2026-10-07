# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent endpoint audit and paired, target-balanced Experiment 2 summaries."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.state_preparation.calibration import CalibratedLocalDepolarizingProvider
from benchmarks.state_preparation.exact_density import density_loss_gradient
from benchmarks.state_preparation.execution import single_threaded_numerics
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_trajectory_comparison as experiment

if TYPE_CHECKING:
    from collections.abc import Sequence


def paired_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return balanced endpoint/cost summaries and hybrid-minus-comparator pairs."""
    if len(rows) != 80 or any(
        not 0 <= cast("float", row["final_fidelity"]) <= 1
        or not np.isfinite(cast("float", row["cpu_seconds"]))
        or cast("float", row["cpu_seconds"]) <= 0
        for row in rows
    ):
        msg = "Expected eighty finite endpoints with positive CPU measurements."
        raise ValueError(msg)
    methods: dict[str, dict[str, object]] = {}
    for kind in experiment.KINDS:
        cases = [row for row in rows if cast("dict", row["method"])["kind"] == kind]
        if len(cases) != 16 or len({(row["target_id"], row["replica"]) for row in cases}) != 16:
            msg = "Expected sixteen distinct paired cases per method."
            raise ValueError(msg)
        families = {}
        for family in experiment.FAMILIES:
            members = [row for row in cases if row["family"] == family]
            targets = sorted({str(row["target_id"]) for row in members})
            if (
                len(members) != 4
                or len(targets) != 2
                or any(
                    {row["replica"] for row in members if row["target_id"] == target} != {11, 29} for target in targets
                )
            ):
                msg = "Expected two targets and two replicas per family."
                raise ValueError(msg)
            families[family] = {
                "fidelity": float(np.mean([row["final_fidelity"] for row in members])),
                "frozen_fidelity": float(np.mean([row["frozen_fidelity"] for row in members])),
                "mean_cpu_seconds": float(np.mean([row["cpu_seconds"] for row in members])),
            }
        methods[kind] = {
            "fidelity": float(np.mean([row["final_fidelity"] for row in cases])),
            "frozen_fidelity": float(np.mean([row["frozen_fidelity"] for row in cases])),
            "noiseless_fidelity": float(np.mean([row["final_noiseless_fidelity"] for row in cases])),
            "mean_cpu_seconds": float(np.mean([row["cpu_seconds"] for row in cases])),
            "mean_wall_seconds": float(np.mean([row["wall_seconds"] for row in cases])),
            "families": families,
        }
    indexed = {(row["target_id"], row["replica"], cast("dict", row["method"])["kind"]): row for row in rows}
    hybrids = [row for row in rows if cast("dict", row["method"])["kind"] == "trajectory_hybrid"]
    comparisons = {}
    for kind in experiment.KINDS:
        if kind == "trajectory_hybrid":
            continue
        paired = []
        for row in hybrids:
            other = indexed[row["target_id"], row["replica"], kind]
            if row["frozen_fidelity"] != other["frozen_fidelity"]:
                msg = "Methods do not share the same starting fidelity."
                raise ValueError(msg)
            paired.append({
                "target_id": row["target_id"],
                "family": row["family"],
                "replica": row["replica"],
                "delta_pp": 100 * (cast("float", row["final_fidelity"]) - cast("float", other["final_fidelity"])),
                "hybrid_over_comparator_cpu": cast("float", row["cpu_seconds"]) / cast("float", other["cpu_seconds"]),
            })
        deltas = np.array([row["delta_pp"] for row in paired])
        family_deltas = {
            family: float(np.mean([row["delta_pp"] for row in paired if row["family"] == family]))
            for family in experiment.FAMILIES
        }
        ratio = cast("float", methods["trajectory_hybrid"]["mean_cpu_seconds"]) / cast(
            "float", methods[kind]["mean_cpu_seconds"]
        )
        comparisons[kind] = {
            "mean_delta_pp": float(np.mean(deltas)),
            "median_delta_pp": float(np.median(deltas)),
            "min_delta_pp": float(np.min(deltas)),
            "max_delta_pp": float(np.max(deltas)),
            "winning_pairs": int(np.sum(deltas > 1e-10)),
            "family_delta_pp": family_deltas,
            "hybrid_over_comparator_cpu": ratio,
            "fidelity_screen": bool(np.mean(deltas) >= 0.5 and ratio <= 1.2 and min(family_deltas.values()) > 0),
            "speed_screen": bool(np.mean(deltas) >= -0.2 and ratio <= 1 / 1.5 and min(family_deltas.values()) >= -0.2),
            "paired": paired,
        }
    strongest = max(experiment.KINDS, key=lambda kind: cast("float", methods[kind]["fidelity"]))
    reference = max(
        (kind for kind in experiment.KINDS if kind != "trajectory_hybrid"),
        key=lambda kind: cast("float", methods[kind]["fidelity"]),
    )
    return {
        "methods": methods,
        "hybrid_comparisons": comparisons,
        "highest_mean_fidelity_method": strongest,
        "strongest_reference": reference,
        "speed_screen_against_strongest_reference": comparisons[reference]["speed_screen"],
    }


def audit(root: Path) -> dict[str, object]:
    """Recheck every stored diagnostic with the independently implemented density adjoint."""
    manifest = pilot.read_record(root / "manifest.json")
    selection = pilot.read_record(root / "selection.json")
    result = pilot.read_record(root / "result.json")
    if (
        result["status"] != "complete"
        or result["manifest_checksum"] != pilot.digest(manifest)
        or result["selection_checksum"] != pilot.digest(selection)
        or manifest["runtime_fingerprint"] != experiment.runtime_fingerprint()
        or experiment.nominate(cast("list[dict[str, object]]", result["development"])) != selection
    ):
        msg = "Experiment provenance or nomination audit failed."
        raise ValueError(msg)
    targets = {row["target_id"]: row for row in cast("list[dict]", manifest["targets"])}
    confirmation = cast("list[dict[str, object]]", result["confirmation"])
    summary = paired_summary(confirmation)
    max_error = 0.0
    checkpoints = []
    for row in confirmation:
        state = pilot.read_record(root / str(row["path"]))
        start = pilot.read_record(root / "starts" / str(row["target_id"]) / f"seed_{row['replica']}" / "start.json")
        target = targets[row["target_id"]]
        trace = cast("list[dict]", state["trace"])
        if (
            pilot.digest(state) != row["checksum"]
            or state["status"] != "complete"
            or state["completed_updates"] != 50
            or len(trace) != 50
            or state["start_checksum"] != pilot.digest(start)
            or state["target_checksum"] != pilot.digest(target)
            or state["config"] != manifest["config"]
            or state["method"] != cast("dict", selection["selected"])[cast("dict", row["method"])["kind"]]
            or state["seed"] != experiment.seed_for("confirmation", str(row["target_id"]), cast("int", row["replica"]))
        ):
            msg = "Endpoint checkpoint audit failed."
            raise ValueError(msg)
        method = experiment.Method(**cast("dict", row["method"]))
        for step, entry in enumerate(trace, 1):
            if (entry["phase"], entry["rate"]) != method.phase_rate(step, experiment.Config()):
                msg = "Phase/rate trace audit failed."
                raise ValueError(msg)
        circuit = pilot.restore_circuit(cast("dict", start["circuit"]))
        vector = experiment.target_vector(target)
        evaluations = cast("list[dict]", state["evaluations"])
        if [e["update"] for e in evaluations] != [0, 10, 25, 50] or evaluations[-1]["theta"] != state["theta"]:
            msg = "Missing diagnostic checkpoint or endpoint mismatch."
            raise ValueError(msg)
        if (
            row["final_fidelity"] != evaluations[-1]["exact_noisy_fidelity"]
            or row["frozen_fidelity"] != evaluations[0]["exact_noisy_fidelity"]
            or row["final_noiseless_fidelity"] != evaluations[-1]["noiseless_fidelity"]
            or row["cpu_seconds"] != sum(entry["cpu_seconds"] for entry in trace)
            or row["wall_seconds"] != sum(entry["wall_seconds"] for entry in trace)
        ):
            msg = "Summary row disagrees with its checkpoint."
            raise ValueError(msg)
        for evaluation in evaluations:
            theta = np.asarray(evaluation["theta"], dtype=float)
            for name, calibration in (
                ("noiseless_fidelity", CalibratedLocalDepolarizingProvider(0, 0)),
                ("exact_noisy_fidelity", CalibratedLocalDepolarizingProvider()),
            ):
                _, _, fidelity = density_loss_gradient(circuit, theta, vector, calibration)
                error = abs(fidelity - evaluation[name])
                max_error = max(max_error, error)
                if error > 1e-10:
                    msg = "Independent fidelity audit failed."
                    raise FloatingPointError(msg)
            checkpoints.append({
                "target_id": row["target_id"],
                "replica": row["replica"],
                "method": method.kind,
                "update": evaluation["update"],
                "fidelity": evaluation["exact_noisy_fidelity"],
                "cpu_seconds": sum(entry["cpu_seconds"] for entry in trace[: evaluation["update"]]),
            })
    return {
        "status": "passed",
        "analysis_source_checksum": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "result_checksum": pilot.digest(result),
        "confirmation_runs": len(confirmation),
        "independent_fidelity_checks": len(checkpoints) * 2,
        "max_fidelity_error": max_error,
        "summary": summary,
        "checkpoints": checkpoints,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write an audited, reproducible summary without touching training artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    with single_threaded_numerics():
        result = audit(arguments.output_root)
    pilot.write_record(arguments.output_root / "audit.json", result)
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
