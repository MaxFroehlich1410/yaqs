# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Verify shared exact deletion adjoints, then reinvest saved CPU in noisy refinement."""

from __future__ import annotations

import argparse
import json
import operator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from benchmarks.state_preparation.exact_density import density_deletion_fidelities
from benchmarks.state_preparation.execution import WorkerSetting, completed_jobs, parse_workers, resolve_workers
from benchmarks.state_preparation.pilot_components import ExactLocalPauliProvider
from experiments import state_preparation_noisy_restructuring as base
from experiments import state_preparation_pilot as pilot
from experiments import state_preparation_sampling_budget as budget
from experiments import state_preparation_trajectory_comparison as comparison
from experiments import state_preparation_trajectory_diagnostics as development

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

VERSION = "yaqs.shared_adjoint_restructuring.v1"
TOLERANCE = 1e-10


def rank_deletions(state: dict[str, Any], target: NDArray[np.complex128]) -> list[dict[str, Any]]:
    """Replace exhaustive propagation without changing eligibility or ranking tie rules."""
    circuit = pilot.restore_circuit(state["circuit"])
    values, _ = density_deletion_fidelities(
        circuit, np.asarray(state["theta"]), target, ExactLocalPauliProvider().calibration
    )
    rows = []
    for index, gate in enumerate(circuit.gates):
        if type(gate.logical_gate_id) is not int:
            msg = "Expected stable integer gate origins."
            raise ValueError(msg)
        if gate.logical_gate_id >= 3 * circuit.num_qubits:
            rows.append({
                "index": index,
                "origin": gate.logical_gate_id,
                "arity": len(gate.sites),
                "fidelity": float(values[index]),
            })
    return sorted(rows, key=lambda r: (-round(r["fidelity"] / 1e-12), r["origin"]))


def compare_scores(reference: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
    """Require all scores and the full ordered gate ranking to reproduce history."""
    identity = operator.itemgetter("index", "origin", "arity")
    if [identity(r) for r in reference] != [identity(r) for r in actual]:
        msg = "Shared deletion adjoint changed the gate ranking."
        raise ValueError(msg)
    errors = [abs(a["fidelity"] - b["fidelity"]) for a, b in zip(reference, actual, strict=True)]
    if not all(np.isfinite(v) and v <= TOLERANCE for v in errors):
        msg = "Shared deletion adjoint disagrees with exhaustive scores."
        raise ValueError(msg)
    return max(errors, default=0.0)


def load_history(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], base.Config]:
    """Authenticate old evidence read-only, allowing the explicitly changed runtime code."""
    manifest = cast("dict[str, Any]", pilot.read_record(root / "manifest.json"))
    result = cast("dict[str, Any]", pilot.read_record(root / "result.json"))
    if (
        manifest["version"] != base.VERSION
        or result["status"] != "complete"
        or result["manifest_checksum"] != pilot.digest(manifest)
        or manifest["noise"] != ExactLocalPauliProvider().scientific_dict()
    ):
        msg = "Expected completed compatible exhaustive restructuring evidence."
        raise ValueError(msg)
    provenance, cases = base.load_cases(Path(manifest["source_root"]))
    # Normalize tuple-valued gate metadata exactly as the historical JSON writer.
    provenance, cases = json.loads(json.dumps([provenance, cases]))
    if provenance != manifest["source"] or cases != manifest["cases"]:
        msg = "Historical construction provenance changed."
        raise ValueError(msg)
    expected = {(c["target"]["target_id"], c["replica"]) for c in cases}
    if len(result["jobs"]) != len(cases) or {(r["target_id"], r["replica"]) for r in result["jobs"]} != expected:
        msg = "Historical case panel is incomplete or duplicated."
        raise ValueError(msg)
    paired = []
    for case in cases:
        name, replica = case["target"]["target_id"], case["replica"]
        row = next(r for r in result["jobs"] if (r["target_id"], r["replica"]) == (name, replica))
        if row["path"] != f"{name}/seed_{replica}.json":
            msg = "Unexpected historical case path."
            raise ValueError(msg)
        record = cast("dict[str, Any]", pilot.read_record(root / row["path"]))
        if (
            pilot.digest(record) != row["checksum"]
            or record["signature"] != pilot.digest([pilot.digest(manifest), case, manifest["config"]])
            or record["status"] != "complete"
        ):
            msg = "Historical case authentication failed."
            raise ValueError(msg)
        seconds = max(base.cpu_cost(arm["ledger"]) for arm in record["structural"].values())
        if not np.isfinite(seconds) or seconds <= 0:
            msg = "Invalid historical CPU budget."
            raise ValueError(msg)
        paired.append({"case": case, "old_job": row, "budget_seconds": seconds})
    return (
        {
            "manifest_checksum": pilot.digest(manifest),
            "result_checksum": pilot.digest(result),
            "source": provenance,
        },
        paired,
        base.Config(**manifest["config"]),
    )


@dataclass(frozen=True)
class Job:
    """One case's verification or four-arm continuation, with private atomic output."""

    root: Path
    history_root: Path
    paired: dict[str, Any]
    config: base.Config
    manifest_checksum: str


def historical_case(job: Job) -> dict[str, Any]:
    """Recheck a source case at worker access, without mutating its artifacts."""
    row = job.paired["old_job"]
    record = cast("dict[str, Any]", pilot.read_record(job.history_root / row["path"]))
    if pilot.digest(record) != row["checksum"]:
        msg = "Historical case changed after manifest creation."
        raise ValueError(msg)
    return record


def signature(job: Job, stage: str) -> str:
    """Bind stage-specific artifacts to scientific source, budget and configuration."""
    return pilot.digest([VERSION, stage, job.manifest_checksum, job.paired, asdict(job.config)])


def artifact_row(job: Job, stage: str, record: dict[str, Any]) -> dict[str, Any]:
    """Describe a private case file for parent-owned aggregation."""
    old = job.paired["old_job"]
    return {**old, "path": f"{stage}/{old['path']}", "checksum": pilot.digest(record)}


def verify_job(job: Job) -> dict[str, Any]:
    """Check all archived scores and freshly benchmark both scorers on first-round states."""
    path = job.root / "verification" / job.paired["old_job"]["path"]
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature(job, "verification") or record["status"] != "passed":
            msg = "Verification checkpoint mismatch."
            raise ValueError(msg)
        return artifact_row(job, "verification", record)
    previous = historical_case(job)
    target = comparison.target_vector(job.paired["case"]["target"])
    rows = []
    for method in base.METHODS:
        for old_round in previous["structural"][method]["rounds"]:
            state = old_round["candidates"][0]["state"]
            # First-round timings pair fresh exhaustive and shared calculations;
            # method order alternates which implementation executes first.
            exhaustive, exhaustive_cost = None, None
            if old_round["round"] == 1 and method == "adam":
                exhaustive, exhaustive_cost = development.measured(partial(base.rank_deletions, state, target))
            actual, cost = development.measured(partial(rank_deletions, state, target))
            if old_round["round"] == 1 and method != "adam":
                exhaustive, exhaustive_cost = development.measured(partial(base.rank_deletions, state, target))
            error = compare_scores(old_round["ranking"], actual)
            if exhaustive is not None:
                error = max(error, compare_scores(exhaustive, actual))
            rows.append({
                "method": method,
                "round": old_round["round"],
                "scores": actual,
                "maximum_error": error,
                "shared_cost": cost,
                "fresh_exhaustive_cost": exhaustive_cost,
            })
    record: dict[str, Any] = {"status": "passed", "signature": signature(job, "verification"), "rows": rows}
    pilot.write_record(path, record)
    return artifact_row(job, "verification", record)


def verify_search(previous: dict[str, Any], actual: dict[str, Any]) -> None:
    """Reject any unintended change to selected gates, repaired parameters or moments."""
    if pilot.digest(previous["state"]) != pilot.digest(actual["state"]) or len(previous["rounds"]) != len(
        actual["rounds"]
    ):
        msg = "Fast search changed the selected circuit or optimizer state."
        raise ValueError(msg)
    for old, new in zip(previous["rounds"], actual["rounds"], strict=True):
        compare_scores(old["ranking"], new["ranking"])
        if old["selected"] != new["selected"] or pilot.digest([v["state"] for v in old["candidates"]]) != pilot.digest([
            v["state"] for v in new["candidates"]
        ]):
            msg = "Fast search changed a repair branch or acceptance decision."
            raise ValueError(msg)


def run_job(job: Job) -> dict[str, Any]:
    """Rerun identical fast search, then refine and rerun controls at frozen case budgets."""
    path = job.root / "optimization" / job.paired["old_job"]["path"]
    if path.exists():
        record = cast("dict[str, Any]", pilot.read_record(path))
        if record["signature"] != signature(job, "optimization") or record["status"] != "complete":
            msg = "Optimization checkpoint mismatch."
            raise ValueError(msg)
        return artifact_row(job, "optimization", record)
    previous = historical_case(job)
    case, config = job.paired["case"], job.config
    target = comparison.target_vector(case["target"])
    start, seed = base.initial_state(case), previous["seed"]
    seconds = job.paired["budget_seconds"]
    order = base.METHODS if case["replica"] % 4 == 3 else base.METHODS[::-1]
    prefixes, searches = {}, {}
    for method in order:
        ledger: list[dict[str, Any]] = []
        initial_score = base.charged(partial(base.fidelity, start, target), ledger, "initial_score")
        if abs(initial_score - case["expected_fidelity"]) > TOLERANCE:
            msg = "Original construction fidelity did not reproduce."
            raise ValueError(msg)
        curve = [base.point(start, initial_score, ledger)]
        warm = base.charged(
            partial(base.train_block, start, target, method, config, seed, 0, config.warmup), ledger, "warmup"
        )
        if warm["state"] != previous["prefixes"][method]["state"]:
            msg = "Original warmup optimizer state did not reproduce."
            raise ValueError(msg)
        curve.append(base.point(warm["state"], warm["fidelity"], ledger))
        prefixes[method] = {**warm, "ledger": ledger, "curve": curve}
        search = base.restructure(prefixes[method], target, method, config, seed, ranker=rank_deletions)
        verify_search(previous["structural"][method], search)
        if base.cpu_cost(search["ledger"]) >= seconds:
            msg = "Fast search exhausted the frozen refinement budget."
            raise ValueError(msg)
        searches[method] = search
    refined, fixed = {}, {}
    for method in order:
        refined[method] = base.fixed_control(
            searches[method],
            target,
            method,
            config,
            seed,
            seconds,
            epoch_start=config.warmup + config.rounds * config.repair,
            role="post_prune_refinement",
        )
        fixed[method] = base.fixed_control(prefixes[method], target, method, config, seed, seconds)
    record = {
        "status": "complete",
        "signature": signature(job, "optimization"),
        "target_id": case["target"]["target_id"],
        "replica": case["replica"],
        "family": case["target"]["family"],
        "budget_seconds": seconds,
        "prefixes": prefixes,
        "searches": searches,
        "refined": refined,
        "fixed": fixed,
    }
    pilot.write_record(path, record)
    return artifact_row(job, "optimization", record)


def summarize(root: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only checkpoints available at the identical four-arm case cutoff."""
    rows = []
    for job in jobs:
        record = cast("dict[str, Any]", pilot.read_record(root / job["path"]))
        if pilot.digest(record) != job["checksum"]:
            msg = "Optimization case changed during aggregation."
            raise ValueError(msg)
        for method in base.METHODS:
            arm = record["refined"][method]
            chosen = base.at_budget(arm, record["budget_seconds"])
            rows.append({
                "target_id": record["target_id"],
                "replica": record["replica"],
                "family": record["family"],
                "method": method,
                "budget_seconds": record["budget_seconds"],
                "fidelity": chosen["fidelity"],
                "selected_update": chosen["state"]["updates"],
                "completed_updates": chosen["completed_updates"],
                "gates": len(chosen["state"]["circuit"]["gates"]),
                "two_qubit_gates": sum(len(g["sites"]) == 2 for g in chosen["state"]["circuit"]["gates"]),
                "after_search_fidelity": record["searches"][method]["fidelity"],
                "search_cpu_seconds": base.cpu_cost(record["searches"][method]["ledger"]),
                "covered_budget": arm["covered_budget"],
                **{
                    f"fixed_{control}_fidelity": base.at_budget(record["fixed"][control], record["budget_seconds"])[
                        "fidelity"
                    ]
                    for control in base.METHODS
                },
                "controls_covered_budget": all(a["covered_budget"] for a in record["fixed"].values()),
            })
    return rows


def run(
    root: Path,
    history_root: Path,
    *,
    workers: WorkerSetting = "auto",
    resume: bool = False,
    stage: str = "all",
) -> dict[str, Any]:
    """Enforce an all-case score-verification barrier before any new optimization."""
    if stage not in {"all", "verify"}:
        msg = "Expected all or verify stage."
        raise ValueError(msg)
    resolve_workers(workers, 1)
    provenance, paired, config = load_history(history_root)
    manifest = json.loads(
        json.dumps({
            "version": VERSION,
            "config": asdict(config),
            "history_root": str(history_root.resolve()),
            "history": provenance,
            "cases": paired,
            "runtime": budget.runtime(),
            "driver": pilot.digest(Path(__file__).read_text(encoding="utf-8")),
            "base_driver": pilot.digest(Path(base.__file__).read_text(encoding="utf-8")),
            "protocol": pilot.digest(Path(__file__).with_suffix(".md").read_text(encoding="utf-8")),
            "score_tolerance": TOLERANCE,
        })
    )
    if root.exists():
        if not resume or not (root / "manifest.json").exists() or pilot.read_record(root / "manifest.json") != manifest:
            msg = "Existing output requires identical-manifest explicit resume."
            raise ValueError(msg)
    else:
        if resume:
            msg = "Cannot resume missing output."
            raise ValueError(msg)
        pilot.write_record(root / "manifest.json", manifest)
    checksum = pilot.digest(manifest)
    jobs = [Job(root, history_root, case, config, checksum) for case in paired]
    order = np.random.default_rng(config.root_seed).permutation(len(jobs))
    verification_rows = {}
    for index, row in completed_jobs(verify_job, [jobs[i] for i in order], workers=workers):
        verification_rows[int(order[index])] = row
        print(f"adjoint verification {len(verification_rows)}/{len(jobs)}", flush=True)
    verification: dict[str, Any] = {
        "status": "passed",
        "manifest_checksum": checksum,
        "jobs": [verification_rows[i] for i in range(len(jobs))],
    }
    save_aggregate(root / "verification.json", verification)
    if stage == "verify":
        return verification
    completed = {}
    for index, row in completed_jobs(run_job, [jobs[i] for i in order], workers=workers):
        completed[int(order[index])] = row
        print(f"adjoint refinement {len(completed)}/{len(jobs)}: {row['target_id']} seed {row['replica']}", flush=True)
    result: dict[str, Any] = {
        "status": "complete",
        "manifest_checksum": checksum,
        "verification_checksum": pilot.digest(verification),
        "jobs": [completed[i] for i in range(len(jobs))],
    }
    result["rows"] = summarize(root, result["jobs"])
    save_aggregate(root / "result.json", result)
    return result


def save_aggregate(path: Path, record: dict[str, Any]) -> None:
    """Preserve completed aggregate bytes and reject inconsistent replay."""
    if path.exists():
        if pilot.read_record(path) != record:
            msg = "Completed aggregate differs from authenticated cases."
            raise ValueError(msg)
    else:
        pilot.write_record(path, record)


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the fixed historical panel with optional verification-only execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "verify"), default="all")
    args = parser.parse_args(argv)
    run(args.output_root, args.history_root, workers=args.workers, resume=args.resume, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
