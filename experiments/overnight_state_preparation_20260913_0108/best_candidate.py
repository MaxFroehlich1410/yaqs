# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Freeze and reproduce the complete ten-case confirmation without dropping controls."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import math
import os
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.state_preparation.targets import load_target
from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import (
    common_control_audit,
    construction,
    lazy_initializer_experiment,
)
from experiments.overnight_state_preparation_20260913_0108 import composed_experiment as base
from experiments.overnight_state_preparation_20260913_0108 import (
    confirmation_construction as constructor,
)
from experiments.overnight_state_preparation_20260913_0108 import (
    lazy_initializer as lazy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

VERSION = "overnight.complete_best_candidate_confirmation.v1"
RECIPES = ("original", "state_preparation_compiled")
TARGETS, REPLICAS = constructor.TARGETS, (11, 29)
METHODS, TRAINED, PRIMARY = base.METHODS, base.TRAINED, base.PRIMARY
CANDIDATE, BASELINE = "structural_mean2_lbfgs", "structural_mean2_weighted_adam"
RESERVE_BYTES = 512 * 1024**2
PER_CASE_BYTES = 130 * 1024**2
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")


def thread_limits() -> dict[str, str | None]:
    """Read all four explicit numerical thread limits for the scientific parent."""
    return {name: os.environ.get(name) for name in THREAD_VARIABLES}


def require_single_threads() -> None:
    """Refuse implicit or oversized native thread pools before scientific preparation or execution."""
    if any(value != "1" for value in thread_limits().values()):
        msg = "Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS and MKL_NUM_THREADS to1."
        raise ValueError(msg)


def peak_rss(raw_value: float | None = None, platform_name: str | None = None) -> dict[str, Any]:
    """Return worker-lifetime peak RSS with explicit platform-dependent raw units."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_value is None else raw_value
    name = sys.platform if platform_name is None else platform_name
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (float, int))
        or not math.isfinite(raw)
        or raw < 0
        or name not in {"darwin", "linux"}
    ):
        msg = "Peak RSS needs a finite nonnegative measurement on macOS or Linux."
        raise ValueError(msg)
    return {
        "raw_value": raw,
        "raw_unit": "bytes" if name == "darwin" else "KiB",
        "bytes": int(raw * (1 if name == "darwin" else 1024)),
        "platform": name,
        "scope": (
            "worker lifetime high-water at scientific body completion; may include prior cases and evidence "
            "authentication; not incremental algorithm memory"
        ),
    }


def read(path: Path) -> dict[str, Any]:
    """Read a checksummed heterogeneous scientific record."""
    return lazy.read(path)


def file_hash(path: Path) -> str:
    """Hash bytes without changing completed artifacts or their physical storage."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel(recipe: str) -> ModuleType:
    """Select only one of the two fully specified development recipes."""
    if recipe not in RECIPES:
        msg = "Only the original and state-preparation-compiled developed recipes are supported."
        raise ValueError(msg)
    return (
        base
        if recipe == "original"
        else importlib.import_module("experiments.overnight_state_preparation_20260913_0108.compiled_composition")
    )


def sources(recipe: str) -> dict[str, str]:
    """Bind all numerical, orchestration and acceptance code in the selected recipe."""
    result = base.sources()
    result.update(lazy.sources())
    result.update(common_control_audit.sources())
    if recipe == "state_preparation_compiled":
        module = importlib.import_module(
            "experiments.overnight_state_preparation_20260913_0108.compiled_composition_experiment"
        )
        result.update(module.sources())
    for path in (Path(__file__), Path(__file__).with_name("BEST_CANDIDATE_PROTOCOL.md")):
        result[str(path.relative_to(base.runner.REPO))] = file_hash(path)
    return result


def acceptance() -> dict[str, Any]:
    """Return the fixed primary and strongest-overall-control confirmation rule."""
    return {
        "candidate": CANDIDATE,
        "primary_baseline": BASELINE,
        "primary_metric": "target-balanced mean of eleven noisy fidelities at four-structural common CPU",
        "rule": {
            "route": "fidelity",
            "criteria": [
                "both replicas of every one of the five fixed targets and all seven roles complete",
                "candidate minus fixed weighted baseline aggregate >=0.005 and every target mean >1e-10",
                "candidate minus strongest aggregate fair control >=0.005 and every target mean >1e-10",
                "all twelve finite exact endpoint scores, full cost custody and complete artifacts required",
            ],
            "complete_panel_required": True,
            "failed_case_policy": "retain_and_fail",
            "minimum_mean_advantage": 0.005,
            "strict_target_advantage": 1e-10,
            "strongest_control": "maximum aggregate among every noncandidate role at the same primary cutoff",
        },
    }


def algorithm(recipe: str, exact: base.exact.Config, stochastic: base.stochastic.Config) -> dict[str, Any]:
    """Freeze the entire recipe without family-dependent budgets or optimizer options."""
    selected = kernel(recipe)
    return {
        "initializer": lazy.specification(),
        "compiler": {
            "generic": lazy.compilation.VERSION,
            "state_preparation": None if recipe == "original" else selected.compiler.VERSION,
            "opportunity": "same initial and final physical compiler opportunity for every role",
        },
        "training": {
            "recipe": recipe,
            "kernel_version": selected.VERSION,
            "exact": asdict(exact),
            "stochastic": asdict(stochastic),
            "training_profiles": ["depolarizing_1s_all", "dephasing_1s_all"],
            "final_profiles": list(base.channels.NOISE_IDS),
            "seed_namespace": "confirmation_training",
        },
        "structure": {
            "ranking": "shared exact finite deletion rank; shortlist2; equally repaired undeleted branch",
            "memory": "fresh curvature on changed structures; historical preserved moments for stochastic arms",
            "protected_initial_gates": 0,
            "policy": "frozen strict improvement and deterministic tie policy; final iterate",
        },
        "endpoint": {
            "selection": "last chronological deployable checkpoint; never best or interpolated",
            "primary_cutoff_roles": list(PRIMARY),
            "scored_at_primary": list(METHODS),
            "sensitivity_cutoff_roles": list(TRAINED),
            "shorter_completed_control": "final point remains deployable at any later cutoff",
            "panels": ["primary", "all_trained_sensitivity", "full_cap"],
        },
    }


def budget_spec(exact: base.exact.Config, stochastic: base.stochastic.Config) -> dict[str, Any]:
    """State charged work caps for every role, including unchanged and noiseless controls."""
    return {
        name: {"updates": 0}
        if name == "unchanged"
        else {"updates": stochastic.total_updates, "paths_per_channel": stochastic.paths_per_channel}
        if name in {"structural_mean2_weighted_adam", "structural_mean2_adaptive_online"}
        else {
            "maximum_joint_oracle_opportunities": exact.channel_evaluations // 2,
            "individual_channel_cap": exact.channel_evaluations // (2 if name == "fixed_noiseless_adam" else 1),
        }
        for name in METHODS
    }


def inventory(root: Path, *, excluded: Sequence[Path] = ()) -> dict[str, str]:
    """Bind the exact complete file set without copying raw histories into outer records."""
    skip = set(excluded)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file() and p not in skip}


def disk_plan(root: Path, cases: int) -> dict[str, Any]:
    """Forecast complete case growth plus reserve before any input loading."""
    parent = root
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    needed = cases * PER_CASE_BYTES + RESERVE_BYTES
    return {
        "cases": cases,
        "per_case_bound_bytes": PER_CASE_BYTES,
        "reserve_bytes": RESERVE_BYTES,
        "required_bytes": needed,
        "available_bytes_at_freeze": free,
        "fits": free >= needed,
        "basis": "104MiB largest complete development child tree excluding outer duplicate, multiplied by1.25",
    }


def guard_disk(root: Path, cases: int = 1) -> None:
    """Stop at a durable phase/case boundary when a complete admitted case cannot fit."""
    if not disk_plan(root, cases)["fits"]:
        msg = "Insufficient storage for the complete remaining admission and512MiB reserve."
        raise RuntimeError(msg)


def phase(root: Path, identity: object, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Resume a measured phase with exact file-set and complete result-field authentication."""
    signature = pilot.digest(identity)
    manifest, ledger_path, result_path, seal_path = [
        root / name for name in ("PHASE.json", "ATTEMPTS.json", "RESULT.json", "COMPLETED.json")
    ]
    ignored = (manifest, ledger_path, result_path, seal_path)
    if manifest.exists() and read(manifest) != {"signature": signature}:
        msg = "Confirmation phase identity changed."
        raise ValueError(msg)
    if not manifest.exists():
        if root.exists() and any(root.iterdir()):
            msg = "Cannot adopt confirmation work without its frozen phase identity."
            raise ValueError(msg)
        pilot.write_record(manifest, {"signature": signature})
    ledger: dict[str, Any] = read(ledger_path) if ledger_path.exists() else {"signature": signature, "attempts": []}
    if ledger["signature"] != signature or any(
        a["status"] not in {"complete", "interrupted", "failed"}
        or any(
            a.get(k) is None
            or isinstance(a[k], bool)
            or not isinstance(a[k], (float, int))
            or not math.isfinite(a[k])
            or a[k] < 0
            for k in ("cpu_seconds", "wall_seconds")
        )
        for a in ledger["attempts"]
    ):
        msg = "Unknown confirmation phase cost or changed identity blocks resume."
        raise ValueError(msg)
    current = inventory(root, excluded=ignored)
    if not ledger["attempts"] and (result_path.exists() or seal_path.exists()):
        msg = "Cannot adopt a completed confirmation result without measured attempts."
        raise ValueError(msg)
    if (ledger["attempts"] and ledger["attempts"][-1]["files"] != current) or (not ledger["attempts"] and current):
        msg = "Confirmation phase artifacts differ from their measured attempt registry."
        raise ValueError(msg)
    if result_path.exists():
        result, sealed = read(result_path), read(seal_path)
        expected = {
            "signature": signature,
            "result_checksum": pilot.digest(result),
            "attempts_checksum": pilot.digest(ledger),
        }
        if (
            sealed != expected
            or result["signature"] != signature
            or result["attempts_checksum"] != pilot.digest(ledger)
            or any(
                result[f"{kind}_seconds"] != sum(a[f"{kind}_seconds"] for a in ledger["attempts"])
                for kind in ("cpu", "wall")
            )
        ):
            msg = "Confirmation phase result fields, scores or complete costs changed."
            raise ValueError(msg)
        return result
    attempt: dict[str, Any] = {"status": "running", "cpu_seconds": None, "wall_seconds": None}
    ledger["attempts"].append(attempt)
    pilot.write_record(ledger_path, ledger)
    cpu, wall = time.process_time(), time.perf_counter()
    error: BaseException | None = None
    value: dict[str, Any] = {}
    try:
        value = function()
    except BaseException as caught:
        error = caught
    finally:
        try:
            attempt["files"] = inventory(root, excluded=ignored)
        except OSError as caught:
            attempt["files"] = None
            error = caught if error is None else error
        attempt.update(
            status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.perf_counter() - wall,
        )
        if error is not None:
            attempt["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_record(ledger_path, ledger)
    if error is not None:
        raise error
    result = {
        "signature": signature,
        "value": value,
        "attempts_checksum": pilot.digest(ledger),
        "cpu_seconds": sum(a["cpu_seconds"] for a in ledger["attempts"]),
        "wall_seconds": sum(a["wall_seconds"] for a in ledger["attempts"]),
    }
    pilot.write_record(
        seal_path,
        {"signature": signature, "result_checksum": pilot.digest(result), "attempts_checksum": pilot.digest(ledger)},
    )
    pilot.write_record(result_path, result)
    return read(result_path)


def read_phase(root: Path, identity: object) -> dict[str, Any]:
    """Authenticate complete stored phase fields and files without permitting any new writes or work."""
    if not all((root / name).is_file() for name in ("PHASE.json", "ATTEMPTS.json", "RESULT.json", "COMPLETED.json")):
        msg = "Read-only phase authentication requires all completed receipts."
        raise ValueError(msg)

    def forbidden() -> dict[str, Any]:
        msg = "Read-only phase authentication cannot execute scientific work."
        raise ValueError(msg)

    return phase(root, identity, forbidden)


def lazy_evidence(root: Path) -> dict[str, str]:
    """Require the entire exact four-case lazy reproduction before scientific design freeze."""
    protocol, summary = read(root / "PROTOCOL.json"), read(root / "SUMMARY.json")
    lazy_initializer_experiment.verify(root, protocol)
    if not summary["accepted"] or summary["protocol_checksum"] != pilot.digest(protocol) or len(summary["cases"]) != 4:
        msg = "All four lazy development reproductions must pass before confirmation."
        raise ValueError(msg)
    files = {str((root / name).resolve()): file_hash(root / name) for name in ("PROTOCOL.json", "SUMMARY.json")}
    for case in protocol["cases"]:
        child = root / f"{case['target_id']}__11"
        result = lazy_initializer_experiment.cached_result(child, pilot.digest([protocol, case]))
        row = next(r for r in summary["cases"] if r["target_id"] == case["target_id"])
        if not result["accepted"] or row["checksum"] != pilot.digest(result):
            msg = "The lazy reproduction summary differs from its complete source-bound cases."
            raise ValueError(msg)
        files.update({str((child / p).resolve()): h for p, h in inventory(child).items()})
    return files


def cached_development_target(initializer_root: Path, target_id: str) -> tuple[NDArray[np.complex128], dict[str, str]]:
    """Read only an already-authenticated development constructor's saved vector, never the fixture loader."""
    original = read(initializer_root / "PROTOCOL.json")
    matches = [s for s in original["starts"] if s["target_id"] == target_id and s["replica"] == 11]
    if target_id not in base.TARGETS or len(matches) != 1:
        msg = "A cached development vector requires its unique authenticated initializer source."
        raise ValueError(msg)
    files = {str((initializer_root / "PROTOCOL.json").resolve()): file_hash(initializer_root / "PROTOCOL.json")}
    vectors = []
    for name, expected in matches[0]["files"].items():
        path = Path(name) if Path(name).is_absolute() else base.runner.REPO / name
        record = read(path)
        if pilot.digest(record) != expected:
            msg = "A cached development constructor receipt changed."
            raise ValueError(msg)
        files[str(path.resolve())] = file_hash(path)
        if "target" in record and "target_vector_sha256" in record:
            target = np.asarray(list(starmap(complex, record["target"])), dtype=np.complex128)
            if (
                record["target_id"] != target_id
                or record["replica"] != 11
                or target.shape != (64,)
                or not np.all(np.isfinite(target))
                or abs(np.linalg.norm(target) - 1) > 1e-12
                or hashlib.sha256(target.tobytes()).hexdigest() != record["target_vector_sha256"]
            ):
                msg = "Cached development vector identity, norm or byte hash changed."
                raise ValueError(msg)
            vectors.append(target)
    if len(vectors) != 1:
        msg = "Exactly one existing complete development vector receipt is required."
        raise ValueError(msg)
    return vectors[0], files


def cached_score(
    root: Path, target: NDArray[np.complex128], config: base.exact.Config, point: dict[str, Any]
) -> dict[str, Any]:
    """Require every physical score receipt before invoking the frozen cache validator."""
    name = "audit_" + pilot.digest({"circuit_hash": point["circuit_hash"], "theta": point["theta"]})
    paths = (
        root / "circuits" / f"{point['circuit_hash']}.json",
        root / "operations" / f"{name}.json",
        root / "operations" / f"{name}.attempts.json",
    )
    if not all(path.is_file() for path in paths):
        msg = "An existing development physical score receipt is missing; fresh evaluation is forbidden."
        raise ValueError(msg)
    result = base.score(base.CompleteStore(root, target, config), point)
    ledger = read(paths[-1])
    if (
        not ledger["attempts"]
        or any(
            a["status"] not in {"complete", "failed", "interrupted"}
            or any(a.get(k) is None or not math.isfinite(a[k]) or a[k] < 0 for k in ("cpu_seconds", "wall_seconds"))
            for a in ledger["attempts"]
        )
        or any(
            result["evaluation"][f"{kind}_seconds"] != sum(a[f"{kind}_seconds"] for a in ledger["attempts"])
            for kind in ("cpu", "wall")
        )
    ):
        msg = "A cached development score has incomplete physical cost provenance."
        raise ValueError(msg)
    return result


def original_record(root: Path, protocol: dict[str, Any], record: dict[str, Any]) -> dict[str, str]:
    """Reconstruct every original control audit field from saved physical source and score receipts."""
    target_id = record["value"]["target_id"]
    entry = next(c for c in protocol["input_panel"]["cases"] if c["target_id"] == target_id)
    source_root = Path(protocol["input_panel"]["root"])
    source_protocol = read(source_root / "PROTOCOL.json")
    source_case = read(source_root / f"{target_id}__11/RESULT.json")
    target, files = cached_development_target(Path(source_protocol["initializer_root"]), target_id)
    child, old_child = root / f"{target_id}__11", source_root / f"{target_id}__11"
    cutoff = min(source_case["arms"][name]["refinement_cpu_seconds"] for name in PRIMARY)
    points = common_control_audit.select_points(source_case["arms"], TRAINED, cutoff)
    config = base.exact.Config(**source_protocol["exact_config"])
    scored = {
        name: cached_score(old_child if name in PRIMARY else child, target, config, point)
        for name, point in points.items()
    }
    unchanged = cached_score(old_child, target, config, source_case["arms"]["unchanged"]["final"])
    expected = {
        "status": "complete",
        "target_id": target_id,
        "refinement_cpu_seconds": cutoff,
        "total_cpu_seconds": source_case["common_primary"]["total_cpu_seconds"],
        "points": points,
        "source_case_checksum": pilot.digest(source_case),
        "source_unchanged_fidelities": unchanged["fidelities"],
        "scores": scored,
        "dependencies": {str(p.relative_to(child)): file_hash(p) for p in child.rglob("*.json") if p.parent != child},
    }
    ledger = read(child / "RESULT.attempts.json")
    signature = pilot.digest(pilot.digest([protocol, entry]))
    if (
        pilot.digest(record["value"]) != pilot.digest(expected)
        or entry["points"] != points
        or record["signature"] != signature
        or ledger["signature"] != signature
        or record["attempts_checksum"] != pilot.digest(ledger)
        or any(
            a["status"] not in {"complete", "failed", "interrupted"}
            or any(a.get(k) is None or not math.isfinite(a[k]) or a[k] < 0 for k in ("cpu_seconds", "wall_seconds"))
            for a in ledger["attempts"]
        )
        or any(
            record[f"{kind}_seconds"] != sum(a[f"{kind}_seconds"] for a in ledger["attempts"])
            for kind in ("cpu", "wall")
        )
    ):
        msg = "Original development audit scores, selected theta or costs differ from their child receipts."
        raise ValueError(msg)
    for path in (source_root / "PROTOCOL.json", source_root / "SUMMARY.json", old_child / "RESULT.json"):
        files[str(path.resolve())] = file_hash(path)
    return files


def development_evidence(recipe: str, root: Path) -> dict[str, str]:
    """Authenticate complete development endpoints and independently reapply the fixed gate."""
    protocol, summary = read(root / "PROTOCOL.json"), read(root / "SUMMARY.json")
    records = [read(root / f"{t}__11/RESULT.json") for t in base.TARGETS]
    linked_files = {}
    if recipe == "original":
        if (
            protocol["version"] != common_control_audit.VERSION
            or protocol["sources"] != common_control_audit.sources()
            or protocol["input_panel"] != common_control_audit.input_panel(Path(protocol["input_panel"]["root"]))
            or summary["protocol_checksum"] != pilot.digest(protocol)
            or summary["cases"] != [pilot.digest(r) for r in records]
        ):
            msg = "Original development strongest-control custody is incomplete."
            raise ValueError(msg)
        for record in records:
            linked_files.update(original_record(root, protocol, record))
        comparison = common_control_audit.summarize(records, base.TARGETS)
        fixed = comparison["paired_comparisons"][BASELINE]
        passed = (
            comparison["same_rule_passes_against_strongest"]
            and fixed["mean_delta_pp"] >= 0.5
            and fixed["target_wins"] == 4
        )
    else:
        module = importlib.import_module(
            "experiments.overnight_state_preparation_20260913_0108.compiled_composition_experiment"
        )
        module.verify(protocol)
        if summary["case_checksums"] != {r["target_id"]: pilot.digest(r) for r in records}:
            msg = "Compiled development summary does not bind all four case records."
            raise ValueError(msg)
        for record in records:
            child = root / f"{record['target_id']}__11"
            source = next(s for s in protocol["starts"] if s["target_id"] == record["target_id"])
            target, files = cached_development_target(Path(protocol["initializer_root"]), record["target_id"])
            linked_files.update(files)
            if not (child / "EXECUTION.json").is_file():
                msg = "A complete compiled execution is required; fresh numerical work is forbidden."
                raise ValueError(msg)
            offset = list(base.TARGETS).index(record["target_id"]) % len(TRAINED)
            execution = kernel(recipe).execute(
                child,
                source["endpoint"],
                target,
                base.exact.Config(**protocol["exact_config"]),
                base.stochastic.Config(**protocol["stochastic_config"]),
                base.runner.seed_for(protocol["seed_namespace"], record["target_id"], 11),
                order=(*TRAINED[offset:], *TRAINED[:offset]),
            )
            expected_record = {
                **execution,
                "execution_signature": execution["signature"],
                "execution_checksum": pilot.digest(execution),
                "signature": pilot.digest([protocol, source]),
                "protocol_checksum": pilot.digest(protocol),
                "target_id": record["target_id"],
                "replica": 11,
                "version": module.VERSION,
            }
            if pilot.digest(record) != pilot.digest(expected_record):
                msg = "Compiled development case differs from its complete execution."
                raise ValueError(msg)
        passed = module.analyze(records)["promoted"]
    if summary["status"] != "complete" or not passed:
        msg = "The selected recipe has not passed its complete development strongest-control gate."
        raise ValueError(msg)
    return {**linked_files, **{str((root / p).resolve()): h for p, h in inventory(root).items()}}


def configuration(
    recipe: str,
    exact: base.exact.Config,
    stochastic: base.stochastic.Config,
    targets: Sequence[str],
    replicas: Sequence[int],
) -> dict[str, Any]:
    """Return the fully explicit configuration frozen before any confirmation vector access."""
    return {
        "algorithm": algorithm(recipe, exact, stochastic),
        "constructor": constructor.constructor_spec(),
        "arms": {
            name: {
                "roles": ["candidate" if name == CANDIDATE else "primary_baseline" if name == BASELINE else "control"],
                "configuration": {"method": name, "same_initializer": True, "same_compilers": True},
            }
            for name in METHODS
        },
        "budgets": budget_spec(exact, stochastic),
        "acceptance": acceptance(),
        "seed_policy": {
            "root_seed": 202609130108,
            "derivation": (
                "runner.seed_for: sha256 JSON [root,namespace,target,replica,epoch,trajectory], "
                "first8 bytes little endian"
            ),
            "namespace": "confirmation_training",
            "training_pairing": "target_id and replica; no method labels in root sampling seed",
            "constructor_namespace": "legacy pilot initialization seed with original root2026090520",
            "arm_order": "rotate TRAINED by complete case index modulo6",
            "case_seeds": {
                f"{target}__{replica}": base.runner.seed_for("confirmation_training", target, replica)
                for target in targets
                for replica in replicas
            },
        },
    }


def freeze_design(root: Path, recipe: str, development_root: Path, lazy_root: Path) -> dict[str, Any]:
    """Freeze the whole algorithm and evidence without loading a held-out vector."""
    kernel(recipe)
    require_single_threads()
    plan = disk_plan(root, len(TARGETS) * len(REPLICAS))
    if not plan["fits"]:
        msg = "The complete ten-case confirmation and reserve must fit before freezing."
        raise RuntimeError(msg)
    evidence = development_evidence(recipe, development_root)
    evidence.update(lazy_evidence(lazy_root))
    selected = configuration(
        recipe, base.exact.Config(protected_initial_gates=0), base.stochastic.Config(protected=0), TARGETS, REPLICAS
    )
    design: dict[str, Any] = {
        "version": VERSION,
        "status": "frozen",
        "phase": "confirmation",
        "recipe": recipe,
        "targets": list(TARGETS),
        "replicas": list(REPLICAS),
        **selected,
        "configuration_sha256": pilot.digest(selected),
        "source_hashes": sources(recipe),
        "benchmark_hashes": base.runner.custody()["benchmark_hashes"],
        "environment": construction.environment(),
        "thread_limits": thread_limits(),
        "development_evidence": evidence,
        "disk_plan": plan,
    }
    path = root / "DESIGN.json"
    if path.exists():
        previous = read(path)
        design["disk_plan"] = previous["disk_plan"]
        if pilot.digest(previous) != pilot.digest(design):
            msg = "Confirmation choices changed after design freeze; never overwrite an existing design."
            raise ValueError(msg)
        return previous
    if root.exists() and any(root.iterdir()):
        msg = "Confirmation design requires a fresh dedicated root."
        raise ValueError(msg)
    pilot.write_record(path, design)
    return read(path)


def verify_parent(root: Path, parent: dict[str, Any]) -> None:
    """Require the complete parent, setup receipt and every unique child before execution."""
    design = read(root / "DESIGN.json")
    verify_design(root, design)

    bound = read_phase(root / "preparation", [VERSION, pilot.digest(design), "all_case_hash_binding"])
    expected = {
        **design,
        "schema": constructor.PARENT_SCHEMA,
        "cases": bound["value"]["cases"],
        "design_checksum": pilot.digest(design),
        "preparation_checksum": pilot.digest(bound),
        "preparation_cpu_seconds": bound["cpu_seconds"],
        "preparation_wall_seconds": bound["wall_seconds"],
    }
    if pilot.digest(parent) != pilot.digest(expected) or pilot.digest(read(root / "PROTOCOL.json")) != pilot.digest(
        parent
    ):
        msg = "Confirmation parent differs from its frozen design or complete input receipt."
        raise ValueError(msg)
    keys = {(c["target_id"], c["replica"]) for c in parent["cases"]}
    if len(keys) != len(parent["cases"]) or keys != {(t, r) for t in parent["targets"] for r in parent["replicas"]}:
        msg = "Confirmation requires the full distinct target/replica product."
        raise ValueError(msg)
    if parent["phase"] == "confirmation":
        constructor.parent_protocol(root / "PROTOCOL.json", pilot.digest(parent))
    for index, case in enumerate(parent["cases"]):
        expected_case = {
            "parent_checksum": pilot.digest(parent),
            "case": case,
            "index": index,
            "training_seed": parent["seed_policy"]["case_seeds"][f"{case['target_id']}__{case['replica']}"],
        }
        if read(root / f"{case['target_id']}__{case['replica']}" / "CASE.json") != expected_case:
            msg = "Confirmation complete child input registry changed."
            raise ValueError(msg)


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    """Keep physical endpoint statistics and the raw evaluation's identity, never duplicate histories."""
    return {
        **{
            key: score[key]
            for key in ("fidelities", "mean_noisy_fidelity", "physical_counts", "ballarin_counts", "parameters")
        },
        "evaluation_checksum": pilot.digest(score["evaluation"]),
    }


def common_panel(
    store: base.CompleteStore, arms: dict[str, Any], cutoff_roles: Sequence[str], initial_cpu: float
) -> dict[str, Any]:
    """Score all controls at one fixed cutoff using chronological deployable checkpoints."""
    cutoff = min(arms[name]["refinement_cpu_seconds"] for name in cutoff_roles)
    curves = {name: {**arm, "curve": arm.get("curve", [arm["final"]])} for name, arm in arms.items()}
    selected = common_control_audit.select_points(curves, METHODS, cutoff)
    return {
        "status": "complete",
        "cutoff_roles": list(cutoff_roles),
        "refinement_cpu_seconds": cutoff,
        "total_cpu_seconds": initial_cpu + cutoff,
        "arms": {
            name: {"point": point, "scores": compact_score(base.score(store, point))}
            for name, point in selected.items()
        },
    }


def train_case(
    root: Path, parent: dict[str, Any], case: dict[str, Any], source: dict[str, Any], target: NDArray[np.complex128]
) -> dict[str, Any]:
    """Run the frozen seven-role recipe and retain all three complete comparison panels."""
    config = parent["algorithm"]["training"]
    exact, sampled = base.exact.Config(**config["exact"]), base.stochastic.Config(**config["stochastic"])
    index = parent["cases"].index(case)
    offset = index % len(TRAINED)
    seed = parent["seed_policy"]["case_seeds"][f"{case['target_id']}__{case['replica']}"]
    result = kernel(parent["recipe"]).execute(
        root, source, target, exact, sampled, seed, order=(*TRAINED[offset:], *TRAINED[:offset])
    )
    if result["status"] != "complete" or set(result["arms"]) != set(METHODS):
        msg = "A confirmation role failed; retain every artifact and fail the complete-panel rule."
        raise RuntimeError(msg)
    arms = {name: read(root / "arms" / f"{name}.json") for name in TRAINED}
    arms["unchanged"] = (
        read(root / "arms/unchanged.json")
        if parent["recipe"] == "state_preparation_compiled"
        else result["arms"]["unchanged"]
    )
    store = base.CompleteStore(root, target, exact)
    panels = {
        "full_cap": {
            "status": "complete",
            "arms": {
                name: {
                    "status": arm["status"],
                    "point": arm["final"],
                    "scores": compact_score(arm["scores"]),
                    "refinement_cpu_seconds": arm["refinement_cpu_seconds"],
                    "total_cpu_seconds": source["initializer_cpu_seconds"] + arm["refinement_cpu_seconds"],
                }
                for name, arm in arms.items()
            },
        },
        "primary": common_panel(store, arms, PRIMARY, source["initializer_cpu_seconds"]),
        "all_trained_sensitivity": common_panel(store, arms, TRAINED, source["initializer_cpu_seconds"]),
    }
    return {
        "status": "complete",
        "panels": panels,
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "arm_receipts": {
            name: {
                "path": f"arms/{name}.json",
                "checksum": pilot.digest(arms[name]),
                "byte_sha256": file_hash(root / "arms" / f"{name}.json"),
                "training_cpu_seconds": arms[name].get("training_cpu_seconds"),
                "refinement_cpu_seconds": arms[name]["refinement_cpu_seconds"],
            }
            for name in TRAINED
        },
        "cost_clock": (
            "deployable initializer plus frozen training/ranking/repair/compiler clocks; "
            "audit and outer work reported separately"
        ),
    }


def execute_case(
    root: Path, parent: dict[str, Any], case: dict[str, Any], loader: Callable[[int, str], NDArray[np.complex128]]
) -> dict[str, Any]:
    """Measure input, lazy initialization and all-role training as disjoint durable phases."""
    batch_root = root.parent.parent
    verify_parent(batch_root, parent)
    guard_disk(root)

    def input_vector() -> dict[str, Any]:
        target = np.asarray(loader(case["num_qubits"], case["target_id"]), dtype=np.complex128)
        if (
            target.shape != (2 ** case["num_qubits"],)
            or not np.all(np.isfinite(target))
            or abs(np.linalg.norm(target) - 1) > 1e-12
            or hashlib.sha256(target.tobytes()).hexdigest() != case["target_vector_sha256"]
        ):
            msg = "Loaded confirmation target differs from its complete frozen parent case."
            raise ValueError(msg)
        return {"target": [[float(z.real), float(z.imag)] for z in target]}

    loaded = phase(root / "input", [pilot.digest(parent), case, "input"], input_vector)
    target = np.asarray(list(starmap(complex, loaded["value"]["target"])), dtype=np.complex128)
    guard_disk(root)

    def initialize() -> dict[str, Any]:
        child = root / "initializer" / "work"
        engineering = parent["phase"] == "engineering"
        binding = (
            None
            if engineering
            else {"path": str((batch_root / "PROTOCOL.json").resolve()), "checksum": pilot.digest(parent)}
        )
        spec = {"fixture": "synthetic rank-two branch only"} if engineering else constructor.constructor_spec()
        manifest = lazy.prepare(
            child,
            target,
            case["target_id"],
            case["replica"],
            phase="engineering" if engineering else "confirmation",
            constructor=spec,
            parent=binding,
        )

        def fallback(path: Path) -> dict[str, Any]:
            if engineering:
                msg = "Small q1-q3 engineering targets must take their exact rank-two branch."
                raise ValueError(msg)
            manifest = constructor.prepare(
                path,
                target,
                case["target_id"],
                case["replica"],
                parent_path=batch_root / "PROTOCOL.json",
                parent_checksum=pilot.digest(parent),
            )
            return constructor.run(path, manifest, target)

        initialized = lazy.run(child, manifest, target, fallback)
        return {
            "endpoint": initialized["endpoint"],
            "initializer_checksum": pilot.digest(initialized),
            "rank_two_selected": initialized["rank_two_selected"],
            "ideal_tt_overlap": initialized["ideal_tt_overlap"],
            "nested_legacy_cpu_seconds": initialized["nested_legacy_cpu_seconds"],
        }

    initial = phase(root / "initializer", [pilot.digest(parent), case, "initializer"], initialize)
    source = copy.deepcopy(initial["value"]["endpoint"])
    source["initializer_cpu_seconds"] = (
        loaded["cpu_seconds"] + initial["cpu_seconds"] + parent["preparation_cpu_seconds"] / len(parent["cases"])
    )
    guard_disk(root)
    trained = phase(
        root / "training",
        [pilot.digest(parent), case, source, "all_roles"],
        lambda: train_case(root / "training/work", parent, case, source, target),
    )
    verify_parent(batch_root, parent)
    return {
        **trained["value"],
        "target_id": case["target_id"],
        "replica": case["replica"],
        "initializer": {
            key: initial["value"][key]
            for key in ("initializer_checksum", "rank_two_selected", "ideal_tt_overlap", "nested_legacy_cpu_seconds")
        },
        "phase_costs": {
            name: {
                "cpu_seconds": row["cpu_seconds"],
                "wall_seconds": row["wall_seconds"],
                "checksum": pilot.digest(row),
            }
            for name, row in (("input", loaded), ("initializer", initial), ("training_and_evaluation", trained))
        },
        "amortized_preparation_cpu_seconds": parent["preparation_cpu_seconds"] / len(parent["cases"]),
        "cost_accounting": (
            "disjoint phase totals inside full outer invocation; nested costs are diagnostic and never added twice"
        ),
        "peak_rss": peak_rss(),
    }


def target_loader(root: Path, parent: dict[str, Any], num_qubits: int, target_id: str) -> NDArray[np.complex128]:
    """Load only a confirmation member after the parent boundary has been authenticated."""
    verify_parent(root, parent)
    if parent["phase"] != "confirmation" or num_qubits != 6 or target_id not in TARGETS:
        msg = "The scientific confirmation loader accepts only the fixed q6 held-out panel."
        raise ValueError(msg)
    return load_target(num_qubits, target_id).state_vector_copy()


def run_case(
    job: dict[str, Any], *, loader: Callable[[int, str], NDArray[np.complex128]] | None = None
) -> dict[str, Any]:
    """Authenticate the entire frozen parent, then run or exactly resume one complete case."""
    root, parent, case = Path(job["root"]), job["parent"], job["case"]
    verify_parent(root, parent)
    if case not in parent["cases"] or (parent["phase"] == "confirmation" and loader is not None):
        msg = "A case or replacement loader is outside the frozen confirmation parent."
        raise ValueError(msg)
    child = root / f"{case['target_id']}__{case['replica']}"
    result = phase(
        child / "execution",
        [pilot.digest(parent), case, "complete_physical_case"],
        lambda: execute_case(
            child / "execution",
            parent,
            case,
            (lambda n, t: target_loader(root, parent, n, t)) if loader is None else loader,
        ),
    )
    record = case_record(parent, case, result)
    path = child / "RESULT.json"
    if path.exists() and pilot.digest(read(path)) != pilot.digest(record):
        msg = "Cached confirmation case fields differ from the complete authenticated physical invocation."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, record)
    return read(path)


def case_record(parent: dict[str, Any], case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every compact outer field from the sealed complete physical invocation."""
    return {
        **result["value"],
        "signature": pilot.digest([parent, case]),
        "parent_checksum": pilot.digest(parent),
        "physical_invocation": {
            "cpu_seconds": result["cpu_seconds"],
            "wall_seconds": result["wall_seconds"],
            "checksum": pilot.digest(result),
            "scope": (
                "whole case including copies, phase setup, initialization, all seven roles, "
                "scoring, custody and replay; excludes final publication"
            ),
        },
    }


def read_case(root: Path, parent: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Read-only authentication for reporting: no target decode, numerical callback or publication."""
    verify_parent(root, parent)
    if case not in parent["cases"]:
        msg = "Read-only confirmation case is outside its complete frozen parent."
        raise ValueError(msg)
    child = root / f"{case['target_id']}__{case['replica']}"
    physical = read_phase(child / "execution", [pilot.digest(parent), case, "complete_physical_case"])
    result = read(child / "RESULT.json")
    if pilot.digest(result) != pilot.digest(case_record(parent, case, physical)):
        msg = "Cached confirmation report fields differ from their complete physical receipts."
        raise ValueError(msg)
    return result


def attempt_case(job: dict[str, Any]) -> dict[str, Any]:
    """Retain explicit failed cases while global custody violations still stop execution."""
    try:
        return run_case(job)
    except (ArithmeticError, RuntimeError, MemoryError, OSError) as error:
        case = job["case"]
        child = Path(job["root"]) / f"{case['target_id']}__{case['replica']}"
        path = child / "execution/ATTEMPTS.json"
        attempts = read(path)["attempts"] if path.exists() else []
        known = bool(attempts) and all(
            a["status"] != "running" and a.get("cpu_seconds") is not None and a.get("wall_seconds") is not None
            for a in attempts
        )
        record = {
            "status": "failed",
            "target_id": case["target_id"],
            "replica": case["replica"],
            "failure": f"{type(error).__name__}: {error}",
            "physical_cpu_seconds": sum(a["cpu_seconds"] for a in attempts) if known else None,
            "physical_wall_seconds": sum(a["wall_seconds"] for a in attempts) if known else None,
            "timing_complete": known,
        }
        pilot.write_record(child / "FAILURE.json", record)
        return record


def analyze(records: Sequence[dict[str, Any]], targets: Sequence[str], replicas: Sequence[int]) -> dict[str, Any]:
    """Apply the fixed target-balanced gate only to a complete panel of valid exact scores."""
    cases = {(r["target_id"], r["replica"]): r for r in records}
    expected = {(t, r) for t in targets for r in replicas}
    if len(cases) != len(records) or set(cases) != expected or any(r["status"] != "complete" for r in records):
        return {
            "status": "incomplete_or_failed",
            "promoted": False,
            "expected_cases": len(expected),
            "complete_cases": sum(r["status"] == "complete" for r in records),
            "missing_cases": [list(key) for key in sorted(expected - set(cases))],
        }
    analysis = {}
    for panel in ("primary", "all_trained_sensitivity", "full_cap"):
        methods: dict[str, Any] = {}
        for name in METHODS:
            values = {}
            for target in targets:
                per_replica = []
                for replica in replicas:
                    selected = cases[target, replica]["panels"][panel]
                    if selected["status"] != "complete" or set(selected["arms"]) != set(METHODS):
                        return {"status": "invalid_role_panel", "promoted": False}
                    scores = selected["arms"][name]["scores"]["fidelities"]
                    if set(scores) != set(base.channels.NOISE_IDS) or any(
                        not math.isfinite(v) or not -1e-10 <= v <= 1 + 1e-10 for v in scores.values()
                    ):
                        return {"status": "invalid_exact_score_panel", "promoted": False}
                    per_replica.append(
                        float(np.mean([scores[noise] for noise in base.channels.NOISE_IDS if noise != "noiseless"]))
                    )
                values[target] = float(np.mean(per_replica))
            methods[name] = {"mean": float(np.mean(list(values.values()))), "by_target": values}
        strongest = max((name for name in METHODS if name != CANDIDATE), key=lambda name: methods[name]["mean"])
        comparisons: dict[str, Any] = {}
        for name in METHODS:
            if name == CANDIDATE:
                continue
            deltas = {
                target: methods[CANDIDATE]["by_target"][target] - methods[name]["by_target"][target]
                for target in targets
            }
            comparisons[name] = {
                "mean_advantage": float(np.mean(list(deltas.values()))),
                "by_target_advantage": deltas,
                "target_wins": sum(value > 1e-10 for value in deltas.values()),
            }
        passes = all(
            comparisons[name]["mean_advantage"] >= 0.005 and comparisons[name]["target_wins"] == len(targets)
            for name in {BASELINE, strongest}
        )
        analysis[panel] = {
            "methods": methods,
            "strongest_control": strongest,
            "comparisons": comparisons,
            "passes_fixed_and_strongest_rule": passes,
        }
    return {
        "status": "complete",
        "promoted": analysis["primary"]["passes_fixed_and_strongest_rule"],
        "candidate": CANDIDATE,
        "primary_baseline": BASELINE,
        "panels": analysis,
        "target_aggregation": "each target: mean two replicas of mean eleven noise conditions; aggregate: mean targets",
    }


def summarize(root: Path, parent: dict[str, Any]) -> dict[str, Any]:
    """Keep every fixed case and role visible; never average an incomplete scientific panel."""
    records, rows = [], []
    for case in parent["cases"]:
        record: dict[str, Any]
        child = root / f"{case['target_id']}__{case['replica']}"
        if (child / "RESULT.json").exists():
            record = read(child / "RESULT.json")

            physical = read_phase(child / "execution", [pilot.digest(parent), case, "complete_physical_case"])
            if pilot.digest(record) != pilot.digest(case_record(parent, case, physical)):
                msg = "Confirmation summary cannot adopt changed case scores or physical receipts."
                raise ValueError(msg)
        elif (child / "FAILURE.json").exists():
            record = read(child / "FAILURE.json")
        else:
            record = {"status": "pending", "target_id": case["target_id"], "replica": case["replica"]}
        records.append(record)
        for panel in ("primary", "all_trained_sensitivity", "full_cap"):
            selected = record.get("panels", {}).get(panel, {})
            for method in METHODS:
                arm = selected.get("arms", {}).get(method, {})
                score = arm.get("scores", {})
                for noise in base.channels.NOISE_IDS:
                    counts = score.get("ballarin_counts" if noise == "ballarin_coupled" else "physical_counts", {})
                    rows.append({
                        "target_id": case["target_id"],
                        "replica": case["replica"],
                        "panel": panel,
                        "method": method,
                        "noise_id": noise,
                        "status": record["status"],
                        "fidelity": score.get("fidelities", {}).get(noise),
                        "one_qubit": counts.get("one_qubit"),
                        "two_qubit": counts.get("two_qubit"),
                        "depth": counts.get("depth"),
                        "parameters": score.get("parameters"),
                        "exactness": "exact_calibrated_channel",
                        "evaluation_checksum": score.get("evaluation_checksum"),
                        "cutoff_total_cpu_seconds": selected.get("total_cpu_seconds"),
                        "endpoint_total_cpu_seconds": arm.get("total_cpu_seconds")
                        if panel == "full_cap"
                        else record.get("initializer_cpu_seconds", 0) + arm["point"]["cpu_seconds"]
                        if "point" in arm
                        else None,
                        "cpu_scope": record.get("cost_clock"),
                        "failure": record.get("failure"),
                    })
    analysis = analyze(records, parent["targets"], parent["replicas"])
    summary = {
        "version": VERSION,
        "parent_checksum": pilot.digest(parent),
        "configuration_sha256": parent["configuration_sha256"],
        "status": "complete" if analysis["status"] == "complete" else "incomplete_or_failed",
        "analysis": analysis,
        "cases": [
            {
                "target_id": r["target_id"],
                "replica": r["replica"],
                "status": r["status"],
                "checksum": pilot.digest(r),
                "failure": r.get("failure"),
                "physical_cpu_seconds": r.get("physical_invocation", {}).get(
                    "cpu_seconds", r.get("physical_cpu_seconds")
                ),
            }
            for r in records
        ],
        "preparation_cpu_seconds": parent["preparation_cpu_seconds"],
        "actual_complete_experiment_cpu_seconds": parent["preparation_cpu_seconds"]
        + sum(r["physical_invocation"]["cpu_seconds"] for r in records)
        if all(r["status"] == "complete" for r in records)
        else None,
    }
    pilot.write_record(root / "SUMMARY.json", summary)
    with (root / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze, bind or run exactly one complete confirmation recipe with explicit resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=base.runner.OUTPUT / "confirmation-best-candidate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-design", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--recipe", choices=RECIPES)
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--lazy-root", type=Path, default=base.runner.OUTPUT / "lazy-initializer-reproduction")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 2:
        parser.error("Allocate one or two single-thread workers for the complete fixed panel.")
    if args.freeze_design:
        if args.recipe is None or args.development_root is None:
            parser.error("Design freeze requires the selected developed recipe and its authenticated evidence root.")
        freeze_design(args.output, args.recipe, args.development_root, args.lazy_root)
        return 0
    if args.recipe is not None or args.development_root is not None:
        parser.error("Recipe choices are only accepted during design freeze.")
    if args.prepare_only:
        prepare(args.output)
        return 0
    parent = read(args.output / "PROTOCOL.json")
    verify_parent(args.output, parent)
    guard_disk(
        args.output,
        sum(not (args.output / f"{c['target_id']}__{c['replica']}" / "RESULT.json").exists() for c in parent["cases"]),
    )
    summarize(args.output, parent)
    jobs = [{"root": str(args.output), "parent": parent, "case": case} for case in parent["cases"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(attempt_case, jobs):
            summarize(args.output, parent)
            print(result["target_id"], result["replica"], result["status"], flush=True)
    final = summarize(args.output, parent)
    return 0 if final["status"] == "complete" else 1


def verify_design(root: Path, design: dict[str, Any]) -> None:
    """Check every fixed configuration field and dependency before any target loader call."""
    engineering = design["phase"] == "engineering"
    if not engineering:
        require_single_threads()
    exact = (
        base.exact.Config(**design["algorithm"]["training"]["exact"])
        if engineering
        else base.exact.Config(protected_initial_gates=0)
    )
    sampled = (
        base.stochastic.Config(**design["algorithm"]["training"]["stochastic"])
        if engineering
        else base.stochastic.Config(protected=0)
    )
    expected = configuration(design["recipe"], exact, sampled, design["targets"], design["replicas"])
    if (
        design["version"] != VERSION
        or design["status"] != "frozen"
        or design["phase"] not in {"confirmation", "engineering"}
        or (not engineering and (design["targets"] != list(TARGETS) or design["replicas"] != list(REPLICAS)))
        or (engineering and (not design["targets"] or any(not t.startswith("synthetic_") for t in design["targets"])))
        or any(pilot.digest(design[k]) != pilot.digest(value) for k, value in expected.items())
        or design["configuration_sha256"] != pilot.digest(expected)
        or design["source_hashes"] != sources(design["recipe"])
        or design["benchmark_hashes"] != base.runner.custody()["benchmark_hashes"]
        or design["environment"] != construction.environment()
        or (not engineering and design.get("thread_limits") != thread_limits())
        or any(file_hash(Path(p)) != h for p, h in design["development_evidence"].items())
        or pilot.digest(read(root / "DESIGN.json")) != pilot.digest(design)
        or (
            not engineering
            and (
                not design["disk_plan"]["fits"]
                or design["disk_plan"]["cases"] != 10
                or design["disk_plan"]["required_bytes"] != 10 * PER_CASE_BYTES + RESERVE_BYTES
            )
        )
    ):
        msg = "Confirmation design, complete algorithm, evidence or numerical custody changed."
        raise ValueError(msg)


def prepare(root: Path, *, vectors: dict[str, NDArray[np.complex128]] | None = None) -> dict[str, Any]:
    """Bind all five vectors only after complete design freeze; never start a constructor."""
    design = read(root / "DESIGN.json")
    verify_design(root, design)
    engineering = design["phase"] == "engineering"
    if engineering != (vectors is not None):
        msg = "Only explicit small synthetic engineering inputs may replace the fixed target loader."
        raise ValueError(msg)
    if vectors is not None and set(vectors) != set(design["targets"]):
        msg = "Engineering input vectors must cover the entire distinct frozen target panel."
        raise ValueError(msg)
    guard_disk(root, len(design["targets"]) * len(design["replicas"]))

    def bind() -> dict[str, Any]:
        cases = []
        for target_id in design["targets"]:
            target = (
                np.asarray(vectors[target_id], dtype=np.complex128)
                if vectors is not None
                else load_target(6, target_id).state_vector_copy()
            )
            if (
                target.ndim != 1
                or target.size not in ({2, 4, 8} if engineering else {64})
                or not np.all(np.isfinite(target))
                or abs(np.linalg.norm(target) - 1) > 1e-12
            ):
                msg = "Confirmation vector has wrong dimension, norm or finite values."
                raise ValueError(msg)
            cases.extend(
                {
                    "target_id": target_id,
                    "replica": replica,
                    "num_qubits": target.size.bit_length() - 1,
                    "target_vector_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                }
                for replica in design["replicas"]
            )
        return {"cases": cases}

    bound = phase(root / "preparation", [VERSION, pilot.digest(design), "all_case_hash_binding"], bind)
    if vectors is not None and any(
        hashlib.sha256(np.asarray(vectors[case["target_id"]], dtype=np.complex128).tobytes()).hexdigest()
        != case["target_vector_sha256"]
        for case in bound["value"]["cases"]
    ):
        msg = "Supplied engineering inputs changed after complete hash binding."
        raise ValueError(msg)
    parent: dict[str, Any] = {
        **design,
        "schema": constructor.PARENT_SCHEMA,
        "cases": bound["value"]["cases"],
        "design_checksum": pilot.digest(design),
        "preparation_checksum": pilot.digest(bound),
        "preparation_cpu_seconds": bound["cpu_seconds"],
        "preparation_wall_seconds": bound["wall_seconds"],
    }
    path = root / "PROTOCOL.json"
    if path.exists() and pilot.digest(read(path)) != pilot.digest(parent):
        msg = "Prepared confirmation parent differs from frozen case hashes."
        raise ValueError(msg)
    if not path.exists():
        pilot.write_record(path, parent)
    if not engineering:
        constructor.parent_protocol(path, pilot.digest(parent))
    for index, case in enumerate(parent["cases"]):
        manifest = {
            "parent_checksum": pilot.digest(parent),
            "case": case,
            "index": index,
            "training_seed": parent["seed_policy"]["case_seeds"][f"{case['target_id']}__{case['replica']}"],
        }
        child = root / f"{case['target_id']}__{case['replica']}" / "CASE.json"
        if child.exists() and read(child) != manifest:
            msg = "Confirmation child manifest changed."
            raise ValueError(msg)
        if not child.exists():
            pilot.write_record(child, manifest)
    return read(path)


if __name__ == "__main__":
    raise SystemExit(main())
