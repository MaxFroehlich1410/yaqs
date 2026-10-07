# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Collect explicitly selected scoreboards without pooling incompatible panels.

Noise conditions, construction replicas within target, then targets receive
equal weight. A failed or missing condition invalidates a method aggregate;
the raw row remains visible. Exact channel scores have no Monte Carlo error.
Development target variation is descriptive, not a population confidence interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import shutil
import statistics
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import channels, runner

if TYPE_CHECKING:
    from collections.abc import Sequence


ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
MANIFEST_PATH = ROOT / "MANIFEST.json"
PREFIX = "experiments/overnight_state_preparation_20260913_0108/"
VERSIONS = {
    "overnight.exact_structural_lbfgs.v1": "structural",
    "overnight.rank_two_development_panel.v1": "synthesis",
    "yaqs.overnight.svrg_adam.v1": "svrg",
    "overnight.ensemble_objective_comparison.v1": "ensemble",
    "overnight.strong_structural_baselines.v1": "baseline",
    "overnight.pure_geometry_comparison.v1": "geometry",
    "overnight.physical_compilation_panel.v1": "compilation",
    "overnight.ensemble_structural.v1": "ensemble_structural",
    "overnight.entropic_risk.v1": "risk",
    "overnight.mean2_stochastic_baselines.v1": "mixture_baseline",
    "overnight.long_ensemble_convergence.v1": "long_ensemble",
    "overnight.initializer_resource_policy_development.v1": "initializer",
    "overnight.conditional_structural_composition.v1": "composed",
    "overnight.zero_state_eigenstate_panel.v1": "state_compilation",
    "overnight.common_control_audit.v1": "common_control",
    "overnight.compiled_conditional_composition_development.v1": "compiled_composition",
    "overnight.complete_best_candidate_confirmation.v1": "best_candidate",
}
MIXTURE_REFERENCE_METHODS = ("structural_mean2_lbfgs", "fixed_mean2_lbfgs", "structural_mean2_adam")
MIXTURE_HISTORICAL_METHODS = ("local_adam", "local_lbfgs", "noiseless_adam", "noiseless_lbfgs", "mean2_lbfgs")
EXACT_LABELS = {
    "exact_calibrated_channel",
    "deterministic_channel_expectation",
    "deterministic exact calibrated channel expectations",
}
NOISE_FAMILIES = {
    "N1_angle_dependent": ("ballarin_coupled",),
    "N2_dephasing": tuple(noise for noise in channels.NOISE_IDS if noise.startswith("dephasing_")),
    "N3_depolarizing": tuple(noise for noise in channels.NOISE_IDS if noise.startswith("depolarizing_")),
}
Cell = tuple[str, str, str, str]


def _read(path: Path) -> dict[str, Any]:
    """Read one checksummed artifact, explaining a missing dependency explicitly.

    Raises:
        ValueError: If a required protocol or result artifact is absent.
    """
    if not path.is_file():
        msg = f"Required reporting provenance artifact is missing: {path}."
        raise ValueError(msg)
    return cast("dict[str, Any]", pilot.read_record(path))


def _kind(protocol: dict[str, Any]) -> str:
    """Identify only explicitly supported frozen numerical driver schemas.

    Raises:
        ValueError: If the protocol schema is unknown or ambiguous.
    """
    if (
        protocol.get("version") == "overnight.complete_best_candidate_confirmation.v1"
        and protocol.get("schema") == "overnight.complete_confirmation_parent.v1"
    ):
        return "best_candidate"
    if "version" in protocol and "schema" in protocol:
        msg = "Unknown ambiguous frozen panel schema."
        raise ValueError(msg)
    tag = protocol.get("version", protocol.get("schema"))
    if tag in VERSIONS:
        return VERSIONS[tag]
    if tag is None and set(protocol) == {"protocol", "source_hashes", "benchmark_hashes"}:
        spec = protocol["protocol"]
        if spec.get("name") in {"wave1", "focused_curvature"} and spec.get("phase") == "development":
            return "runner"
    msg = "Unknown frozen panel schema; reporting fails closed."
    raise ValueError(msg)


def _required_sources(kind: str) -> set[str]:
    """Return the minimum numerical dependency coverage for a known driver."""
    if kind == "best_candidate":
        module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
        return set(module.sources("original"))
    if kind == "compiled_composition":
        module = importlib.import_module(
            "experiments.overnight_state_preparation_20260913_0108.compiled_composition_experiment"
        )
        return set(module.sources())
    if kind in {"state_compilation", "common_control"}:
        name = "state_preparation_compilation_experiment" if kind == "state_compilation" else "common_control_audit"
        module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108." + name)
        return set(module.sources())
    if kind in {"initializer", "composed"}:
        module = importlib.import_module(
            "experiments.overnight_state_preparation_20260913_0108." + kind + "_experiment"
        )
        return set(module.sources())
    required = set(runner.source_hashes())
    extras = {
        "runner": (),
        "structural": ("structural.py", "structural_experiment.py"),
        "synthesis": ("synthesis.py", "synthesis_experiment.py"),
        "svrg": ("variance.py", "svrg_experiment.py", "SVRG_PROTOCOL.md"),
        "ensemble": ("ensemble_objective.py", "ensemble_experiment.py", "ENSEMBLE_PROTOCOL.md"),
        "baseline": ("structural.py", "baseline_experiment.py", "BASELINE_PROTOCOL.md"),
        "geometry": ("geometry.py", "geometry_experiment.py", "GEOMETRY_PROTOCOL.md"),
        "ensemble_structural": (
            "ensemble_objective.py",
            "ensemble_experiment.py",
            "ENSEMBLE_PROTOCOL.md",
            "ensemble_deletion.py",
            "ensemble_structural_experiment.py",
            "ENSEMBLE_STRUCTURAL_PROTOCOL.md",
        ),
        "compilation": ("compilation.py", "compilation_experiment.py", "COMPILATION_PROTOCOL.md"),
    }
    if kind == "risk":
        extras[kind] = (*extras["ensemble_structural"], "risk_objective.py", "risk_experiment.py", "RISK_PROTOCOL.md")
    if kind == "mixture_baseline":
        extras[kind] = (
            *extras["ensemble_structural"],
            "trajectory_mixture.py",
            "MIXTURE_PROTOCOL.md",
            "mixture_baseline_experiment.py",
            "MIXTURE_BASELINE_PROTOCOL.md",
        )
        required.add("experiments/state_preparation_trajectory_diagnostics.py")
    if kind == "long_ensemble":
        extras[kind] = (*extras["ensemble"], "long_ensemble_experiment.py", "LONG_ENSEMBLE_PROTOCOL.md")
    required.update(PREFIX + name for name in extras[kind])
    if kind == "baseline":
        required.update({
            "experiments/state_preparation_noisy_restructuring.py",
            "experiments/state_preparation_trajectory_diagnostics.py",
        })
    return required


def _verify_sources(protocol: dict[str, Any], kind: str) -> None:
    """Verify benchmark custody and actual bytes of the entire frozen source map.

    Raises:
        ValueError: If the startup benchmark, coverage or any frozen bytes differ.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    if protocol.get("benchmark_hashes") != manifest["benchmark_hashes"]:
        msg = "Panel benchmark hashes differ from the overnight startup manifest."
        raise ValueError(msg)
    sources = protocol.get("source_hashes", protocol.get("sources", {}))
    required = _required_sources(kind)
    if kind == "best_candidate" and protocol.get("recipe") == "state_preparation_compiled":
        module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
        required |= set(module.sources("state_preparation_compiled"))
    if not required.issubset(sources):
        msg = "Frozen source hash map omits required numerical dependencies."
        raise ValueError(msg)
    for fingerprint in (sources, protocol["benchmark_hashes"]):
        for name, expected in fingerprint.items():
            path = REPO / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                msg = f"Reporting source or benchmark hash mismatch: {name}."
                raise ValueError(msg)


def _validate_tags(row: dict[str, str]) -> None:
    """Reject explicit alternative calibration or approximate-scoring labels.

    Raises:
        ValueError: If the row contradicts the supported exact benchmark semantics.
    """
    if row.get("convention", "") not in {"", "calibrated_exact"}:
        msg = "Scoreboard convention is not calibrated_exact."
        raise ValueError(msg)
    if row.get("exactness") and row["exactness"] not in EXACT_LABELS:
        msg = "Scoreboard exactness label is not a supported exact calibrated score."
        raise ValueError(msg)


def _case_sources(protocol: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map frozen start identities, rejecting duplicate source entries.

    Raises:
        ValueError: If source target/replica identities repeat.
    """
    starts = protocol.get("starts", [])
    result = {(source["target_id"], str(source["replica"])): source for source in starts}
    if len(result) != len(starts):
        msg = "Frozen protocol has duplicate start identities."
        raise ValueError(msg)
    return result


def _structure_references(protocol: dict[str, Any], cases: set[tuple[str, str]]) -> dict[tuple[str, str], set[str]]:
    """Derive historical methods from the frozen, checksummed reference inputs.

    Raises:
        ValueError: If a historical input or expected case is missing or changed.
    """
    root = Path(protocol["historical_root"])
    summary = _read(root / "result.json")
    if pilot.digest(summary) != protocol["historical_result_checksum"]:
        msg = "Frozen structural historical-reference summary changed."
        raise ValueError(msg)
    result = {}
    for case in sorted(cases):
        jobs = [job for job in summary["jobs"] if (job["target_id"], str(job["replica"])) == case]
        if len(jobs) != 1:
            msg = "Structural historical-reference case is missing or duplicated."
            raise ValueError(msg)
        archived = _read(root / jobs[0]["path"])
        if pilot.digest(archived) != jobs[0]["checksum"]:
            msg = "Frozen structural historical-reference run changed."
            raise ValueError(msg)
        result[case] = {"historical_structural_" + method for method in archived["refined"]}
    return result


def baseline_references(protocol: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Authenticate the complete structural comparator behind the baseline panel.

    Raises:
        ValueError: If any referenced protocol, case or signature has changed.
    """
    root = Path(protocol["reference_root"])
    reference = _read(root / "PROTOCOL.json")
    if _kind(reference) != "structural" or pilot.digest(reference) != protocol["reference_protocol_checksum"]:
        msg = "Baseline reference protocol changed."
        raise ValueError(msg)
    _verify_sources(reference, "structural")
    result = {}
    for case, source in _case_sources(protocol).items():
        record = _read(root / f"{case[0]}__{case[1]}" / "RESULT.json")
        if record["signature"] != pilot.digest({"protocol": reference, "source": source}):
            msg = "Baseline reference case signature changed."
            raise ValueError(msg)
        result[case] = {
            "checksum": pilot.digest(record),
            "historical_methods": [name for name in record["arms"] if name.startswith("historical_")],
        }
    return result


def compilation_sources(protocol: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Authenticate each compilation input and its complete source-label panel.

    Raises:
        ValueError: If an input is duplicated, omitted or changed on disk.
    """
    sources = protocol["sources"]
    mapped = {(row["target_id"], str(row["replica"]), row["label"]): row for row in sources}
    expected = set(itertools.product(protocol["targets"], map(str, protocol["replicas"]), protocol["labels"]))
    if len(mapped) != len(sources) or set(mapped) != expected:
        msg = "Compilation source labels are missing or duplicated."
        raise ValueError(msg)
    checked = {}
    for source in sources:
        for name, expected_hash in source["file_hashes"].items():
            if name not in checked:
                path = Path(name)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                checked[name] = digest.hexdigest()
            if checked[name] != expected_hash:
                msg = "Compilation input artifact changed."
                raise ValueError(msg)
    return mapped


def _expected(protocol: dict[str, Any], kind: str) -> dict[tuple[str, str], set[str]]:
    """Derive every required case and method from its supported frozen protocol.

    Raises:
        ValueError: If the declared panel or frozen source coverage is invalid.
    """
    if kind == "common_control":
        module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.common_control_audit")
        entries = protocol["input_panel"]["cases"]
        if len(entries) != len(module.composed.TARGETS) or {e["target_id"] for e in entries} != set(
            module.composed.TARGETS
        ):
            msg = "Common-control input cases must cover every distinct development target."
            raise ValueError(msg)
        return {
            (entry["target_id"], "11"): {f"{method} [common_primary]" for method in module.composed.METHODS}
            for entry in entries
        }
    spec = protocol.get("protocol", protocol.get("spec", protocol))
    targets = spec["targets"]
    replicas = spec.get("replicas", [spec.get("replica")])
    if not targets or not replicas or len(set(targets)) != len(targets) or len(set(replicas)) != len(replicas):
        msg = "Frozen panel requires distinct nonempty target and replica lists."
        raise ValueError(msg)
    if any(type(replica) is not int or replica < 0 for replica in replicas):
        msg = "Frozen panel construction replicas must be nonnegative integers."
        raise ValueError(msg)
    cases = {(target, str(replica)) for target in targets for replica in replicas}
    if kind == "state_compilation":
        pairs = protocol["pairs"]
        identities = {(pair["target_id"], str(pair["replica"]), pair["label"]) for pair in pairs}
        if len(identities) != len(pairs) or identities != {
            (target, replica, label) for target, replica in cases for label in protocol["labels"]
        }:
            msg = "State compiler source pairs are missing, duplicated or outside the fixed panel."
            raise ValueError(msg)
        return {
            case: {f"{label}__{side}" for label in protocol["labels"] for side in ("before", "after")} for case in cases
        }
    if kind == "compiled_composition":
        if set(_case_sources(protocol)) != cases:
            msg = "Compiled composition starts do not cover every declared case."
            raise ValueError(msg)
        return {
            case: {
                f"{method} [{panel}]"
                for method in protocol["methods"]
                for panel in ("full_cap", "common_primary", "common_all_trained")
            }
            for case in cases
        }
    if kind == "best_candidate":
        if {(case["target_id"], str(case["replica"])) for case in protocol["cases"]} != cases or len(
            protocol["cases"]
        ) != len(cases):
            msg = "Confirmation requires every distinct target and replica."
            raise ValueError(msg)
        return {
            case: {
                f"{method} [{panel}]"
                for method in protocol["arms"]
                for panel in ("full_cap", "primary", "all_trained_sensitivity")
            }
            for case in cases
        }
    if (
        kind
        in {
            "structural",
            "svrg",
            "ensemble",
            "baseline",
            "geometry",
            "ensemble_structural",
            "risk",
            "mixture_baseline",
            "long_ensemble",
            "initializer",
            "composed",
        }
        and set(_case_sources(protocol)) != cases
    ):
        msg = "Frozen starts do not cover the declared target/replica panel."
        raise ValueError(msg)
    if kind == "compilation":
        sources = compilation_sources(protocol)
        if {(target, replica) for target, replica, _ in sources} != cases:
            msg = "Compilation sources do not cover the declared cases."
            raise ValueError(msg)
        return {
            case: {f"{label}__{arm}" for label in spec["labels"] for arm in ("uncompiled", "compiled")}
            for case in cases
        }
    if kind == "structural":
        history = _structure_references(protocol, cases)
        return {case: {"fixed_lbfgs", "structural_lbfgs", "unchanged_reference"} | history[case] for case in cases}
    if kind == "baseline":
        references = baseline_references(protocol)
        methods = {
            "fixed_lbfgs",
            "structural_lbfgs",
            "structural_exact_adam",
            "structural_weighted_adam",
            "structural_adaptive_online",
            "fixed_adaptive_online",
        }
        return {
            case: {f"{method} [{panel}]" for method in methods for panel in ("full_cap", "common_cpu")}
            | {f"{method} [historical_unmatched]" for method in references[case]["historical_methods"]}
            for case in cases
        }
    if kind == "mixture_baseline":
        history = mixture_history_paths(protocol)
        methods = {*protocol["methods"], *MIXTURE_REFERENCE_METHODS}
        return {
            case: {f"{method} [{panel}]" for method in methods for panel in ("full_cap", "common_cpu")}
            | {"unchanged_start [unchanged]"}
            | {f"{method} [historical_unmatched]" for method in MIXTURE_HISTORICAL_METHODS if case in history}
            for case in cases
        }
    if kind == "long_ensemble":
        return {
            case: {f"{method} [{panel}]" for method in protocol["arms"] for panel in ("full_cap", "common_cpu")}
            for case in cases
        }
    if kind == "composed":
        return {
            case: {f"{method} [full_cap]" for method in protocol["methods"]}
            | {f"{method} [common_primary]" for method in protocol["primary"]}
            | {f"{method} [common_all_trained]" for method in protocol["methods"] if method != "unchanged"}
            for case in cases
        }
    methods = set(protocol["arms"]) if kind in {"ensemble", "geometry"} else set(spec["methods"])
    if kind == "synthesis":
        methods.update("reference_" + method for method in spec["reference_methods"])
    if not methods:
        msg = "Frozen panel must declare at least one method."
        raise ValueError(msg)
    return dict.fromkeys(cases, methods)


def _terminal_path(root: Path) -> Path:
    """Choose one complete or explicitly failed result, rejecting ambiguity.

    Raises:
        ValueError: If no terminal result or two conflicting terminal files exist.
    """
    paths = [path for path in (root / "RESULT.json", root / "FAILED.json") if path.exists()]
    if len(paths) != 1:
        msg = f"Expected exactly one terminal result artifact in {root}."
        raise ValueError(msg)
    return paths[0]


def target_hash(target_id: str) -> str:
    """Return the immutable q6 target byte identity used by structural drivers."""
    return hashlib.sha256(runner.load_target(6, target_id).state_vector_copy().tobytes()).hexdigest()


def ensemble_structural_signature(protocol: dict[str, Any], source: dict[str, Any]) -> str:
    """Return the complete construction, target and protocol-bound case signature."""
    return pilot.digest({
        "version": protocol["version"],
        "circuit": source["candidate"]["circuit"],
        "theta": source["candidate"]["theta"],
        "target": target_hash(source["target_id"]),
        "config": protocol["config"],
        "target_id": source["target_id"],
        "replica": source["replica"],
        "protocol_checksum": pilot.digest(protocol),
    })


def file_sha256(path: Path) -> str:
    """Hash large immutable artifacts without retaining their serialized copies."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mixture_history_paths(protocol: dict[str, Any]) -> dict[tuple[str, str], Path]:
    """Map the optional complete historical panel without opening large cases.

    Raises:
        ValueError: If a frozen history file is changed, repeated or unpaired.
    """
    cases = _case_sources(protocol)
    result = {}
    for filename, fingerprint in protocol["unmatched_reference_files"].items():
        path = Path(filename)
        matches = [case for case in cases if path.parent.name == f"{case[0]}__{case[1]}"]
        if len(matches) != 1 or matches[0] in result or not path.is_file() or file_sha256(path) != fingerprint:
            msg = "Mixture historical source file is changed, duplicated or unpaired."
            raise ValueError(msg)
        result[matches[0]] = path
    if result and set(result) != set(cases):
        msg = "Mixture historical sources do not cover every frozen paired case."
        raise ValueError(msg)
    return result


def mixture_references(protocol: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Authenticate primary and historical source cases, retaining only digests.

    Large source cases are read once each and discarded before the next case.
    Historical labels stay bound to their original unmatched budget protocol.

    Raises:
        ValueError: If a source protocol, start, case or completion ledger differs.
    """
    root = Path(protocol["reference_root"])
    reference = _read(root / "PROTOCOL.json")
    summary = _read(root / "RESULT.json")
    sources = _case_sources(protocol)
    starts_path = runner.OUTPUT / "starts.json"
    original = _read(starts_path)
    selected = {
        (source["target_id"], str(source["replica"])): source
        for source in original["starts"]
        if (source["target_id"], str(source["replica"])) in sources
    }
    if (
        _kind(reference) != "ensemble_structural"
        or pilot.digest(reference) != protocol["reference_protocol_checksum"]
        or _case_sources(reference) != sources
        or selected != sources
        or file_sha256(starts_path) != protocol["source_starts_sha256"]
        or summary.get("status") != "complete"
        or summary.get("protocol_checksum") != pilot.digest(reference)
    ):
        msg = "Mixture primary reference protocol, starts or completion changed."
        raise ValueError(msg)
    _verify_sources(reference, "ensemble_structural")
    paths = mixture_history_paths(protocol)
    history_protocols: dict[Path, dict[str, Any]] = {}
    history_checksums: dict[Path, list[str]] = defaultdict(list)
    result = {}
    for case, source in sources.items():
        record = _read(root / f"{case[0]}__{case[1]}" / "RESULT.json")
        if (
            record.get("signature") != ensemble_structural_signature(reference, source)
            or record.get("status") != "complete"
            or (record["target_id"], str(record["replica"])) != case
            or set(record["arms"]) != set(MIXTURE_REFERENCE_METHODS)
        ):
            msg = "Mixture primary reference case identity, signature or methods changed."
            raise ValueError(msg)
        reference_hash = pilot.digest(record)
        arm_hashes = {
            method: pilot.digest({
                **arm,
                **(
                    {
                        "comparison": "objective_matched_deterministic_reference_heterogeneous_call_cap",
                        "cpu_scope": protocol["cpu_scope"],
                    }
                    if arm["status"] == "complete"
                    else {}
                ),
            })
            for method, arm in record["arms"].items()
        }
        del record
        historical = {}
        if case in paths:
            path = paths[case]
            history_root = path.parent.parent
            if history_root not in history_protocols:
                historical_protocol = _read(history_root / "PROTOCOL.json")
                if _kind(historical_protocol) != "ensemble" or _case_sources(historical_protocol) != sources:
                    msg = "Mixture historical protocol or paired construction changed."
                    raise ValueError(msg)
                _verify_sources(historical_protocol, "ensemble")
                history_protocols[history_root] = historical_protocol
            historical_protocol = history_protocols[history_root]
            archived = _read(path)
            identity = {
                "protocol": historical_protocol,
                "source": source,
                "order_index": historical_protocol["targets"].index(case[0]),
            }
            if (
                archived.get("signature") != pilot.digest(identity)
                or (archived["target_id"], str(archived["replica"])) != case
            ):
                msg = "Mixture historical case identity or signature changed."
                raise ValueError(msg)
            history_checksums[history_root].append(pilot.digest(archived))
            historical = {
                method: {
                    "comparison": "unmatched_historical_fixed_1100_individual_channel_call_cap",
                    "source_sha256": protocol["unmatched_reference_files"][str(path)],
                    "source_file": str(path),
                    "record": archived["arms"][method],
                }
                for method in MIXTURE_HISTORICAL_METHODS
            }
        result[case] = {
            "checksum": reference_hash,
            "arm_hashes": arm_hashes,
            "historical_checksum": pilot.digest(historical),
        }
    if sorted(summary.get("results", [])) != sorted(value["checksum"] for value in result.values()):
        msg = "Mixture primary reference completion ledger changed."
        raise ValueError(msg)
    for history_root, checksums in history_checksums.items():
        summary = _read(history_root / "RESULT.json")
        if (
            summary.get("status") not in {"complete", "complete_with_failed_arms", "complete_with_failures"}
            or summary.get("protocol_checksum") != pilot.digest(history_protocols[history_root])
            or sorted(summary.get("results", [])) != sorted(checksums)
        ):
            msg = "Mixture historical completion ledger changed."
            raise ValueError(msg)
    return result


def verify_mixture_case(folder: Path, record: dict[str, Any], reference: dict[str, Any]) -> None:
    """Bind copied references, durable artifacts and context to a numerical case.

    Raises:
        ValueError: If copied scores, artifact bytes or auxiliary context differ.
    """
    if record["reference_case_checksum"] != reference["checksum"] or any(
        pilot.digest(record["arms"].get(method)) != fingerprint
        for method, fingerprint in reference["arm_hashes"].items()
    ):
        msg = "Mixture copied primary reference case or arm changed."
        raise ValueError(msg)
    fingerprints = record["artifact_hashes"]
    actual = {
        str(path.relative_to(folder))
        for path in folder.rglob("*.json")
        if path not in {folder / "RESULT.json", folder / "REFERENCES.json"}
    }
    if (
        not fingerprints
        or set(fingerprints) != actual
        or any(
            not (folder / name).resolve().is_relative_to(folder.resolve()) or file_sha256(folder / name) != fingerprint
            for name, fingerprint in fingerprints.items()
        )
    ):
        msg = "Mixture durable artifact coverage or bytes changed."
        raise ValueError(msg)
    context = _read(folder / "REFERENCES.json")
    if (
        set(context) != {"case_checksum", "historical_unmatched"}
        or context["case_checksum"] != pilot.digest(record)
        or pilot.digest(context["historical_unmatched"]) != reference["historical_checksum"]
    ):
        msg = "Mixture auxiliary historical context differs from its frozen source case."
        raise ValueError(msg)


def verify_long_ensemble_case(folder: Path, record: dict[str, Any]) -> None:
    """Verify all durable optimizer, invocation and score files behind a long case.

    Raises:
        ValueError: If a recorded artifact is missing, added or changed.
    """
    actual = {str(path.relative_to(folder)) for path in folder.rglob("*.json") if path != folder / "RESULT.json"}
    fingerprints = record["artifact_hashes"]
    if (
        not fingerprints
        or set(fingerprints) != actual
        or any(
            not (folder / name).resolve().is_relative_to(folder.resolve()) or file_sha256(folder / name) != fingerprint
            for name, fingerprint in fingerprints.items()
        )
    ):
        msg = "Long-ensemble durable artifact coverage or bytes changed."
        raise ValueError(msg)


def initializer_sources(protocol: dict[str, Any]) -> None:
    """Authenticate construction inputs without calling any reconstruction routine.

    Frozen numerical environments must agree with one another; reporting does
    not require its own thread settings to match a past numerical execution.

    Raises:
        ValueError: If settings, fresh endpoints, archive inputs or measured costs differ.
    """
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.initializer_experiment")
    if (
        protocol["phase"] != "development"
        or protocol["targets"] != list(module.TARGETS)
        or protocol["replicas"] != [11]
        or protocol["methods"] != list(module.METHODS)
        or protocol["threshold"] != module.initializer_policy.OVERLAP_THRESHOLD
        or protocol["config"] != asdict(module.initializer_training.Config())
    ):
        msg = "Initializer frozen panel, threshold or optimizer configuration changed."
        raise ValueError(msg)
    cached = {}
    for source in protocol["starts"]:
        files = source["files"]
        for filename, fingerprint in files.items():
            if filename not in cached:
                cached[filename] = _read(Path(filename))
            if pilot.digest(cached[filename]) != fingerprint:
                msg = "Initializer construction source file changed."
                raise ValueError(msg)
        named = {Path(filename).name: Path(filename) for filename in files}
        if len(files) != 5 or set(named) != {
            "RESULT.json",
            "AUDIT.json",
            "PROTOCOL.json",
            "BATCH_PROTOCOL.json",
            "SUMMARY.json",
        }:
            msg = "Initializer construction source coverage is incomplete or ambiguous."
            raise ValueError(msg)
        built, audit, manifest, batch, summary = (
            cached[str(named[name])]
            for name in ("RESULT.json", "AUDIT.json", "PROTOCOL.json", "BATCH_PROTOCOL.json", "SUMMARY.json")
        )
        case_root = named["RESULT.json"].parent
        candidates = [
            case for case in batch["cases"] if (case["target_id"], case["replica"]) == (source["target_id"], 11)
        ]
        if len(candidates) != 1:
            msg = "Initializer fresh construction case is missing or duplicated."
            raise ValueError(msg)
        case = candidates[0]
        cost = source["construction_cpu_seconds"]
        circuit = pilot.restore_circuit(built["circuit"])
        if (
            batch["version"] != module.construction_experiment.VERSION
            or batch["targets"] != protocol["targets"]
            or batch["replicas"] != [11]
            or batch["phase"] != "development"
            or len(batch["cases"]) != 4
            or {case["target_id"] for case in batch["cases"]} != set(protocol["targets"])
            or set(summary["case_checksums"]) != set(protocol["targets"])
            or summary["status"] != "complete"
            or not summary["all_archive_reproductions_match"]
            or summary["protocol_checksum"] != pilot.digest(batch)
            or summary["case_checksums"][source["target_id"]] != pilot.digest(audit)
            or Path(case["root"]) != case_root
            or case["manifest_checksum"] != pilot.digest(manifest)
            or manifest["version"] != module.construction.VERSION
            or manifest["target_id"] != source["target_id"]
            or manifest["replica"] != 11
            or manifest["target_vector_sha256"] != target_hash(source["target_id"])
            or manifest["config"] != asdict(module.construction.Config())
            or built["signature"] != pilot.digest(manifest)
            or audit["signature"] != pilot.digest({"protocol": batch, "case": case})
            or audit["construction_checksum"] != pilot.digest(built)
            or audit["status"] != "complete"
            or not audit["matches"]
            or built["status"] != "complete"
            or not built["archive_reproduction"]["matches"]
            or not built["timing_complete"]
            or not source["timing_complete"]
            or cost is None
            or not math.isfinite(cost)
            or cost <= 0
            or cost != built["construction_cpu_seconds"]
            or cost != audit["construction_cpu_seconds"]
            or built["physical_counts"]["one_qubit"] != 78
            or built["physical_counts"]["two_qubit"] != 30
            or built["parameters"] != 108
            or built["updates"] != 570
            or circuit.num_qubits != 6
            or circuit.num_params != 108
            or len(circuit.gates) != 108
            or runner.counts(circuit) != built["physical_counts"]
            or pilot.digest([source["circuit"], source["theta"]]) != pilot.digest([built["circuit"], built["theta"]])
        ):
            msg = "Initializer construction identity, configuration, audit or measured cost changed."
            raise ValueError(msg)
        for input_protocol in (batch, manifest):
            if (
                input_protocol["environment"] != protocol["environment"]
                or input_protocol["benchmark_hashes"] != protocol["benchmark_hashes"]
                or not input_protocol["sources"]
                or any(
                    protocol["sources"].get(name) != fingerprint
                    for name, fingerprint in input_protocol["sources"].items()
                )
            ):
                msg = "Initializer construction numerical source or environment custody changed."
                raise ValueError(msg)
        archive = manifest["archive_reference"]
        if archive is None or any(
            pilot.digest(_read(Path(archive[name]))) != archive[checksum]
            for name, checksum in (("path", "checksum"), ("manifest_path", "manifest_checksum"))
        ):
            msg = "Initializer source archive reference changed."
            raise ValueError(msg)
        if built["attempts_checksum"] != pilot.digest(_read(case_root / "ATTEMPTS.json")) or any(
            block["checksum"] != pilot.digest(_read(case_root / "blocks" / f"{name}.json"))
            for name, block in built["blocks"].items()
        ):
            msg = "Initializer construction block or timing ledger changed."
            raise ValueError(msg)


def initializer_records(
    root: Path, protocol: dict[str, Any], expected: dict[tuple[str, str], set[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return authenticated six-arm cases and their target-keyed summary ledger.

    Raises:
        ValueError: If a case, initializer, training reference or durable artifact differs.
    """
    initializer_sources(protocol)
    summary = _read(root / "SUMMARY.json")
    if summary["status"] not in {"complete", "partial_failure"} or summary["protocol_checksum"] != pilot.digest(
        protocol
    ):
        msg = "Initializer SUMMARY is nonterminal or belongs to another protocol."
        raise ValueError(msg)
    records = []
    for case, source in _case_sources(protocol).items():
        folder = root / f"{case[0]}__{case[1]}"
        record = _read(folder / "RESULT.json")
        actual = {str(path.relative_to(folder)) for path in folder.rglob("*.json") if path != folder / "RESULT.json"}
        if (
            record["signature"] != pilot.digest({"protocol": protocol, "source": source})
            or (record["target_id"], str(record["replica"])) != case
            or record["status"] not in {"complete", "partial_failure", "failed"}
            or set(record["arms"]) != expected[case]
            or set(record["dependencies"]) != actual
            or any(
                not (folder / name).resolve().is_relative_to(folder.resolve())
                or file_sha256(folder / name) != fingerprint
                for name, fingerprint in record["dependencies"].items()
            )
        ):
            msg = "Initializer case identity, methods or durable dependencies changed."
            raise ValueError(msg)
        if "initializer_checksum" in record:
            initials = _read(folder / "INITIALIZERS.json")
            if record["initializer_checksum"] != pilot.digest(initials) or initials[
                "attempts_checksum"
            ] != pilot.digest(_read(folder / "INITIALIZER_ATTEMPTS.json")):
                msg = "Initializer setup or its attempt ledger changed."
                raise ValueError(msg)
        for method, arm in record["arms"].items():
            if arm["status"] == "complete":
                if arm["training_checksum"] is not None and arm["training_checksum"] != pilot.digest(
                    _read(folder / method / "TRAINING.json")
                ):
                    msg = "Initializer arm training reference changed."
                    raise ValueError(msg)
                if pilot.digest(arm["scores"]) != pilot.digest(
                    _read(folder / "scores" / f"{arm['scores']['signature']}.json")
                ):
                    msg = "Initializer arm score reference changed."
                    raise ValueError(msg)
        records.append(record)
    checksums = {record["target_id"]: pilot.digest(record) for record in records}
    if summary["case_checksums"] != checksums:
        msg = "Initializer SUMMARY checksums do not cover the complete target panel."
        raise ValueError(msg)
    return records, summary


def initializer_availability(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact authenticated availability facts without inventing score rows."""
    return {
        record["target_id"]: {
            "case_checksum": pilot.digest(record),
            **{
                name: {
                    "status": record.get(name, {}).get("status", "not_evaluated"),
                    "clock": record.get(name, {}).get("clock"),
                    "common_cpu_seconds": record.get(name, {}).get("common_cpu_seconds"),
                    "points": [
                        {
                            "fraction": point["fraction"],
                            "cpu_seconds": point["cpu_seconds"],
                            "all_available": point["all_available"],
                            "arms": {method: arm["status"] for method, arm in point["arms"].items()},
                            "point_checksum": pilot.digest(point),
                        }
                        for point in record.get(name, {}).get("points", [])
                    ],
                }
                for name in ("common_total_cpu", "common_refinement_cpu")
            },
        }
        for record in records
    }


def composed_inputs(protocol: dict[str, Any]) -> None:
    """Authenticate conditional endpoints through the completed initializer panel.

    Raises:
        ValueError: If the source initializer, panel, protection rule or input hashes differ.
    """
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.composed_experiment")
    if (
        protocol["phase"] != "development"
        or protocol["targets"] != list(module.TARGETS)
        or protocol["replicas"] != [11]
        or protocol["methods"] != list(module.METHODS)
        or protocol["primary"] != list(module.PRIMARY)
        or protocol["exact_config"] != asdict(module.exact.Config(protected_initial_gates=0))
        or protocol["stochastic_config"] != asdict(module.stochastic.Config(protected=0))
    ):
        msg = "Composition frozen panel, methods or configuration changed."
        raise ValueError(msg)
    root = Path(protocol["initializer_root"])
    original = _read(root / "PROTOCOL.json")
    _verify_sources(original, "initializer")
    records, summary = initializer_records(root, original, _expected(original, "initializer"))
    if (
        _kind(original) != "initializer"
        or summary["status"] != "complete"
        or any(record["status"] != "complete" for record in records)
        or original["environment"] != protocol["environment"]
        or original["benchmark_hashes"] != protocol["benchmark_hashes"]
    ):
        msg = "Composition requires a complete matching initializer source panel."
        raise ValueError(msg)
    expected_files = {root / "PROTOCOL.json", root / "SUMMARY.json"}
    starts = []
    for record in records:
        folder = root / f"{record['target_id']}__11"
        initializers = _read(folder / "INITIALIZERS.json")
        starts.append({"target_id": record["target_id"], "replica": 11, "endpoint": initializers["conditional"]})
        expected_files.update({folder / "RESULT.json", folder / "INITIALIZERS.json"})
    if pilot.digest(starts) != pilot.digest(protocol["starts"]) or protocol["input_files"] != {
        str(path.resolve()): file_sha256(path) for path in expected_files
    }:
        msg = "Composition conditional source endpoint or frozen input files changed."
        raise ValueError(msg)


def composed_score(folder: Path, point: dict[str, Any], score: dict[str, Any], target: str) -> None:
    """Bind a reported endpoint to its exact all12 operation and physical attempt ledger.

    Raises:
        ValueError: If a circuit, parameter identity, score or operation ledger differs.
    """
    circuit = _read(folder / "circuits" / f"{point['circuit_hash']}.json")
    identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
    operation_path = folder / "operations" / f"audit_{pilot.digest(identity)}.json"
    operation = _read(operation_path)
    expected_signature = pilot.digest({
        "version": "overnight.ensemble_structural.v1",
        "circuit": point["circuit_hash"],
        "theta": point["theta"],
        "target": target_hash(target),
        "ranking": False,
        "profiles": list(channels.NOISE_IDS),
    })
    if (
        pilot.digest(circuit) != point["circuit_hash"]
        or operation["signature"] != expected_signature
        or pilot.digest(operation) != pilot.digest(score["evaluation"])
        or score["fidelities"] != operation["value"]["fidelities"]
        or set(score["fidelities"]) != set(channels.NOISE_IDS)
        or operation["attempts_checksum"] != pilot.digest(_read(operation_path.with_suffix(".attempts.json")))
    ):
        msg = "Composition deployed endpoint, score or operation ledger changed."
        raise ValueError(msg)


def composed_common(folder: Path, record: dict[str, Any], methods: tuple[str, ...], panel: str) -> None:
    """Authenticate chronological availability without any quality-based selection.

    Raises:
        ValueError: If methods, cutoff, endpoint availability or scores are inconsistent.
    """
    comparison = record[panel]
    if comparison["methods"] != list(methods):
        msg = "Composition common-CPU method panel changed."
        raise ValueError(msg)
    if any(record["arms"][method]["status"] != "complete" for method in methods):
        if comparison["status"] != "unavailable_failed_arm" or comparison["arms"]:
            msg = "Composition failed common-CPU arm cannot be dropped or scored as available."
            raise ValueError(msg)
        return
    cutoff = min(record["arms"][method]["refinement_cpu_seconds"] for method in methods)
    selected = {
        method: [point for point in record["arms"][method]["curve"] if point["cpu_seconds"] <= cutoff]
        for method in methods
    }
    if any(not points for points in selected.values()):
        if comparison["status"] != "unavailable_no_checkpoint" or comparison["arms"]:
            msg = "Composition missing checkpoint cannot become an available endpoint."
            raise ValueError(msg)
        return
    if (
        comparison["status"] != "complete"
        or set(comparison["arms"]) != set(methods)
        or comparison["refinement_cpu_seconds"] != cutoff
        or comparison["total_cpu_seconds"] != record["source"]["initializer_cpu_seconds"] + cutoff
    ):
        msg = "Composition common-CPU cutoff or initialization cost changed."
        raise ValueError(msg)
    for method, points in selected.items():
        arm = comparison["arms"][method]
        if pilot.digest(arm["point"]) != pilot.digest(points[-1]):
            msg = "Composition common-CPU endpoint is not the last available iterate."
            raise ValueError(msg)
        composed_score(folder, arm["point"], arm["scores"], record["target_id"])


def composed_records(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return complete source-bound cases and their inclusive physical CPU provenance.

    Raises:
        ValueError: If cases, subset arm dependencies, invocation costs or summary links differ.
    """
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.composed_experiment")
    composed_inputs(protocol)
    summary = _read(root / "SUMMARY.json")
    if summary["status"] not in {"complete", "partial_failure"} or summary["protocol_checksum"] != pilot.digest(
        protocol
    ):
        msg = "Composition SUMMARY is nonterminal or belongs to another protocol."
        raise ValueError(msg)
    records = []
    for case, source in _case_sources(protocol).items():
        folder = root / f"{case[0]}__{case[1]}"
        record = _read(folder / "RESULT.json")
        signature = pilot.digest([protocol, source])
        actual = {str(path.relative_to(folder)) for path in folder.rglob("*.json") if path != folder / "RESULT.json"}
        if (
            record["signature"] != signature
            or (record["target_id"], str(record["replica"])) != case
            or record["status"] not in {"complete", "partial_failure"}
            or set(record["arms"]) != set(module.METHODS)
            or pilot.digest(record["source"]) != pilot.digest(source["endpoint"])
            or set(record["dependencies"]) != actual
            or any(
                not (folder / name).resolve().is_relative_to(folder.resolve())
                or file_sha256(folder / name) != fingerprint
                for name, fingerprint in record["dependencies"].items()
            )
        ):
            msg = "Composition case, conditional source or durable dependencies changed."
            raise ValueError(msg)
        invocation = _read(folder / "CASE_ATTEMPTS.json")
        physical = record["physical_invocation"]
        if (
            invocation["signature"] != signature
            or any(attempt["status"] == "running" for attempt in invocation["attempts"])
            or physical["attempts_checksum"] != pilot.digest(invocation)
            or any(
                not math.isfinite(attempt[key]) or attempt[key] < 0
                for attempt in invocation["attempts"]
                for key in ("cpu_seconds", "wall_seconds")
            )
            or any(
                physical[key] != sum(attempt[key] for attempt in invocation["attempts"])
                for key in ("cpu_seconds", "wall_seconds")
            )
        ):
            msg = "Composition inclusive physical invocation timing changed or remains unknown."
            raise ValueError(msg)
        for method, arm in record["arms"].items():
            if method in module.TRAINED:
                expected_signature = pilot.digest([
                    protocol["version"],
                    source["endpoint"],
                    target_hash(case[0]),
                    protocol["exact_config"],
                    protocol["stochastic_config"],
                    runner.seed_for("composition_training", case[0], 11),
                    method,
                ])
                if (
                    arm["signature"] != expected_signature
                    or pilot.digest(arm) != pilot.digest(_read(folder / "arms" / f"{method}.json"))
                    or any(
                        Path(name).name in {"CASE_ATTEMPTS.json", "RESULT.json"}
                        or record["dependencies"].get(name) != fingerprint
                        for name, fingerprint in arm["dependencies"].items()
                    )
                ):
                    msg = "Composition trained arm or subset dependency identity changed."
                    raise ValueError(msg)
            if arm["status"] == "complete":
                if (
                    arm["total_cpu_seconds"]
                    != source["endpoint"]["initializer_cpu_seconds"] + arm["refinement_cpu_seconds"]
                ):
                    msg = "Composition arm omitted or changed its shared initializer cost."
                    raise ValueError(msg)
                composed_score(folder, arm["final"], arm["scores"], case[0])
        composed_common(folder, record, tuple(module.PRIMARY), "common_primary")
        composed_common(folder, record, tuple(module.TRAINED), "common_all_trained")
        records.append(record)
    if summary["case_checksums"] != {record["target_id"]: pilot.digest(record) for record in records}:
        msg = "Composition SUMMARY checksums do not cover every paired case."
        raise ValueError(msg)
    return records, summary


def diagnostic_sources(protocol: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Authenticate every inherited composition endpoint before reporting diagnostics."""
    root = Path(protocol["composition_root"] if kind == "state_compilation" else protocol["input_panel"]["root"])
    original = _read(root / "PROTOCOL.json")
    _verify_sources(original, "composed")
    cases, summary = composed_records(root, original)
    if summary["status"] != "complete" or any(case["status"] != "complete" for case in cases):
        msg = "Endpoint diagnostics require the complete original composition panel."
        raise ValueError(msg)
    if (
        protocol["environment"] != original["environment"]
        or protocol["benchmark_hashes"] != original["benchmark_hashes"]
    ):
        msg = "Diagnostic environment or benchmark differs from the frozen source."
        raise ValueError(msg)
    if kind == "common_control":
        module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.common_control_audit")
        expected = {
            "root": str(root.resolve()),
            "protocol_checksum": pilot.digest(original),
            "summary_checksum": pilot.digest(summary),
            "cases": [
                {
                    "target_id": case["target_id"],
                    "checksum": pilot.digest(case),
                    "points": module.select_points(
                        case["arms"], module.composed.TRAINED, case["common_primary"]["refinement_cpu_seconds"]
                    ),
                }
                for case in cases
            ],
        }
        if protocol["phase"] != "development_diagnostic" or pilot.digest(protocol["input_panel"]) != pilot.digest(
            expected
        ):
            msg = "Common-control source panel, cutoff or selected points changed."
            raise ValueError(msg)
        return cases
    module = importlib.import_module(
        "experiments.overnight_state_preparation_20260913_0108.state_preparation_compilation_experiment"
    )
    pairs, files = [], dict(original["input_files"])
    for case in cases:
        target = case["target_id"]
        folder = root / f"{target}__11"
        initials = _read(Path(original["initializer_root"]) / f"{target}__11" / "INITIALIZERS.json")
        endpoints = {f"{name}_initializer": initials[name] for name in ("legacy", "conditional")}
        for label in module.composition.PRIMARY:
            point = case["arms"][label]["final"]
            circuit = _read(folder / "circuits" / f"{point['circuit_hash']}.json")
            endpoints[label] = {"circuit": circuit, "theta": point["theta"]}
        pairs.extend(
            {
                "target_id": target,
                "replica": 11,
                "label": label,
                "endpoint": {key: endpoints[label][key] for key in ("circuit", "theta")},
                "source_case_checksum": pilot.digest(case),
                "target_sha256": target_hash(target),
            }
            for label in module.LABELS
        )
        files[str((folder / "RESULT.json").resolve())] = file_sha256(folder / "RESULT.json")
        files.update({str((folder / name).resolve()): value for name, value in case["dependencies"].items()})
    files.update({str((root / name).resolve()): file_sha256(root / name) for name in ("PROTOCOL.json", "SUMMARY.json")})
    if (
        protocol["phase"] != "development"
        or protocol["targets"] != list(module.TARGETS)
        or protocol["labels"] != list(module.LABELS)
        or protocol["replicas"] != [11]
        or protocol["noise_ids"] != list(channels.NOISE_IDS)
        or protocol["tolerance"] != module.compiler.EIGENSTATE_ATOL
        or protocol["cpu_scope"] != module.CPU_SCOPE
        or protocol["source_selection"] != "all fixed method final endpoints; no quality selection"
        or pilot.digest(protocol["pairs"]) != pilot.digest(pairs)
        or protocol["input_files"] != files
    ):
        msg = "State compiler frozen source pair, metadata or complete artifact graph changed."
        raise ValueError(msg)
    return cases


def endpoint_resources(circuit: dict[str, Any], theta: list[float]) -> dict[str, Any]:
    """Return gate metadata by materializing gates only, without state propagation."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.composed_experiment")
    restored = module.restore_physical(circuit)
    ballarin = module.exact.materialize_ballarin_circuit(restored, module.np.asarray(theta)).circuit
    return {
        "physical_counts": runner.counts(restored),
        "ballarin_counts": runner.counts(ballarin),
        "parameters": restored.num_params,
    }


def terminal_attempts(path: Path, record: dict[str, Any], signature: str) -> None:
    """Authenticate an inclusive timer and reject unfinished or unknown work."""
    ledger = _read(path)
    attempts = ledger["attempts"]
    if (
        ledger["signature"] != signature
        or record["attempts_checksum"] != pilot.digest(ledger)
        or not attempts
        or attempts[-1]["status"] != "complete"
        or any(a["status"] not in {"complete", "failed", "interrupted"} for a in attempts)
        or any(
            not isinstance(a[key], (int, float)) or not math.isfinite(a[key]) or a[key] < 0
            for a in attempts
            for key in ("cpu_seconds", "wall_seconds")
        )
        or any(record[key] != sum(a[key] for a in attempts) for key in ("cpu_seconds", "wall_seconds"))
    ):
        msg = "Diagnostic attempt identity or inclusive timer changed or remains unknown."
        raise ValueError(msg)


def state_compilation_records(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate all complete or explicitly failed compiler pairs and child costs."""
    diagnostic_sources(protocol, "state_compilation")
    module = importlib.import_module(
        "experiments.overnight_state_preparation_20260913_0108.state_preparation_compilation_experiment"
    )
    summary, failures = _read(root / "SUMMARY.json"), _read(root / "FAILURES.json")
    if summary["status"] not in {"complete", "partial_failure"} or summary["protocol_checksum"] != pilot.digest(
        protocol
    ):
        msg = "State compiler SUMMARY differs from its terminal protocol."
        raise ValueError(msg)
    records, missing = [], []
    for source in protocol["pairs"]:
        key = module.pair_key(source)
        folder = root / key
        if not (folder / "RESULT.json").exists():
            if (
                key not in failures
                or failures[key].get("status") != "failed"
                or any(failures[key].get(name) != source[name] for name in ("target_id", "replica", "label"))
            ):
                msg = "A missing compiler pair lacks an explicit matching failure record."
                raise ValueError(msg)
            if (folder / "FAILURE.json").exists() and pilot.digest(_read(folder / "FAILURE.json")) != pilot.digest(
                failures[key]
            ):
                msg = "The compiler pair failure record changed."
                raise ValueError(msg)
            missing.append(key)
            continue
        record = _read(folder / "RESULT.json")
        signature = pilot.digest([protocol, source])
        if (
            record["signature"] != signature
            or any(record.get(name) != source[name] for name in ("target_id", "replica", "label"))
            or record["dependencies"] != module.artifact_dependencies(folder)
        ):
            msg = "State compiler pair identity or complete child map changed."
            raise ValueError(msg)
        module.authenticate_result(folder, source, record)
        terminal_attempts(folder / "CASE_ATTEMPTS.json", record["physical_invocation"], signature)
        terminal_attempts(
            folder / "COMPILATION.attempts.json", record["compilation"], pilot.digest([module.VERSION, source])
        )
        for side, endpoint in {"before": source["endpoint"], "after": record["compilation"]["value"]}.items():
            scored = record[side]
            score_signature = pilot.digest({
                "version": module.measured.VERSION,
                "circuit": endpoint["circuit"],
                "theta": endpoint["theta"],
                "target": source["target_sha256"],
            })
            terminal_attempts(folder / side / "scores" / score_signature / "ATTEMPTS.json", scored, score_signature)
            resources = endpoint_resources(endpoint["circuit"], endpoint["theta"])
            if (
                scored["signature"] != score_signature
                or scored["theta"] != endpoint["theta"]
                or scored["logical_counts"] != resources["physical_counts"]
                or scored["ballarin_counts"] != resources["ballarin_counts"]
                or len(scored["theta"]) != resources["parameters"]
            ):
                msg = "State compiler score endpoint or independently materialized resource counts changed."
                raise ValueError(msg)
        records.append(record)
    if (
        set(failures) != set(missing)
        or set(summary["missing_pairs"]) != set(missing)
        or summary["case_checksums"] != {module.pair_key(record): pilot.digest(record) for record in records}
    ):
        msg = "State compiler completion/failure inventory does not match every required pair."
        raise ValueError(msg)
    return records, summary


def common_control_records(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind every common-cutoff control to its original point and exact score receipt."""
    sources = {case["target_id"]: case for case in diagnostic_sources(protocol, "common_control")}
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.common_control_audit")
    summary = _read(root / "SUMMARY.json")
    if summary["status"] != "complete" or summary["protocol_checksum"] != pilot.digest(protocol):
        msg = "Common-control SUMMARY is incomplete or has a different protocol."
        raise ValueError(msg)
    records = []
    for entry in protocol["input_panel"]["cases"]:
        target = entry["target_id"]
        source = sources[target]
        folder = root / f"{target}__11"
        record = _read(folder / "RESULT.json")
        ledger = _read(folder / "RESULT.attempts.json")
        signature = pilot.digest(pilot.digest([protocol, entry]))
        terminal_attempts(folder / "RESULT.attempts.json", record, signature)
        value = record["value"]
        metadata = {
            "status": "complete",
            "target_id": target,
            "refinement_cpu_seconds": source["common_primary"]["refinement_cpu_seconds"],
            "total_cpu_seconds": source["common_primary"]["total_cpu_seconds"],
            "points": entry["points"],
            "source_case_checksum": entry["checksum"],
            "source_unchanged_fidelities": source["arms"]["unchanged"]["scores"]["fidelities"],
        }
        actual = {
            str(path.relative_to(folder)): file_sha256(path) for path in folder.rglob("*.json") if path.parent != folder
        }
        if (
            record["signature"] != signature
            or ledger["signature"] != signature
            or record["attempts_checksum"] != pilot.digest(ledger)
            or any(a["status"] == "running" or a["cpu_seconds"] is None for a in ledger["attempts"])
            or record["cpu_seconds"] != sum(a["cpu_seconds"] for a in ledger["attempts"])
            or any(pilot.digest(value.get(key)) != pilot.digest(expected) for key, expected in metadata.items())
            or actual != value["dependencies"]
            or set(value["scores"]) != set(module.composed.TRAINED)
        ):
            msg = "Common-control case timing, source points or complete child map changed."
            raise ValueError(msg)
        for name in module.composed.TRAINED:
            score = value["scores"][name]
            if name in module.composed.PRIMARY:
                if pilot.digest(score) != pilot.digest(source["common_primary"]["arms"][name]["scores"]):
                    msg = "Reused common-control score differs from its frozen source."
                    raise ValueError(msg)
            else:
                composed_score(folder, entry["points"][name], score, target)
                point = entry["points"][name]
                circuit = _read(folder / "circuits" / f"{point['circuit_hash']}.json")
                identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
                terminal_attempts(
                    folder / "operations" / f"audit_{pilot.digest(identity)}.attempts.json",
                    score["evaluation"],
                    score["evaluation"]["signature"],
                )
                operation_attempts = _read(folder / "operations" / f"audit_{pilot.digest(identity)}.attempts.json")[
                    "attempts"
                ]
                charged = {
                    "individual_channel_calls": 12,
                    "density_direction_passes": 12,
                    "physical_attempted_channel_calls": sum(a["attempted_channel_calls"] for a in operation_attempts),
                    "physical_completed_channel_calls": sum(a["completed_channel_calls"] for a in operation_attempts),
                    "physical_completed_direction_passes": sum(
                        a["completed_channel_calls"] for a in operation_attempts
                    ),
                }
                if any(score["evaluation"][key] != expected for key, expected in charged.items()):
                    msg = "Common-control charged component counts differ from their physical receipts."
                    raise ValueError(msg)
                resources = endpoint_resources(circuit, point["theta"])
                if any(score[key] != expected for key, expected in resources.items()):
                    msg = "Common-control independently materialized resource counts changed."
                    raise ValueError(msg)
        records.append(record)
    if sorted(summary["cases"]) != sorted(pilot.digest(record) for record in records):
        msg = "Common-control SUMMARY does not authenticate the complete distinct case panel."
        raise ValueError(msg)
    expected_summary = {
        "status": "complete",
        "protocol_checksum": pilot.digest(protocol),
        "cases": [pilot.digest(record) for record in records],
        **module.summarize(records, module.composed.TARGETS),
    }
    if pilot.digest(summary) != pilot.digest(expected_summary):
        msg = "Common-control strongest-control or paired aggregate changed."
        raise ValueError(msg)
    return records, summary


def common_control_rows(protocol: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all seven controls at the unchanged primary clock, without extra scoring."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.common_control_audit")
    source_root = Path(protocol["input_panel"]["root"])
    rows = []
    for record in records:
        value = record["value"]
        target = value["target_id"]
        source = _read(source_root / f"{target}__11" / "RESULT.json")
        for method in module.composed.METHODS:
            unchanged = method == "unchanged"
            reused = unchanged or method in module.composed.PRIMARY
            score = source["arms"]["unchanged"]["scores"] if unchanged else value["scores"][method]
            point = source["arms"]["unchanged"]["final"] if unchanged else value["points"][method]
            for noise in channels.NOISE_IDS:
                counts = score["ballarin_counts" if noise == "ballarin_coupled" else "physical_counts"]
                rows.append({
                    "target_id": target,
                    "replica": 11,
                    "panel": "common_primary",
                    "method": method,
                    "noise_id": noise,
                    "status": "complete",
                    "fidelity": score["fidelities"][noise],
                    **counts,
                    "parameters": score["parameters"],
                    "selected_cpu_seconds": point["cpu_seconds"],
                    "refinement_cpu_cutoff": value["refinement_cpu_seconds"],
                    "total_cpu_cutoff": value["total_cpu_seconds"],
                    "circuit_hash": point["circuit_hash"],
                    "compiled_final": point.get("compiled_final", False),
                    "score_origin": "reused_source" if reused else "fresh_audit",
                    "inclusive_case_audit_cpu_seconds": record["cpu_seconds"],
                    "evaluation_cpu_seconds": 0.0 if reused else score["evaluation"]["cpu_seconds"],
                    **{
                        key: 0 if reused else score["evaluation"][key]
                        for key in (
                            "physical_attempted_channel_calls",
                            "physical_completed_channel_calls",
                            "physical_completed_direction_passes",
                        )
                    },
                    "comparison": (
                        "all controls at the unchanged four-structural primary CPU cutoff; "
                        "diagnostic scoring excluded from comparison clock"
                    ),
                })
    return rows


def write_common_control_scoreboard(root: Path) -> Path:
    """Create the missing derived CSV only from fully authenticated control artifacts."""
    protocol = _read(root / "PROTOCOL.json")
    if _kind(protocol) != "common_control":
        msg = "The derived common-control writer requires its exact protocol kind."
        raise ValueError(msg)
    _verify_sources(protocol, "common_control")
    records, _ = common_control_records(root, protocol)
    rows = common_control_rows(protocol, records)
    path = root / "SCOREBOARD.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def point_score(root: Path, point: dict[str, Any], target_sha256: str) -> dict[str, Any]:
    """Return a score reconstructed solely from physical gate and operation receipts."""
    circuit = _read(root / "circuits" / f"{point['circuit_hash']}.json")
    identity = {"circuit_hash": point["circuit_hash"], "theta": point["theta"]}
    path = root / "operations" / f"audit_{pilot.digest(identity)}.json"
    operation = _read(path)
    signature = pilot.digest({
        "version": "overnight.ensemble_structural.v1",
        "circuit": point["circuit_hash"],
        "theta": point["theta"],
        "target": target_sha256,
        "ranking": False,
        "profiles": list(channels.NOISE_IDS),
    })
    terminal_attempts(path.with_suffix(".attempts.json"), operation, signature)
    fidelities = operation["value"]["fidelities"]
    if (
        pilot.digest(circuit) != point["circuit_hash"]
        or operation["signature"] != signature
        or set(fidelities) != set(channels.NOISE_IDS)
        or any(not math.isfinite(v) or not -1e-10 <= v <= 1 + 1e-10 for v in fidelities.values())
    ):
        msg = "Compiled endpoint circuit, target, profile set or operation identity changed."
        raise ValueError(msg)
    return {
        "fidelities": fidelities,
        "mean_noisy_fidelity": float(np.mean([fidelities[n] for n in channels.NOISE_IDS if n != "noiseless"])),
        **endpoint_resources(circuit, point["theta"]),
        "evaluation": operation,
    }


def compiler_point(root: Path, point: dict[str, Any], *, state: bool, role: str) -> dict[str, Any]:
    """Reconstruct compiler availability without invoking a compiler or numerical cache API."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.compiled_composition")
    circuit = _read(root / "circuits" / f"{point['circuit_hash']}.json")
    if pilot.digest(circuit) != point["circuit_hash"]:
        msg = "Compiler input circuit differs from its physical identity."
        raise ValueError(msg)
    source = {"circuit": circuit, "theta": point["theta"]}
    folder = "state_compilations" if state else "compilations"
    path = root / folder / f"{pilot.digest(source)}.json"
    receipt = _read(path)
    signature = pilot.digest(
        [module.VERSION, module.compiler.VERSION, source] if state else [module.base.VERSION, source]
    )
    terminal_attempts(path.with_suffix(".attempts.json"), receipt, signature)
    value = receipt["value"]
    if receipt["signature"] != signature or (
        state
        and (
            value["source_checksum"] != pilot.digest(source)
            or value["version"] != module.compiler.VERSION
            or value["input_state"] != "all_zero_only"
            or value["independent_audit"]["passed"] is not True
        )
    ):
        msg = "Compiler receipt or independent ideal-state audit changed."
        raise ValueError(msg)
    output_hash = pilot.digest(value["circuit"])
    if pilot.digest(_read(root / "circuits" / f"{output_hash}.json")) != output_hash:
        msg = "Compiler output circuit differs from its bank identity."
        raise ValueError(msg)
    result = {
        "theta": value["theta"],
        "circuit_hash": output_hash,
        "cpu_seconds": point["cpu_seconds"] + receipt["cpu_seconds"],
        "stage": role,
    }
    if state:
        result.update(
            state_compiler_cpu_seconds=receipt["cpu_seconds"],
            state_compiler_receipt=str(path.relative_to(root)),
            state_compiler_checksum=pilot.digest(receipt),
            removed_gates=value["input_gates"] - value["output_gates"],
        )
    else:
        result.update(
            compiled_final=True, compiler_cpu_seconds=receipt["cpu_seconds"], compiler_checksum=pilot.digest(receipt)
        )
    return result


def compiled_execution(
    root: Path,
    source: dict[str, Any],
    target_sha256: str,
    config: dict[str, Any],
    stochastic_config: dict[str, Any],
    seed: int,
    order: list[str],
) -> dict[str, Any]:
    """Authenticate all compact kernel fields through complete immutable child receipts."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.compiled_composition")
    result = _read(root / "EXECUTION.json")
    signature = pilot.digest({
        "version": module.VERSION,
        "source": source,
        "target_sha256": target_sha256,
        "exact_config": config,
        "stochastic_config": stochastic_config,
        "seed": seed,
        "order": order,
    })
    if (
        result["signature"] != signature
        or result["version"] != module.VERSION
        or result["dependencies"] != module.artifacts(root)
    ):
        msg = "Compiled execution signature or complete child inventory changed."
        raise ValueError(msg)
    terminal_attempts(root / "CASE_ATTEMPTS.json", result["physical_invocation"], signature)
    original = {
        "circuit_hash": pilot.digest(source["circuit"]),
        "theta": source["theta"],
        "cpu_seconds": 0.0,
        "stage": "generic_initializer",
    }
    initial = compiler_point(root, original, state=True, role="state_preparation_initializer")
    arms = {}
    for method in module.METHODS:
        path = root / "arms" / f"{method}.json"
        arm = _read(path)
        identity = pilot.digest([signature, method, initial])
        if (
            arm["signature"] != identity
            or any(
                name == "CASE_ATTEMPTS.json" or name.startswith("arms/") or result["dependencies"].get(name) != value
                for name, value in arm["dependencies"].items()
            )
            or pilot.digest(result["arms"].get(method)) != pilot.digest(module.compact_arm(arm, path, root))
        ):
            msg = "Compiled arm identity, compact fields or subset dependency map changed."
            raise ValueError(msg)
        if arm["status"] == "failed":
            failure = _read(root / "failures" / f"{method}.json")
            if failure != {"status": "failed", "failure": arm["failure"], "signature": identity}:
                msg = "Compiled failed arm differs from its retained failure receipt."
                raise ValueError(msg)
            arms[method] = arm
            continue
        if arm["status"] != "complete":
            msg = "Compiled arm is nonterminal."
            raise ValueError(msg)
        curve = arm["curve"]
        training_cpu = 0.0
        if method == "unchanged":
            raw = initial
            valid_curve = len(curve) == 4 and curve[:2] == [original, initial]
        else:
            receipt = _read(root / "training_results" / f"{method}.json")
            trained = receipt["training"]
            last = trained.get("state", trained)
            training_cpu = trained["cpu_seconds"]
            raw = {
                "theta": last["theta"],
                "circuit_hash": last["circuit_hash"],
                "cpu_seconds": initial["cpu_seconds"] + training_cpu,
                "stage": "raw_final",
            }
            valid_curve = (
                arm["training_receipt"] == f"training_results/{method}.json"
                and arm["training_checksum"] == pilot.digest(receipt)
                and len(curve) >= 5
                and curve[:2] == [original, initial]
                and curve[-3] == raw
                and receipt["raw_final"] == raw
                and pilot.digest(curve[2:-3]) == receipt["shifted_curve_checksum"]
            )
        generic = compiler_point(root, raw, state=False, role="generic_final")
        final = compiler_point(root, generic, state=True, role="state_preparation_final")
        expected = {
            "final": final,
            "initializer_cpu_seconds": source["initializer_cpu_seconds"],
            "initial_state_compiler_cpu_seconds": initial["state_compiler_cpu_seconds"],
            "training_cpu_seconds": training_cpu,
            "generic_final_compiler_cpu_seconds": generic["compiler_cpu_seconds"],
            "state_final_compiler_cpu_seconds": final["state_compiler_cpu_seconds"],
            "refinement_cpu_seconds": final["cpu_seconds"],
            "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
            "scores": point_score(root, final, target_sha256),
        }
        if (
            not valid_curve
            or curve[-2:] != [generic, final]
            or any(pilot.digest(arm.get(key)) != pilot.digest(value) for key, value in expected.items())
        ):
            msg = "Compiled arm curve, endpoint, physical score or charged cost differs from child receipts."
            raise ValueError(msg)
        arms[method] = arm
    expected_execution: dict[str, Any] = {
        "status": "complete" if all(a["status"] == "complete" for a in arms.values()) else "partial_failure",
        "source_checksum": pilot.digest(source),
        "initial": initial,
        "initializer_cpu_seconds": source["initializer_cpu_seconds"],
        "physical_initial_compiler_receipt": initial["state_compiler_receipt"],
        "initial_compiler_scope": "executed once physically, charged to every declared procedure",
    }
    for panel, methods in (("common_primary", module.PRIMARY), ("common_all_trained", module.TRAINED)):
        if any(arm["status"] != "complete" for arm in arms.values()):
            expected_panel = {"status": "unavailable_failed_arm", "cutoff_methods": list(methods), "arms": {}}
        else:
            cutoff = min(arms[method]["refinement_cpu_seconds"] for method in methods)
            points = module.common_control_audit.select_points(arms, module.METHODS, cutoff)
            expected_panel = {
                "status": "complete",
                "cutoff_methods": list(methods),
                "methods": list(module.METHODS),
                "refinement_cpu_seconds": cutoff,
                "total_cpu_seconds": source["initializer_cpu_seconds"] + cutoff,
                "arms": {
                    method: {"point": point, "scores": point_score(root, point, target_sha256)}
                    for method, point in points.items()
                },
            }
        expected_execution[panel] = expected_panel
    if set(result["arms"]) != set(module.METHODS) or any(
        pilot.digest(result.get(key)) != pilot.digest(value) for key, value in expected_execution.items()
    ):
        msg = "Compiled execution status, selected points, all-role scores or clocks changed."
        raise ValueError(msg)
    return result


def compiled_records(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return all complete or explicitly failed development cases with source custody."""
    module = importlib.import_module(
        "experiments.overnight_state_preparation_20260913_0108.compiled_composition_experiment"
    )
    composed_inputs(protocol)
    fixed = {
        "version": module.VERSION,
        "baseline": module.kernel.BASELINE,
        "candidate": module.kernel.CANDIDATE,
        "seed_namespace": "compiled_composition_training",
        "primary_cutoff": "minimum four structural final refinement CPU; score all seven at same cutoff",
        "promotion": {"mean_delta_pp": 0.5, "target_wins": 4, "fixed_and_strongest_controls_required": True},
    }
    if any(pilot.digest(protocol.get(key)) != pilot.digest(value) for key, value in fixed.items()):
        msg = "Compiled development configuration or fixed promotion rule changed."
        raise ValueError(msg)
    records = []
    for source in protocol["starts"]:
        folder = root / f"{source['target_id']}__11"
        record = _read(folder / "RESULT.json")
        signature = pilot.digest([protocol, source])
        if record["status"] == "failed":
            failure = _read(folder / "CASE_FAILURE.json")
            expected = {
                **failure,
                "case_failure_checksum": pilot.digest(failure),
                "dependencies": module.kernel.artifacts(folder),
            }
            if (
                record != expected
                or record["signature"] != signature
                or record["target_id"] != source["target_id"]
                or record["replica"] != 11
            ):
                msg = "Compiled failed case or retained artifact inventory changed."
                raise ValueError(msg)
        else:
            index = list(module.TARGETS).index(source["target_id"]) % len(module.kernel.TRAINED)
            order = [*module.kernel.TRAINED[index:], *module.kernel.TRAINED[:index]]
            execution = compiled_execution(
                folder,
                source["endpoint"],
                target_hash(source["target_id"]),
                protocol["exact_config"],
                protocol["stochastic_config"],
                runner.seed_for(protocol["seed_namespace"], source["target_id"], 11),
                order,
            )
            expected = {
                **execution,
                "execution_signature": execution["signature"],
                "execution_checksum": pilot.digest(execution),
                "signature": signature,
                "protocol_checksum": pilot.digest(protocol),
                "target_id": source["target_id"],
                "replica": 11,
                "version": module.VERSION,
            }
            if pilot.digest(record) != pilot.digest(expected):
                msg = "Compiled outer case differs from its compact execution receipt."
                raise ValueError(msg)
        records.append(record)
    summary = _read(root / "SUMMARY.json")
    if summary["protocol_checksum"] != pilot.digest(protocol) or summary["case_checksums"] != {
        record["target_id"]: pilot.digest(record) for record in records
    }:
        msg = "Compiled SUMMARY does not cover the complete distinct case panel."
        raise ValueError(msg)
    return records, summary


def best_parent(root: Path, parent: dict[str, Any]) -> None:
    """Authenticate a prepared complete parent without loading any target vector."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
    design = _read(root / "DESIGN.json")
    engineering = design["phase"] == "engineering"
    exact = (
        module.base.exact.Config(**design["algorithm"]["training"]["exact"])
        if engineering
        else module.base.exact.Config(protected_initial_gates=0)
    )
    sampled = (
        module.base.stochastic.Config(**design["algorithm"]["training"]["stochastic"])
        if engineering
        else module.base.stochastic.Config(protected=0)
    )
    expected_config = module.configuration(design["recipe"], exact, sampled, design["targets"], design["replicas"])
    if (
        design["version"] != module.VERSION
        or design["status"] != "frozen"
        or design["phase"] not in {"confirmation", "engineering"}
        or (
            not engineering
            and (design["targets"] != list(module.TARGETS) or design["replicas"] != list(module.REPLICAS))
        )
        or (engineering and any(not t.startswith("synthetic_") for t in design["targets"]))
        or any(pilot.digest(design.get(k)) != pilot.digest(v) for k, v in expected_config.items())
        or design["configuration_sha256"] != pilot.digest(expected_config)
        or any(
            not Path(name).is_file() or file_sha256(Path(name)) != value
            for name, value in design["development_evidence"].items()
        )
    ):
        msg = "Confirmation design, complete configuration or development evidence changed."
        raise ValueError(msg)
    bound = module.read_phase(root / "preparation", [module.VERSION, pilot.digest(design), "all_case_hash_binding"])
    expected_parent = {
        **design,
        "schema": module.constructor.PARENT_SCHEMA,
        "cases": bound["value"]["cases"],
        "design_checksum": pilot.digest(design),
        "preparation_checksum": pilot.digest(bound),
        "preparation_cpu_seconds": bound["cpu_seconds"],
        "preparation_wall_seconds": bound["wall_seconds"],
    }
    if pilot.digest(parent) != pilot.digest(expected_parent):
        msg = "Confirmation parent differs from its sealed pre-execution design and complete case binding."
        raise ValueError(msg)
    expected_keys = {(t, r) for t in parent["targets"] for r in parent["replicas"]}
    if (
        len(parent["cases"]) != len(expected_keys)
        or {(c["target_id"], c["replica"]) for c in parent["cases"]} != expected_keys
    ):
        msg = "Confirmation case identities do not cover the complete paired panel."
        raise ValueError(msg)
    for index, case in enumerate(parent["cases"]):
        expected = {
            "parent_checksum": pilot.digest(parent),
            "case": case,
            "index": index,
            "training_seed": parent["seed_policy"]["case_seeds"][f"{case['target_id']}__{case['replica']}"],
        }
        if (
            _read(root / f"{case['target_id']}__{case['replica']}" / "CASE.json") != expected
            or case["num_qubits"] not in ({1, 2, 3} if engineering else {6})
            or len(case["target_vector_sha256"]) != 64
        ):
            msg = "Confirmation child manifest, target digest or training seed changed."
            raise ValueError(msg)


def best_panels(root: Path, parent: dict[str, Any], case: dict[str, Any], record: dict[str, Any]) -> None:
    """Bind all confirmation endpoints and costs to stored gates, arms and score receipts."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
    loaded = module.read_phase(root / "input", [pilot.digest(parent), case, "input"])
    initialization = module.read_phase(root / "initializer", [pilot.digest(parent), case, "initializer"])
    source = {**initialization["value"]["endpoint"], "initializer_cpu_seconds": record["initializer_cpu_seconds"]}
    cost = record["phase_costs"]
    if (
        cost["input"]
        != {key: loaded[key] for key in ("cpu_seconds", "wall_seconds")} | {"checksum": pilot.digest(loaded)}
        or cost["initializer"]["checksum"] != pilot.digest(initialization)
        or cost["initializer"]["cpu_seconds"] != initialization["cpu_seconds"]
        or cost["initializer"]["wall_seconds"] != initialization["wall_seconds"]
        or record["initializer"]
        != {
            key: initialization["value"][key]
            for key in ("initializer_checksum", "rank_two_selected", "ideal_tt_overlap", "nested_legacy_cpu_seconds")
        }
        or source["initializer_cpu_seconds"]
        != cost["input"]["cpu_seconds"]
        + initialization["cpu_seconds"]
        + parent["preparation_cpu_seconds"] / len(parent["cases"])
        or record["amortized_preparation_cpu_seconds"] != parent["preparation_cpu_seconds"] / len(parent["cases"])
    ):
        msg = "Confirmation initializer or amortized preparation cost changed."
        raise ValueError(msg)
    trained = module.read_phase(root / "training", [pilot.digest(parent), case, source, "all_roles"])
    if (
        cost["training_and_evaluation"]["checksum"] != pilot.digest(trained)
        or cost["training_and_evaluation"]["cpu_seconds"] != trained["cpu_seconds"]
        or cost["training_and_evaluation"]["wall_seconds"] != trained["wall_seconds"]
        or set(record["arm_receipts"]) != set(module.TRAINED)
        or any(pilot.digest(record.get(key)) != pilot.digest(value) for key, value in trained["value"].items())
    ):
        msg = "Confirmation training view differs from its sealed complete phase."
        raise ValueError(msg)
    work = root / "training/work"
    config = parent["algorithm"]["training"]
    seed = parent["seed_policy"]["case_seeds"][f"{case['target_id']}__{case['replica']}"]
    target_sha = case["target_vector_sha256"]
    if parent["recipe"] == "state_preparation_compiled":
        offset = parent["cases"].index(case) % len(module.TRAINED)
        compiled_execution(
            work,
            source,
            target_sha,
            config["exact"],
            config["stochastic"],
            seed,
            [*module.TRAINED[offset:], *module.TRAINED[:offset]],
        )
    arms: dict[str, Any] = {}
    for name in module.TRAINED:
        path = work / "arms" / f"{name}.json"
        arm = _read(path)
        if arm["status"] != "complete" or any(
            not (work / p).is_file() or file_sha256(work / p) != h for p, h in arm["dependencies"].items()
        ):
            msg = "Confirmation arm or its physical dependencies changed."
            raise ValueError(msg)
        if parent["recipe"] == "original":
            signature = pilot.digest([
                module.base.VERSION,
                source,
                target_sha,
                config["exact"],
                config["stochastic"],
                seed,
                name,
            ])
            raw = arm["raw_training"]
            last = raw.get("state", raw)
            point = {"theta": last["theta"], "circuit_hash": last["circuit_hash"], "cpu_seconds": raw["cpu_seconds"]}
            final = compiler_point(work, point, state=False, role="generic_final")
            final.pop("stage")
            initial = {
                "circuit_hash": pilot.digest(source["circuit"]),
                "theta": source["theta"],
                "cpu_seconds": 0.0,
                "initial": True,
            }
            expected = {
                "signature": signature,
                "final": final,
                "curve": [initial, *raw["curve"], point, final],
                "initializer_cpu_seconds": source["initializer_cpu_seconds"],
                "training_cpu_seconds": raw["cpu_seconds"],
                "refinement_cpu_seconds": final["cpu_seconds"],
                "total_cpu_seconds": source["initializer_cpu_seconds"] + final["cpu_seconds"],
                "scores": point_score(work, final, target_sha),
            }
            if any(pilot.digest(arm.get(k)) != pilot.digest(v) for k, v in expected.items()):
                msg = "Original confirmation arm identity, compiler endpoint, score or clock changed."
                raise ValueError(msg)
        receipt = {
            "path": f"arms/{name}.json",
            "checksum": pilot.digest(arm),
            "byte_sha256": file_sha256(path),
            "training_cpu_seconds": arm.get("training_cpu_seconds"),
            "refinement_cpu_seconds": arm["refinement_cpu_seconds"],
        }
        if record["arm_receipts"].get(name) != receipt:
            msg = "Confirmation compact arm receipt changed."
            raise ValueError(msg)
        arms[name] = arm
    if parent["recipe"] == "original":
        initial = {
            "circuit_hash": pilot.digest(source["circuit"]),
            "theta": source["theta"],
            "cpu_seconds": 0.0,
            "initial": True,
        }
        arms["unchanged"] = {
            "status": "complete",
            "final": initial,
            "curve": [initial],
            "scores": point_score(work, initial, target_sha),
            "refinement_cpu_seconds": 0.0,
        }
    else:
        arms["unchanged"] = _read(work / "arms/unchanged.json")
    full = {
        "status": "complete",
        "arms": {
            name: {
                "status": arm["status"],
                "point": arm["final"],
                "scores": module.compact_score(point_score(work, arm["final"], target_sha)),
                "refinement_cpu_seconds": arm["refinement_cpu_seconds"],
                "total_cpu_seconds": source["initializer_cpu_seconds"] + arm["refinement_cpu_seconds"],
            }
            for name, arm in arms.items()
        },
    }
    panels: dict[str, Any] = {"full_cap": full}
    for panel, methods in (("primary", module.PRIMARY), ("all_trained_sensitivity", module.TRAINED)):
        cutoff = min(arms[name]["refinement_cpu_seconds"] for name in methods)
        selected = module.common_control_audit.select_points(arms, module.METHODS, cutoff)
        panels[panel] = {
            "status": "complete",
            "cutoff_roles": list(methods),
            "refinement_cpu_seconds": cutoff,
            "total_cpu_seconds": source["initializer_cpu_seconds"] + cutoff,
            "arms": {
                name: {"point": point, "scores": module.compact_score(point_score(work, point, target_sha))}
                for name, point in selected.items()
            },
        }
    if pilot.digest(record["panels"]) != pilot.digest(panels):
        msg = "Confirmation selected points, gate resources or all12 score panels changed."
        raise ValueError(msg)


def best_rows_summary(
    parent: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return canonical confirmation cells and complete-panel analysis without filesystem writes."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
    rows = []
    for record in records:
        for panel in ("primary", "all_trained_sensitivity", "full_cap"):
            selected = record.get("panels", {}).get(panel, {})
            for method in module.METHODS:
                arm = selected.get("arms", {}).get(method, {})
                score = arm.get("scores", {})
                for noise in channels.NOISE_IDS:
                    counts = score.get("ballarin_counts" if noise == "ballarin_coupled" else "physical_counts", {})
                    rows.append({
                        "target_id": record["target_id"],
                        "replica": record["replica"],
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
    analysis = module.analyze(records, parent["targets"], parent["replicas"])
    summary = {
        "version": module.VERSION,
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
    return rows, summary


def best_records(root: Path, parent: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve complete, explicitly failed and still-pending cases without target access."""
    module = importlib.import_module("experiments.overnight_state_preparation_20260913_0108.best_candidate")
    best_parent(root, parent)
    records = []
    for case in parent["cases"]:
        child = root / f"{case['target_id']}__{case['replica']}"
        if (child / "RESULT.json").exists():
            physical = module.read_phase(child / "execution", [pilot.digest(parent), case, "complete_physical_case"])
            record = _read(child / "RESULT.json")
            if pilot.digest(record) != pilot.digest(module.case_record(parent, case, physical)):
                msg = "Confirmation published case differs from its sealed physical invocation."
                raise ValueError(msg)
            best_panels(child / "execution", parent, case, record)
        elif (child / "FAILURE.json").exists():
            record = _read(child / "FAILURE.json")
            if record["status"] != "failed" or (record["target_id"], record["replica"]) != (
                case["target_id"],
                case["replica"],
            ):
                msg = "Confirmation failure identity changed."
                raise ValueError(msg)
            ledger = child / "execution/ATTEMPTS.json"
            saved = _read(ledger) if ledger.exists() else None
            attempts = saved["attempts"] if saved is not None else []
            if saved is not None and (
                saved["signature"] != pilot.digest([pilot.digest(parent), case, "complete_physical_case"])
                or any(
                    a["status"] not in {"running", "complete", "failed", "interrupted"}
                    or any(
                        a.get(key) is not None
                        and (
                            isinstance(a[key], bool)
                            or not isinstance(a[key], (int, float))
                            or not math.isfinite(a[key])
                            or a[key] < 0
                        )
                        for key in ("cpu_seconds", "wall_seconds")
                    )
                    for a in attempts
                )
            ):
                msg = "Confirmation failure attempt identity, status or measured duration is invalid."
                raise ValueError(msg)
            known = bool(attempts) and all(
                a["status"] != "running" and a.get("cpu_seconds") is not None and a.get("wall_seconds") is not None
                for a in attempts
            )
            if (
                record["timing_complete"] != known
                or record["physical_cpu_seconds"] != (sum(a["cpu_seconds"] for a in attempts) if known else None)
                or record["physical_wall_seconds"] != (sum(a["wall_seconds"] for a in attempts) if known else None)
            ):
                msg = "Confirmation failed-work accounting differs from its attempts."
                raise ValueError(msg)
        else:
            record = {"status": "pending", "target_id": case["target_id"], "replica": case["replica"]}
        records.append(record)
    summary = _read(root / "SUMMARY.json")
    if pilot.digest(summary) != pilot.digest(best_rows_summary(parent, records)[1]):
        msg = "Confirmation SUMMARY gate, missing-case inventory or total resource accounting changed."
        raise ValueError(msg)
    return records, summary


def _records(
    root: Path, protocol: dict[str, Any], kind: str, expected: dict[tuple[str, str], set[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate every expected run/case and its top-level completion ledger.

    Raises:
        ValueError: If identity, signature, summary links or completion differ.
    """
    if kind == "initializer":
        return initializer_records(root, protocol, expected)
    if kind == "composed":
        return composed_records(root, protocol)
    if kind == "state_compilation":
        return state_compilation_records(root, protocol)
    if kind == "common_control":
        return common_control_records(root, protocol)
    if kind == "compiled_composition":
        return compiled_records(root, protocol)
    if kind == "best_candidate":
        return best_records(root, protocol)
    summary = _read(root / "RESULT.json")
    protocol_key = "runtime_checksum" if kind == "runner" else "protocol_checksum"
    statuses = {
        "complete",
        "partial_failure",
        "contains_failures",
        "complete_with_failed_arms",
        "complete_with_failures",
    }
    if summary.get("status") not in statuses or summary.get(protocol_key) != pilot.digest(protocol):
        msg = "Panel RESULT is not terminal or does not authenticate its frozen protocol."
        raise ValueError(msg)
    result: list[dict[str, Any]] = []
    sources = _case_sources(protocol)
    compilation_inputs = compilation_sources(protocol) if kind == "compilation" else {}
    mixture_inputs = mixture_references(protocol) if kind == "mixture_baseline" else {}
    for (target, replica), methods in sorted(expected.items()):
        if kind == "compilation":
            for label in protocol["labels"]:
                source = compilation_inputs[target, replica, label]
                record = _read(root / f"{target}__{replica}__{label}.json")
                if (
                    record["signature"] != pilot.digest([pilot.digest(protocol), source])
                    or (record["target_id"], str(record["replica"]), record["source_label"]) != (target, replica, label)
                    or record["source_checksum"] != pilot.digest(source)
                ):
                    msg = "Compilation source identity or signature mismatch."
                    raise ValueError(msg)
                result.append(record)
            continue
        if kind in {"runner", "svrg"}:
            for method in sorted(methods):
                path = (
                    root / f"{target}__{replica}__{method}.json"
                    if kind == "runner"
                    else _terminal_path(root / f"{target}__{replica}__{method}")
                )
                record = _read(path)
                if kind == "runner":
                    job = record["job"]
                    signature = pilot.digest({key: value for key, value in job.items() if key != "path"})
                    valid = (
                        (job["target_id"], str(job["replica"]), job["label"]) == (target, replica, method)
                        and job["config"] == protocol["protocol"]["methods"][method]
                        and job.get("runtime_checksum") == pilot.digest(protocol)
                        and job.get("source_hash") == pilot.digest(protocol["source_hashes"])
                    )
                else:
                    signature = pilot.digest({
                        "protocol": pilot.digest(protocol),
                        "source": pilot.digest(sources[target, replica]),
                        "method": method,
                    })
                    valid = (record["target_id"], str(record["replica"]), record["method"]) == (target, replica, method)
                if not valid or record["signature"] != signature:
                    msg = "Run identity or frozen-protocol signature mismatch."
                    raise ValueError(msg)
                result.append(record)
        else:
            folder = root / f"{target}__{replica}"
            record = _read(_terminal_path(folder) if kind == "structural" else folder / "RESULT.json")
            if kind == "synthesis":
                identity = {"protocol": protocol, "target_id": target, "replica": int(replica)}
            else:
                identity = {"protocol": protocol, "source": sources[target, replica]}
                if kind in {"ensemble", "geometry"}:
                    identity["order_index"] = protocol["targets"].index(target)
            expected_signature = (
                pilot.digest([protocol, sources[target, replica]]) if kind == "baseline" else pilot.digest(identity)
            )
            if kind in {"ensemble_structural", "risk"}:
                expected_signature = ensemble_structural_signature(protocol, sources[target, replica])
            if kind == "long_ensemble":
                expected_signature = pilot.digest({
                    "version": protocol["version"],
                    "source": sources[target, replica],
                    "config": protocol["config"],
                    "target": target_hash(target),
                    "order_index": protocol["targets"].index(target),
                    "protocol_checksum": pilot.digest(protocol),
                })
                verify_long_ensemble_case(folder, record)
            if kind == "mixture_baseline":
                source = sources[target, replica]
                reference = mixture_inputs[target, replica]
                expected_signature = pilot.digest([
                    protocol["version"],
                    source["candidate"]["circuit"],
                    source["candidate"]["theta"],
                    target_hash(target),
                    protocol["config"],
                    target,
                    source["replica"],
                    pilot.digest(protocol),
                    reference["checksum"],
                ])
                verify_mixture_case(folder, record, reference)
            if record["signature"] != expected_signature or (record["target_id"], str(record["replica"])) != (
                target,
                replica,
            ):
                msg = "Case identity or frozen-protocol signature mismatch."
                raise ValueError(msg)
            result.append(record)
    if kind == "baseline":
        references = baseline_references(protocol)
        for record in result:
            reference = references[record["target_id"], str(record["replica"])]
            if record["reference_case_checksum"] != reference["checksum"]:
                msg = "Baseline result points to a changed structural reference."
                raise ValueError(msg)
    checksum_key = (
        "cases"
        if kind in {"baseline", "compilation", "mixture_baseline"}
        else "runs"
        if kind in {"runner", "svrg"}
        else "results"
    )
    checksums = summary.get(checksum_key, [])
    if sorted(checksums) != sorted(pilot.digest(record) for record in result):
        msg = "Panel RESULT checksums do not exactly cover every expected run or case."
        raise ValueError(msg)
    if any(record.get("status") not in statuses | {"failed"} for record in result):
        msg = "Panel contains a nonterminal run or case."
        raise ValueError(msg)
    return result, summary


def normalize_strata(rows: list[dict[str, str]], kind: str) -> list[dict[str, str]]:
    """Keep full-budget, common-CPU and historical cells as separate identities.

    Raises:
        ValueError: If a baseline row has an unknown comparison stratum.
    """
    if kind == "compilation":
        return [{**row, "method": f"{row['source']}__{row['arm']}"} for row in rows]
    if kind == "state_compilation":
        if any(row.get("endpoint") not in {"before", "after"} for row in rows):
            msg = "Unknown state compiler endpoint identity."
            raise ValueError(msg)
        return [{**row, "method": f"{row['label']}__{row['endpoint']}"} for row in rows]
    if kind not in {
        "baseline",
        "mixture_baseline",
        "long_ensemble",
        "composed",
        "common_control",
        "compiled_composition",
        "best_candidate",
    }:
        return rows
    result = []
    for row in rows:
        stratum = row.get("panel")
        allowed = {"full_cap", "common_cpu", "historical_unmatched"}
        if kind == "long_ensemble":
            allowed.remove("historical_unmatched")
        if kind in {"composed", "compiled_composition"}:
            allowed = {"full_cap", "common_primary", "common_all_trained"}
        if kind == "best_candidate":
            allowed = {"full_cap", "primary", "all_trained_sensitivity"}
        if kind == "common_control":
            allowed = {"common_primary"}
        if kind == "mixture_baseline":
            allowed.add("unchanged")
        if stratum not in allowed:
            msg = "Unknown baseline comparison stratum."
            raise ValueError(msg)
        copied = dict(row)
        copied.pop("panel")
        copied.update(source_method=row["method"], comparison_panel=stratum, method=f"{row['method']} [{stratum}]")
        result.append(copied)
    return result


def canonical_rows(
    kind: str,
    records: list[dict[str, Any]],
    *,
    context_root: Path | None = None,
    protocol: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Regenerate source-derived cells, retaining authenticated auxiliary inputs.

    Raises:
        ValueError: If mixture regeneration lacks its required source context.
    """
    module_name = "state_preparation_compilation_experiment" if kind == "state_compilation" else kind + "_experiment"
    module = (
        runner
        if kind in {"runner", "common_control", "best_candidate"}
        else importlib.import_module("experiments.overnight_state_preparation_20260913_0108." + module_name)
    )
    with tempfile.TemporaryDirectory(prefix="yaqs-report-verify-") as directory:
        root = Path(directory)
        if kind == "mixture_baseline":
            if context_root is None:
                msg = "Mixture canonical writer requires authenticated historical context."
                raise ValueError(msg)
            for record in records:
                relative = Path(f"{record['target_id']}__{record['replica']}") / "REFERENCES.json"
                (root / relative).parent.mkdir(parents=True)
                shutil.copyfile(context_root / relative, root / relative)
        if kind in {"common_control", "best_candidate"}:
            if protocol is None:
                msg = "Common-control canonical writer requires its frozen protocol."
                raise ValueError(msg)
            if kind == "best_candidate":
                rows, regenerated = best_rows_summary(protocol, records)
                if pilot.digest(regenerated) != pilot.digest(summary):
                    msg = "Confirmation canonical summary differs from authenticated cases."
                    raise ValueError(msg)
            else:
                rows = common_control_rows(protocol, records)
            with (root / "SCOREBOARD.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        elif kind in {"initializer", "composed", "state_compilation", "compiled_composition"}:
            if protocol is None or summary is None:
                msg = "Initializer/composition canonical writer requires its frozen protocol and summary."
                raise ValueError(msg)
            if kind == "compiled_composition":
                module.summarize(root, protocol, records)
            else:
                module.write_summary(root, protocol, records)
            regenerated = _read(root / "SUMMARY.json")
            if pilot.digest(regenerated) != pilot.digest(summary):
                msg = "Initializer/composition SUMMARY aggregates or selection fields differ from authenticated cases."
                raise ValueError(msg)
        elif kind == "runner":
            module.scoreboard(records, root)
        else:
            module.write_scoreboard(root, records)
        filename = "scoreboard.csv" if kind == "state_compilation" else "SCOREBOARD.csv"
        with (root / filename).open(newline="") as stream:
            return normalize_strata(list(csv.DictReader(stream)), kind)


def _cell(row: dict[str, str]) -> Cell:
    """Return the complete target/replica/method/noise cell identity."""
    return row["target_id"], row["replica"], row["method"], row["noise_id"]


def authenticate_panel(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Verify source provenance, terminal results and every supplied CSV cell.

    Missing CSV cells are materialized as incomplete, even when their underlying
    result exists. They cannot silently shrink the panel or its method list.

    Raises:
        ValueError: If a row contradicts its authenticated numerical record.
    """
    protocol = _read(path.parent / "PROTOCOL.json")
    kind = _kind(protocol)
    _verify_sources(protocol, kind)
    expected = _expected(protocol, kind)
    records, summary = _records(path.parent, protocol, kind, expected)
    canonical = {
        _cell(row): row
        for row in canonical_rows(kind, records, context_root=path.parent, protocol=protocol, summary=summary)
    }
    expected_cells = {
        (target, replica, method, noise)
        for (target, replica), methods in expected.items()
        for method in methods
        for noise in channels.NOISE_IDS
    }
    if set(canonical) - expected_cells:
        msg = "Authenticated writer contains methods/cases outside the frozen protocol."
        raise ValueError(msg)
    with path.open(encoding="utf-8", newline="") as stream:
        raw = normalize_strata(list(csv.DictReader(stream)), kind)
    observed = set()
    rows = []
    for row in raw:
        _validate_tags(row)
        cell = _cell(row)
        if cell in observed or cell not in expected_cells or cell not in canonical:
            msg = "Duplicate, unexpected or unauthenticated scoreboard cell."
            raise ValueError(msg)
        observed.add(cell)
        known = canonical[cell]
        for key, value in known.items():
            if key in {"convention", "exactness"} and not row.get(key):
                continue
            if row.get(key) != value:
                msg = f"CSV cell differs from its authenticated result: {cell}, field {key}."
                raise ValueError(msg)
        rows.append({**row, "comparison": known.get("comparison", "resource_comparison_defined_by_frozen_protocol")})
    for target, replica, method, noise in sorted(expected_cells - observed):
        rows.append({
            "target_id": target,
            "replica": replica,
            "method": method,
            "noise_id": noise,
            "status": "missing",
            "fidelity": "",
            "failure": "missing_expected_csv_cell",
            "comparison": canonical.get((target, replica, method, noise), {}).get("comparison", "not_evaluated"),
        })
    for row in rows:
        _validate_tags(row)
        row["verified_convention"] = "calibrated_exact"
        row["verified_exactness_source"] = kind + ":frozen_sources_and_checksummed_results"
        row["protocol_checksum"] = pilot.digest(protocol)
    if kind == "compiled_composition":
        case_lookup = {(record["target_id"], str(record["replica"])): record for record in records}
        for row in rows:
            for key in ("one_qubit", "two_qubit", "depth"):
                row["source_" + key] = row.get(key, "")
            row["source_count_scope"] = "physical circuit before profile-specific lowering"
            row["count_scope"] = "per-condition materialized physical circuit"
            if row["status"] != "complete":
                continue
            record = case_lookup[row["target_id"], row["replica"]]
            panel = row["comparison_panel"]
            arm = (record["arms"] if panel == "full_cap" else record[panel]["arms"])[row["source_method"]]
            point = arm["final"] if panel == "full_cap" else arm["point"]
            folder = path.parent / f"{row['target_id']}__{row['replica']}"
            resources = endpoint_resources(_read(folder / "circuits" / f"{point['circuit_hash']}.json"), point["theta"])
            counts = resources["ballarin_counts" if row["noise_id"] == "ballarin_coupled" else "physical_counts"]
            row.update({key: str(value) for key, value in counts.items()})
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol_path": str((path.parent / "PROTOCOL.json").resolve()),
        "protocol_checksum": pilot.digest(protocol),
        "result_checksum": pilot.digest(summary),
        "result_status": summary["status"],
        "schema_kind": kind,
        "verified_source_files": len(protocol.get("source_hashes", protocol.get("sources", {}))),
        "expected_cells": len(expected_cells),
        "missing_csv_cells": len(expected_cells - observed),
        "authenticated_runs_or_cases": len(records),
        "verification": "all source bytes, expected identities, result checksums and canonical CSV scores/resources",
    }
    if kind == "initializer":
        metadata["common_cpu_availability"] = initializer_availability(records)
        metadata["csv_endpoint_scope"] = "original full endpoints only; common CPU scores are not invented as CSV rows"
    if kind == "composed":
        metadata["physical_invocations"] = {record["target_id"]: record["physical_invocation"] for record in records}
        metadata["physical_cpu_aggregation"] = (
            "inclusive invocation clocks retained; nested numerical clocks are not added"
        )
    if kind == "state_compilation":
        metadata["physical_invocations"] = {
            f"{record['target_id']}__11/{record['label']}": record["physical_invocation"] for record in records
        }
        metadata["missing_or_failed_pairs"] = summary["missing_pairs"]
        metadata["physical_work"] = {
            f"{record['target_id']}__11/{record['label']}": {
                key: record.get(key)
                for key in (
                    "accounting",
                    "compiler_attempts",
                    "compiler_completed_attempts",
                    "failed_compiler_gate_work",
                    "scoring",
                )
            }
            for record in records
        }
        metadata["cpu_scope"] = protocol["cpu_scope"]
        metadata["physical_cpu_aggregation"] = (
            "inclusive pair invocation clocks retained once per pair; nested compiler and score clocks are not added"
        )
    if kind == "common_control":
        metadata["physical_invocations"] = {
            record["value"]["target_id"]: {
                key: record[key] for key in ("cpu_seconds", "wall_seconds", "attempts_checksum")
            }
            for record in records
        }
        metadata["strongest_control"] = summary["strongest_control"]
        metadata["fresh_score_operations"] = {
            record["value"]["target_id"]: {
                score["evaluation"]["signature"]: score["evaluation"]
                for method, score in record["value"]["scores"].items()
                if method in {"fixed_mean2_lbfgs", "fixed_noiseless_adam"}
            }
            for record in records
        }
        metadata["physical_cpu_aggregation"] = (
            "inclusive case audit clocks retained once per case; "
            "nested scoring clocks and copied source work are not added"
        )
        metadata["endpoint_clock"] = (
            "unchanged structural primary refinement cutoff plus the shared measured initializer; "
            "audit scoring excluded"
        )
    if kind == "compiled_composition":
        metadata["physical_invocations"] = {
            record["target_id"]: record.get("physical_invocation") for record in records
        }
        metadata["source_gate_count_scope"] = "source CSV repeats pre-profile physical counts even for Ballarin"
        metadata["canonical_gate_count_scope"] = (
            "independent profile materialization; original counts kept in source_* columns"
        )
        metadata["physical_cpu_aggregation"] = (
            "inclusive kernel invocation once per case; nested compiler/training clocks are not added"
        )
    if kind == "best_candidate":
        metadata["physical_invocations"] = {
            f"{record['target_id']}__{record['replica']}": record.get("physical_invocation") for record in records
        }
        metadata["preparation_cpu_seconds"] = protocol["preparation_cpu_seconds"]
        metadata["actual_complete_experiment_cpu_seconds"] = summary["actual_complete_experiment_cpu_seconds"]
        metadata["confirmation_analysis"] = summary["analysis"]
        metadata["case_completion"] = summary["cases"]
        metadata["failed_case_costs"] = {
            f"{record['target_id']}__{record['replica']}": {
                key: record.get(key)
                for key in ("failure", "physical_cpu_seconds", "physical_wall_seconds", "timing_complete")
            }
            for record in records
            if record["status"] == "failed"
        }
        metadata["physical_cpu_aggregation"] = (
            "preparation once plus inclusive physical case invocations once; nested phase and arm clocks are not added"
        )
        metadata["peak_rss"] = {
            f"{record['target_id']}__{record['replica']}": record.get("peak_rss") for record in records
        }
    return rows, metadata


def family(target: str) -> str:
    """Map immutable target identities to the four benchmark families."""
    for prefix in ("gaussian", "tfim", "haar_random", "random_mps"):
        if target.startswith(prefix):
            return prefix
    return target


def balanced(values: dict[tuple[str, str], float]) -> dict[str, float]:
    """Average replicas first, retaining a separate mean for every target."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for (target, _), value in values.items():
        grouped[target].append(value)
    return {target: statistics.mean(group) for target, group in sorted(grouped.items())}


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize one panel and every paired method contrast without exclusions."""
    grouped: dict[str, dict[tuple[str, str], dict[str, dict[str, str]]]] = {}
    all_cases = set()
    for row in rows:
        _validate_tags(row)
        if row["noise_id"] not in channels.NOISE_IDS:
            msg = f"Unknown benchmark noise condition: {row['noise_id']}."
            raise ValueError(msg)
        key = row["target_id"], row["replica"]
        all_cases.add(key)
        case = grouped.setdefault(row["method"], {}).setdefault(key, {})
        if row["noise_id"] in case:
            msg = "Duplicate target/replica/method/noise scoreboard cell."
            raise ValueError(msg)
        case[row["noise_id"]] = row
    summaries: dict[str, Any] = {}
    means: dict[str, dict[tuple[str, str], float]] = {}
    for method, cases in sorted(grouped.items()):
        issues = []
        scores = {}
        condition_values: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        for key in sorted(all_cases):
            case = cases.get(key, {})
            if set(case) != set(channels.NOISE_IDS):
                issues.append({"case": key, "reason": "missing noise conditions or case"})
                continue
            if any(row["status"] != "complete" for row in case.values()):
                issues.append({"case": key, "reason": "failed or incomplete run"})
                continue
            values = {noise: float(row["fidelity"]) for noise, row in case.items()}
            if any(not math.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10 for value in values.values()):
                msg = "Completed scoreboard contains a nonfinite or unphysical fidelity."
                raise ValueError(msg)
            scores[key] = statistics.mean(value for noise, value in values.items() if noise != "noiseless")
            for noise, value in values.items():
                condition_values[noise][key] = value
        if issues:
            summaries[method] = {"status": "incomplete", "issues": issues, "complete_cases": len(scores)}
            continue
        targets = balanced(scores)
        families: dict[str, list[float]] = defaultdict(list)
        for target, value in targets.items():
            families[family(target)].append(value)
        target_family_means = {name: statistics.mean(group) for name, group in families.items()}
        noise_means = {
            noise: statistics.mean(balanced(condition_values[noise]).values()) for noise in channels.NOISE_IDS
        }
        noise_family_means = {
            name: statistics.mean(noise_means[noise] for noise in noises) for name, noises in NOISE_FAMILIES.items()
        }
        means[method] = scores
        summaries[method] = {
            "status": "complete",
            "cases": len(scores),
            "targets": len(targets),
            "mean_noisy_fidelity": statistics.mean(targets.values()),
            "median_target_noisy_fidelity": statistics.median(targets.values()),
            "worst_target_noisy_fidelity": min(targets.values()),
            "worst_target": min(targets, key=lambda target: targets[target]),
            "worst_target_family_noisy_fidelity": min(target_family_means.values()),
            "worst_target_family": min(target_family_means, key=lambda name: target_family_means[name]),
            "worst_noise_family_fidelity": min(noise_family_means.values()),
            "worst_noise_family": min(noise_family_means, key=lambda name: noise_family_means[name]),
            "by_target": targets,
            "by_target_family": target_family_means,
            "by_noise": noise_means,
            "by_noise_family": noise_family_means,
        }
    pairs = []
    for candidate, baseline in itertools.permutations(sorted(means), 2):
        delta = {case: means[candidate][case] - means[baseline][case] for case in sorted(all_cases)}
        targets = balanced(delta)
        pairs.append({
            "candidate": candidate,
            "baseline": baseline,
            "mean_delta_pp": 100 * statistics.mean(targets.values()),
            "median_target_delta_pp": 100 * statistics.median(targets.values()),
            "worst_target_delta_pp": 100 * min(targets.values()),
            "target_wins": sum(value > 1e-12 for value in targets.values()),
            "target_ties": sum(abs(value) <= 1e-12 for value in targets.values()),
            "target_count": len(targets),
            "case_wins": sum(value > 1e-12 for value in delta.values()),
            "case_count": len(delta),
            "by_target_delta_pp": {target: 100 * value for target, value in targets.items()},
            "comparison": "unmatched_reference_contrast"
            if any(
                "unmatched" in row.get("comparison", "").lower()
                for row in rows
                if row["method"] in {candidate, baseline}
            )
            else "resource_matching_defined_by_frozen_protocol",
        })
    return {"methods": summaries, "paired_all_noisy": pairs}


def collect(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Authenticate and preserve source columns and separate panel aggregates.

    Raises:
        ValueError: If sources are absent, repeated, or lack complete provenance.
    """
    if not paths:
        msg = "At least one authenticated panel scoreboard is required."
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    panels = {}
    sources = {}
    for path in paths:
        panel = path.parent.name
        if panel in sources:
            msg = "Each source scoreboard must have a distinct parent panel name."
            raise ValueError(msg)
        rows, provenance = authenticate_panel(path)
        if provenance["schema_kind"] in {
            "baseline",
            "mixture_baseline",
            "long_ensemble",
            "composed",
            "common_control",
            "compiled_composition",
            "best_candidate",
        }:
            strata = ["full_cap", "common_cpu", "historical_unmatched"]
            if provenance["schema_kind"] in {"composed", "compiled_composition"}:
                strata = ["full_cap", "common_primary", "common_all_trained"]
            if provenance["schema_kind"] == "best_candidate":
                strata = ["full_cap", "primary", "all_trained_sensitivity"]
            if provenance["schema_kind"] == "common_control":
                strata = ["common_primary"]
            if provenance["schema_kind"] == "long_ensemble":
                strata.remove("historical_unmatched")
            if provenance["schema_kind"] == "mixture_baseline":
                strata.append("unchanged")
            for stratum in strata:
                selected = [row for row in rows if row["method"].endswith(f"[{stratum}]")]
                panels[f"{panel}/{stratum}"] = summarize(selected)
        else:
            panels[panel] = summarize(rows)
        sources[panel] = provenance
        all_rows.extend({**row, "panel": panel} for row in rows)
    fieldnames = ("panel", "target_id", "replica", "method", "noise_id", "status", "fidelity")
    remaining = sorted({key for row in all_rows for key in row} - set(fieldnames))
    all_rows.sort(key=lambda row: tuple(row[key] for key in fieldnames[:5]))
    with (output / "SCOREBOARD.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*fieldnames, *remaining))
        writer.writeheader()
        writer.writerows(all_rows)
    result: dict[str, Any] = {
        "sources": sources,
        "weighting": "eleven noisy conditions equally; replicas equally within target; targets equally",
        "endpoint_rule": "as frozen in each panel; historical references explicitly unmatched",
        "comparison_limit": "panel contrasts are descriptive unless protocols establish matched resources",
        "panels": panels,
        "raw_rows": len(all_rows) - sum(source["missing_csv_cells"] for source in sources.values()),
        "rows_including_missing": len(all_rows),
        "reporting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    pilot.write_record(output / "SCOREBOARD.json", result)
    lines = [
        "# Overnight development scoreboard",
        "",
        "Exact fidelity percentages; noise conditions, replicas within target, then targets receive equal weight.",
        "Panels retain their own budgets and architectures. This is not a cross-panel ranking.",
        "All raw resource columns and failed cells are preserved in SCOREBOARD.csv.",
        "Historical checkpoint references remain unmatched.",
        "Each source protocol, numerical source snapshot, completion ledger, and CSV score/resource cell is verified.",
        "Missing expected cells remain visible and invalidate the affected method aggregate.",
        "Noise families follow the benchmark: N1 angle dependent, N2 dephasing, N3 depolarizing.",
        "",
        (
            "| Panel | Method | Targets / cases | Mean noisy % | Median target % | "
            "Worst target % | Worst target family % | Worst noise family % |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for panel, summary in panels.items():
        for method, values in summary["methods"].items():
            if values["status"] != "complete":
                lines.append(f"| {panel} | {method} | INCOMPLETE | — | — | — | — | — |")
                continue
            metrics = [
                f"{100 * values[key]:.6f}"
                for key in (
                    "mean_noisy_fidelity",
                    "median_target_noisy_fidelity",
                    "worst_target_noisy_fidelity",
                    "worst_target_family_noisy_fidelity",
                    "worst_noise_family_fidelity",
                )
            ]
            lines.append(f"| {panel} | {method} | {values['targets']} / {values['cases']} | {' | '.join(metrics)} |")
    lines.extend([
        "",
        "All within-panel paired contrasts and target/family/channel results are in SCOREBOARD.json.",
        "",
    ])
    (output / "SCOREBOARD.md").write_text("\n".join(lines))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Build an auditable overview from explicitly named completed panel CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("scoreboards", type=Path, nargs="+")
    args = parser.parse_args(argv)
    collect(args.scoreboards, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
