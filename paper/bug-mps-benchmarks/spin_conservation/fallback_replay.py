#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Replay saved fallback events from identical compressed states."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import fallback_models as models
import numpy as np
from fallback_pilot import canonical_vector, energy, load_snapshot, write_json

from mqt.yaqs import track_conservation
from mqt.yaqs.core.methods.conservation import restore_invariant
from mqt.yaqs.core.parallel_utils import limit_worker_threads


def replay_event(directory: Path, case: dict, event: dict) -> dict:
    """Check recovery, gauge, dimensions, and deterministic stage counts."""
    original = load_snapshot(directory / event["snapshot"])
    mpo, dense, _ = models.hamiltonians(case["length"], case["model"], case.get("width_per_site"))
    if original.flipped:
        mpo = mpo.reflected()
    before = canonical_vector(original)
    checks = {}
    for enabled in [False, True]:
        state = deepcopy(original)
        with track_conservation() as stats:
            changed = restore_invariant(state, mpo, event["target"], tol=case["conserve_tol"], fallback=enabled)
        after = canonical_vector(state)
        residual = abs(energy(after, dense) - event["target"])
        preserved = state.orthogonality_center == original.orthogonality_center and state.flipped == original.flipped
        preserved &= all(a.shape == b.shape for a, b in zip(state.tensors, original.tensors, strict=True))
        exact_rollback = all(np.array_equal(a, b) for a, b in zip(state.tensors, original.tensors, strict=True))
        threshold = max(case["conserve_tol"], 64 * np.finfo(float).eps) * max(1, abs(event["target"]))
        valid = preserved and (residual <= 2 * threshold if changed else exact_rollback)
        if enabled:
            valid &= stats.to_dict() == event["counts"]
            valid &= abs(energy(after, dense) - event["post_energy"]) <= 2 * threshold
        else:
            valid &= not changed and stats.scalar_successes == stats.already_satisfied == 0
        checks["fallback" if enabled else "scalar"] = {
            "changed": changed,
            "energy_residual": residual,
            "counts": stats.to_dict(),
            "displacement_norm": float(np.linalg.norm(after - before)),
            "gauge_and_dimensions_preserved": preserved,
            "exact_rollback": exact_rollback,
            "valid": bool(valid),
        }
    return {
        "step": event["step"],
        "call_in_step": event["call_in_step"],
        "snapshot": event["snapshot"],
        "checks": checks,
        "valid": all(check["valid"] for check in checks.values()),
    }


def replay_all(output: Path) -> dict:
    """Replay every event and write one audit file per eventful case."""
    limit_worker_threads(1)
    summary = {"events": 0, "passed": 0, "failed": 0, "cases": []}
    for path in sorted((output / "cases").glob("*/result.json")):
        result = json.loads(path.read_text())
        arm = next(arm for arm in result["arms"] if arm["variant"] == "fallback")
        if not arm["events"]:
            continue
        audits = [replay_event(path.parent / "fallback", result["case"], event) for event in arm["events"]]
        write_json(path.parent / "replay.json", audits)
        passed = sum(audit["valid"] for audit in audits)
        summary["events"] += len(audits)
        summary["passed"] += passed
        summary["failed"] += len(audits) - passed
        summary["cases"].append({"key": result["key"], "events": len(audits), "passed": passed})
        print(f"{result['key']}: {passed}/{len(audits)} replay checks passed", flush=True)
    write_json(output / "replay_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summary = replay_all(parser.parse_args().output)
    raise SystemExit(int(summary["failed"] > 0))
