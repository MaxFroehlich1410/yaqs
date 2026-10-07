# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reporting regressions for dominance, paired denominators and independent paths."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pytest

from experiments import topdown_depth_analysis as analysis
from experiments import topdown_depth_search as study
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def test_pareto_keeps_tradeoffs_and_exact_duplicates() -> None:
    """Depth/resources and fidelity tradeoffs survive; strictly dominated points do not."""
    points = np.array([[10, 4, -0.9], [8, 4, -0.9], [9, 5, -0.88], [12, 6, -0.95], [8, 4, -0.9]])
    assert analysis.pareto_indices(points) == [1, 3, 4]
    assert not analysis.pareto_indices(np.empty((0, 3)))


def test_independent_report_path_replays_single_and_correlated_jumps() -> None:
    """The audit's Qiskit trajectory includes the same physical noise as vector validation."""
    circuit = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("ry", (2,), 0),
            ParameterizedGate("rxx", (2, 0), 1),
            ParameterizedGate("rz", (1,), 2),
            ParameterizedGate("ryy", (0, 1), 3),
        ],
        4,
    )
    theta = np.array([0.3, 0.8, -0.4, 0.7])
    rng = np.random.default_rng(8)
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target /= np.linalg.norm(target)
    record = {
        "hardware": "Google",
        "theta": theta.tolist(),
        "circuit": {"num_qubits": 3, "num_params": 4, "gates": [asdict(gate) for gate in circuit.gates]},
    }
    codes = np.array([[0, 0, 0, 0], [1, 8, 0, 4], [0, 2, 3, 9]], dtype=np.uint8)
    explicit = [analysis.qiskit_path_fidelity(record, target, row) for row in codes]
    actual = study.validation_score(circuit, theta, target, "Google", codes)
    assert actual["fidelity"] == pytest.approx(np.mean(explicit), abs=1e-14)


def test_summary_uses_paired_means_and_total_search_cost() -> None:
    """Unequal timing denominators and intermediate failures cannot inflate reported gain."""
    rows = []
    keys = ("logical_two_qubit", "compiled_two_qubit", "compiled_two_qubit_depth")
    for hardware in study.HARDWARE:
        for variant in study.VARIANTS:
            for seed, old, new, old_cpu, new_cpu in ((300, 0.8, 0.9, 2, 8), (400, 0.95, 0.85, 4, 20)):
                rows.append({
                    "target_id": "example",
                    "seed": seed,
                    "hardware": hardware,
                    "variant": variant,
                    "baseline_fidelity": old,
                    "gain_pp": 100 * (new - old),
                    "baseline_cost": {"total_search_cpu_seconds": old_cpu},
                    "search_cost": {
                        "total_search_cpu_seconds": new_cpu,
                        "failed_candidates": 1,
                        "construction_cpu_seconds": 1,
                        "pruning_cpu_seconds": 2,
                        "fine_tuning_cpu_seconds": 3,
                        "validation_cpu_seconds": 4,
                    },
                    "baseline_resources": dict.fromkeys(keys, 10),
                    "stop_depth": 5,
                    "stop_reason": "patience",
                    "selection_regret_pp": 0.2,
                    "retrospective_best_depth": 3,
                    "selected": {
                        "fidelity": new,
                        "depth": 2,
                        "construction_ideal": 0.88,
                        "resources": dict.fromkeys(keys, 8),
                    },
                })
    groups = analysis.aggregate(rows)
    assert len(groups) == 4
    for group in groups:
        assert group["gain_pp"] == pytest.approx(0)
        assert group["baseline_mean_percent"] == pytest.approx(87.5)
        assert group["new_mean_percent"] == pytest.approx(87.5)
        assert group["cpu_ratio"] == pytest.approx(14 / 3)
        assert group["wins"] == group["losses"] == 1
        assert group["candidate_failures"] == 2
        assert group["selected_before_ideal_90"] == 2
        assert group["exact_winner_mismatches"] == 2


@pytest.mark.parametrize("training_paths", [100, 400])
def test_audit_counts_and_regenerates_failed_training_banks(tmp_path: Path, training_paths: int) -> None:
    """Failed candidates retain the configured raw sampling cost without training."""
    root = tmp_path / "case"
    search = root / "search"
    search.mkdir(parents=True)
    np.save(root / "target.npy", np.array([1, 0], dtype=complex))
    study.write_json(root / "selection_locked.json", {})
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    tracker = study.Patience()
    decisions = []
    for depth in range(1, 4):
        study.save_checkpoint(
            search,
            f"depth_{depth:02d}_D_IBM",
            circuit,
            np.array([0.2]),
            {
                "depth": depth,
                "variant": "D",
                "hardware": "IBM",
                "status": "no_jump_training_bank",
                "training": None,
                "validation": None,
            },
        )
        decisions.append({"depth": depth, "score": None, "standard_error": 0.0, **tracker.observe(depth, None)})
    physical = {"trace_error": 0.0, "hermiticity_error": 0.0, "minimum_eigenvalue": 0.0}
    baseline = {
        "vector_matrix_error": 0.0,
        "resources": {"logical_unitary_max_error_up_to_global_phase": 0.0},
        "ideal": physical,
        "exact": physical,
        "lindblad": physical,
    }
    study.write_json(
        root / "complete.json",
        {
            "optimizer_seed": 300,
            "artifacts": {},
            "selection_sha256": study.file_hash(root / "selection_locked.json"),
            "results": [
                {
                    "variant": "D",
                    "hardware": "IBM",
                    "decisions": decisions,
                    "selected_depth": None,
                    "stop_reason": "patience",
                    "baseline": baseline,
                    "evaluations": [],
                }
            ],
        },
    )
    audit = analysis.audit_case((str(tmp_path), "case", training_paths))
    assert audit["failed_training_sampled"] == 3 * training_paths
    assert audit["training_sampled"] == 0
    assert audit["training_banks"] == 0
    assert audit["coverage_failures"] == 3
