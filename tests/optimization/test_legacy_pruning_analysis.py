# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Missing-data pairing and an independently replayed pruning audit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import qasm2
from qiskit.quantum_info import Operator

from experiments import legacy_pruning_ablation as study
from experiments import legacy_pruning_analysis as analysis
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from pathlib import Path


def test_pairing_uses_only_shared_completed_cases() -> None:
    """A missing outcome is not zero or a reason to change the comparison subset."""
    rows = [
        {
            "target_id": "target",
            "optimizer_seed": seed,
            "hardware": "IBM",
            "variant": variant,
            "status": "complete" if fidelity is not None else "no_jump_bank",
            "fidelity": fidelity,
        }
        for seed, variant, fidelity in [(300, "A", 0.4), (300, "B", 0.6), (400, "A", None), (400, "B", 0.9)]
    ]
    result = analysis.paired(rows, "IBM", "A", "B")
    assert result["pairs"] == 1
    assert result["mean_delta_pp"] == pytest.approx(20)
    assert result["mean_before_percent"] == pytest.approx(40)
    assert result["mean_after_percent"] == pytest.approx(60)
    assert result["wins"] == 1
    absent = analysis.paired(rows, "Google", "A", "B")
    assert absent["pairs"] == 0
    assert absent["mean_delta_pp"] is None


def test_end_to_end_audit_with_empty_and_failed_circuits(tmp_path: Path) -> None:
    """Audit saved decisions, Qiskit fidelities, coverage and resource artifacts."""
    folder = "small_case"
    target = np.array([0, 1], dtype=complex)
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), 0)], 1)
    theta = np.array([np.pi])
    source = study.Checkpoints(tmp_path / "source" / folder, target)
    source.save("unpruned_initial", circuit, theta, {})
    root = tmp_path / "output"
    store = study.Checkpoints(root / folder, target)
    study.write_json(root / "manifest.json", {"source_root": str(source.root.parent)})
    results = []
    for variant in study.VARIANTS:
        pruned, values, trace = study.prune_variant(circuit, theta, target, variant)
        store.save(variant + "_initial", pruned, values, trace)
        results.extend(study.fine_tune(store, variant, hardware, 300) for hardware in study.HARDWARE)
    study.write_json(
        store.root / "complete.json",
        {
            "results": results,
            "artifacts": {path.name: study.file_hash(path) for path in store.root.iterdir() if path.is_file()},
        },
    )
    audit = analysis.audit_target((str(root), folder))
    assert audit["initial_candidates_checked_with_qiskit"] == 1
    assert audit["empty_circuit_conditions"] == 2
    assert audit["coverage_failures"] == 6
    assert audit["completed_banks"] == 2
    assert audit["maximum_deletion_or_repair_fidelity_error"] < 1e-12


def test_markdown_retains_missing_literal() -> None:
    """Reported missing values remain explicit rather than numeric zero."""
    assert analysis.markdown_table(["Setting", "F"], [["A", "missing"]]) == (
        "| Setting | F |\n| --- | --- |\n| A | missing |"
    )


def test_exported_sx_qasm_roundtrip(tmp_path: Path) -> None:
    """The audit reads sx gates in Qiskit's extended qelib1 export correctly."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), 0)], 1)
    compiled, _ = study.compile_resources(circuit, np.array([0.37]))
    path = tmp_path / "compiled.qasm"
    path.write_text(qasm2.dumps(compiled))
    restored = analysis.read_compiled_qasm(path)
    assert restored.count_ops()["sx"] > 0
    assert Operator(restored).equiv(Operator(compiled))
