# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Selection separation, authenticated resume and competitor sweep orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.state_preparation_competitors import run as study
from experiments.state_preparation_competitors.common import Candidate

if TYPE_CHECKING:
    import argparse

    from experiments.state_preparation_competitors.evaluation import CompiledCandidate


def candidate(identifier: str = "one", *, empty: bool = True, status: str = "complete") -> Candidate:
    """Return a physical two-qubit candidate with explicit completion metadata."""
    circuit = QuantumCircuit(2)
    if not empty:
        circuit.ry(np.pi, 0)
    return Candidate(circuit, {"candidate_id": identifier, "method": "toy", "status": status})


def args_for(output: Path) -> argparse.Namespace:
    """Return a small valid CLI configuration using the real argument parser."""
    return study.make_parser().parse_args([
        "--output",
        str(output),
        "--targets",
        study.TARGET_IDS[0],
        "--methods",
        "aqce",
        "--hardware",
        "IBM",
        "--depths",
        "1",
        "--aqce-gates",
        "0",
        "--aqce-sweeps",
        "0",
        "--construction-steps",
        "0",
        "--online-steps",
        "0",
        "--noisy-steps",
        "0",
        "--noisy-online-steps",
        "0",
        "--validation-paths",
        "8",
    ])


def evaluate(folder: Path, candidates: list[Candidate] | None = None, **changes: object) -> dict:
    """Return a toy evaluation group with all context fields stated explicitly."""
    kwargs: dict[str, Any] = {
        "hardware": "IBM",
        "validation_paths": 8,
        "identity": {"target_id": "toy", "method": "toy", "seed": 300, "hardware": "IBM"},
        "construction": {"construction_cpu_seconds": 1.25},
        "construction_receipt": "a" * 64,
        "construction_cache": "construction/toy",
    }
    kwargs.update(changes)
    target = kwargs.pop("target", np.array([1, 0, 0, 0], dtype=complex))
    return study.evaluate_group(folder, [candidate()] if candidates is None else candidates, target, **kwargs)


def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail when a resumed stage unexpectedly repeats already authenticated work."""
    pytest.fail("A completed stage must be loaded rather than recomputed.")


def controlled_bank(circuit: QuantumCircuit, _hardware: str, count: int, _entropy: list[int]) -> np.ndarray:
    """Return a deterministic toy bank with one real Pauli event when gates exist."""
    codes = np.zeros((count, len(circuit.data)), dtype=np.uint8)
    if circuit.data:
        codes[0, 0] = 1
    return codes


def test_finite_atomic_json_and_digest_preserve_existing_valid_artifacts(tmp_path: Path) -> None:
    """Serialization errors occur before replacing any existing authenticated value."""
    path = tmp_path / "nested" / "value.json"
    study.write_json(path, {"z": [1, 2], "a": 0.5})
    original = path.read_bytes()
    assert json.loads(original) == {"a": 0.5, "z": [1, 2]}
    assert study.digest(path) == hashlib.sha256(original).hexdigest()
    assert not path.with_suffix(".json.tmp").exists()
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="JSON compliant"):
            study.write_json(path, {"nested": [invalid]})
        assert path.read_bytes() == original


def test_bundle_roundtrip_preserves_circuits_metadata_and_seals(tmp_path: Path) -> None:
    """QPY stores actual complex-angle circuits rather than only resource summaries."""
    first, second = candidate().circuit, candidate(empty=False).circuit
    second.s(1)
    second.cx(1, 0)
    second.global_phase = 0.31
    record = {"identity": ["example", 2], "scalar": 0.9}
    study.save_bundle(tmp_path, record, [first, second])
    actual, circuits = study.load_bundle(tmp_path)
    assert actual == record
    assert circuits == [first, second]
    seal = json.loads((tmp_path / "seal.json").read_text())
    assert seal == {name: study.digest(tmp_path / name) for name in ("record.json", "circuits.qpy")}


def test_qpy_roundtrip_preserves_routed_output_layout_and_exact_target_overlap(tmp_path: Path) -> None:
    """A resumed routed circuit retains the wire permutation needed for final density scoring."""
    source = QuantumCircuit(3)
    source.h(0)
    source.s(0)
    source.ry(0.31, 1)
    source.cx(0, 2)
    target = Statevector.from_instruction(source).data
    compiled = study.evaluation.compile_candidate(Candidate(source, {}), target)
    assert compiled.metadata["logical_to_physical_output"] != [0, 1, 2]
    study.save_bundle(tmp_path, {"compilation": compiled.metadata}, [compiled.circuit])
    record, circuits = study.load_bundle(tmp_path)
    assert circuits[0].layout is not None
    assert circuits[0].layout.final_index_layout() == compiled.metadata["logical_to_physical_output"]
    mapped = study.evaluation.physical_target(target, record["compilation"]["logical_to_physical_output"])
    restored = study.evaluation.CompiledCandidate(circuits[0], mapped, record["compilation"])
    assert study.evaluation.exact_score(restored, "IBM") == study.evaluation.exact_score(compiled, "IBM")


@pytest.mark.parametrize("member", ["record.json", "circuits.qpy"])
@pytest.mark.parametrize("mutation", ["append", "remove"])
def test_bundle_detects_modified_or_missing_members(tmp_path: Path, member: str, mutation: str) -> None:
    """Metadata and numerical circuits are both protected before any resume reads."""
    study.save_bundle(tmp_path, {}, [candidate().circuit])
    path = tmp_path / member
    if mutation == "append":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="integrity failure"):
        study.load_bundle(tmp_path)


def test_bundle_rejects_incomplete_seal(tmp_path: Path) -> None:
    """A receipt missing the numerical payload cannot certify a complete stage."""
    study.save_bundle(tmp_path, {}, [candidate().circuit])
    study.write_json(tmp_path / "seal.json", {"record.json": study.digest(tmp_path / "record.json")})
    with pytest.raises(ValueError, match="integrity failure"):
        study.load_bundle(tmp_path)


def test_replacing_a_bundle_removes_its_unused_history_archive(tmp_path: Path) -> None:
    """A completed replacement does not leave stale unsealed parameter data behind."""
    record = {"optimizer": {"records": [{"theta": [0.1], "fidelity": 0.2}]}}
    study.save_bundle(tmp_path, record, [candidate().circuit])
    assert (tmp_path / "histories.npz").is_file()
    study.save_bundle(tmp_path, {"status": "complete"}, [candidate().circuit])
    assert not (tmp_path / "histories.npz").exists()
    restored, _ = study.load_bundle(tmp_path)
    assert restored == {"status": "complete"}


def test_validation_entropy_is_stable_and_separates_every_job_role() -> None:
    """Target, method, hardware, replica and candidate affect independent validation seeds."""
    identity = {"target_id": "toy", "method": "aqce", "seed": 300, "hardware": "IBM"}
    baseline = study.validation_entropy(identity, "depth_01")
    assert baseline == study.validation_entropy(dict(reversed(list(identity.items()))), "depth_01")
    assert baseline[:2] == [20260920, 0x56414C]
    alternatives = [study.validation_entropy(identity, "depth_02")]
    for key, value in (("target_id", "other"), ("method", "layerwise"), ("seed", 400), ("hardware", "Google")):
        alternatives.append(study.validation_entropy({**identity, key: value}, "depth_01"))
    assert len({tuple(value) for value in [baseline, *alternatives]}) == 6
    assert all(type(value) is int and value >= 0 for value in baseline)
    assert baseline != [5300]  # The legacy training stream is not reused.


def test_selection_uses_only_eligible_validation_then_declared_resource_ties() -> None:
    """Neither ideal construction fidelity nor exact noisy test scores can choose winners."""
    records = []
    for name, fidelity, gates, depth, eligible in (
        ("failed", 1.0, 0, 0, False),
        ("lower_fidelity", 0.7, 0, 0, True),
        ("more_gates", 0.8, 3, 1, True),
        ("deeper", 0.8, 2, 3, True),
        ("z", 0.8, 2, 1, True),
        ("a", 0.8, 2, 1, True),
    ):
        records.append({
            "eligible": eligible,
            "validation": {"fidelity": fidelity},
            "compilation": {"compiled_two_qubit": gates, "compiled_two_qubit_depth": depth, "ideal_fidelity": 1},
            "method": {"candidate_id": name},
            "exact": {"fidelity": 0 if name == "a" else 1},
        })
    assert study.select_candidate(records) == 5
    assert study.select_candidate([]) is None
    assert study.select_candidate([{**record, "eligible": False} for record in records]) is None


def test_prepare_candidates_caches_full_grid_and_original_cost(tmp_path: Path) -> None:
    """A second hardware evaluation reuses the same authenticated construction receipt."""
    grid = [candidate("empty"), candidate("rotation", empty=False)]
    returned, receipt = study.prepare_candidates(tmp_path, lambda: grid)
    again, cached = study.prepare_candidates(tmp_path, fail_if_called)
    assert receipt == cached
    assert receipt["construction_cpu_seconds"] >= 0
    assert receipt["construction_wall_seconds"] >= 0
    assert [item.metadata for item in returned] == [item.metadata for item in again]
    assert [item.circuit for item in returned] == [item.circuit for item in again]


@pytest.mark.parametrize("identifiers", [[], [""], [None], ["duplicate", "duplicate"]])
def test_prepare_candidates_requires_a_nonempty_unique_identifier_grid(tmp_path: Path, identifiers: list) -> None:
    """Missing or duplicate labels cannot create ambiguous selection receipts."""
    grid = [Candidate(QuantumCircuit(2), {"candidate_id": value}) for value in identifiers]
    with pytest.raises(ValueError, match=r"nonempty.*unique"):
        study.prepare_candidates(tmp_path, lambda: grid)
    assert not (tmp_path / "seal.json").exists()


def test_prepare_candidates_rejects_authenticated_record_circuit_count_mismatch(tmp_path: Path) -> None:
    """Even individually valid files must agree on the number of candidates."""
    study.save_bundle(tmp_path, {"candidates": [candidate().metadata, candidate("two").metadata]}, [QuantumCircuit(2)])
    with pytest.raises(ValueError, match="count differ"):
        study.prepare_candidates(tmp_path, fail_if_called)


def test_exact_density_runs_only_after_persisted_lock_and_cannot_reverse_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately misleading validation sample cannot be repaired using test scores."""
    grid = [candidate("validation_winner", empty=False), candidate("exact_winner")]
    original_validation = study.evaluation.validation_score
    original_exact = study.evaluation.exact_score
    exact_calls = 0

    def ranked_validation(compiled: CompiledCandidate, hardware: str, codes: np.ndarray) -> dict:
        value = original_validation(compiled, hardware, codes)
        value["fidelity"] = 0.9 if compiled.circuit.data else 0.4
        return value

    def guarded_exact(compiled: CompiledCandidate, hardware: str) -> dict:
        nonlocal exact_calls
        assert (tmp_path / "selection_locked/seal.json").is_file()
        locked, stored = study.load_bundle(tmp_path / "selection_locked")
        assert locked["selected_index"] == 0
        assert len(stored) == 2
        assert all("exact" not in row for row in locked["records"])
        exact_calls += 1
        return original_exact(compiled, hardware)

    monkeypatch.setattr(study.evaluation, "sample_validation", controlled_bank)
    monkeypatch.setattr(study.evaluation, "validation_score", ranked_validation)
    monkeypatch.setattr(study.evaluation, "exact_score", guarded_exact)
    result = evaluate(tmp_path, grid)
    assert exact_calls == 2
    assert result["selected_index"] == 0
    assert result["final_scores"][0]["exact"]["fidelity"] < result["final_scores"][1]["exact"]["fidelity"]
    assert result["final_scores"][1]["exact"]["fidelity"] == pytest.approx(1)
    assert result["records"][1]["eligible"]  # An empty circuit needs no artificial jump.
    assert result["records"][1]["validation"]["jump_paths"] == 0
    assert result["selection_lock_sha256"] == study.digest(tmp_path / "selection_locked/seal.json")
    for name in ("compile_candidate", "sample_validation", "validation_score", "exact_score"):
        monkeypatch.setattr(study.evaluation, name, fail_if_called)
    monkeypatch.setattr(study, "select_candidate", fail_if_called)
    assert evaluate(tmp_path, []) == result


def test_failed_training_and_missing_validation_jumps_remain_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete exact final scores do not make unsuccessful training banks eligible."""
    grid = [candidate("failed_training", status="no_jump_coverage"), candidate("missing_jumps", empty=False)]
    monkeypatch.setattr(
        study.evaluation,
        "sample_validation",
        lambda circuit, _hardware, count, _entropy: np.zeros((count, len(circuit.data)), dtype=np.uint8),
    )
    result = evaluate(tmp_path, grid)
    assert result["selected_index"] is None
    assert result["status"] == "no_eligible_candidate"
    assert not any(row["eligible"] for row in result["records"])
    assert result["records"][0]["validation_coverage"] == "complete"
    assert result["records"][1]["validation_coverage"] == "no_jump_coverage"
    assert len(result["final_scores"]) == 2


def test_interrupted_final_evaluation_resumes_the_persisted_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed exact calculation leaves an immutable winner for a later retry."""
    original = study.evaluation.exact_score

    def interrupted(*_args: object, **_kwargs: object) -> None:
        assert (tmp_path / "selection_locked/seal.json").exists()
        msg = "synthetic interruption"
        raise RuntimeError(msg)

    monkeypatch.setattr(study.evaluation, "exact_score", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluate(tmp_path)
    assert not (tmp_path / "final/seal.json").exists()
    lock_hash = study.digest(tmp_path / "selection_locked/seal.json")
    monkeypatch.setattr(study.evaluation, "exact_score", original)
    for name in ("compile_candidate", "sample_validation", "validation_score"):
        monkeypatch.setattr(study.evaluation, name, fail_if_called)
    monkeypatch.setattr(study, "select_candidate", fail_if_called)
    result = evaluate(tmp_path, [])
    assert result["selection_lock_sha256"] == lock_hash
    assert result["selected_index"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"hardware": "Google"},
        {"validation_paths": 16},
        {"construction_receipt": "b" * 64},
        {"identity": {"method": "other"}},
        {"target": np.array([0, 1, 0, 0], dtype=complex)},
    ],
)
def test_evaluation_resume_rejects_changed_context(tmp_path: Path, changes: dict) -> None:
    """A valid seal cannot authorize using a different target, bank or construction."""
    evaluate(tmp_path)
    with pytest.raises(ValueError, match="evaluation context"):
        evaluate(tmp_path, **changes)


def test_final_receipt_must_bind_to_the_actual_selection_lock(tmp_path: Path) -> None:
    """A well-formed final bundle from another lock is rejected on resume."""
    result = evaluate(tmp_path)
    _, circuits = study.load_bundle(tmp_path / "final")
    result["selection_lock_sha256"] = "b" * 64
    study.save_bundle(tmp_path / "final", result, circuits)
    with pytest.raises(ValueError, match="selection lock"):
        evaluate(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("targets", []),
        ("methods", ["aqce", "aqce"]),
        ("hardware", []),
        ("seeds", [300, 300]),
        ("seeds", [-1]),
        ("aqce_sweeps", -1),
        ("aqce_initial_gates", -1),
        ("aqce_growth_increment", 0),
        ("construction_steps", -1),
        ("training_paths", 0),
        ("validation_paths", 1),
        ("online_steps", 1),
        ("noisy_online_steps", 1),
        ("depths", [2, 1]),
        ("depths", [-1]),
        ("aqce_gates", [1, 0]),
        ("aqce_gates", [-1]),
        ("baa_losses", [-0.1]),
        ("baa_losses", [float("nan")]),
        ("baa_losses", [1]),
    ],
)
def test_config_validation_rejects_invalid_lists_and_budgets(tmp_path: Path, field: str, value: object) -> None:
    """Invalid campaigns fail before manifests or construction artifacts are created."""
    args = args_for(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=r"nonempty|nonnegative|path|budgets|sorted|lie"):
        study.validate_config(args)


def test_layerwise_config_requires_contiguous_ancestors_and_parser_checks_method(tmp_path: Path) -> None:
    """CLI parsing rejects unknown algorithms and layerwise grids cannot skip parents."""
    args = args_for(tmp_path)
    args.methods, args.depths = ["layerwise"], [1, 3]
    with pytest.raises(ValueError, match="contiguous"):
        study.validate_config(args)
    with pytest.raises(SystemExit):
        study.make_parser().parse_args(["--output", str(tmp_path), "--methods", "adapt"])


def test_default_methods_include_only_noise_trained_incumbent_and_controls(tmp_path: Path) -> None:
    """An unspecified method list excludes noiseless synthesis from future campaigns."""
    args = study.make_parser().parse_args(["--output", str(tmp_path)])
    study.validate_config(args)
    assert args.methods == ["current_b_lbfgs_frozen", "fixed_vqsp", "layerwise"]
    assert not {"aqce", "baa_lrsp"}.intersection(args.methods)


@pytest.mark.parametrize("methods", [["aqce"], ["baa_lrsp"], ["aqce", "baa_lrsp"]])
def test_noiseless_historical_methods_remain_explicitly_selectable(tmp_path: Path, methods: list[str]) -> None:
    """Historical synthesis runs can be reproduced without restoring them to defaults."""
    args = study.make_parser().parse_args(["--output", str(tmp_path), "--methods", *methods])
    study.validate_config(args)
    assert args.methods == methods


@pytest.mark.parametrize(("budgets", "initial", "increment"), [([0, 1], 2, 1), ([0, 2], 1, 2)])
def test_aqce_config_requires_reachable_positive_budgets(
    tmp_path: Path, budgets: list[int], initial: int, increment: int
) -> None:
    """Reject impossible M0/delta M grids before creating experiment output."""
    args = args_for(tmp_path)
    args.aqce_gates, args.aqce_initial_gates, args.aqce_growth_increment = budgets, initial, increment
    with pytest.raises(ValueError, match="reachable"):
        study.validate_config(args)


def test_aqce_paper_controls_are_exposed_in_cli_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Record changed M0/delta M explicitly so resume cannot mix algorithms."""
    args = study.make_parser().parse_args([
        "--output",
        str(tmp_path),
        "--methods",
        "aqce",
        "--aqce-gates",
        "0",
        "2",
        "4",
        "--aqce-initial-gates",
        "2",
        "--aqce-growth-increment",
        "2",
        "--aqce-sweeps",
        "3",
    ])
    study.validate_config(args)
    mock_pinned_versions(monkeypatch)
    manifest = study.build_manifest(args)
    assert manifest["config"]["aqce_initial_gates"] == 2
    assert manifest["config"]["aqce_growth_increment"] == 2
    assert manifest["config"]["aqce_sweeps"] == 3
    assert "experiments/state_preparation_competitors/aqce_schedule.py" in manifest["sources"]


def mock_pinned_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid requiring optional synthesis packages in orchestration unit tests."""
    requirements = Path(study.__file__).with_name("requirements.txt")
    pins = dict(
        line.split("==") for line in requirements.read_text().splitlines() if "==" in line and not line.startswith("#")
    )
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: pins[name])


def test_manifest_binds_numerical_dependencies_config_and_package_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume authentication includes the reused L-BFGS, gate and noise implementations."""
    mock_pinned_versions(monkeypatch)
    args = args_for(tmp_path)
    first = study.build_manifest(args)
    for path in (
        "experiments/overnight_state_preparation_20260913_0108/optimizers.py",
        "experiments/state_preparation_pilot.py",
        "src/mqt/yaqs/core/libraries/circuit_library.py",
        "src/mqt/yaqs/core/data_structures/noise_model.py",
        "src/mqt/yaqs/optimization/statevector.py",
    ):
        assert first["sources"][path] == study.digest(study.ROOT / path)
    assert "output" not in first["config"]
    assert first["schema"] == study.SCHEMA
    assert "interrupted unsealed attempts" in first["cost_scope"]
    second_args = copy.deepcopy(args)
    second_args.output = tmp_path / "another"
    assert study.build_manifest(second_args) == first
    second_args.seeds = [700]
    assert study.build_manifest(second_args)["config"] != first["config"]
    original_digest = study.digest
    monkeypatch.setattr(
        study, "digest", lambda path: "changed" if path.name == "optimizers.py" else original_digest(path)
    )
    assert study.build_manifest(args)["sources"] != first["sources"]
    monkeypatch.setattr(study.importlib.metadata, "version", lambda _name: "wrong")
    with pytest.raises(ValueError, match="Pinned"):
        study.build_manifest(args)


@pytest.mark.parametrize("method", study.METHODS)
def test_dispatch_forwards_exact_method_specific_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """Published adapters receive their own grids; only iterative controls receive training settings."""
    args = args_for(tmp_path)
    target = np.array([1, 0, 0, 0], dtype=complex)
    observed = {}

    def factory(*values: object, **options: object) -> list[Candidate]:
        observed.update({"values": values, "options": options})
        return [candidate()]

    for owner, name in (
        (study.aqce, "aqce_candidates"),
        (study.baa, "baa_candidates"),
        (study.controls, "fixed_vqsp_candidates"),
        (study.controls, "layerwise_candidates"),
    ):
        monkeypatch.setattr(owner, name, factory)

    def reference_factory(*values: object) -> Candidate:
        observed["values"] = values
        observed["options"] = {}
        return candidate("archived")

    monkeypatch.setattr(study, "frozen_reference", reference_factory)
    returned = study.construct(args, method, "toy", target, 400, "Google")
    assert len(returned) == 1
    options = observed["options"]
    if method == "aqce":
        assert options == {
            "gate_budgets": tuple(args.aqce_gates),
            "sweeps": args.aqce_sweeps,
            "initial_gates": args.aqce_initial_gates,
            "growth_increment": args.aqce_growth_increment,
        }
    elif method == "baa_lrsp":
        assert options == {"fidelity_losses": tuple(args.baa_losses)}
    elif method == "current_b_lbfgs_frozen":
        assert observed["values"][:4] == (args.reference_root, "toy", 400, "Google")
        np.testing.assert_array_equal(observed["values"][4], target)
    else:
        expected = {
            "depths": tuple(args.depths),
            "seed": 400,
            "hardware": "Google",
            "construction_steps": 0,
            "noisy_steps": 0,
            "training_paths": 100,
        }
        if method == "layerwise":
            expected.update({"online_steps": 0, "noisy_online_steps": 0})
        assert options == expected


def test_unknown_dispatch_method_fails(tmp_path: Path) -> None:
    """Programmatic callers cannot silently fall back to another method."""
    with pytest.raises(ValueError, match="Unknown competitor"):
        study.construct(args_for(tmp_path), "unknown", "toy", np.array([1, 0]), 300, "IBM")


def test_full_sweep_reuses_deterministic_construction_and_all_authenticated_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real two-qubit evaluation covers every method/profile while construction is stubbed."""
    args = args_for(tmp_path)
    args.methods, args.hardware = list(study.METHODS), ["IBM", "Google"]
    target = np.array([1, 0, 0, 0], dtype=complex)
    collection = SimpleNamespace(load=lambda _qubits, _name: SimpleNamespace(state_vector_copy=target.copy))
    monkeypatch.setattr(study, "load_target_collection", lambda: collection)
    monkeypatch.setattr(
        study, "build_manifest", lambda _args: {"schema": study.SCHEMA, "config": "frozen", "sources": "frozen"}
    )
    calls = []

    def construct(
        _args: object, method: str, target_id: str, vector: np.ndarray, seed: int, hardware: str
    ) -> list[Candidate]:
        calls.append((method, target_id, seed, hardware))
        np.testing.assert_array_equal(vector, target)
        return [Candidate(QuantumCircuit(2), {"method": method, "candidate_id": "empty", "status": "complete"})]

    monkeypatch.setattr(study, "construct", construct)
    first = study.run(args)
    assert len(first["groups"]) == 16
    assert len(calls) == 14  # One construction per deterministic method, four for each other arm.
    assert sum(method == "aqce" for method, *_ in calls) == 1
    assert sum(method == "baa_lrsp" for method, *_ in calls) == 1
    for method in study.DETERMINISTIC:
        pair = [group for group in first["groups"] if group["context"]["identity"]["method"] == method]
        assert len(pair) == 2
        assert pair[0]["construction_cache"] == pair[1]["construction_cache"]
        assert pair[0]["construction"] == pair[1]["construction"]
        assert all(group["context"]["identity"]["seed"] is None for group in pair)
    for group in first["groups"]:
        assert group["status"] == "complete"
        assert group["selected_index"] == 0
        assert group["final_scores"][0]["exact"]["fidelity"] == pytest.approx(1)
        assert "once" in group["cost_note"]
    streams = [tuple(group["records"][0]["validation"]["entropy"]) for group in first["groups"]]
    assert len(set(streams)) == len(streams)
    assert len({group["construction_cache"] for group in first["groups"]}) == len(calls)
    assert json.loads((tmp_path / "summary.json").read_text()) == first
    monkeypatch.setattr(study, "construct", fail_if_called)
    for name in ("compile_candidate", "sample_validation", "validation_score", "exact_score"):
        monkeypatch.setattr(study.evaluation, name, fail_if_called)
    assert study.run(args) == first


@pytest.mark.parametrize("field", ["config", "sources"])
def test_resume_rejects_changed_manifest_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """A source or configuration change cannot reuse previously generated controls."""
    original = {"config": {"seed": 300}, "sources": {"optimizer.py": "abc"}}
    study.write_json(tmp_path / "manifest.json", original)
    changed = {**original, field: "changed"}
    monkeypatch.setattr(study, "build_manifest", lambda _args: changed)
    monkeypatch.setattr(study, "load_target_collection", fail_if_called)
    with pytest.raises(ValueError, match="manifest changed"):
        study.run(args_for(tmp_path))


def test_nonempty_output_without_manifest_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated output directory remains intact when no matching protocol exists."""
    (tmp_path / "unrelated.txt").write_text("keep")
    monkeypatch.setattr(study, "build_manifest", lambda _args: {})
    with pytest.raises(ValueError, match="nonempty output"):
        study.run(args_for(tmp_path))
    assert (tmp_path / "unrelated.txt").read_text() == "keep"


def test_cli_main_parses_and_dispatches_without_changing_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module entry point forwards selected methods, grids and runtime profiles."""
    captured = []
    monkeypatch.setattr(study, "run", captured.append)
    study.main(["--output", str(tmp_path), "--methods", "baa_lrsp", "--baa-losses", "0", "0.03", "--seeds", "400"])
    assert len(captured) == 1
    args = captured[0]
    assert args.output == tmp_path
    assert args.methods == ["baa_lrsp"]
    assert args.baa_losses == [0, 0.03]
    assert args.seeds == [400]
    assert args.targets == list(study.TARGET_IDS)
    assert args.validation_paths == 32768
    assert args.hardware == list(study.evaluation.HARDWARE)
