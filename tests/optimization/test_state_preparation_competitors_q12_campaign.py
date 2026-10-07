# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check frozen science, real adapter parity, held-out scoring and queue safety."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit

from experiments import topdown_backend_support as checkpoints
from experiments import topdown_mps
from experiments.state_preparation_competitors import q12_campaign as campaign
from experiments.state_preparation_competitors import q12_cluster as cluster
from experiments.state_preparation_competitors.run import digest, load_bundle, write_json

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_matrix_preserves_full_budgets_and_sequential_parents() -> None:
    """Only independent competitor depths are split into separate jobs."""
    settings = campaign.protocol()
    jobs = campaign.jobs(settings)
    assert len(jobs) == len({job["id"] for job in jobs}) == 39
    assert sum(j["kind"] == "candidate" for j in jobs) == 28
    assert sum(j["kind"] == "chain" for j in jobs) == 4
    assert sum(j["kind"] == "final" for j in jobs) == 7
    assert settings["ours"] == asdict(topdown_mps.Config(qubits=12, max_depth=4))
    assert settings["robuststate"] == {
        "noiseless_steps": 500,
        "noisy_steps": 50,
        "learning_rate": 0.005,
        "feedback": "pauli_shadows",
        "num_bases": 400,
        "shots_per_basis": 1024,
    }
    assert not any(j["method"] == "fontana" and j["target"] == "haar_random_1" for j in jobs)
    assert all(len(j["dependencies"]) == 4 for j in jobs if j["kind"] == "final")
    assert campaign.source_hashes()["experiments/state_preparation_competitors/q12_cluster.py"]


@pytest.mark.parametrize("method", ["robuststate", "fontana"])
def test_real_candidate_adapter_matches_serial_and_compact_resume(tmp_path: Path, method: str) -> None:
    """A tiny numerical fixture exercises the original adapters without q12 training."""
    settings = campaign.protocol()
    settings.update({"validation_paths": 2048, "final_paths_per_batch": 1024, "final_batches": 2, "depths": [1]})
    settings["robuststate"].update({"noiseless_steps": 2, "noisy_steps": 2, "num_bases": 4, "shots_per_basis": 32})
    settings["fontana"].update({"construction_steps": 2, "noisy_steps": 2})
    target = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    job = next(j for j in campaign.jobs(settings) if j["kind"] == "candidate" and j["method"] == method)
    result = campaign.candidate(tmp_path, job, settings, target)
    factory = (
        campaign.robuststate.robuststate_candidates if method == "robuststate" else campaign.fontana.fontana_candidates
    )
    expected = factory(target, hardware="IBM", seed=300, depths=(1,), noise_basis="logical", **settings[method])[0]
    training = tmp_path / result["training_reference"]
    record, circuits = load_bundle(training)
    assert circuits[0] == expected.circuit
    assert result["validation"]["jump_paths"] > 0
    assert campaign.candidate(tmp_path, job, settings, target) == result
    if method == "robuststate":
        trace = record["candidates"][0]["training"]
        assert trace["measurement_calls"] == 2
        assert trace["bases_per_call"] == 4
        assert "parameter_history" in trace
        assert all("theta" not in r for r in trace["records"])
    checkpoints.save_checkpoint(tmp_path / "jobs" / job["id"] / "result", result, {})
    final_job = next(j for j in campaign.jobs(settings) if j["kind"] == "final" and j["method"] == method)
    final = campaign.finalize(tmp_path, final_job, settings, target)
    assert final["selected_job"] == job["id"]
    assert final["status"] == "complete"
    assert campaign.finalize(tmp_path, final_job, settings, target) == final
    test = json.loads((tmp_path / final["test_batches"][0]).read_text())
    assert test["bank_sha256"] != result["validation"]["bank_sha256"]
    assert test["entropy"] != result["validation"]["entropy"]
    assert final["total_search_cpu_seconds"] >= result["training_cpu_seconds"]
    lock = tmp_path / "jobs" / final_job["id"] / "selection_locked.json"
    write_json(lock, {"selected_job": "tampered"})
    with pytest.raises(ValueError, match="selection"):
        campaign.finalize(tmp_path, final_job, settings, target)


def test_entropy_and_coverage_do_not_condition_on_jumps(tmp_path: Path) -> None:
    """Fresh roles differ; missing jumps are reported rather than resampled."""
    job = campaign.jobs(campaign.protocol())[1]
    entropy = campaign.assessment_entropy(job, 300, "validation")
    assert entropy == campaign.assessment_entropy(job, 300, "validation")
    assert entropy != campaign.assessment_entropy(job, 300, "test")
    assert campaign.assessment_entropy(job, 300, "test", 0) != campaign.assessment_entropy(job, 300, "test", 1)
    with pytest.raises(ValueError, match="role"):
        campaign.assessment_entropy(job, 300, "training")
    circuit = QuantumCircuit(2)
    score = campaign.assess(tmp_path / "empty", circuit, np.array([1, 0, 0, 0]), 2, entropy)
    assert score["status"] == "complete"
    assert score["fidelity"] == 1
    circuit.x(0)
    score = campaign.assess(tmp_path / "gated", circuit, np.array([0, 1, 0, 0]), 2, entropy)
    assert score["status"] == "no_jump_bank"
    with pytest.raises(ValueError, match="resume"):
        campaign.assess(tmp_path / "gated", circuit, np.array([0, 1, 0, 0]), 3, entropy)


def test_manifest_authentication_and_execute_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject stale sources, versions and manifests, and forward the frozen config."""
    root = tmp_path / "campaign"
    manifest = cluster.prepare(root)
    assert campaign.authenticate(root) == manifest
    with pytest.raises(ValueError, match="new campaign"):
        cluster.prepare(root)
    job = manifest["jobs"][0]
    monkeypatch.setattr(campaign, "load_target", lambda *_: SimpleNamespace(state_vector_copy=lambda: np.array([1, 0])))
    calls = []

    def fake_chain(folder: Path, *_args: object, **kwargs: object) -> dict:
        calls.append(kwargs)
        return checkpoints.save_checkpoint(folder / "final", {"status": "complete", "fidelity": 0.9}, {})

    monkeypatch.setattr(topdown_mps, "run_case", fake_chain)
    assert campaign.execute(root, job["id"])["fidelity"] == pytest.approx(0.9)
    assert calls[0]["target_id"] == job["target"]
    original = campaign.source_hashes
    monkeypatch.setattr(campaign, "source_hashes", dict)
    with pytest.raises(ValueError, match="snapshot"):
        campaign.authenticate(root)
    monkeypatch.setattr(campaign, "source_hashes", original)
    monkeypatch.setattr(campaign.importlib.metadata, "version", lambda _: "bad")
    with pytest.raises(ValueError, match="package"):
        campaign.authenticate(root)
    (root / "manifest.sha256").write_text("bad")
    with pytest.raises(ValueError, match="checksum"):
        campaign.authenticate(root)


def test_capacity_plan_and_default_cli_never_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Heterogeneous memory/load caps are respected, and planning is non-executing."""
    root = tmp_path / "campaign"
    manifest = cluster.prepare(root)
    sample = {"cpus": 144, "load": [1, 2, 3], "available_bytes": 333 * 1024**3, "total_bytes": 755 * 1024**3}
    assert cluster.slots(sample, manifest["execution"]) == 16
    assert cluster.slots({**sample, "available_bytes": 110 * 1024**3}, manifest["execution"]) == 2
    assert cluster.slots({**sample, "load": [100, 100, 100]}, manifest["execution"]) == 0
    monkeypatch.setattr(cluster, "probe", lambda _host, _manifest: sample)
    result = cluster.plan(root, manifest)
    assert len(result["initial_assignment_preview"]) == 32
    assert result["launched"] is False
    assert result["initial_assignment_preview"][0]["job"].startswith("fontana")
    monkeypatch.setattr(cluster, "launch", lambda *_: pytest.fail("No launch was authorized by the plan command"))
    cluster.main(["--root", str(root)])
    assert not (root / "launch.json").exists()
    assert not (root / "claims").exists()


def test_atomic_claims_dependencies_failures_and_status(tmp_path: Path) -> None:
    """Concurrent workers claim every independent job exactly once on shared storage."""
    manifest = {"jobs": campaign.jobs(campaign.protocol())}
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: cluster.claim(tmp_path, manifest), range(40)))
    acquired = [job for job in claims if job is not None]
    assert len(acquired) == len({job["id"] for job in acquired}) == 32
    assert cluster.claim(tmp_path, manifest) is None
    final_job = next(j for j in manifest["jobs"] if j["kind"] == "final")
    for dependency in final_job["dependencies"]:
        folder = tmp_path / "jobs" / dependency / "result"
        checkpoints.save_checkpoint(folder, {"status": "complete"}, {})
        write_json(tmp_path / "done" / f"{dependency}.json", {"result_seal_sha256": digest(folder / "complete.json")})
    assert cluster.claim(tmp_path, manifest) == final_job
    assert cluster.status(tmp_path, manifest)[final_job["dependencies"][0]] == "done"
    other = acquired[0]["id"]
    write_json(tmp_path / "failed" / f"{other}.json", {"status": "time_limit"})
    assert cluster.status(tmp_path, manifest)[other] in {"done", "failed"}


def test_guard_receipts_distinguish_completion_failure_and_missing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No failed or missing result can satisfy downstream dependencies."""
    manifest = {"execution": {"seconds_per_job": 60, "memory_bytes_per_job": 1024**3}}

    def failed_worker(*_args: object, observer: Callable[[int, int, int], None], **_kwargs: object) -> dict:
        """Exercise heartbeat callbacks with a typed synthetic child process.

        Returns:
            A synthetic resource-limit receipt without starting a process.
        """
        observer(1234, 1000, 0)
        observer(1234, 1000, 0)
        return {"status": "time_limit"}

    monkeypatch.setattr(cluster, "run_bounded", failed_worker)
    assert cluster.guarded_job(tmp_path, manifest, {"id": "failed"})["status"] == "time_limit"
    assert json.loads((tmp_path / "claims/failed/worker.json").read_text())["pid"] == 1234
    monkeypatch.setattr(cluster, "run_bounded", lambda *_args, **_kwargs: {"status": "completed"})
    assert cluster.guarded_job(tmp_path, manifest, {"id": "missing"})["status"] == "missing_result"
    checkpoints.save_checkpoint(tmp_path / "jobs" / "good" / "result", {"status": "complete"}, {})
    assert cluster.guarded_job(tmp_path, manifest, {"id": "good"})["scientific_status"] == "complete"
    assert not (tmp_path / "done" / "missing.json").exists()
    assert (tmp_path / "done" / "good.json").exists()


def test_remote_command_probe_stage_and_explicit_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock transport verifies strict keys, source checks and opt-in detached startup."""
    root = tmp_path / "campaign"
    manifest = cluster.prepare(root)
    host = "leonhard-13"
    assert "StrictHostKeyChecking=yes" in cluster.ssh_arguments(host, manifest)
    with pytest.raises(ValueError, match="Host"):
        cluster.ssh_arguments("unknown", manifest)
    command = cluster.remote_command(manifest, "check")
    assert "OPENBLAS_NUM_THREADS=1" in command
    assert "nice -n 10" in command
    assert "PYTHONPATH=" in command
    assert " check " in command
    monkeypatch.setattr(cluster, "invoke", lambda _: json.dumps({"host": host, "packages": manifest["packages"]}))
    assert cluster.probe(host, manifest)["host"] == host
    with pytest.raises(ValueError, match="identity"):
        cluster.probe("leonhard-16", manifest)
    with pytest.raises(ValueError, match="Stage"):
        cluster.launch(root, manifest)
    # Keep the administrative archive test small; real preparation authenticates all sources.
    manifest["sources"] = {"pyproject.toml": digest(campaign.ROOT / "pyproject.toml")}
    calls = []
    monkeypatch.setattr(cluster, "invoke", lambda argv: calls.append(argv) or "12345")
    cluster.stage(root, manifest)
    assert not any("supervise" in " ".join(argv) for argv in calls)
    assert (root / "staged.json").exists()
    monkeypatch.setattr(cluster, "plan", lambda *_: {"hosts": {host: {"slots": 2}}})
    assert cluster.launch(root, manifest)[host]["supervisor_pid"] == "12345"
    assert any("nohup" in " ".join(argv) for argv in calls)
    with pytest.raises(ValueError, match="relaunch"):
        cluster.launch(root, manifest)


def test_transport_errors_and_supervisor_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker finishes under supervision and releases only its own host lock."""
    assert cluster.invoke([sys.executable, "-c", "print('ok')"]).strip() == "ok"
    with pytest.raises(subprocess.CalledProcessError):
        cluster.invoke([sys.executable, "-c", "raise SystemExit(2)"])
    root = tmp_path / "campaign"
    manifest = cluster.prepare(root)
    monkeypatch.setattr(cluster, "probe", lambda *_: (_ for _ in ()).throw(ValueError("unavailable")))
    assert not cluster.plan(root, manifest)["initial_assignment_preview"]
    manifest["jobs"] = [manifest["jobs"][0]]
    monkeypatch.setattr(cluster.socket, "gethostname", lambda: "leonhard-13")
    monkeypatch.setattr(cluster, "local_snapshot", dict)
    monkeypatch.setattr(cluster, "slots", lambda *_: 1)
    monkeypatch.setattr(cluster.time, "sleep", lambda _: None)

    def fake_guard(output: Path, _manifest: dict, job: dict) -> dict:
        write_json(output / "failed" / f"{job['id']}.json", {"status": "time_limit"})
        return {"status": "time_limit"}

    monkeypatch.setattr(cluster, "guarded_job", fake_guard)
    cluster.supervise(root, manifest)
    assert not (root / "supervisor-leonhard-13.lock").exists()
    assert cluster.status(root, manifest)[manifest["jobs"][0]["id"]] == "failed"


def test_local_capacity_and_cli_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse Linux memory units and exercise nonlaunch CLI paths with no experiment."""
    original_read = cluster.Path.read_text
    monkeypatch.setattr(cluster.Path, "read_text", lambda *_args, **_kwargs: "MemTotal: 200 kB\nMemAvailable: 100 kB\n")
    snapshot = cluster.local_snapshot()
    assert snapshot["total_bytes"] == 200 * 1024
    assert snapshot["available_bytes"] == 100 * 1024
    monkeypatch.setattr(cluster.Path, "read_text", original_read)
    root = tmp_path / "campaign"
    cluster.main(["prepare", "--root", str(root)])
    cluster.main(["check", "--root", str(root)])
    cluster.main(["status", "--root", str(root)])
    with pytest.raises(SystemExit, match="2"):
        cluster.main(["work", "--root", str(root)])
    monkeypatch.setattr(campaign, "execute", lambda *_: {"status": "mocked"})
    identity = json.loads((root / "manifest.json").read_text())["jobs"][0]["id"]
    cluster.main(["work", "--root", str(root), "--job", identity])


def test_first_wave_balances_long_jobs_and_can_steal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first node cannot claim all deeper Fontana runs before node two starts."""
    root = tmp_path / "campaign"
    manifest = cluster.prepare(root)
    monkeypatch.setattr(cluster.socket, "gethostname", lambda: "leonhard-13")
    first = [cluster.claim(root, manifest) for _ in range(16)]
    assert all(job is not None for job in first)
    assert sum(job["method"] == "fontana" for job in first if job is not None) == 6
    # If the second host is unavailable, work remains accessible to the first.
    stolen = cluster.claim(root, manifest)
    assert stolen is not None
    assert stolen["method"] == "fontana"
