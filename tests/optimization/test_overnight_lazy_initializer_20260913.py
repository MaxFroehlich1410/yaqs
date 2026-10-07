# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test real lazy branching, exact old-policy endpoints and complete attempt costs."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from experiments import state_preparation_pilot as pilot
from experiments.overnight_state_preparation_20260913_0108 import initializer_policy
from experiments.overnight_state_preparation_20260913_0108 import lazy_initializer as lazy

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze temporary metadata without changing any numerical primitive."""
    monkeypatch.setattr(lazy, "sources", lambda: {"kernel": "fixed"})
    monkeypatch.setattr(lazy.construction, "environment", lambda: {"runtime": "fixed"})
    monkeypatch.setattr(lazy.runner, "custody", lambda: {"benchmark_hashes": {"immutable": "fixed"}})


def target(n: int) -> np.ndarray:
    """Return only small synthetic inputs, including an explicit q4 rank-loss case."""
    if n == 4:
        vector = np.zeros(16, dtype=complex)
        vector[[0, 5, 10, 15]] = 0.5
        return vector
    rng = np.random.default_rng(130 + n)
    vector = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return vector / np.linalg.norm(vector)


def legacy(n: int) -> dict[str, Any]:
    """Return a synthetic physical scalar endpoint for old-policy parity checks."""
    circuit = lazy.construction.circuit_at_depth(n, 1)
    return {
        "circuit": pilot.circuit_record(circuit),
        "theta": np.linspace(-0.3, 0.4, circuit.num_params).tolist(),
        "construction_cpu_seconds": 0.001,
        "construction_wall_seconds": 0.001,
        "timing_complete": True,
    }


def factory(vector: np.ndarray, source: dict[str, Any], calls: list[Path]) -> Callable[[Path], dict[str, Any]]:
    """Return an engineering-only constructor with real owned result/ledger receipts.

    Returns:
        A durable synthetic callback; it never simulates or loads a q6 state.
    """

    def construct(root: Path) -> dict[str, Any]:
        calls.append(root)
        if (root / "RESULT.json").exists():
            return lazy.read(root / "RESULT.json")
        cpu, wall = time.process_time(), time.perf_counter()
        protocol: dict[str, Any] = {
            "target_id": "synthetic_case",
            "replica": 11,
            "target_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        }
        pilot.write_record(root / "PROTOCOL.json", protocol)
        used_cpu, used_wall = time.process_time() - cpu, time.perf_counter() - wall
        ledger: dict[str, Any] = {
            "signature": pilot.digest(protocol),
            "attempts": [{"status": "complete", "cpu_seconds": used_cpu, "wall_seconds": used_wall}],
        }
        pilot.write_record(root / "ATTEMPTS.json", ledger)
        result = {
            **source,
            "status": "complete",
            "target_id": "synthetic_case",
            "replica": 11,
            "signature": pilot.digest(protocol),
            "attempts_checksum": pilot.digest(ledger),
            "blocks": {},
            "construction_cpu_seconds": used_cpu,
            "construction_wall_seconds": used_wall,
        }
        pilot.write_record(root / "RESULT.json", result)
        return lazy.read(root / "RESULT.json")

    return construct


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("n", [1, 2, 3])
def test_rank_two_branch_never_invokes_legacy_and_matches_frozen_policy(tmp_path: Path, n: int) -> None:
    """The selected physical circuit/angles agree exactly while unused construction disappears."""
    vector, source = target(n), legacy(n)
    expected = initializer_policy.build_initializers(vector, source)
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    result = lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("selected rank two invoked legacy"))
    assert result["rank_two_selected"]
    assert result["ideal_tt_overlap"] == expected["ideal_tt_overlap"]
    assert pilot.digest(result["endpoint"]["circuit"]) == pilot.digest(expected["conditional"]["circuit"])
    assert result["endpoint"]["theta"] == expected["conditional"]["theta"]
    assert not (tmp_path / "legacy").exists()
    assert result["nested_legacy_cpu_seconds"] is None
    assert result["initializer_cpu_seconds"] > 0
    assert result["initializer_cpu_seconds"] == result["endpoint"]["initializer_cpu_seconds"]
    assert lazy.read(tmp_path / "ATTEMPTS.json")["attempts"][0]["constructor_called"] is False


@pytest.mark.usefixtures("custody")
def test_rank_loss_calls_owned_constructor_once_and_retains_probe_cost(tmp_path: Path) -> None:
    """A true rank-loss case takes the original compiled fallback with full physical accounting."""
    vector, source, calls = target(4), legacy(4), []
    expected = initializer_policy.build_initializers(vector, source)
    assert not expected["rank_two_selected"]
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    result = lazy.run(tmp_path, manifest, vector, factory(vector, source, calls))
    assert calls == [tmp_path / "legacy"]
    assert not result["rank_two_selected"]
    assert result["discarded_probe_cpu_seconds"] > 0
    assert result["ideal_tt_overlap"] == expected["ideal_tt_overlap"]
    assert result["endpoint"]["theta"] == expected["conditional"]["theta"]
    assert pilot.digest(result["endpoint"]["circuit"]) == pilot.digest(expected["conditional"]["circuit"])
    ledger = lazy.read(tmp_path / "ATTEMPTS.json")
    assert result["initializer_cpu_seconds"] == pytest.approx(sum(a["cpu_seconds"] for a in ledger["attempts"]))
    assert result["initializer_cpu_seconds"] > result["nested_legacy_cpu_seconds"]
    assert result["initializer_cpu_seconds"] > result["probe_cpu_seconds"]
    assert lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("cached callback")) == result


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("stage", ["probe", "constructor"])
def test_caught_interruption_reuses_owned_stages_and_charges_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Exact stage outputs survive interruption without rerunning the physical rank-two probe."""
    vector, source, calls = target(4), legacy(4), []
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    real_write = pilot.write_record
    selected = tmp_path / ("PROBE.json" if stage == "probe" else "legacy/RESULT.json")

    def interrupt(path: Path, record: dict[str, Any]) -> None:
        real_write(path, record)
        if path == selected:
            raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        lazy.run(tmp_path, manifest, vector, factory(vector, source, calls))
    monkeypatch.setattr(pilot, "write_record", real_write)
    monkeypatch.setattr(lazy, "probe", lambda _: pytest.fail("completed probe replayed"))
    result = lazy.run(tmp_path, manifest, vector, factory(vector, source, calls))
    ledger = lazy.read(tmp_path / "ATTEMPTS.json")
    assert [a["status"] for a in ledger["attempts"]] == ["interrupted", "complete"]
    assert result["initializer_cpu_seconds"] == pytest.approx(sum(a["cpu_seconds"] for a in ledger["attempts"]))
    assert all(a["cpu_seconds"] > 0 for a in ledger["attempts"])
    assert len(calls) == (1 if stage == "probe" else 2)


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("changed", ["target", "source", "probe", "unknown", "orphan"])
def test_changed_custody_or_unknown_work_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    """No numerical restart can bypass changed inputs, orphan stages or unknown costs."""
    vector = target(2)
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    if changed in {"probe", "unknown"}:
        lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("legacy"))
    if changed == "target":
        vector = vector.copy()
        vector[0] *= -1
    elif changed == "source":
        monkeypatch.setattr(lazy, "sources", lambda: {"kernel": "changed"})
    elif changed == "probe":
        record = lazy.read(tmp_path / "PROBE.json")
        record["ideal_tt_overlap"] -= 0.01
        pilot.write_record(tmp_path / "PROBE.json", record)
    elif changed == "unknown":
        ledger = lazy.read(tmp_path / "ATTEMPTS.json")
        ledger["attempts"][0].update(status="running", cpu_seconds=None, wall_seconds=None)
        pilot.write_record(tmp_path / "ATTEMPTS.json", ledger)
    else:
        pilot.write_record(tmp_path / "PROBE.json", {"orphan": True})
    monkeypatch.setattr(lazy, "probe", lambda _: pytest.fail("invalid case ran probe"))
    with pytest.raises(ValueError, match=r"target|custody|changed|Unknown|without"):
        lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("invalid callback"))


@pytest.mark.usefixtures("custody")
def test_probe_numerical_failure_cannot_trigger_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure is an explicit paid failed attempt, never an alternative architecture selector."""
    vector = target(2)
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})

    def fail(_: np.ndarray) -> dict[str, Any]:
        msg = "numerical probe failed"
        raise FloatingPointError(msg)

    monkeypatch.setattr(lazy, "probe", fail)
    with pytest.raises(FloatingPointError, match="probe failed"):
        lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("failure triggered fallback"))
    attempt = lazy.read(tmp_path / "ATTEMPTS.json")["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["cpu_seconds"] > 0
    assert not attempt["constructor_called"]


@pytest.mark.usefixtures("custody")
def test_confirmation_requires_parent_and_development_rejects_heldout(tmp_path: Path) -> None:
    """A phase label alone cannot access confirmation or use an unvalidated constructor."""
    vector = np.ones(64, dtype=complex) / 8
    with pytest.raises(ValueError, match="parent"):
        lazy.prepare(
            tmp_path / "confirmation", vector, "tfim_ferro", 11, phase="confirmation", constructor={"fixture": True}
        )
    with pytest.raises(ValueError, match="development"):
        lazy.prepare(tmp_path / "heldout", vector, "tfim_ferro", 11, phase="development", constructor={"fixture": True})
    with pytest.raises(ValueError, match="configuration"):
        lazy.prepare(
            tmp_path / "changed",
            vector,
            lazy.construction.TARGETS[0],
            11,
            phase="development",
            constructor={"fixture": True},
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("field", ["theta", "overlap", "cost", "branch"])
def test_complete_result_is_reconstructed_from_stages_before_cache_return(tmp_path: Path, n: int, field: str) -> None:
    """Rewriting a correctly serialized result cannot detach endpoints or claimed costs from evidence."""
    vector, calls = target(n), []
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    result = lazy.run(tmp_path, manifest, vector, factory(vector, legacy(n), calls))
    if field == "theta":
        result["endpoint"]["theta"][0] += 0.1
    elif field == "overlap":
        result["ideal_tt_overlap"] += 0.01
    elif field == "cost":
        result["initializer_cpu_seconds"] += 1
        result["endpoint"]["initializer_cpu_seconds"] += 1
    else:
        result["rank_two_selected"] = not result["rank_two_selected"]
    pilot.write_record(tmp_path / "RESULT.json", result)
    with pytest.raises(ValueError, match="authenticated stages"):
        lazy.run(tmp_path, manifest, vector, lambda _: pytest.fail("tampered cache constructed"))


@pytest.mark.usefixtures("custody")
@pytest.mark.parametrize("field", ["sum", "negative", "unknown"])
def test_fallback_receipt_requires_complete_real_ledger_totals(tmp_path: Path, field: str) -> None:
    """A callback cannot claim a complete but inconsistent or unknown constructor cost."""
    vector = target(4)
    manifest = lazy.prepare(tmp_path, vector, "synthetic_case", 11, phase="engineering", constructor={"fixture": True})
    child = tmp_path / "legacy"
    result = factory(vector, legacy(4), [])(child)
    ledger = lazy.read(child / "ATTEMPTS.json")
    if field == "sum":
        result["construction_cpu_seconds"] += 1
    elif field == "negative":
        ledger["attempts"][0]["cpu_seconds"] = -1
    else:
        ledger["attempts"][0]["wall_seconds"] = None
    pilot.write_record(child / "ATTEMPTS.json", ledger)
    result["attempts_checksum"] = pilot.digest(ledger)
    pilot.write_record(child / "RESULT.json", result)
    with pytest.raises(ValueError, match="authenticated complete constructor"):
        lazy.receipt(child, result, manifest)
