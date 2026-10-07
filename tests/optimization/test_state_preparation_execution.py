# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded, single-threaded process execution of experiment jobs."""

from __future__ import annotations

import argparse
import os
from typing import cast

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from benchmarks.state_preparation import execution


def _probe(value: int) -> tuple[int, int, list[int], list[str | None]]:
    """Return execution identity and actual numerical-library thread counts."""
    _u, _s, _vh = np.linalg.svd(np.eye(8))
    return (
        value * value,
        os.getpid(),
        [pool["num_threads"] for pool in threadpool_info()],
        [os.environ.get(name) for name in execution.THREAD_ENVIRONMENT],
    )


def _fail(value: int) -> int:
    """Raise a picklable job error to exercise cleanup in both execution modes.

    Raises:
        ValueError: Always, to simulate a failing job.
    """
    msg = f"job {value} failed"
    raise ValueError(msg)


def test_worker_resolution_and_cli_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto uses available CPUs, never exceeding pending jobs or explicit caps."""
    monkeypatch.setattr(execution.os, "cpu_count", lambda: 8)
    assert execution.resolve_workers("auto", 20) == 8
    assert execution.resolve_workers("auto", 3) == 3
    assert execution.resolve_workers(2, 20) == 2
    assert execution.resolve_workers(100, 20) == 8
    assert execution.resolve_workers("auto", 0) == 0
    assert execution.parse_workers("auto") == "auto"
    assert execution.parse_workers("8") == 8
    for invalid in (True, False, 0, -1, 1.5, "8", "wrong"):
        with pytest.raises(ValueError, match="workers"):
            execution.resolve_workers(cast("execution.WorkerSetting", invalid), 2)
    for invalid in ("0", "-2", "wrong", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            execution.parse_workers(invalid)
    with pytest.raises(ValueError, match="job_count"):
        execution.resolve_workers(1, -1)
    monkeypatch.setattr(execution.os, "cpu_count", lambda: None)
    assert execution.resolve_workers("auto", 2) == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_execution_limits_threads_and_restores_environment(workers: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial and spawned execution keep one thread and restore parent settings."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    original = {name: os.environ.get(name) for name in execution.THREAD_ENVIRONMENT}
    pools_before = threadpool_info()
    results = sorted(execution.completed_jobs(_probe, list(range(4)), workers=workers))
    assert [result[0] for _index, result in results] == [0, 1, 4, 9]
    for _index, (_value, pid, threads, environment) in results:
        assert all(count == 1 for count in threads)
        assert environment == ["1"] * len(execution.THREAD_ENVIRONMENT)
        if execution.resolve_workers(workers, 4) > 1:
            assert pid != os.getpid()
        else:
            assert pid == os.getpid()
    assert {name: os.environ.get(name) for name in execution.THREAD_ENVIRONMENT} == original
    assert threadpool_info() == pools_before


@pytest.mark.parametrize("workers", [1, 2])
def test_job_errors_restore_parent_settings(workers: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures propagate without leaking thread settings into subsequent work."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    with pytest.raises(ValueError, match=r"job .* failed"):
        list(execution.completed_jobs(_fail, [1, 2], workers=workers))
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert list(execution.completed_jobs(_probe, [], workers=workers)) == []
