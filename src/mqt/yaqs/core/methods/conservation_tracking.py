# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run-local counters for conservation recovery, including explicit integration loops."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class ConservationStats:
    """Counts of correction calls, not physical timesteps.

    BUG calls the correction twice per timestep, TDVP once. ``fallback_attempts``
    counts rejected scalar corrections that enter recovery; ``fallback_successes``
    counts successful restorations. Unresolved attempts are partitioned into
    ``infeasible``, ``solver_failures``, ``no_minimizer``, and ``gauge_failures``.
    Multi-observable joint solves are counted separately; the scalar fallback is
    not applied to a failed joint constraint system.
    """

    scalar_calls: int = 0
    scalar_successes: int = 0
    already_satisfied: int = 0
    fallback_attempts: int = 0
    scan_sites: int = 0
    scan_successes: int = 0
    frames_checked: int = 0
    frame_attempts: int = 0
    frame_successes: int = 0
    infeasible: int = 0
    solver_failures: int = 0
    no_minimizer: int = 0
    gauge_failures: int = 0
    joint_calls: int = 0
    joint_unresolved: int = 0

    @property
    def fallback_successes(self) -> int:
        """Number of recovery calls that restored the target."""
        return self.scan_successes + self.frame_successes

    @property
    def fallback_failures(self) -> int:
        """Number of recovery calls that retained the compressed state."""
        return self.fallback_attempts - self.fallback_successes

    def merge(self, other: ConservationStats) -> None:
        """Add counters returned by another trajectory or worker."""
        for item in fields(self):
            setattr(self, item.name, getattr(self, item.name) + getattr(other, item.name))

    def to_dict(self) -> dict[str, int]:
        """Return JSON-ready counts, including aggregate successes and failures.

        Returns:
            Independent dictionary of counters.
        """
        return {
            **asdict(self),
            "fallback_successes": self.fallback_successes,
            "fallback_failures": self.fallback_failures,
        }


_ACTIVE_STATS: ContextVar[ConservationStats | None] = ContextVar("conservation_stats", default=None)


def current_conservation_stats() -> ConservationStats | None:
    """Return the collector for the current run, if tracking is active.

    Returns:
        Current collector or ``None``.
    """
    return _ACTIVE_STATS.get()


@contextmanager
def track_conservation() -> Iterator[ConservationStats]:
    """Collect a fresh set of counters for an explicit integration loop.

    Simulator workers use the same scope and return the counters to the parent
    process. Nested scopes are independent, and cleanup also runs on exceptions.

    Yields:
        Mutable collector, usable after the context exits.
    """
    stats = ConservationStats()
    token = _ACTIVE_STATS.set(stats)
    try:
        yield stats
    finally:
        _ACTIVE_STATS.reset(token)
