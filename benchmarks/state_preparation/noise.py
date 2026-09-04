# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen standard-noise profiles for the state-preparation benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import GateNoiseContext, TJMNoiseInstruction

from .constants import (
    MULTI_QUBIT_STRENGTH,
    ONE_QUBIT_STRENGTH,
    STANDARD_NOISE_IDS,
)

if TYPE_CHECKING:
    import numpy as np

    from mqt.yaqs.optimization import GateNoiseInstruction

NoiseFamily = Literal["dephasing", "depolarizing"]
NoiseSupport = Literal["single_site", "two_site", "single_site_two_site"]
NoisePlacement = Literal["single_qubit_gates", "multi_qubit_gates", "all_gates"]


@dataclass(frozen=True, slots=True)
class StandardNoiseProfile:
    """One frozen standard dephasing or depolarizing configuration."""

    noise_id: str
    family: NoiseFamily
    support: NoiseSupport
    placement: NoisePlacement

    @property
    def content_checksum(self) -> str:
        """Return a stable checksum of the scientific profile definition."""
        payload = {
            "family": self.family,
            "multi_qubit_strength": MULTI_QUBIT_STRENGTH,
            "noise_id": self.noise_id,
            "one_qubit_strength": ONE_QUBIT_STRENGTH,
            "placement": self.placement,
            "support": self.support,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


_PROFILES = {
    "dephasing_1s_1q": StandardNoiseProfile("dephasing_1s_1q", "dephasing", "single_site", "single_qubit_gates"),
    "dephasing_1s_2q": StandardNoiseProfile("dephasing_1s_2q", "dephasing", "single_site", "multi_qubit_gates"),
    "dephasing_1s_all": StandardNoiseProfile("dephasing_1s_all", "dephasing", "single_site", "all_gates"),
    "dephasing_2s_2q": StandardNoiseProfile("dephasing_2s_2q", "dephasing", "two_site", "multi_qubit_gates"),
    "dephasing_1s2s_all": StandardNoiseProfile("dephasing_1s2s_all", "dephasing", "single_site_two_site", "all_gates"),
    "depolarizing_1s_1q": StandardNoiseProfile(
        "depolarizing_1s_1q", "depolarizing", "single_site", "single_qubit_gates"
    ),
    "depolarizing_1s_2q": StandardNoiseProfile(
        "depolarizing_1s_2q", "depolarizing", "single_site", "multi_qubit_gates"
    ),
    "depolarizing_1s_all": StandardNoiseProfile("depolarizing_1s_all", "depolarizing", "single_site", "all_gates"),
    "depolarizing_2s_2q": StandardNoiseProfile("depolarizing_2s_2q", "depolarizing", "two_site", "multi_qubit_gates"),
    "depolarizing_1s2s_all": StandardNoiseProfile(
        "depolarizing_1s2s_all", "depolarizing", "single_site_two_site", "all_gates"
    ),
}


def standard_noise_profile(noise_id: str) -> StandardNoiseProfile:
    """Return one frozen standard-noise profile.

    Raises:
        ValueError: If the identifier is not part of the benchmark.
    """
    try:
        return _PROFILES[noise_id]
    except KeyError as error:
        msg = f"Unknown standard-noise identifier {noise_id!r}; expected one of {STANDARD_NOISE_IDS}."
        raise ValueError(msg) from error


def _applies(profile: StandardNoiseProfile, arity: int) -> bool:
    """Return whether a profile applies after a gate of this arity."""
    return (
        profile.placement == "all_gates"
        or (profile.placement == "single_qubit_gates" and arity == 1)
        or (profile.placement == "multi_qubit_gates" and arity == 2)
    )


def _single_site_processes(
    profile: StandardNoiseProfile,
    context: GateNoiseContext,
    strength: float,
) -> list[dict[str, object]]:
    """Return the profile's one-site jump processes."""
    labels = ("z",) if profile.family == "dephasing" else ("x", "y", "z")
    return [
        {"name": f"pauli_{label}", "sites": [site], "strength": strength} for site in context.sites for label in labels
    ]


def _two_site_processes(
    profile: StandardNoiseProfile,
    context: GateNoiseContext,
    strength: float,
) -> list[dict[str, object]]:
    """Return the profile's frozen two-site jump processes."""
    if context.arity != 2:
        return []
    labels = ("zz",) if profile.family == "dephasing" else tuple(left + right for left in "xyz" for right in "xyz")
    return [{"name": f"crosstalk_{label}", "sites": list(context.sites), "strength": strength} for label in labels]


@dataclass(frozen=True, slots=True)
class StandardNoiseProvider:
    """Build gate-local TJM models for one standard benchmark profile."""

    noise_id: str

    def __post_init__(self) -> None:
        """Validate the profile identifier."""
        standard_noise_profile(self.noise_id)

    @property
    def profile(self) -> StandardNoiseProfile:
        """Return the frozen profile definition."""
        return standard_noise_profile(self.noise_id)

    @property
    def content_checksum(self) -> str:
        """Return a stable checksum of the scientific profile definition."""
        return self.profile.content_checksum

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> GateNoiseInstruction | None:
        """Return a fresh gate-local TJM instruction or no instruction."""
        del rng
        profile = self.profile
        if not _applies(profile, context.arity):
            return None
        strength = ONE_QUBIT_STRENGTH if context.arity == 1 else MULTI_QUBIT_STRENGTH
        processes: list[dict[str, object]] = []
        if profile.support in {"single_site", "single_site_two_site"}:
            processes.extend(_single_site_processes(profile, context, strength))
        if profile.support in {"two_site", "single_site_two_site"}:
            processes.extend(_two_site_processes(profile, context, strength))
        if not processes:
            return None
        return TJMNoiseInstruction(NoiseModel(processes), channel_id=self.noise_id)


__all__ = [
    "NoiseFamily",
    "NoisePlacement",
    "NoiseSupport",
    "StandardNoiseProfile",
    "StandardNoiseProvider",
    "standard_noise_profile",
]
