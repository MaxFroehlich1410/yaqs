# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Executable state-preparation benchmark components."""

from __future__ import annotations

from .ballarin import (
    BallarinCircuitMaterialization,
    BallarinNoiseProvider,
    ballarin_epsilon,
    ballarin_pauli_probability,
    canonicalize_angle,
    materialize_ballarin_circuit,
)
from .constants import (
    BALLARIN_NOISE_ID,
    MINIMUM_NOISE_IDS,
    NOISE_IDS,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_IDS,
)
from .methods import KrotovBatchMethod, StatePreparationMethod, TrainingOutcome, TrainingSettings
from .noise import StandardNoiseProfile, StandardNoiseProvider, standard_noise_profile
from .statistics import CircuitStatistics, FidelityEstimate, circuit_statistics, fidelity_estimate
from .targets import TargetCollection, TargetRecord, iter_targets, load_target, load_target_collection

__all__ = [
    "BALLARIN_NOISE_ID",
    "MINIMUM_NOISE_IDS",
    "NOISE_IDS",
    "STANDARD_NOISE_IDS",
    "SUPPORTED_QUBIT_COUNTS",
    "TARGET_IDS",
    "BallarinCircuitMaterialization",
    "BallarinNoiseProvider",
    "CircuitStatistics",
    "FidelityEstimate",
    "KrotovBatchMethod",
    "StandardNoiseProfile",
    "StandardNoiseProvider",
    "StatePreparationMethod",
    "TargetCollection",
    "TargetRecord",
    "TrainingOutcome",
    "TrainingSettings",
    "ballarin_epsilon",
    "ballarin_pauli_probability",
    "canonicalize_angle",
    "circuit_statistics",
    "fidelity_estimate",
    "iter_targets",
    "load_target",
    "load_target_collection",
    "materialize_ballarin_circuit",
    "standard_noise_profile",
]
