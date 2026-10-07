# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hardware-error calibration for the local-depolarizing causal experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import GateNoiseContext, TJMNoiseInstruction

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class CalibratedLocalDepolarizingProvider:
    """Match exact local Pauli-Lindblad channels to average gate errors.

    This provider assumes a unit circuit-TJM window, ``dt=1``. It retains the
    recent experiment's three/six local Pauli operators and gate placement.
    It does not change the legacy standard profiles or the one-jump sampler.

    Attributes:
        one_qubit_error: Target complete one-qubit average gate infidelity.
        two_qubit_error: Target complete two-qubit average gate infidelity.
        one_qubit_strength: Integrated coefficient of each one-qubit process.
        two_qubit_strength: Integrated coefficient of each two-qubit process.
    """

    one_qubit_error: float = 0.00064
    two_qubit_error: float = 0.0051
    one_qubit_strength: float = field(init=False)
    two_qubit_strength: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate physical error targets and compute exact coefficients.

        Raises:
            ValueError: If a target is nonfinite or outside the finite-rate range.
        """
        for name, limit in (("one_qubit_error", 0.5), ("two_qubit_error", 0.75)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < limit:
                msg = f"{name} must be finite and in [0, {limit})."
                raise ValueError(msg)
        one = -math.log1p(-2 * self.one_qubit_error) / 4
        # Rationalize sqrt(1-x)-1 to avoid cancellation at weak noise.
        root = math.sqrt(1 - 1.25 * self.two_qubit_error)
        two = -math.log1p(-5 * self.two_qubit_error / (3 * (1 + root))) / 4
        object.__setattr__(self, "one_qubit_strength", one)
        object.__setattr__(self, "two_qubit_strength", two)

    def scientific_dict(self) -> dict[str, object]:
        """Return complete calibration and finite-step error metadata."""
        return {
            "version": "local_depolarizing_average_gate_infidelity.v1",
            "noise_id": "depolarizing_1s_all",
            "metric": "average_gate_infidelity",
            "calibration_channel": "exact_pauli_lindblad_exponential",
            "execution_channel": "circuit_tjm_at_most_one_jump",
            "dt": 1.0,
            "one_qubit_error": self.one_qubit_error,
            "two_qubit_error": self.two_qubit_error,
            "one_qubit_strength": self.one_qubit_strength,
            "two_qubit_strength": self.two_qubit_strength,
            "one_qubit_operators": ["X", "Y", "Z"],
            "two_qubit_operators": [f"{p}I" for p in "XYZ"] + [f"I{p}" for p in "XYZ"],
            "one_jump_one_qubit_error": -2 / 3 * math.expm1(-3 * self.one_qubit_strength),
            "one_jump_two_qubit_error": -4 / 5 * math.expm1(-6 * self.two_qubit_strength),
            "placement": "after_every_abstract_gate_on_its_support",
            "scope": "hardware_motivated_homogeneous_model_not_native_device_calibration",
            "reference": "https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf",
        }

    @property
    def content_checksum(self) -> str:
        """Return a stable identity distinct from every legacy profile."""
        encoded = json.dumps(self.scientific_dict(), sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def __call__(self, context: GateNoiseContext, rng: np.random.Generator, /) -> TJMNoiseInstruction:
        """Build independent local Pauli generators for the active gate.

        Returns:
            A unit-window TJM instruction with calibrated per-operator rates.
        """
        del rng
        strength = self.one_qubit_strength if context.arity == 1 else self.two_qubit_strength
        processes = [
            {"name": f"pauli_{label}", "sites": [site], "strength": strength}
            for site in context.sites
            for label in "xyz"
        ]
        return TJMNoiseInstruction(
            NoiseModel(processes),
            channel_id="depolarizing_1s_all:average_gate_infidelity_v1",
        )
