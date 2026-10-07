# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded numerical audits required before a twelve-qubit competitor campaign.

These checks never substitute reference gradients into training. A failed audit
is a readiness failure, not a poor benchmark fidelity. The identity-origin
limitation is retained and measured rather than hidden by parameter jitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from benchmarks.state_preparation.targets import load_target
from experiments.state_preparation_competitors.iqsp_comb_backend import (
    CombLayout,
    CombOptions,
    CombTensorNetworkObjective,
    comb_target_from_vector,
)
from experiments.state_preparation_competitors.iqsp_tci import TCIOptions, tci_target_bank
from experiments.state_preparation_competitors.iqsp_tn import GAUSSIAN_TARGET_ID, interpolation_target, iqsp_tn_ansatz
from experiments.state_preparation_competitors.iqsp_tn_backend import CircuitTensorGates
from experiments.state_preparation_exact_adam import AdamConfig, adam_step

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class StatevectorAuditObjective(CircuitTensorGates):
    """Independent exact ket contraction used solely as a small-system oracle."""

    def loss_gradient(self, theta: ArrayLike, target: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Differentiate the exact untruncated SU(4) circuit on at most twelve sites."""
        n = self.template.num_qubits
        if n > 12 or self.hardware is not None:
            msg = "The statevector audit is noiseless and bounded to twelve qubits."
            raise ValueError(msg)
        vector = np.asarray(target, dtype=np.complex128)
        if vector.shape != (2**n,) or not np.isfinite(vector).all() or abs(np.vdot(vector, vector) - 1) > 1e-10:
            msg = "Require a normalized finite target matching the audit circuit."
            raise ValueError(msg)
        values = self._theta(theta, gradients=True)
        state = torch.zeros(2**n, dtype=torch.complex128)
        state[0] = 1
        for index, item in enumerate(self.template.data):
            axes = [n - 1 - self.template.find_bit(q).index for q in reversed(item.qubits)]
            remaining = [axis for axis in range(n) if axis not in axes]
            order = axes + remaining
            transformed = self._matrix(index, values) @ state.reshape((2,) * n).permute(order).reshape(
                2 ** len(axes), -1
            )
            state = transformed.reshape((2,) * n).permute(np.argsort(order).tolist()).reshape(-1)
        loss = 1 - torch.abs(torch.vdot(torch.tensor(vector), state)).square()
        gradient = torch.autograd.grad(loss, values)[0].detach().numpy() if self.parameters else np.empty(0)
        return float(loss.detach()), gradient


def audit_gaussian_initialization(
    *,
    num_qubits: int = 12,
    depth: int = 1,
    steps: int = 20,
    options: CombOptions | None = None,
    tci_options: TCIOptions | None = None,
) -> dict[str, Any]:
    """Check actual Gaussian startup with the unchanged zero initialization.

    A bounded set of normal Adam updates at lambda=0.05 must make progress and recover
    gradient agreement. The distorted derivative at exactly zero is recorded,
    never used to claim exact equivalence. Failure prevents a campaign launch.

    Args:
        num_qubits: Existing six- or twelve-qubit Gaussian fixture.
        depth: Declared even/odd SU(4) layer count.
        steps: Positive bounded startup budget, at most twenty updates.
        options: Explicit author compression settings.
        tci_options: Use the published TCI bridge; None retains the historical SVD audit.

    Returns:
        Auditable fidelity/gradient errors and an explicit readiness decision.

    Raises:
        ValueError: If the fixture or startup budget is outside this audit's scope.
    """
    if num_qubits not in {6, 12} or type(steps) is not int or not 1 <= steps <= 20:
        msg = "Use q6/q12 and one to twenty startup updates."
        raise ValueError(msg)
    target = interpolation_target(
        load_target(num_qubits, GAUSSIAN_TARGET_ID).state_vector, 0.05, target_id=GAUSSIAN_TARGET_ID
    )
    circuit = iqsp_tn_ansatz(num_qubits, depth)
    selected_options = options or CombOptions()
    construction = None
    if tci_options is not None:
        if selected_options.target_tolerance != 0:
            msg = "TCI cannot use a dense-SVD target_tolerance."
            raise ValueError(msg)
        targets, construction = tci_target_bank([target], gaussian_coordinates=[0.05], options=tci_options)
        target_comb = targets[0]
    else:
        target_comb = comb_target_from_vector(
            target, CombLayout(num_qubits), relative_tolerance=selected_options.target_tolerance
        )
    objective = CombTensorNetworkObjective(
        circuit,
        target_comb,
        options=selected_options,
    )
    reference = StatevectorAuditObjective(circuit)
    theta = np.zeros(circuit.num_parameters)
    first, second = theta.copy(), theta.copy()
    config = AdamConfig(updates=steps, learning_rate=0.01)
    records = []
    for iteration in range(steps + 1):
        loss, gradient = objective.loss_gradient(theta)
        exact_loss, exact_gradient = reference.loss_gradient(theta, target)
        records.append({
            "iteration": iteration,
            "fidelity": 1 - loss,
            "exact_fidelity": 1 - exact_loss,
            "fidelity_error": abs(loss - exact_loss),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "relative_gradient_error": float(
                np.linalg.norm(gradient - exact_gradient) / max(np.linalg.norm(exact_gradient), 1e-14)
            ),
        })
        if iteration < steps:
            theta, first, second = adam_step(theta, gradient, first, second, iteration + 1, config)
    passed = (
        max(row["exact_fidelity"] for row in records[1:]) > records[0]["exact_fidelity"] + 1e-6
        and max(row["fidelity_error"] for row in records) < 1e-5
        and records[-1]["relative_gradient_error"] < 0.01
    )
    return {
        "passed": bool(passed),
        "scope": "Gaussian first interpolation step only; not a global gradient guarantee",
        "initialization": "unchanged zero exponential-SU4 parameters after Hadamards",
        "gradient_fallback": False,
        "num_qubits": num_qubits,
        "depth": depth,
        "records": records,
        "final_theta": theta.tolist(),
        "tensor_diagnostics": objective.diagnostics(theta),
        "target_construction_receipt": construction,
    }
