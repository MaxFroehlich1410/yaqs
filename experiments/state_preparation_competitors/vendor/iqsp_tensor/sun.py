# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2020-present TorchQuantum Authors
# SPDX-License-Identifier: MIT

"""SU(4) exponential adapted from the pinned author's ``functional/sun.py``.

The generator order and positive exponential sign are unchanged. This extract
removes TorchQuantum registration/batching and uses explicit complex128.
See LICENSE and PROVENANCE.md in this directory.
"""

from __future__ import annotations

import torch


def su4_matrix(parameters: torch.Tensor) -> torch.Tensor:
    """Return ``exp(+i sum(theta_ab P_a tensor P_b))``, excluding identity.

    Args:
        parameters: Fifteen real coefficients in row-major Pauli-product order, excluding identity.

    Returns:
        A complex128 four-dimensional unitary in the supplied local basis.

    Raises:
        ValueError: If the parameter shape or values are invalid.
    """
    if parameters.shape != (15,) or parameters.is_complex() or not bool(torch.isfinite(parameters).all()):
        msg = "SU4 requires fifteen finite real coefficients."
        raise ValueError(msg)
    paulis = torch.tensor(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
        device=parameters.device,
    )
    generators = torch.stack([torch.kron(paulis[index // 4], paulis[index % 4]) for index in range(1, 16)])
    return torch.matrix_exp(1j * torch.einsum("p,pij->ij", parameters.to(torch.complex128), generators))
