#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Small-system stress Hamiltonians with independent MPO and bit-basis assembly.

Spin interactions use S = sigma/2. Tilted Ising uses Pauli operators. Power-law
couplings use ring chord distance; the other families have open boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import fixtures_n as fx
import numpy as np
from scipy import sparse

from mqt.yaqs.core.data_structures.mpo import MPO


@dataclass(frozen=True)
class Model:
    """Physical parameters in the unscaled Hamiltonian."""

    family: str
    alpha: float = 2.0
    delta: float = 1.0
    j2: float = 0.0
    hx: float = 1.05
    hz: float = 0.0


STRESS_MODELS = {
    **{f"power_a{a:g}_d{d:g}": Model("power", alpha=a, delta=d) for a in (1, 1.5, 2, 3) for d in (1, 2)},
    **{f"j1j2_{j:g}": Model("j1j2", j2=j) for j in (0.3, 0.5, 1)},
    **{f"tilted_ising_{h:g}": Model("tilted_ising", hz=h) for h in (0.2, 0.5, 1)},
    **{f"xxz_d{d:g}": Model("xxz", delta=d) for d in (2, 4, 8)},
}
MODELS = (*fx.MODELS, *STRESS_MODELS)
_SPIN = (
    np.array([[0, 1], [1, 0]], dtype=complex) / 2,
    np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
    np.array([[1, 0], [0, -1]], dtype=complex) / 2,
)


def coefficients(length: int, model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return spin couplings J[i,j,axis] and fields h[i,axis]."""
    pair = np.zeros((length, length, 3))
    field = np.zeros((length, 3))
    for i in range(length):
        if model.family == "tilted_ising":
            field[i] = (-2 * model.hx, 0, -2 * model.hz)
        for j in range(i + 1, length):
            if model.family == "power":
                chord = length / np.pi * np.sin(np.pi * (j - i) / length)
                pair[i, j] = np.array([1, 1, model.delta]) / chord**model.alpha
            elif model.family == "j1j2":
                pair[i, j] = 1 if j == i + 1 else model.j2 if j == i + 2 else 0
            elif model.family == "xxz":
                pair[i, j] = [1, 1, model.delta] if j == i + 1 else 0
            elif model.family == "tilted_ising":
                pair[i, j, 2] = -4 if j == i + 1 else 0
            else:
                raise ValueError(f"Unknown family: {model.family}")
    return pair, field


def pair_mpo(pair: np.ndarray, field: np.ndarray) -> MPO:
    """Build an exact FSM, retaining only channels with a future nonzero bond."""
    length = len(field)
    # Each bond has a waiting channel, open (site, axis) channels, and a completed channel.
    cuts = [["wait"]]
    for cut in range(1, length):
        cuts.append(["wait", *[(i, a) for i in range(cut) for a in range(3) if np.any(pair[i, cut:, a])], "done"])
    cuts.append(["done"])
    tensors = []
    for site in range(length):
        left, right = cuts[site], cuts[site + 1]
        block = np.zeros((2, 2, len(left), len(right)), dtype=complex)
        for il, channel in enumerate(left):
            if channel in right:
                block[:, :, il, right.index(channel)] += np.eye(2)
            if channel == "wait":
                if "done" in right:
                    block[:, :, il, right.index("done")] += sum(field[site, a] * _SPIN[a] for a in range(3))
                for a in range(3):
                    if (site, a) in right:
                        block[:, :, il, right.index((site, a))] += _SPIN[a]
            elif channel != "done" and "done" in right:
                start, axis = channel
                block[:, :, il, right.index("done")] += pair[start, site, axis] * _SPIN[axis]
        tensors.append(block)
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    return mpo


def bit_matrix(pair: np.ndarray, field: np.ndarray) -> sparse.csr_matrix:
    """Assemble independently through spin flips in the site-0-LSB basis."""
    length = len(field)
    size = 2**length
    rows, columns, values = [], [], []
    for ket in range(size):
        signs = [1 - 2 * ((ket >> i) & 1) for i in range(length)]
        diagonal = 0.0
        for i in range(length):
            diagonal += field[i, 2] * signs[i] / 2
            rows.append(ket ^ (1 << i))
            columns.append(ket)
            values.append((field[i, 0] + 1j * signs[i] * field[i, 1]) / 2)
            for j in range(i + 1, length):
                diagonal += pair[i, j, 2] * signs[i] * signs[j] / 4
                rows.append(ket ^ (1 << i) ^ (1 << j))
                columns.append(ket)
                values.append((pair[i, j, 0] - signs[i] * signs[j] * pair[i, j, 1]) / 4)
        rows.append(ket)
        columns.append(ket)
        values.append(diagonal)
    result = sparse.csr_matrix((values, (rows, columns)), shape=(size, size), dtype=complex)
    result.eliminate_zeros()
    return result


def hamiltonians(length: int, name: str, width_per_site: float | None = None) -> tuple[MPO, sparse.csr_matrix, dict]:
    """Build both representations and optionally fix spectral width per site.

    The scale is L * width_per_site / (E_max - E_min), computed from the exact
    small-system spectrum. Both operators receive the identical multiplier.
    No shift of the energy origin or approximate MPO compression is performed.
    """
    if name in fx.MODELS:
        mpo, matrix = fx.hamiltonian_mpo(length, name), fx.sparse_hamiltonian(length, name)
        parameters = {"family": name}
    else:
        model = STRESS_MODELS[name]
        pair, field = coefficients(length, model)
        mpo, matrix = pair_mpo(pair, field), bit_matrix(pair, field)
        parameters = asdict(model)
    spectrum = np.linalg.eigvalsh(matrix.toarray())
    width = float(spectrum[-1] - spectrum[0])
    if width_per_site is not None and (not np.isfinite(width_per_site) or width_per_site <= 0):
        raise ValueError("Spectral width per site must be finite and positive")
    scale = 1.0 if width_per_site is None else length * width_per_site / width
    mpo.tensors[0] *= scale
    matrix = matrix * scale
    return (
        mpo,
        matrix,
        {
            "parameters": parameters,
            "boundary": "ring chord" if name == "hs" or parameters["family"] == "power" else "open",
            "normalization": "none" if width_per_site is None else "spectral_width_per_site",
            "raw_min_energy": float(spectrum[0]),
            "raw_max_energy": float(spectrum[-1]),
            "raw_spectral_width": width,
            "scale": scale,
            "spectral_width_per_site": scale * width / length,
        },
    )
