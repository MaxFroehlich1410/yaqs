#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
# SPDX-License-Identifier: MIT
# Licensed under the MIT License

"""Independent full-state reference without a dense 2**L by 2**L matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import fallback_models as models
import fixtures_n as fx
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply

from mqt.yaqs.core.data_structures.mps import MPS


@dataclass(frozen=True)
class Accuracy:
    """User-adjustable error thresholds, in normalized infidelity and spin units."""

    infidelity: float = 1e-3
    observable: float = 1e-3
    reference: float = 1e-9

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in (self.infidelity, self.observable, self.reference)):
            raise ValueError("Accuracy thresholds must be finite and positive")


class SpinOperator(LinearOperator):
    """Apply spin-half fields and same-axis pair couplings by bit flips.

    Only O(L * 2**L) diagonal/sign data are stored. Site zero is the least
    significant bit. Operator actions are independent of the MPO contraction.
    All terms are traceless, allowing an exact traceA=0 for expm_multiply.
    """

    def __init__(self, pair, field):
        self.length = len(field)
        self.indices = np.arange(2**self.length, dtype=np.int64)
        self.signs = np.array([1 - 2 * ((self.indices >> i) & 1) for i in range(self.length)], dtype=np.int8)
        self.diagonal = np.zeros(len(self.indices))
        self.flips = []
        for i in range(self.length):
            self.diagonal += field[i, 2] * self.signs[i] / 2
            if field[i, 0] or field[i, 1]:
                self.flips.append((1 << i, field[i, 0] / 2, 1j * field[i, 1] / 2, (i,)))
            for j in range(i + 1, self.length):
                self.diagonal += pair[i, j, 2] * self.signs[i] * self.signs[j] / 4
                if pair[i, j, 0] or pair[i, j, 1]:
                    self.flips.append(((1 << i) | (1 << j), pair[i, j, 0] / 4, -pair[i, j, 1] / 4, (i, j)))
        super().__init__(dtype=np.dtype(complex), shape=(len(self.indices), len(self.indices)))

    def _matvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        result = self.diagonal * vector
        for mask, constant, signed, sites in self.flips:
            sign = self.signs[sites[0]] if len(sites) == 1 else self.signs[sites[0]] * self.signs[sites[1]]
            result = result.astype(complex, copy=False)
            result[self.indices ^ mask] += (constant + signed * sign) * vector
        return result.astype(complex, copy=False)

    def _rmatvec(self, vector):
        return self._matvec(vector)


def model_coefficients(length, name):
    """Resolve supported legacy, stress, and dimer-quench Hamiltonians."""
    if name.startswith("dimer_"):
        weak = float(name.removeprefix("dimer_"))
        if not 0 <= weak <= 1:
            raise ValueError("Dimer weak/strong coupling must lie in [0, 1]")
        pair, field = np.zeros((length, length, 3)), np.zeros((length, 3))
        for i in range(length - 1):
            pair[i, i + 1] = 1.0 if i % 2 == 0 else weak
        return pair, field, {"family": "dimerized_heisenberg", "strong": 1.0, "weak": weak, "boundary": "open"}
    legacy = {
        "hs": models.Model("power", alpha=2, delta=1),
        "xxx": models.Model("xxz", delta=1),
        "xxz": models.Model("xxz", delta=fx.DELTA_XXZ),
        "tfim": models.Model("tilted_ising", hz=0),
    }
    specification = legacy.get(name) or models.STRESS_MODELS.get(name)
    if specification is None:
        raise ValueError(f"Unsupported reference model: {name}")
    pair, field = models.coefficients(length, specification)
    return pair, field, asdict(specification)


def spectral_edges(operator):
    """Compute reproducible extremal Ritz values and verify their residuals."""
    start = np.random.default_rng(719).normal(size=operator.shape[0])
    edges, residuals = [], []
    for which in ("SA", "LA"):
        values, vectors = eigsh(operator, k=1, which=which, v0=start, tol=1e-12, maxiter=10000)
        value, vector = float(values[0]), vectors[:, 0]
        residual = float(np.linalg.norm(operator @ vector - value * vector))
        if residual > 1e-10 * max(1, abs(value)):
            raise ValueError(f"Unconverged spectral normalization: {residual}")
        edges.append(value)
        residuals.append(residual)
    if edges[1] <= edges[0]:
        raise ValueError("Hamiltonian must have a positive spectral width")
    return edges, residuals


def hamiltonians(length, name, width_per_site=1.0, *, scale=None):
    """Build independent representations; reuse a saved scale for event audits."""
    pair, field, parameters = model_coefficients(length, name)
    reference = SpinOperator(pair, field)
    metadata = {
        "parameters": parameters,
        "reference": "matrix_free_bit_basis",
        "normalization": "spectral_width_per_site",
    }
    if scale is None:
        edges, residuals = spectral_edges(reference)
        scale = 1.0 if width_per_site is None else length * width_per_site / (edges[1] - edges[0])
        metadata.update(
            raw_min_energy=edges[0],
            raw_max_energy=edges[1],
            spectral_residuals=residuals,
            spectral_width_per_site=scale * (edges[1] - edges[0]) / length,
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Hamiltonian scale must be finite and positive")
    mpo = models.pair_mpo(pair, field)
    mpo.tensors[0] *= scale
    metadata["scale"] = scale
    return mpo, scale * reference, metadata


def mpo_action(mpo, vector):
    """Contract an MPO into a full vector using O(D_MPO * 2**L) storage."""
    block = np.asarray(vector).reshape(1, 1, -1)
    for tensor in mpo.tensors:
        out_prefix, bond, remaining = block.shape
        block = np.einsum(
            "abri,oibc->oacr", block.reshape(out_prefix, bond, remaining // 2, 2), tensor, optimize=True
        ).reshape(2 * out_prefix, tensor.shape[3], remaining // 2)
    return block.reshape(-1)


def check_operators(mpo, reference):
    """Check operator actions on two independent, deterministic full-state probes."""
    rng = np.random.default_rng(843)
    error = 0.0
    for _ in range(2):
        vector = rng.normal(size=reference.shape[0]) + 1j * rng.normal(size=reference.shape[0])
        vector /= np.linalg.norm(vector)
        expected = reference @ vector
        error = max(error, float(np.linalg.norm(mpo_action(mpo, vector) - expected) / max(1, np.linalg.norm(expected))))
    if error > 1e-11:
        raise ValueError(f"Independent operator actions disagree: {error}")
    return error


def singlet_product(length, *, seed=fx.SEED, pad=False):
    """Prepare a canonical product of singlets, the decoupled-dimer ground state."""
    if length % 2 or length < 2:
        raise ValueError("Singlet products require a positive even length")
    tensors = []
    for _ in range(length // 2):
        tensors.append((np.eye(2) / np.sqrt(2)).astype(complex).reshape(2, 1, 2))
        tensors.append(np.array([[0, -1], [1, 0]], dtype=complex).reshape(2, 2, 1))
    if pad:
        rng = np.random.default_rng(seed)
        for bond in range(length - 1):
            left, right = tensors[bond], tensors[bond + 1]
            old = left.shape[2]
            if old >= 2:
                continue
            padded_left = np.zeros((2, left.shape[1], 2), dtype=complex)
            padded_right = np.zeros((2, 2, right.shape[2]), dtype=complex)
            padded_left[:, :, :old], padded_right[:, :old, :] = left, right
            for block in (padded_left[:, :, old:], padded_right[:, old:, :]):
                block[:] = fx.NOISE_SCALE * (rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape))
            tensors[bond], tensors[bond + 1] = padded_left, padded_right
    state = MPS(length, tensors=tensors)
    # Right canonicalize without dropping the exact Schmidt rank.
    for site in range(length - 1, 0, -1):
        state.set_center(site)
        state.shift_orthogonality_center_left(site)
    state.set_center(0)
    state.tensors[0] /= np.linalg.norm(state.tensors[0])
    return state


class Observables:
    """Measure all local spins and nearest-neighbor same-axis spin correlations."""

    def __init__(self, length):
        self.length = length
        self.indices = np.arange(2**length)
        self.signs = [1 - 2 * ((self.indices >> i) & 1) for i in range(length)]

    def values(self, vector):
        """Return Sx,Sy,Sz per site, then SxSx,SySy,SzSz per nearest-neighbor bond."""
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
            raise ValueError("Observable measurement requires a finite nonzero state")
        v = np.asarray(vector) / np.linalg.norm(vector)
        probability = abs(v) ** 2
        values = []
        for i in range(self.length):
            flipped = v[self.indices ^ (1 << i)]
            values.extend([
                np.vdot(flipped, v).real / 2,
                np.vdot(flipped, 1j * self.signs[i] * v).real / 2,
                np.dot(probability, self.signs[i]).real / 2,
            ])
        for i in range(self.length - 1):
            sign = self.signs[i] * self.signs[i + 1]
            flipped = v[self.indices ^ (1 << i) ^ (1 << (i + 1))]
            values.extend([
                np.vdot(flipped, v).real / 4,
                -np.vdot(flipped, sign * v).real / 4,
                np.dot(probability, sign).real / 4,
            ])
        return np.array(values, dtype=float)

    def compare(self, vector, reference, thresholds):
        """Flag accuracy using both global infidelity and the worst observable error."""
        overlap = abs(np.vdot(vector, reference)) ** 2 / (
            np.vdot(vector, vector).real * np.vdot(reference, reference).real
        )
        infidelity = float(np.clip(1 - overlap, 0, 1))
        obs = self.values(vector)
        error = float(np.max(np.abs(obs - self.values(reference))))
        return {
            "infidelity": infidelity,
            "max_observable_error": error,
            "within_thresholds": infidelity <= thresholds.infidelity and error <= thresholds.observable,
        }


def propagate(operator, vector, duration):
    """Propagate a single full-state vector with a known zero operator trace."""
    return expm_multiply((-1j * duration) * operator, vector, traceA=0.0)


def event_time(case, event):
    """Locate a BUG half-step or TDVP end-step correction on the physical clock."""
    if case["integrator"] not in ("bug", "tdvp"):
        raise ValueError("Unknown event integrator")
    calls = 2 if case["integrator"] == "bug" else 1
    call = event["call_in_step"]
    if not 1 <= call <= calls or event["step"] < 1:
        raise ValueError("Invalid correction position")
    return (event["step"] - 1 + call / calls) * case["dt"]
