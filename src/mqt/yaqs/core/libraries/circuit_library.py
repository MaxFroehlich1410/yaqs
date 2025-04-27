# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of useful quantum circuits.

This module provides functions for creating quantum circuits that simulate
the dynamics of the Ising and Heisenberg models. The functions create_ising_circuit
and create_Heisenberg_circuit construct Qiskit QuantumCircuit objects based on specified
parameters such as the number of qubits, interaction strengths, time steps, and total simulation time.
These circuits are used to simulate the evolution of quantum many-body systems under the
respective Hamiltonians.
"""

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit
from scipy.linalg import expm

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


def create_ising_circuit(
    L: int, J: float, g: float, dt: float, timesteps: int, *, periodic: bool = False
) -> QuantumCircuit:
    """Ising Trotter circuit with optional periodic boundary conditions.

    Create a quantum circuit for simulating the Ising model. When periodic is True,
    a long-range rzz gate is added between the last and first qubits in each timestep.

    Args:
        L (int): Number of qubits in the circuit.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.
        periodic (bool, optional): If True, add a long-range gate between qubits 0 and L-1.
                                   Defaults to False.

    Returns:
        QuantumCircuit: A quantum circuit representing the Ising model evolution.
    """
    # Angle on X rotation
    alpha = -2 * dt * g
    # Angle on ZZ rotation
    beta = -2 * dt * J

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Apply RX rotations on all qubits.
        for site in range(L):
            circ.rx(theta=alpha, qubit=site)

        # Even-odd nearest-neighbor interactions.
        for site in range(L // 2):
            circ.rzz(beta, qubit1=2 * site, qubit2=2 * site + 1)

        # Odd-even nearest-neighbor interactions.
        for site in range(1, L // 2):
            circ.rzz(beta, qubit1=2 * site - 1, qubit2=2 * site)

        # For odd L > 1, handle the last pair.
        if L % 2 != 0 and L != 1:
            circ.rzz(beta, qubit1=L - 2, qubit2=L - 1)

        # If periodic, add an additional long-range gate between qubit L-1 and qubit 0.
        if periodic and L > 1:
            circ.rzz(beta, qubit1=0, qubit2=L - 1)
        circ.barrier()

    return circ


def create_2d_ising_circuit(
    num_rows: int, num_cols: int, J: float, g: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Ising Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Ising model evolution with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Single-qubit rotation and ZZ interaction angles.
    alpha = -2 * dt * g
    beta = -2 * dt * J

    for _ in range(timesteps):
        # Apply RX rotations to all qubits according to the snaking order.
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rx(alpha, q)

        # Horizontal interactions: within each row, apply rzz gates between adjacent qubits.
        for row in range(num_rows):
            # Even bonds in the row.
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)
            # Odd bonds in the row.
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)

        # Vertical interactions: between adjacent rows.
        for col in range(num_cols):
            # Even bonds vertically.
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
            # Odd bonds vertically.
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
        circ.barrier()

    return circ


def create_heisenberg_circuit(
    L: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """Heisenberg Trotter circuit.

    Create a quantum circuit for simulating the Heisenberg model.

    Args:
        L (int): Number of qubits (sites) in the circuit.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Magnetic field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.

    Returns:
        QuantumCircuit: A quantum circuit representing the Heisenberg model evolution.
    """
    theta_xx = -2 * dt * Jx
    theta_yy = -2 * dt * Jy
    theta_zz = -2 * dt * Jz
    theta_z = -2 * dt * h

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Z application
        for site in range(L):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rzz(theta=theta_zz, qubit1=L - 2, qubit2=L - 1)

        # XX application
        for site in range(L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rxx(theta=theta_xx, qubit1=L - 2, qubit2=L - 1)

        # YY application
        for site in range(L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.ryy(theta=theta_yy, qubit1=L - 2, qubit2=L - 1)
        circ.barrier()

    return circ


def create_2d_heisenberg_circuit(
    num_rows: int, num_cols: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """2D Heisenberg Trotter circuit on a rectangular grid using a snaking MPS ordering.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Single-qubit Z-field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: A quantum circuit representing the 2D Heisenberg model evolution
                       with MPS-friendly ordering.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Define the Trotter angles
    theta_xx = -2.0 * dt * Jx
    theta_yy = -2.0 * dt * Jy
    theta_zz = -2.0 * dt * Jz
    theta_z = -2.0 * dt * h

    for _ in range(timesteps):
        # (1) Apply single-qubit Z rotations to all qubits
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rz(theta_z, q)

        # (2) ZZ interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)

        # (3) XX interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)

        # (4) YY interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)
        circ.barrier()

    return circ



def create_highly_entangling_circuit(n_qubits, n_layers, seed=None):
    """
    Creates a highly entangling circuit with the given number of qubits and layers.
    
    Each layer applies random single-qubit rotations (Rz-Ry-Rz) on every qubit and
    then applies entangling CNOT gates in a ring pattern and between next-nearest neighbors.
    
    Parameters:
      n_qubits (int): Number of qubits.
      n_layers (int): Number of layers.
      seed (int, optional): Seed for random number generation.
      
    Returns:
      QuantumCircuit: The resulting highly entangling circuit.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    qc = QuantumCircuit(n_qubits)
    
    for layer in range(n_layers):
        # Apply random single-qubit rotations to each qubit.
        for q in range(n_qubits):
            theta = random.uniform(0, 2*np.pi)
            phi = random.uniform(0, 2*np.pi)
            lam = random.uniform(0, 2*np.pi)
            qc.rz(theta, q)
            qc.ry(phi, q)
            qc.rz(lam, q)
        
        # Apply entangling gates:
        # 1. Nearest-neighbor entanglement (ring connectivity)
        for i in range(n_qubits):
            qc.cx(i, (i+1) % n_qubits)
        # 2. Next-nearest neighbor entanglement
        for i in range(n_qubits):
            qc.cx(i, (i+2) % n_qubits)
        
        # Add a barrier to separate layers.
        qc.barrier()
        
    return qc




def entanglement_circuit(n_qubits, n_layers, seed=None):
    """
    Creates a multi-layer entangling circuit using Hadamard and CNOT gates.
    
    Parameters:
        n_qubits (int): Number of qubits in the circuit.
        n_layers (int): Number of entangling layers to apply.
        seed (int, optional): Seed for reproducibility of CNOT patterns.

    Returns:
        QuantumCircuit: A Qiskit circuit object.
    """
    import random
    if seed is not None:
        random.seed(seed)

    qc = QuantumCircuit(n_qubits)

    for layer in range(n_layers):
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qc.h(i)

        # Apply random CNOT entanglement (linear nearest neighbor or random pairings)
        for i in range(0, n_qubits - 1, 2):
            if random.random() < 0.5:
                control, target = i, i+1
            else:
                control, target = i+1, i
            qc.cx(control, target)

        qc.barrier()

    return qc




def brickwork_random_circuit(n_qubits: int, layers: int, seed: int = None) -> QuantumCircuit:
    if seed is not None:
        np.random.seed(seed)

    qc = QuantumCircuit(n_qubits)
    for layer in range(layers):
        start = layer % 2  # Even or odd layer
        for i in range(start, n_qubits - 1, 2):
            theta, phi, lam = 2 * np.pi * np.random.rand(3)
            qc.cx(i, i + 1)
            qc.u(theta, phi, lam, i)
            qc.u(theta, phi, lam, i + 1)
        qc.barrier()
    return qc


def qaoa_maxcut_ring(n_qubits: int, layers: int, gamma: float = 0.8, beta: float = 0.7) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    
    # Initial state: uniform superposition
    for i in range(n_qubits):
        qc.h(i)
    
    for _ in range(layers):
        # Cost unitary: ZZ terms for MaxCut
        for i in range(n_qubits):
            j = (i + 1) % n_qubits  # Periodic boundary
            qc.(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        
        # Mixer unitary: X rotations
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
        qc.barrier()
    
    return qc


def floquet_circuit(n_qubits: int, layers: int, J: float = 1.0, h: float = 1.0) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    
    for _ in range(layers):
        # Entangling ZZ layer (Trotterized)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * J, i + 1)
            qc.cx(i, i + 1)
        
        # Global X field
        for i in range(n_qubits):
            qc.rx(2 * h, i)
        qc.barrier()
    
    return qc




def hard_local_circuit(n_qubits: int, depth: int, seed: int = 42) -> QuantumCircuit:
    """
    Builds a brickwork-style circuit with only nearest-neighbor CZ gates
    and random non-Clifford single-qubit rotations. Generates volume-law entanglement.

    Parameters:
        n_qubits (int): Number of qubits in the circuit.
        depth (int): Number of brickwork layers (2-qubit + 1-qubit).
        seed (int): Random seed for reproducibility.

    Returns:
        QuantumCircuit: The constructed Qiskit circuit.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(depth):
        # 1. Single-qubit non-Clifford rotations
        for i in range(n_qubits):
            theta = 2 * np.pi * rng.random()
            phi = 2 * np.pi * rng.random()
            lam = 2 * np.pi * rng.random()
            qc.u(theta, phi, lam, i)  # Arbitrary SU(2) rotation

        # 2. Nearest-neighbor CZ gates (brickwork pattern)
        start = layer % 2  # even vs. odd layer
        for i in range(start, n_qubits - 1, 2):
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.h(i + 1)  # CZ via Hadamard sandwich
        qc.barrier()

    return qc


def local_plus_next_nearest_circuit(n_qubits: int, depth: int, seed: int = 123) -> QuantumCircuit:
    """
    Builds a circuit with nearest-neighbor and next-nearest-neighbor 2-qubit gates,
    plus random single-qubit unitaries.

    Parameters:
        n_qubits (int): Number of qubits in the circuit.
        depth (int): Number of layers.
        seed (int): RNG seed.

    Returns:
        QuantumCircuit: The generated circuit.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(depth):
        # 1. Single-qubit unitaries
        for i in range(n_qubits):
            theta = 2 * np.pi * rng.random()
            phi = 2 * np.pi * rng.random()
            lam = 2 * np.pi * rng.random()
            qc.u(theta, phi, lam, i)

        # 2. Nearest-neighbor CZ-like gates (layer parity alternates even/odd)
        start = layer % 2
        for i in range(start, n_qubits - 1, 2):
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.h(i + 1)

        # 3. Next-nearest-neighbor CZ-like gates (layer parity alternates)
        start = (layer + 1) % 2
        for i in range(start, n_qubits - 2, 2):
            qc.h(i + 2)
            qc.cx(i, i + 2)
            qc.h(i + 2)
        qc.barrier()

    return qc


def local_range_k_circuit(n_qubits: int, depth: int, k: int, seed: int = 42) -> QuantumCircuit:
    """
    Generates a circuit with single-qubit unitaries and entangling gates up to range k.

    Parameters:
        n_qubits (int): Number of qubits.
        depth (int): Number of layers (each includes single-qubit and range-k entanglers).
        k (int): Maximum range of two-qubit gates (i, i + r) for r in [1, k].
        seed (int): RNG seed for reproducibility.

    Returns:
        QuantumCircuit: The generated circuit.
    """
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(seed)

    for layer in range(depth):
        # 1. Random single-qubit unitaries
        for i in range(n_qubits):
            theta, phi, lam = 2 * np.pi * rng.random(3)
            qc.u(theta, phi, lam, i)

        # 2. Two-qubit CZ-like gates at increasing distances
        for r in range(1, k + 1):
            offset = (layer + r) % r  # stagger to prevent gate collisions
            for i in range(offset, n_qubits - r, 2 * r):
                qc.h(i + r)
                qc.cx(i, i + r)
                qc.barrier()
                qc.h(i + r)
        qc.barrier()
    print(qc)

    return qc




def nearest_neighbour_random_circuit(
    n_qubits: int,
    layers: int,
    seed: int = 42,
) -> QuantumCircuit:
    """Creates a random circuit with single and two-qubit nearest-neighbor gates.

    Gates are sampled following the prescription in https://arxiv.org/abs/2002.07730.

    Returns:
        A `QuantumCircuit` on `n_qubits` implementing `layers` of alternating
        random single-qubit rotations and nearest-neighbor CZ/CX entanglers.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(layers):
        # Single-qubit random rotations
        for qubit in range(n_qubits):
            add_random_single_qubit_rotation(qc, qubit, rng)

        # Two-qubit entangling gates
        if layer % 2 == 0:
            # Even layer → pair (1,2), (3,4), ...
            pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
        else:
            # Odd layer → pair (0,1), (2,3), ...
            pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]

        for q1, q2 in pairs:
            if rng.random() < 0.5:
                qc.cz(q1, q2)
            else:
                qc.cx(q1, q2)

        qc.barrier()

    return qc


def extract_u_parameters(
    matrix: NDArray[np.complex128],
) -> tuple[float, float, float]:
    """Extract θ, φ, λ from a 2x2 SU(2) unitary `matrix`.

    This removes any global phase and then solves
    matrix = U3(θ,φ,λ) exactly.

    Args:
        matrix: 2x2 complex array (must be SU(2), det=1 up to phase).

    Returns:
        A tuple (θ, φ, λ) of real gate angles.
    """
    assert matrix.shape == (2, 2), "Input must be a 2x2 matrix."

    # strip global phase
    u: NDArray[np.complex128] = matrix.astype(np.complex128)
    u *= np.exp(-1j * np.angle(u[0, 0]))

    a, b = u[0, 0], u[0, 1]
    c, d = u[1, 0], u[1, 1]

    theta = 2 * np.arccos(np.clip(np.abs(a), -1.0, 1.0))
    sin_th2: float = float(np.sin(theta / 2))
    if np.isclose(sin_th2, 0.0):
        phi = 0.0
        lam = np.angle(d) - np.angle(a)
    else:
        phi = np.angle(c)
        lam = np.angle(-b)

    return float(theta), float(phi), float(lam)


def add_random_single_qubit_rotation(
    qc: QuantumCircuit,
    qubit: int,
    rng: Generator | None = None,
) -> None:
    """Append a random single-qubit rotation exp(-i θ n sigma) as a U3 gate.

    Samples:
      - θ ∈ [0, 2π)
      - axis n uniformly on the Bloch sphere

    Decomposes the resulting 2x2 into U3(θ,φ,λ) and does `qc.u(...)`.

    Args:
        qc: the circuit to modify.
        qubit: which wire to rotate.
        rng: if given, used instead of the global `np.random`.
    """
    sampler = rng if rng is not None else np.random

    # sample angles
    theta = sampler.uniform(0, 2 * np.pi)
    alpha = sampler.uniform(0, np.pi)
    phi = sampler.uniform(0, 2 * np.pi)

    # Bloch-sphere axis
    nx = np.sin(alpha) * np.cos(phi)
    ny = np.sin(alpha) * np.sin(phi)
    nz = np.cos(alpha)

    # Pauli matrices
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    h = nx * x + ny * y + nz * z
    u_mat = expm(-1j * theta * h)

    th_u3, ph_u3, lam_u3 = extract_u_parameters(u_mat)
    qc.u(th_u3, ph_u3, lam_u3, qubit)