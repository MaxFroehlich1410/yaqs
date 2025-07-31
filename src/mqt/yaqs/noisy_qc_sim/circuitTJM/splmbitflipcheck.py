import numpy as np


pauli_x = np.array([[0, 1], [1, 0]])
identity = np.array([[1, 0], [0, 1]])


import numpy as np

def random_density_matrix(num_qubits):
    dim = 2 ** num_qubits
    # Step 1: Create a random complex matrix
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    
    # Step 2: Construct a Hermitian positive semidefinite matrix
    rho = A @ A.conj().T

    # Step 3: Normalize to trace 1
    rho /= np.trace(rho)

    return rho


def kraus_channel(kraus_factors, rho):
    rho1 = kraus_factors[0] * np.kron(pauli_x, identity) @ rho @ np.kron(pauli_x, identity).conj().T
    rho1 += kraus_factors[1] * np.kron(identity, pauli_x) @ rho @ np.kron(identity, pauli_x).conj().T
    rho1 += kraus_factors[2] * np.kron(pauli_x, pauli_x) @ rho @ np.kron(pauli_x, pauli_x).conj().T
    return rho1

def splm_channel(splm_factors, rho):
    rho1 = splm_factors[0] * rho + (1 - splm_factors[0]) * np.kron(pauli_x, identity) @ rho @ np.kron(pauli_x, identity).conj().T
    rho2 = rho1 @ (splm_factors[1] * rho + (1 - splm_factors[1]) * np.kron(identity, pauli_x) @ rho @ np.kron(identity, pauli_x).conj().T)
    rho3 = rho2 @ (splm_factors[2] * rho + (1 - splm_factors[2]) * np.kron(pauli_x, pauli_x) @ rho @ np.kron(pauli_x, pauli_x).conj().T)
    return rho3


def fidelity(rho1, rho2):
    return np.trace(rho1 @ rho2)

def splm_omega(factor):
    return (1 + np.exp(-2*factor))* 0.5



kraus_factors = [0.8, 0.1, 0.1]
splm_factors = [splm_omega(0.8), splm_omega(0.1), splm_omega(0.1)]

fidelity_list = []

for i in range(100):
    rho = random_density_matrix(2)

    rho_kraus = kraus_channel(kraus_factors, rho)
    rho_splm = splm_channel(splm_factors, rho)

    fidelity_list.append(fidelity(rho_kraus, rho_splm))

print(np.mean(fidelity_list))
