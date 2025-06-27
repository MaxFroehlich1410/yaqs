import numpy as np



hadamard_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
rho0 = np.array([[1, 0], [0, 0]])
identity = np.array([[1, 0], [0, 1]])



def single_qubit_sim(rho0, noise_strength, gate, kraus_op):
    print('--------------------------------')

    rho1 = gate @ rho0 @ gate.conj().T
    rho1_noisy = rho1 * (1 - noise_strength) + noise_strength * kraus_op @ rho1 @ kraus_op.conj().T
    # print(f"Rho after {gate}:", rho1)
    # print("Rho after noise:", rho1_noisy)
    print(f"Noise strength: {noise_strength}")
    print("Z expectation:", np.trace(pauli_z @ rho1_noisy))
    print("X expectation:", np.trace(pauli_x @ rho1_noisy))
    print("Y expectation:", np.trace(pauli_y @ rho1_noisy))
    print('--------------------------------')
    return 

def two_qubit_sim(rho0, noise_strength, gate, kraus_op):
    print('--------------------------------')

    rho1 = gate @ rho0 @ gate.conj().T
    kraus_first = np.kron(kraus_op, identity)
    kraus_second = np.kron(identity, kraus_op)
    kraus_double = np.kron(kraus_op, kraus_op)
    rho1_noisy = rho1 * (1 - noise_strength) + noise_strength/3 * kraus_first @ rho1 @ kraus_first.conj().T + noise_strength/3 * kraus_second @ rho1 @ kraus_second.conj().T + noise_strength/3 * kraus_double @ rho1 @ kraus_double.conj().T
    rho1_noisy = rho1_noisy / np.trace(rho1_noisy)
    # print(f"Rho after {gate}:", rho1) good
    # print("Rho after noise:", rho1_noisy)
    print(f"Noise strength: {noise_strength}")
    print("Z expectation 1st qubit:", np.trace(np.kron(pauli_z, identity) @ rho1_noisy))
    print("Z expectation 2nd qubit:", np.trace(np.kron(identity, identity) @ rho1_noisy))
    print("Z expectation 1st and 2nd qubit:", np.trace(np.kron(pauli_z, pauli_z) @ rho1_noisy))
    # print("X expectation:", np.trace(pauli_x @ rho1_noisy))
    # print("Y expectation:", np.trace(pauli_y @ rho1_noisy))
    print('--------------------------------')
    return 

if __name__ == "__main__":

    rho0_twoqubits = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) 
    gate = np.eye(4)
    kraus_op = pauli_x

    two_qubit_sim(rho0_twoqubits, 0, gate, kraus_op)
    two_qubit_sim(rho0_twoqubits, 0.3, gate, kraus_op)

    # single_qubit_sim(rho0, 0, identity, pauli_x)
    # single_qubit_sim(rho0, 0.3, identity, pauli_x)
    # single_qubit_sim(rho0, 0, hadamard_gate, pauli_y)
    # single_qubit_sim(rho0, 0, hadamard_gate, pauli_z)
    # single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_x)
    # single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_y)
    # single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_z)








