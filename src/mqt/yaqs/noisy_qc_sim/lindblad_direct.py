import numpy as np
from scipy.linalg import expm


def lindblad_liouvillian(L_ops, gammas, d):
    """Return the full Liouvillian superoperator for the provided jump ops and rates."""
    I = np.eye(d)
    L = np.zeros((d*d, d*d), dtype=np.complex128)
    for L_op, gamma in zip(L_ops, gammas):
        L_dag_L = L_op.conj().T @ L_op
        # Left action
        L_left = np.kron(L_op, L_op.conj())
        # Dissipator
        dissipator = (
            np.kron(L_op, L_op.conj()) -
            0.5 * (np.kron(L_dag_L, I) + np.kron(I, L_dag_L.T))
        )
        L += gamma * dissipator
    return L


noise_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kraus_results_list = []
lindblad_results_list = []
d = 4

for noise_strength in noise_strengths:
    print("-"*100)
    print(f"Starting noisy quantum circuit simulator comparison tests for noise strength: {noise_strength}")
    print("-"*100)

    gamma = -0.5 * np.log(1-noise_strength)

    # direct vector matrix simulation
    #########################################################

    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    rho0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # rho1 = CNOT @ rho0 @ CNOT.T

    I_X = np.kron(np.eye(2), pauli_x)
    X_I = np.kron(pauli_x, np.eye(2))
    X_X = np.kron(pauli_x, pauli_x)

    p = noise_strength
    K0 = np.sqrt((1-p)**2) * np.eye(4)
    K1 = np.sqrt(p*(1-p)) * X_I
    K2 = np.sqrt(p*(1-p)) * I_X
    K3 = np.sqrt(p**2) * X_X

    rho_noise = (
        K0 @ rho0 @ K0.conj().T +
        K1 @ rho0 @ K1.conj().T +
        K2 @ rho0 @ K2.conj().T +
        K3 @ rho0 @ K3.conj().T
    )
    
    vector_matrix_sim_expvals = []
    vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(pauli_z, np.eye(2))))
    vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(np.eye(2), pauli_z)))
    vector_matrix_sim_expvals = np.array([[np.real(val) for val in vector_matrix_sim_expvals]])

    kraus_results_list.append(vector_matrix_sim_expvals)
    
    print(f"Kraus_sim_expvals: {vector_matrix_sim_expvals}")

    # --- Lindblad master equation ---
    # Rates for the three jumps
    p_lindblad = -0.5 * np.log(1-noise_strength*2)

    gamma_xi = (1-p_lindblad)*p_lindblad   # scale to match your code
    gamma_ix = (1-p_lindblad)*p_lindblad
    gamma_xx = (p_lindblad**2)
    L_ops = [X_I, I_X, X_X]
    gammas = [gamma_xi, gamma_ix, gamma_xx]

    # Build Liouvillian
    L = lindblad_liouvillian(L_ops, gammas, d)
    # Vectorize initial state (column-stacked)
    rho_vec = rho0.flatten()
    # Exponentiate for time t=1
    rho_final_vec = expm(L) @ rho_vec
    rho_final = rho_final_vec.reshape((d, d))

    lindblad_expect = [
        np.real(np.trace(rho_final @ np.kron(pauli_z, np.eye(2)))),
        np.real(np.trace(rho_final @ np.kron(np.eye(2), pauli_z))),
    ]
    lindblad_results_list.append(lindblad_expect)
    print(f"Lindblad ME expvals: {lindblad_expect}")


import matplotlib.pyplot as plt
kraus_results_array = np.array(kraus_results_list)
lindblad_results_array = np.array(lindblad_results_list)
plt.plot(noise_strengths, kraus_results_array[:,0], 'o-', label="Kraus Z0")
plt.plot(noise_strengths, lindblad_results_array[:,0], 's-', label="Lindblad Z0")
plt.xlabel("Noise strength")
plt.ylabel("Z0 expectation")
plt.legend()
plt.show()