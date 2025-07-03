# """
# Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
# Uses dephasing noise models to validate both approaches give consistent results.
# """

# import numpy as np
# from qiskit import QuantumCircuit
# from qiskit_aer.primitives import Estimator
# from qiskit.quantum_info import SparsePauliOp
# from qiskit_aer import Aer



# from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
# from mqt.yaqs.noisy_qc_sim.densitymatrix_sim import (
#     create_all_zero_density_matrix, 
#     evolve_noisy_circuit, 
#     circuit_to_unitary_list, 
#     z_expectations,
#     two_qubit_reverse
# )
# from mqt.yaqs.noisy_qc_sim.qiskit_noisemodels import qiskit_dephasing_noise, qiskit_bitflip_noise, qiskit_bitflip_noise_2
# from mqt.yaqs.core.data_structures.noise_model import NoiseModel
# from mqt.yaqs.noisy_qc_sim.yaqs_noisemodels import create_yaqs_bitflip_noise, create_yaqs_bitflip_noise_2, create_yaqs_dephasing_noise
# from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
# from mqt.yaqs.core.data_structures.simulation_parameters import Observable
# from mqt.yaqs.core.libraries.gate_library import Z
# from mqt.yaqs import simulator
# from mqt.yaqs.core.data_structures.networks import MPS



# if __name__ == "__main__":

#     for noise_strength in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         print("-"*100)
#         print(f"Starting noisy quantum circuit simulator comparison tests for noise strength: {noise_strength}")
#         print("-"*100)


#         # direct vector matrix simulation
#         #########################################################

#         pauli_x = np.array([[0, 1], [1, 0]])
#         pauli_z = np.array([[1, 0], [0, -1]])

#         rho0 = create_all_zero_density_matrix(2)

#         CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
#         rho1 = CNOT @ rho0 @ CNOT.T

#         I_X = np.kron(np.eye(2), pauli_x)
#         X_I = np.kron(pauli_x, np.eye(2))
#         X_X = np.kron(pauli_x, pauli_x)


#         gamma = (1+np.exp(-2*noise_strength))*0.5

#         p = noise_strength

#         rho_noise = (gamma*rho1+(1-gamma)*I_X@rho1@I_X.T)@(gamma*rho1+(1-gamma)*X_I@rho1@X_I.T)@(gamma*rho1+(1-gamma)*X_X@rho1@X_X.T)

#         vector_matrix_sim_expvals = []
#         vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(pauli_z, np.eye(2))))
#         vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(np.eye(2), pauli_z)))
#         vector_matrix_sim_expvals = np.array([[np.real(val) for val in vector_matrix_sim_expvals]])
        
#         print(f"vector_matrix_sim_expvals: {vector_matrix_sim_expvals}")


#         #########################################################

#         qc = QuantumCircuit(2)
#         qc.cx(0, 1)

#         ### 
#         # Qiskit simulation
#         #########################################################

#         # noise_model = qiskit_bitflip_noise(num_qubits=2, noise_strengths=[noise_strength])
#         # qiskit_results = qiskit_noisy_simulator(qc, noise_model, 2, 1)
#         # print(f"qiskit_results: {qiskit_results}")



        
#         processes_yaqs = [{
#                 "name": "xx",
#                 "sites": [0, 1],
#                     "strength": noise_strength
#             },
#             {
#                 "name": "x",
#                 "sites": [0],
#                 "strength": noise_strength
#             }, 
#             {
#                 "name": "x",
#                 "sites": [1],
#                 "strength": noise_strength
#             }
#             ]


    
#         ### 
#         # Kraus channel func evolve noisy circuit simulation
#         #########################################################

#         # processes = [{
#         #         "name": "xx",
#         #         "sites": [0, 1],
#         #         "strength": noise_strength**2
#         #     },
#         #     {
#         #         "name": "x",
#         #         "sites": [0],
#         #         "strength": (1-noise_strength)*noise_strength
#         #     }, 
#         #     {
#         #         "name": "x",
#         #         "sites": [1],
#         #         "strength": (1-noise_strength)*noise_strength
#         #     }
#         #     ]
#         # noise_model = NoiseModel(processes)
    

#         # rho0 = create_all_zero_density_matrix(2)

#         # gate_list = circuit_to_unitary_list(qc)
        
#         # kraus_results = evolve_noisy_circuit(rho0, gate_list, noise_model, 1)
        
#         # print(f"evolve_noisy_circuit results: {kraus_results}")


#         ### 
#         # YAQS simulation
#         #########################################################

#         sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(2)], num_traj=1000, max_bond_dim=2, threshold=1e-14, window_size=0, get_state=False)

#         noise_model_yaqs = NoiseModel(processes_yaqs)
        
#         initial_mps = MPS(2, state = "zeros", pad=2)
        
#         simulator.run(initial_mps, qc, sim_params, noise_model_yaqs)
#         yaqs_results = []
#         for i in range(2):
#             yaqs_results.append(sim_params.observables[i].results)
        
#         print(f"yaqs_results: {yaqs_results}")
        
        
        
        
        
        
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Your data from above (using the first YAQS result for each case)
noise_strengths = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ref_results = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0])
yaqs_results = np.array([1.0, 0.816, 0.652, 0.526, 0.434, 0.3, 0.17, 0.06, 0.012, -0.13, -0.228])

# Compute the difference for each noise strength
diffs = yaqs_results - ref_results

plt.figure(figsize=(8,4))
plt.plot(noise_strengths, diffs, marker="o")
plt.axhline(0, color="k", linestyle="--", lw=1)
plt.xlabel("Noise Strength (p)")
plt.ylabel("YAQS - Reference")
plt.title("Difference between YAQS and Reference Simulation")
plt.show()

# Now, try scaling the noise_strength for YAQS and interpolate the YAQS values
def scaled_yaqs_loss(factor):
    # Interpolate the YAQS results at scaled noise strengths (clip to [0,1])
    scaled_p = np.clip(noise_strengths * factor, 0, 1)
    scaled_yaqs = np.interp(scaled_p, noise_strengths, yaqs_results)
    # Mean squared error
    return np.mean((scaled_yaqs - ref_results)**2)

# Find the optimal scaling factor
res = minimize_scalar(scaled_yaqs_loss, bounds=(0.5, 3), method="bounded")
best_factor = res.x

print(f"Best scaling factor: {best_factor:.3f}")

# Plot the rescaled result
plt.figure(figsize=(8,4))
plt.plot(noise_strengths, ref_results, "o-", label="Reference (Kraus/Qiskit)")
plt.plot(noise_strengths, yaqs_results, "o-", label="YAQS (original)")
plt.plot(noise_strengths, np.interp(np.clip(noise_strengths*best_factor,0,1), noise_strengths, yaqs_results), "o-", label=f"YAQS (scaled, factor={best_factor:.2f})")
plt.xlabel("Noise Strength (p)")
plt.ylabel("Expectation Value")
plt.title("Effect of Scaling YAQS Noise Parameter")
plt.legend()
plt.show()
