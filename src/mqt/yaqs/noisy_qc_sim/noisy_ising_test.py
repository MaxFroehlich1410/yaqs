"""
Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
Uses dephasing noise models to validate both approaches give consistent results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
import matplotlib.pyplot as plt


from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.noisy_qc_sim.densitymatrix_sim import (
    create_all_zero_density_matrix, 
    evolve_noisy_circuit, 
    circuit_to_unitary_list, 
    z_expectations,
    two_qubit_reverse
)
from mqt.yaqs.noisy_qc_sim.qiskit_noisemodels import qiskit_dephasing_noise, qiskit_bitflip_noise, qiskit_bitflip_noise_2
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.noisy_qc_sim.yaqs_noisemodels import create_yaqs_bitflip_noise, create_yaqs_bitflip_noise_2, create_yaqs_dephasing_noise
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS


def kraus_noisemodel(num_qubits, noise_strength):
    processes = []
    for i in range(num_qubits-1):
        processes.append({"name": "xx",
                "sites": [i, i+1],
                "strength": noise_strength**2})
    for j in range(num_qubits):
        processes.append({
                "name": "x",
                "sites": [j],
                "strength": noise_strength*(1-noise_strength)
            })
    return NoiseModel(processes)


def yaqs_noisemodel(num_qubits, noise_strength):
    processes = []
    for i in range(num_qubits-1):
        processes.append({"name": "xx",
                "sites": [i, i+1],
                "strength": -0.5 * np.log(1-(noise_strength**2)*2)})
    for j in range(num_qubits):
        processes.append({
                "name": "x",
                "sites": [j],
                "strength": -0.5 * np.log(1-(noise_strength*(1-noise_strength))*2)
            })
    for j in range(num_qubits):
        processes.append({
                "name": "identity",
                "sites": [j],
                "strength": -0.5 * np.log((1-(1-noise_strength)**2))
            })
    return NoiseModel(processes)



if __name__ == "__main__":

    vector_matrix_sim_expvals_list = []
    kraus_results_list = []
    yaqs_results_list = []
    qiskit_results_list = []

    noise_strengths = [0.0, 0.01, 0.02, 0.03, 0.04] #, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    # noise_strengths = [0.0, 0.1, 0.2, 0.3, 0.4]
    # noise_strengths = [0.05, 0.06]
        
    num_qubits = 3
    

    for noise_strength in noise_strengths:

 
        print("-"*100)
        print(f"Starting noisy quantum circuit simulator comparison tests for noise strength: {noise_strength}")
        print("-"*100)

        gamma = -0.5 * np.log(1-noise_strength*2)

        #########################################################

        qc = create_ising_circuit(num_qubits, J=1.0, g=0.5, dt=1.0, timesteps=1)

  

        ### 
        # Qiskit simulation
        #########################################################

        noise_model = qiskit_bitflip_noise(num_qubits=num_qubits, noise_strengths=[noise_strength])
        qiskit_results = qiskit_noisy_simulator(qc, noise_model, num_qubits, 1)
        print(f"qiskit_results: {qiskit_results}")
        qiskit_results_list.append(qiskit_results)
        ### 
        # Kraus channel func evolve noisy circuit simulation
        #########################################################

        noise_model = kraus_noisemodel(num_qubits, noise_strength)

        rho0 = create_all_zero_density_matrix(num_qubits)
        gate_list = circuit_to_unitary_list(qc)
        kraus_results = evolve_noisy_circuit(rho0, gate_list, noise_model, 1)
        kraus_results_list.append(kraus_results)
        print(f"evolve_noisy_circuit results: {kraus_results}")

        ### 
        # YAQS simulation
        #########################################################

        noise_model_yaqs = yaqs_noisemodel(num_qubits, noise_strength)
        sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(num_qubits)], num_traj=2000, max_bond_dim=4, threshold=1e-14, window_size=0, get_state=False)
        initial_mps = MPS(num_qubits, state = "zeros", pad=2)
        simulator.run(initial_mps, qc, sim_params, noise_model_yaqs, parallel = False)
        yaqs_results = []
        for i in range(num_qubits):
            yaqs_results.append(sim_params.observables[i].results)

        yaqs_results_list.append(yaqs_results)
        
        print(f"yaqs_results: {yaqs_results}")

    num_noise_strengths = len(kraus_results_list)

    for q in range(num_qubits):
        kraus_q = [kraus_results_list[i][0, q] for i in range(num_noise_strengths)]
        yaqs_q = [yaqs_results_list[i][q][0] for i in range(num_noise_strengths)]
        diff_q = [k - y for k, y in zip(kraus_q, yaqs_q)]
        
        plt.plot(noise_strengths, kraus_q, marker="o", label=f'Kraus q{q}')
        plt.plot(noise_strengths, yaqs_q, marker="s", label=f'YAQS q{q}')
        plt.plot(noise_strengths, diff_q, marker="x", linestyle="--", label=f'Kraus - YAQS q{q}')


    plt.xlabel('Noise Strength')
    plt.ylabel('Expectation Value')
    plt.title("Comparison of Kraus and YAQS Simulation (per qubit)")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    
    
        
        
        
        