"""
Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
Uses bitflip noise models to validate both approaches give consistent results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
import matplotlib.pyplot as plt


from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
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
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS



if __name__ == "__main__":
    num_qubits = 8


    kraus_results_list = []
    yaqs_results_list = []
    qiskit_results_list = []

    noise_strengths = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] #, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    # noise_strengths = [0.0, 0.1, 0.2, 0.3, 0.4]
        
    

    for noise_strength in noise_strengths:
        print("-"*100)
        print(f"Starting noisy quantum circuit simulator comparison tests for noise strength: {noise_strength}")
        print("-"*100)

        gamma = -0.5 * np.log(1-noise_strength*2)

        #########################################################

        qc = create_ising_circuit(num_qubits, 1.0, 0.5, 0.1, 10)

           

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

        processes = [{
                "name": "xx",
                "sites": [0, 1],
                "strength": noise_strength**2
            },
            {
                "name": "xx",
                "sites": [1, 2],
                "strength": noise_strength**2
            },
            {
                "name": "x",
                "sites": [0],
                "strength": (1-noise_strength)*noise_strength
            }, 
            {
                "name": "x",
                "sites": [1],
                "strength": (1-noise_strength)*noise_strength
            }, 
            {
                "name": "x",
                "sites": [2],
                "strength": (1-noise_strength)*noise_strength
            }
            ]
        noise_model = NoiseModel(processes)

        rho0 = create_all_zero_density_matrix(num_qubits)
        gate_list = circuit_to_unitary_list(qc)
        kraus_results = evolve_noisy_circuit(rho0, gate_list, noise_model, 1)
        kraus_results_list.append(kraus_results)
        print(f"evolve_noisy_circuit results: {kraus_results}")

        ### 
        # YAQS simulation
        #########################################################

        processes_yaqs = [{
                "name": "xx",
                "sites": [0, 1],
                    "strength": gamma**2
            },
            {
                "name": "xx",
                "sites": [1, 2],
                "strength": gamma**2
            },
            {
                "name": "x",
                "sites": [0],
                "strength": (1-gamma)*gamma
            }, 
            {
                "name": "x",
                "sites": [1],
                "strength": (1-gamma)*gamma
            },
            {
                "name": "x",
                "sites": [2],
                "strength": (1-gamma)*gamma
            }
            ]
        noise_model_yaqs = NoiseModel(processes_yaqs)
        sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(num_qubits)], num_traj=2000, max_bond_dim=2, threshold=1e-14, window_size=0, get_state=False)
        initial_mps = MPS(num_qubits, state = "zeros", pad=2)
        simulator.run(initial_mps, qc, sim_params, noise_model_yaqs)
        yaqs_results = []
        for i in range(num_qubits):
            yaqs_results.append(sim_params.observables[i].results)

        yaqs_results_list.append(yaqs_results)
        
        print(f"yaqs_results: {yaqs_results}")

    # print(f"kraus_results_list shape: {np.array(kraus_results_list).shape}")
    # print(f"yaqs_results_list shape: {np.array(yaqs_results_list).shape}")
    # print(f"qiskit_results shape: {np.array(qiskit_results).shape}")
    # difference = [kraus_results_list[i][0] - yaqs_results_list[i][0] for i in range(len(kraus_results_list))]
    # kraus_results_list = [kraus_results_list[i][0] for i in range(len(kraus_results_list))]
    # yaqs_results_list = [yaqs_results_list[i][0] for i in range(len(yaqs_results_list))]
    # # plot difference between kraus and yaqs results    
    # plt.plot(noise_strengths, kraus_results_list, marker="o", label='Kraus')
    # plt.plot(noise_strengths, yaqs_results_list, marker="s", label='YAQS')
    # plt.plot(noise_strengths, difference, marker="x", linestyle="--", label='Kraus - YAQS')

    # plt.xlabel('Noise Strength')
    # plt.ylabel('Expectation Value')
    # plt.title("Comparison of Kraus and YAQS Simulation")
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

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

    
    
    
        
        
        
        