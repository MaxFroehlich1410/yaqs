"""
Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
Uses dephasing noise models to validate both approaches give consistent results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer



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
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS


if __name__ == "__main__":
    print("Starting noisy quantum circuit simulator comparison tests...")
    print("Comparing Qiskit Estimator vs Kraus Channel simulation")

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    print(f"qc: {qc}")
    print(f"qc.draw(output='text'): {qc.draw(output='text')}")


   

    noise_strength = 0.3
    # processes = [{
    #         "name": "xx",
    #         "sites": [0, 1],
    #         "strength": noise_strength**2
    #     },
    #     {
    #         "name": "x",
    #         "sites": [0],
    #         "strength": (1-noise_strength)*noise_strength
    #     }, 
    #     {
    #         "name": "x",
    #         "sites": [1],
    #         "strength": (1-noise_strength)*noise_strength
    #     }
    #     ]
    
    # processes_yaqs = [{
    #         "name": "xx",
    #         "sites": [0, 1],
    #         "strength": noise_strength**2*1.4
    #     },
    #     {
    #         "name": "x",
    #         "sites": [0],
    #         "strength": (1-noise_strength)*noise_strength*1.4
    #     }, 
    #     {
    #         "name": "x",
    #         "sites": [1],
    #         "strength": (1-noise_strength)*noise_strength*1.4
    #     },
    #     {
    #         "name": "id",
    #         "sites": [0],
    #         "strength": 1-noise_strength**2-2*(1-noise_strength)*noise_strength
    #     },
    #     {
    #         "name": "id",
    #         "sites": [1],
    #         "strength": 1-noise_strength**2-2*(1-noise_strength)*noise_strength
    #     }
    #     ]

    processes = [{
            "name": "xx",
            "sites": [0, 1],
            "strength": noise_strength
        },
        {
            "name": "x",
            "sites": [0],
            "strength": noise_strength
        }, 
        {
            "name": "x",
            "sites": [1],
            "strength": noise_strength
        }
        ]
  

    noise_model = NoiseModel(processes)
    # noise_model_yaqs = NoiseModel(processes_yaqs)

    rho0 = create_all_zero_density_matrix(2)

    gate_list = circuit_to_unitary_list(qc)
    
    kraus_results = evolve_noisy_circuit(rho0, gate_list, noise_model, 1)
    
    print(f"kraus_results: {kraus_results}")


    ### 
    # YAQS simulation
    #########################################################

    sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(2)], num_traj=10000, max_bond_dim=2, threshold=1e-6, window_size=0, get_state=False)
    
    initial_mps = MPS(2, state = "zeros")
    
    simulator.run(initial_mps, qc, sim_params, noise_model)

    for i in range(2):
        print(f"Observable {i}: {sim_params.observables[i].results}")
    
    