import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit.quantum_info import SparsePauliOp, Pauli 
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer import Aer


from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import PhysicsSimParams, Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS

'''
This script tries to match the results of the circuitTJM solver with the Qiskit noisy simulator.

simlating circuitTJM and ising ciruit.

# Navigate to the project directory
cd /Users/maximilianfrohlich/Documents/GitHub/mqt-yaqs

# Activate the virtual environment
source env/bin/activate

# Run the script
python3 -m mqt.yaqs.noisy_qc_sim.circuitTJM.circuitTJMIsing'''



if __name__ == "__main__":


    num_qubits = 4
    J = 1.0
    g = 0.5
    timestepsize = 0.1
    elapsed_time =5
    timesteps = int(elapsed_time/timestepsize)
    # Create observables for the simulation
    observables = [Observable(Z(), i) for i in range(num_qubits)]
    sim_params = PhysicsSimParams(observables, elapsed_time=elapsed_time, dt=timestepsize)
    qutip_noise_factor = 0.01
    qiskit_noise_factor = 0.01*np.sqrt(timestepsize)

    qc_test = create_ising_circuit(num_qubits, J, g, timestepsize, 1, periodic=False)
    # qc_test.draw(output="mpl")

    
    # qiskit simulation
    generators_two_qubit = [Pauli("IX"), Pauli("XI"), Pauli("XX")] # [Pauli("IX"), Pauli("XI"), Pauli("XX")]
    generators_single_qubit = [Pauli("X")]
    SPLM_error_two_qubit = PauliLindbladError(generators_two_qubit, [qiskit_noise_factor, qiskit_noise_factor, qiskit_noise_factor]) #, qiskit_noise_factor])
    SPLM_error_single_qubit = PauliLindbladError(generators_single_qubit, [qiskit_noise_factor])
    noise_model = QiskitNoiseModel()
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(SPLM_error_two_qubit, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # noise_model.add_quantum_error(SPLM_error_single_qubit, ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"], [qubit])


    qiskit_results_list = [np.array([1.0, 1.0, 1.0, 1.0])]
    for i in range(timesteps):
        # Qiskit noisy simulator
        qc = create_ising_circuit(num_qubits, J, g, timestepsize, i+1, periodic=False)
        qiskit_results = qiskit_noisy_simulator(qc, noise_model, num_qubits, 1)
        qiskit_results_list.append(qiskit_results)

    print(len(qiskit_results_list))
    print(len(qiskit_results_list[0]))
    print("qiskit_results_list: ", qiskit_results_list)



    # yaqs simulation
    yaqs_results_list = [np.array([1.0, 1.0, 1.0, 1.0])]
    processes = []
    for qubit in range(num_qubits-1):
        processes.append({
            "name": "xx",
            "sites": [qubit, qubit + 1],
            "strength": qiskit_noise_factor
        })

    for qubit in range(num_qubits):
        processes.append({
            "name": "x",
            "sites": [qubit],
            "strength": qiskit_noise_factor
        })
    noise_model_yaqs = NoiseModel(processes)
    yaqs_results_list = [np.array([1.0, 1.0, 1.0, 1.0])]
    for i in range(timesteps):
        qc = create_ising_circuit(num_qubits, J, g, timestepsize, i+1, periodic=False)
        sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(num_qubits)], num_traj=100, max_bond_dim=4, threshold=1e-14, window_size=0, get_state=False)
        initial_mps = MPS(num_qubits, state = "zeros", pad=2)
        simulator.run(initial_mps, qc, sim_params, noise_model_yaqs, parallel = True)

        yaqs_results = []
        for i in range(num_qubits):
            yaqs_results.append(sim_params.observables[i].results)

        yaqs_results_list.append(yaqs_results)

    print("yaqs_results_list: ", yaqs_results_list)
    print("qiskit_results_list: ", qiskit_results_list)

    # plot
    plt.figure(figsize=(12, 8))
    for i in range(num_qubits):
        plt.subplot(num_qubits, 1, i+1)
        
        # Extract Qiskit results for this observable
        qiskit_obs_values = []
        for timestep in range(len(qiskit_results_list)):
            if timestep == 0:
                # First element is 1D array
                qiskit_obs_values.append(qiskit_results_list[timestep][i])
            else:
                # Other elements are 2D arrays
                qiskit_obs_values.append(qiskit_results_list[timestep][0, i])
        
        # Extract YAQS results for this observable
        yaqs_obs_values = []
        for timestep in range(len(yaqs_results_list)):
            if timestep == 0:
                # First element is 1D array
                yaqs_obs_values.append(yaqs_results_list[timestep][i])
            else:
                # Other elements are lists of arrays
                yaqs_obs_values.append(yaqs_results_list[timestep][i][0].real)  # Take real part of complex number
        
        time_points = [timestepsize*i for i in range(len(qiskit_results_list))]
        
        plt.plot(time_points, qiskit_obs_values, label=f"Qiskit Observable {i+1}")
        plt.plot(time_points, yaqs_obs_values, label=f"YAQS Observable {i+1}")
        plt.plot(time_points, np.array(qiskit_obs_values) - np.array(yaqs_obs_values), label=f"Qiskit - YAQS Observable {i+1}", linestyle="--")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Expectation Value")
        plt.legend()
        plt.title(f"Observable {i+1}")
    
    plt.tight_layout()
    plt.show()
