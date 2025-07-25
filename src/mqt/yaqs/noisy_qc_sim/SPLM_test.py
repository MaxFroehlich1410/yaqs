
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit.quantum_info import SparsePauliOp, Pauli 
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer import Aer
import matplotlib.pyplot as plt


from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS



if __name__ == "__main__":

    # circuit
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    qc.rzz(np.pi/4, 0, 1)
    qc.rzz(np.pi/4, 1, 2)
    # qc.rzz(np.pi/4, 2, 3)

    yaqs_noise_rates = [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.04, 0.04, 0.04], [0.05, 0.05, 0.05]]
    qiskit_results_list = []
    yaqs_results_list = []


    for noise_rate in yaqs_noise_rates:

        # qiskit simulation
        generators = [Pauli("IX"), Pauli("XI"), Pauli("XX")]
        SPLM_error = PauliLindbladError(generators, noise_rate)
        noise_model = QiskitNoiseModel()
        for qubit in range(num_qubits - 1):
            noise_model.add_quantum_error(SPLM_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])

        qiskit_results = qiskit_noisy_simulator(qc, noise_model, num_qubits, 1)
        qiskit_results_list.append(qiskit_results)

        # yaqs simulation
        processes = []
        for qubit in range(num_qubits-1):
            processes.append({
                "name": "xx",
                "sites": [qubit, qubit + 1],
                "strength": noise_rate[2]
            })

        for qubit in range(num_qubits):
            processes.append({
                "name": "x",
                "sites": [qubit],
                "strength": noise_rate[1]
            })
        noise_model_yaqs = NoiseModel(processes)
        sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(num_qubits)], num_traj=1000, max_bond_dim=2, threshold=1e-14, window_size=0, get_state=False)
        initial_mps = MPS(num_qubits, state = "zeros", pad=2)
        simulator.run(initial_mps, qc, sim_params, noise_model_yaqs, parallel = False)

        yaqs_results = []
        for i in range(num_qubits):
            yaqs_results.append(sim_params.observables[i].results)

        yaqs_results_list.append(yaqs_results)
        # print('yaqs_results', yaqs_results)
    print('yaqs_results_list', len(yaqs_results_list))
    print('yaqs_results_list B', len(yaqs_results_list[0]))

    print('qiskit_results_list', len(qiskit_results_list))
    print('qiskit_results_list B', qiskit_results_list[0])




    # plot
    for q in range(num_qubits):
        qiskit_q = [qiskit_results_list[i][0, q] for i in range(len(yaqs_noise_rates))]
        yaqs_q = [yaqs_results_list[i][q][0] for i in range(len(yaqs_noise_rates))]
        diff_q = [k - y for k, y in zip(qiskit_q, yaqs_q)]

        
        plt.plot(range(len(yaqs_noise_rates)), qiskit_q, marker="o", label=f'Qiskit q{q}')
        plt.plot(range(len(yaqs_noise_rates)), yaqs_q, marker="s", label=f'YAQS q{q}')
        plt.plot(range(len(yaqs_noise_rates)), diff_q, marker="x", linestyle="--", label=f'Qiskit - YAQS q{q}')


    plt.xlabel('Noise Rate')
    plt.ylabel('Expectation Value')
    plt.title("Comparison of Qiskit and SPLM Simulation (per qubit)")
    plt.legend()
    plt.grid(True)
    plt.show()
