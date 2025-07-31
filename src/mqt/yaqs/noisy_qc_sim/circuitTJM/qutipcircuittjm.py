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
from mqt.yaqs.core.data_structures.simulation_parameters import PhysicsSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS

'''
This script tries to match the results of the QuTiP Lindblad solver with the Qiskit noisy simulator.

simlating SPLM and ising ciruit. Spoiler: it does not work.

# Navigate to the project directory
cd /Users/maximilianfrohlich/Documents/GitHub/mqt-yaqs

# Activate the virtual environment
source env/bin/activate

# Run the script
python3 -m mqt.yaqs.noisy_qc_sim.circuitTJM.qutipcircuittjm'''



if __name__ == "__main__":


    L = 4
    J = 1.0
    g = 0.5
    timestepsize = 0.1
    elapsed_time = 10
    timesteps = int(elapsed_time/timestepsize)
    # Create observables for the simulation
    observables = [Observable(Z(), i) for i in range(L)]
    sim_params = PhysicsSimParams(observables, elapsed_time=elapsed_time, dt=timestepsize)
    qutip_noise_factor = 0.01
    qiskit_noise_factor = 0.01*np.sqrt(timestepsize)

    qc_test = create_ising_circuit(L, J, g, timestepsize, 1, periodic=False)
    # qc_test.draw(output="mpl")

    # Time vector
    t = np.arange(0, sim_params.elapsed_time + sim_params.dt, sim_params.dt)

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L-1):
        H += J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

    # Construct collapse operators
    c_ops = []

    # Single qubit bitflip operators
    for i in range(L):
        c_ops.append(np.sqrt(qutip_noise_factor) * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)]))

    # # Two qubit bitflip operators
    # for i in range(L-1):
    #     c_ops.append(np.sqrt(qutip_noise_factor) * qt.tensor([sx if n==i or n==i+1 else qt.qeye(2) for n in range(L)]))

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, sz_list, progress_bar=True)

    # Convert lindblad results to numpy array
    lindblad_expect = np.array(result_lindblad.expect)
    
    # qiskit simulation
    generators_two_qubit = [Pauli("IX"), Pauli("XI"), Pauli("XX")] # [Pauli("IX"), Pauli("XI"), Pauli("XX")]
    generators_single_qubit = [Pauli("X")]
    SPLM_error_two_qubit = PauliLindbladError(generators_two_qubit, [qiskit_noise_factor, qiskit_noise_factor, qiskit_noise_factor]) #, qiskit_noise_factor])
    SPLM_error_single_qubit = PauliLindbladError(generators_single_qubit, [qiskit_noise_factor])
    noise_model = QiskitNoiseModel()
    for qubit in range(L - 1):
        noise_model.add_quantum_error(SPLM_error_two_qubit, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # noise_model.add_quantum_error(SPLM_error_single_qubit, ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"], [qubit])


    qiskit_results_list = [np.array([1.0, 1.0, 1.0, 1.0])]
    for i in range(timesteps):
        # Qiskit noisy simulator
        qc = create_ising_circuit(L, J, g, timestepsize, i+1, periodic=False)
        qiskit_results = qiskit_noisy_simulator(qc, noise_model, L, 1)
        qiskit_results_list.append(qiskit_results)

    print(len(qiskit_results_list))
    print(len(qiskit_results_list[0]))
    print("qiskit_results_list: ", qiskit_results_list)
    print("lindblad: ", result_lindblad.expect)

    # Transform qiskit results to match lindblad format
    # Extract the expectation values from qiskit results and reshape
    qiskit_expect_values = []
    
    # Handle the mixed structure (first element is 1D, rest are 2D)
    num_observables = len(lindblad_expect)  # Use the actual number from QuTiP results
    for i in range(num_observables):  # For each observable
        observable_values = []
        for j in range(len(qiskit_results_list)):  # For each time step
            if j == 0:
                # First element is 1D array
                observable_values.append(qiskit_results_list[j][i])
            else:
                # Rest are 2D arrays
                observable_values.append(qiskit_results_list[j][0][i])
        qiskit_expect_values.append(observable_values)
    
    qiskit_expect_values = np.array(qiskit_expect_values)
    lindblad_expect = np.array(result_lindblad.expect)
    
    print("qiskit_expect_values shape: ", qiskit_expect_values.shape)
    print("lindblad.expect shape: ", lindblad_expect.shape)

    # plot
    plt.figure(figsize=(12, 8))
    for i in range(len(lindblad_expect)):
        plt.subplot(len(lindblad_expect), 1, i+1)
        plt.plot(t, lindblad_expect[len(lindblad_expect)-i-1], label=f"QuTiP Observable {i+1}")
        plt.plot(t[:len(qiskit_expect_values[i])], qiskit_expect_values[i], label=f"Qiskit Observable {i+1}")
        plt.plot(t[:len(qiskit_expect_values[i])], qiskit_expect_values[i]-lindblad_expect[len(lindblad_expect)-i-1], label=f"difference Observable {i+1}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Expectation Value")
        plt.legend()
        plt.title(f"Observable {i+1}")
    
    plt.tight_layout()
    plt.show()
