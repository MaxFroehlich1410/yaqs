from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS


import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit.quantum_info import SparsePauliOp, Pauli 
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer import Aer
import matplotlib.pyplot as plt



num_qubits = 4
qc = QuantumCircuit(num_qubits)
qc.rxx(np.pi/4, 1, 2)
qc.rzz(np.pi/4, 2, 3)



state = MPS(num_qubits, state="zeros")

sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(num_qubits)], max_bond_dim=2, threshold=1e-14, window_size=0, get_state=False)


simulator.run(state, qc, sim_params, noise_model=None)

for obs in sim_params.observables:
    print(obs.results)






