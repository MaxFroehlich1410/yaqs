from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliError
from qiskit_aer.noise import depolarizing_error
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer.primitives import Estimator
from qiskit_aer import Aer
import numpy as np

def qiskit_dephasing_noise(num_qubits: int, noise_strengths: list) -> QiskitNoiseModel:
    """Create a Qiskit noise model with dephasing noise for single qubits and qubit pairs."""

    noise_model = QiskitNoiseModel()
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength


    # Single qubit dephasing
    single_qubit_dephasing = PauliError([Pauli('I'), Pauli('Z')], [1-single_qubit_strength, single_qubit_strength])
    # Two qubit ZZ dephasing
    two_qubit_dephasing = PauliError([Pauli('II'), Pauli('ZZ')], [1-pair_qubit_strength, pair_qubit_strength])

    for qubit in range(num_qubits):
        noise_model.add_quantum_error(single_qubit_dephasing, ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3"], [qubit])
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(two_qubit_dephasing, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])

    return noise_model

def qiskit_bitflip_noise(num_qubits: int, noise_strengths: list) -> QiskitNoiseModel:
    """Create a Qiskit noise model with dephasing noise for single qubits and qubit pairs."""

    noise_model = QiskitNoiseModel()
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength


    # Single qubit dephasing
    single_qubit_dephasing = PauliError([Pauli('I'), Pauli('X')], [1-single_qubit_strength, single_qubit_strength])
    # Two qubit ZZ dephasing
    two_qubit_dephasing = PauliError([Pauli('II'), Pauli('XX')], [1-pair_qubit_strength, pair_qubit_strength])

    for qubit in range(num_qubits):
        noise_model.add_quantum_error(single_qubit_dephasing, ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3"], [qubit])
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(two_qubit_dephasing, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])

    return noise_model

def qiskit_bitflip_noise_2(num_qubits: int, noise_strengths: list) -> QiskitNoiseModel:
    """Create a Qiskit noise model with dephasing noise for single qubits and qubit pairs."""

    noise_model = QiskitNoiseModel()
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength


    # Single qubit dephasing
    single_qubit_dephasing = PauliError([Pauli('I'), Pauli('X')], [1-single_qubit_strength, single_qubit_strength])
    single_qubit_dephasing_2 = PauliError([Pauli('I'), Pauli('X')], [1-single_qubit_strength*2, single_qubit_strength*2])
    # Two qubit ZZ dephasing
    two_qubit_dephasing = PauliError([Pauli('II'), Pauli('XX')], [1-pair_qubit_strength, pair_qubit_strength])

    for qubit in range(num_qubits):
        if qubit == 0:
            noise_model.add_quantum_error(single_qubit_dephasing, ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3"], [qubit])
        else:
            noise_model.add_quantum_error(single_qubit_dephasing_2, ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3"], [qubit])
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(two_qubit_dephasing, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])

    return noise_model







