import numpy as np
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def create_yaqs_dephasing_noise(num_qubits: int, noise_strengths: list) -> NoiseModel:
    """Create a YAQS noise model with dephasing noise for single qubits and qubit pairs."""
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength
    
    processes = []
    
    # Single qubit dephasing
    for qubit in range(num_qubits):
        processes.append({
            "name": "dephasing",
            "sites": [qubit],
            "strength": single_qubit_strength
        })
    
    # Two qubit ZZ dephasing (equivalent to double_dephasing in NoiseLibrary)
    for qubit in range(num_qubits - 1):
        processes.append({
            "name": "double_dephasing",
            "sites": [qubit, qubit + 1],
            "strength": pair_qubit_strength
        })
    
    return NoiseModel(processes)

def create_yaqs_bitflip_noise_2(num_qubits: int, noise_strengths: list) -> NoiseModel:
    """Create a YAQS noise model with dephasing noise for single qubits and qubit pairs."""
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength
    
    processes = []
    
    # Single qubit dephasing
    for qubit in range(num_qubits):
        if qubit == 0:
            processes.append({
                "name": "x",
                "sites": [qubit],
                "strength": single_qubit_strength
            })
        else:
            processes.append({
                "name": "x",
                "sites": [qubit],
                "strength": single_qubit_strength*2
            })
    
    # Two qubit XX bitflip (equivalent to double_dephasing in NoiseLibrary)
    for qubit in range(num_qubits - 1):
        processes.append({
            "name": "xx",
            "sites": [qubit, qubit + 1],
            "strength": pair_qubit_strength
        })
    
    return NoiseModel(processes)

def create_yaqs_bitflip_noise(num_qubits: int, noise_strengths: list) -> NoiseModel:
    """Create a YAQS noise model with dephasing noise for single qubits and qubit pairs."""
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength
    
    processes = []

    
    # Single qubit dephasing
    for qubit in range(num_qubits):
        processes.append({
            "name": "x",
            "sites": [qubit],
            "strength": single_qubit_strength
        })

    
    # Two qubit XX bitflip (equivalent to double_dephasing in NoiseLibrary)
    for qubit in range(num_qubits - 1):
        processes.append({
            "name": "xx",
            "sites": [qubit, qubit + 1],
            "strength": pair_qubit_strength
        })
    
    for process in processes:
        process["strength"] = process["strength"]/np.sqrt(len(processes))
    
    return NoiseModel(processes)

if __name__ == "__main__":
    num_qubits = 2
    noise_strengths = [0.4, 0.2]
    noise_model = create_yaqs_bitflip_noise(num_qubits, noise_strengths)
    print(noise_model)