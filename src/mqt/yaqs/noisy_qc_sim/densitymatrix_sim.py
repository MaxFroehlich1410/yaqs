import numpy as np
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit


from mqt.yaqs.circuits.utils.dag_utils import convert_dag_to_tensor_algorithm
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.noise_library import NoiseLibrary

from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerSimulator
import matplotlib.pyplot as plt


def expand_operator(local_op: np.ndarray, site: int | list[int], n_qubits: int) -> np.ndarray:
    """Expand a single-qubit operator to act on 'site' in an n-qubit system."""
    ops = [np.eye(2)] * n_qubits
    if isinstance(site, list):
        ops[site[0]] = local_op
        ops.pop(site[1])
    else:
        ops[site] = local_op
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def KrausChannel(rho, noisemodel, sites):
    """
    Apply a Kraus channel (from NoiseModel) selectively to 'sites' of the density matrix 'rho'.
    If len(sites)==1, all local ops for that site are used (standard local channel).
    If len(sites)==2, Kraus ops are all local ops for site 0 (embedded), and all for site 1 (embedded),
    but **never** tensor products where both are non-identity.
    """
    if noisemodel is None or not noisemodel.processes:
        return rho
    # print(f"KrausChannel called with sites={sites}")
    
    n_qubits = int(np.log2(rho.shape[0]))
    kraus_ops_global = []
    total_strength = 0
    

    # For all processes in noisemodel, apply those that act on exactly these sites (for one- and two-site channels)
    for process in noisemodel.processes:
        if len(sites) == 1 and process["sites"] == sites:
            total_strength += process["strength"]
            # single-site channel
            # print(f'strength len sites 1: {process["strength"]}')
            local_K = np.sqrt(process["strength"]) * process["jump_operator"]
            global_K = expand_operator(local_K, sites[0], n_qubits)
            # print(f'global_K: {global_K}')
            kraus_ops_global.append(global_K)
        elif len(sites) == 2:
            # collect any process that acts exactly on [i], [i+1], or [i, i+1]
            if process["sites"] == [sites[0]] or process["sites"] == [sites[1]]:
                total_strength += process["strength"]
                # single-site process, embed on correct site
                site_idx = process["sites"][0]
                # print(f'strength len sites 2 single operator: {process["strength"]}')
                local_K = np.sqrt(process["strength"]) * process["jump_operator"]
                global_K = expand_operator(local_K, site_idx, n_qubits)
                # print(f'global_K: {global_K}')
                kraus_ops_global.append(global_K)
            elif process["sites"] == sites:
                # two-site process, acts on both sites simultaneously
                total_strength += process["strength"]
                # print(f'strength len sites 2 double operator: {process["strength"]}')
                local_K = np.sqrt(process["strength"]) * process["jump_operator"]
                # print(f'local_K: {local_K}', f'process["strength"]: {process["strength"]}')
                global_K = expand_operator(local_K, sites, n_qubits)
                # print(f'global_K: {global_K}')
                kraus_ops_global.append(global_K)

    # print(f'total_strength: {total_strength}')
    kraus_ops_global.append(np.sqrt(1-total_strength) * np.eye(rho.shape[0]))
    # Kraus channel application
    result = np.zeros_like(rho, dtype=complex)
    #print(f'length of kraus_ops_global: {len(kraus_ops_global)}')
    for K in kraus_ops_global:
        # print(f"Applying Kraus operator {K} to density matrix {rho}")
        # print(f"Applying Kraus operator {K}")
        result += K @ rho @ K.conj().T

    return result

def z_expectations(rho, num_qubits):
    """
    Compute <Z> for each qubit for the given density matrix.
    """
    z_vals = []
    sz = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    for i in range(num_qubits):
        op = 1
        for j in range(num_qubits):
            op = np.kron(op, sz if i == j else I)
        z_vals.append(np.real(np.trace(rho @ op)))
    return np.array(z_vals)

def two_qubit_reverse(mat):
    """
    For a 4x4 gate acting on qubits (a, b), swap a and b.
    Only needed if cx order is reversed.
    """
    SWAP = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
    mat = SWAP @ mat @ SWAP 
    return mat

    
def create_all_zero_density_matrix(n_qubits):
    """
    Creates the density matrix for an n-qubit all-zero state (|0...0><0...0|).

    Args:
        n_qubits (int): The number of qubits.

    Returns:
        np.ndarray: The (2^n x 2^n) density matrix.
    """
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0  # Set the |0...0><0...0| element to 1
    return rho

def circuit_to_unitary_list(circuit):
    """
    Convert a Qiskit circuit to a list of gates with qubit indices.
    Returns a list of tuples (unitary, [qubit indices])
    """
    dag = circuit_to_dag(circuit)
    return convert_dag_to_tensor_algorithm(dag)




def evolve_noisy_circuit(rho0, gate_list, noisemodel, num_layers):
    """
    Evolve a density matrix rho0 through the list of gates,
    applying Kraus noise after every gate as specified in kraus_channel_map.
    kraus_channel_map: dict with keys '1q' and '2q' for 1- and 2-qubit noise
    Returns: final density matrix
    """
    n = int(np.log2(rho0.shape[0]))
    rho = np.copy(rho0)
    z_expvals = []
    # print(f"Evolving circuit with {len(gate_list)} gates")
    
    for layer in range(num_layers):
        for gate in gate_list:
            # Expand gate to full Hilbert space
            if len(gate.sites) == 1:
                U = np.eye(1)
                for i in range(n):
                    U = np.kron(U, gate.matrix if i == gate.sites[0] else np.eye(2))
        
            elif len(gate.sites) == 2:
                if np.abs(gate.sites[-1] - gate.sites[0]) > 1:
                    raise ValueError("Non-adjacent two-qubit gates not supported")

                idx0, idx1 = gate.sites[0], gate.sites[1]
                if idx0 > idx1:
                    idx0, idx1 = idx1, idx0
                    gate.matrix = two_qubit_reverse(gate.matrix)
                U = np.eye(1)
                i = 0
                while i < n:
                    if len(gate.sites) == 2 and i == idx0:
                        U = np.kron(U, gate.matrix)
                        i += 2  # skip both qubits (idx0, idx1)
                    else:
                        U = np.kron(U, np.eye(2))
                        i += 1
            else:
                raise ValueError("Only 1- and 2-qubit gates supported")
            
            # Apply unitary    
            rho = U @ rho @ U.conj().T

            # Apply noise
            # print(f'noisemodel processes: {noisemodel.processes}')
            rho = KrausChannel(rho, noisemodel, gate.sites)

        z_expvals.append(z_expectations(rho, n))
        # print('diagonal elements of rho: ', np.diag(rho))

    return np.array(z_expvals)




