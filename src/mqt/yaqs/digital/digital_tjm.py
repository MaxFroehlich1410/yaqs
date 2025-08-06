# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital Tensor Jump Method.

This module provides functions for simulating quantum circuits using the Tensor Jump Method (TJM). It includes
utilities for converting quantum circuits to DAG representations, processing gate layers, applying gates to
matrix product states (MPS) and constructing generator MPOs.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from qiskit.converters import circuit_to_dag


from ..core.data_structures.networks import MPO, MPS
from ..core.data_structures.noise_model import NoiseModel
from ..core.data_structures.simulation_parameters import WeakSimParams, StrongSimParams
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.stochastic_process import stochastic_process
from ..core.methods.tdvp import two_site_tdvp
from .utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode

    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import StrongSimParams
    from ..core.libraries.gate_library import BaseGate

def debug_print(msg):
    """Helper function for debug output"""
    print(f"🔍 DEBUG: {msg}")

def analyze_circuit(circuit, name):
    """Analyze circuit structure for debugging"""
    debug_print(f"\n=== {name} Analysis ===")
    debug_print(f"Total instructions: {len(circuit.data)}")
    debug_print(f"Circuit size: {circuit.size()}")
    debug_print(f"Circuit depth: {circuit.depth()}")
    
    # Analyze gate types
    gate_counts = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    debug_print(f"Gate breakdown: {gate_counts}")
    
    # Convert to DAG and analyze
    dag = circuit_to_dag(circuit)
    dag_gates = [n for n in dag.op_nodes() if n.op.name not in {"measure", "barrier"}]
    debug_print(f"DAG effective gates (no measure/barrier): {len(dag_gates)}")
    
    # Show first few gates
    debug_print("First 10 gates:")
    for i, instruction in enumerate(circuit.data[:10]):
        # Fix: Use circuit.find_bit() to get qubit indices
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        debug_print(f"  {i}: {instruction.operation.name} on qubits {qubits}")
    
    debug_print(f"=== End {name} Analysis ===\n")
    return len(dag_gates)


def create_local_noise_model(noise_model: NoiseModel, first_site: int, last_site: int) -> NoiseModel:
    """Create local noise model.
    
    Create a local noise model from a global noise model for a given gate.
    
    Args: 
        noise_model (NoiseModel): The global noise model.
        first_site (int): The first site of the gate.
        last_site (int): The last site of the gate.

    Returns:
        NoiseModel: The local noise model.
    """
    local_processes = []
    gate_sites = [[i] for i in range(first_site, last_site+1)]
    neighbor_pairs = [[i, i+1] for i in range(first_site, last_site)]
    noise_model_copy = copy.deepcopy(noise_model)

    for process in noise_model_copy.processes:
        if process["sites"] in neighbor_pairs:
            local_processes.append(process)
        elif process["sites"] in gate_sites:
            local_processes.append(process)
    return NoiseModel(local_processes)



def process_layer(dag: DAGCircuit) -> tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]:
    """Process quantum circuit layer before applying to MPS.

    Processes the current layer of a DAGCircuit and categorizes nodes into single-qubit, even-indexed two-qubit,
    and odd-indexed two-qubit gates.

    Args:
        dag (DAGCircuit): The directed acyclic graph representing the quantum circuit.

    Returns:
        tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]: A tuple containing three lists:
            - single_qubit_nodes: Nodes corresponding to single-qubit gates.
            - even_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is even.
            - odd_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is odd.

    Raises:
        NotImplementedError: If a node with more than two qubits is encountered.
    """
    # Extract the current layer
    current_layer = dag.front_layer()

    # Prepare groups for even/odd two-qubit gates.
    single_qubit_nodes = []
    even_nodes = []
    odd_nodes = []

    # Separate the current layer into single-qubit and two-qubit gates.
    for node in current_layer:
        # Remove measurement and barrier nodes.
        if node.op.name in {"measure", "barrier"}:
            dag.remove_op_node(node)
            continue

        if len(node.qargs) == 1:
            single_qubit_nodes.append(node)
        elif len(node.qargs) == 2:
            # Group two-qubit gates by even/odd based on the lower qubit index.
            q0, q1 = node.qargs[0]._index, node.qargs[1]._index  # noqa: SLF001
            if min(q0, q1) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)
        else:
            raise NotImplementedError

    return single_qubit_nodes, even_nodes, odd_nodes


def apply_single_qubit_gate(state: MPS, node: DAGOpNode) -> None:
    """Apply single qubit gate.

    This function applies a single-qubit gate to the MPS, used during circuit simulation.

    Parameters:
    state (MPS): The matrix product state (MPS) representing the quantum state.
    node (DAGOpNode): The directed acyclic graph (DAG) operation node representing the gate to be applied.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    state.tensors[gate.sites[0]] = oe.contract("ab, bcd->acd", gate.tensor, state.tensors[gate.sites[0]])


def construct_generator_mpo(gate: BaseGate, length: int) -> tuple[MPO, int, int]:
    """Construct Generator MPO.

    Constructs a Matrix Product Operator (MPO) representation of a generator for a given gate over a
      specified length.

    Args:
        gate (BaseGate): The gate containing the generator and the sites it acts on.
        length (int): The total number of sites in the system.

    Returns:
        tuple[MPO, int, int]: A tuple containing the constructed MPO, the first site index, and the last site index.
    """
    tensors = []

    if gate.sites[0] < gate.sites[1]:
        first_gen = 0
        second_gen = 1
    else:
        first_gen = 1
        second_gen = 0

    first_site = gate.sites[first_gen]
    last_site = gate.sites[second_gen]
    for site in range(length):
        if site == first_site:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[first_gen]
            tensors.append(w)
        elif site == last_site:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[second_gen]
            tensors.append(w)
        else:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = np.eye(2)
            tensors.append(w)

    mpo = MPO()
    mpo.init_custom(tensors)
    return mpo, first_site, last_site


def apply_window(state: MPS, mpo: MPO, first_site: int, last_site: int, window_size: int) -> tuple[MPS, MPO, list[int]]:
    """Apply Window.

    Apply a window to the given MPS and MPO for a local update.

    Args:
        state (MPS): The matrix product state (MPS) to be updated.
        mpo (MPO): The matrix product operator (MPO) to be applied.
        first_site (int): The index of the first site in the window.
        last_site (int): The index of the last site in the window.
        window_size: Number of sites on each side of first and last site

    Returns:
        tuple[MPS, MPO, list[int]]: A tuple containing the shortened MPS, the shortened MPO, and the window indices.
    """
    # Define a window for a local update.
    window = [first_site - window_size, last_site + window_size]
    window[0] = max(window[0], 0)
    window[1] = min(window[1], state.length - 1)

    # Shift the orthogonality center for sites before the window.
    for i in range(window[0]):
        state.shift_orthogonality_center_right(i)

    short_mpo = MPO()
    short_mpo.init_custom(mpo.tensors[window[0] : window[1] + 1], transpose=False)
    assert window[1] - window[0] + 1 > 1, "MPS cannot be length 1"
    short_state = MPS(length=window[1] - window[0] + 1, tensors=state.tensors[window[0] : window[1] + 1])

    return short_state, short_mpo, window


def apply_two_qubit_gate(state: MPS, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> tuple[int, int]:
    """Apply two-qubit gate.

    Applies a two-qubit gate to the given Matrix Product State (MPS) with dynamic TDVP.

    Args:
        state (MPS): The Matrix Product State to which the gate will be applied.
        node (DAGOpNode): The node representing the two-qubit gate in the Directed Acyclic Graph (DAG).
        sim_params (StrongSimParams | WeakSimParams): Simulation parameters that determine the behavior
        of the algorithm.

    """
    # Construct the MPO for the two-qubit gate.
    gate = convert_dag_to_tensor_algorithm(node)[0]
    mpo, first_site, last_site = construct_generator_mpo(gate, state.length)

    window_size = 1
    short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)
    two_site_tdvp(short_state, short_mpo, sim_params)
    # Replace the updated tensors back into the full state.
    for i in range(window[0], window[1] + 1):
        state.tensors[i] = short_state.tensors[i - window[0]]
    
    return first_site, last_site


def digital_tjm(
    args: tuple[int, MPS, NoiseModel | None, StrongSimParams | WeakSimParams, QuantumCircuit],
) -> NDArray[np.float64]:
    """Circuit Tensor Jump Method.

    Simulates a quantum circuit using the Tensor Jump Method.

    Args:
        args (tuple): A tuple containing the following elements:
            - int: An index or identifier, primarily for parallelization
            - MPS: The initial state of the system represented as a Matrix Product State.
            - NoiseModel | None: The noise model to be applied during the simulation, or None if no noise is
                to be applied.
            - StrongSimParams | WeakSimParams: Parameters for the simulation, either for strong or weak simulation.
            - QuantumCircuit: The quantum circuit to be simulated.

    Returns:
        NDArray[np.float64]: The results of the simulation. If StrongSimParams are used, the results
        are the measured observables.
        If WeakSimParams are used, the results are the measurement outcomes for each shot.
    """
    trajectory_id, initial_state, noise_model, sim_params, circuit = args
        
    state = copy.deepcopy(initial_state)

    dag = circuit_to_dag(circuit)

    # Debug print helper
    def debug_print(msg):
        print(f"🔧 TJM DEBUG [Traj {trajectory_id}]: {msg}")

    debug_print("=== STARTING DIGITAL TJM ===")
    debug_print(f"Circuit qubits: {circuit.num_qubits}")
    debug_print(f"Circuit size: {circuit.size()}")
    debug_print(f"Initial DAG nodes: {len(list(dag.op_nodes()))}")

    # Check for layer sampling
    layer_sampling = (
        isinstance(sim_params, StrongSimParams) and 
        getattr(sim_params, 'sample_layers', False)
    )
    debug_print(f"Layer sampling enabled: {layer_sampling}")
    
    if layer_sampling:
        num_layers = getattr(sim_params, 'num_layers', None)
        basis_circuit = getattr(sim_params, 'basis_circuit', None)
        
        debug_print(f"num_layers: {num_layers}")
        debug_print(f"basis_circuit provided: {basis_circuit is not None}")
        
        if num_layers is None:
            raise ValueError("num_layers must be provided when sample_layers=True")
        if basis_circuit is None:
            raise ValueError("basis_circuit must be provided when sample_layers=True")
        
        # Analyze basis circuit
        basis_dag = circuit_to_dag(basis_circuit)
        basis_gates = [n for n in basis_dag.op_nodes() if n.op.name not in {"measure", "barrier"}]
        gates_per_layer = len(basis_gates)
        
        analyze_circuit(basis_circuit, "BASIS CIRCUIT")
        
        # Validate main circuit
        total_gates = len([n for n in dag.op_nodes() if n.op.name not in {"measure", "barrier"}])
        expected_total = gates_per_layer * num_layers
        
        debug_print(f"=== CONCATENATED CIRCUIT VALIDATION ===")
        debug_print(f"Expected gates: {gates_per_layer} × {num_layers} = {expected_total}")
        debug_print(f"Actual gates: {total_gates}")
        debug_print(f"Match: {total_gates == expected_total}")
        
        if total_gates != expected_total:
            debug_print(f"⚠️  WARNING: Circuit structure mismatch!")
            # Don't raise error, just warn and continue
        
        # Sample initial state (layer 0) - store directly in trajectories!
        debug_print("=== SAMPLING INITIAL STATE (Layer 0) ===")
        temp_state = copy.deepcopy(state)
        for obs_idx, observable in enumerate(sim_params.observables):
            expectation = temp_state.expect(observable)
            sim_params.observables[obs_idx].trajectories[trajectory_id, 0] = expectation
            print(f" sim_params.observables[obs_idx].trajectories[trajectory_id, 0]: {sim_params.observables[obs_idx].trajectories[trajectory_id, 0]}")
            debug_print(f"Initial obs[{obs_idx}] = {expectation:.6f}")
            debug_print(f"trajectories obsevable: {sim_params.observables[obs_idx].trajectories}")
            # debug_print(f"trajectories sorted_observable: {sim_params.sorted_observables[obs_idx].trajectories}")
            # debug_print("test")
        
        current_layer = 1
        gates_processed_in_current_layer = 0
        debug_print(f"Layer sampling initialized: targeting {num_layers} layers")

    layer_count = 0
    total_gates_processed = 0
    
    while dag.op_nodes():
        layer_count += 1
        debug_print(f"\n--- Processing DAG Layer {layer_count} ---")

        single_qubit_nodes, even_nodes, odd_nodes = process_layer(dag)
        
        debug_print(f"Layer gates - Single: {len(single_qubit_nodes)}, Even: {len(even_nodes)}, Odd: {len(odd_nodes)}")

        # Process single-qubit gates
        debug_print("Processing single-qubit gates:")
        for i, node in enumerate(single_qubit_nodes):
            qubits = [dag.qubits.index(q) for q in node.qargs]  # Use DAG's qubit list to find indices
            debug_print(f"  Single[{i}]: {node.op.name} on qubit {qubits}")
            
            apply_single_qubit_gate(state, node)
            dag.remove_op_node(node)
            total_gates_processed += 1
            
            if layer_sampling:
                gates_processed_in_current_layer += 1
                debug_print(f"    Layer progress: {gates_processed_in_current_layer}/{gates_per_layer}")
                
                # Check if we completed a layer
                if gates_processed_in_current_layer == gates_per_layer and current_layer <= num_layers:
                    debug_print(f"🎯 LAYER 222 {current_layer} COMPLETED! Sampling observables...")
                    temp_state = copy.deepcopy(state)
                    for obs_idx, observable in enumerate(sim_params.observables):
                        expectation = temp_state.expect(observable)
                        sim_params.observables[obs_idx].trajectories[trajectory_id, current_layer] = expectation
                        debug_print(f" trajectory 111: {sim_params.observables[obs_idx].trajectories}")
                        debug_print(f"Layer {current_layer} obs[{obs_idx}] = {expectation}")
                    
                    current_layer += 1
                    gates_processed_in_current_layer = 0
                    debug_print(f"Reset for next layer. Current layer now: {current_layer}")

        # Process two-qubit gates in even/odd sweeps.
        for group_name, group in [("even", even_nodes), ("odd", odd_nodes)]:
            debug_print(f"Processing {group_name} two-qubit gates:")
            for i, node in enumerate(group):
                qubits = [dag.qubits.index(q) for q in node.qargs]  # Use DAG's qubit list to find indices
                debug_print(f"  {group_name}[{i}]: {node.op.name} on qubits {qubits}")
                
                first_site, last_site = apply_two_qubit_gate(state, node, sim_params)
                
                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    # Normalizes state
                    debug_print("    No noise - just normalizing")
                    state.normalize(form="B", decomposition="QR")
                else:
                    debug_print(f"    Applying noise model on sites {first_site}-{last_site}")
                    state.normalize(form="B", decomposition="QR")

                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    debug_print(f"    Local noise processes: {len(local_noise_model.processes)}")

                    apply_dissipation(state, local_noise_model, dt=1, sim_params=sim_params)
                    debug_print("    Applied dissipation")
                    debug_print(f"    State norm after dissipation: {state.norm()}")

                    state = stochastic_process(state, local_noise_model, dt=1, sim_params=sim_params)
                    debug_print("    Applied stochastic process")

                    state.normalize(form="B", decomposition="QR")

                dag.remove_op_node(node)
                total_gates_processed += 1
                
                if layer_sampling:
                    gates_processed_in_current_layer += 1
                    debug_print(f"    Layer progress: {gates_processed_in_current_layer}/{gates_per_layer}")
                    
                    # Check if we completed a layer
                    if gates_processed_in_current_layer == gates_per_layer and current_layer <= num_layers:
                        debug_print(f"🎯 LAYER 111 {current_layer} COMPLETED! Sampling observables...")
                        temp_state = copy.deepcopy(state)
                        for obs_idx, observable in enumerate(sim_params.observables):
                            expectation = temp_state.expect(observable)
                            sim_params.observables[obs_idx].trajectories[trajectory_id, current_layer] = expectation
                            debug_print(f"Layer {current_layer} obs[{obs_idx}] = {expectation:.6f}")
                            debug_print(f" trajectory 222: {sim_params.observables[obs_idx].trajectories}")
                            debug_print(f"Layer {current_layer} obs[{obs_idx}] = {expectation:.6f}")
                        
                        current_layer += 1
                        gates_processed_in_current_layer = 0
                        debug_print(f"Reset for next layer. Current layer now: {current_layer}")

    debug_print(f"\n=== SIMULATION COMPLETE ===")
    debug_print(f"Total DAG layers processed: {layer_count}")
    debug_print(f"Total gates processed: {total_gates_processed}")
    if layer_sampling:
        debug_print(f"Final layer: {current_layer - 1}, Gates in partial layer: {gates_processed_in_current_layer}")

    if isinstance(sim_params, WeakSimParams):
        if not noise_model or all(proc["strength"] == 0 for proc in noise_model.processes):
            # All shots can be done at once in noise-free model
            if sim_params.get_state:
                sim_params.output_state = state
            debug_print("Returning measurement shots")
            return state.measure_shots(sim_params.shots)
        # Each shot is an individual trajectory
        debug_print("Returning single shot measurement")
        return state.measure_shots(shots=1)
    
    # StrongSimParams
    if layer_sampling:
        # For layer sampling, we've already stored everything in trajectories
        # Return the final layer results for compatibility
        debug_print("Returning layer sampling results")
        final_results = np.zeros((len(sim_params.observables), 1))
        for obs_idx in range(len(sim_params.observables)):
            final_results[obs_idx, 0] = sim_params.observables[obs_idx].trajectories[trajectory_id, -1]
            debug_print(f"Final result obs[{obs_idx}] = {final_results[obs_idx, 0]:.6f}")
        return final_results
    else:
        # Standard StrongSimParams logic (unchanged)
        debug_print("Standard observable measurement")
        results = np.zeros((len(sim_params.observables), 1))
        temp_state = copy.deepcopy(state)
        if sim_params.get_state:
            sim_params.output_state = state

        last_site = 0
        for obs_index, observable in enumerate(sim_params.observables):
            if isinstance(observable.sites, list):
                idx = observable.sites[0]
            elif isinstance(observable.sites, int):
                idx = observable.sites

            if idx > last_site:
                for site in range(last_site, idx):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = idx
            
            expectation = temp_state.expect(observable)
            results[obs_index, 0] = expectation
            debug_print(f"Standard obs[{obs_index}] = {expectation:.6f}")

        return results