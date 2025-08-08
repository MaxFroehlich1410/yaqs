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
from ..core.data_structures.simulation_parameters import WeakSimParams
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
    # DEBUG
    # print(f"[DEBUG][NOISE] Local noise model for sites [{first_site},{last_site}] -> {len(local_processes)} processes: {[(p['name'], p['sites'], p['strength']) for p in local_processes]}")
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

    # DEBUG
    # print(f"[DEBUG][LAYER] front_layer sizes -> 1Q:{len(single_qubit_nodes)} 2Q-even:{len(even_nodes)} 2Q-odd:{len(odd_nodes)}")

    return single_qubit_nodes, even_nodes, odd_nodes


def apply_single_qubit_gate(state: MPS, node: DAGOpNode) -> None:
    """Apply single qubit gate.

    This function applies a single-qubit gate to the MPS, used during circuit simulation.

    Parameters:
    state (MPS): The matrix product state (MPS) representing the quantum state.
    node (DAGOpNode): The directed acyclic graph (DAG) operation node representing the gate to be applied.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    # DEBUG
    # print(f"[DEBUG][APPLY-1Q] Gate {gate.name} on site {gate.sites[0]}")
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
    # DEBUG
    # print(f"[DEBUG][GEN-MPO] Gate {gate.name} sites {gate.sites} -> first_site={first_site} last_site={last_site}")
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

    # DEBUG
    # print(f"[DEBUG][WINDOW] window_size={window_size} -> window={window} short_len={short_state.length}")

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
    # DEBUG
    # print(f"[DEBUG][APPLY-2Q] Gate {gate.name} sites {gate.sites} -> TDVP on window {window}")
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
    _i, initial_state, noise_model, sim_params, circuit = args
        
    state = copy.deepcopy(initial_state)

    if sim_params.sample_layers:
        if sim_params.num_layers is None:
            raise ValueError("num_layers must be provided when sample_layers=True")
        if sim_params.basis_circuit is None:
            raise ValueError("basis_circuit must be provided when sample_layers=True")
        results = np.zeros((len(sim_params.sorted_observables), sim_params.num_layers + 1))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_layers:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            state.evaluate_observables(sim_params, results, 0)
        # DEBUG initial measurements
        # print(f"[DEBUG][LS] Initial column 0 measurement: norm={state.norm()} len={state.length}")
        # for obs_index, observable in enumerate(sim_params.sorted_observables):
            # print(f"[DEBUG][LS] obs[{obs_index}] {observable.gate.name} sites={observable.sites} -> {results[obs_index, 0]}")
       

        # Analyze basis circuit
        basis_dag = circuit_to_dag(sim_params.basis_circuit)
        basis_gates = [n for n in basis_dag.op_nodes() if n.op.name not in {"measure", "barrier"}]
        gates_per_layer = len(basis_gates)
        # print(f"[DEBUG][LS] gates_per_layer from basis_circuit = {gates_per_layer}")

        dag = circuit_to_dag(circuit)

        # Validate main circuit
        total_gates = len([n for n in dag.op_nodes() if n.op.name not in {"measure", "barrier"}])
        expected_total = gates_per_layer * sim_params.num_layers

        # print(f"[DEBUG][LS] total_gates(main)={total_gates} expected_total={expected_total}")
        if total_gates != expected_total:
            raise ValueError("Circuit structure mismatch!")
    else:
        dag = circuit_to_dag(circuit)
        gates_per_layer = 0  # Not used in non-layer sampling mode


    
    
    
    


    layer_count = 0
    gates_processed_in_current_layer = 0
    while dag.op_nodes():
        

        single_qubit_nodes, even_nodes, odd_nodes = process_layer(dag)

        for node in single_qubit_nodes:
            apply_single_qubit_gate(state, node)
            dag.remove_op_node(node)

            if sim_params.sample_layers:
                gates_processed_in_current_layer += 1
                # print(f"[DEBUG][LS] ++1Q gate processed; gates_in_layer={gates_processed_in_current_layer}/{gates_per_layer} layer={layer_count}")
                
                # Check if we completed a layer
                if gates_processed_in_current_layer == gates_per_layer and layer_count <= sim_params.num_layers - 1:
                    layer_count += 1
                    temp_state = copy.deepcopy(state)
                    # print(f"[DEBUG][LS] === LAYER {layer_count} boundary after 1Q; norm={temp_state.norm()} canonical={temp_state.check_canonical_form()}")
                    temp_state.evaluate_observables(sim_params, results, layer_count)
                    # for obs_index, observable in enumerate(sim_params.sorted_observables):
                        # print(f"[DEBUG][LS] layer={layer_count} obs[{obs_index}] {observable.gate.name} sites={observable.sites} -> {results[obs_index, layer_count]}")
                    gates_processed_in_current_layer = 0
                

        # Process two-qubit gates in even/odd sweeps.
        for group_name, group in [("even", even_nodes), ("odd", odd_nodes)]:
            for node in group:
                first_site, last_site = apply_two_qubit_gate(state, node, sim_params)
                
                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    # Normalizes state
                    state.normalize(form="B", decomposition="QR")
                else:
                    state.normalize(form="B", decomposition="QR")
                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    apply_dissipation(state, local_noise_model, dt=1, sim_params=sim_params)
                    state = stochastic_process(state, local_noise_model, dt=1, sim_params=sim_params)
                    state.normalize(form="B", decomposition="QR")

                dag.remove_op_node(node)

                if sim_params.sample_layers:
                    gates_processed_in_current_layer += 1
                    # print(f"[DEBUG][LS] ++2Q gate processed({group_name}); gates_in_layer={gates_processed_in_current_layer}/{gates_per_layer} layer={layer_count} norm={state.norm()}")
                    
                    # Check if we completed a layer
                    if gates_processed_in_current_layer == gates_per_layer and layer_count <= sim_params.num_layers - 1:
                        layer_count += 1
                        temp_state = copy.deepcopy(state)
                        # print(f"[DEBUG][LS] === LAYER {layer_count} boundary after 2Q; norm={temp_state.norm()} canonical={temp_state.check_canonical_form()}")
                        temp_state.evaluate_observables(sim_params, results, layer_count)
                        # for obs_index, observable in enumerate(sim_params.sorted_observables):
                            # print(f"[DEBUG][LS] layer={layer_count} obs[{obs_index}] {observable.gate.name} sites={observable.sites} -> {results[obs_index, layer_count]}")
                        gates_processed_in_current_layer = 0
                    


    if isinstance(sim_params, WeakSimParams):
        if not noise_model or all(proc["strength"] == 0 for proc in noise_model.processes):
            # All shots can be done at once in noise-free model
            if sim_params.get_state:
                sim_params.output_state = state
            return state.measure_shots(sim_params.shots)
        # Each shot is an individual trajectory
        return state.measure_shots(shots=1)
    
    # StrongSimParams
    temp_state = copy.deepcopy(state)
    if sim_params.get_state:
        sim_params.output_state = state

    last_site = 0
    for obs_index, observable in enumerate(sim_params.sorted_observables):
        if isinstance(observable.sites, list):
            idx = observable.sites[0]
        elif isinstance(observable.sites, int):
            idx = observable.sites

        if idx > last_site:
            for site in range(last_site, idx):
                temp_state.shift_orthogonality_center_right(site)
            last_site = idx
        
        expectation = temp_state.expect(observable)
        results[obs_index, sim_params.num_layers] = expectation

    return results