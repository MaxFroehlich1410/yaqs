# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit Tensor Jump Method.

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
from ..core.data_structures.simulation_parameters import WeakSimParams
from ..core.data_structures.noise_model import NoiseModel
from ..core.methods.dissipation import apply_dissipation, apply_circuit_dissipation
from ..core.methods.stochastic_process import circuit_stochastic_process, stochastic_process
from ..core.methods.tdvp import local_dynamic_tdvp, two_site_tdvp
from .utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode

    from ..core.data_structures.simulation_parameters import StrongSimParams
    from ..core.libraries.gate_library import BaseGate


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


def apply_window(state: MPS, mpo: MPO | None, first_site: int, last_site: int, window_size: int) -> tuple[MPS, MPO, list[int]]:
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
    print("-"*80)
    print(f"DEBUG: apply_window called")
    print(f"DEBUG: first_site: {first_site}, last_site: {last_site}, window_size: {window_size}")
    print(f"DEBUG: state canonical form before apply_window: {state.check_canonical_form()}")
    print(f"DEBUG: state norm before apply_window site 0: {state.norm(0)}")
    print("-"*80)
    # Define a window for a local update.
    window = [first_site - window_size, last_site + window_size]
    window[0] = max(window[0], 0)
    window[1] = min(window[1], state.length - 1)

    # Shift the orthogonality center for sites before the window.
    print(f"DEBUG: Window: {window}")
    print(f"DEBUG: State length: {state.length}")
    print(f"DEBUG: window[0], window[1]: {window[0]}, {window[1]}")
    for i in range(window[0]):
        state.shift_orthogonality_center_right(i)
    print(f"DEBUG: State canonical form after shift: {state.check_canonical_form()}")
    if mpo is not None:
        short_mpo = MPO()
        short_mpo.init_custom(mpo.tensors[window[0] : window[1] + 1], transpose=False)
    else:
        short_mpo = None
    assert window[1] - window[0] + 1 > 1, "MPS cannot be length 1"
    short_state = MPS(length=window[1] - window[0] + 1, tensors=state.tensors[window[0] : window[1] + 1])
    print(f"DEBUG: short state canonical form: {short_state.check_canonical_form()}")
    copy_state = copy.deepcopy(state)
    copy_state.set_canonical_form(0)
    print(f"DEBUG: copy state norm after set canonical form 0: {copy_state.norm(0)}")
    print("-"*80)
    print("apply_window done")
    print("-"*80)
    return short_state, short_mpo, window


def apply_two_qubit_gate(state: MPS, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> None:
    """Apply two-qubit gate.

    Applies a two-qubit gate to the given Matrix Product State (MPS) with dynamic TDVP.

    Args:
        state (MPS): The Matrix Product State to which the gate will be applied.
        node (DAGOpNode): The node representing the two-qubit gate in the Directed Acyclic Graph (DAG).
        sim_params (StrongSimParams | WeakSimParams): Simulation parameters that determine the behavior
        of the algorithm.

    .
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]

    # Construct the MPO for the two-qubit gate.
    mpo, first_site, last_site = construct_generator_mpo(gate, state.length)

    window_size = 1
    print(f"DEBUG: State canonical form INSIDE APPLY_TWO_QUBIT_GATE before apply_window: {state.check_canonical_form()}")
    print(f"DEGUB: first_site: {first_site}, last_site: {last_site}")
    short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)
    if np.abs(first_site - last_site) == 1:
        # Apply two-site TDVP for nearest-neighbor gates.
        two_site_tdvp(short_state, short_mpo, sim_params)
    else:
        local_dynamic_tdvp(short_state, short_mpo, sim_params)
    print(f"DEBUG: Short state canonical form after TDVP: {short_state.check_canonical_form()}")
    # Replace the updated tensors back into the full state.
    for i in range(window[0], window[1] + 1):
        state.tensors[i] = short_state.tensors[i - window[0]]
    print(f"DEBUG: State canonical form after tensor replacement: {state.check_canonical_form()}")
    return first_site, last_site


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
        
    # print(f"DEBUG: Gate sites: {gate_sites}")
    # print(f"DEBUG: Neighbor pairs: {neighbor_pairs}")
    # print(f"DEBUG: Gate acts on sites: {first_site} to {last_site}")
    noise_model_copy = copy.deepcopy(noise_model)

    for process in noise_model_copy.processes:
        # print(f"DEBUG: Checking process {process['sites']} against gate_sites: {gate_sites}")
        # print(f"DEBUG: Checking process {process['sites']} against neighbor_pairs: {neighbor_pairs}")
        if process["sites"] in neighbor_pairs:
            # print(f"DEBUG: Adding neighbor process: {process}")
            local_processes.append(process)
        elif process["sites"] in gate_sites:
            # print(f"DEBUG: Adding gate site process: {process}")
            local_processes.append(process)
        # else:
        #     # print(f"DEBUG: Process {process['sites']} not included in local noise model")

    return NoiseModel(local_processes)

# def apply_noisy_two_qubit_gate(state: MPS, noise_model: NoiseModel, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> None:
#     """Apply two-qubit gate.

#     Applies a two-qubit gate to the given Matrix Product State (MPS) with dynamic TDVP.

#     Args:
#         state (MPS): The Matrix Product State to which the gate will be applied.
#         node (DAGOpNode): The node representing the two-qubit gate in the Directed Acyclic Graph (DAG).
#         sim_params (StrongSimParams | WeakSimParams): Simulation parameters that determine the behavior
#         of the algorithm.

#     .
#     """
#     # print("="*80)
#     # print("DEBUG: apply_noisy_two_qubit_gate called")
#     # print(f"DEBUG: Gate node: {node.op.name}")
#     # print(f"DEBUG: Gate qubits: {[q._index for q in node.qargs]}")
#     # print(f"DEBUG: State length: {state.length}")
#     # print(f"DEBUG: State norm before gate: {state.norm()}")
    
#     gate = convert_dag_to_tensor_algorithm(node)[0]
#     # print(f"DEBUG: Gate tensor shape: {gate.tensor.shape}")
#     # print(f"DEBUG: Gate sites: {gate.sites}")

#     # Construct the MPO for the two-qubit gate.
#     mpo, first_site, last_site = construct_generator_mpo(gate, state.length)
#     # print(f"DEBUG: MPO constructed, first_site: {first_site}, last_site: {last_site}")

#     window_size = 1
#     print(f"DEBUG: State canonical form before window: {state.check_canonical_form()}")
#     short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)
#     first_local = first_site - window[0]
#     last_local = last_site - window[0]
#     print(f"DEBUG: first_local: {first_local}, last_local: {last_local}")
#     #print(f"DEBUG: Window: {window}, short_state length: {short_state.length}")
#     # print(f"DEBUG: Short state norm before TDVP: {short_state.norm()}")
#     print(f"DEBUG: Short state canonical form after window: {short_state.check_canonical_form()}")
#     print(f"DEBUG: state canonical form after window: {state.check_canonical_form()}")
#     if np.abs(first_site - last_site) == 1:
#         # Apply two-site TDVP for nearest-neighbor gates.
#         # print("DEBUG: Applying two-site TDVP")
#         two_site_tdvp(short_state, short_mpo, sim_params)
#     else:
#         # print("DEBUG: Applying local dynamic TDVP")
#         local_dynamic_tdvp(short_state, short_mpo, sim_params)
#     print(f"DEBUG: Short state canonical form after TDVP: {short_state.check_canonical_form()}")
    
#     # set canonical form to first site
#     short_state.set_canonical_form(first_local)
#     print(f"DEBUG: Now short state canonical form should be first_local: {first_local}: {short_state.check_canonical_form()}")
#     affected_state = MPS(length=2, tensors=short_state.tensors[first_local : last_local + 1])

#     # get local noise model from global noise model
#     local_noise_model = create_local_noise_model(noise_model, first_site, last_site)   

#     # apply noise to qubits affected by the gate

#     print(f"DEBUG: Affected state canonical form before dissipation CIRCUIT TJM: {affected_state.check_canonical_form()}")
#     apply_circuit_dissipation(affected_state, local_noise_model, dt=1, global_start=first_site, sim_params=sim_params)
    
#     print(f"DEBUG: 111 short state canonical form after dissipation CIRCUIT TJM: {short_state.check_canonical_form()}")
#     print(f"DEBUG: 111 affected state canonical form after dissipation CIRCUIT TJM: {affected_state.check_canonical_form()}")
#     print(f"DEBUG: 111 full state canonical form after dissipation CIRCUIT TJM: {state.check_canonical_form()}")
#     print(f"DEBUG: 111 affected state norm after dissipation CIRCUIT TJM: {affected_state.norm(0)}")
    
#     affected_state = circuit_stochastic_process(affected_state, local_noise_model, dt=1, global_start=first_site, sim_params=sim_params)
#     # print(f"DEBUG: Short state norm after stochastic process: {short_state.norm()}")
#     print(f"DEBUG: 222 Affected state canonical form after stochastic process CIRCUIT TJM: {affected_state.check_canonical_form()}")
#     print(f"DEBUG: 222 short state canonical form after stochastic process CIRCUIT TJM: {short_state.check_canonical_form()}")
#     print(f"DEBUG: 222 full state canonical form after stochastic process CIRCUIT TJM: {state.check_canonical_form()}")
    
#     affected_state.set_canonical_form(0)
#     short_state.set_canonical_form(first_local)
#     print(f"DEBUG: 222 short state canonical form after set canonical form CIRCUIT TJM. Should be first_local: {first_local}: {short_state.check_canonical_form()}")

#     # replace tensors in window with affected state
#     short_state.tensors[first_local : last_local + 1] = affected_state.tensors[:]

#     short_state.set_canonical_form(window[0])
#     print(f"DEBUG: window[0]: {window[0]}")
#     print(f"DEBUG: 333 short state canonical form after tensor replacement CIRCUIT TJM: {short_state.check_canonical_form()}")
   

#     # Replace the updated tensors back into the full state.
#     state.set_canonical_form(window[0])
#     print(f"DEBUG: 333 state canonical form after set canonical form CIRCUIT TJM. Should be window[0]: {window[0]}: {state.check_canonical_form()}")
#     for i in range(window[0], window[1] + 1):
#         state.tensors[i] = short_state.tensors[i - window[0]]
#     state.normalize("B", "QR")

#     # Replace the updated tensors back into the full state.
#     # print(f"DEBUG: Replacing tensors from window {window}")
#     # state.tensors[first_site] = affected_state.tensors[0]
#     # state.tensors[last_site] = affected_state.tensors[1]
#     print(f"DEBUG: State canonical form after tensor replacement and after diss+jump: {state.check_canonical_form()}")

    
#     # print(f"DEBUG: State norm after tensor replacement: {state.norm()}")
#     # print("="*80)



def circuit_tjm(
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
    
    # print(f"\n{'#'*80}")
    # print(f"DEBUG: Starting circuit_tjm simulation (trajectory {_i})")
    # print(f"DEBUG: Circuit has {circuit.num_qubits} qubits")
    # print(f"DEBUG: Circuit depth: {circuit.depth()}")
    # print(f"DEBUG: Initial state norm: {initial_state.norm()}")
    #if noise_model is not None:
        # print(f"DEBUG: Noise model has {len(noise_model.processes)} processes")
        # for i, proc in enumerate(noise_model.processes):
        #     print(f"DEBUG: Global noise process {i}: {proc}")
    # print(f"{'#'*80}\n")
    
    state = copy.deepcopy(initial_state)
    state_copy = copy.deepcopy(state)
    print(f"DEBUG: Canonical form in circuit_tjm: {state_copy.check_canonical_form()}")

    dag = circuit_to_dag(circuit)
    # print(f"DEBUG: DAG has {len(dag.op_nodes())} operation nodes")

    layer_count = 0
    while dag.op_nodes():
        layer_count += 1
        # print(f"\nDEBUG: Processing layer {layer_count}")
        
        single_qubit_nodes, even_nodes, odd_nodes = process_layer(dag)
        # print(f"DEBUG: Layer {layer_count} - Single qubit gates: {len(single_qubit_nodes)}")
        # print(f"DEBUG: Layer {layer_count} - Even two-qubit gates: {len(even_nodes)}")
        # print(f"DEBUG: Layer {layer_count} - Odd two-qubit gates: {len(odd_nodes)}")

        for node in single_qubit_nodes:
            # print(f"DEBUG: Applying single qubit gate: {node.op.name} on qubit {node.qargs[0]._index}")
            apply_single_qubit_gate(state, node)
            # print(f"DEBUG: State norm after single qubit gate: {state.norm()}")
            dag.remove_op_node(node)

        # Process two-qubit gates in even/odd sweeps.
        for group_name, group in [("even", even_nodes), ("odd", odd_nodes)]:
            for node in group:
                # print(f"\nDEBUG: Processing {group_name} two-qubit gate: {node.op.name}")
                # print(f"DEBUG: Gate acts on qubits: {[q._index for q in node.qargs]}")
                # print(f"DEBUG: State norm before gate: {state.norm()}")
                print("-"*80)
                print(f"DEBUG: Applying {group_name} two-qubit gate: {node.op.name}")
                print("-"*80)
                print(f"DEBUG: state canonical form before noiseless two-qubit gate: {state.check_canonical_form()}")
                first_site, last_site = apply_two_qubit_gate(state, node, sim_params)
                print(f"DEBUG: state canonical form after apply_two_qubit_gate: {state.check_canonical_form()}")
                print(f"DEBUG: first_site: {first_site}, last_site: {last_site}")
                
                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    # Normalizes state
                    state.normalize(form="B", decomposition="QR")
                else:
                    state.normalize(form="B", decomposition="QR")
                    #  print(f"DEBUG: Applying noisy two-qubit gate")
                    print(f"DEBUG: noise model: {noise_model}")
                    print(f"DEBUG: noise model processes: {noise_model.processes}")
                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    for process in local_noise_model.processes:
                        if process["sites"] == [first_site, last_site]:
                            process["sites"] = [0, 1]
                        elif process["sites"] == [first_site]:
                            process["sites"] = [0]
                        elif process["sites"] == [last_site]:
                            process["sites"] = [1]
                    print(f"DEBUG: local noise model: {local_noise_model}")
                    print(f"DEBUG: local noise model processes: {local_noise_model.processes}")
                    affected_state, _, window = apply_window(state, None, first_site, last_site, window_size=0)
                    print(f"DEBUG: affected state canonical form after apply_window: {affected_state.check_canonical_form()}")
                    print(f"DEBUG: full state canonical form after apply_window: {state.check_canonical_form()}")
                    print("-"*80)
                    print(f"DEBUG: Applying dissipation")
                    print("-"*80)
                    apply_dissipation(affected_state, local_noise_model, dt=1, sim_params=sim_params)
                    print(f"DEBUG: affected state canonical form after dissipation: {affected_state.check_canonical_form()}")
                    print(f"DEBUG: full state canonical form after dissipation: {state.check_canonical_form()}")
                    print("-"*80)
                    print(f"DEBUG: Applying stochastic process")
                    print("-"*80)
                    affected_state = stochastic_process(affected_state, local_noise_model, dt=1, sim_params=sim_params)
                    print(f"DEBUG: affected state canonical form after stochastic process: {affected_state.check_canonical_form()}")
                    print(f"DEBUG: full state canonical form after stochastic process: {state.check_canonical_form()}")
                    print("-"*80)
                    print(f"DEBUG: Replacing tensors in window")
                    print("-"*80)
                    for i in range(window[0], window[1] + 1):
                        state.tensors[i] = affected_state.tensors[i - window[0]]
                    state.normalize(form="B", decomposition="QR")
                    print(f"DEBUG: State canonical form after normalization: {state.check_canonical_form()}")
                    print(f"DEBUG: State norm after gate+dissipation+jump: {state.norm(0)}")

                dag.remove_op_node(node)

    # print(f"\nDEBUG: Circuit execution completed. Final state norm: {state.norm()}")

    if isinstance(sim_params, WeakSimParams):
        if not noise_model or all(proc["strength"] == 0 for proc in noise_model.processes):
            # All shots can be done at once in noise-free model
            if sim_params.get_state:
                sim_params.output_state = state
            return state.measure_shots(sim_params.shots)
        # Each shot is an individual trajectory
        return state.measure_shots(shots=1)
    
    # StrongSimParams
    # print(f"\nDEBUG: Computing observables for {len(sim_params.observables)} observables")
    results = np.zeros((len(sim_params.observables), 1))
    temp_state = copy.deepcopy(state)
    if sim_params.get_state:
        sim_params.output_state = state

    last_site = 0
    for obs_index, observable in enumerate(sim_params.sorted_observables):
        if isinstance(observable.sites, list):
            idx = observable.sites[0]
        elif isinstance(observable.sites, int):
            idx = observable.sites
        
        # print(f"DEBUG: Computing observable {obs_index} at site {idx}")
        # print(f"DEBUG: Observable gate: {observable.gate.name}")
        
        if idx > last_site:
            # print(f"DEBUG: Shifting orthogonality center from {last_site} to {idx}")
            for site in range(last_site, idx):
                temp_state.shift_orthogonality_center_right(site)
            last_site = idx
        
        expectation = temp_state.expect(observable)
        results[obs_index, 0] = expectation
        # print(f"DEBUG: Expectation value for observable {obs_index}: {expectation}")
    
    # print(f"DEBUG: Final results: {results.flatten()}")
    # print(f"{'#'*80}\n")
    return results
