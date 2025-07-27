'''Check for noise model with 3 processes on 2 sites if norm after dissipation is same 
if applied to 2 sites or 4 sites'''


import numpy as np
from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from ..core.methods.dissipation import apply_dissipation, apply_circuit_dissipation



# # yaqs simulation
# processes = []
# processes.append({
#         "name": "xx",
#         "sites": [1, 2],
#         "strength": 0.01
#     })
# processes.append({
#         "name": "x",
#         "sites": [1],
#         "strength": 0.01
#     })
# processes.append({
#         "name": "x",
#         "sites": [2],
#         "strength": 0.01
#     })
# noise_model_yaqs = NoiseModel(processes)

# state = MPS(length=4, state="random")

# print("state canonical form:", state.check_canonical_form())
# print("state norm:", state.norm())
# state.shift_orthogonality_center_right(0)
# # add ortho shift before slice
# print("state canonical form after shift:", state.check_canonical_form())
# short_state = MPS(length=2, tensors=state.tensors[1:3])
# print("short state canonical form:", short_state.check_canonical_form())
# print("short state norm:", short_state.norm())

# measurements = [Observable(Z(), site) for site in range(1,3)]

# simparams = StrongSimParams(measurements)
# # # add placeholder simparams
# apply_circuit_dissipation(short_state, noise_model_yaqs, dt=1, global_start=1, sim_params=simparams)
# apply_circuit_dissipation(state, noise_model_yaqs, dt=1, global_start=1, sim_params=simparams)
# print("state canonical form:", state.check_canonical_form())
# print("short state canonical form:", short_state.check_canonical_form())
# print("state norm after dissipation:", state.norm())
# print("short state norm after dissipation:", short_state.norm())


state = MPS(length=4, state="random")
first_site = 1
last_site = 2

print("state canonical form:", state.check_canonical_form())
print("state norm:", state.norm())

noise_model = NoiseModel([
    {"name": "x", "sites": [1], "strength": 0.01},
    {"name": "x", "sites": [2], "strength": 0.01}
])
apply_dissipation(state, noise_model, dt=1, sim_params=StrongSimParams(observables=[Observable(Z(), site) for site in range(1,3)]))
state.shift_orthogonality_center_right(0)
state.shift_orthogonality_center_right(1)

print("state canonical form after shift:", state.check_canonical_form())

state.set_canonical_form(first_site)
# for i in reversed(range(first_site,current_ortho_center[0] + 1)):
#     state.shift_orthogonality_center_left(i)

print("state canonical form after shift:", state.check_canonical_form())

state.normalize("B", "QR")
print("state norm after normalize:", state.norm())
print("state canonical form after normalize:", state.check_canonical_form())