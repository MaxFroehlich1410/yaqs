from ..worker_functions.yaqs_simulator import run_yaqs, build_noise_models

# 
# Initialize noise models (YAQS)

num_qubits = 2
noise_strength = 0.1
processes = [
    {"name": "pauli_x", "sites": [i], "strength": noise_strength}
    for i in range(num_qubits)
    ] + [
    {"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
    for i in range(num_qubits - 1)
] 
noise_model_normal, noise_model_projector, noise_model_unitary_2pt, noise_model_unitary_gauss = build_noise_models(processes)

for process in noise_model_projector.processes:
    print(process)