import numpy as np


from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    Observable,
    PhysicsSimParams,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import XX, YY, ZZ, X, Z

if __name__ == "__main__":

    length = 10
    initial_state = MPS(length, state="zeros")

    H = MPO()
    H.init_ising(length, J=1, g=0.5)
    elapsed_time = 1
    dt = 0.1
    sample_timesteps = False
    num_traj = 100
    max_bond_dim = 4
    threshold = 0
    order = 2

    measurements = [Observable(Z(), site) for site in range(length)]
    sim_params = PhysicsSimParams(
        measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps
    )
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(length) for name in ["relaxation", "dephasing"]
    ])


    simulator.run(initial_state, H, sim_params, noise_model)

