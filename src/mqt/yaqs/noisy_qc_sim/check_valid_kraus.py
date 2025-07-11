import numpy as np 
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

def check_valid_kraus(noise_model: NoiseModel):
    """
    Check if the Kraus operators are valid.
    """
    strength_list = []
    for process in noise_model.processes:
        strength_list.append(process["strength"])
    p = 0
    for strength in strength_list:
        p+=strength
    print(p)
    return np.allclose(p, 1)




if __name__ == "__main__":
    noise_strength = 0.1
    processes = [{
            "name": "xx",
            "sites": [0, 1],
            "strength": noise_strength**2*(1-noise_strength)
        },
        {
            "name": "xx",
            "sites": [1, 2],
            "strength": noise_strength**2*(1-noise_strength)
        },
        {
            "name": "x",
            "sites": [0],
            "strength": (1-noise_strength)**2*noise_strength
        }, 
        {
            "name": "x",
            "sites": [1],
            "strength": (1-noise_strength)**2*noise_strength
        }, 
        {
            "name": "x",
            "sites": [2],
            "strength": (1-noise_strength)**2*noise_strength
        }
        ]
    noise_model = NoiseModel(processes)
    print(check_valid_kraus(noise_model))
