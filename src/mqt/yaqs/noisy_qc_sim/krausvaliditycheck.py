import numpy as np

def strength_check_2qubits(p):

    return p**2 + 2*(1-p)*p+ (1-p)**2

# def strength_check_3qubits(p):

#     a = p**3 + 2*
    

strengths = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,100]

# for p in strengths:
#     print(strength_check_2qubits(p))

from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gamma = 0.1

processes = [{
        "name": "xx",
        "sites": [0, 1],
            "strength": gamma**2
    },
    {
        "name": "x",
        "sites": [0],
        "strength": (1-gamma)*gamma
    }, 
    {
        "name": "x",
        "sites": [1],
        "strength": (1-gamma)*gamma
    }
    ]

noise_model = NoiseModel(processes)



# print(noise_model.processes)

local_processes = []

for process in noise_model.processes:
    if process["sites"]==[0]:
        print(process)



