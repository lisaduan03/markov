from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix

def practicefunc(env_size, sim_num):
    """
    for set 1: 
        Broken DB
        weak coupling
        non 0 stored info 
    """
    with open("tryingagain.csv", "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environmesadfafsdafdsnt: ' + str(env_size)]
        header_2 = ["system states: " '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")
    
        trajectories = []
        randomly_generated_env_size_x = [x for x in range(10)]
        for s in range(10):
            writer.writerow(["examplfjdaslkfj;lsdkajflasdfe: " + str(s)])
            for i in range(2, 11):
                writer.writerow(["states in syafsdasdfafsddsfstem: " + str(i)])

practicefunc(2, 2)
print("help")
