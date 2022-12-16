from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix
import sys
"""
IMPORTANT: environment size is 2. 10:30 am- just practice right now with 2 sims,
 once it works run with 1000 
"""


def func(number_of_sims: int, env_size: int, file_name: str):
    """
    for set 1: 
        Broken DB
        weak coupling
        non 0 stored info 
    """
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: ' + str(env_size)]
        header_2 = ["example #", "theoretical SI", "SI using MI formula", "SI using simluated steady state time slice", "2", "3" "4", "5", "6", "7", "8", "9", "10"] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        row = []
        randomly_generated_env_size_x = [generate_random_matrix(env_size) for i in range(10)]
        for i in range(1):            
            s = 0
            while s < number_of_sims:
                row = []
                row.append("example: " + str(i + 1))
                trajectory = Markov_flex_sys_flex_env(randomly_generated_env_size_x[i], generate_random_matrix(2), generate_random_matrix(2)) # abritrary system states here 
                row.append(trajectory.stored_information())
                trajectory.generate_ensemble(100)
                trajectory.calculate_probs()
                row.append(trajectory.verify_stored_information_mi())
                row.append(trajectory.verify_stored_information_steady_state())
                for sys_size in range (2, 11):
                    trajectory = Markov_flex_sys_flex_env(randomly_generated_env_size_x[i], generate_random_matrix(sys_size), generate_random_matrix(sys_size))
                    trajectory.generate_ensemble(100)
                    trajectory.calculate_probs()
                    trajectory.I_pred_one_step()
                    row.append(sum(np.array(trajectory.I_pred_one_step()[50:])[np.isfinite(trajectory.I_pred_one_step()[50:])])/len(np.array(trajectory.I_pred_one_step()[50:])[np.isfinite(trajectory.I_pred_one_step()[50:])]))
                s = s + 1
                writer.writerow(row) # indented this once

file_name = "8_8_2_state_nonzero_si_strong_coupling.csv"
func(100, 2, file_name)