from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix
import sys
"""
8/8. environment size is 3, system size is 3. strong coupling 
"""


def func(number_of_sims: int, env_size: int, file_name: str):
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: ' + str(env_size)]
        header_2 = ["theoretical SI", "SI using MI formula", "SI using simluated steady state time slice", "3 state system"]
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        row = []
        s = 0
        while s < number_of_sims:
            row = []
            trajectory = Markov_flex_sys_flex_env(np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]), \
                np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
                    np.array([[.05, .9, .05], [.05, .9, .05], [.05, .09, .05]]), \
                        p_t3 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))
            row.append(trajectory.stored_information())
            trajectory.generate_ensemble(10000)
            trajectory.calculate_probs()
            row.append(trajectory.verify_stored_information_mi())
            row.append(trajectory.verify_stored_information_steady_state())
            trajectory.I_pred_one_step()
            row.append(sum(np.array(trajectory.I_pred_one_step()[50:])[np.isfinite(trajectory.I_pred_one_step()[50:])])/len(np.array(trajectory.I_pred_one_step()[50:])[np.isfinite(trajectory.I_pred_one_step()[50:])]))
            s = s + 1
            writer.writerow(row) # indented this once

file_name = "8_8_3_state_zero_si_strong_coupling_" + sys.argv[1] + ".csv"

func(10000, 3, file_name)