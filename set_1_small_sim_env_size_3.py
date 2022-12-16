from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix

def small_sim_env_size_3(number_of_sims: int, file_name: str):
    """
    for set 1: 
        Broken DB
        weak coupling
        non 0 stored info 
    """
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: ' + str(3)]
        header_2 = ["system states: " '3', '4', '5'] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        trajectories = []
        randomly_generated_env_size_x = [generate_random_matrix(3) for i in range(10)]
        for s in range(10):
            writer.writerow(["example: " + str(s)])
            for i in range(3, 6):
                writer.writerow(["states in system: " + str(i)])
                trajectories = [Markov_flex_sys_flex_env(randomly_generated_env_size_x[s], generate_random_matrix(i), generate_random_matrix(i), p_t3= generate_random_matrix(i)) for x in range(number_of_sims)]
                I_pred_max_one_list = []
                stored_info_list = []
                for traj in trajectories:
                    traj.generate_ensemble(2000)
                    traj.calculate_probs()
                    traj.I_pred_one_step()
                    I_pred_max_one_list.append(sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])]))
                    stored_info_list.append(traj.stored_information())
                writer.writerow(I_pred_max_one_list)
                writer.writerow(stored_info_list)

small_sim_env_size_3(1000, "small_sim_env_size_3.csv")
print("hi")