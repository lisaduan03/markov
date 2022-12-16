from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix


def set_3_env_10(number_of_sims: int, file_name: str):
    """
    10 state environment
    for set 3: 
        SI = 0
        DB (only one steady state matrix we know so far) 
        weak coupling
    """
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: 10']
        header_2 = ["system states: " '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        trajectories = []
        for i in range(2, 11):
            writer.writerow(["states in system: " + str(i)])
            trajectories = [Markov_flex_sys_flex_env(np.array([[.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                     [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                        [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                                [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                                    [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                                        [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                                            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], \
                                                [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]]), generate_random_matrix(i), generate_random_matrix(i), p_t3= generate_random_matrix(i), p_t4= generate_random_matrix(i), p_t5= generate_random_matrix(i), p_t6 = generate_random_matrix(i), p_t7 = generate_random_matrix(i), p_t8 =generate_random_matrix(i), p_t9 =generate_random_matrix(i), p_t99 = generate_random_matrix(i)) for x in range(number_of_sims)]
            I_pred_max_one_list = []
            stored_info_list = []
            for traj in trajectories:
                traj.generate_ensemble(1000)
                traj.calculate_probs()
                traj.I_pred_one_step()
                I_pred_max_one_list.append(sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])]))
                stored_info_list.append(traj.stored_information())
            writer.writerow(I_pred_max_one_list)
            writer.writerow(stored_info_list) # this position may be incorrect



set_3_env_10(1000, "set_3_env_size_10.csv")
print("hi")