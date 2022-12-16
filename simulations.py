from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix

def set_1(number_of_sims: int, env_size: int, file_name: str):
    """
    for set 1: 
        Broken DB
        weak coupling
        non 0 stored info 
    """
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: ' + str(env_size)]
        header_2 = ["system states: " '2', '3', '4', '5', '6', '7', '8', '9', '10'] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        trajectories = []
        randomly_generated_env_size_x = [generate_random_matrix(env_size) for i in range(10)]
        for s in range(10):
            writer.writerow(["example: " + str(s)])
            for i in range(2, 11):
                writer.writerow(["states in system: " + str(i)])
                trajectories = [Markov_flex_sys_flex_env(randomly_generated_env_size_x[s], generate_random_matrix(i), generate_random_matrix(i)) for x in range(number_of_sims)]
                I_pred_max_one_list = []
                for traj in trajectories:
                    traj.generate_ensemble(1000)
                    traj.calculate_probs()
                    traj.I_pred_one_step()
                    I_pred_max_one_list.append(sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])]))
                writer.writerow(I_pred_max_one_list)

set_1(1000, 2, "1000sims.csv")

            


# def practice(name):
#     header = ['name', 'area', 'country_code2', 'country_code3']
#     data = ['Afghanistan', 652090, 'AF', 'AFG']

#     with open(name, 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)

#         # write the header
#         writer.writerow(header)

#         # write the data
#         writer.writerow(data)


# practice("/home/lic776/Desktop/markov/set_1_two_state_env.csv")














    # stored_info = env.stored_information()
    # plt.figure()
    # plt.axhline(y=stored_info, color='pink', linestyle='--', label = "stored info")
    # plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    # plt.ylim(0, 1.8)
    # plt.ylabel("information (nats)")
    # plt.title("max predictive power and SI")
    # trajectories = [Markov]
    # traj.generate_ensemble(10000)
    # traj.calculate_probs()
    #     I_pred_max_one_step_seven_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])

    # legend =  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # legend.get_frame().set_facecolor('w')
    # plt.show()
    # fig, ax = plt.subplots()
    # table_data=[
    # ["stored information", stored_info],
    # ["I(s+t,x+t+1) three states", I_pred_max_one_step_three_sys],
    # ["I(s+t,x+t+1) four states", I_pred_max_one_step_four_sys],
    # ["I(s+t,x+t+1) five states", I_pred_max_one_step_five_sys],
    # ["I(s+t,x+t+1) six states", I_pred_max_one_step_six_sys],
    # ["I(s+t,x+t+1) seven states", I_pred_max_one_step_seven_sys]]
    # table = ax.table(cellText=table_data, loc='center')
    # #modify table
    # table.set_fontsize(14)
    # table.scale(1,4)
    # ax.axis('off')


        
