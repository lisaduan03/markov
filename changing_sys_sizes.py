from three_sys_five_env import Markov_three_sys_five_env
from four_sys_five_env import Markov_four_sys_five_env
from five_sys_five_env import Markov_five_sys_five_env
from six_sys_five_env import Markov_six_sys_five_env
from seven_sys_five_env import Markov_seven_sys_five_env
from five_state_env import Environment_five
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt 
from generating_matrices import generate_random_matrix

"""
changing the size of the system, but keeping environment constant . 
1) randomly generated (little/no coupling)
2) strong coupling by hand-- verify transition matrices
"""

def growth_func(x, m, a):
	return m * (1-(np.power(np.e, (-a * x))))

def stored_info_vs_mi_no_coupling_change_system_size(number: int, env_mat: np.array):
    env = Environment_five(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    plt.ylim(0, 1.8)
    plt.xlabel("time step")
    plt.ylabel("information (nats)")
    plt.title("max predictive power and SI in 5 state environment with litte/no coupling")
    trajectories_three = [Markov_three_sys_five_env(env_mat, generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3)) for i in range(number)]
    for traj in trajectories_three:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'red')
    trajectories_four = [Markov_four_sys_five_env(env_mat, generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4)) for i in range(number)]
    for traj in trajectories_four:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'orange')
    trajectories_five = [Markov_five_sys_five_env(env_mat, generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5)) for i in range(number)]
    for traj in trajectories_five:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'yellow')
    trajectories_six = [Markov_six_sys_five_env(env_mat, generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6)) for i in range(number)]
    for traj in trajectories_six:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'green')
    trajectories_seven = [Markov_seven_sys_five_env(env_mat, generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7)) for i in range(number)]
    for traj in trajectories_seven:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'blue')
    plt.plot(x, y, '--', label = "stored info", color = 'pink')
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

# env_mat = np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]])
# stored_info_vs_mi_no_coupling_change_system_size(1, env_mat)

def stored_info_vs_mi_no_coupling_change_system_size_average(number: int, env_mat: np.array):
    """
    lines of average values 
    """
    env = Environment_five(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    plt.ylim(0, 1.8)
    plt.xlabel("time step")
    plt.ylabel("information (nats)")
    plt.title("avg max predictive power and SI in 5 state environment with litte/no coupling")
    trajectories_three = [Markov_three_sys_five_env(env_mat, generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3)) for i in range(number)]
    for traj in trajectories_three:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.axhline(y=sum(traj.I_pred_one_step()[50:100])/50, color='red', label = "3 state system")
    trajectories_four = [Markov_four_sys_five_env(env_mat, generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4)) for i in range(number)]
    for traj in trajectories_four:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.axhline(y=sum(traj.I_pred_one_step()[50:100])/50, color='orange', label = "4 state system")
    trajectories_five = [Markov_five_sys_five_env(env_mat, generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5)) for i in range(number)]
    for traj in trajectories_five:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.axhline(y=sum(traj.I_pred_one_step()[50:100])/50, color='yellow', label = "5 state system")
    trajectories_six = [Markov_six_sys_five_env(env_mat, generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6)) for i in range(number)]
    for traj in trajectories_six:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.axhline(y=sum(traj.I_pred_one_step()[50:100])/50, color='green', label = "6 state system")
    trajectories_seven = [Markov_seven_sys_five_env(env_mat, generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7)) for i in range(number)]
    for traj in trajectories_seven:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.axhline(y=sum(traj.I_pred_one_step()[50:100])/50, color='blue', label = "7 state system")
    plt.plot(x, y, '--', label = "stored info", color = 'pink')
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

# env_mat = np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]])
# stored_info_vs_mi_no_coupling_change_system_size_average(1, env_mat)


def stored_info_vs_mi_with_coupling_change_system_size(number: int, env_mat: np.array):
    env = Environment_five(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    plt.ylim(0, 1.8)
    plt.xlabel("time step")
    plt.ylabel("information (nats)")
    plt.title("max predictive power and SI in 5 state environment with strong coupling")
    trajectories_three = [Markov_three_sys_five_env(env_mat, np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
        np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
            np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
    for traj in trajectories_three:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'red')
    trajectories_four = [Markov_four_sys_five_env(env_mat, np.array([[.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04]]), np.array([[.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04]]), \
        np.array([[.03, .9, .03, .04], [.03, .9, .03, .04], [.03, .9, .03, .04], [.03, .9, .03, .04]]), np.array([[.03, .03, .9, .04], [.03, .03, .9, .04], [.03, .03, .9, .04], [.03, .03, .9, .04]]), \
            np.array([[.03, .03, .04, .9], [.03, .03, .04, .9], [.03, .03, .04, .9], [.03, .03, .04, .9]])) for i in range(number)]
    for traj in trajectories_four:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'orange')
    trajectories_five = [Markov_five_sys_five_env(env_mat, np.array([[.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025]]), \
        np.array([[.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025]]), \
            np.array([[.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025]]), \
                    np.array([[.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025]]), \
                        np.array([[.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9]])) for i in range(number)]
    for traj in trajectories_five:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'yellow')
    trajectories_six = [Markov_six_sys_five_env(env_mat, np.array([[.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025]]), \
        np.array([[.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02]]), \
            np.array([[.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02]]), \
                np.array([[.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02]]), \
                    np.array([[.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9]])) for i in range(number)]
    for traj in trajectories_six:
        traj.generate_ensemble(200)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'green')
    trajectories_seven = [Markov_seven_sys_five_env(env_mat, np.array([[.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02]]), \
        np.array([[.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02]]), \
            np.array([[.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02]]), \
                np.array([[.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02]]), \
                    np.array([[.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9]])) for i in range(number)]
    for traj in trajectories_seven:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step(), color = 'blue')
    plt.plot(x, y, '--', label = "stored info", color = 'pink')
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

def stored_info_vs_mi_with_coupling_change_system_size_average(number: int, env_mat: np.array):
    I_pred_max_one_step_three_sys = 0
    I_pred_max_one_step_four_sys = 0
    I_pred_max_one_step_five_sys = 0
    I_pred_max_one_step_six_sys = 0
    I_pred_max_one_step_seven_sys = 0
    env = Environment_five(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    plt.axhline(y=stored_info, color='pink', linestyle='--', label = "stored info")
    plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    plt.ylim(0, 1.8)
    plt.ylabel("information (nats)")
    plt.title("avg max predictive power and SI in 5 state environment with strong coupling")
    trajectories_three = [Markov_three_sys_five_env(env_mat, np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
        np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
            np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
    for traj in trajectories_three:
        traj.generate_ensemble(10000)
        traj.calculate_probs()
        I_pred_max_one_step_three_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])
    trajectories_four = [Markov_four_sys_five_env(env_mat, np.array([[.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04]]), np.array([[.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04], [.9, .03, .03, .04]]), \
        np.array([[.03, .9, .03, .04], [.03, .9, .03, .04], [.03, .9, .03, .04], [.03, .9, .03, .04]]), np.array([[.03, .03, .9, .04], [.03, .03, .9, .04], [.03, .03, .9, .04], [.03, .03, .9, .04]]), \
            np.array([[.03, .03, .04, .9], [.03, .03, .04, .9], [.03, .03, .04, .9], [.03, .03, .04, .9]])) for i in range(number)]
    for traj in trajectories_four:
        traj.generate_ensemble(10000)
        traj.calculate_probs()
        I_pred_max_one_step_four_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])
    trajectories_five = [Markov_five_sys_five_env(env_mat, np.array([[.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025]]), \
        np.array([[.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025]]), \
            np.array([[.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025]]), \
                    np.array([[.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025]]), \
                        np.array([[.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9]])) for i in range(number)]
    for traj in trajectories_five:
        traj.generate_ensemble(10000)
        traj.calculate_probs()
        I_pred_max_one_step_five_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])
    trajectories_six = [Markov_six_sys_five_env(env_mat, np.array([[.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025], [.45, .45, .025, .025, .025, .025]]), \
        np.array([[.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02], [.02, .02, .9, .02, .02, .02]]), \
            np.array([[.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02], [.02, .02, .02, .9, .02, .02]]), \
                np.array([[.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02], [.02, .02, .02, .02, .9, .02]]), \
                    np.array([[.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9], [.02, .02, .02, .02, .02, .9]])) for i in range(number)]
    for traj in trajectories_six:
        traj.generate_ensemble(10000)
        traj.calculate_probs()
        I_pred_max_one_step_six_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])
    trajectories_seven = [Markov_seven_sys_five_env(env_mat, np.array([[.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02]]), \
        np.array([[.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02]]), \
            np.array([[.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02]]), \
                np.array([[.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02]]), \
                    np.array([[.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9]])) for i in range(number)]
    for traj in trajectories_seven:
        traj.generate_ensemble(10000)
        traj.calculate_probs()
        I_pred_max_one_step_seven_sys = sum(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])/len(np.array(traj.I_pred_one_step()[50:])[np.isfinite(traj.I_pred_one_step()[50:])])
    plt.axhline(I_pred_max_one_step_three_sys, color = "red", label = "3 state sys")
    plt.axhline(I_pred_max_one_step_four_sys, color = "orange", label = "4 state sys")
    plt.axhline(I_pred_max_one_step_five_sys, color = "yellow", label = "5 state sys")
    plt.axhline(I_pred_max_one_step_six_sys, color = "green", label = "6 state sys")
    plt.axhline(I_pred_max_one_step_seven_sys, color = "blue", label = "7 state sys")
    legend = plt.legend(loc='upper right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()
    fig, ax = plt.subplots()
    table_data=[
    ["stored information", stored_info],
    ["I(s+t,x+t+1) three states", I_pred_max_one_step_three_sys],
    ["I(s+t,x+t+1) four states", I_pred_max_one_step_four_sys],
    ["I(s+t,x+t+1) five states", I_pred_max_one_step_five_sys],
    ["I(s+t,x+t+1) six states", I_pred_max_one_step_six_sys],
    ["I(s+t,x+t+1) seven states", I_pred_max_one_step_seven_sys]]
    table = ax.table(cellText=table_data, loc='center')
    #modify table
    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')
    print("I(s_t, x_t+1) 3 state sys:")
    print(I_pred_max_one_step_three_sys)
    print("I(s_t, x_t+1) 4 state sys:")
    print(I_pred_max_one_step_four_sys)
    print("I(s_t, x_t+1) 5 state sys:")
    print(I_pred_max_one_step_five_sys)
    print("I(s_t, x_t+1) 6 state sys:")
    print(I_pred_max_one_step_six_sys)
    print("I(s_t, x_t+1) 7 state sys:")
    print(I_pred_max_one_step_seven_sys)

# env_mat = np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]])
# stored_info_vs_mi_with_coupling_change_system_size_average(100, env_mat)


# env_mat = np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]])
# stored_info_vs_mi_with_coupling_change_system_size(10, env_mat)

# env_mat_strong_cycle = np.array([[.00025, .999, .00025, .00025, .00025], [.00025, .00025, .999, .00025, .00025], [.00025, .00025, .00025, .999, .00025], [.00025, .00025, .00025, .00025, .999], [.999, .00025, .00025, .00025, .00025]])
# stored_info_vs_mi_with_coupling_change_system_size(10, env_mat_strong_cycle)

# env_mat_db_with_si = np.array([[.025, .9, .025, .025, .025], [.9, .025, .025, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .9, .025, .025], [.1, .1, .1, .1, .6]])
# stored_info_vs_mi_with_coupling_change_system_size_average(100, env_mat_db_with_si)

env_mat_db_no_si = np.array([[1/5, 1/5, 1/5, 1/5, 1/5], [1/5, 1/5, 1/5, 1/5, 1/5], [1/5, 1/5, 1/5, 1/5, 1/5], [1/5, 1/5, 1/5, 1/5, 1/5], [1/5, 1/5, 1/5, 1/5, 1/5]])
stored_info_vs_mi_with_coupling_change_system_size_average(5, env_mat_db_no_si)



def pred_si_changing_sys(number: int):
    """
	plotting maximum predictive power one step ahead I(st, xt+1) in randomly generated 5 state environments,
	where size of system ranges from 3-7
	randomly generated systems so little/no coupling
    scatterplot with slope
	"""
    list_I_pred_max_one_step_three_sys = []
    list_I_pred_max_one_step_four_sys = []
    list_I_pred_max_one_step_five_sys = []
    list_I_pred_max_one_step_six_sys = []
    list_I_pred_max_one_step_seven_sys = []
    list_stored_info_three_sys = []
    list_stored_info_four_sys = []
    list_stored_info_five_sys = []
    list_stored_info_six_sys = []
    list_stored_info_seven_sys = []
    x = [x for x in range(99)]
    trajectories_three = [Markov_three_sys_five_env(generate_random_matrix(5), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3)) for i in range(number)]
    for traj in trajectories_three:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        list_stored_info_three_sys.append(Environment_five(traj.p_tenv).stored_information())
        list_I_pred_max_one_step_three_sys.append(sum(traj.I_pred_one_step()[50:])/50)
    trajectories_four = [Markov_four_sys_five_env(generate_random_matrix(5), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4), generate_random_matrix(4)) for i in range(number)]
    for traj in trajectories_four:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        list_stored_info_four_sys.append(Environment_five(traj.p_tenv).stored_information())
        list_I_pred_max_one_step_four_sys.append(sum(traj.I_pred_one_step()[50:])/50)
    trajectories_five = [Markov_five_sys_five_env(generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5)) for i in range(number)]
    for traj in trajectories_five:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        list_stored_info_five_sys.append(Environment_five(traj.p_tenv).stored_information())
        list_I_pred_max_one_step_five_sys.append(sum(traj.I_pred_one_step()[50:])/50)
    trajectories_six = [Markov_six_sys_five_env(generate_random_matrix(5), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6), generate_random_matrix(6)) for i in range(number)]
    for traj in trajectories_six:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        list_stored_info_six_sys.append(Environment_five(traj.p_tenv).stored_information())
        list_I_pred_max_one_step_six_sys.append(sum(traj.I_pred_one_step()[50:])/50)
    trajectories_seven = [Markov_seven_sys_five_env(generate_random_matrix(5), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7)) for i in range(number)]
    for traj in trajectories_seven:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        list_stored_info_seven_sys.append(Environment_five(traj.p_tenv).stored_information())
        list_I_pred_max_one_step_seven_sys.append(sum(traj.I_pred_one_step()[50:])/50)
    plt.figure()
    plt.title("max predictive power and SI with systems of different sizes")
    plt.xlabel("SI")
    plt.ylabel("max I(s_t, x_t+1)")
    plt.axhline(y=np.log(5), color='purple', linestyle='--', label = "ln(5)")
    plt.plot(list_stored_info_three_sys, list_I_pred_max_one_step_three_sys, 'ro', label = '3 state system', color = "red")
    plt.plot(list_stored_info_four_sys, list_I_pred_max_one_step_four_sys, 'ro', label = '4 state system', color = "orange")
    plt.plot(list_stored_info_five_sys, list_I_pred_max_one_step_five_sys, 'ro', label = '5 state system', color = "yellow")
    plt.plot(list_stored_info_six_sys, list_I_pred_max_one_step_six_sys, 'ro', label = '6 state system', color = "green")
    plt.plot(list_stored_info_seven_sys, list_I_pred_max_one_step_seven_sys, 'ro', label = '7 state system', color = "blue")
    plt.legend()
    plt.show()

#pred_si_changing_sys(100)





    





