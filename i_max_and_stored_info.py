
"""simulations to show (maybe) that I(s_t, x_t+1) <= I(x_t, x_t+1)

"""

from generating_matrices import generate_coupled_cw, generate_random_matrix
from three_state_env import Environment
from markov_three import Markov_three
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
from fitting_curves import growth_func
import numpy as np
from seven_sys_five_env import Markov_seven_sys_five_env

"""
no coupling, takes in env so DB can be modified 
"""
def stored_info_vs_mi_no_coupling(number: int, env_mat: np.array):
    env = Environment(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.plot(x, y, '--', label = "stored info")
    plt.ylim(0, 1.5)
    plt.xlabel("time step")
    plt.ylabel("information (nats)")
    plt.title("mutual information and stored information between given matrix with litte/no coupling")
    trajectories = [Markov_three(env_mat, generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3)) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step())
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

def stored_info_vs_mi_moderate_coupling(number: int, env_mat: np.array):
    env = Environment(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.plot(x, y, '--', label = "stored info")
    plt.xlabel("time step")
    plt.ylim(0, 1.5)
    plt.ylabel("information (nats)")
    plt.title("mutual information and stored information between given matrix with moderate coupling")
    trajectories = [Markov_three(env_mat, generate_coupled_cw(1), generate_coupled_cw(2), generate_coupled_cw(3)) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step())
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

def stored_info_vs_mi_strong_coupling(number: int, env_mat: np.array):
    env = Environment(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.plot(x, y, '--', label = "stored info")
    plt.xlabel("time step")
    plt.ylim(0, 1.5)
    plt.ylabel("information (nats)")
    plt.title("I(s_t, x_t+1) and I(x_t, x_t+1) in given matrix with strong coupling")
    trajectories = [Markov_three(env_mat, p_t1 = np.array([[.05, .9, .05],
         [.05, .9, .05],
         [.05, .9, .05]]), p_t2 = np.array([[.05, .05, .9],
         [.05, .05, .9],
         [.05, .05, 9]]), p_t3 = np.array([[.9, .05, .05],
         [.9, .05, .05],
         [.9, .05, .05]])) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(300)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step())
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

# env_mat = np.array([[1/3, 1/3, 1/3],
#  [1/3, 1/3, 1/3],
#  [1/3,  1/3, 1/3]])
# print(env_mat)
# # stored_info_vs_mi_no_coupling(10, env_mat)
# # stored_info_vs_mi_moderate_coupling(10, env_mat)
# stored_info_vs_mi_strong_coupling(10, env_mat)

def test_seven_system_coupling(number: int, env_mat: np.array):
    env = Environment(env_mat)
    stored_info = env.stored_information()
    plt.figure()
    x = [x for x in range(100)]
    y = [stored_info for y in range(100)]
    plt.plot(x, y, '--', label = "stored info")
    plt.xlabel("time step")
    plt.ylim(0, 1.5)
    plt.ylabel("information (nats)")
    plt.title("I(s_t, x_t+1) and I(x_t, x_t+1) in given matrix with strong coupling")
    trajectories = [Markov_seven_sys_five_env(env_mat, np.array([[.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02]]), \
        np.array([[.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02]]), \
            np.array([[.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02]]), \
                np.array([[.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02]]), \
                    np.array([[.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9]])) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(1000)
        traj.calculate_probs()
        plt.plot(traj.I_pred_one_step())
    legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
    legend.get_frame().set_facecolor('w')
    plt.show()

env_mat = np.array([[1/5, 1/5, 1/5, 1/5, 1/5], 
[1/5, 1/5, 1/5, 1/5, 1/5], 
[1/5, 1/5, 1/5, 1/5, 1/5], 
[1/5, 1/5, 1/5, 1/5, 1/5], 
[1/5, 1/5, 1/5, 1/5, 1/5]])
test_seven_system_coupling(2, env_mat)


# stored_info_random = random_env.stored_information()
# random_Markov_three = Markov_three(random_env, generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))
# random_Markov_three.calculate_probs()
# mi_random = random_Markov_three.mutual_information()


# def stored_info_vs_mi_scatter(number: int):
# 	global list_I_max
# 	list_I_max = []
# 	list_entropy = []
# 	x = [x for x in range(100)]
# 	trajectories = [Markov_three(generate_random_matrix()
# , p_t1 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
# 		 p_t2 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, 9]]), \
# 			 p_t3 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]])) for i in range(number)]
# 	for traj in trajectories:
# 		traj.generate_ensemble(237)
# 		traj.calculate_probs()
# 		popt, pcov = curve_fit(growth_func, x, traj.mutual_information())
# 		list_I_max.append(popt[0])
# 		list_entropy.append(traj.entropy(traj.env_probabilities))
# 		list_cw_to_ccw_ratio.append(cw_to_ccw_ratio(traj.p_tenv))
# 		list_slope.append(popt[1])
# 		list_entropy_production_rate.append(entropy_rate_env(traj.p_tenv))

# plt.figure()
# plt.xlabel("cw to ccw ratio")
# plt.ylabel("I_max")
# plt.title("I_max vs cw to cc ratio")
# plt.plot(list_cw_to_ccw_ratio, list_I_max, 'ro')
# plt.show()
