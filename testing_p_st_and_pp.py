from flexible_size_systems import Markov_flex_sys_flex_env
import numpy as np
from three_state_simulation import stationary_distribution, build_environment, couple, couple_helper, time_steps, env_plot, sys_plot
from matplotlib import pyplot as plt
from generating_matrices import generate_random_matrix, generate_db_matrices

def plot_sys_states(markov):
    num_states = len(markov.sys_probabilities[0])
    sys_probs_list = [[0 for i in range(time_steps)]  for x in range(num_states)]
    for time in range(time_steps):
        for row in range(num_states):
            sys_probs_list[row][time] = markov.sys_probabilities[time][row]
    plt.figure()
    plt.ylim(0, 1)
    plt.xlabel("time steps")
    plt.ylabel("probability")
    x = [x for x in range(time_steps)]
    for states in sys_probs_list:
        plt.plot(x, states)
    plt.show()

def plot_pred_power(markov):
    plt.figure()
    plt.title("predictive power over time")
    plt.xlabel("time step")
    plt.ylabel("pp (nats)")
    plt.ylim(0,1)
    plt.plot(markov.I_pred_one_step())
    plt.show()

test = Markov_flex_sys_flex_env(generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), p_t3= generate_random_matrix(3))
test.generate_ensemble(1000)
test.calculate_probs()
plot_pred_power(test)
# test_cycle_strong_coupling = Markov_flex_sys_flex_env(np.array([[.05, .9, .05], [.05, .05, .9], [.9, .05, .05]]), \
#     np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), p_t3= np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))
# test_cycle_strong_coupling.generate_ensemble(1000)
# test_cycle_strong_coupling.calculate_probs()
# plot_sys_states(test_cycle_strong_coupling)
# t_1 = generate_random_matrix(3)
# print(stationary_distribution(t_1))
# t_2 = generate_random_matrix(3)
# print(stationary_distribution(t_2))
# t_3 = generate_random_matrix(3)
# print(stationary_distribution(t_3))
# test_cycle_weak_coupling = Markov_flex_sys_flex_env(t_env, t_1, t_2, p_t3 = t_3)
# test_cycle_weak_coupling.generate_ensemble(1000)
# test_cycle_weak_coupling.calculate_probs()
# plot_sys_states(test_cycle_weak_coupling)
# test_db_strong_coupling = Markov_flex_sys_flex_env(np.array([[.2, .4, .4],
#           [.1, .6, .3],
#           [.2, .6, .2]]), 
#           np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), p_t3= np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))
# test_db_strong_coupling.generate_ensemble(1000)
# test_db_strong_coupling.calculate_probs()
# plot_sys_states(test_db_strong_coupling)
# test_db_weak_coupling = Markov_flex_sys_flex_env(np.array([[.2, .4, .4],
#           [.1, .6, .3],
#           [.2, .6, .2]]),  t_1, t_2, p_t3 = t_3)
# test_db_weak_coupling.generate_ensemble(1000)
# test_db_weak_coupling.calculate_probs()
# plot_sys_states(test_db_weak_coupling)



