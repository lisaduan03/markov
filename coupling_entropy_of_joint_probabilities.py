"""
7/21. trying to quantify the idea of coupling through calculating entropy in joint probabilities
should work for any size combo of system and environment 
"""

import numpy as np
from three_state_simulation import time_steps, stationary_distribution
from generating_matrices import generate_random_matrix
from flexible_size_systems import Markov_flex_sys_flex_env
import pandas as pd
from matplotlib import pyplot as plt

# def entropy_of_joint_probs(markov):
#     entropy_val = 0
#     entropy_list = []
#     for t in range(time_steps):
#         for row in range(len(markov.joint_probs[t])):
#             for col in range(len(markov.joint_probs[t][0])):
#                 if markov.joint_probs[t][row][col] != 0:
#                     entropy_val = entropy_val + (markov.joint_probs[t][row][col] * np.log(markov.joint_probs[t][row][col]))
#         entropy_list.append(-entropy_val)
#         entropy_val = 0
#     return sum(entropy_list)/len(entropy_list)


def entropy_of_conditional_probs_sx(markov):
    entropy_list = []
    entropy_val = 0
    for t in range(time_steps):
        for row in range(len(markov.joint_probs[t])):
            for col in range(len(markov.joint_probs[t][0])):
                if markov.joint_probs[t][row][col] == 0:
                    markov.joint_probs[t][row][col] = .00001
                if markov.joint_probs[t][row][col] == 0:
                    markov.joint_probs[t][row][col] = .00001
                entropy_val = entropy_val + (markov.joint_probs[t][row][col] *\
                    np.log(markov.env_probabilities[t][col]/markov.joint_probs[t][row][col]))
        entropy_list.append(entropy_val)
        entropy_val = 0
    return sum(entropy_list)/len(entropy_list)


def entropy_env(markov):
    """
    H(X)
    """
    steady_state = stationary_distribution(markov.p_tenv)
    h_x = 0
    for x in steady_state:
        h_x = h_x - (x * np.log(x))
    return h_x

def entropy_sys(markov):
    """
    H(S)
    """
    h = [0] * 20
    for x in range(80,100):
        for j in range(len(markov.sys_probabilities[0])):
            if markov.sys_probabilities[x][j] == 0:
                markov.sys_probabilities[x][j] = .0001
            h[x-80] = h[x-80] - (markov.sys_probabilities[x][j]*np.log(markov.sys_probabilities[x][j])) 
    return sum(h)/20

# def entropy_sys(markov):
#     """
#     H(S)
#     8/5- modified version where I find the steady state first, then find that entropy 
#     gave up, can fix later
#     """
#     for 
#     steady_state_numerical = sum(markov.sys_probabilities[90:])/10
#     h_s = 0
#     for x in steady_state_numerical:
#         h_s = h_s - (x * np.log(x))
#     return h_s


def verify(markov): # already verified, this works
    """
    H(S, X) = H(S|X) + H(X)
    """
    h_xs_cond = entropy_of_conditional_probs_sx(markov)
    h_x = entropy_env(markov)
    h_xs_joint = h_xs_cond + h_x 
    return h_xs_joint

def modified_entropy(markov):
    """ 
    oops, this is wrong.
    H(S|X)/H(X)
    """
    return entropy_of_conditional_probs_sx(markov)/entropy_env(markov)

def modified_entropy_correct(markov):
    """
    H(S|X)/H(S)
    """
    h_xs = entropy_of_conditional_probs_sx(markov)
    h_s = entropy_sys(markov)
    return h_xs/h_s

"""
testing 8/3
"spreading out" probabilities or one-to-one?
testing out strong coupling:

"""
def strongly_coupled_cycle(num_sims):
    """
    sys states = env states
    """
    table = [None] * num_sims
    for i in range(num_sims):
        two = Markov_flex_sys_flex_env(generate_random_matrix(2), np.array([[.9, .1], [.9, .1]]),\
            np.array([[.1, .9], [.1, .9]]))
        three = Markov_flex_sys_flex_env(generate_random_matrix(3), np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
            np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
                p_t3= np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))
        four = Markov_flex_sys_flex_env(generate_random_matrix(4), np.array([[.9, 1/30, 1/30, 1/30], [.9, 1/30, 1/30, 1/30], [.9, 1/30, 1/30, 1/30], [.9, 1/30, 1/30, 1/30]]), \
            np.array([[1/30, .9, 1/30, 1/30], [1/30, .9, 1/30, 1/30], [1/30, .9, 1/30, 1/30], [1/30, .9, 1/30, 1/30]]), \
                p_t3= np.array([[1/30, 1/30, .9, 1/30], [1/30, 1/30, .9, 1/30], [1/30, 1/30, .9, 1/30], [1/30, 1/30, .9, 1/30]]), \
                    p_t4= np.array([[1/30, 1/30, 1/30, .9], [1/30, 1/30, 1/30, .9], [1/30, 1/30, 1/30, .9], [1/30, 1/30, 1/30, .9]]))
        five = Markov_flex_sys_flex_env(generate_random_matrix(5), np.array([[.9, 1/40, 1/40, 1/40, 1/40], [.9, 1/40, 1/40, 1/40, 1/40], [.9, 1/40, 1/40, 1/40, 1/40], [.9, 1/40, 1/40, 1/40, 1/40], [.9, 1/40, 1/40, 1/40, 1/40]]), \
            np.array([[1/40, .9, 1/40, 1/40, 1/40], [1/40, .9, 1/40, 1/40, 1/40], [1/40, .9, 1/40, 1/40, 1/40], [1/40, .9, 1/40, 1/40, 1/40], [1/40, .9, 1/40, 1/40, 1/40]]), \
                p_t3 = np.array([[1/40, 1/40, .9, 1/40, 1/40], [1/40, 1/40, .9, 1/40, 1/40], [1/40, 1/40, .9, 1/40, 1/40], [1/40, 1/40, .9, 1/40, 1/40], [1/40, 1/40, .9, 1/40, 1/40]]), \
                    p_t4 = np.array([[1/40, 1/40, 1/40, .9, 1/40], [1/40, 1/40, 1/40, .9, 1/40], [1/40, 1/40, 1/40, .9, 1/40], [1/40, 1/40, 1/40, .9, 1/40], [1/40, 1/40, 1/40, .9, 1/40]]), \
                        p_t5 = np.array([[1/40, 1/40, 1/40, 1/40, .9], [1/40, 1/40, 1/40, 1/40, .9], [1/40, 1/40, 1/40, 1/40, .9], [1/40, 1/40, 1/40, 1/40, .9], [1/40, 1/40, 1/40, 1/40, .9]]))
        six = Markov_flex_sys_flex_env(generate_random_matrix(6), np.array([[.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50]]), \
            np.array([[1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50]]), \
                p_t3 = np.array([[1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50]]), \
                    p_t4 = np.array([[1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50]]), \
                        p_t5 = np.array([[1/50, 1/50, 1/50, 1/50, .9, 1/50], [1/50, 1/50, 1/50, 1/50, .9, 1/50], [1/50, 1/50, 1/50, 1/50, .9, 1/50], [1/50, 1/50, 1/50, 1/50, .9, 1/50], [1/50, 1/50, 1/50, 1/50, .9, 1/50], [1/50, 1/50, 1/50, 1/50, .9, 1/50]]), \
                            p_t6= np.array([[1/50, 1/50, 1/50, 1/50, 1/50, .9], [1/50, 1/50, 1/50, 1/50, 1/50, .9], [1/50, 1/50, 1/50, 1/50, 1/50, .9], [1/50, 1/50, 1/50, 1/50, 1/50, .9], [1/50, 1/50, 1/50, 1/50, 1/50, .9], [1/50, 1/50, 1/50, 1/50, 1/50, .9]]))
        seven = Markov_flex_sys_flex_env(generate_random_matrix(7), np.array([[.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60], [.9, 1/60, 1/60, 1/60, 1/60, 1/60, 1/60]]), \
            np.array([[1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60], [1/60, .9, 1/60, 1/60, 1/60, 1/60, 1/60]]), \
                 p_t3 = np.array([[1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60], [1/60, 1/60, .9, 1/60, 1/60, 1/60, 1/60]]), \
                    p_t4 = np.array([[1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60], [1/60, 1/60, 1/60, .9, 1/60, 1/60, 1/60]]), \
                         p_t5 = np.array([[1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60], [1/60, 1/60, 1/60, 1/60, .9, 1/60, 1/60]]), \
                            p_t6= np.array([[1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60], [1/60, 1/60, 1/60, 1/60, 1/60, .9, 1/60]]), \
                                p_t7= np.array([[1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9], [1/60, 1/60, 1/60, 1/60, 1/60, 1/60, .9]]))
        #eight = Markov_flex_sys_flex_env(generate_random_matrix(8), np.array([[.9, 1/70, 1/70, 1/70, 1/70, 1/70]]), generate_random_matrix(8), p_t3 = generate_random_matrix(8), p_t4 =generate_random_matrix(8), p_t5 =generate_random_matrix(8), p_t6= generate_random_matrix(8), p_t7= generate_random_matrix(8), p_t8 =generate_random_matrix(8))
        #nine = Markov_flex_sys_flex_env(generate_random_matrix(9), generate_random_matrix(9), generate_random_matrix(9), p_t3 = generate_random_matrix(9), p_t4 =generate_random_matrix(9), p_t5 =generate_random_matrix(9), p_t6= generate_random_matrix(9), p_t7= generate_random_matrix(9), p_t8 =generate_random_matrix(9), p_t9 =generate_random_matrix(9))
        #ten = Markov_flex_sys_flex_env(generate_random_matrix(10), generate_random_matrix(10), generate_random_matrix(10), p_t3 = generate_random_matrix(10), p_t4 =generate_random_matrix(10), p_t5 =generate_random_matrix(10), p_t6= generate_random_matrix(10), p_t7= generate_random_matrix(10), p_t8 =generate_random_matrix(10), p_t9 =generate_random_matrix(10), p_t99 =generate_random_matrix(10))
        two.generate_ensemble(200)
        two.calculate_probs()
        x_2 = modified_entropy_correct(two)
        three.generate_ensemble(200)
        three.calculate_probs()
        x_3 = modified_entropy_correct(three)
        four.generate_ensemble(200)
        four.calculate_probs()
        x_4 = modified_entropy_correct(four)
        five.generate_ensemble(200)
        five.calculate_probs()
        x_5 = modified_entropy_correct(five)
        six.generate_ensemble(200)
        six.calculate_probs()
        x_6 = modified_entropy_correct(six)
        seven.generate_ensemble(200)
        seven.calculate_probs()
        x_7 = modified_entropy_correct(seven)
        # eight.generate_ensemble(100)
        # eight.calculate_probs()
        # x_8 = modified_entropy_correct(eight)
        # nine.generate_ensemble(100)
        # nine.calculate_probs()
        # x_9 = modified_entropy_correct(nine)
        # ten.generate_ensemble(100)
        # ten.calculate_probs()
        # x_10 = modified_entropy_correct(ten)
        table[i] = [x_2, x_3, x_4, x_5, x_6, x_7]
    df = pd.DataFrame(table, columns = ['2', '3', '4', '5', '6', '7'])
    print(df)

#strongly_coupled_cycle(5)

def test_coupling_and_pred(num_sims):
    """
    is there a one-to-one correspondence between our coupling measure and the saturation PP?
    starting with simple 3 state env 3 state system 
    """
    list_coupling_vals = []
    list_sat_pp = []
    list_traj = []
    list_sys_probs = []
    list_just_h_sx = []
    list_env_matrices = [] 
    list_markovs = []
    for i in range(num_sims):
        three = Markov_flex_sys_flex_env(np.array([[.5, .2, .3], [.1, .8, .1], [.2, .4, .4]]), \
            generate_random_matrix(3), generate_random_matrix(3), p_t3= generate_random_matrix(3))
        three.generate_ensemble(500)
        three.calculate_probs()
        list_coupling_vals.append(modified_entropy_correct(three))
        list_sat_pp.append(sum(np.array(three.I_pred_one_step()[50:])[np.isfinite(three.I_pred_one_step()[50:])])/\
            len(np.array(three.I_pred_one_step()[50:])[np.isfinite(three.I_pred_one_step()[50:])]))
        list_traj.append(three.I_pred_one_step())
        list_just_h_sx.append(entropy_of_conditional_probs_sx(three))
        list_markovs.append(three)
    plt.figure() # scatterplot of coupling vals and PP 
    plt.xlabel("coupling vals")
    plt.ylabel("max pp")
    plt.plot(list_coupling_vals, list_sat_pp, 'ro')
    plt.show()
    plt.figure() # scatterplot of entropy of conditional probs and PP
    plt.plot(list_just_h_sx, list_sat_pp, 'ro')
    plt.xlabel("H(S|X)")
    plt.ylabel("max pp")
    plt.figure() # plot PP trajectories 
    x = [x for x in range(99)]
    for traj in list_traj:
        plt.plot(x, traj)
    plt.show()
    print(list_coupling_vals)
    print(list_sat_pp)
    print(list_markovs)

test_coupling_and_pred(100)
        


    








""""
testing 7/25
"""
# # entropy should be low for coupled
# # 3 states in env, 3 in sys 
# three_strongly_coupled = Markov_three(np.array([[.05, .9, .05], [.05, .05, .9], [.9, .05, .05]]), \
#     np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
#         np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
#             np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))

# # env states > sys states: 4 states in env, 3 in sys
# four_strongly_coupled = Markov_four(np.array([[.03, .9, .03, .04], [.03, .03, .9, .04], [.03, .03, .04, .9], [.9, .03, .03, .04]]), \
#     np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), \
#         np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
#             np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]), \
#                 np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))

# # 5 states in env, 5 in sys
# five_strongly_coupled = test_1 = Markov_five_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
#     np.array([[.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025]]), \
#         np.array([[.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025]]), \
#             np.array([[.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025]]), \
#                     np.array([[.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025]]), \
#                         np.array([[.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9]]))

# # sys states > env states: 5 states in env, 7 states in sys
# five_env_seven_sys_strongly_coupled = Markov_seven_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
#     np.array([[.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02], [.45, .45, .02, .02, .02, .02, .02]]), \
#         np.array([[.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02], [.02, .02, .45, .45, .02, .02, .02]]), \
#             np.array([[.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02], [.015, .015, .015, .015, .9, .02, .02]]), \
#                 np.array([[.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02], [.015, .015, .015, .015, .02, .9, .02]]), \
#                     np.array([[.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9], [.015, .015, .015, .015, .02, .02, .9]]))


# # entropy should be high for randomly generated 
# three_1 = Markov_three(np.array([[.05, .9, .05], [.05, .05, .9], [.9, .05, .05]]), \
#     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))
# three_2 = Markov_three(generate_random_matrix(3), \
#     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))
# # three_3 = Markov_three(np.array([[.05, .9, .05], [.05, .05, .9], [.9, .05, .05]]), \
# #     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))

# four_1 = Markov_four(np.array([[.03, .9, .03, .04], [.03, .03, .9, .04], [.03, .03, .04, .9], [.9, .03, .03, .04]]), \
#     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))
# four_2 = Markov_four(generate_random_matrix(4), \
#     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))
# # four_3 = Markov_four(np.array([[.03, .9, .03, .04], [.03, .03, .9, .04], [.03, .03, .04, .9], [.9, .03, .03, .04]]), \
# #     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))

# five_1 = Markov_five_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
#     generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5))
# five_2 = Markov_five_sys_five_env(generate_random_matrix(5), \
#     generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5))
# # five_3 = Markov_four(np.array([[.03, .9, .03, .04], [.03, .03, .9, .04], [.03, .03, .04, .9], [.9, .03, .03, .04]]), \
# #     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3))

# five_env_seven_sys_1 = Markov_seven_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
#     generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5), generate_random_matrix(5))
# five_env_seven_sys_2 = Markov_seven_sys_five_env(generate_random_matrix(5), \
#     generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7), generate_random_matrix(7))
# # five_env_seven_sys_1 = Markov_seven_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
# #     generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate)


# # need to gen ensemble calc probs first 
# print("3 env 3 sys")
# three_strongly_coupled.generate_ensemble(200)
# three_strongly_coupled.calculate_probs()
# print("strongly coupled by hand: " + str(modified_entropy_correct(three_strongly_coupled)))
# three_1.generate_ensemble(200)
# three_1.calculate_probs()
# print("cycle in env, random system: "+ str(modified_entropy_correct(three_1)))
# three_2.generate_ensemble(200)
# three_2.calculate_probs()
# print("random env, random sys: "+ str(modified_entropy_correct(three_2)))
# # three_3.generate_ensemble(200)
# # three_3.calculate_probs()
# # print("cycle in env, non cyclic sys by hand: %d %s" % (entropy_of_joint_probs(three_3), entropy_of_conditional_probs_sx(three_3), verify(three_3)))


# print("4 env 3 sys strong coupling")
# four_strongly_coupled.generate_ensemble(200)
# four_strongly_coupled.calculate_probs()
# print("strongly coupled by hand: " + str(modified_entropy_correct(four_strongly_coupled)))
# four_1.generate_ensemble(200)
# four_1.calculate_probs()
# print("cycle in env, random system: " +  str(modified_entropy_correct(four_1)))
# four_2.generate_ensemble(200)
# four_2.calculate_probs()
# print("random env, random sys: " + str(modified_entropy_correct(four_2)))
# # four_3.generate(200)
# # four_3.calculate_probs()
# # print("cycle in env, non cyclic sys by hand: %d %s" % (entropy_of_joint_probs(four_3), entropy_of_conditional_probs_sx(four_3), verify(four_3)))

# print("5 env 5 sys strong coupling")
# five_strongly_coupled.generate_ensemble(200)
# five_strongly_coupled.calculate_probs()
# print("strongly coupled by hand: " + str(modified_entropy_correct(five_strongly_coupled)))
# five_1.generate_ensemble(200)
# five_1.calculate_probs()
# print("cycle in env, random system: " + str(modified_entropy_correct(five_1)))
# five_2.generate_ensemble(200)
# five_2.calculate_probs()
# print("random env, random sys: " + str(modified_entropy_correct(five_1)))
# # five_3.generate(200)
# # five_3.calculate_probs()
# # print("cycle in env, non cyclic sys by hand: %d %s" % (entropy_of_joint_probs(five_3), entropy_of_conditional_probs_sx(five_3), verify(five_3)))


# print("5 env 7 sys strong coupling")
# five_env_seven_sys_strongly_coupled.generate_ensemble(200)
# five_env_seven_sys_strongly_coupled.calculate_probs()
# print("strongly coupled by hand: " + str(modified_entropy_correct(five_env_seven_sys_strongly_coupled)))
# five_env_seven_sys_1.generate_ensemble(200)
# five_env_seven_sys_1.calculate_probs()
# print("cycle in env, random system: " + str(modified_entropy_correct(five_env_seven_sys_1)))
# five_env_seven_sys_2.generate_ensemble(200)
# five_env_seven_sys_2.calculate_probs()
# print("random env, random sys: " + str(modified_entropy_correct(five_env_seven_sys_2)))
# # five_env_seven_sys_3.generate_ensemble(200)
# # five_env_seven_sys_3.calculate_probs()
# # print("cycle in env, non cyclic sys by hand: %d %s" % entropy_of_joint_probs(five_env_seven_sys_3), entropy_of_conditional_probs_sx(five_env_seven_sys_3), verify(five_env_seven_sys_3))






