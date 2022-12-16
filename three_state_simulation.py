"""
6/16/22. Simulate chain and generate Markov sequence.
"""
# draw an initial state s_t from p_s, where s_t ~ Multinomial(1, p_s)
# draw next state s_t+1 ~ Multinomial(1, p_Ti) where i is the index of the state

from operator import length_hint
import numpy as np
from scipy.stats import multinomial
from typing import List
import matplotlib.pyplot as plt

time_steps = 100

def stationary_distribution(p_transition):
    """
    Calculates stationary distribution.
    """
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq

def markov_sequence(p_init: np.array, p_transition: np.array, sequence_length: int) \
    -> List[int]:
    """
    Generate a Markov sequence based on p_init and p_transition
    """
    if p_init is None:
        p_init = stationary_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)
    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states

def build_environment(p_tenv):
    p_init_env = stationary_distribution(p_tenv)
    global env_states
    env_states = markov_sequence(p_init_env, p_tenv, sequence_length=100)
    env_states = [x + 1 for x in env_states]
    return env_states

def couple_helper(p_transition: np.array, prev_state: int):
    p_tr = p_transition[prev_state]
    new_state = list(multinomial.rvs(1, p_tr)).index(1)
    return new_state

def couple(env_states: List[int], sys_probs: List[np.array]):
    """
    takes in env_states and list of matrices representing transition probabilities 
    of system depending on environemnt state, returns coupled sys_states, which is a list
    """
    curr_probability_mat = sys_probs[env_states[0]-1]
    initial_state = list(multinomial.rvs(1, stationary_distribution(np.array(curr_probability_mat)))).index(1)
    sys_states = [initial_state]
    for t in range(1, len(env_states)):
        updated_probability_mat = sys_probs[env_states[t]-1]
        updated_sys_state = couple_helper(updated_probability_mat, sys_states[-1])
        sys_states.append(updated_sys_state)
    sys_states = [x + 1 for x in sys_states]
    return sys_states


# "shifts" states up for clarify (starts at state 1 instead of state 0)
def env_plot(env_states):
    plt.xlabel("time step")
    plt.ylabel("state")
    plt.title("environment states")
    plt.stairs(env_states)
    plt.show()

# later- int in argument gives ability to plot multiple simulations on top of each other in same graph
def sys_plot(sys_states):
    plt.xlabel("time step")
    plt.ylabel("state")
    plt.title("system states after coupling with environment")
    plt.stairs(sys_states)
    plt.show()


# # plotting a 3 state where transition matrix probabilities are identical
# tends to state 3 > state 2 > state 1 
# p_transition_env = np.array([[.05, .9, .05],
#                              [.05, .05, .9],
#                              [.9, .05, .05]])
# p_transition_env = np.array([[.005, .99, .005],
#                              [.005, .005, .99],
#                              [.99, .005, .005]])
# p_transition_env = np.array([[1/3, 1/3, 1/3],
#                               [1/3, 1/3, 1/3],
#                               [1/3, 1/3, 1/3]])
# p_transition_env = np.array([[.2, .4, .4],
#                               [.1, .6, .3],
#                               [.2, .6, .2]])
# p_transition_env = np.array([[0.07884272643629353, 0.30809612969402367, 0.6130611438696828],
#   [0.30809612969402367, 0.5717289824886116, 0.12017488781736474],
#   [0.6130611438696828, 0.12017488781736474, 0.26676396831295246]])
# this does not satisfy even though stationary dist is uniform:      
# p_transition_env = np.array([[0, 2/3, 1/3],
#                              [1/3, 0, 2/3],
#                              [2/3, 1/3, 0]])  
# p_transition_env = np.array([[.005, .99, .005],
#                              [.005, .005, .99],
#                              [.99, .005, .005]])              
# p_transition_env = np.array([[0.50168961, 0.49741287, 0.00089752],
#  [0.25255844, 0.5321737,  0.21526786],
#  [0.25959383, 0.62619742, 0.11420875]])
# p_transition_env = np.array([[0.57065764, 0.26678846, 0.1625539 ],
#  [0.09170592, 0.89289868, 0.0153954 ],
#  [0.72871164, 0.12407509, 0.14721327]])
# p_transition_env = np.array([[.001, .9, .099],
# [.3, .4, .3],
# [.9998, .0001, .0001]]) 
# p_transition_env = np.array([[42/100, 50/100, 2/100],
#           [1/100, 94/100, 40/100],
#           [5/100, 5/100, 70/100]])



# now, add the environment, assume initial is steady state so it's in the 
# build environment function 

# if environment in state A, will tend to state 1 
# p_t1 = np.array([[1/3, 1/3, 1/3],
#                               [1/3, 1/3, 1/3],
#                               [1/3, 1/3, 1/3]])
# p_t1 = np.array([[.1, .5, .4],
#                               [.4, .3, .3],
#                               [.1, .8, .1]])
# p_t1 = np.array([[.05, .9, .05],
#          [.05, .9, .05],
#          [.05, .9, .05]])
# if environment in state B, will tend to state 2
# p_t2 = np.array([[.2, .4, .4],
#                               [.1, .6, .3],
#                               [.2, .6, .2]])
# p_t2 = np.array([[.1, .3, .6],
#                               [.3, .1, .6],
#                               [.2, .6, .2]])
# p_t2 = np.array([[.05, .05, .9],
#          [.05, .05, .9],
#          [.05, .05, 9]])
# if environment in state C, is random 
# p_t3 = np.array([[.2, .4, .4],
#                               [.1, .6, .3],
#                               [.2, .6, .2]])
# p_t3 = np.array([[.1, .7, .2],
#                               [.6, .1, .3],
#                               [.2, .3, .5]])

# p_t3 = np.array([[.9, .05, .05],
#          [.9, .05, .05],
#          [.9, .05, .05]])

# test_env = build_environment(p_transition_env)
# env_plot(test_env)
# sys_states = couple(test_env, [p_t1, p_t2, p_t3])
# sys_plot(sys_states)




