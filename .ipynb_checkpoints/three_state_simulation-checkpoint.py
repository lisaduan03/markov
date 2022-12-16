"""
6/16/22. Simulate chain and generate Markov sequence.
"""
# draw an initial state s_t from p_s, where s_t ~ Multinomial(1, p_s)
# draw next state s_t+1 ~ Multinomial(1, p_Ti) where i is the index of the state

from operator import length_hint
from tkinter import N
import numpy as np
from scipy.stats import multinomial
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt


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
    env_states = markov_sequence(p_init_env, p_tenv, sequence_length=100)
    env_states = [x + 1 for x in env_states]
    return env_states

def couple(env_states: List[int], p_init_sys: np.array, transition_probs: List[np.array]):
    """
    takes in env_states and list of matrices representing transition probabilities 
    of system depending on environemnt state, returns coupled sys_states, which is a list
    """
    initial_state = list(multinomial.rvs(1, p_init_sys)).index(1)
    sys_states = [initial_state]
    curr_probability_mat = transition_probs[env_states[0]-1]
    for state in env_states[1:]:
        updated_probability_mat = transition_probs[state -1]
        #what would be the inital dist? would it be the stationary distribution
        # of matrix associated with the current environemnt? 
        updated_sys_state = \
            markov_sequence(stationary_distribution(np.array(curr_probability_mat)), updated_probability_mat, 1)
        sys_states = sys_states + updated_sys_state
        curr_probability_mat = updated_probability_mat
    sys_states = [x + 1 for x in sys_states]
    return sys_states


# "shifts" states up for clarify (starts at state 1 instead of state 0)
def env_plot(env_states):
    plt.xlabel("time step")
    plt.ylabel("state")
    plt.yticks([1, 2, 3], ["A", "B", "C"])
    plt.title("environment states")
    plt.stairs(env_states)
    plt.show()

# later- int in argument gives ability to plot multiple simulations on top of each other in same graph
def sys_plot(sys_states):
    plt.xlabel("time step")
    plt.ylabel("state")
    plt.yticks([1, 2, 3])
    plt.title("system states after coupling with environment")
    plt.stairs(sys_states)
    plt.show()


# # plotting a 3 state where transition matrix probabilities are identical
# tends to state 3 > state 2 > state 1 
p_transition_env = np.array([[.05, .9, .05],
                             [.05, .05, .9],
                             [.9, .05, .05]])
# now, add the environment, assume initial is steady state so it's in the 
# build environment function 

# assume uniform for now, will probably chance later
p_init_sys = [1/3, 1/3, 1/3]
# if environment in state A, will tend to state 1 
p_t1 = [[.05, .9, .05],
        [.05, .9, .05],
        [.05, .9, .05]]
# if environment in state B, will tend to state 2
p_t2 = [[.05, .05, .9],
        [.05, .05, .9],
        [.05, .05, .9]]
# if environment in state C, is random 
p_t3 = [[.9, .05, .05],
        [.9, .05, .05],
        [.9, .05, .05]]

#test_env = build_environment(p_transition_env)
#generate_env_plot(test_env)
#generate_sys_plot(test_env, 1)


def generate_ensemble(int):
    list_of_tuples = [None] * int
    for x in range(int):
        temp_env = build_environment(p_transition_env)
        tuple = (temp_env, couple(temp_env, p_init_sys, [p_t1, p_t2, p_t3]))
        list_of_tuples[x] = tuple
        # can add labels or figure out how to "group" pairs
        #env_plot(tuple[0])
        #sys_plot(tuple[1])
    return list_of_tuples

generate_ensemble(15)

