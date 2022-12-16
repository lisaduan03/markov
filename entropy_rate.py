from re import T
from three_state_simulation import stationary_distribution
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
from typing import List
# from calculations_entropy_mutual_information import mutual_information, stationary_distribution, calculate_probs, entropy


def generate_random_matrix():
    """
    generates a list of 3 by 3 matrices with length given by num_of_matrices 
    """
    mat = np.random.dirichlet(np.ones(3),size=3)
    return mat


def generate_db_matrices(num: int):
    """
    generate symmetric matrices. stationary always 1/3, 1/3, 1/3
    """
    global db_list 
    db_list = [None] * num
    for x in range(num):
        n = 3
        r = np.random.rand(n*(n+1)//2)
        sym = np.zeros((n,n))
        for i in range(n):
            t = i*(i+1)//2
            sym[i,0:i+1] = r[t:t+i+1]
            sym[0:i,i] = r[t:t+i]  
        sym[0][0] = 1 - (sym[0][1] + sym[0][2])
        sym[1][1] = 1 - (sym[1][0] + sym[1][2])
        sym[2][2] = 1 - (sym[2][0] + sym[2][1])
        if (sym[0][0] < 0) or (sym[1][1] < 0) or (sym[2][2] < 0):
            generate_db_matrices_helper(sym)
        db_list[x] = sym.tolist()
    return db_list

def generate_db_matrices_helper(sym: np.array):
    n = 3
    r = np.random.rand(n*(n+1)//2)
    for i in range(n):
        t = i*(i+1)//2
        sym[i,0:i+1] = r[t:t+i+1]
        sym[0:i,i] = r[t:t+i]  
    sym[0][0] = 1 - (sym[0][1] + sym[0][2])
    sym[1][1] = 1 - (sym[1][0] + sym[1][2])
    sym[2][2] = 1 - (sym[2][0] + sym[2][1])
    if (sym[0][0] < 0) or (sym[1][1] < 0) or (sym[2][2] < 0):
        generate_db_matrices_helper(sym)
    else: 
        return sym


# generate_db_matrices(10)
# plot_many_mi(10)
    
# calculate entropy production rate for a matrix
def entropy_rate_env(matrix):
    global stationary_dist_env
    stationary_dist_env = stationary_distribution(matrix)
    entropy_rate_env_val = 0
    if (matrix[0][1] == 0):
        matrix[0][1] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][0]) - \
        (stationary_dist_env[0] * matrix[0][1]))* np.log((stationary_dist_env[1] * matrix[1][0])/(stationary_dist_env[0] * matrix[0][1]))
    if (matrix[0][2] == 0):
        matrix[0][1] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][0]) - \
        (stationary_dist_env[0] * matrix[0][2]))* np.log((stationary_dist_env[2] * matrix[2][0])/(stationary_dist_env[0] * matrix[0][2]))
    if (matrix[1][0] == 0):
        matrix[1][0] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][1]) - \
        (stationary_dist_env[1] * matrix[1][0]))* np.log((stationary_dist_env[0] * matrix[0][1])/(stationary_dist_env[1] * matrix[1][0]))
    if (matrix[1][2] == 0):
        matrix[1][2] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][1]) - \
        (stationary_dist_env[1] * matrix[1][2]))* np.log((stationary_dist_env[2] * matrix[2][1])/(stationary_dist_env[1] * matrix[1][2]))
    if (matrix[2][0] == 0):
        matrix[2][0] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][2]) - \
        (stationary_dist_env[2] * matrix[2][0]))* np.log((stationary_dist_env[0] * matrix[0][2])/(stationary_dist_env[2] * matrix[2][0]))
    if (matrix[2][1] == 0):
        matrix[2][1] = .0000001
    entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][2]) - \
        (stationary_dist_env[2] * matrix[2][1]))* np.log((stationary_dist_env[1] * matrix[1][2])/(stationary_dist_env[2] * matrix[2][1]))
    entropy_rate_env_val = entropy_rate_env_val * (1/2)
    return entropy_rate_env_val

# def average_entropy_rate_sys(env_matrix, l_o_m): # this is for the system. fix this 
#     stationary_dist_env = stationary_distribution(env_matrix)
#     p_a = stationary_dist_env[0]
#     p_b = stationary_dist_env[1]
#     p_c = stationary_dist_env[2]
#     h_a = entropy(l_o_m[0])
#     h_b = entropy(l_o_m[1])
#     h_c = entropy(l_o_m[2])
#     average_entropy_rate = (p_a * h_a) + (p_b * h_b) + (p_c * h_c)
#     return average_entropy_rate 


# def generate_random_matrix_cw_ccw(ratio: int):
#     """
#     generates a list of 3 by 3 matrices with length given by num_of_matrices 
#     and cw to ccw ratio 
#     """
#     mat = np.random.dirichlet(np.ones(3),size=3)
#     cw = mat[0][1] * mat[1][2] * mat[2][0]
#     ccw = np.random.dirichlet(np.ones(1), size=3)

#     mat[0][1] * 

#     return mat

p_transition_env = np.array([[.05, .9, .05],
          [.05, .05, .9],
          [.9, .05, .05]])

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

entropy_rate_env(np.array([[.0005, .999, .0005],
                                       [.999, .0005, .0005],
                                       [.0005, .999, .0005]]))