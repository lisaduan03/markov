import numpy as np
import random
"""
7/6/22
created class for generating matrices
can generate random of any size, 3 by 3 DB, 4 by 4 DB"""


def generate_random_matrix(num: int):
    """
    generates a num by num matrix 
    """
    mat = np.random.dirichlet(np.ones(num),size=num)
    return mat

def generate_four_state_cycle():
    """
    generates a 4 by 4 matrix where opposing sides are not connected

    """
    mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(3),size=4)
    mat[0][0] = vals[0][0]
    mat[0][1] = vals[0][1]
    mat[0][3] = 1- (vals[0][0] + vals[0][1])
    mat[1][0] = vals[1][0]
    mat[1][1] = vals[1][1]
    mat[1][2] = 1- (vals[1][0] + vals[1][1])
    mat[2][1] = vals[2][0]
    mat[2][2] = vals[2][1]
    mat[2][3] = 1- (vals[2][0] + vals[2][1])
    mat[3][0] = vals[3][0]
    mat[3][2] = vals[3][1]
    mat[3][3] = 1- (vals[3][0] + vals[3][1])
    return np.array(mat)

def generate_three_state_line():
    mat = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(2),size=2)
    mat[0][0] = vals[0][0]
    mat[0][1] = 1- vals[0][0]
    mat[2][1] = vals[1][0]
    mat[2][2] = 1- vals[1][0]
    vals_2 = np.random.dirichlet(np.ones(3), size=1)
    mat[1] = vals_2[0].tolist()
    return mat

def generate_four_state_line():
    mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(2),size=2)
    vals_2 = np.random.dirichlet(np.ones(3), size=2)
    mat[0][0] = vals[0][0]
    mat[0][1] = 1 -vals[0][0]
    mat[3][2] = vals[1][0]
    mat[3][3] = 1- vals[1][0]
    mat[1][0] = vals_2[0][0]
    mat[1][1] = vals_2[0][1]
    mat[1][2] = 1 - (vals_2[0][0] + vals_2[0][1])
    mat[2][1] = vals_2[1][0]
    mat[2][2] = vals_2[1][1]
    mat[2][3] = 1- (vals_2[1][0] + vals_2[1][1])
    return mat

def generate_four_state_two_cycle():
    mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(4),size=2)
    vals_2 = np.random.dirichlet(np.ones(3), size=2)
    mat[0] = vals[0].tolist()
    mat[2] = vals[1].tolist()
    mat[1][0] = vals_2[0][0] 
    mat[1][1] = vals_2[0][1]
    mat[1][2] = 1- (vals_2[0][0] + vals_2[0][1])
    mat[3][0] = vals_2[1][0] 
    mat[3][1] = vals_2[1][1]
    mat[3][2] = 1 - (vals_2[1][0] + vals_2[1][1])
    return mat

def generate_four_state_star():
    mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(4),size=1)
    vals_2 = np.random.dirichlet(np.ones(2), size=3)
    mat[0] = vals[0].tolist()
    mat[1][0] = vals_2[0][0]
    mat[1][1] = 1- vals_2[0][0]
    mat[2][0] = vals_2[1][0]
    mat[2][2] = 1- vals_2[1][0]
    mat[3][0] = vals_2[2][0]
    mat[3][3] = 1- vals_2[2][0]
    return mat

def generate_four_state_with_island():
    mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    vals = np.random.dirichlet(np.ones(4),size=1)
    vals_1 = np.random.dirichlet(np.ones(3),size=2)
    vals_2 = np.random.dirichlet(np.ones(2),size=1)
    mat[2] = vals[0].tolist()
    mat[0][0] = vals_1[0][0]
    mat[0][1] = vals_1[0][1]
    mat[0][2] = 1- (vals_1[0][0] + vals_1[0][1])
    mat[1][0] = vals_1[1][0]
    mat[1][1] = vals_1[1][1]
    mat[1][2] = 1- (vals_1[1][0] + vals_1[1][1])
    mat[3][2] = vals_2[0][0]
    mat[3][3] = 1-vals_2[0][0]
    return mat
    
    
# def_db_matrix(sys_states: int):
#     """
#     8/8 trying to generate db matrix that is not symmetric
#     """




def generate_db_matrices(num: int):
    """
    hard coded 3 by 3
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

def generate_coupled_cw(mat_num: int):
    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    num = random.uniform(1/3, 1)
    if mat_num == 1:
        mat[0][1] = num
        mat[1][1] = num
        mat[2][1] = num
        other_num = (1 - num)/2
        mat[0][0] = other_num
        mat[0][2] = other_num
        mat[1][0] = other_num
        mat[1][2] = other_num
        mat[2][0] = other_num
        mat[2][2] = other_num
        return np.array(mat)
    if mat_num == 2:
        mat[0][2] = num
        mat[1][2] = num
        mat[2][2] = num
        other_num = (1 - num)/2
        mat[0][0] = other_num
        mat[0][1] = other_num
        mat[1][0] = other_num
        mat[1][1] = other_num
        mat[2][0] = other_num
        mat[2][1] = other_num
        return np.array(mat)
    else:
        mat[0][0] = num
        mat[1][0] = num
        mat[2][0] = num
        other_num = (1 - num)/2
        mat[0][1] = other_num
        mat[0][2] = other_num
        mat[1][1] = other_num
        mat[1][2] = other_num
        mat[2][1] = other_num
        mat[2][2] = other_num
        return np.array(mat)