"""
7/19- 5 state sys, 5 state env
"""

from re import T
from three_state_simulation import stationary_distribution, build_environment, couple, couple_helper, time_steps
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
from typing import List

class Markov_five_sys_five_env:
    def __init__(self, p_transition_env, p_t1, p_t2, p_t3, p_t4, p_t5):
        self.p_tenv = p_transition_env
        self.p_t1 = p_t1
        self.p_t2 = p_t2
        self.p_t3 = p_t3
        self.p_t4 = p_t4
        self.p_t5 = p_t5

    sys_start_time = 50 # this can change
    sys_probabilities = [[None, None, None, None, None] \
        for row in range(time_steps)]
    list_of_joint_probs = [None for row in range(time_steps)]
    env_probabilities = [[None, None, None, None, None] \
        for row in range(time_steps)]
    h_xy = [0] * time_steps
    h_yx = [0] * time_steps
    mutual_info_c_xs = [0] * time_steps
    mutual_info_c_sx = [0] * time_steps
    list_of_joint_probs_pred = [None] * (time_steps - (sys_start_time))
    sys_probabilities_pred = sys_probabilities[sys_start_time]
    env_probabilities_pred = env_probabilities[sys_start_time:]
    list_of_joint_probs_pred_one_step = [None] * (time_steps -1)  
    h_xs_verify = [0] * (time_steps-1)
    pred_one_step_h_xs = [None] * (time_steps -1) 

    def generate_ensemble(self, int):
        list_of_tuples = [None] * int
        for x in range(int):
            temp_env = build_environment(self.p_tenv)
            tuple = (temp_env, couple(temp_env, [self.p_t1, self.p_t2, self.p_t3, self.p_t4, self.p_t5]))
            list_of_tuples[x] = tuple
            # can add labels or figure out how to "group" pairs
            # env_plot(tuple[0])
            # sys_plot(tuple[1])
        global list_of_env_sys_tuples
        list_of_env_sys_tuples = list_of_tuples
        return list_of_env_sys_tuples
    
    def calculate_probs(self):
        """
        fills up sys_probabilities matrix, env_probabilities matrix, 
        where each row is the time step up to n time steps
        and each column is the probability of being at state 1 (A for environment), 2 (B), or 3 (C) respectively. 
        ex: [[P(S_0 = 1), P(S_0 = 2), P(S_0 = 3)],
            [P(S_1 = 1), P(S_1 = 2), P(S_1 = 3)],
            ...
            [P(S_n = 1), P(S_n = 2), P(S_n = 3)]]
        
        fills up list_of_joint_probs list of matrices, with the following format:
        
        """
        state_1_count = 0
        state_2_count = 0
        state_3_count = 0
        state_4_count = 0
        state_5_count = 0
        a_count = 0
        b_count = 0
        c_count = 0
        d_count = 0
        e_count = 0
        a_1 = 0
        b_1 = 0
        c_1 = 0
        d_1 = 0
        e_1 = 0
        a_2 = 0
        b_2 = 0
        c_2 = 0
        d_2 = 0
        e_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
        d_3 = 0
        e_3 = 0 
        a_4 = 0
        b_4 = 0
        c_4 = 0
        d_4 = 0
        e_4 = 0 
        a_5 = 0
        b_5 = 0
        c_5 = 0
        d_5 = 0
        e_5 = 0 
        n = len(list_of_env_sys_tuples)
        for t in range(time_steps): #  gives us number of time steps
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][t] == 1:
                    state_1_count = state_1_count+1
                    # looking through associated environments to set up joint_probabilities_list
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_1 = a_1 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_1 = b_1 + 1
                        b_count = b_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 3:
                        c_1 = c_1 + 1
                        c_count = c_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 4:
                        d_1 = d_1 + 1
                        d_count = d_count + 1
                    else: 
                        e_1 = e_1 + 1 
                        e_count = e_count + 1
                elif list_of_env_sys_tuples[e][1][t] == 2:
                    state_2_count = state_2_count+1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_2 = a_2 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_2 = b_2 + 1
                        b_count = b_count + 1
                    elif list_of_env_sys_tuples[e][0][t] == 3:
                        c_2 = c_2 + 1
                        c_count = c_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 4:
                        d_2 = d_2 + 1
                        d_count = d_count + 1
                    else: 
                        e_2 = e_2 + 1 
                        e_count = e_count + 1
                elif list_of_env_sys_tuples[e][1][t] == 3:
                    state_3_count = state_3_count+1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_3 = a_3 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_3 = b_3 + 1
                        b_count = b_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 3:
                        c_3 = c_3 + 1
                        c_count = c_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 4:
                        d_3 = d_3 + 1
                        d_count = d_count + 1
                    else: 
                        e_3 = e_3 + 1 
                        e_count = e_count + 1
                elif list_of_env_sys_tuples[e][1][t] == 4:
                    state_4_count = state_4_count + 1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_4 = a_4 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_4 = b_4 + 1
                        b_count = b_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 3:
                        c_4 = c_4 + 1
                        c_count = c_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 4:
                        d_4 = d_4 + 1
                        d_count = d_count + 1
                    else: 
                        e_4 = e_4 + 1 
                        e_count = e_count + 1
                else: 
                    state_5_count = state_5_count + 1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_5 = a_5 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_5 = b_5 + 1
                        b_count = b_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 3:
                        c_5 = c_5 + 1
                        c_count = c_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 4:
                        d_5 = d_5 + 1
                        d_count = d_count + 1
                    else: 
                        e_5 = e_5 + 1 
                        e_count = e_count + 1

            # filling the data structures
            prob_array = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[0][3] = d_1/n
            prob_array[0][4] = e_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[1][3] = d_2/n
            prob_array[1][4] = e_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            prob_array[2][3] = d_3/n
            prob_array[2][4] = e_3/n
            prob_array[3][0] = a_4/n
            prob_array[3][1] = b_4/n
            prob_array[3][2] = c_4/n
            prob_array[3][3] = d_4/n
            prob_array[3][4] = e_4/n
            prob_array[4][0] = a_5/n
            prob_array[4][1] = b_5/n
            prob_array[4][2] = c_5/n
            prob_array[4][3] = d_5/n  
            prob_array[4][4] = e_5/n                      
            self.list_of_joint_probs[t] = prob_array
            self.sys_probabilities[t][0] = state_1_count/n
            self.sys_probabilities[t][1] = state_2_count/n
            self.sys_probabilities[t][2] = state_3_count/n
            self.sys_probabilities[t][3] = state_4_count/n
            self.sys_probabilities[t][4] = state_5_count/n
            self.env_probabilities[t][0] = a_count/n
            self.env_probabilities[t][1] = b_count/n
            self.env_probabilities[t][2] = c_count/n
            self.env_probabilities[t][3] = d_count/n
            self.env_probabilities[t][4] = e_count/n
            # resetting the values 
            state_1_count = 0
            state_2_count = 0
            state_3_count = 0
            state_4_count = 0
            state_5_count = 0
            a_count = 0
            b_count = 0
            c_count = 0
            d_count = 0
            e_count = 0
            a_1 = 0
            b_1 = 0
            c_1 = 0
            d_1 = 0
            e_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            d_2 = 0
            e_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0
            d_3 = 0
            e_3 = 0
            a_4 = 0
            b_4 = 0
            c_4 = 0
            d_4 = 0
            e_4 = 0
            a_5 = 0
            b_5 = 0
            c_5 = 0
            d_5 = 0
            e_5 = 0
    
    def entropy(self, env_probs):
        """
        uses Shannon entropy formula to find system entropy at each time in ENVIRONMENT step as a list"""
        h = [None] * time_steps
        for x in range(time_steps):
            for i in range(5):
                if env_probs[x][i] == 0:
                    env_probs[x][0] = .000001
            h[x] = -((env_probs[x][0]*np.log(env_probs[x][0])) \
                + (env_probs[x][1]*np.log(env_probs[x][1])) \
                    + (env_probs[x][2]*np.log(env_probs[x][2])) \
                        + (env_probs[x][3]*np.log(env_probs[x][3])) \
                            + (env_probs[x][4]*np.log(env_probs[x][4])))        
        return h
    
    def pred_one_step_ahead(self):
        """  
        fills up list_of_joint_probs list of matrices, with the following format:
        t is NOT fixed 
        default should be start at 0 

        [[[P(S_t_X_t+1 = a_1), P(S_t_X_t+1 = b_1), P(S_t_X_t+1 = c_1)], [[P(S_t+1_X_t+2 = a_1), P(S_t+1_X_t+2 = b_1), P(S_t+1_X_t+2 = c_1)],      
        [P(S_t_X_t+1 = a_2), P(S_t_X_t+1 = b_2), P(S_t_X_t+1 = c_2)],  [P(S_t+1_X_t+2 = a_2), P(S_t+1_X_t+2 = b_2), P(S_t+1_X_t+2 = c_2)], 
        [P(S_t_X_t+1 = a_3), P(S_t_X_t+1 = b_3), P(S_t_X_t+1 = c_3)]], [P(S_t+1_X_t+2 = a_3), P(S_t+1_X_t+2 = b_3), P(S_t+1_X_t+2 = c_3)]], ... 
        """
        a_1 = 0
        b_1 = 0
        c_1 = 0
        d_1 = 0
        e_1 = 0
        a_2 = 0
        b_2 = 0
        c_2 = 0
        d_2 = 0
        e_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
        d_3 = 0
        e_3 = 0
        a_4 = 0
        b_4 = 0
        c_4 = 0
        d_4 = 0
        e_4 = 0
        a_5 = 0
        b_5 = 0
        c_5 = 0
        d_5 = 0
        e_5 = 0
        list_of_joint_probs_index = 0
        n = len(list_of_env_sys_tuples)
        for t in range(time_steps-1): # starts at sys_start, so first entry is s_t and x_t
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][t] == 1:
                    # looking through associated environments to set up joint_probabilities_list
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_1 = a_1 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_1 = b_1 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 3:
                        c_1 = c_1 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 4:
                        d_1 = d_1 + 1
                    else:
                        e_1 = e_1 + 1
                elif list_of_env_sys_tuples[e][1][t] == 2:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_2 = a_2 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_2 = b_2 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 3:
                        c_2 = c_2 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 4:
                        d_2 = d_2 + 1
                    else:
                        e_2 = e_2 + 1
                elif list_of_env_sys_tuples[e][1][t] == 3:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_3 = a_3 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_3 = b_3 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 3:
                        c_3 = c_3 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 4:
                        d_3 = d_3 + 1
                    else:
                        e_3 = e_3 + 1
                elif list_of_env_sys_tuples[e][1][t] == 4:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_4 = a_4 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_4 = b_4 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 3:
                        c_4 = c_4 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 4:
                        d_4 = d_4 + 1
                    else:
                        e_4 = e_4 + 1
                else:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_5 = a_5 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_5 = b_5 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 3:
                        c_5 = c_5 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 4:
                        d_5 = d_5 + 1
                    else:
                        e_5 = e_5 + 1

            # filling the data structures
            prob_array = [[None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None], [None, None, None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[0][3] = d_1/n
            prob_array[0][4] = e_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[1][3] = d_2/n
            prob_array[1][4] = e_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            prob_array[2][3] = d_3/n
            prob_array[2][4] = e_3/n
            prob_array[3][0] = a_4/n
            prob_array[3][1] = b_4/n
            prob_array[3][2] = c_4/n
            prob_array[3][3] = d_4/n
            prob_array[3][4] = e_4/n
            prob_array[4][0] = a_5/n
            prob_array[4][1] = b_5/n
            prob_array[4][2] = c_5/n
            prob_array[4][3] = d_5/n
            prob_array[4][4] = e_5/n
            self.list_of_joint_probs_pred_one_step[list_of_joint_probs_index] = prob_array
            # resetting the values 
            a_1 = 0
            b_1 = 0
            c_1 = 0
            d_1 = 0
            e_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            d_2 = 0
            e_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0 
            d_3 = 0 
            e_3 = 0
            a_4 = 0
            b_4 = 0
            c_4 = 0 
            d_4 = 0 
            e_4 = 0
            a_5 = 0
            b_5 = 0
            c_5 = 0 
            d_5 = 0 
            e_5 = 0
            list_of_joint_probs_index = list_of_joint_probs_index + 1 

    def I_pred_one_step(self):
        self.pred_one_step_ahead()
        global I_pred_list_one_step
        I_pred_list_one_step = [0] * len(self.list_of_joint_probs_pred_one_step)
        """
        calculates predictive information S_t has about env one step ahead X_t+1
        """
        for t in range(len(self.list_of_joint_probs_pred_one_step)):
            p_a_1 = self.list_of_joint_probs_pred_one_step[t][0][0]
            p_a_2 = self.list_of_joint_probs_pred_one_step[t][1][0]
            p_a_3 = self.list_of_joint_probs_pred_one_step[t][2][0]
            p_a_4 = self.list_of_joint_probs_pred_one_step[t][3][0]
            p_a_5 = self.list_of_joint_probs_pred_one_step[t][4][0]
            p_b_1 = self.list_of_joint_probs_pred_one_step[t][0][1]
            p_b_2 = self.list_of_joint_probs_pred_one_step[t][1][1]
            p_b_3 = self.list_of_joint_probs_pred_one_step[t][2][1]
            p_b_4 = self.list_of_joint_probs_pred_one_step[t][3][1]
            p_b_5 = self.list_of_joint_probs_pred_one_step[t][4][1]
            p_c_1 = self.list_of_joint_probs_pred_one_step[t][0][2]
            p_c_2 = self.list_of_joint_probs_pred_one_step[t][1][2]
            p_c_3 = self.list_of_joint_probs_pred_one_step[t][2][2]
            p_c_4 = self.list_of_joint_probs_pred_one_step[t][3][2]
            p_c_5 = self.list_of_joint_probs_pred_one_step[t][4][2]
            p_d_1 = self.list_of_joint_probs_pred_one_step[t][0][3]
            p_d_2 = self.list_of_joint_probs_pred_one_step[t][1][3]
            p_d_3 = self.list_of_joint_probs_pred_one_step[t][2][3]
            p_d_4 = self.list_of_joint_probs_pred_one_step[t][3][3]
            p_d_5 = self.list_of_joint_probs_pred_one_step[t][4][3]
            p_e_1 = self.list_of_joint_probs_pred_one_step[t][0][4]
            p_e_2 = self.list_of_joint_probs_pred_one_step[t][1][4]
            p_e_3 = self.list_of_joint_probs_pred_one_step[t][2][4]
            p_e_4 = self.list_of_joint_probs_pred_one_step[t][3][4]
            p_e_5 = self.list_of_joint_probs_pred_one_step[t][4][4]


            p_a = self.env_probabilities[t+1][0]
            p_b = self.env_probabilities[t+1][1]
            p_c = self.env_probabilities[t+1][2]
            p_d = self.env_probabilities[t+1][3]
            p_e = self.env_probabilities[t+1][4]
            p_1 = self.sys_probabilities[t][0]
            p_2 = self.sys_probabilities[t][1]
            p_3 = self.sys_probabilities[t][2]
            p_4 = self.sys_probabilities[t][3]
            p_5 = self.sys_probabilities[t][4]
            if p_a_1 != 0 and p_a != 0 and p_1 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_1, mp.log(p_a_1 / mp.fmul(p_a, p_1)))) 
            if p_a_1 != 0 and p_a != 0 and p_2 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_2, mp.log(p_a_2 / mp.fmul(p_a, p_2))))
            if p_a_3 != 0 and p_a != 0 and p_3 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_3, mp.log(p_a_3 / mp.fmul(p_a, p_3))))
            if p_a_4 != 0 and p_a != 0 and p_4 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_4, mp.log(p_a_4 / mp.fmul(p_a, p_4))))
            if p_a_5 != 0 and p_a != 0 and p_5 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_5, mp.log(p_a_5 / mp.fmul(p_a, p_5))))
            if p_b_1 != 0 and p_b != 0 and p_1 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_1, mp.log(p_b_1 / mp.fmul(p_b, p_1))))
            if p_b_2 != 0 and p_b != 0 and p_2 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_2, mp.log(p_b_2 / mp.fmul(p_b, p_2))))
            if p_b_3 != 0 and p_b != 0 and p_3 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_3, mp.log(p_b_3 / mp.fmul(p_b, p_3))))
            if p_b_4 != 0 and p_b != 0 and p_4 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_4, mp.log(p_b_4 / mp.fmul(p_b, p_4))))
            if p_b_5 != 0 and p_b != 0 and p_5 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_5, mp.log(p_b_5 / mp.fmul(p_b, p_5))))
            if p_c_1 != 0 and p_c != 0 and p_1 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_1, mp.log(p_c_1 / mp.fmul(p_c, p_1))))
            if p_c_2 != 0 and p_c != 0 and p_2 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_2, mp.log(p_c_2 / mp.fmul(p_c, p_2))))
            if p_c_3 != 0 and p_c != 0 and p_3 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_3, mp.log(p_c_3 / mp.fmul(p_c, p_3))))
            if p_c_4 != 0 and p_c != 0 and p_4 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_4, mp.log(p_c_4 / mp.fmul(p_c, p_4))))
            if p_c_5 != 0 and p_c != 0 and p_5 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_5, mp.log(p_c_5 / mp.fmul(p_c, p_5))))
            if p_d_1 != 0 and p_d != 0 and p_1 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_d_1, mp.log(p_d_1 / mp.fmul(p_d, p_1))))
            if p_d_2 != 0 and p_d != 0 and p_2 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_d_2, mp.log(p_d_2 / mp.fmul(p_d, p_2))))
            if p_d_3 != 0 and p_d != 0 and p_3 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_d_3, mp.log(p_d_3 / mp.fmul(p_d, p_3))))
            if p_d_4 != 0 and p_d != 0 and p_4 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_d_4, mp.log(p_d_4 / mp.fmul(p_d, p_4))))
            if p_d_5 != 0 and p_d != 0 and p_5 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_d_5, mp.log(p_d_5 / mp.fmul(p_d, p_5))))
            if p_e_1 != 0 and p_e != 0 and p_1 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_e_1, mp.log(p_e_1 / mp.fmul(p_e, p_1))))
            if p_e_2 != 0 and p_e != 0 and p_2 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_e_2, mp.log(p_e_2 / mp.fmul(p_e, p_2))))
            if p_e_3 != 0 and p_e != 0 and p_3 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_e_3, mp.log(p_e_3 / mp.fmul(p_e, p_3))))
            if p_e_4 != 0 and p_e != 0 and p_4 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_e_4, mp.log(p_e_4 / mp.fmul(p_e, p_4))))
            if p_e_5 != 0 and p_e != 0 and p_5 != 0:
                I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_e_5, mp.log(p_e_5 / mp.fmul(p_e, p_5))))
            if I_pred_list_one_step[t] == None:
                I_pred_list_one_step[t] = 0
        return I_pred_list_one_step



    def verify_I_pred_one_step(self, joint_prob: List[np.array], cond_term: np.array ):
        """
        For H[X|S], where X env and S is sys, plug the joint probs and S (sys) probabilities into this method
        """
        for t in range(len(self.list_of_joint_probs_pred_one_step)):
            if joint_prob[t][0][0] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][0]*np.log(cond_term[t][0]/joint_prob[t][0][0]))
            if joint_prob[t][1][0] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][0]*np.log(cond_term[t][1]/joint_prob[t][1][0]))
            if joint_prob[t][2][0] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][0]*np.log(cond_term[t][2]/joint_prob[t][2][0]))
            if joint_prob[t][3][0] and cond_term[t][3] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][3][0]*np.log(cond_term[t][3]/joint_prob[t][3][0])) 
            if joint_prob[t][4][0] and cond_term[t][4] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][4][0]*np.log(cond_term[t][4]/joint_prob[t][4][0]))   
            if joint_prob[t][0][1] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][1]*np.log(cond_term[t][0]/joint_prob[t][0][1])) 
            if joint_prob[t][1][1] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][1]*np.log(cond_term[t][1]/joint_prob[t][1][1]))
            if joint_prob[t][2][1] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][1]*np.log(cond_term[t][2]/joint_prob[t][2][1])) 
            if joint_prob[t][3][1] and cond_term[t][3] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][3][1]*np.log(cond_term[t][3]/joint_prob[t][3][1]))
            if joint_prob[t][4][1] and cond_term[t][4] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][4][1]*np.log(cond_term[t][4]/joint_prob[t][4][1]))   
            if joint_prob[t][0][2] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][2]*np.log(cond_term[t][0]/joint_prob[t][0][2])) 
            if joint_prob[t][1][2] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][2]*np.log(cond_term[t][1]/joint_prob[t][1][2])) 
            if joint_prob[t][2][2] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][2]*np.log(cond_term[t][2]/joint_prob[t][2][2]))
            if joint_prob[t][3][2] and cond_term[t][3] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][3][2]*np.log(cond_term[t][3]/joint_prob[t][3][2]))
            if joint_prob[t][4][2] and cond_term[t][4] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][4][2]*np.log(cond_term[t][4]/joint_prob[t][4][2]))         
            if joint_prob[t][0][3] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][3]*np.log(cond_term[t][0]/joint_prob[t][0][3])) 
            if joint_prob[t][1][3] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][3]*np.log(cond_term[t][1]/joint_prob[t][1][3])) 
            if joint_prob[t][2][3] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][3]*np.log(cond_term[t][2]/joint_prob[t][2][3]))
            if joint_prob[t][3][3] and cond_term[t][3] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][3][3]*np.log(cond_term[t][3]/joint_prob[t][3][3])) 
            if joint_prob[t][4][3] and cond_term[t][4] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][4][3]*np.log(cond_term[t][4]/joint_prob[t][4][3]))    
            if joint_prob[t][0][4] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][4]*np.log(cond_term[t][0]/joint_prob[t][0][4])) 
            if joint_prob[t][1][4] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][4]*np.log(cond_term[t][1]/joint_prob[t][1][4])) 
            if joint_prob[t][2][4] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][4]*np.log(cond_term[t][2]/joint_prob[t][2][4]))
            if joint_prob[t][3][4] and cond_term[t][3] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][3][4]*np.log(cond_term[t][3]/joint_prob[t][3][4]))  
            if joint_prob[t][4][4] and cond_term[t][4] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][4][4]*np.log(cond_term[t][4]/joint_prob[t][4][4]))   
        return self.h_xs_verify

    def pred_one_step_using_conditional_xs(self): 
        """
        to verify that I[s_t, x_t+1] is correct 
        """
        env_entropy = self.entropy(self.env_probabilities)
        cond_entropy_xs = self.verify_I_pred_one_step(self.list_of_joint_probs_pred_one_step, self.sys_probabilities)
        for t in range(time_steps-1):
            self.pred_one_step_h_xs[t] = env_entropy[t+1] - cond_entropy_xs[t]
        return self.pred_one_step_h_xs

    def plot_I_pred_one_step(self): # shift the x axis so it starts at 0 
        self.I_pred_one_step()
        # x = [x for x in range(1, len(env_probabilities_pred))]
        # string_x = [str(t) for t in x]
        # x_labels = ['t +' + string_x[t] for t in range(len(string_x))]
        x_axis = [x for x in range(time_steps - 1)]
        # plt.xticks(x_axis_list ,x_labels)
        plt.ylim(0, 1.5)
        plt.xlabel("time step")
        plt.ylabel("predictive power I[s_t, x_t+1]")
        plt.title("predictive power of system about environment one step ahead I[s_t, x_t+1]")
        plt.plot(x_axis, I_pred_list_one_step, label = "using MI formula")
        plt.plot(x_axis, self.pred_one_step_using_conditional_xs(), label = "using conditional")
        legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
        legend.get_frame().set_facecolor('w')
        plt.show()

# strongly coupled 
test_1 = Markov_five_sys_five_env(np.array([[.025, .9, .025, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .025, .9], [.9, .025, .025, .025, .025]]), \
    np.array([[.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025], [.9, .025, .025, .025, .025]]), \
        np.array([[.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025], [.025, .9, .025, .025, .025]]), \
            np.array([[.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025], [.025, .025, .9, .025, .025]]), \
                    np.array([[.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025], [.025, .025, .025, .9, .025]]), \
                        np.array([[.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9], [.025, .025, .025, .025, .9]]))

# test_1.generate_ensemble(100)
# test_1.calculate_probs()
# test_1.plot_I_pred_one_step() 