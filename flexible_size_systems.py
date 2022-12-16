"""
7/21
for any given system graph topology, calculate predictive power and stored information
how to specify amount of coupling?  
"""

from re import T
from three_state_simulation import stationary_distribution, build_environment, couple, couple_helper, time_steps, env_plot, sys_plot
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
from typing import List
from generating_matrices import generate_random_matrix, generate_db_matrices
from collections import OrderedDict

class Markov_flex_sys_flex_env:
    """
    takes in an environment matrix of up to 10x10,m and up to 10 system transition matrices (one for each environmental state), 
    with the number of rows/columns in each matrix representing the number of system states 
    """
    def __init__(self, p_transition_env, p_t1, p_t2, **kwargs):
        self.p_env = p_transition_env
        self.p_t1 = p_t1
        self.p_t2 = p_t2
        for attr in ('p_t3', 'p_t4', 'p_t5', 'p_t6', 'p_t7', 'p_t8', 'p_t9', 'p_t99'): #tenth is called p_t99, bad temp fix 
            setattr(self, attr, kwargs.get(attr))

    sys_start_time = 50 # this can change
    # system size is determined by rows in p_t1
    sys_probabilities = [None for x in range(time_steps)]
    joint_probs = [None for x in range(time_steps)]
    cond_probs = [None for x in range(time_steps)]
    # environment size is determined by number of p_ts 
    env_probabilities = [None for x in range(time_steps)]
    env_env_probabilities = [None for x in range(time_steps)]
    h_xy = [0] * time_steps
    h_yx = [0] * time_steps
    mutual_info_c_xs = [0] * time_steps
    mutual_info_c_sx = [0] * time_steps
    list_of_joint_probs_pred = [None] * (time_steps - (sys_start_time))
    sys_probabilities_pred = sys_probabilities[sys_start_time]
    env_probabilities_pred = env_probabilities[sys_start_time:]
    list_of_joint_probs_pred_one_step = [None] * (time_steps -1) 
    list_of_cond_probs_pred_one_step_given_xt = [None] * (time_steps -1) 
    h_xs_verify = [0] * (time_steps-1)
    pred_one_step_h_xs = [None] * (time_steps -1) 

    def generate_ensemble(self, int):
        list_of_tuples = [None] * int
        list_of_p_tmats = [getattr(self, attr) for attr in dir(self) if attr.startswith('p_t') and getattr(self, attr) is not None]
        # or, list_of_p_mats = dir(sel)[1:len(self.p_env) + 1]
        for x in range(int):
            temp_env = build_environment(self.p_env)
            tuple = (temp_env, couple(temp_env, list_of_p_tmats))
            list_of_tuples[x] = tuple
            #env_plot(tuple[0])
            #sys_plot(tuple[1])
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
        n = len(list_of_env_sys_tuples)
        for t in range(time_steps): #  gives us number of time steps
            dict_env_state_counts = {}
            dict_sys_state_counts =  {}
            dict_joint_counts = {}
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][t] not in dict_sys_state_counts:
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] = {}
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] = 1
                elif list_of_env_sys_tuples[e][1][t] in dict_sys_state_counts:
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] += 1
                if list_of_env_sys_tuples[e][0][t] not in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] = {}
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] = 1
                elif list_of_env_sys_tuples[e][0][t] in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] += 1
                if (str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][1][t])) not in dict_joint_counts:
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][1][t]))] = {}
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][1][t]))] = 1
                elif (str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][1][t])) in dict_joint_counts:
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][1][t]))] += 1
            self.env_probabilities[t] = (self.sys_and_env_prob_helper(dict(sorted(dict_env_state_counts.items())), len(self.p_env), n))
            self.sys_probabilities[t] = (self.sys_and_env_prob_helper(dict(sorted(dict_sys_state_counts.items())), len(self.p_t1), n))
            self.joint_probs[t] = (self.joint_probs_helper(dict(sorted(dict_joint_counts.items())), len(self.p_env), len(self.p_t1), n))
            self.cond_probs[t] = (self.cond_probs_helper(dict(sorted(dict_joint_counts.items())), len(self.p_env), len(self.p_t1), n, t))

# dumb method I never use
    def env_calc_probs(self, int):
        env_ensembles = [None] * int
        for e in range(int):
            env = build_environment(self.p_env)
            env_ensembles[e] = env
        global list_of_env_ensembles 
        list_of_env_ensembles = env_ensembles
        for t in range(time_steps):
            dict_env_state_counts = {}
            for e in range(int):
                if list_of_env_ensembles[e][t] not in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_ensembles[e][t]] = {}
                    dict_env_state_counts[list_of_env_ensembles[e][t]] = 1
                elif list_of_env_ensembles[e][t] in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_ensembles[e][t]] += 1
            self.env_env_probabilities[t] = (self.sys_and_env_prob_helper(dict(sorted(dict_env_state_counts.items())), len(self.p_env), int))

    

    def sys_and_env_prob_helper(self, dictionary, num_states, num_ensembles):
        """
        retunrs a list of probabilities at time step t, ensuring that the size of the list 
        is equal to the number of states in the environment or the system
        """
        prob_list = [None] * num_states
        for x in range(num_states):
            state_num = x + 1 
            if state_num in dictionary:
                prob_list[x] = dictionary.get(state_num)/num_ensembles
            elif state_num not in dictionary:
                prob_list[x] = 0
        return prob_list
    
    def joint_probs_helper(self, dictionary, num_env_states, num_sys_states, num_ensembles):
        """
        returns a list of list of joint probabilities at time step t, ensuring that the 
        number of rows = # of env states
        and number of columns = # of sys states
        so you would index into joint_probs[env_state][sys_state]
        this is different than before, when rows = # of sys states and cols = # of env states
        checks if there is no entry in the dictionary, in prob array value is 0
        alternative: fill up dictionary with 0 
        """
        prob_array = [[None for i in range(num_sys_states)] for i in range(num_env_states)]
        for row in range(num_env_states):
            for col in range(num_sys_states):
                entry = str(row+1) + str(col+1)
                if entry in dictionary:
                    prob_array[row][col] = dictionary.get(entry)/num_ensembles
                else:
                    prob_array[row][col] = 0
        return prob_array
    
    def cond_probs_helper(self, dictionary, num_env_states, num_sys_states, num_ensembles, time):
        """
        calc p(s|x) by getting p(s,x)/p(x)
        """
        prob_array = [[None for i in range(num_sys_states)] for i in range(num_env_states)]
        for row in range(num_env_states):
            for col in range(num_sys_states):
                entry = str(row+1) + str(col+1)
                if entry in dictionary:
                    prob_array[row][col] = (dictionary.get(entry)/num_ensembles)/(self.env_probabilities[time][row])
                else:
                    prob_array[row][col] = 0
        return prob_array

    def calc_mutual_info(self):
        """
        I(S, X)
        """
        mutual_info = [0] * time_steps
        for t in range(time_steps):
            for env_row in range(len(self.env_probabilities[0])): # of environment states 
                for sys_col in range(len(self.sys_probabilities[0])): # of system states
                    if self.joint_probs[t][env_row][sys_col] == 0:
                         self.joint_probs[t][env_row][sys_col] = .00001
                    mutual_info[t] = mutual_info[t] + (self.joint_probs[t][env_row][sys_col] * np.log(self.joint_probs[t][env_row][sys_col]/(self.sys_probabilities[t][sys_col]*self.env_probabilities[t][env_row])))
        return mutual_info

    def entropy(self, env_probs):
        """
        uses Shannon entropy formula to find system entropy at each time in System step as a list"""
        h = [0] * time_steps
        for t in range(time_steps):
            for i in range(len(self.env_probs[0])): # gives number of env states 
                if env_probs[t][i] == 0:
                    env_probs[t][0] = .000001
                h[t] = h[t] - (env_probs[t][i]*np.log(env_probs[t][i]))
        return h

    def sys_entropy(self, sys_probs):
        """
        uses Shannon entropy formula to find system entropy at each time in SYSTEM step as a list"""
        h = [0] * time_steps
        for t in range(time_steps):
            for i in range(len(self.p_t1[0])): # gives number of SYS states 
                if sys_probs[t][i] == 0:
                    sys_probs[t][0] = .000001
                h[t] = h[t] - (sys_probs[t][i]*np.log(sys_probs[t][i]))
        return h
    
    def pred_given_xt_helper(self, dictionary, num_env_states, num_sys_states, num_ensembles, time):
        prob_3darray = [[[None for col in range(num_sys_states)] for row in range(num_env_states)] for row in range(num_env_states)]
        for row_xt in range(num_env_states):
            for row_xtplus1 in range(num_env_states):
                for col in range(num_sys_states):
                    entry = str(row_xt + 1) + str(row_xtplus1+1) + str(col+1)
                    if entry in dictionary:
                        prob_3darray[row_xt][row_xtplus1][col] = (dictionary.get(entry)/num_ensembles)/(self.env_probabilities[time][row_xt])
                    else:
                        prob_3darray[row_xt][row_xtplus1][col] = 0
        return prob_3darray

    
    def pred_one_step_ahead(self):
        """  
        fills up list_of_joint_probs list of matrices, with the following format:
        t is NOT fixed 
        default should be start at 0 

        [[[P(S_t_X_t+1 = a_1), P(S_t_X_t+1 = b_1), P(S_t_X_t+1 = c_1)], [[P(S_t+1_X_t+2 = a_1), P(S_t+1_X_t+2 = b_1), P(S_t+1_X_t+2 = c_1)],      
        [P(S_t_X_t+1 = a_2), P(S_t_X_t+1 = b_2), P(S_t_X_t+1 = c_2)],  [P(S_t+1_X_t+2 = a_2), P(S_t+1_X_t+2 = b_2), P(S_t+1_X_t+2 = c_2)], 
        [P(S_t_X_t+1 = a_3), P(S_t_X_t+1 = b_3), P(S_t_X_t+1 = c_3)]], [P(S_t+1_X_t+2 = a_3), P(S_t+1_X_t+2 = b_3), P(S_t+1_X_t+2 = c_3)]], ... 
        """
        n = len(list_of_env_sys_tuples)
        for t in range(time_steps-1): #  gives us number of time steps
            dict_env_state_counts = {}
            dict_sys_state_counts =  {}
            dict_joint_counts = {}
            dict_joint_counts_given_xt = {} # added 8/11 to verify
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][t] not in dict_sys_state_counts:
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] = {}
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] = 1
                elif list_of_env_sys_tuples[e][1][t] in dict_sys_state_counts:
                    dict_sys_state_counts[list_of_env_sys_tuples[e][1][t]] += 1
                if list_of_env_sys_tuples[e][0][t+1] not in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t+1]] = {}
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t+1]] = 1
                elif list_of_env_sys_tuples[e][0][t+1] in dict_env_state_counts:
                    dict_env_state_counts[list_of_env_sys_tuples[e][0][t+1]] += 1
                if (str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t])) not in dict_joint_counts:
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] = {}
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] = 1
                elif (str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t])) in dict_joint_counts:
                    dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] += 1
                # adding 8/11 xt, xt+1, s_t
                if (str(list_of_env_sys_tuples[e][0][t])+ str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t])) not in dict_joint_counts_given_xt:
                    dict_joint_counts_given_xt[str(list_of_env_sys_tuples[e][0][t]) + (str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] = {}
                    dict_joint_counts_given_xt[str(list_of_env_sys_tuples[e][0][t]) + (str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] = 1
                elif (str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t])) in dict_joint_counts_given_xt:
                    dict_joint_counts_given_xt[str(list_of_env_sys_tuples[e][0][t]) + (str(list_of_env_sys_tuples[e][0][t+1]) + str(list_of_env_sys_tuples[e][1][t]))] += 1
            self.list_of_joint_probs_pred_one_step[t] = (self.joint_probs_helper(dict(sorted(dict_joint_counts.items())), len(self.p_env), len(self.p_t1), n))
            self.list_of_cond_probs_pred_one_step_given_xt[t] = (self.pred_given_xt_helper(dict(sorted(dict_joint_counts_given_xt.items())), len(self.p_env), len(self.p_t1), n, t))

    def I_pred_one_step(self):
        self.pred_one_step_ahead()
        global I_pred_list_one_step
        I_pred_list_one_step = [0] * len(self.list_of_joint_probs_pred_one_step)
        """
        calculates predictive information S_t has about env one step ahead X_t+1
        """
        for t in range(len(self.list_of_joint_probs_pred_one_step)):
            for env_row in range(len(self.env_probabilities[0])): # of environment states 
                for sys_col in range(len(self.sys_probabilities[0])): # of system states
                    if self.list_of_joint_probs_pred_one_step[t][env_row][sys_col] == 0:
                         self.list_of_joint_probs_pred_one_step[t][env_row][sys_col] = .00001
                    if self.sys_probabilities[t][sys_col] == 0:
                        self.sys_probabilities[t][sys_col] = .00001
                    if self.env_probabilities[t+1][env_row] == 0:
                        self.env_probabilities[t+1][env_row] = .00001
                    I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(self.list_of_joint_probs_pred_one_step[t][env_row][sys_col], mp.log(self.list_of_joint_probs_pred_one_step[t][env_row][sys_col] / mp.fmul(self.env_probabilities[t+1][env_row], self.sys_probabilities[t][sys_col])))) 
        return I_pred_list_one_step


    def verify_I_pred_one_step(self, joint_prob: List[np.array], cond_term: np.array):
        """
        For H[X|S], where X env and S is sys, plug the joint probs and S (sys) probabilities into this method
        """
        for t in range(len(self.list_of_joint_probs_pred_one_step)):
            for env_row in range(len(self.env_probabilities[0])): # of environment states 
                for sys_col in range(len(self.sys_probabilities[0])): # of system states
                    if joint_prob[t][env_row][sys_col] and cond_term[t][sys_col] != 0:
                        self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][env_row][sys_col] * np.log(cond_term[t][sys_col]/joint_prob[t][env_row][sys_col]))
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
        plt.ylim(0, 2)
        plt.xlabel("time step")
        plt.ylabel("predictive power I[s_t, x_t+1]")
        plt.title("predictive power of system one step ahead")
        plt.plot(x_axis, self.pred_one_step_using_conditional_xs(), label = "using conditional")
        plt.plot(x_axis, I_pred_list_one_step, label = "using MI formula")
        legend =  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        legend.get_frame().set_facecolor('w')
        plt.show()
    
    def stored_information(self):
    # right now assume none of matrix entries are 0
    # double check that this is sum of all i and j even when i=j, 9 combos not 6
        stored_info = 0 
        steady_state = stationary_distribution(self.p_env)
        for i in range(len(self.p_env)):
            for j in range(len(self.p_env)):
                if self.p_env[i][j] != 0:
                    stored_info = stored_info + steady_state[i] * self.p_env[i][j] * np.log(self.p_env[i][j]/steady_state[j])
        return stored_info

    def calc_joint_prob_time_slice(self, t):
        n = len(list_of_env_sys_tuples)
        dict_env_state_counts = {}
        dict_env_plus_one_state_counts =  {}
        dict_joint_counts = {}
        global joint_prob_time_slice
        joint_prob_time_slice = [[None for i in range(len(self.p_env))] for i in range(len(self.p_env))]
        for e in range(n): # gives us number of ensembles
            if list_of_env_sys_tuples[e][0][t] not in dict_env_state_counts:
                dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] = {}
                dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] = 1
            elif list_of_env_sys_tuples[e][0][t] in dict_env_state_counts:
                dict_env_state_counts[list_of_env_sys_tuples[e][0][t]] += 1
            if list_of_env_sys_tuples[e][0][t+1] not in dict_env_plus_one_state_counts:
                dict_env_plus_one_state_counts[list_of_env_sys_tuples[e][0][t+1]] = {}
                dict_env_plus_one_state_counts[list_of_env_sys_tuples[e][0][t+1]] = 1
            elif list_of_env_sys_tuples[e][0][t+1] in dict_env_plus_one_state_counts:
                dict_env_plus_one_state_counts[list_of_env_sys_tuples[e][0][t+1]] += 1
            if (str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1])) not in dict_joint_counts:
                dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1]))] = {}
                dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1]))] = 1
            elif (str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1])) in dict_joint_counts:
                dict_joint_counts[(str(list_of_env_sys_tuples[e][0][t]) + str(list_of_env_sys_tuples[e][0][t+1]))] += 1
        joint_prob_time_slice = (self.joint_probs_helper(dict(sorted(dict_joint_counts.items())), len(self.p_env), len(self.p_env), n))
        """
        returns a single value of the joint probability of time step t and t + 1
        can be used in calc_mutual_info method to find the mutual information at given time slice
        pick a time slice at steady state to verify mutual info
        """
     

    def verify_stored_information_mi(self):
        """
        to confirm that stored info and entropy/cond entropy are correct
        I[S_t, S_t+1]
        """
        self.calc_joint_prob_time_slice(98)
        time_prob = self.env_probabilities[98]
        time_plus_one_prob = self.env_probabilities[99]
        stored_info = 0
        for row in range(len(self.p_env)):
            for col in range(len(self.p_env)):
                if joint_prob_time_slice[row][col] == 0: 
                    joint_prob_time_slice[row][col] = .00001
                stored_info = stored_info + float(mp.fmul(joint_prob_time_slice[row][col], mp.log(joint_prob_time_slice[row][col]/(time_prob[row]*time_plus_one_prob[col]))))
        return stored_info

    def verify_stored_information_steady_state(self):
        """
        to confirm that stored info is correct using the stored info formula, but the steady states is just a random time slice
        how to ensure that time slice is accurate? doing 90 right now 
        take average? 
        """
        stored_info = 0 
        for i in range(len(self.p_env)):
            for j in range(len(self.p_env)):
                if self.p_env[i][j] != 0:
                    stored_info = stored_info + self.env_probabilities[90][i] * self.p_env[i][j] * np.log(self.p_env[i][j]/ self.env_probabilities[90][j])
        return stored_info
    
    def assign_energies(self):
        """
        for strongly connected, later must check when there are missing edges
        for db matrices, energies 0. must fix and try to generate db matrices that are not symmetric
        """
        list_of_p_tmats = [getattr(self, attr) for attr in dir(self) if attr.startswith('p_t') and getattr(self, attr) is not None]
        global energies
        energies = [[None for i in range(len(self.p_t1))] for x in range(len(self.p_env))]
        for row in range(len(self.p_env)): # of env states 
            temp_mat = list_of_p_tmats[row]
            energies[row][0] = 0
            for col in range(1, len(self.p_t1)):
                energies[row][col] = -np.log((temp_mat[0][col])/(temp_mat[col][0]))
        return energies
    
    def work(self):
        """
        EDIT: returns work at each time step averaged over number ensembles 
      
        can also "flip" and get average work per trajectory, then average, but this gives us work per 
        time step which i thought would be more useful 
        """
        total_work = [None] * time_steps 
        for t in range(1, time_steps):
            avg_work_time_t = 0 
            work_time_t = 0
            for traj in list_of_env_sys_tuples:
                s_t_minus_1 = traj[1][t-1] - 1 # -1 because i added 1 for readability
                x_t_minus_1 = traj[0][t-1] -1 
                x_t = traj[0][t] - 1 
                work_time_t = work_time_t + (energies[x_t][s_t_minus_1] - energies[x_t_minus_1][s_t_minus_1])
            avg_work_time_t = work_time_t/len(list_of_env_sys_tuples)
            total_work[t] = avg_work_time_t
        return total_work[1:]
            
    def heat(self):
        """
        EDIT: returns heat at each time step averaged over number ensembles 

        """
        heat_list = [None] * time_steps
        for t in range(1, time_steps):
            avg_heat_time_t = 0 
            heat_time_t = 0
            for traj in list_of_env_sys_tuples:
                s_t = traj[1][t] - 1 # -1 because i added 1 for readability
                s_t_minus_1 = traj[1][t-1] -1 
                x_t = traj[0][t] - 1 
                heat_time_t = heat_time_t + (energies[x_t][s_t] - energies[x_t][s_t_minus_1])
            avg_heat_time_t = heat_time_t/len(list_of_env_sys_tuples)
            heat_list[t] = avg_heat_time_t
        return heat_list[1:]
    
    def check_work_and_heat_with_energy(self):
        """
        total change in energy = energy at time tau minus energy at time 0 
        should be equal to w + q 
        finding energy per traj, which is flipped from before 
        """
        work = sum(self.work())
        heat = sum(self.heat())
        total_energy = 0
        energy_work_plus_heat = work + heat 
        for traj in list_of_env_sys_tuples:
            energy_per_traj = 0 
            x_0 = traj[0][0] - 1
            s_0 = traj[1][0] -1 
            x_tau = traj[0][time_steps-1] -1
            s_tau = traj[1][time_steps-1] -1 
            energy_per_traj = (energies[x_tau][s_tau] - energies[x_0][s_0])
            total_energy = total_energy + energy_per_traj
        total_energy = total_energy/len(list_of_env_sys_tuples)
        print(energy_work_plus_heat, total_energy)
    
    def energies(self):
        """
        list of average energy of all ensembles per time step 
        """
        energies_list = [None] * time_steps
        for t in range(time_steps):
            energy_avg_time_t = 0
            for traj in list_of_env_sys_tuples:
                energy_total = 0
                x_t = traj[0][t] - 1 
                s_t = traj[1][t] - 1
                energy_total = energy_total + energies[x_t][s_t]
            energy_avg_time_t = energy_total/len(list_of_env_sys_tuples)
            energies_list[t] = energy_avg_time_t
        return energies_list
    
    def non_equilibrium_free_energies(self):
        neq_free_energies_list = [None] * time_steps
        energies_list = self.energies()
        entropy_list = self.sys_entropy(self.sys_probabilities)
        for t in range(time_steps):
            neq_free_energies_list[t] = energies_list[t] + entropy_list[t]
        return neq_free_energies_list


    def dissipation(self):
        """
        calculate dissapation at each time step = W - neq free energy
        then i will print out Imem - Ipred at each time step
        shoudld be equal 
        """
        work_list = self.work()
        neq_list = self.non_equilibrium_free_energies()
        dissipation_list = [None] * (time_steps - 1)
        for t in range(len(dissipation_list)):
            dissipation_list[t] = work_list[t] - neq_list[t]
        return dissipation_list

    def non_predictive_info(self):
        non_predictive_info = [None] * (time_steps - 1)
        imem = self.calc_mutual_info()
        ipred = self.I_pred_one_step()
        for t in range(len(non_predictive_info)):
            non_predictive_info[t] = imem[t] - ipred[t]
        return non_predictive_info

    def plot_mem_and_pred(self):
        imem = self.calc_mutual_info()
        ipred = self.I_pred_one_step()
        plt.figure()
        plt.plot(imem, label = 'mem')
        plt.plot(ipred, label = 'pred')
        plt.show()


    def test_proportional(self):
        list = [None] *(time_steps -1)
        dissipation = test.dissipation()
        non_predictive_info = test.non_predictive_info()
        for t in range(time_steps-1):
            list[t] = non_predictive_info[t]/dissipation[t]
        return list 
    
    """
    8/11 testing
    # """
    def check_probs(self, time, x_t, x_tplus1, s_t):
        for t in range(time, time_steps - 1):
            lhs = self.list_of_cond_probs_pred_one_step_given_xt[t][x_t][x_tplus1][s_t]
            rhs = self.cond_probs[t][x_t][s_t] * self.p_env[x_t][x_tplus1]
            print(lhs, rhs)
                
# testing the claim holds on the last day
# test = Markov_flex_sys_flex_env(generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), p_t3= generate_random_matrix(3))
# print("3 by 3, random env and sys")
# test.generate_ensemble(10000)
# test.calculate_probs()
# test.pred_one_step_ahead()
# test.check_probs(0, 1, 2, 0)

# test_strong_coupling = Markov_flex_sys_flex_env(np.array([[.05, .9, .05], [.05, .05, .9], [.9, .05, .05]]), np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]), np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), p_t3= np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]))
# print("3 by 3, strong coupling")
# test_strong_coupling.generate_ensemble(10000)
# test_strong_coupling.calculate_probs()
# test_strong_coupling.pred_one_step_ahead()
# test_strong_coupling.check_probs(0, 2, 1, 1)



"""
testing thermodynamic calculations
"""
# test = Markov_flex_sys_flex_env(generate_random_matrix(3), np.array([[.2, .4, .4],
#          [.1, .6, .3],
#           [.2, .6, .2]]), np.array([[0, 1/4, 1/4], [1/2, 1/2, 1/4], [1/2, 1/4, 1/2]]), p_t3 = generate_db_matrices(1)[0])
# test.generate_ensemble(1000)
# test.calculate_probs()
# test.assign_energies()
# test.check_work_and_heat_with_energy()
# test.plot_mem_and_pred()
# # print("dissipation", test.dissipation())
# # print("nonpredictive info", test.non_predictive_info())
# print(test.dissipation()/test.non_predictive_info())


    # def check_entropies(self, time):
    #     # for H[X_(t+1)|S_t]
    #     h_x = self.entropy(self.env_probabilities)
    #     h_x_t_plus_1 = h_x[time + 1]
    #     h_x_t_plus_1_given_s_t = 0
    #     for env_row in range(len(self.env_probabilities[0])): # of environment states 
    #             for sys_col in range(len(self.sys_probabilities[0])): # of system states
    #                 h_x_t_plus_1_given_s_t  = h_x_t_plus_1_given_s_t + (self.list_of_joint_probs_pred_one_step[time][env_row][sys_col] * np.log(self.sys_probabilities[time][sys_col]/self.list_of_joint_probs_pred_one_step[time][env_row][sys_col]))
    #     print("h_x_t_plus_1", h_x_t_plus_1)
    #     print("h_x_t_plus_1_given_s_t", h_x_t_plus_1_given_s_t)

"""
8/10 testing what ugur told to test 
"""
# test = Markov_flex_sys_flex_env(generate_random_matrix(3), np.array([[.2, .4, .4],
#          [.1, .6, .3],
#           [.2, .6, .2]]), np.array([[0, 1/4, 1/4], [1/2, 1/2, 1/4], [1/2, 1/4, 1/2]]), p_t3 = generate_db_matrices(1)[0])
# test.generate_ensemble(5000)
# test.calculate_probs()
# test.pred_one_step_ahead()
# test.check_entropies(90)
# print("3 state env, 3 state sys, db environment and db system")

# test_1 = Markov_flex_sys_flex_env(generate_random_matrix(10), generate_random_matrix(10), generate_random_matrix(10), p_t3= generate_random_matrix(10), p_t4= generate_random_matrix(10), p_t5= generate_random_matrix(10), p_t6= generate_random_matrix(10), \
#     p_t7 = generate_random_matrix(10), p_t8 =generate_random_matrix(10), p_t9 =generate_random_matrix(10), p_t99 = generate_random_matrix(10))
# test_1.generate_ensemble(5000)
# test_1.calculate_probs()
# test_1.pred_one_step_ahead()
# test_1.check_entropies(90)
# print("10 state env, 10 state sys, randomly generated env and system")

# test_2 = Markov_flex_sys_flex_env(generate_random_matrix(5), generate_random_matrix(2), generate_random_matrix(2), p_t3= generate_random_matrix(2), p_t4= generate_random_matrix(2), p_t5= generate_random_matrix(2))
# test_2.generate_ensemble(5000)
# test_2.calculate_probs()
# test_2.pred_one_step_ahead()
# test_2.check_entropies(90)
# print("5 state env, 2 state sys, randomly generated env and system")

# test_3 = Markov_flex_sys_flex_env(generate_random_matrix(4), np.array([[.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50], [.9, 1/50, 1/50, 1/50, 1/50, 1/50]]), \
#     np.array([[1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50], [1/50, .9, 1/50, 1/50, 1/50, 1/50]]), \
#         p_t3 = np.array([[1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50], [1/50, 1/50, .9, 1/50, 1/50, 1/50]]), \
#             p_t4 = np.array([[1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50], [1/50, 1/50, 1/50, .9, 1/50, 1/50]]))
# test_3.generate_ensemble(5000)
# test_3.calculate_probs()
# test_3.pred_one_step_ahead()
# test_3.check_entropies(90)
# print("4 state env, 6 state sys, random env and strong coupling")

# test_4 = Markov_flex_sys_flex_env(generate_random_matrix(4), generate_random_matrix(6), generate_random_matrix(6), p_t3= generate_random_matrix(6), p_t4= generate_random_matrix(6))
# test_4.generate_ensemble(5000)
# test_4.calculate_probs()
# test_4.pred_one_step_ahead()
# test_4.check_entropies(90)
# print("4 state env, 6 state sys, random env and random systems")

# test_5 = Markov_flex_sys_flex_env(generate_random_matrix(2), generate_random_matrix(10), generate_random_matrix(10))
# test_5.generate_ensemble(5000)
# test_5.calculate_probs()
# test_5.pred_one_step_ahead()
# test_5.check_entropies(90)
# print("2 state env, 10 state sys, random env and random systems")

