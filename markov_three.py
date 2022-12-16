from re import T
from three_state_simulation import stationary_distribution, build_environment, couple, couple_helper, time_steps
from matplotlib import pyplot as plt
import numpy as np
import mpmath as mp
from typing import List

    
# data to keep track of 
class Markov_three:
    def __init__(self, p_transition_env, p_t1, p_t2, p_t3):
        self.p_tenv = p_transition_env
        self.p_t1 = p_t1
        self.p_t2 = p_t2
        self.p_t3 = p_t3

    sys_start_time = 50 # this can change
    sys_probabilities = [[None, None, None] \
        for row in range(time_steps)]
    list_of_joint_probs = [None for row in range(time_steps)]
    env_probabilities = [[None, None, None] \
        for row in range(time_steps)]
    h_xy = [0] * time_steps
    h_yx = [0] * time_steps
    mutual_info_c_xs = [0] * time_steps
    mutual_info_c_sx = [0] * time_steps
    list_of_joint_probs_pred = [None] * (time_steps - (sys_start_time))
    sys_probabilities_pred = sys_probabilities[sys_start_time]
    env_probabilities_pred = env_probabilities[sys_start_time:]
    sys_probabilities_pred_and_mem = sys_probabilities[(sys_start_time - (time_steps - sys_start_time)):]
    env_probabilities_pred_and_mem = env_probabilities[(sys_start_time - (time_steps - sys_start_time)):]
    list_of_joint_probs_pred_one_step = [None] * (time_steps -1)  
    h_xs_verify = [0] * (time_steps-1)
    pred_one_step_h_xs = [None] * (time_steps -1) 
    list_of_joint_probs_pred_and_mem = [None] *  ((time_steps - sys_start_time) * 2)
    I_pred_list_and_mem = [0] * len(sys_probabilities_pred_and_mem)
    h = [None] * time_steps
    h_sk_st = [0] * ((time_steps - sys_start_time) * 2)
    conditional_entropy_mutual_info_verification = [0] * (time_steps - (sys_start_time)) * 2




    def generate_ensemble(self, int):
        list_of_tuples = [None] * int
        for x in range(int):
            temp_env = build_environment(self.p_tenv)
            tuple = (temp_env, couple(temp_env, [self.p_t1, self.p_t2, self.p_t3]))
            list_of_tuples[x] = tuple
            # can add labels or figure out how to "group" pairs
            # env_plot(tuple[0])
            # sys_plot(tuple[1])
        global list_of_env_sys_tuples
        list_of_env_sys_tuples = list_of_tuples
        return list_of_env_sys_tuples

    # for prediction

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
        a_count = 0
        b_count = 0
        c_count = 0
        a_1 = 0
        b_1 = 0
        c_1 = 0
        a_2 = 0
        b_2 = 0
        c_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
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
                    else:
                        c_1 = c_1 + 1
                        c_count = c_count + 1 
                elif list_of_env_sys_tuples[e][1][t] == 2:
                    state_2_count = state_2_count+1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_2 = a_2 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_2 = b_2 + 1
                        b_count = b_count + 1
                    else:
                        c_2 = c_2 + 1
                        c_count = c_count + 1 
                else:
                    state_3_count = state_3_count+1
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_3 = a_3 + 1
                        a_count = a_count + 1 
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_3 = b_3 + 1
                        b_count = b_count + 1 
                    else:
                        c_3 = c_3 + 1
                        c_count = c_count + 1 
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            self.list_of_joint_probs[t] = prob_array
            self.sys_probabilities[t][0] = state_1_count/n
            self.sys_probabilities[t][1] = state_2_count/n
            self.sys_probabilities[t][2] = state_3_count/n
            self.env_probabilities[t][0] = a_count/n
            self.env_probabilities[t][1] = b_count/n
            self.env_probabilities[t][2] = c_count/n
            # resetting the values 
            state_1_count = 0
            state_2_count = 0
            state_3_count = 0
            a_count = 0
            b_count = 0
            c_count = 0
            a_1 = 0
            b_1 = 0
            c_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0

    def plot_state_probs(self):
        """
        first gathers probabilities of being at state 1, 2, or 3 at given time step,
        then plots the system probabilities
        """
        data_1 = []
        data_2 = []
        data_3 = []
        for x in range(time_steps):
            data_1.append(self.sys_probabilities[x][0])
            data_2.append(self.sys_probabilities[x][1])
            data_3.append(self.sys_probabilities[x][2])
        plt.plot(data_1, label = 'state 1 ')
        plt.plot(data_2, label = 'state 2')
        plt.plot(data_3, label = 'state 3')
        legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
        legend.get_frame().set_facecolor('w')
        plt.xlabel("time step")
        plt.ylabel("probability")
        plt.title("system probability of being at state 1, 2, or 3")
        plt.show()

    def entropy(self, env_or_sys_probs):
        """
        uses Shannon entropy formula to find system entropy at each time step as a list"""
        for x in range(time_steps):
            if env_or_sys_probs[x][0] == 0:
                env_or_sys_probs[x][0] = .000001
            if env_or_sys_probs[x][1] == 0:
                env_or_sys_probs[x][1] = .000001
            if env_or_sys_probs[x][2] == 0:
                env_or_sys_probs[x][2] = .000001
            self.h[x] = -((env_or_sys_probs[x][0]*np.log(env_or_sys_probs[x][0])) \
                + (env_or_sys_probs[x][1]*np.log(env_or_sys_probs[x][1])) \
                    + (env_or_sys_probs[x][2]*np.log(env_or_sys_probs[x][2])))
        return self.h
        
    def plot_sys_entropy(self, sys_prob):
        """
        plot entropy of either environment or system over time
        """
        plt.xlabel("time step")
        plt.ylabel("entropy")
        plt.title("entropy of the system over time")
        plt.plot(self.entropy(self.sys_probabilities))
        plt.show()

    def plot_env_entropy(self, env_probs):
        """
        plot entropy of either environment or system over time
        """
        plt.xlabel("time step")
        plt.ylabel("entropy")
        plt.title("entropy of the environment over time")
        plt.plot(self.entropy(self.env_probabilities))
        plt.show()


    def verify_stationary(self): 
        """
        quick method to verify that env states over many ensembles converge to stationary/steady-state dist
        for transition matrix p_transition_env = np.array([[.1, .4, .5],
                                                        [.1, .4, .5],
                                                        [.1, .5, .4]]),
        should get [0.1       , 0.44545455, 0.45454545] based on linear algebra                                              
        """
        a_stationary = 0
        b_stationary = 0
        c_stationary = 0
        for x in range(time_steps):
            a_stationary = a_stationary + self.env_probabilities[x][0]
            b_stationary = b_stationary + self.env_probabilities[x][1]
            c_stationary = c_stationary + self.env_probabilities[x][2]
        stationary_dist_list = [None, None, None]
        stationary_dist_list[0] = a_stationary/time_steps
        stationary_dist_list[1] = b_stationary/time_steps
        stationary_dist_list[2] = c_stationary/time_steps
        return stationary_dist_list


    def mutual_information(self):
        """
        calculates mutual information at each time step 
        """
        mutual_info = [0] * time_steps
        for t in range(time_steps):
            p_a_1 = self.list_of_joint_probs[t][0][0]
            p_a_2 = self.list_of_joint_probs[t][1][0]
            p_a_3 = self.list_of_joint_probs[t][2][0]
            p_b_1 = self.list_of_joint_probs[t][0][1]
            p_b_2 = self.list_of_joint_probs[t][1][1]
            p_b_3 = self.list_of_joint_probs[t][2][1]
            p_c_1 = self.list_of_joint_probs[t][0][2]
            p_c_2 = self.list_of_joint_probs[t][1][2]
            p_c_3 = self.list_of_joint_probs[t][2][2]
            p_a = self.env_probabilities[t][0]
            p_b = self.env_probabilities[t][1]
            p_c = self.env_probabilities[t][2]
            p_1 = self.sys_probabilities[t][0]
            p_2 = self.sys_probabilities[t][1]
            p_3 = self.sys_probabilities[t][2]
            if p_a_1 == 0:
                p_a_1 = .00001
            if p_a_2 == 0:
                p_a_2 = .00001
            if p_a_3 == 0:
                p_a_3 = .00001
            if p_b_1 == 0:
                p_b_1 = .00001
            if p_b_2 == 0:
                p_b_2 = .00001
            if p_b_3 == 0:
                p_b_3 = .00001
            if p_c_1 == 0:
                p_c_1 = .00001
            if p_c_2 == 0:
                p_c_2 = .00001
            if p_c_3 == 0:
                p_c_3 = .00001
            if p_a_1 and p_a and p_1 != 0: #  was getting divide by 0 and ln 0 errors
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_a_1, mp.log(p_a_1 / mp.fmul(p_a, p_1)))) 
            if p_a_1 and p_a and p_2 != 0: 
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_a_2, mp.log(p_a_2 / mp.fmul(p_a, p_2))))
            if p_a_3 and p_a and p_3 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_a_3, mp.log(p_a_3 / mp.fmul(p_a, p_3))))
            if p_b_1 and p_b and p_1 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_b_1, mp.log(p_b_1 / mp.fmul(p_b, p_1))))
            if p_b_2 and p_b and p_2 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_b_2, mp.log(p_b_2 / mp.fmul(p_b, p_2))))
            if p_b_3 and p_b and p_3 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_b_3, mp.log(p_b_3 / mp.fmul(p_b, p_3))))
            if p_c_1 and p_c and p_1 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_c_1, mp.log(p_c_1 / mp.fmul(p_c, p_1))))
            if p_c_2 and p_c and p_2 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_c_2, mp.log(p_c_2 / mp.fmul(p_c, p_2))))
            if p_c_3 and p_c and p_3 != 0:
                mutual_info[t] = mutual_info[t] + float(mp.fmul(p_c_3, mp.log(p_c_3 / mp.fmul(p_c, p_3))))
        return mutual_info
        print("list of joint probs:")
        print(list_of_joint_probs)
        print("env probabilities:")
        print(env_probabilities)
        print("sys probabilities:")
        print(sys_probabilities)
        self.mutual_information()
        print("mutual info:")
        print(mutual_info)

    def conditional_entropy_xs(self, joint_prob: List[np.array], cond_term: np.array):
        """
        For H[X|S], where X env and S is sys, plug the joint probs and S (sys) probabilities into this method
        """
        for t in range(time_steps):
            if joint_prob[t][0][0] and cond_term[t][0] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][0][0]*np.log(cond_term[t][0]/joint_prob[t][0][0]))
            if joint_prob[t][1][0] and cond_term[t][1] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][1][0]*np.log(cond_term[t][1]/joint_prob[t][1][0]))
            if joint_prob[t][2][0] and cond_term[t][2] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][2][0]*np.log(cond_term[t][2]/joint_prob[t][2][0])) 
            if joint_prob[t][0][1] and cond_term[t][0] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][0][1]*np.log(cond_term[t][0]/joint_prob[t][0][1])) 
            if joint_prob[t][1][1] and cond_term[t][1] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][1][1]*np.log(cond_term[t][1]/joint_prob[t][1][1]))
            if joint_prob[t][2][1] and cond_term[t][2] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][2][1]*np.log(cond_term[t][2]/joint_prob[t][2][1])) 
            if joint_prob[t][0][2] and cond_term[t][0] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][0][2]*np.log(cond_term[t][0]/joint_prob[t][0][2])) 
            if joint_prob[t][1][2] and cond_term[t][1] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][1][2]*np.log(cond_term[t][1]/joint_prob[t][1][2])) 
            if joint_prob[t][2][2] and cond_term[t][2] != 0:
                self.h_yx[t] = self.h_yx[t] + (joint_prob[t][2][2]*np.log(cond_term[t][2]/joint_prob[t][2][2]))
        return self.h_yx

    def conditional_entropy_sx(self, joint_prob: List[np.array], cond_term: np.array):
        """
        For H[S|X], where S is sys and X is env, plug in env X probabilities"""
        for t in range(time_steps):
            if joint_prob[t][0][0] and cond_term[t][0] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][0][0]*np.log(cond_term[t][0]/joint_prob[t][0][0]))
            if joint_prob[t][1][0] and cond_term[t][0] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][1][0]*np.log(cond_term[t][0]/joint_prob[t][1][0])) 
            if joint_prob[t][2][0] and cond_term[t][0] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][2][0]*np.log(cond_term[t][0]/joint_prob[t][2][0])) 
            if joint_prob[t][0][1] and cond_term[t][1] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][0][1]*np.log(cond_term[t][1]/joint_prob[t][0][1])) 
            if joint_prob[t][1][1] and cond_term[t][1] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][1][1]*np.log(cond_term[t][1]/joint_prob[t][1][1])) 
            if joint_prob[t][2][1] and cond_term[t][1] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][2][1]*np.log(cond_term[t][1]/joint_prob[t][2][1])) 
            if joint_prob[t][0][2] and cond_term[t][2] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][0][2]*np.log(cond_term[t][2]/joint_prob[t][0][2])) 
            if joint_prob[t][1][2] and cond_term[t][2] != 0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][1][2]*np.log(cond_term[t][2]/joint_prob[t][1][2])) 
            if joint_prob[t][2][2] and cond_term[t][2] !=0:
                self.h_xy[t] = self.h_xy[t] + (joint_prob[t][2][2]*np.log(cond_term[t][2]/joint_prob[t][2][2]))
        return self.h_xy

    # for testing: I[X|Y] = H(X) - H(X|Y)
    # entropy(sys_probabilities)
    # conditional_entropy_xy(list_of_joint_probs, env_probabilities)

    def mutual_info_using_conditional_sx(self): # this is working, matches mi 
        """
        to verify that mutual info is correct
        """
        sys_entropy = self.entropy(self.sys_probabilities)
        cond_entropy_sx = self.conditional_entropy_sx(self.list_of_joint_probs, self.env_probabilities)
        for t in range(time_steps):
            self.mutual_info_c_sx[t] = sys_entropy[t] - cond_entropy_sx[t]
        return self.mutual_info_c_sx



    def mutual_info_using_conditional_xs(self): # this is not working 
        """
        to verify that mutual info is correct
        """
        env_entropy = self.entropy(self.env_probabilities)
        cond_entropy_xs = self.conditional_entropy_xs(self.list_of_joint_probs, self.sys_probabilities)
        for t in range(time_steps):
            self.mutual_info_c_xs[t] = env_entropy[t] - cond_entropy_xs[t]
        return self.mutual_info_c_xs

    def plot_mutual_information(self):
        mutual_info = self.mutual_information()
        """
        plots mutual information over time
        plots all 3 ways of getting mutual information- using formula and entropies
        """
        plt.xlabel("time step") 
        plt.ylabel("mutual information")
        plt.title("mutual information between system S and environment X")
        plt.plot(mutual_info, label = 'I[S,X]')
        plt.plot(self.mutual_info_c_xs, label = 'H[X] - H[X|S]')
        plt.plot(self.mutual_info_c_sx, label = 'H[S] - H[S|X]')
        legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
        legend.get_frame().set_facecolor('w')
        plt.show()

    # plot_state_probs()



    # need to modify way we're filling up data strutures for prediction I[s_t, x_t + 1]

    def calculate_probs_pred(self, sys_start_time : int): 
        """  
        fills up list_of_joint_probs list of matrices, with the following format:

        [[[P(S_t_X_t+1 = a_1), P(S_t_X_t+1 = b_1), P(S_t_X_t+1 = c_1)], [[P(S_t_X_t+2 = a_1), P(S_t_X_t+2 = b_1), P(S_t_X_t+2 = c_1)],      
        [P(S_t_X_t+1 = a_2), P(S_t_X_t+1 = b_2), P(S_t_X_t+1 = c_2)],  [P(S_t_X_t+2 = a_2), P(S_t_X_t+2 = b_2), P(S_t_X_t+2 = c_2)], 
        [P(S_t_X_t+1 = a_3), P(S_t_X_t+1 = b_3), P(S_t_X_t+1 = c_3)]], [P(S_t_X_t+2 = a_3), P(S_t_X_t+2 = b_3), P(S_t_X_t+2 = c_3)]], ... 
        """
        # memory, where sys and env are in parallel 
        a_1 = 0
        b_1 = 0
        c_1 = 0
        a_2 = 0
        b_2 = 0
        c_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
        list_of_joint_probs_index = 0
        n = len(list_of_env_sys_tuples)
        for t in range(sys_start_time, time_steps): # starts at sys_start, so first entry is s_t and x_t
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][sys_start_time] == 1:
                    # looking through associated environments to set up joint_probabilities_list
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_1 = a_1 + 1
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_1 = b_1 + 1
                    else:
                        c_1 = c_1 + 1
                elif list_of_env_sys_tuples[e][1][sys_start_time] == 2:
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_2 = a_2 + 1
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_2 = b_2 + 1
                    else:
                        c_2 = c_2 + 1
                else:
                    if list_of_env_sys_tuples[e][0][t] == 1:
                        a_3 = a_3 + 1
                    elif list_of_env_sys_tuples[e][0][t] == 2:
                        b_3 = b_3 + 1
                    else:
                        c_3 = c_3 + 1
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            self.list_of_joint_probs_pred[list_of_joint_probs_index] = prob_array
            # resetting the values 
            a_1 = 0
            b_1 = 0
            c_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0  
            list_of_joint_probs_index = list_of_joint_probs_index + 1 

    def I_pred(self):
        global I_pred_list
        I_pred_list = [0] * len(self.env_probabilities_pred)
        """
        calculates predictive information S_t has about X_t+1, X_t+1, etc 
        """
        self.calculate_probs_pred(800)
        for t in range(len(self.env_probabilities_pred)):
            p_a_1 = self.list_of_joint_probs_pred[t][0][0]
            p_a_2 = self.list_of_joint_probs_pred[t][1][0]
            p_a_3 = self.list_of_joint_probs_pred[t][2][0]
            p_b_1 = self.list_of_joint_probs_pred[t][0][1]
            p_b_2 = self.list_of_joint_probs_pred[t][1][1]
            p_b_3 = self.list_of_joint_probs_pred[t][2][1]
            p_c_1 = self.list_of_joint_probs_pred[t][0][2]
            p_c_2 = self.list_of_joint_probs_pred[t][1][2]
            p_c_3 = self.list_of_joint_probs_pred[t][2][2]
            p_a = self.env_probabilities_pred[t][0]
            p_b = self.env_probabilities_pred[t][1]
            p_c = self.env_probabilities_pred[t][2]
            p_1 = self.sys_probabilities_pred[0]
            p_2 = self.sys_probabilities_pred[1]
            p_3 = self.sys_probabilities_pred[2]
            if p_a_1 == 0:
                p_a_1 = .00001
            if p_a_2 == 0:
                p_a_2 = .00001
            if p_a_3 == 0:
                p_a_3 = .00001
            if p_b_1 == 0:
                p_b_1 = .00001
            if p_b_2 == 0:
                p_b_2 = .00001
            if p_b_3 == 0:
                p_b_3 = .00001
            if p_c_1 == 0:
                p_c_1 = .00001
            if p_c_2 == 0:
                p_c_2 = .00001
            if p_c_3 == 0:
                p_c_3 = .00001
            if p_a == 0:
                p_a = .00001
            if p_b == 0:
                p_b = .00001
            if p_c == 0:
                p_c = .00001
            if p_1 == 0:
                p_1 = .00001
            if p_2 == 0:
                p_2 = .00001
            if p_3 == 0:
                p_3 = .00001
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_a_1, mp.log(p_a_1 / mp.fmul(p_a, p_1)))) 
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_a_2, mp.log(p_a_2 / mp.fmul(p_a, p_2))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_a_3, mp.log(p_a_3 / mp.fmul(p_a, p_3))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_b_1, mp.log(p_b_1 / mp.fmul(p_b, p_1))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_b_2, mp.log(p_b_2 / mp.fmul(p_b, p_2))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_b_3, mp.log(p_b_3 / mp.fmul(p_b, p_3))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_c_1, mp.log(p_c_1 / mp.fmul(p_c, p_1))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_c_2, mp.log(p_c_2 / mp.fmul(p_c, p_2))))
            I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_c_3, mp.log(p_c_3 / mp.fmul(p_c, p_3))))
        return I_pred_list


    def plot_I_pred(self): # shift the x axis so it starts at 0 
        self.I_pred()
        # x = [x for x in range(1, len(env_probabilities_pred))]
        # string_x = [str(t) for t in x]
        # x_labels = ['t +' + string_x[t] for t in range(len(string_x))]
        x_axis = [x for x in range(len(self.env_probabilities_pred))]
        # plt.xticks(x_axis_list ,x_labels)
        plt.ylim(0, 1)
        plt.xlabel("time steps after t")
        plt.ylabel("predictive power")
        plt.title("predictive power of S_20")
        plt.plot(x_axis, I_pred_list)
        plt.show()

    def plot_non_predictive_info(self): 
        """
        not sure if this definition aligns with the paper 
        sum of I_mem(t) - sum of I_pred(t), where t = 0
        """
        non_predictive_info_list = [None] * len(self.mutual_info)
        for x in range(len(self.mutual_info)):
            non_predictive_info_list[x] = self.mutual_info[x] - I_pred_list[x]
        plt.ylim(0, 1)
        plt.xlabel("time steps after 0")
        plt.ylabel("info")
        plt.title("instantaneous non predictive info")
        plt.plot(non_predictive_info_list)
        plt.show()

    def plot_I_pred_zoomed_in(self):
        x_axis = [x for x in range(len(self.env_probabilities_pred))]
        x_axis_plus_one = [x + 1 for x in x_axis]
        # plt.xticks(x_axis_list ,x_labels)
        plt.ylim(0, 1)
        plt.xlabel("time steps after t")
        plt.ylabel("predictive power")
        plt.title("predictive power of S_20")
        plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        plt.plot(x_axis_plus_one[0:19], I_pred_list[1:20])
        plt.show()

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
        a_2 = 0
        b_2 = 0
        c_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
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
                    else:
                        c_1 = c_1 + 1
                elif list_of_env_sys_tuples[e][1][t] == 2:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_2 = a_2 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_2 = b_2 + 1
                    else:
                        c_2 = c_2 + 1
                else:
                    if list_of_env_sys_tuples[e][0][t+1] == 1:
                        a_3 = a_3 + 1
                    elif list_of_env_sys_tuples[e][0][t+1] == 2:
                        b_3 = b_3 + 1
                    else:
                        c_3 = c_3 + 1
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            self.list_of_joint_probs_pred_one_step[list_of_joint_probs_index] = prob_array
            # resetting the values 
            a_1 = 0
            b_1 = 0
            c_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0  
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
            p_b_1 = self.list_of_joint_probs_pred_one_step[t][0][1]
            p_b_2 = self.list_of_joint_probs_pred_one_step[t][1][1]
            p_b_3 = self.list_of_joint_probs_pred_one_step[t][2][1]
            p_c_1 = self.list_of_joint_probs_pred_one_step[t][0][2]
            p_c_2 = self.list_of_joint_probs_pred_one_step[t][1][2]
            p_c_3 = self.list_of_joint_probs_pred_one_step[t][2][2]
            p_a = self.env_probabilities[t+1][0]
            p_b = self.env_probabilities[t+1][1]
            p_c = self.env_probabilities[t+1][2]
            p_1 = self.sys_probabilities[t][0]
            p_2 = self.sys_probabilities[t][1]
            p_3 = self.sys_probabilities[t][2]
            if p_a_1 == 0:
                p_a_1 = .00001
            if p_a_2 == 0:
                p_a_2 = .00001
            if p_a_3 == 0:
                p_a_3 = .00001
            if p_b_1 == 0:
                p_b_1 = .00001
            if p_b_2 == 0:
                p_b_2 = .00001
            if p_b_3 == 0:
                p_b_3 = .00001
            if p_c_1 == 0:
                p_c_1 = .00001
            if p_c_2 == 0:
                p_c_2 = .00001
            if p_c_3 == 0:
                p_c_3 = .00001
            if p_a == 0:
                p_a = .00001
            if p_b == 0:
                p_b = .00001
            if p_c == 0:
                p_c = .00001
            if p_1 == 0:
                p_1 = .00001
            if p_2 == 0:
                p_2 = .00001
            if p_3 == 0:
                p_3 = .00001
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_1, mp.log(p_a_1 / mp.fmul(p_a, p_1)))) 
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_2, mp.log(p_a_2 / mp.fmul(p_a, p_2))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_a_3, mp.log(p_a_3 / mp.fmul(p_a, p_3))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_1, mp.log(p_b_1 / mp.fmul(p_b, p_1))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_2, mp.log(p_b_2 / mp.fmul(p_b, p_2))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_b_3, mp.log(p_b_3 / mp.fmul(p_b, p_3))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_1, mp.log(p_c_1 / mp.fmul(p_c, p_1))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_2, mp.log(p_c_2 / mp.fmul(p_c, p_2))))
            I_pred_list_one_step[t] = I_pred_list_one_step[t] + float(mp.fmul(p_c_3, mp.log(p_c_3 / mp.fmul(p_c, p_3))))
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
            if joint_prob[t][0][1] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][1]*np.log(cond_term[t][0]/joint_prob[t][0][1])) 
            if joint_prob[t][1][1] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][1]*np.log(cond_term[t][1]/joint_prob[t][1][1]))
            if joint_prob[t][2][1] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][1]*np.log(cond_term[t][2]/joint_prob[t][2][1])) 
            if joint_prob[t][0][2] and cond_term[t][0] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][0][2]*np.log(cond_term[t][0]/joint_prob[t][0][2])) 
            if joint_prob[t][1][2] and cond_term[t][1] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][1][2]*np.log(cond_term[t][1]/joint_prob[t][1][2])) 
            if joint_prob[t][2][2] and cond_term[t][2] != 0:
                self.h_xs_verify[t] = self.h_xs_verify[t] + (joint_prob[t][2][2]*np.log(cond_term[t][2]/joint_prob[t][2][2]))
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
        plt.ylim(0, 1)
        plt.xlabel("time step")
        plt.ylabel("predictive power I[s_t, x_t+1]")
        plt.title("predictive power of system about environment one step ahead I[s_t, x_t+1]")
        plt.plot(x_axis, I_pred_list_one_step, label = "using MI formula")
        plt.plot(x_axis, self.pred_one_step_using_conditional_xs(), label = "using conditional")
        legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
        legend.get_frame().set_facecolor('w')
        plt.show()

        """
        new stuff 
        trying I[st, sk]
        """
    
    def calc_I_pred_and_mem(self):
        """
        I[st, sk] and I[st, sh]
        """
        a_1 = 0
        b_1 = 0
        c_1 = 0
        a_2 = 0
        b_2 = 0
        c_2 = 0
        a_3 = 0
        b_3 = 0
        c_3 = 0
        list_of_joint_probs_index = 0
        n = len(list_of_env_sys_tuples)
        for t in range((self.sys_start_time - (time_steps - self.sys_start_time)), time_steps): 
            for e in range(n): # gives us number of ensembles
                if list_of_env_sys_tuples[e][1][self.sys_start_time] == 1:
                    # looking through associated environments to set up joint_probabilities_list
                    if list_of_env_sys_tuples[e][1][t] == 1:
                        a_1 = a_1 + 1
                    elif list_of_env_sys_tuples[e][1][t] == 2:
                        b_1 = b_1 + 1
                    else:
                        c_1 = c_1 + 1
                elif list_of_env_sys_tuples[e][1][self.sys_start_time] == 2:
                    if list_of_env_sys_tuples[e][1][t] == 1:
                        a_2 = a_2 + 1
                    elif list_of_env_sys_tuples[e][1][t] == 2:
                        b_2 = b_2 + 1
                    else:
                        c_2 = c_2 + 1
                else:
                    if list_of_env_sys_tuples[e][1][t] == 1:
                        a_3 = a_3 + 1
                    elif list_of_env_sys_tuples[e][1][t] == 2:
                        b_3 = b_3 + 1
                    else:
                        c_3 = c_3 + 1
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = a_1/n
            prob_array[0][1] = b_1/n
            prob_array[0][2] = c_1/n
            prob_array[1][0] = a_2/n
            prob_array[1][1] = b_2/n
            prob_array[1][2] = c_2/n
            prob_array[2][0] = a_3/n
            prob_array[2][1] = b_3/n
            prob_array[2][2] = c_3/n
            self.list_of_joint_probs_pred_and_mem[list_of_joint_probs_index] = prob_array
            # resetting the values 
            a_1 = 0
            b_1 = 0
            c_1 = 0
            a_2 = 0
            b_2 = 0
            c_2 = 0
            a_3 = 0
            b_3 = 0
            c_3 = 0  
            list_of_joint_probs_index = list_of_joint_probs_index + 1 

    def I_pred_and_mem(self):
        """
        calculates predictive information S_t has about X_t+1, X_t+1, etc 
        """
        self.calc_I_pred_and_mem()
        for t in range(len(self.env_probabilities_pred_and_mem)):
            p_a_1 = self.list_of_joint_probs_pred_and_mem[t][0][0]
            p_a_2 = self.list_of_joint_probs_pred_and_mem[t][1][0]
            p_a_3 = self.list_of_joint_probs_pred_and_mem[t][2][0]
            p_b_1 = self.list_of_joint_probs_pred_and_mem[t][0][1]
            p_b_2 = self.list_of_joint_probs_pred_and_mem[t][1][1]
            p_b_3 = self.list_of_joint_probs_pred_and_mem[t][2][1]
            p_c_1 = self.list_of_joint_probs_pred_and_mem[t][0][2]
            p_c_2 = self.list_of_joint_probs_pred_and_mem[t][1][2]
            p_c_3 = self.list_of_joint_probs_pred_and_mem[t][2][2]
            p_a = self.sys_probabilities_pred_and_mem[t][0]
            p_b = self.sys_probabilities_pred_and_mem[t][1]
            p_c = self.sys_probabilities_pred_and_mem[t][2]
            p_1 = self.sys_probabilities[self.sys_start_time][0]
            p_2 = self.sys_probabilities[self.sys_start_time][1]
            p_3 = self.sys_probabilities[self.sys_start_time][2]
            if p_a_1 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_a_1 * (np.log(p_a_1 / (float(p_a) * float(p_1)))))
            if p_a_2 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_a_2 * (np.log(p_a_2 / (float(p_a) * float(p_2)))))
            if p_a_3 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_a_3 * (np.log(p_a_3 / (float(p_a) * float(p_3)))))
            if p_b_1 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_b_1 * (np.log(p_b_1 / (float(p_b) * float(p_1)))))
            if p_b_2 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_b_2 * (np.log(p_b_2 / (float(p_b) * float(p_2)))))
            if p_b_3 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_b_3 * (np.log(p_b_3 / (float(p_b) * float(p_3)))))
            if p_c_1 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_c_1 * (np.log(p_c_1 / (float(p_c) * float(p_1)))))
            if p_c_2 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_c_2 * (np.log(p_c_2 / (float(p_c) * float(p_2)))))
            if p_c_3 != 0:
                self.I_pred_list_and_mem[t] = self.I_pred_list_and_mem[t] + (p_c_3 * (np.log(p_c_3 / (float(p_c) * float(p_3)))))
        return self.I_pred_list_and_mem
    

    def conditional_entropy_ss(self, cond_term: list):
        """
        modified conditional entropy, for finding H[S_k|S_t]. plug in env probabilities at time slice t
        """
        self.calc_I_pred_and_mem()
        for t in range(len(self.h_sk_st)):
            if self.list_of_joint_probs_pred_and_mem[t][0][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][0][0], mp.log(cond_term[0]/self.list_of_joint_probs_pred_and_mem[t][0][0])))
            if self.list_of_joint_probs_pred_and_mem[t][1][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][1][0], mp.log(cond_term[1]/self.list_of_joint_probs_pred_and_mem[t][1][0])))
            if self.list_of_joint_probs_pred_and_mem[t][2][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][2][0], mp.log(cond_term[2]/self.list_of_joint_probs_pred_and_mem[t][2][0])))
            if self.list_of_joint_probs_pred_and_mem[t][0][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][0][1], mp.log(cond_term[0]/self.list_of_joint_probs_pred_and_mem[t][0][1])))
            if self.list_of_joint_probs_pred_and_mem[t][1][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][1][1], np.log(cond_term[1]/self.list_of_joint_probs_pred_and_mem[t][1][1])))
            if self.list_of_joint_probs_pred_and_mem[t][2][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][2][1], np.log(cond_term[2]/self.list_of_joint_probs_pred_and_mem[t][2][1])))
            if self.list_of_joint_probs_pred_and_mem[t][0][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][0][2], np.log(cond_term[0]/self.list_of_joint_probs_pred_and_mem[t][0][2])))
            if self.list_of_joint_probs_pred_and_mem[t][1][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][1][2], np.log(cond_term[1]/self.list_of_joint_probs_pred_and_mem[t][1][2])))
            if self.list_of_joint_probs_pred_and_mem[t][2][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_pred_and_mem[t][2][2], np.log(cond_term[2]/self.list_of_joint_probs_pred_and_mem[t][2][2])))

    def conditional_entropy_mutual_info_verify_ss(self):
        """
        verifying that mutual info is correct using conditional entropy
        """
        self.entropy(self.sys_probabilities)
        self.conditional_entropy_ss(self.sys_probabilities[self.sys_start_time])
        for t in range((time_steps - self.sys_start_time) * 2):
            self.conditional_entropy_mutual_info_verification[t] = self.h[t] - self.h_sk_st[t]

    def plot_I_pred_and_mem(self): # shift the x axis so it starts at 0 
        self.I_pred_and_mem()
        self.conditional_entropy_mutual_info_verify_ss()
        # x = [x for x in range(1, len(env_probabilities_pred))]
        # string_x = [str(t) for t in x]
        # x_labels = ['t +' + string_x[t] for t in range(len(string_x))]
        x_axis_past = [-x for x in range(50)]
        x_axis_past = [(x - 1) for x in x_axis_past]
        x_axis_past.reverse()
        x_axis_future = [x for x in range(50)]
        # plt.xticks(x_axis_list ,x_labels)
        plt.ylim(0, 2)
        plt.xlabel("time steps after t")
        plt.ylabel("predictive power")
        plt.title("predictive power and memory of S_800")
        plt.plot(x_axis_past, self.I_pred_list_and_mem[150: 200], label = "mutual info formula")
        plt.plot(x_axis_future, self.I_pred_list_and_mem[200: 250], label = "mutual info formula")
        plt.plot(x_axis_past, self.conditional_entropy_mutual_info_verification[150:200], label = "conditional entropy formula")
        plt.plot(x_axis_future, self.conditional_entropy_mutual_info_verification[200:250], label = "conditional entropy formula")
        plt.legend()
        plt.show()

"""testing"""
# test_1 = Markov_three(p_transition_env = np.array([[.05, .9, .05],
#                               [.05, .05, .9],
#                            [.9, .05, .05]]), p_t1 = np.array([[.05, .9, .05],
#          [.05, .9, .05],
#          [.05, .9, .05]]), p_t2 = np.array([[.05, .05, .9],
#          [.05, .05, .9],
#          [.05, .05, 9]]), p_t3 = np.array([[.9, .05, .05],
#          [.9, .05, .05],
#          [.9, .05, .05]]))


# test_1.generate_ensemble(500)
# test_1.calculate_probs()
# test_1.plot_state_probs()
# test_1.I_pred_and_mem()
# test_1.plot_I_pred_and_mem()

# test_1.calculate_probs()
# test_1.mutual_info_using_conditional_xs()
# test_1.mutual_info_using_conditional_sx()
# test_1.plot_mutual_information()
# test_1.plot_state_probs()
# test_1.plot_mutual_information()
# test_1.calculate_probs_pred(20)
# test_1.plot_I_pred()
# test_1.plot_I_pred_zoomed_in()
# test_1.plot_I_pred_one_step()






# for entropy rate for system, what is the stationary distribution since there are 3 transition matrices?








# can calculate non equilibrium free energy 



# heat work, then dissapation
# then information theory - quantifiy mutual information, memory, predictive power 


# dissapation connects with infromation theory 
# infromation between state of system and driving signal

# learning involves mutual information between states of envrionemnt and system are increasing, and some aspect of memory and prediction

# not clear if there is an increase in mutual infromation yet?

# faster process = higher dissipation so less memory 
