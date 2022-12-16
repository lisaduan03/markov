from struct import calcsize
from time import time
from entropy_rate import generate_db_matrices, generate_random_matrix
from three_state_simulation import stationary_distribution, markov_sequence
from three_state_simulation import markov_sequence
import numpy as np
from matplotlib import pyplot as plt 
import mpmath as mp


class Environment:
    def __init__(self, matrix):
        self.matrix = matrix

    """
    first, using build_environment to build a markov chain representing the environment 
    """
    sys_start_time = 50 # this can cange 
    time_steps = 100
    number_of_ensembles = 500 # # of ensembles 
    list_of_env_chains = [[None, None, None] for row in range(number_of_ensembles)]
    env_probs = [[None, None, None] for row in range(time_steps)]
    joint_prob = [[None, None, None], [None, None, None], [None, None, None]]
    t_probs = [None, None, None]
    h_yx = 0 
    I_pred_env_list = []
    list_of_joint_probs_t_k = [None] * (time_steps - sys_start_time)
    env_probabilities_pred = env_probs[sys_start_time:]



    def build_environment(self):
        p_init_env = [.05, .05, .9] # this can change
        global env_states
        env_states = markov_sequence(p_init_env, self.matrix, self.time_steps)
        env_states = [x + 1 for x in env_states]
        return env_states

    def gen_ensemble(self):
        for x in range(self.number_of_ensembles):
            self.list_of_env_chains[x] = self.build_environment()
        return self.list_of_env_chains

    def calc_env_probs(self):
        """
        a modified version of calc probs that just fills up probabilities for given system

        joint probs:
        [[P(X_0 = 1), P(X_0 = 2), P(X_0 = 3)], 
        [P(X_1 = 1), P(X_1 = 1), P(X_1 = 3)], 
        ...
        ]
        """
        t_1 = 0
        t_2 = 0
        t_3 = 0
        for x in range(self.time_steps):
            for l in range(self.number_of_ensembles):
                if self.list_of_env_chains[l][x] == 1:
                    t_1 = t_1 + 1
                if self.list_of_env_chains[l][x] == 2:
                    t_2 = t_2 + 1
                if self.list_of_env_chains[l][x] == 3:
                    t_3 = t_3 + 1
            self.env_probs[x][0] = t_1/self.number_of_ensembles
            self.env_probs[x][1] = t_2/self.number_of_ensembles
            self.env_probs[x][2] = t_3/self.number_of_ensembles
            t_1 = 0
            t_2 = 0
            t_3 = 0

    def calc_joint_prob(self, time):
        # first t, then t+1
        t_1_1 = 0
        t_1_2 = 0
        t_1_3 = 0
        t_2_1 = 0
        t_2_2 = 0
        t_2_3 = 0
        t_3_1 = 0
        t_3_2 = 0
        t_3_3 = 0
        for l in range(self.number_of_ensembles):
            if self.list_of_env_chains[l][time] == 1:
                if self.list_of_env_chains[l][time + 1] == 1:
                    t_1_1 = t_1_1 + 1
                if self.list_of_env_chains[l][time + 1] == 2:
                    t_1_2 = t_1_2 +1
                if self.list_of_env_chains[l][time + 1] == 3:
                    t_1_3 = t_1_3 +1
            if self.list_of_env_chains[l][time] == 2:
                if self.list_of_env_chains[l][time+ 1] == 1:
                    t_2_1 = t_2_1 + 1
                if self.list_of_env_chains[l][time+1] == 2:
                    t_2_2 = t_2_2 +1
                if self.list_of_env_chains[l][time+1] == 3:
                    t_2_3 = t_2_3 +1
            if self.list_of_env_chains[l][time] == 3:
                if self.list_of_env_chains[l][time+ 1] == 1:
                    t_3_1 = t_3_1 + 1
                if self.list_of_env_chains[l][time+1] == 2:
                    t_3_2 = t_3_2 +1
                if self.list_of_env_chains[l][time+1] == 3:
                    t_3_3 = t_3_3 +1
        self.joint_prob[0][0] = t_1_1/self.number_of_ensembles
        self.joint_prob[0][1] = t_1_2/self.number_of_ensembles
        self.joint_prob[0][2] = t_1_3/self.number_of_ensembles
        self.joint_prob[1][0] = t_2_1/self.number_of_ensembles
        self.joint_prob[1][1] = t_2_2/self.number_of_ensembles
        self.joint_prob[1][2] = t_2_3/self.number_of_ensembles
        self.joint_prob[2][0] = t_3_1/self.number_of_ensembles
        self.joint_prob[2][1] = t_3_2/self.number_of_ensembles
        self.joint_prob[2][2] = t_3_3/self.number_of_ensembles


    def plot_state_probs(self):
            """
            first gathers probabilities of being at state 1, 2, or 3 at given time step,
            then plots the system probabilities
            """
            data_1 = []
            data_2 = []
            data_3 = []
            for x in range(self.time_steps):
                data_1.append(self.env_probs[x][0])
                data_2.append(self.env_probs[x][1])
                data_3.append(self.env_probs[x][2])
            plt.plot(data_1, label = 'state 1')
            plt.plot(data_2, label = 'state 2')
            plt.plot(data_3, label = 'state 3')
            legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
            legend.get_frame().set_facecolor('w')
            plt.xlabel("time step")
            plt.ylabel("probability")
            plt.ylim(0, 1)
            plt.title("probability of being at state 1, 2, or 3")
            plt.show()

    def entropy(self, time):
        """
        modified. shannon entropy at time t
        """
        h = 0
        if self.env_probs[time][0] == 0:
            self.env_probs[time][0] = .000001
        if self.env_probs[time][1] == 0:
            self.env_probs[time][1] = .000001
        if self.env_probs[time][2] == 0:
            self.env_probs[time][2] = .000001
        h = -((self.env_probs[time][0]*np.log(self.env_probs[time][0])) \
            + (self.env_probs[time][1]*np.log(self.env_probs[time][1])) \
                + (self.env_probs[time][2]*np.log(self.env_probs[time][2])))
        return h

    def conditional_entropy(self, cond_term: list):
        """
        modified condition entropy, for finding H[S_t+1|S_t]
        """
        h_t_1_t = 0
        for i in range(3):
            if cond_term[i] == 0:
                cond_term[i] = .000001
        for x in range(3):
            for y in range(3):
                if self.joint_prob[x][y] == 0:
                    self.joint_prob[x][y] = .000001
        h_t_1_t = (self.joint_prob[0][0]*np.log(cond_term[0]/self.joint_prob[0][0])) \
                + (self.joint_prob[1][0]*np.log(cond_term[1]/self.joint_prob[1][0])) \
                + (self.joint_prob[2][0]*np.log(cond_term[2]/self.joint_prob[2][0])) \
                + (self.joint_prob[0][1]*np.log(cond_term[0]/self.joint_prob[0][1])) \
                + (self.joint_prob[1][1]*np.log(cond_term[1]/self.joint_prob[1][1])) \
                + (self.joint_prob[2][1]*np.log(cond_term[2]/self.joint_prob[2][1])) \
                + (self.joint_prob[0][2]*np.log(cond_term[0]/self.joint_prob[0][2])) \
                + (self.joint_prob[1][2]*np.log(cond_term[1]/self.joint_prob[1][2])) \
                + (self.joint_prob[2][2]*np.log(cond_term[2]/self.joint_prob[2][2])) 
        return h_t_1_t

    def cw_to_ccw_ratio(self):
        cw = self.matrix[0][1] * self.matrix[1][2] * self.matrix[2][0]
        ccw = self.matrix[0][2] * self.matrix[2][1] * self.matrix[1][0]
        ratio = cw/ccw
        return ratio

    def stored_information(self):
        # right now assume none of matrix entries are 0
        # double check that this is sum of all i and j even when i=j, 9 combos not 6 
        steady_state = stationary_distribution(self.matrix)
        stored_info = (steady_state[0] * self.matrix[0][0] * np.log(self.matrix[0][0]/steady_state[0])) \
                    + (steady_state[0] * self.matrix[0][1] * np.log(self.matrix[0][1]/steady_state[1])) \
                    + (steady_state[0] * self.matrix[0][2] * np.log(self.matrix[0][2]/steady_state[2])) \
                    + (steady_state[1] * self.matrix[1][0] * np.log(self.matrix[1][0]/steady_state[0])) \
                    + (steady_state[1] * self.matrix[1][1] * np.log(self.matrix[1][1]/steady_state[1])) \
                    + (steady_state[1] * self.matrix[1][2] * np.log(self.matrix[1][2]/steady_state[2])) \
                    + (steady_state[2] * self.matrix[2][0] * np.log(self.matrix[2][0]/steady_state[0])) \
                    + (steady_state[2] * self.matrix[2][1] * np.log(self.matrix[2][1]/steady_state[1])) \
                    + (steady_state[2] * self.matrix[2][2] * np.log(self.matrix[2][2]/steady_state[2]))
        return stored_info  

    def calc_mutual_info(self, time, next_time):
        """
        to confirm that stored info and entropy/cond entropy are correct
        I[S_t, S_t+1]
        """
        time_prob = self.env_probs[time]
        time_plus_one_prob = self.env_probs[time + 1]
        mutual_info = (self.joint_prob[0][0] * np.log(self.joint_prob[0][0]/(time_prob[0]*time_plus_one_prob[0])) \
                    + (self.joint_prob[0][1] * np.log(self.joint_prob[0][1]/(time_prob[0]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[0][2] * np.log(self.joint_prob[0][2]/(time_prob[0]*time_plus_one_prob[2]))) \
                    + (self.joint_prob[1][0] * np.log(self.joint_prob[1][0]/(time_prob[1]*time_plus_one_prob[0]))) \
                    + (self.joint_prob[1][1] * np.log(self.joint_prob[1][1]/(time_prob[1]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[1][2] * np.log(self.joint_prob[1][2]/(time_prob[1]*time_plus_one_prob[2]))) \
                    + (self.joint_prob[2][0] * np.log(self.joint_prob[2][0]/(time_prob[2]*time_plus_one_prob[0]))) \
                    + (self.joint_prob[2][1] * np.log(self.joint_prob[2][1]/(time_prob[2]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[2][2] * np.log(self.joint_prob[2][2]/(time_prob[2]*time_plus_one_prob[2]))))
        return mutual_info
    
    def entropy_rate_env(self, matrix):
        global stationary_dist_env
        stationary_dist_env = stationary_distribution(matrix)
        entropy_rate_env_val = 0
        # if (matrix[0][0] == 0):
        #     matrix[0][0] = .0000001 # make it arbitrarily small so entropy prod rate blows up
        # entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][0]) - (stationary_dist_env[0] * matrix[0][0]))* np.log((stationary_dist_env[0] * matrix[0][0])/(stationary_dist_env[0] * matrix[0][0]))
        if (matrix[0][1] == 0):
            matrix[0][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][0]) - (stationary_dist_env[0] * matrix[0][1]))* np.log((stationary_dist_env[1] * matrix[1][0])/(stationary_dist_env[0] * matrix[0][1]))
        if (matrix[0][2] == 0):
            matrix[0][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][0]) - (stationary_dist_env[0] * matrix[0][2]))* np.log((stationary_dist_env[2] * matrix[2][0])/(stationary_dist_env[0] * matrix[0][2]))
        if (matrix[1][0] == 0):
            matrix[1][0] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][1]) - (stationary_dist_env[1] * matrix[1][0]))* np.log((stationary_dist_env[0] * matrix[0][1])/(stationary_dist_env[1] * matrix[1][0]))
        # if (matrix[1][1] == 0):
        #     matrix[1][1] = .0000001
        #entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][1]) - (stationary_dist_env[1] * matrix[1][1]))* np.log((stationary_dist_env[1] * matrix[1][1])/(stationary_dist_env[1] * matrix[1][1]))
        if (matrix[1][2] == 0):
            matrix[1][2] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][1]) - (stationary_dist_env[1] * matrix[1][2]))* np.log((stationary_dist_env[2] * matrix[2][1])/(stationary_dist_env[1] * matrix[1][2]))
        if (matrix[2][0] == 0):
            matrix[2][0] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][2]) - (stationary_dist_env[2] * matrix[2][0]))* np.log((stationary_dist_env[0] * matrix[0][2])/(stationary_dist_env[2] * matrix[2][0]))
        if (matrix[2][1] == 0):
            matrix[2][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][2]) - (stationary_dist_env[2] * matrix[2][1]))* np.log((stationary_dist_env[1] * matrix[1][2])/(stationary_dist_env[2] * matrix[2][1]))
        # if (matrix[2][2] == 0):
        #     matrix[2][2] = .0000001
        #entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][2]) - (stationary_dist_env[2] * matrix[2][2]))* np.log((stationary_dist_env[2] * matrix[2][2])/(stationary_dist_env[2] * matrix[2][2]))
        entropy_rate_env_val = entropy_rate_env_val * (1/2)
        return entropy_rate_env_val

    # def I_pred_env(self, time):
    #     for x in range(time, self.time_steps):
    #         self.I_pred_env_list.append(self.calc_mutual_info(time, x))
    #     return self.I_pred_env_list
        

    # def calculate_probs_pred_env(self, sys_start_time : int): 
    #     """
    #     fills up list_of_joint_probs list of matrices, with the following format:
    # [[[P(S_t_X_t+1 = a_1), P(S_t_X_t+1 = b_1), P(S_t_X_t+1 = c_1)], [[P(S_t_X_t+2 = a_1), P(S_t_X_t+2 = b_1), P(S_t_X_t+2 = c_1)],      
    # [P(S_t_X_t+1 = a_2), P(S_t_X_t+1 = b_2), P(S_t_X_t+1 = c_2)],  [P(S_t_X_t+2 = a_2), P(S_t_X_t+2 = b_2), P(S_t_X_t+2 = c_2)], 
    # [P(S_t_X_t+1 = a_3), P(S_t_X_t+1 = b_3), P(S_t_X_t+1 = c_3)]], [P(S_t_X_t+2 = a_3), P(S_t_X_t+2 = b_3), P(S_t_X_t+2 = c_3)]], ... 
    #     """
    #     one_1 = 0
    #     two_1 = 0
    #     three_1 = 0
    #     one_2 = 0
    #     two_2 = 0
    #     three_2 = 0
    #     one_3 = 0
    #     two_3 = 0
    #     three_3 = 0
    #     list_of_joint_probs_index = 0
    #     n = len(self.list_of_env_chains)
    #     for t in range(sys_start_time, self.time_steps): # starts at sys_start, so first entry is s_t and x_t
    #         for e in range(n): # gives us number of ensembles
    #             if self.list_of_env_chains[e][1][sys_start_time] == 1:
    #                 # looking through associated environments to set up joint_probabilities_list
    #                 if self.list_of_env_chains[e][0][t] == 1:
    #                     one_1 = one_1 + 1
    #                 elif self.list_of_env_chains[e][0][t] == 2:
    #                     two_1 = two_1 + 1
    #                 else:
    #                     three_1 = three_1 + 1
    #             elif self.list_of_env_chains[e][1][sys_start_time] == 2:
    #                 if self.list_of_env_chains[e][0][t] == 1:
    #                     one_2 = one_2 + 1
    #                 elif self.list_of_env_chains[e][0][t] == 2:
    #                     two_2 = two_2 + 1
    #                 else:
    #                     three_2 = three_2 + 1
    #             else:
    #                 if self.list_of_env_chains[e][0][t] == 1:
    #                     one_3 = one_3 + 1
    #                 elif self.list_of_env_chains[e][0][t] == 2:
    #                     two_3 = two_3 + 1
    #                 else:
    #                     three_3 = three_3 + 1
    #         # filling the data structures
    #         prob_array = [[None, None, None], [None, None, None], [None, None, None]]
    #         prob_array[0][0] = one_1/n
    #         prob_array[0][1] = two_1/n
    #         prob_array[0][2] = three_1/n
    #         prob_array[1][0] = one_2/n
    #         prob_array[1][1] = two_2/n
    #         prob_array[1][2] = three_2/n
    #         prob_array[2][0] = one_3/n
    #         prob_array[2][1] = two_3/n
    #         prob_array[2][2] = three_3/n
    #         self.list_of_joint_probs_pred[list_of_joint_probs_index] = prob_array
    #         # resetting the values 
    #         one_1 = 0
    #         two_1 = 0
    #         three_1 = 0
    #         one_2 = 0
    #         two_2 = 0
    #         three_2 = 0
    #         one_3 = 0
    #         two_3 = 0
    #         three_3 = 0
    #         list_of_joint_probs_index = list_of_joint_probs_index + 1 

    # def I_pred_env(self, start_time):
    #     env_probabilities_pred = self.env_probs[start_time:]
    #     global I_pred_list
    #     I_pred_list = [0] * len(self.env_probabilities_pred)
    #     """
    #     calculates predictive information S_t has about X_t+1, X_t+1, etc 
    #     """
    #     self.calculate_probs_pred(20)
    #     for t in range(len(self.env_probabilities_pred)):
    #         p_one_1 = self.list_of_joint_probs_pred[t][0][0]
    #         p_one_2 = self.list_of_joint_probs_pred[t][1][0]
    #         p_one_3 = self.list_of_joint_probs_pred[t][2][0]
    #         p_two_1 = self.list_of_joint_probs_pred[t][0][1]
    #         p_two_2 = self.list_of_joint_probs_pred[t][1][1]
    #         p_two_3 = self.list_of_joint_probs_pred[t][2][1]
    #         p_three_1 = self.list_of_joint_probs_pred[t][0][2]
    #         p_three_2 = self.list_of_joint_probs_pred[t][1][2]
    #         p_three_3 = self.list_of_joint_probs_pred[t][2][2]
    #         p_one = self.env_probs_pred[t][0]
    #         p_two = self.env_probs_pred[t][1]
    #         p_three = self.env_probs_pred[t][2]
    #         p_1 = self.env_probs_pred[start_time][0]
    #         p_2 = self.sys_probs_pred[start_time][1]
    #         p_3 = self.sys_probs_pred[start_time][2]
    #         if p_one_1 == 0:
    #             p_one_1 = .00001
    #         if p_one_2 == 0:
    #             p_one_2 = .00001
    #         if p_one_3 == 0:
    #             p_one_3 = .00001
    #         if p_two_1 == 0:
    #             p_two_1 = .00001
    #         if p_two_2 == 0:
    #             p_two_2 = .00001
    #         if p_two_3 == 0:
    #             p_two_3 = .00001
    #         if p_three_1 == 0:
    #             p_three_1 = .00001
    #         if p_three_2 == 0:
    #             p_three_2 = .00001
    #         if p_three_3 == 0:
    #             p_three_3 = .00001
    #         if p_a == 0:
    #             p_a = .00001
    #         if p_b == 0:
    #             p_b = .00001
    #         if p_c == 0:
    #             p_c = .00001
    #         if p_1 == 0:
    #             p_1 = .00001
    #         if p_2 == 0:
    #             p_2 = .00001
    #         if p_3 == 0:
    #             p_3 = .00001
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_one_1, mp.log(p_one_1 / mp.fmul(p_a, p_1)))) 
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_one_2, mp.log(p_one_2 / mp.fmul(p_a, p_2))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_one_3, mp.log(p_one_3 / mp.fmul(p_a, p_3))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_two_1, mp.log(p_two_1 / mp.fmul(p_b, p_1))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_two_2, mp.log(p_two_2 / mp.fmul(p_b, p_2))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_two_3, mp.log(p_two_3 / mp.fmul(p_b, p_3))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_one_3, mp.log(p_one_3 / mp.fmul(p_c, p_1))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_three_2, mp.log(p_three_2 / mp.fmul(p_c, p_2))))
    #         I_pred_list[t] = I_pred_list[t] + float(mp.fmul(p_three_3, mp.log(p_three_3 / mp.fmul(p_c, p_3))))
    #     return I_pred_list
        
    def calculate_probs_pred(self, sys_start_time : int): 
        self.gen_ensemble()
        self.calc_env_probs()
        """  
        fills up list_of_joint_probs_t_k list of matrices, with the following format:

        [[[P(X_t_X_t+1 = a_1), P(X_t_X_t+1 = b_1), P(X_t_X_t+1 = c_1)], [[P(X_t_X_t+2 = a_1), P(S_t_X_t+2 = b_1), P(S_t_X_t+2 = c_1)],      
        [P(X_t_X_t+1 = a_2), P(X_t_X_t+1 = b_2), P(X_t_X_t+1 = c_2)],  [P(X_t_X_t+2 = a_2), P(S_t_X_t+2 = b_2), P(S_t_X_t+2 = c_2)], 
        [P(X_t_X_t+1 = a_3), P(X_t_X_t+1 = b_3), P(X_t_X_t+1 = c_3)]], [P(S_t_X_t+2 = a_3), P(S_t_X_t+2 = b_3), P(S_t_X_t+2 = c_3)]], ... 
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
        for t in range(sys_start_time, self.time_steps): # starts at sys_start, so first entry is s_t and x_t
            for e in range(self.number_of_ensembles): # gives us number of ensembles
                if self.list_of_env_chains[e][sys_start_time] == 1:
                    # looking through associated environments to set up joint_probabilities_list
                    if self.list_of_env_chains[e][t] == 1:
                        a_1 = a_1 + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        b_1 = b_1 + 1
                    else:
                        c_1 = c_1 + 1
                elif self.list_of_env_chains[e][sys_start_time] == 2:
                    if self.list_of_env_chains[e][t] == 1:
                        a_2 = a_2 + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        b_2 = b_2 + 1
                    else:
                        c_2 = c_2 + 1
                else:
                    if self.list_of_env_chains[e][t] == 1:
                        a_3 = a_3 + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        b_3 = b_3 + 1
                    else:
                        c_3 = c_3 + 1
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = a_1/self.number_of_ensembles
            prob_array[0][1] = b_1/self.number_of_ensembles
            prob_array[0][2] = c_1/self.number_of_ensembles
            prob_array[1][0] = a_2/self.number_of_ensembles
            prob_array[1][1] = b_2/self.number_of_ensembles
            prob_array[1][2] = c_2/self.number_of_ensembles
            prob_array[2][0] = a_3/self.number_of_ensembles
            prob_array[2][1] = b_3/self.number_of_ensembles
            prob_array[2][2] = c_3/self.number_of_ensembles
            self.list_of_joint_probs_t_k[list_of_joint_probs_index] = prob_array
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
        I_pred_list = [0] * (self.time_steps - self.sys_start_time)
        """
        calculates predictive information S_t has about X_t+1, X_t+1, etc 
        """
        self.calculate_probs_pred(50)
        for t in range(self.time_steps- self.sys_start_time): # is this correct
            p_a_1 = self.list_of_joint_probs_t_k[t][0][0]
            p_a_2 = self.list_of_joint_probs_t_k[t][1][0]
            p_a_3 = self.list_of_joint_probs_t_k[t][2][0]
            p_b_1 = self.list_of_joint_probs_t_k[t][0][1]
            p_b_2 = self.list_of_joint_probs_t_k[t][1][1]
            p_b_3 = self.list_of_joint_probs_t_k[t][2][1]
            p_c_1 = self.list_of_joint_probs_t_k[t][0][2]
            p_c_2 = self.list_of_joint_probs_t_k[t][1][2]
            p_c_3 = self.list_of_joint_probs_t_k[t][2][2]
            # these are at time k 
            p_a = self.env_probabilities_pred[t][0]
            p_b = self.env_probabilities_pred[t][1]
            p_c = self.env_probabilities_pred[t][2]
            # these are at time t
            p_1 = self.env_probs[self.sys_start_time][0]
            p_2 = self.env_probs[self.sys_start_time][1]
            p_3 = self.env_probs[self.sys_start_time][2]
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
# testing
# in detailed balance
# print("broken DB (by hand, probabilities stay in place):")
# broken_db_matrix_6 = Environment(np.array([[.9, .05, .05],
#                                       [.05, .9, .05],
#                                       [.05, .05, .9]]))
# print(broken_db_matrix_6.matrix)
# broken_db_matrix_6.calc_env_probs(broken_db_matrix_6.gen_ensemble(broken_db_matrix_6.matrix))
# broken_db_matrix_6.plot_state_probs()
# joint_probs = broken_db_matrix_6.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_6.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_6_cond_term = broken_db_matrix_6.env_probs[950]
# print(broken_db_matrix_6.conditional_entropy(broken_db_matrix_6_cond_term))
# print("stored information:")
# print(broken_db_matrix_6.stored_information())
# print("mutual info:")
# print(broken_db_matrix_6.calc_mutual_info(950))

# print("uniform in DB:")
# db_matrix_uniform = System(np.array([[1/3, 1/3, 1/3],
#                               [1/3, 1/3, 1/3],
#                                [1/3, 1/3, 1/3]]))
# print(db_matrix_uniform.matrix)
# db_matrix_uniform.calc_env_probs(db_matrix_uniform.gen_ensemble(db_matrix_uniform.matrix))
# db_matrix_uniform.plot_state_probs()
# db_matrix_uniform.calc_joint_prob(950)
# print("entropy at time 951:")
# print(db_matrix_uniform.entropy(951))
# print("conditional entropy at 951 | 950:")
# db_matrix_cond_term = db_matrix_uniform.env_probs[950]
# print(db_matrix_uniform.conditional_entropy(db_matrix_cond_term))
# print("stored information:")
# print(db_matrix_uniform.stored_information(db_matrix_uniform.matrix))
# print("mutual info:")
# print(db_matrix_uniform.calc_mutual_info(950))

# print("in DB:")
# db_matrix = Environment(np.array([[.2, .4, .4],
#                               [.1, .6, .3],
#                                [.2, .6, .2]]))
# print(db_matrix.matrix)
# db_matrix.calc_env_probs(db_matrix.gen_ensemble(db_matrix.matrix))
# db_matrix.plot_state_probs()
# db_matrix.calc_joint_prob(950)
# print("entropy at time 951:")
# print(db_matrix.entropy(951))
# print("conditional entropy at 951 | 950:")
# db_matrix_cond_term = db_matrix.env_probs[950]
# print(db_matrix.conditional_entropy(db_matrix_cond_term))
# print("stored information:")
# print(db_matrix.stored_information())
# print("mutual info:")
# print(db_matrix.calc_mutual_info(950))

# print("broken DB:")
# random_broken_db_matrix = System(generate_random_matrix())
# print(random_broken_db_matrix.matrix)
# random_broken_db_matrix.calc_env_probs(random_broken_db_matrix.gen_ensemble(random_broken_db_matrix.matrix))
# random_broken_db_matrix.plot_state_probs()
# joint_probs = random_broken_db_matrix.calc_joint_prob(950)
# print("entropy at time 951:")
# print(random_broken_db_matrix.entropy(951))
# print("conditional entropy at 951 | 950:")
# random_broken_db_matrix_cond_term = random_broken_db_matrix.env_probs[950]
# print(random_broken_db_matrix.conditional_entropy(random_broken_db_matrix_cond_term))
# print("stored information:")
# print(random_broken_db_matrix.stored_information(random_broken_db_matrix.matrix))
# print("mutual info:")
# print(random_broken_db_matrix.calc_mutual_info(950))

# print("broken DB (by hand):")
# broken_db_matrix_1 = System(np.array([[.05, .9, .05],
#                                       [.05, .05, .9],
#                                       [.9, .05, .05]]))
# print(broken_db_matrix_1.matrix)
# broken_db_matrix_1.calc_env_probs(broken_db_matrix_1.gen_ensemble(broken_db_matrix_1.matrix))
# broken_db_matrix_1.plot_state_probs()
# joint_probs = broken_db_matrix_1.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_1.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_1_cond_term = broken_db_matrix_1.env_probs[950]
# print(broken_db_matrix_1.conditional_entropy(broken_db_matrix_1_cond_term))
# print("stored information:")
# print(broken_db_matrix_1.stored_information(broken_db_matrix_1.matrix))
# print("mutual info:")
# print(broken_db_matrix_1.calc_mutual_info(950))

# print("broken DB (by hand, more broken):")
# broken_db_matrix_2 = Environment(np.array([[.005, .99, .005],
#                                       [.005, .005, .99],
#                                       [.99, .005, .005]]))
# print(broken_db_matrix_2.matrix)
# broken_db_matrix_2.calc_env_probs(broken_db_matrix_2.gen_ensemble(broken_db_matrix_2.matrix))
# broken_db_matrix_2.plot_state_probs()
# joint_probs = broken_db_matrix_2.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_2.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_2_cond_term = broken_db_matrix_2.env_probs[950]
# print(broken_db_matrix_2.conditional_entropy(broken_db_matrix_2_cond_term))
# print("stored information:")
# print(broken_db_matrix_2.stored_information(broken_db_matrix_2.matrix))
# print("mutual info:")
# print(broken_db_matrix_2.calc_mutual_info(950))

# print("broken DB (by hand, extreme):")
# broken_db_matrix_3 = System(np.array([[.0005, .999, .0005],
#                                       [.0005, .0005, .999],
#                                       [.999, .0005, .0005]]))
# print(broken_db_matrix_3.matrix)
# broken_db_matrix_3.calc_env_probs(broken_db_matrix_3.gen_ensemble(broken_db_matrix_3.matrix))
# broken_db_matrix_3.plot_state_probs()
# joint_probs = broken_db_matrix_3.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_3.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_3_cond_term = broken_db_matrix_3.env_probs[950]
# print(broken_db_matrix_3.conditional_entropy(broken_db_matrix_3_cond_term))
# print("stored information:")
# print(broken_db_matrix_3.stored_information(broken_db_matrix_3.matrix))
# print("mutual info:")
# print(broken_db_matrix_3.calc_mutual_info(950))

# print("DB (by hand, probabilities tend to one state):")
# db_matrix_4 = Environment(np.array([[.9, .05, .05],
#                                       [.9, .05, .05],
#                                       [.9, .05, .05]]))
# print(db_matrix_4.matrix)
# db_matrix_4.calc_env_probs(db_matrix_4.gen_ensemble(db_matrix_4.matrix))
# db_matrix_4.plot_state_probs()
# joint_probs = db_matrix_4.calc_joint_prob(950)
# print("entropy at time 951:")
# print(db_matrix_4.entropy(951))
# print("conditional entropy at 951 | 950:")
# db_matrix_4_cond_term = db_matrix_4.env_probs[950]
# print(db_matrix_4.conditional_entropy(db_matrix_4_cond_term))
# print("stored information:")
# print(db_matrix_4.stored_information())
# print("mutual info:")
# print(db_matrix_4.calc_mutual_info(950))

# print("broken DB (by hand, probabilities tend to two state):")
# db_matrix_5 = System(np.array([[.475, .475, .05],
#                                       [.475, .475, .05],
#                                       [.475, .475, .05]]))
# print(db_matrix_5.matrix)
# db_matrix_5.calc_env_probs(db_matrix_5.gen_ensemble(db_matrix_5.matrix))
# db_matrix_5.plot_state_probs()
# joint_probs = db_matrix_5.calc_joint_prob(950)
# print("entropy at time 951:")
# print(db_matrix_5.entropy(951))
# print("conditional entropy at 951 | 950:")
# db_matrix_5_cond_term = db_matrix_5.env_probs[950]
# print(db_matrix_5.conditional_entropy(db_matrix_5_cond_term))
# print("stored information:")
# print(db_matrix_5.stored_information())
# print("mutual info:")
# print(db_matrix_5.calc_mutual_info(950))


# print("broken DB (by hand, probabilities oscillate between two states):")
# broken_db_matrix_7 = Environment(np.array([[.0005, .999, .0005],
#                                       [.999, .0005, .0005],
#                                       [.0005, .999, .0005]]))
# print(broken_db_matrix_7.matrix)
# broken_db_matrix_7.calc_env_probs(broken_db_matrix_7.gen_ensemble(broken_db_matrix_7.matrix))
# broken_db_matrix_7.plot_state_probs()
# joint_probs = broken_db_matrix_7.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_7.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_7_cond_term = broken_db_matrix_7.env_probs[950]
# print(broken_db_matrix_7.conditional_entropy(broken_db_matrix_7_cond_term))
# print("stored information:")
# print(broken_db_matrix_7.stored_information())
# print("mutual info:")
# print(broken_db_matrix_7.calc_mutual_info(950))

# print("DB (by hand, probabilities oscillate between two states, can get stuck in another state):")
# broken_db_matrix_8 = Environment(np.array([[.0005, .999, .0005],
#                                       [.999, .0005, .0005],
#                                       [.0005, .0005, .999]]))
# print(broken_db_matrix_8.matrix)
# broken_db_matrix_8.calc_env_probs(broken_db_matrix_8.gen_ensemble(broken_db_matrix_8.matrix))
# broken_db_matrix_8.plot_state_probs()
# joint_probs = broken_db_matrix_8.calc_joint_prob(9500)
# print("entropy at time 9501:")
# print(broken_db_matrix_8.entropy(9501))
# print("conditional entropy at 9501 | 9500:")
# broken_db_matrix_8_cond_term = broken_db_matrix_8.env_probs[9500]
# print(broken_db_matrix_8.conditional_entropy())
# print("stored information:")
# print(broken_db_matrix_8.stored_information())
# print("mutual info:")
# print(broken_db_matrix_8.calc_mutual_info(9500))



