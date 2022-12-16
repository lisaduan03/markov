from struct import calcsize
from time import time
from generating_matrices import generate_random_matrix
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
    env_start_time = 50 # this can cange 
    time_steps = 100
    number_of_ensembles = 100 # # of ensembles 
    list_of_env_chains = [[None, None, None] for row in range(number_of_ensembles)]
    env_probs = [[None, None, None] for row in range(time_steps)]
    t_probs = [None, None, None]
    joint_prob_time_slice = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    h_sk_st = [0] * ((time_steps - env_start_time) * 2)
    I_pred_env_list = []
    h = [0] * ((time_steps - env_start_time) * 2)
    conditional_entropy_mutual_info_verification = [0] * (time_steps - (env_start_time)) * 2
    list_of_joint_probs_t_k = [None] * (time_steps - (env_start_time)) * 2
    env_probabilities_pred = env_probs[(env_start_time - (time_steps - env_start_time)):]
    I_pred_list = [0] * ((time_steps - env_start_time) * 2)
    entropy_ss = 0




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
        self.gen_ensemble()

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

    def calc_joint_prob_time_slice(self, time):
        """
        returns a single value of the joint probability of time step t and t + 1
        can be used in calc_mutual_info method to find the mutual information at given time slice
        pick a time slice at steady state to verify mutual info
        """
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
                if self.list_of_env_chains[l][time + 1] == 1:
                    t_2_1 = t_2_1 + 1
                if self.list_of_env_chains[l][time +1] == 2:
                    t_2_2 = t_2_2 +1
                if self.list_of_env_chains[l][time +1] == 3:
                    t_2_3 = t_2_3 +1
            if self.list_of_env_chains[l][time] == 3:
                if self.list_of_env_chains[l][time + 1] == 1:
                    t_3_1 = t_3_1 + 1
                if self.list_of_env_chains[l][time +1 ] == 2:
                    t_3_2 = t_3_2 +1
                if self.list_of_env_chains[l][time +1 ] == 3:
                    t_3_3 = t_3_3 +1
        self.joint_prob_time_slice[0][0] = t_1_1/self.number_of_ensembles
        self.joint_prob_time_slice[0][1] = t_1_2/self.number_of_ensembles
        self.joint_prob_time_slice[0][2] = t_1_3/self.number_of_ensembles
        self.joint_prob_time_slice[1][0] = t_2_1/self.number_of_ensembles
        self.joint_prob_time_slice[1][1] = t_2_2/self.number_of_ensembles
        self.joint_prob_time_slice[1][2] = t_2_3/self.number_of_ensembles
        self.joint_prob_time_slice[2][0] = t_3_1/self.number_of_ensembles
        self.joint_prob_time_slice[2][1] = t_3_2/self.number_of_ensembles
        self.joint_prob_time_slice[2][2] = t_3_3/self.number_of_ensembles


    def plot_env_probs(self):
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
            plt.plot(data_1, label = 'state A')
            plt.plot(data_2, label = 'state B')
            plt.plot(data_3, label = 'state C')
            legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
            legend.get_frame().set_facecolor('w')
            plt.xlabel("time step")
            plt.ylabel("probability")
            plt.ylim(0, 1)
            plt.title("env probability of being at state A, B, or C")
            plt.show()

    def entropy_verify(self):
        """
        modified. shannon entropy at time t
        """
        self.calc_env_probs()
        for x in range(len(self.h)):
            if self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][0] != 0:
                self.h[x] = self.h[x] + float(mp.fmul(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][0], mp.log(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][0])))
            if self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][1] != 0:
                self.h[x] = self.h[x] + float(mp.fmul(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][1], mp.log(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][1])))
            if self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][2] != 0:
                self.h[x] = self.h[x] + float(mp.fmul(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][2], mp.log(self.env_probs[(self.env_start_time - ((self.time_steps - self.env_start_time))) + x][2])))
            self.h[x] = -self.h[x]
    
    def entropy(self):
        """
        entropy of steady state 
        """
        stationary_dist_env = stationary_distribution(self.matrix)
        if stationary_dist_env[0] != 0:
                self.entropy_ss = self.entropy_ss + float(mp.fmul(stationary_dist_env[0], mp.log(stationary_dist_env[0])))
        if stationary_dist_env[1] != 0:
                self.entropy_ss = self.entropy_ss + float(mp.fmul(stationary_dist_env[1], mp.log(stationary_dist_env[1])))
        if stationary_dist_env[2] != 0:
                self.entropy_ss = self.entropy_ss + float(mp.fmul(stationary_dist_env[1], mp.log(stationary_dist_env[1])))
        self.entropy_ss = -self.entropy_ss
        return self.entropy_ss


    def conditional_entropy(self, cond_term: list):
        """
        modified conditional entropy, for finding H[S_k|S_t]. 
        """
        self.calculate_probs_pred()
        for t in range(len(self.h_sk_st)):
            if self.list_of_joint_probs_t_k[t][0][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][0][0], mp.log(cond_term[0]/self.list_of_joint_probs_t_k[t][0][0])))
            if self.list_of_joint_probs_t_k[t][1][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][1][0], mp.log(cond_term[1]/self.list_of_joint_probs_t_k[t][1][0])))
            if self.list_of_joint_probs_t_k[t][2][0] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][2][0], mp.log(cond_term[2]/self.list_of_joint_probs_t_k[t][2][0])))
            if self.list_of_joint_probs_t_k[t][0][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][0][1], mp.log(cond_term[0]/self.list_of_joint_probs_t_k[t][0][1])))
            if self.list_of_joint_probs_t_k[t][1][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][1][1], np.log(cond_term[1]/self.list_of_joint_probs_t_k[t][1][1])))
            if self.list_of_joint_probs_t_k[t][2][1] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][2][1], np.log(cond_term[2]/self.list_of_joint_probs_t_k[t][2][1])))
            if self.list_of_joint_probs_t_k[t][0][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][0][2], np.log(cond_term[0]/self.list_of_joint_probs_t_k[t][0][2])))
            if self.list_of_joint_probs_t_k[t][1][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][1][2], np.log(cond_term[1]/self.list_of_joint_probs_t_k[t][1][2])))
            if self.list_of_joint_probs_t_k[t][2][2] != 0:
                self.h_sk_st[t] = self.h_sk_st[t] + float(mp.fmul(self.list_of_joint_probs_t_k[t][2][2], np.log(cond_term[2]/self.list_of_joint_probs_t_k[t][2][2])))


    def conditional_entropy_mutual_info_verify(self):
        """
        verifying that mutual info is correct using conditional entropy
        """
        self.entropy_verify()
        self.conditional_entropy(self.env_probs[self.env_start_time])
        for t in range((self.time_steps - self.env_start_time) * 2):
            self.conditional_entropy_mutual_info_verification[t] = self.h[t] - self.h_sk_st[t]

    def cw_to_ccw_ratio(self):
        cw = self.matrix[0][1] * self.matrix[1][2] * self.matrix[2][0]
        ccw = self.matrix[0][2] * self.matrix[2][1] * self.matrix[1][0]
        ratio = cw/ccw
        return ratio
    

    def cycle_affinity_flux(self):
        """
        should give the same value as entropy production rate 
        """ 
        global stationary_dist_env
        stationary_dist_env = stationary_distribution(self.matrix)
        cycle_affinity = np.log(self.cw_to_ccw_ratio())
        flux = (stationary_dist_env[0] * self.matrix[0][1]) - (stationary_dist_env[2] * self.matrix[1][0])
        return cycle_affinity * flux

    def stored_information(self):
        # right now assume none of matrix entries are 0
        # double check that this is sum of all i and j even when i=j, 9 combos not 6 
        steady_state = stationary_distribution(self.matrix)
        stored_info = 0 
        if self.matrix[0][0] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[0], mp.fmul(self.matrix[0][0], mp.log(self.matrix[0][0]/steady_state[0]))))
        if self.matrix[0][1] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[0], mp.fmul(self.matrix[0][1], mp.log(self.matrix[0][1]/steady_state[1]))))
        if self.matrix[0][2] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[0], mp.fmul(self.matrix[0][2], mp.log(self.matrix[0][2]/steady_state[2]))))
        if self.matrix[1][0] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[1], mp.fmul(self.matrix[1][0], mp.log(self.matrix[1][0]/steady_state[0]))))
        if self.matrix[1][1] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[1], mp.fmul(self.matrix[1][1], mp.log(self.matrix[1][1]/steady_state[1]))))
        if self.matrix[1][2] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[1], mp.fmul(self.matrix[1][2], mp.log(self.matrix[1][2]/steady_state[2]))))
        if self.matrix[2][0] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[2], mp.fmul(self.matrix[2][0], mp.log(self.matrix[2][0]/steady_state[0])))) 
        if self.matrix[2][1] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[2], mp.fmul(self.matrix[2][1], mp.log(self.matrix[2][1]/steady_state[1]))))
        if self.matrix[2][2] != 0:
            stored_info = stored_info + float(mp.fmul(steady_state[2], mp.fmul(self.matrix[2][2], mp.log(self.matrix[2][2]/steady_state[2]))))
        return stored_info  

    def calc_mutual_info(self, time):
        """
        to confirm that stored info and entropy/cond entropy are correct
        I[S_t, S_t+1]
        """
        time_prob = self.env_probs[time]
        time_plus_one_prob = self.env_probs[time+1]
        mutual_info = 0
        if self.joint_prob_time_slice[0][0] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[0][0], mp.log(self.joint_prob_time_slice[0][0]/(time_prob[0]*time_plus_one_prob[0]))))
        if self.joint_prob_time_slice[0][1] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[0][1], mp.log(self.joint_prob_time_slice[0][1]/(time_prob[0]*time_plus_one_prob[1])))) 
        if self.joint_prob_time_slice[0][2] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[0][2], mp.log(self.joint_prob_time_slice[0][2]/(time_prob[0]*time_plus_one_prob[2]))))
        if self.joint_prob_time_slice[1][0] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[1][0], mp.log(self.joint_prob_time_slice[1][0]/(time_prob[1]*time_plus_one_prob[0]))))
        if self.joint_prob_time_slice[1][1] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[1][1], mp.log(self.joint_prob_time_slice[1][1]/(time_prob[1]*time_plus_one_prob[1]))))
        if self.joint_prob_time_slice[1][2] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[1][2], mp.log(self.joint_prob_time_slice[1][2]/(time_prob[1]*time_plus_one_prob[2]))))    
        if self.joint_prob_time_slice[2][0] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[2][0], mp.log(self.joint_prob_time_slice[2][0]/(time_prob[2]*time_plus_one_prob[0]))))
        if self.joint_prob_time_slice[2][1] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[2][1], np.log(self.joint_prob_time_slice[2][1]/(time_prob[2]*time_plus_one_prob[1])))) 
        if self.joint_prob_time_slice[2][2] != 0:
            mutual_info = mutual_info + float(mp.fmul(self.joint_prob_time_slice[2][2], np.log(self.joint_prob_time_slice[2][2]/(time_prob[2]*time_plus_one_prob[2]))))
        return mutual_info
    
    def entropy_rate_env(self):
        global stationary_dist_env
        stationary_dist_env = stationary_distribution(self.matrix)
        entropy_rate_env_val = 0
        # if (matrix[0][0] == 0):
        #     matrix[0][0] = .0000001 # make it arbitrarily small so entropy prod rate blows up
        # entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * matrix[0][0]) - (stationary_dist_env[0] * matrix[0][0]))* np.log((stationary_dist_env[0] * matrix[0][0])/(stationary_dist_env[0] * matrix[0][0]))
        if (self.matrix[0][1] == 0):
            self.matrix[0][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] *self.matrix[1][0]) - (stationary_dist_env[0] * self.matrix[0][1]))* np.log((stationary_dist_env[1] * self.matrix[1][0])/(stationary_dist_env[0] * self.matrix[0][1]))
        if (self.matrix[0][2] == 0):
            self.matrix[0][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * self.matrix[2][0]) - (stationary_dist_env[0] * self.matrix[0][2]))* np.log((stationary_dist_env[2] * self.matrix[2][0])/(stationary_dist_env[0] * self.matrix[0][2]))
        if (self.matrix[1][0] == 0):
            self.matrix[1][0] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * self.matrix[0][1]) - (stationary_dist_env[1] * self.matrix[1][0]))* np.log((stationary_dist_env[0] * self.matrix[0][1])/(stationary_dist_env[1] * self.matrix[1][0]))
        # if (matrix[1][1] == 0):
        #     matrix[1][1] = .0000001
        #entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * matrix[1][1]) - (stationary_dist_env[1] * matrix[1][1]))* np.log((stationary_dist_env[1] * matrix[1][1])/(stationary_dist_env[1] * matrix[1][1]))
        if (self.matrix[1][2] == 0):
            self.matrix[1][2] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * self.matrix[2][1]) - (stationary_dist_env[1] * self.matrix[1][2]))* np.log((stationary_dist_env[2] * self.matrix[2][1])/(stationary_dist_env[1] * self.matrix[1][2]))
        if (self.matrix[2][0] == 0):
            self.matrix[2][0] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[0] * self.matrix[0][2]) - (stationary_dist_env[2] * self.matrix[2][0]))* np.log((stationary_dist_env[0] * self.matrix[0][2])/(stationary_dist_env[2] * self.matrix[2][0]))
        if (self.matrix[2][1] == 0):
            self.matrix[2][1] = .0000001
        entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[1] * self.matrix[1][2]) - (stationary_dist_env[2] *self. matrix[2][1]))* np.log((stationary_dist_env[1] * self.matrix[1][2])/(stationary_dist_env[2] * self.matrix[2][1]))
        # if (matrix[2][2] == 0):
        #     matrix[2][2] = .0000001
        #entropy_rate_env_val = entropy_rate_env_val + ((stationary_dist_env[2] * matrix[2][2]) - (stationary_dist_env[2] * matrix[2][2]))* np.log((stationary_dist_env[2] * matrix[2][2])/(stationary_dist_env[2] * matrix[2][2]))
        entropy_rate_env_val = entropy_rate_env_val * (1/2)
        return entropy_rate_env_val


    def calculate_probs_pred(self): 
        self.gen_ensemble()
        self.calc_env_probs()
        """  
        fills up list_of_joint_probs_t_k list of matrices, with the following format:

        [[[P(X_t_X_t+1 = a_1), P(X_t_X_t+1 = b_1), P(X_t_X_t+1 = c_1)], [[P(X_t_X_t+2 = a_1), P(S_t_X_t+2 = b_1), P(S_t_X_t+2 = c_1)],      
        [P(X_t_X_t+1 = a_2), P(X_t_X_t+1 = b_2), P(X_t_X_t+1 = c_2)],  [P(X_t_X_t+2 = a_2), P(S_t_X_t+2 = b_2), P(S_t_X_t+2 = c_2)], 
        [P(X_t_X_t+1 = a_3), P(X_t_X_t+1 = b_3), P(X_t_X_t+1 = c_3)]], [P(S_t_X_t+2 = a_3), P(S_t_X_t+2 = b_3), P(S_t_X_t+2 = c_3)]], ... 
        """
        # memory, where sys and env are in parallel 
        t_1_a = 0
        t_1_b = 0
        t_1_c = 0
        t_2_a = 0
        t_2_b = 0
        t_2_c = 0
        t_3_a = 0
        t_3_b = 0
        t_3_c = 0
        list_of_joint_probs_index = 0
        for t in range((self.env_start_time - (self.time_steps - self.env_start_time)), self.time_steps): 
            for e in range(self.number_of_ensembles): # gives us number of ensembles
                if self.list_of_env_chains[e][self.env_start_time] == 1:
                    # looking through associated environments to set up joint_probabilities_list
                    if self.list_of_env_chains[e][t] == 1:
                        t_1_a = t_1_a + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        t_1_b = t_1_b + 1
                    else:
                        t_1_c = t_1_c + 1
                elif self.list_of_env_chains[e][self.env_start_time] == 2:
                    if self.list_of_env_chains[e][t] == 1:
                        t_2_a = t_2_a + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        t_2_b = t_2_b + 1
                    else:
                        t_2_c = t_2_c + 1
                else:
                    if self.list_of_env_chains[e][t] == 1:
                        t_3_a = t_3_a + 1
                    elif self.list_of_env_chains[e][t] == 2:
                        t_3_b = t_3_b + 1
                    else:
                        t_3_c = t_3_c + 1
            # filling the data structures
            prob_array = [[None, None, None], [None, None, None], [None, None, None]]
            prob_array[0][0] = t_1_a/self.number_of_ensembles
            prob_array[0][1] = t_1_b/self.number_of_ensembles
            prob_array[0][2] = t_1_c/self.number_of_ensembles
            prob_array[1][0] = t_2_a/self.number_of_ensembles
            prob_array[1][1] = t_2_b/self.number_of_ensembles
            prob_array[1][2] = t_2_c/self.number_of_ensembles
            prob_array[2][0] = t_3_a/self.number_of_ensembles
            prob_array[2][1] = t_3_b/self.number_of_ensembles
            prob_array[2][2] = t_3_c/self.number_of_ensembles
            self.list_of_joint_probs_t_k[list_of_joint_probs_index] = prob_array
            # resetting the values 
            t_1_a = 0
            t_1_b = 0
            t_1_c = 0
            t_2_a = 0
            t_2_b = 0
            t_2_c = 0
            t_3_a = 0
            t_3_b = 0
            t_3_c = 0  
            list_of_joint_probs_index = list_of_joint_probs_index + 1 

    def I_pred(self):
        """
        calculates predictive information X_t has about X_t+1, X_t+1, etc 
        """
        for t in range(len(self.env_probabilities_pred)): # is this correct
            p_1_a = self.list_of_joint_probs_t_k[t][0][0]
            p_2_a = self.list_of_joint_probs_t_k[t][1][0]
            p_3_a = self.list_of_joint_probs_t_k[t][2][0]
            p_1_b = self.list_of_joint_probs_t_k[t][0][1]
            p_2_b = self.list_of_joint_probs_t_k[t][1][1]
            p_3_b = self.list_of_joint_probs_t_k[t][2][1]
            p_1_c = self.list_of_joint_probs_t_k[t][0][2]
            p_2_c = self.list_of_joint_probs_t_k[t][1][2]
            p_3_c = self.list_of_joint_probs_t_k[t][2][2]
            # these are at time k 
            p_a = self.env_probabilities_pred[t][0]
            p_b = self.env_probabilities_pred[t][1]
            p_c = self.env_probabilities_pred[t][2]
            # these are at time t
            p_1 = self.env_probs[self.env_start_time][0]
            p_2 = self.env_probs[self.env_start_time][1]
            p_3 = self.env_probs[self.env_start_time][2]
            if p_1_a != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_1_a, mp.log(p_1_a / mp.fmul(p_a, p_1)))) 
            if p_2_a != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_2_a, mp.log(p_2_a / mp.fmul(p_a, p_2))))
            if p_3_a != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_3_a, mp.log(p_3_a / mp.fmul(p_a, p_3))))
            if p_1_b != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_1_b, mp.log(p_1_b / mp.fmul(p_b, p_1))))
            if p_2_b != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_2_b, mp.log(p_2_b / mp.fmul(p_b, p_2))))
            if p_3_b != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_3_b, mp.log(p_3_b / mp.fmul(p_b, p_3))))
            if p_1_c != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_1_c, mp.log(p_1_c / mp.fmul(p_c, p_1))))
            if p_2_c != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_2_c, mp.log(p_2_c / mp.fmul(p_c, p_2))))
            if p_3_c != 0:
                self.I_pred_list[t] = self.I_pred_list[t] + float(mp.fmul(p_3_c, mp.log(p_3_c / mp.fmul(p_c, p_3))))
        return self.I_pred_list
    
    def plot_I_pred(self): # shift the x axis so it starts at 0 
        self.I_pred()
        self.conditional_entropy_mutual_info_verify()
        # x = [x for x in range(1, len(env_probabilities_pred))]
        # string_x = [str(t) for t in x]
        # x_labels = ['t +' + string_x[t] for t in range(len(string_x))]
        x_axis_past = [-x for x in range(self.time_steps - self.env_start_time)]
        x_axis_past = [(x - 1) for x in x_axis_past]
        x_axis_past.reverse()
        x_axis_future = [x for x in range(self.time_steps - self.env_start_time)]
        # plt.xticks(x_axis_list ,x_labels)
        plt.ylim(0, 1.3)
        plt.xlabel("time steps after t")
        plt.ylabel("predictive power")
        plt.title("predictive power and memory of X_800")
        plt.plot(x_axis_past, self.I_pred_list[0: 200], label = "mutual info formula")
        plt.plot(x_axis_future, self.I_pred_list[200: 400], label = "mutual info formula")
        plt.plot(x_axis_past, self.conditional_entropy_mutual_info_verification[0:200], label = "conditional entropy formula")
        plt.plot(x_axis_future, self.conditional_entropy_mutual_info_verification[200:400], label = "conditional entropy formula")
        plt.legend()
        plt.show()

# testing
# broken_db_matrix = Environment(np.array([[.005, .99, .005],
#                                        [.005, .005, .99],
#                                        [.99, .005, .005]]))
# print(broken_db_matrix.matrix)
# broken_db_matrix.calculate_probs_pred()
# broken_db_matrix.plot_env_probs()
# broken_db_matrix.plot_I_pred()

# db_matrix = Environment(np.array([[.2, .4, .4],
#                        [.1, .6, .3],
#                        [.2, .6, .2]]))
# print(db_matrix.matrix)
# db_matrix.calculate_probs_pred()
# db_matrix.plot_env_probs()
# db_matrix.plot_I_pred()

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



