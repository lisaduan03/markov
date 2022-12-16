"""
7/6/22.
4 state env 4 state system simulation
"""
from struct import calcsize
from generating_matrices import generate_random_matrix, generate_four_state_cycle, generate_db_matrices, generate_db_matrices_helper
from three_state_simulation import stationary_distribution, markov_sequence
from three_state_simulation import markov_sequence
import numpy as np
from matplotlib import pyplot as plt 

class Environment_four:
    def __init__(self, matrix):
        self.matrix = matrix

    """
    first, using build_environment to build a markov chain representing the environment 
    """
    time_steps = 1000
    number_of_ensembles = 1000 # # of ensembles 
    list_of_sys_chains = [[None, None, None, None] for row in range(number_of_ensembles)]
    sys_probs = [[None, None, None, None] for row in range(time_steps)]
    joint_prob = [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None]]
    t_probs = [None, None, None, None]
    h_yx = 0 

    def build_environment(self, matrix):
        p_init_env = [.05, .02, .9, .03] # this can change
        global env_states
        env_states = markov_sequence(p_init_env, self.matrix, self.time_steps)
        env_states = [x + 1 for x in env_states]
        return env_states

    def gen_ensemble(self, sys_matrix: np.array):
        for x in range(self.number_of_ensembles):
            self.list_of_sys_chains[x] = self.build_environment(sys_matrix)
        return self.list_of_sys_chains

    def calc_sys_probs(self, list_of_sys_chains):
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
        t_4 = 0
        for x in range(self.time_steps):
            for l in range(self.number_of_ensembles):
                if list_of_sys_chains[l][x] == 1:
                    t_1 = t_1 + 1
                if list_of_sys_chains[l][x] == 2:
                    t_2 = t_2 + 1
                if list_of_sys_chains[l][x] == 3:
                    t_3 = t_3 + 1
                if list_of_sys_chains[l][x] == 4:
                    t_4 = t_4 + 1
            self.sys_probs[x][0] = t_1/self.number_of_ensembles
            self.sys_probs[x][1] = t_2/self.number_of_ensembles
            self.sys_probs[x][2] = t_3/self.number_of_ensembles
            self.sys_probs[x][3] = t_4/self.number_of_ensembles
            t_1 = 0
            t_2 = 0
            t_3 = 0
            t_4 = 0

    def calc_joint_prob(self, time):
        # first t, then t+1
        t_1_1 = 0
        t_1_2 = 0
        t_1_3 = 0
        t_1_4 = 0
        t_2_1 = 0
        t_2_2 = 0
        t_2_3 = 0
        t_2_4 = 0
        t_3_1 = 0
        t_3_2 = 0
        t_3_3 = 0
        t_3_4 = 0
        t_4_1 = 0
        t_4_2 = 0
        t_4_3 = 0
        t_4_4 = 0
        for l in range(self.number_of_ensembles):
            if self.list_of_sys_chains[l][time] == 1:
                if self.list_of_sys_chains[l][time + 1] == 1:
                    t_1_1 = t_1_1 + 1
                if self.list_of_sys_chains[l][time + 1] == 2:
                    t_1_2 = t_1_2 +1
                if self.list_of_sys_chains[l][time + 1] == 3:
                    t_1_3 = t_1_3 +1
                if self.list_of_sys_chains[l][time + 1] == 4:
                    t_1_4 = t_1_4 +1
            if self.list_of_sys_chains[l][time] == 2:
                if self.list_of_sys_chains[l][time+ 1] == 1:
                    t_2_1 = t_2_1 + 1
                if self.list_of_sys_chains[l][time+1] == 2:
                    t_2_2 = t_2_2 +1
                if self.list_of_sys_chains[l][time+1] == 3:
                    t_2_3 = t_2_3 +1
                if self.list_of_sys_chains[l][time + 1] == 4:
                    t_2_4 = t_2_4 +1
            if self.list_of_sys_chains[l][time] == 3:
                if self.list_of_sys_chains[l][time+ 1] == 1:
                    t_3_1 = t_3_1 + 1
                if self.list_of_sys_chains[l][time+1] == 2:
                    t_3_2 = t_3_2 +1
                if self.list_of_sys_chains[l][time+1] == 3:
                    t_3_3 = t_3_3 +1
                if self.list_of_sys_chains[l][time + 1] == 4:
                    t_3_4 = t_3_4 +1
            if self.list_of_sys_chains[l][time] == 4:
                if self.list_of_sys_chains[l][time+ 1] == 1:
                    t_4_1 = t_4_1 + 1
                if self.list_of_sys_chains[l][time+1] == 2:
                    t_4_2 = t_4_2 +1
                if self.list_of_sys_chains[l][time+1] == 3:
                    t_4_3 = t_4_3 +1
                if self.list_of_sys_chains[l][time + 1] == 4:
                    t_4_4 = t_4_4 +1
        self.joint_prob[0][0] = t_1_1/self.number_of_ensembles
        self.joint_prob[0][1] = t_1_2/self.number_of_ensembles
        self.joint_prob[0][2] = t_1_3/self.number_of_ensembles
        self.joint_prob[0][3] = t_1_4/self.number_of_ensembles
        self.joint_prob[1][0] = t_2_1/self.number_of_ensembles
        self.joint_prob[1][1] = t_2_2/self.number_of_ensembles
        self.joint_prob[1][2] = t_2_3/self.number_of_ensembles
        self.joint_prob[1][3] = t_2_4/self.number_of_ensembles
        self.joint_prob[2][0] = t_3_1/self.number_of_ensembles
        self.joint_prob[2][1] = t_3_2/self.number_of_ensembles
        self.joint_prob[2][2] = t_3_3/self.number_of_ensembles
        self.joint_prob[2][3] = t_3_4/self.number_of_ensembles
        self.joint_prob[3][0] = t_4_1/self.number_of_ensembles
        self.joint_prob[3][1] = t_4_2/self.number_of_ensembles
        self.joint_prob[3][2] = t_4_3/self.number_of_ensembles
        self.joint_prob[3][3] = t_4_4/self.number_of_ensembles



    def plot_state_probs(self):
            """
            first gathers probabilities of being at state 1, 2, or 3 at given time step,
            then plots the system probabilities
            """
            data_1 = []
            data_2 = []
            data_3 = []
            data_4 = []
            for x in range(self.time_steps):
                data_1.append(self.sys_probs[x][0])
                data_2.append(self.sys_probs[x][1])
                data_3.append(self.sys_probs[x][2])
                data_4.append(self.sys_probs[x][3])
            plt.plot(data_1, label = 'state 1')
            plt.plot(data_2, label = 'state 2')
            plt.plot(data_3, label = 'state 3')
            plt.plot(data_4, label = 'state 4')
            legend = plt.legend(loc='lower right', shadow=True, fontsize = 'medium') 
            legend.get_frame().set_facecolor('w')
            plt.xlabel("time step")
            plt.ylabel("probability")
            plt.ylim(0, 1)
            plt.title("probability of being at state 1, 2, 3, or 4")
            plt.show()

    def entropy(self, time):
        """
        modified. shannon entropy at time t
        """
        h = 0
        if self.sys_probs[time][0] == 0:
            self.sys_probs[time][0] = .000001
        if self.sys_probs[time][1] == 0:
            self.sys_probs[time][1] = .000001
        if self.sys_probs[time][2] == 0:
            self.sys_probs[time][2] = .000001
        if self.sys_probs[time][3] == 0:
            self.sys_probs[time][3] = .000001
        h = -((self.sys_probs[time][0]*np.log(self.sys_probs[time][0])) \
            + (self.sys_probs[time][1]*np.log(self.sys_probs[time][1])) \
                + (self.sys_probs[time][2]*np.log(self.sys_probs[time][2]))
                + (self.sys_probs[time][3]*np.log(self.sys_probs[time][3])))
        return h

    def conditional_entropy(self, cond_term: list):
        """
        modified condition entropy, for finding H[S_t+1|S_t]
        """
        h_t_1_t = 0
        for i in range(4):
            if cond_term[i] == 0:
                cond_term[i] = .000001
        for x in range(4):
            for y in range(4):
                if self.joint_prob[x][y] == 0:
                    self.joint_prob[x][y] = .000001
        h_t_1_t = (self.joint_prob[0][0]*np.log(cond_term[0]/self.joint_prob[0][0])) \
                + (self.joint_prob[1][0]*np.log(cond_term[1]/self.joint_prob[1][0])) \
                + (self.joint_prob[2][0]*np.log(cond_term[2]/self.joint_prob[2][0])) \
                + (self.joint_prob[3][0]*np.log(cond_term[3]/self.joint_prob[3][0])) \
                + (self.joint_prob[0][1]*np.log(cond_term[0]/self.joint_prob[0][1])) \
                + (self.joint_prob[1][1]*np.log(cond_term[1]/self.joint_prob[1][1])) \
                + (self.joint_prob[2][1]*np.log(cond_term[2]/self.joint_prob[2][1])) \
                + (self.joint_prob[3][1]*np.log(cond_term[3]/self.joint_prob[3][1])) \
                + (self.joint_prob[0][2]*np.log(cond_term[0]/self.joint_prob[0][2])) \
                + (self.joint_prob[1][2]*np.log(cond_term[1]/self.joint_prob[1][2])) \
                + (self.joint_prob[2][2]*np.log(cond_term[2]/self.joint_prob[2][2])) \
                + (self.joint_prob[3][2]*np.log(cond_term[3]/self.joint_prob[3][1])) \
                + (self.joint_prob[0][3]*np.log(cond_term[0]/self.joint_prob[0][3])) \
                + (self.joint_prob[1][3]*np.log(cond_term[1]/self.joint_prob[1][3])) \
                + (self.joint_prob[2][3]*np.log(cond_term[2]/self.joint_prob[2][3])) \
                + (self.joint_prob[3][3]*np.log(cond_term[3]/self.joint_prob[3][3]))
        return h_t_1_t


    def stored_information(self):
        # right now assume none of matrix entries are 0
        # double check that this is sum of all i and j even when i=j, 9 combos not 6
        stored_info = 0 
        steady_state = stationary_distribution(self.matrix)
        if self.matrix[0][0] != 0:
            stored_info = stored_info + (steady_state[0] * self.matrix[0][0] * np.log(self.matrix[0][0]/steady_state[0])) 
        if self.matrix[0][1] != 0:
            stored_info = stored_info + (steady_state[0] * self.matrix[0][1] * np.log(self.matrix[0][1]/steady_state[1])) 
        if self.matrix[0][2] != 0:
            stored_info = stored_info + (steady_state[0] * self.matrix[0][2] * np.log(self.matrix[0][2]/steady_state[2]))
        if self.matrix[0][3] != 0:
            stored_info = stored_info + (steady_state[0] * self.matrix[0][3] * np.log(self.matrix[0][3]/steady_state[3])) 
        if self.matrix[1][0] != 0:
            stored_info = stored_info + (steady_state[1] * self.matrix[1][0] * np.log(self.matrix[1][0]/steady_state[0])) 
        if self.matrix[1][1] != 0:
            stored_info = stored_info + (steady_state[1] * self.matrix[1][1] * np.log(self.matrix[1][1]/steady_state[1])) 
        if self.matrix[1][2] != 0:
            stored_info = stored_info + (steady_state[1] * self.matrix[1][2] * np.log(self.matrix[1][2]/steady_state[2]))
        if self.matrix[1][3] != 0:
            stored_info = stored_info + (steady_state[1] * self.matrix[1][3] * np.log(self.matrix[1][3]/steady_state[3])) 
        if self.matrix[2][0] != 0:
            stored_info = stored_info + (steady_state[2] * self.matrix[2][0] * np.log(self.matrix[2][0]/steady_state[0])) 
        if self.matrix[2][1] != 0:
            stored_info = stored_info + (steady_state[2] * self.matrix[2][1] * np.log(self.matrix[2][1]/steady_state[1])) 
        if self.matrix[2][2] != 0:
            stored_info = stored_info + (steady_state[2] * self.matrix[2][2] * np.log(self.matrix[2][2]/steady_state[2]))
        if self.matrix[2][3] != 0:
            stored_info = stored_info + (steady_state[2] * self.matrix[2][3] * np.log(self.matrix[2][3]/steady_state[3])) 
        if self.matrix[3][0] != 0:
            stored_info = stored_info + (steady_state[3] * self.matrix[3][0] * np.log(self.matrix[3][0]/steady_state[0])) 
        if self.matrix[3][1] != 0:
            stored_info = stored_info + (steady_state[3] * self.matrix[3][1] * np.log(self.matrix[3][1]/steady_state[1])) 
        if self.matrix[3][2] != 0:
            stored_info = stored_info + (steady_state[3] * self.matrix[3][2] * np.log(self.matrix[3][2]/steady_state[2]))
        if self.matrix[3][3] != 0:
            stored_info = stored_info + (steady_state[3] * self.matrix[3][3] * np.log(self.matrix[3][3]/steady_state[3])) 
        return stored_info  

    def calc_mutual_info(self, time):
        """
        to confirm that stored info and entropy/cond entropy are correct
        I[S_t, S_t+1]
        """
        time_prob = self.sys_probs[time]
        time_plus_one_prob = self.sys_probs[time + 1]
        mutual_info = (self.joint_prob[0][0] * np.log(self.joint_prob[0][0]/(time_prob[0]*time_plus_one_prob[0])) \
                    + (self.joint_prob[0][1] * np.log(self.joint_prob[0][1]/(time_prob[0]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[0][2] * np.log(self.joint_prob[0][2]/(time_prob[0]*time_plus_one_prob[2]))) \
                    + (self.joint_prob[0][3] * np.log(self.joint_prob[0][3]/(time_prob[0]*time_plus_one_prob[3]))) \
                    + (self.joint_prob[1][0] * np.log(self.joint_prob[1][0]/(time_prob[1]*time_plus_one_prob[0]))) \
                    + (self.joint_prob[1][1] * np.log(self.joint_prob[1][1]/(time_prob[1]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[1][2] * np.log(self.joint_prob[1][2]/(time_prob[1]*time_plus_one_prob[2]))) \
                    + (self.joint_prob[1][3] * np.log(self.joint_prob[1][3]/(time_prob[1]*time_plus_one_prob[3]))) \
                    + (self.joint_prob[2][0] * np.log(self.joint_prob[2][0]/(time_prob[2]*time_plus_one_prob[0]))) \
                    + (self.joint_prob[2][1] * np.log(self.joint_prob[2][1]/(time_prob[2]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[2][2] * np.log(self.joint_prob[2][2]/(time_prob[2]*time_plus_one_prob[2])))) \
                    + (self.joint_prob[2][3] * np.log(self.joint_prob[2][3]/(time_prob[2]*time_plus_one_prob[3]))) \
                    + (self.joint_prob[3][0] * np.log(self.joint_prob[3][0]/(time_prob[3]*time_plus_one_prob[0]))) \
                    + (self.joint_prob[3][1] * np.log(self.joint_prob[3][1]/(time_prob[3]*time_plus_one_prob[1]))) \
                    + (self.joint_prob[3][2] * np.log(self.joint_prob[3][2]/(time_prob[3]*time_plus_one_prob[2]))) \
                    + (self.joint_prob[3][3] * np.log(self.joint_prob[3][3]/(time_prob[3]*time_plus_one_prob[3])))
        return mutual_info

# testing
# in detailed balance
# print("broken DB, cw cycle:")
# broken_db_matrix_6 = Environment_four(np.array([[.0003, .999, .0003, .0004],
#                                       [.0003, .0003, .999, .0004],
#                                       [.0003, .0004, .0003, .999],
#                                       [.999, .0003, .0003, .0004]]))

# print(broken_db_matrix_6.matrix)
# broken_db_matrix_6.calc_sys_probs(broken_db_matrix_6.gen_ensemble(broken_db_matrix_6.matrix))
# broken_db_matrix_6.plot_state_probs()
# joint_probs = broken_db_matrix_6.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_6.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_6_cond_term = broken_db_matrix_6.sys_probs[950]
# print(broken_db_matrix_6.conditional_entropy(broken_db_matrix_6_cond_term))
# print("stored information:")
# print(broken_db_matrix_6.stored_information())
# print("mutual info:")
# print(broken_db_matrix_6.calc_mutual_info(950))

# print("db, oscillate between 2 states")
# broken_db_matrix_7 = Environment_four(np.array([[.0003, .999, .0003, .0004],
#                                       [.999, .0003, .0003, .0004],
#                                       [.0003, .999, .0003, .0004],
#                                       [.0003, .999, .0003, .0004]]))

# print(broken_db_matrix_7.matrix)
# broken_db_matrix_7.calc_sys_probs(broken_db_matrix_7.gen_ensemble(broken_db_matrix_7.matrix))
# broken_db_matrix_7.plot_state_probs()
# joint_probs = broken_db_matrix_7.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_7.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_7_cond_term = broken_db_matrix_7.sys_probs[950]
# print(broken_db_matrix_7.conditional_entropy(broken_db_matrix_7_cond_term))
# print("stored information:")
# print(broken_db_matrix_7.stored_information())
# print("mutual info:")
# print(broken_db_matrix_7.calc_mutual_info(950))

# print("db, oscillate between 3 states")
# broken_db_matrix_8 = Environment_four(np.array([[.0003, .999, .0003, .0004],
#                                       [.0003, .0003, .999, .0004],
#                                       [.0003, .0004, .0003, .999],
#                                       [.0003, .999, .0003, .0004]]))

# print(broken_db_matrix_8.matrix)
# broken_db_matrix_8.calc_sys_probs(broken_db_matrix_8.gen_ensemble(broken_db_matrix_8.matrix))
# broken_db_matrix_8.plot_state_probs()
# joint_probs = broken_db_matrix_8.calc_joint_prob(950)
# print("entropy at time 951:")
# print(broken_db_matrix_8.entropy(951))
# print("conditional entropy at 951 | 950:")
# broken_db_matrix_8_cond_term = broken_db_matrix_8.sys_probs[950]
# print(broken_db_matrix_8.conditional_entropy(broken_db_matrix_8_cond_term))
# print("stored information:")
# print(broken_db_matrix_8.stored_information())
# print("mutual info:")
# print(broken_db_matrix_8.calc_mutual_info(950))