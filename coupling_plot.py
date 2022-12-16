from markov_three import Markov_three
import numpy as np
import matplotlib.pyplot as plt
import random 
from generating_matrices import generate_random_matrix, generate_db_matrices, generate_coupled_cw, generate_db_matrices_helper


def plot_many_mi(number: int):
    plots = [] 
    trajectories = [Markov_three(generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3), generate_random_matrix(3)) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.mutual_information())
    plt.ylim(0, 1)
    plt.title("mutual information with randomly generated matrices")
    plt.xlabel("time steps")
    plt.ylabel("mutual information")

def plot_many_mi_coupled(number: int):
    plots = [] 
    trajectories = [Markov_three(generate_random_matrix(3), generate_coupled_cw(1), generate_coupled_cw(2), generate_coupled_cw(3)) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.mutual_information())
    plt.ylim(0, 1)
    plt.title("mutual information with randomly generated environment and coupling")
    plt.xlabel("time steps")
    plt.ylabel("mutual information")

def plot_many_mi_strongly_coupled(number: int):
    plots = [] 
    trajectories = [Markov_three(generate_random_matrix(3), p_t1 = np.array([[.05, .9, .05],
         [.05, .9, .05],
         [.05, .9, .05]]), p_t2 = np.array([[.05, .05, .9],
         [.05, .05, .9],
         [.05, .05, 9]]), p_t3 = np.array([[.9, .05, .05],
         [.9, .05, .05],
         [.9, .05, .05]])) for i in range(number)]
    for traj in trajectories:
        traj.generate_ensemble(100)
        traj.calculate_probs()
        plt.plot(traj.mutual_information())
    plt.ylim(0, 1)
    plt.title("mutual information with randomly generated environment and strong coupling")
    plt.xlabel("time steps")
    plt.ylabel("mutual information")


#plot_many_mi(10)
# plot_many_mi_coupled(10)
# plot_many_mi_strongly_coupled(10)
