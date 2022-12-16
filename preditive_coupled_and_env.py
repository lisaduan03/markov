from markov_three import Markov_three
import numpy as np
from matplotlib import pyplot as plt
from three_state_env import Environment
from generating_matrices import generate_random_matrix, generate_db_matrices
from three_state_simulation import stationary_distribution

test_1 = Markov_three(p_transition_env = np.array(generate_db_matrices(1)), \
    p_t1 = np.array([[.05, .9, .05],
         [.05, .9, .05],
         [.05, .9, .05]]), p_t2 = np.array([[.05, .05, .9],
         [.05, .05, .9],
         [.05, .05, .9]]), p_t3 = np.array([[.9, .05, .05],
         [.9, .05, .05],
         [.9, .05, .05]]))

test_1.generate_ensemble(500)
test_1.calculate_probs()
test_1.calculate_probs_pred(50)

test_1_env = Environment(test_1.p_tenv)

coupled_decay = test_1.I_pred()
env_decay = test_1_env.I_pred()


#plotting 
x_axis = [x for x in range(len(test_1.env_probabilities_pred))]
plt.ylim(0, 1.5)
plt.xlabel("time steps after t")
plt.ylabel("information (nats)")
plt.title("information decay starting at time step 50 in coupled model and environment only")
plt.plot(x_axis, coupled_decay, label = 'I[s_t, x_k]')
plt.plot(x_axis, env_decay, label = 'I[x_t, x_k]')
plt.legend()
plt.show()

x_axis = [x for x in range(len(test_1.env_probabilities_pred))]
plt.xlabel("time steps after t")
plt.ylabel("information (nats)")
plt.title("zoomed in- information decay starting at time step 50 in coupled model and environment only")
plt.plot(x_axis[1:25], coupled_decay[1:25], label = 'I[s_t, x_k]')
plt.plot(x_axis[1:25], env_decay[1:25], label = 'I[x_t, x_k]')
plt.legend()
plt.show()

print(test_1_env.entropy_rate_env(test_1_env.matrix))
print(test_1_env.cw_to_ccw_ratio())
print(test_1_env.matrix)
print(stationary_distribution(np.array(test_1_env.matrix)))

