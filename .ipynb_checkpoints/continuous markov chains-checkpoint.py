# """
# 6/16/22. First time working with continuous Markov chains
# """
import numpy as np

# first solve for steady state distribution using linear algebra
def stationary_distribution(p_transition):
    """
    Calculates stationary distribution.
    """
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq

def steady_state(p_transition):
    (np.transpose(p_transition).stack(np.vector([1,1,1])).solve_right(np.vector([0,0,0,1])))

Q_matrix = np.array([[-3, 2, 1],
                     [1, -5, 4],
                     [1, 8, -9]])
steady_state_probs = stationary_distribution(Q_matrix)
print(steady_state_probs)

