import markov_three
import random
import statistics
import matplotlib as plt
import numpy as np
import scipy
from markov_three import I_pred_list, mutual_info

# define the true objective function
def objective(t, a, b):
	return a+b*np.log(t)

x = [x for x in range(len(mutual_info))]
y_data = mutual_info

log_x_data = np.log(x)

curve = np.polyfit(log_x_data, y_data, 1)
print(curve)
