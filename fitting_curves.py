from cmath import e
import markov_three
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from markov_three import Markov_three
from generating_matrices import generate_four_state_cycle, generate_four_state_line, generate_four_state_star, generate_four_state_two_cycle, generate_four_state_with_island, generate_random_matrix, generate_three_state_line, generate_db_matrices
from entropy_rate import entropy_rate_env
from three_state_simulation import stationary_distribution
import statistics 
from three_state_env import Environment
from four_state_env import Environment_four
from markov_four import Markov_four
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""
scatterplots on mutual information, preditive power, entropy, etc\
"""


def growth_func(x, m, a):
	return m * (1-(np.power(np.e, (-a * x))))

def decay_func(x, A, K, C):
    return A * np.exp(-K * x) + C

def linear_func(x, m, a):
	return (m * x) + a

def fit_curve_growth(markov):
	markov.generate_ensemble(250)
	markov.calculate_probs()
	mutual_info = markov.mutual_information()

	x = [x for x in range(len(mutual_info))]
	y = mutual_info
	plt.plot(x, y, 'ro',label="Original Data")

	x = np.array(x, dtype=float) #transform your data in a numpy array of floats 
	y = np.array(y, dtype=float) #so the curve_fit can work


	popt, pcov = curve_fit(growth_func, x, y)

	print(popt[0], popt[1])

	plt.plot(x, growth_func(x, *popt), label="Fitted Curve")
	plt.legend(loc='lower right')

	plt.ylabel("mutual information")
	plt.xlabel("time steps")
	plt.show

def fit_curve_decay(markov):
	markov.generate_ensemble(250)
	markov.calculate_probs()
	markov.calculate_probs_pred(20)
	pred_info = markov.I_pred()

	x = [x for x in range(len(pred_info))]
	y = pred_info
	plt.plot(x, y, 'ro', label = "Original Data")

	x = np.array(x, dtype=float) #transform your data in a numpy array of floats 
	y = np.array(y, dtype=float) #so the curve_fit can work

	popt, pcov = curve_fit(decay_func, x, y)

	print(popt[0], popt[1], popt[2])

	plt.plot(x, decay_func(x, *popt), label = "Fitted Curve")
	plt.legend(loc = "lower right")

	plt.ylabel("I_pred")
	plt.xlabel("time steps after system start time")
	plt.title("I_pred with randomly generated envrionment and fitted curve")
	plt.show()


# test = Markov_three(p_transition_env = generate_random_matrix(3), p_t1 = np.array([[.05, .9, .05],
#          [.05, .9, .05],
#          [.05, .9, .05]]), p_t2 = np.array([[.05, .05, .9],
#          [.05, .05, .9],
#          [.05, .05, 9]]), p_t3 = np.array([[.9, .05, .05],
#          [.9, .05, .05],
#          [.9, .05, .05]]))

# def I_pred_plots(number: int):
# 	list_slope = []
# 	list_I_pred_min = []
# 	list_entropy = []
# 	list_entropy_production_rate = []
# 	list_cw_to_ccw_ratio = []
# 	x = [x for x in range(80)]
# 	trajectories = [Markov_three(generate_random_matrix()
# , p_t1 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
# 		 p_t2 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, 9]]), \
# 			 p_t3 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]])) for i in range(number)]
# 	for traj in trajectories:
# 		traj.generate_ensemble(237)
# 		traj.calculate_probs()
# 		traj.calculate_probs_pred(20)
# 		popt, pcov = curve_fit(decay_func, x, traj.I_pred())
# 		list_I_pred_min.append(popt[2])
# 		list_slope.append(popt[1])
# 		# 3 quants about environment
# 		list_cw_to_ccw_ratio.append(cw_to_ccw_ratio(traj.p_tenv))
# 		list_entropy_production_rate.append(entropy_rate_env(traj.p_tenv))
# 		list_entropy.append(traj.entropy(traj.env_probabilities))
# 	list_entropy_avg = []
# 	for list in list_entropy:
# 		list_entropy_avg.append(statistics.mean(list))

# 	# plotting

# 	plt.figure()
# 	plt.xlabel("entropy production rate")
# 	plt.ylabel("cw to ccw ratio")
# 	plt.title("cw to ccw ratio of env vs. entropy production rate of env")
# 	plt.plot(list_entropy_production_rate, list_cw_to_ccw_ratio, 'ro')		 
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("entropy production rate")
# 	plt.ylabel("avg shannon entropy")
# 	plt.title("avg shannon entropy vs. entropy production rate of env")
# 	plt.plot(list_entropy_production_rate, list_entropy_avg, 'ro')		 
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("cw to ccw ratio")
# 	plt.ylabel("avg shannon entropy")
# 	plt.title("avg shannon entropy vs. cw to ccw ratio")
# 	plt.plot(list_cw_to_ccw_ratio, list_entropy_avg, 'ro')		 
# 	plt.show()

# 	# about I_pred 
# 	plt.figure()
# 	plt.xlabel("cw to ccw ratio")
# 	plt.ylabel('rate of decay')
# 	plt.title("rate of decay vs cw to ccw ratio of environment")
# 	plt.plot(list_cw_to_ccw_ratio, list_slope, 'ro')
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("entropy production rate")
# 	plt.ylabel("rate of decay")
# 	plt.title("rate of decay vs. entropy production rate of environment")
# 	plt.plot(list_entropy_production_rate, list_slope, 'ro')		 
# 	plt.show()
	
# 	plt.figure()
# 	plt.xlabel("shannon entropy")
# 	plt.ylabel("rate of decay")
# 	plt.title("rate of decay vs. shannon entropy of environment")
# 	plt.plot(list_entropy_avg, list_slope, 'ro')		 
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("entropy production rate")
# 	plt.ylabel("I_pred_min")
# 	plt.title("least predictive info vs. entropy production rate of environment")
# 	plt.plot(list_entropy_production_rate, list_I_pred_min, 'ro')		 
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("shannon entropy")
# 	plt.ylabel("I_pred_min")
# 	plt.title("least predictive info vs. shannon entropy of environment")
# 	plt.plot(list_entropy_avg, list_I_pred_min, 'ro')		 
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("rate of decay")
# 	plt.ylabel("I_pred_min")
# 	plt.title("least predictive info vs. rate of decay")
# 	plt.plot(list_slope, list_I_pred_min, 'ro')		 
# 	plt.show()

def check_db_and_stored_info(number: int):
	"""
	first half not in db, second half in db
	"""
	list_db_entropy_prod_rate = []
	list_non_db_entropy_prod_rate = []
	list_db_stored_info = []
	list_non_db_stored_info = []
	list_db_entropy= []
	list_non_db_entropy = []
	trajectories = [Environment(generate_random_matrix(3)) for i in range(number)]
	trajectories = trajectories + [Environment(np.array(generate_db_matrices(1)[0])) for i in range(number-1)]
	trajectories.append(Environment(np.array([[.2, .4, .4],
                               [.1, .6, .3],
                                [.2, .6, .2]])))
	for traj in trajectories[0: number]:
		list_non_db_entropy.append(traj.entropy())
		list_non_db_entropy_prod_rate.append(traj.entropy_rate_env())
		list_non_db_stored_info.append(traj.stored_information())
	for traj in trajectories[number:]:
		list_db_entropy.append(traj.entropy())
		list_db_entropy_prod_rate.append(traj.entropy_rate_env())
		list_db_stored_info.append(traj.stored_information())
	
	plt.figure()
	plt.ylabel("stored information")
	plt.xlabel("entropy")
	plt.title("stored information in randomly generated environments")
	y_db = list_db_stored_info
	x_db = list_db_entropy
	y_non_db = list_non_db_stored_info
	x_non_db = list_non_db_entropy
	plt.plot(x_non_db, y_non_db, 'ro', label = 'broken db')
	plt.plot(x_db, y_db, 'ro', color = 'green', label = 'db')
	plt.legend()
	plt.show()

# check_db_and_stored_info(1000)
	
	
def plots(number: int):
	"""
	max pred info one step ahead I(s_t, x_t+1) and stored information 
	"""
	list_slope = []
	list_I_pred_max_one_step = []
	list_envs = []
	list_stored_info_three = []
	list_stored_info_four = []
	list_Ixtxt = []
	list_I_max = []
	x = [x for x in range(999)]
	# trajectories = [Markov_four(np.array(generate_four_state_two_cycle()), p_t1 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]),
	# 	 p_t2 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
	# 		 p_t3 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]), \
	# 			 p_t4 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
	# trajectories = [Markov_four(generate_random_matrix(4), p_t1 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]),
	# 	 p_t2 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
	# 		 p_t3 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]), \
	# 			 p_t4 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
	trajectories = [Markov_three(np.array(generate_three_state_line()), p_t1 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]),
		 p_t2 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
			 p_t3 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
	trajectories = trajectories + [Markov_four(np.array(generate_four_state_line()), p_t1 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]]),
	 	 p_t2 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
	 		 p_t3 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]]), \
				 p_t4 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, .9]])) for i in range(number)]
	for traj in trajectories:
		traj.generate_ensemble(237)
		traj.calculate_probs()
		# popt, pcov = curve_fit(growth_func, x, traj.mutual_information())
		list_envs.append(traj.p_tenv)
		popt, pcov = curve_fit(growth_func, x, traj.I_pred_one_step())
		list_I_pred_max_one_step.append(popt[0])
		# list_slope.append(popt[1])
	for env in list_envs[0:number]:
		# Environment(env).calculate_probs_pred()
		# list_Ixtxt.append(Environment(env).I_pred()[200])
		# list_stored_info.append(Environment_four(env).stored_information())
		list_stored_info_three.append(Environment(env).stored_information())
	for env in list_envs[number:]:
		list_stored_info_four.append(Environment_four(env).stored_information())


	# plt.figure()
	# plt.ylabel("max I[s_t, x_t+1]")
	# plt.xlabel("stored information")
	# plt.title("maximum predictive power one step ahead and stored information in environment")
	# y = list_I_pred_max_one_step
	# x = list_stored_info
	# plt.plot(x, y, 'ro')
	# plt.show()

	plt.figure()
	plt.ylabel("I[s_t, x_t+1]")
	plt.xlabel("stored information")
	plt.title("predictive power and stored information in randomly generated environments")
	x_3 = list_stored_info_three
	x_3 = np.array([x_3], dtype=float)
	x_3 = x_3.reshape(-1, 1)
	x_4 = list_stored_info_four
	x_4 = np.array([x_4], dtype=float)
	x_4 = x_4.reshape(-1, 1)
	y = list_I_pred_max_one_step
	y = np.array([y], dtype=float) 
	y = y.reshape(-1, 1)
	model_3 = LinearRegression()
	model_4 = LinearRegression()
	slope_3 = model_3.fit(x_3, y[0:number])
	slope_4 = model_4.fit(x_4, y[number:])
	y_pred_x_3 = model_3.predict(x_3)
	y_pred_x_4 = model_4.predict(x_4)
	r2_3 = r2_score(y[0:number], y_pred_x_3)
	r2_4 = r2_score(y[number:], y_pred_x_4)
	# #a, b = curve_fit(linear_func, x, y)	
	# #a, b = np.polyfit(x, y, 1)	 
	# #plt.plot(x, a*x+b, color='steelblue', linestyle='--', linewidth=2)
	# #plt.text(1, 17, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
	# #line = (a[0]*x) + a[1]
	#plt.plot(x, line, '--', color = 'blue', label = 'line of best fit')
	print('Slope 3 state environment:' ,slope_3.coef_)
	print('Slope 4 state environment:' ,slope_4.coef_)
	print('R2 score 3 state environment: ', r2_3)
	print('R2 score 4 state environment: ', r2_4)
	print('n= ', number)
	plt.plot(x_3, y[0:number], 'ro', color = 'blue')
	plt.plot(x_4, y[number:], 'ro')
	plt.plot(x_3, y_pred_x_3, color = 'b', linestyle='--', label = '3 state environment')
	plt.plot(x_4, y_pred_x_4, color = 'red', linestyle='--', label = '4 state environment')
	plt.legend()
	plt.show()

# plots(100)


 

def mutual_info_plots(number: int):
	# list_entropy_production_rate = []
	global list_I_max
	list_I_max = []
	list_entropy_ss = []
	#list_slope = []
	#list_entropy = []
	#list_entropy_production_rate = []
	# list_cw_to_ccw_ratio = []
	x = [x for x in range(100)]
	trajectories = [Markov_three(generate_random_matrix(3)
, p_t1 = np.array([[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]]), \
		 p_t2 = np.array([[.05, .05, .9], [.05, .05, .9], [.05, .05, 9]]), \
			 p_t3 = np.array([[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]])) for i in range(number)]
	# plt.figure()
	# plt.ylabel("shannon entropy of environment")
	# plt.xlabel("time steps")
	# plt.title("shannon entropy of randomly generated environments")
	for traj in trajectories:
		traj.generate_ensemble(100)
		traj.calculate_probs()
		popt, pcov = curve_fit(growth_func, x, traj.mutual_information())
		list_I_max.append(popt[0])
		list_entropy_ss.append(traj.entropy(traj.sys_probabilities))
		# list_entropy.append(traj.entropy(traj.env_probabilities))
		# list_cw_to_ccw_ratio.append(cw_to_ccw_ratio(traj.p_tenv))
		# list_slope.append(popt[1])
		# list_entropy_production_rate.append(entropy_rate_env(traj.p_tenv))

# 	plt.figure()
# 	plt.xlabel("cw to ccw ratio")
# 	plt.ylabel("I_max")
# 	plt.title("I_max vs cw to cc ratio")
# 	plt.plot(list_cw_to_ccw_ratio, list_I_max, 'ro')
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("cw to ccw ratio")
# 	plt.ylabel("slope")
# 	plt.title("slope vs cw to cc ratio")
# 	plt.plot(list_cw_to_ccw_ratio, list_slope, 'ro')
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("slope")
# 	plt.ylabel("I_max")
# 	plt.title("I_max vs slope")
# 	plt.plot(list_slope, list_I_max, 'ro')
# 	plt.show()

# 	plt.figure()
# 	plt.xlabel("entropy production rate")
# 	plt.ylabel("slope")
# 	plt.title("entropy production rate vs. slope")
# 	plt.plot(list_entropy_production_rate, list_slope, 'ro')
# 	plt.show()

	# list_entropy_avg = []
	# for list in list_entropy:
	# 	list_entropy_avg.append(statistics.mean(list))
	plt.figure()
	plt.xlabel("shannon entropy of steady state")
	plt.ylabel("I_max")
	plt.title("mutual information vs entropy of system")
	plt.plot(list_entropy_ss, list_I_max, 'ro')
	# plt.show()
	# plt.figure()
	# plt.xlabel("avg shannon entropy of env")
	# plt.ylabel("slope")
	# plt.title("avg shannon entropy vs. slope")
	# plt.plot(list_entropy_avg, list_slope, 'ro')
	# plt.show()

mutual_info_plots(100)
#fit_curve_decay(test)
#mutual_info_plots(100)
# I_pred_plots(1000)


def three_state_stored_info(number: int):
	"""
	showing that stored info for a 3 state env shouldn't exceed ln3
	strongly coupled cyclic system right now, maybe should change 
	"""
	stored_info_list = []
	entropy_prod_rate_list = []
	list_envs = []
	trajectories = [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	trajectories = trajectories + [Environment(generate_random_matrix(3))for i in range(number)]
	for traj in trajectories:
		list_envs.append(traj)
	for env in list_envs:
		stored_info_list.append(env.stored_information())
		entropy_prod_rate_list.append(Environment(env).entropy_rate_env(env.matrix))
	# manually adding these two points that I know approach the limit
	stored_info_list.append(Environment(np.array([[5.00e-04, 9.99e-01, 5.00e-04], [9.99e-01, 5.00e-04, 5.00e-04],\
		 [5.00e-04, 5.00e-04, 9.99e-01]])).stored_information())
	entropy_prod_rate_list.append((Environment([[5.00e-04, 9.99e-01, 5.00e-04], [9.99e-01, 5.00e-04, 5.00e-04], \
		[5.00e-04, 5.00e-04, 9.99e-01]]).entropy_rate_env(np.array([[5.00e-04, 9.99e-01, 5.00e-04], [9.99e-01, 5.00e-04, 5.00e-04],\
			 [5.00e-04, 5.00e-04, 9.99e-01]]))))
	stored_info_list.append(Environment(np.array([[5.00e-04, 9.99e-01, 5.00e-04], [5.00e-04, 5.00e-04, 9.99e-01], \
		[9.99e-01, 5.00e-04, 5.00e-04]])).stored_information())
	entropy_prod_rate_list.append((Environment([[5.00e-04, 9.99e-01, 5.00e-04], [5.00e-04, 5.00e-04, 9.99e-01], \
		[9.99e-01, 5.00e-04, 5.00e-04]]).entropy_rate_env(np.array([[5.00e-04, 9.99e-01, 5.00e-04], [5.00e-04, 5.00e-04, 9.99e-01], \
			[9.99e-01, 5.00e-04, 5.00e-04]]))))
	plt.figure()
	plt.xlabel("entropy production rate")
	plt.ylabel("stored information in environment")
	plt.title("stored information for three state environment")
	x = entropy_prod_rate_list
	x = np.array(x, dtype=float) 
	y = stored_info_list
	y = np.array(y, dtype=float) 
	plt.plot(x, y, 'ro')
	plt.axhline(y=np.log(3), color='purple', linestyle='--', label = "ln(3)")
	plt.legend()
	plt.show()


def four_state_stored_info(number: int):
	"""
	showing that stored info for a 4 state env shouldn't exceed ln4
	strongly coupled cyclic system right now, maybe should change 
	"""
	stored_info_list = []
	entropy_prod_rate_list = []
	list_envs = []
	trajectories = [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]	
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)]
	trajectories = trajectories + [Environment_four(generate_random_matrix(4)) for i in range(number)] 
	for env in trajectories:
		stored_info_list.append(Environment(env.matrix).stored_information())
		entropy_prod_rate_list.append(Environment(env.matrix).entropy_rate_env(np.array(env.matrix)))
 
	stored_info_list.append(Environment_four(np.array([[3.00e-04, 9.99e-01, 3.00e-04, 4.00e-04],
 [3.00e-04, 3.00e-04, 9.99e-01, 4.00e-04],
 [3.00e-04, 4.00e-04, 3.00e-04, 9.99e-01],
 [9.99e-01, 3.00e-04, 3.00e-04, 4.00e-04]])).stored_information())
	entropy_prod_rate_list.append((entropy_rate_env(np.array(np.array([[3.00e-04, 9.99e-01, 3.00e-04, 4.00e-04],
 [3.00e-04, 3.00e-04, 9.99e-01, 4.00e-04],
 [3.00e-04, 4.00e-04, 3.00e-04, 9.99e-01],
 [9.99e-01, 3.00e-04, 3.00e-04, 4.00e-04]])))))
	
	stored_info_list.append(Environment_four(np.array([[3.00e-04, 9.99e-01, 3.00e-04, 4.00e-04],
 [9.99e-01, 3.00e-04, 3.00e-04, 4.00e-04],
 [3.00e-04, 4.00e-04, 9.99e-01, 3.00e-04],
 [4.00e-04, 3.00e-04, 3.00e-04, 9.99e-01]])).stored_information())
	entropy_prod_rate_list.append(entropy_rate_env(np.array(np.array([[3.00e-04, 9.99e-01, 3.00e-04, 4.00e-04],
 [9.99e-01, 3.00e-04, 3.00e-04, 4.00e-04],
 [3.00e-04, 4.00e-04, 9.99e-01, 3.00e-04],
 [4.00e-04, 3.00e-04, 3.00e-04, 9.99e-01]]))))

	plt.figure()
	plt.xlabel("entropy production rate")
	plt.ylabel("stored information in environment")
	plt.title("stored information for four state environment")
	x = entropy_prod_rate_list
	x = np.array(x, dtype=float) 
	y = stored_info_list
	y = np.array(y, dtype=float) 
	plt.plot(x, y, 'ro')
	plt.axhline(y=np.log(4), color='purple', linestyle='--', label = "ln(4)")
	plt.legend()
	plt.show()

"""
"""
# three_state_stored_info(100000)
# four_state_stored_info(100000)

	
	
	






