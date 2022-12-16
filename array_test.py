from flexible_size_systems import Markov_flex_sys_flex_env
import csv
import numpy as np
from generating_matrices import generate_random_matrix
import os
import sys 
"""
IMPORTANT: environment size is 2. 10:30 am- just practice right now with 2 sims,
 once it works run with 1000 
"""

number_of_sims = sys.argv[1]

def array_test(number_of_sims: int, file_name: str):
    """
    for set 1: 
        Broken DB
        weak coupling
        non 0 stored info 
    """
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        header_1 = ['states in environment: ' + str(2)]
        header_2 = ["example #", "theoretical SI", "SI using MI formula", "SI using simluated steady state time slice", "2", "3" "4", "5", "6", "7", "8", "9", "10"] 
        writer.writerow(header_1)
        writer.writerow(header_2)
        print("hi")

        row = []
        for i in range(10):            
            s = 0
            while s < number_of_sims:
                row = []
                row.append("example: " + str(i + 1))
                row.append(np.random.uniform(0, 1, 1))
                row.append(np.random.uniform(0, 1, 1))
                row.append(np.random.uniform(0, 1, 1))
                for sys_size in range (2, 11):
                    row.append(sys_size)
                s = s + 1
                writer.writerow(row) # indented this once

file_name = "array_test" + sys.argv[1] + ".csv"
array_test(10, file_name)


