import GPyOpt
import numpy as np

# Function that takes in data and makes a suggested next parameter combo
def next_hyps(x_dat,y_dat,vars_dom):
       my_prob = GPyOpt.methods.BayesianOptimization(f = None, X = x_dat, Y = y_dat, domain = vars_dom)
       return my_prob.suggested_sample

#### Example use:
# Input parameter combos
x_init = np.array([[1,2,-2],
                   [2,3,-1],
                   [-1,5,-4],
                   [-5,6,9]])
# Output of model evaluated at inputs
y_init = np.array([[5],
                   [6],
                   [-2],
                   [-1]])
# Details about the domain of variables. It seems like continuos variables
# should be listed first for some reason
bds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-10,10)},
       {'name': 'var_2', 'type': 'continuous', 'domain': (-10,10)},
       {'name': 'var_3', 'type': 'discrete', 'domain': tuple(range(-10,11))}]

# Display output
print next_hyps(x_init,y_init,bds)
