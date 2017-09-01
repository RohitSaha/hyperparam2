import GPyOpt
import numpy as np

x_init = np.array([[1,2],
                  [2,3]])

y_init = np.array([[5],
                  [6]])

def eval_func(x):
    return x[:,0]**2 + x[:,1]**2

bds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-2,10)},
       {'name': 'var_2', 'type': 'continuous', 'domain': (-2,10)}]

#my_prob = GPyOpt.methods.BayesianOptimization(f = eval_func, X = x_init, Y = y_init, domain = bds, report_file = 'history.txt')
my_prob = GPyOpt.methods.BayesianOptimization(f = None, X = x_init, Y = y_init, domain = bds, report_file = 'history.txt')
#my_prob.run_optimization(1)
#my_prob.plot_acquisition()
#my_prob.suggested_sample

####Psuedocode
#Load tested hyperparameters (x_init) and results (y_init) into method
my_prob = GPyOpt.methods.BayesianOptimization(f = None, X = x_init, Y = y_init, domain = bds, report_file = 'history.txt')

#Get next suggested hyperparameter combo with suggested_sample method
x_init = np.vstack([x_init, my_prob.suggested_sample])

#Evaluate model at these hyperparameters and update y_init
y_init = np.vstack([y_init, eval_model(my_prob.suggested_sample)])

#Iterate
