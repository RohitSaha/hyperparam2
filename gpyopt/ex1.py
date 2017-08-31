from GPyOpt.methods import BayesianOptimization
import numpy as np

# --- Define your problem
# def f(x): return (6*x-2)**2*np.sin(12*x-4)
bounds = [(0,5)]
inputs = np.array([1])
output = np.array([2])

# --- Solve your problem
#myBopt = BayesianOptimization(f=f, bounds=bounds)
myBopt = BayesianOptimization(f= None, X = inputs, Y = output, bounds=bounds)
myBopt.run_optimization(1)
print myBopt.suggested_sample
