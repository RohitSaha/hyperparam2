import GPy
import GPyOpt
import numpy as np

bounds = [(1,5),(0,240)]
x = 2
y = 5

myBopt = GPyOpt.methods.BayesianOptimization(f = None, bounds = bounds,
X = x, Y = y, acquisition_type='EI')
myBopt.run_optimization(1)
# a = myBopt.suggested_sample
