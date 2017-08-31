# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np

# --- Define your problem
def my_func(x,y): return (6*x-2)**2*np.sin(12*x-4)+y**2
bounds = [{'name': 'x','type': 'continuous', 'domain': (0,1)},
          {'name': 'y','type': 'continuous', 'domain': (0,1)}]

# --- Solve your problem
myBopt = BayesianOptimization(f=my_func, domain=bounds)
myBopt.run_optimization(max_iter=15)
myBopt.plot_acquisition()
