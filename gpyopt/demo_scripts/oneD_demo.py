#You may need to run these to run this code:
#pip install gpy
#pip install gpyopt

from GPyOpt.methods import BayesianOptimization
import numpy as np
from func_file import simp_func

bounds = [(-2,3)]

#x_init = np.array([[0.25],[-1],[0.5]])
x_init = np.array([[1],[0.5],[-1.5]])
y_init = simp_func(x_init)
it = 1
import ipdb; ipdb.set_trace()

# --- Solve your problem
while it < 15:

    #Input data into BayesianOptimization method
    myBopt = BayesianOptimization(f=None, X = x_init, Y = y_init, bounds=bounds)

    #Display the next suggested sample from our function
    print "The next suggested sample is: " + str(float(myBopt.suggested_sample[[0]]))

    #Plot the current data, function distribution, and expected improvement curve
    myBopt.plot_acquisition()

    #Append new sample and output to data
    x_init = np.vstack([x_init,myBopt.suggested_sample])
    y_init = np.vstack([y_init,simp_func(myBopt.suggested_sample)])
    it += 1
