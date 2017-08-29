# KERAS IMPORTS ASSUMED
from hyperopt import fmin, tpe, hp, STATUS_OK
# tpe is the search method, fmin is optimization method which will wrap the model

# Defining the model in a function which accepts the Hyperparameter

def model(x):
	loss = x[0]+x[1]
	# This function needs to return a dictinary of loss and some status flags to hyperopt
	return {'loss':loss, 'status': STATUS_OK}


# Note that we are ooptimizing the test loss so it will give best parameter setting which is generalizable
# Now the model is defined, let's define the search space for each of the paramter. Search space is a tuple with each entry being the hyperparamter
space = (
	hp.uniform('x', 0, 1),
	hp.uniform('y', 0, 1)
	)

# choice is a bernoulli random variable for selecting between relu and tanh activation
# Now lets define the hyperopt function which will search for best paramter setting among all

best = fmin(model, space=space, algo=tpe.suggest, max_evals=10)
# best will have the best setting of parameters which optimizes test loss
# This setting can be used as final setting
print best
