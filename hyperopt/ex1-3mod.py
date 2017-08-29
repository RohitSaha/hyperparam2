import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(x):
    return {
        'loss': x[0]**2 + x[1]**2,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        }
trials = Trials() #this stores all of the above info
space=(hp.uniform('x', -10, 10),hp.uniform('y',-10,10))
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)

print best
