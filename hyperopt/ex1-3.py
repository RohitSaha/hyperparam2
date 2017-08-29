import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(x,y):
    return {
        'loss': x ** 2 * y ** 2,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments':
            {'time_module': pickle.dumps(time.time)}
        }
trials = Trials() #this stores all of the above info
space={
    'x': hp.uniform('x', -10, 10),
    'y': hp.uniform('y',-10,10)
    }
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print best
