import hyperopt
from hyperopt import Trials, Domain

domain = Domain(fn = None)

docs = hyperopt.rand.suggest(range(10), domain, Trials(), seed=123)
