"""Bayesian hyperparameter tuning module
"""
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK #, STATUS_FAIL


class BayesOpt:
    """
    Parameters to hyperopt
    - Categorical: Relative frequency (%) to encode as 'other'
    - Categorical encoder (unify mean-encoding, one-hot, and label-encoding)
    - Create special is_na feature or not for numerical and fillna for numerical features (mean, average)
    - Model hyperparameters defined in the config files

    """

    def __init__(self, config_loader):
        self.n_iterations = config_loader['n_iterations']
        self.prob_parameters_space = None

    def _define_parameters_prob_space(self):
        generic_space = {'countthresholder': {'min_rel_freq': hp.uniform('min_rel_freq', 0.0, 1.0)},
                         'categoricalencoder': {'encoder': hp.choice('encoder', ['onehot', 'mean'])},
                         'simpleimputer': {'strategy': hp.choice('strategy', ['mean', 'median'])}}
        config_space = {'logisticregression': {'C': hp.uniform('C', 0.2, 1.0),
                                               'penalty': hp.choice('penalty', ['l1', 'l2'])}}
        self.search_space = {**generic_space, **config_space}

    def _load_models_candidates(self):
        pass

    def optimization(self, pipeline, X, y):
        def objective_function(search_space):
            loss_avg = cross_val_score(pipeline(search_space), X, y).mean()
            status = STATUS_OK
            return {'loss': loss_avg, 'status': status}
        self._define_parameters_prob_space()
        trials = Trials()
        best = fmin(objective_function, self.search_space, algo=tpe.suggest, max_evals=10, trials=trials)
        return best
