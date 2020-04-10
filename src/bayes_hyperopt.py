"""Bayesian hyperparameter tuning module
"""
from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK  # , STATUS_FAIL


class BayesOpt:
    """
    Parameters to hyperopt
    - Categorical: Relative frequency (%) to encode as 'other'
    - Categorical encoder (unify mean-encoding, one-hot, and label-encoding)
    - Create special is_na feature or not for numerical and fillna for numerical features (mean, average)
    - Model hyperparameters defined in the config files

    """

    def __init__(self, config_loader):
        self._config_loader = config_loader
        self._n_iterations = self._config_loader['n_iterations']
        self._generic_space = {'countthresholder': {'min_rel_freq': hp.uniform('min_rel_freq', 0.0, 1.0)},
                               'categoricalencoder': {'encoder': hp.choice('encoder', ['onehot', 'mean'])},
                               'simpleimputer': {'strategy': hp.choice('strategy', ['mean', 'median'])}}

    def _define_search_space(self):
        model_space = {'logisticregression': self._load_models_candidates()}
        self.search_space = {**self._generic_space, **model_space}

    def _load_models_candidates(self):
        # TODO: Implement logic to handle more than one model at the same time and unit tests
        model_space = {}
        model_parameters = self._config_loader['model_parameters']
        for param in model_parameters.keys():
            if 'uniform' in model_parameters:
                min_value, max_value = model_parameters[param]['uniform']
                model_space[param] = hp.choice(param, min_value, max_value)
            elif 'choice' in model_parameters:
                model_space[param] = hp.choice(param, model_parameters[param]['choice'])
        return model_space

    def optimization(self, pipeline, X, y):
        def objective_function(search_space):
            # TODO: Add custom metrics for evaluation
            loss_avg = cross_val_score(pipeline(search_space), X, y).mean()
            status = STATUS_OK
            return {'loss': loss_avg, 'status': status}

        self._define_search_space()
        trials = Trials()
        best_trial = fmin(objective_function, self.search_space, algo=tpe.suggest, max_evals=self._n_iterations,
                          trials=trials)
        best_parameters = space_eval(self.search_space, best_trial)
        return best_parameters
