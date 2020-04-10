"""Bayesian hyperparameter tuning module"""

from sklearn.model_selection import cross_val_score
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK  # , STATUS_FAIL


class BayesOpt:
    """
    Bayesian hyperparameter tuning of `scikit-learn` pipelines.

    The encoding process is static and the parameters space is defined in
    the attribute `_generic_space`. It does not depend on the selected
    model. The following parameters are optimized:
        - Relative frequency for the CountThresholder
        - The categorical encoder choice (`onehot` or `mean`)
        - How to fill the nan values for the numerical features
        - Model hyperparameters defined in the config file
    """

    def __init__(self, config_loader):
        self._config_loader = config_loader
        self._n_iterations = self._config_loader['n_iterations']
        self._generic_space = {'countthresholder': {'min_rel_freq': hp.uniform('min_rel_freq', 0.0, 1.0)},
                               'categoricalencoder': {'encoder': hp.choice('encoder', ['onehot', 'mean'])},
                               'simpleimputer': {'strategy': hp.choice('strategy', ['mean', 'median'])}}

    @property
    def model_settings(self):
        """Model hyperparameter space"""
        # TODO: Implement logic to handle more than one model at the same time and unit tests
        model_space = {}
        model_parameters = self._config_loader['model_parameters']
        for param in model_parameters.keys():
            if 'uniform' in model_parameters[param]:
                min_value, max_value = model_parameters[param]['uniform']
                model_space[param] = hp.uniform(param, min_value, max_value)
            elif 'choice' in model_parameters[param]:
                model_space[param] = hp.choice(param, model_parameters[param]['choice'])
        return model_space

    @property
    def search_space(self):
        """Pipeline hyperparameter space"""
        model_space = {'logisticregression': self.model_settings}
        return {**self._generic_space, **model_space}

    def optimization(self, pipeline, X, y):
        """
        Find the optimal pipeline hyperparameters.

        The objetive functions are defined inside the method and
        `_n_iterations` are executed sequentially to find the best
        hyperparameter space, which is returned.

        Parameters
        ----------
        pipeline : function
            Function that defines the sklearn.pipeline.Pipeline
        X : pandas.DataFrame
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like
            Target vector relative to X.

        Returns
        -------
        dict
            Best hyperparameters for the objetive function in the
            validation set
        """

        def objective_function(search_space):
            # TODO: Add custom metrics for evaluation
            loss_avg = cross_val_score(pipeline(search_space), X, y).mean()
            status = STATUS_OK
            return {'loss': loss_avg, 'status': status}
        trials = Trials()
        best_trial = fmin(objective_function, self.search_space, algo=tpe.suggest, max_evals=self._n_iterations,
                          trials=trials)
        best_parameters = space_eval(self.search_space, best_trial)
        return best_parameters
