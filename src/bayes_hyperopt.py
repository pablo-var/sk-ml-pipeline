"""Bayesian hyperparameter tuning module"""


class BayesOpt:
    """
    Parameters to hyperopt
    - Categorical: Relative frequency (%) to encode as 'other'
    - Categorical encoder
    - Create special is_na feature or not for numerical
    - Fillna for numerical features (mean, average)
    - Model hyperparameters defined in the config files

    """


def __init__(self, config_loader):
    self.n_iterations = config_loader['n_iterations']
    self.prob_parameters_space = None


def _define_parameters_prob_space(self):
    generic_prob_space = {}
    config_prob_model_space = {}
    self.prob_parameters_space = {**generic_prob_space, **config_prob_model_space}


def optimization(self, train_data, target_name):
    pass
