import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression

from src.transformers import SelectDtypeColumns, CountThresholder, CategoricalEncoder
from src.bayes_hyperopt import BayesOpt
from src.config_loader import ConfigLoader


@pytest.fixture(scope="module")
def config():
    config_test_path = 'tests/resources/config.yaml'
    config = ConfigLoader()
    config.load(config_test_path)
    return config


@pytest.fixture(scope="module")
def X_float():
    data = {'x1_float': [np.nan, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            'x2_float': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, np.nan]}
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def X_int():
    data = {'x1_int': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'x2_int': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def X_bool():
    data = {'x1_bool': [True, False, False, False, True, True, True, True, False, True]}
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def X_cat_float_int__bool():
    data = {'x_cat': ['category_A', 'category_A', 'category_B', 'category_A', 'other', 'category_A', 'category_A',
                      'category_B', 'category_A', 'other'],
            'x_float': [np.nan, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            'x_int': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'x_bool': [True, False, False, False, True, True, True, True, False, True]
            }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def y_testing():
    return np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 1])


def test_bayesopt___n_iterations(config):
    bayes = BayesOpt(config)
    assert bayes._n_iterations == 2


def test_bayesopt___generic_space(config):
    bayes = BayesOpt(config)
    assert list(bayes._generic_space.keys()) == ['countthresholder', 'categoricalencoder', 'simpleimputer']


def test_bayesopt___load_model_settings__correct_response(config):
    bayes = BayesOpt(config)
    assert list(bayes.model_settings.keys()) == ['C', 'penalty']
    assert bayes.model_settings.values()


def test_bayesopt___define_search_space__correct_response(config):
    bayes = BayesOpt(config)
    assert list(bayes.search_space.keys()) == ['countthresholder', 'categoricalencoder', 'simpleimputer',
                                               'logisticregression']


def test__bayesopt__optimization__cat_features__correct_response(config, X_int, X_float, X_bool, y_testing):
    bayes = BayesOpt(config)

    def create_pipeline(search_space):
        pipeline = make_pipeline(SimpleImputer(**search_space['simpleimputer']),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline

    bayes.optimization(create_pipeline, X_float, y_testing)
    bayes.optimization(create_pipeline, X_int, y_testing)
    bayes.optimization(create_pipeline, X_bool, y_testing)


def test__bayesopt__optimization__cat_and_numeric_features__correct_response(config, X_cat_float_int__bool, y_testing):
    bayes = BayesOpt(config)

    def create_pipeline(search_space):
        pipeline = make_pipeline(make_union(make_pipeline(SelectDtypeColumns(exclude=['number', 'bool']),
                                                          CountThresholder(**search_space['countthresholder']),
                                                          CategoricalEncoder(**search_space['categoricalencoder'])),
                                            make_pipeline(SelectDtypeColumns(include=['number', 'bool']),
                                                          SimpleImputer(**search_space['simpleimputer']))),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline
    bayes.optimization(create_pipeline, X_cat_float_int__bool, y_testing)
