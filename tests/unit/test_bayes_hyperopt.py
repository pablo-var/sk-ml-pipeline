import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression

from src.transformers import CountThresholder, CategoricalEncoder
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
def y_test():
    return np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 1])


def test__A__correct_response(config, X_int, X_float, X_bool, y_test):
    bayes = BayesOpt(config)

    def create_pipeline(search_space):
        pipeline = make_pipeline(SimpleImputer(**search_space['simpleimputer']),
                                 # CategoricalEncoder(**search_space['categoricalencoder']), ##CountThresholder(**search_space['countthresholder']),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline

    bayes.optimization(create_pipeline, X_float, y_test)
    bayes.optimization(create_pipeline, X_int, y_test)
    bayes.optimization(create_pipeline, X_bool, y_test)


def test__B__correct_response(config, X_cat_float_int__bool, y_test):
    bayes = BayesOpt(config)

    # select_categorical = FunctionTransformer(lambda df: df.select_dtypes(exclude=['number', 'bool']).values)
    # select_numeric = FunctionTransformer(lambda df: df.select_dtypes(include=['number', 'bool']).values)
    class SelectDtypeColumnsTransfomer(TransformerMixin):

        def __init__(self, include=None, exclude=None):
            self.include_ = include
            self.exclude_ = exclude

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_transformed = X.select_dtypes(include=self.include_, exclude=self.exclude_).values
            return X_transformed

    def create_pipeline(search_space):
        pipeline = make_pipeline(make_union(make_pipeline(SelectDtypeColumnsTransfomer(exclude=['number', 'bool']),
                                                          CountThresholder(**search_space['countthresholder']),
                                                          CategoricalEncoder(**search_space['categoricalencoder'])),
                                            make_pipeline(SelectDtypeColumnsTransfomer(include=['number', 'bool']),
                                                          SimpleImputer(**search_space['simpleimputer']))),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline

    bayes.optimization(create_pipeline, X_cat_float_int__bool, y_test)
