from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pytest

from src.transformers import CountThresholder, CategoricalEncoder
from src.bayes_hyperopt import BayesOpt
from src.config_loader import ConfigLoader


@pytest.fixture(scope="module")
def config():
    config_test_path = 'tests/resources/config.yaml'
    config = ConfigLoader()
    config.load(config_test_path)
    return config


def test__config_loader__correct_response(config):
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    bayes = BayesOpt(config)

    def create_pipeline(search_space):
        pipeline = make_pipeline(SimpleImputer(**search_space['simpleimputer']), #CategoricalEncoder(**search_space['categoricalencoder']), ##CountThresholder(**search_space['countthresholder']),
                                 SimpleImputer(**search_space['simpleimputer']),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline
        bayes.optimization(create_pipeline, X, y)
