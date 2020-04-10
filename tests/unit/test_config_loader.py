"""
Unit tests for the config_loader module
"""

from src.config_loader import ConfigLoader


def test__config_loader__correct_response():
    config_test_path = 'tests/resources/config.yaml'
    config = ConfigLoader()
    config.load(config_test_path)
    assert config['data_url'] == 'http:data'
    assert config['model_type'] == 'linear_model.LogisticRegression'
    assert config['n_iterations'] == 10
    assert list(config['model_parameters'].keys()) == ['C', 'penalty']
    assert config['model_parameters']['C']['uniform'] == [0.2, 1]
    assert config['model_parameters']['penalty']['choice'] == ['l1', 'l2', 'elasticnet']
