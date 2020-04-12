"""Unit tests for the config_loader module"""

from src.config_loader import ConfigLoader


def test__config_loader__correct_response():
    config_test_path = 'tests/resources/config.yaml'
    config = ConfigLoader()
    config.load(config_test_path)
    assert config['string_key'] == 'value'
    assert config['int_key'] == 1
    assert config['float_key'] == 1.5
    assert config['list_key'] == [1, 'value2', 3]
    assert list(config['model_parameters'].keys()) == ['C', 'penalty']
    assert config['model_parameters']['C']['uniform'] == [0.2, 1]
    assert config['model_parameters']['penalty']['choice'] == ['l1', 'l2']
    assert config.get('non_existent_key') is None
