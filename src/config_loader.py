"""Module to load config yaml files"""
import yaml


class ConfigLoader:

    def __init__(self):
        self._config = None
        yaml.add_constructor('!join', self._join)

    def load(self, config_file):
        with open(config_file, 'rt') as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

    @staticmethod
    def _join(loader, node):
        """
        Constructor to join placeholders and text
        """
        return ''.join([str(i) for i in loader.construct_sequence(node)])

    def __getitem__(self, item):
        return self._config[item]
