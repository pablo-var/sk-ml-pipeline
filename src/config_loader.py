"""Module to load config yaml files"""

import yaml


class ConfigLoader:
    """Load config files with YAML format"""
    def __init__(self):
        self._config = None
        yaml.add_constructor('!join', self._join)

    def load(self, config_file):
        """
        Load a specific YAML file.

        Parameters
        ----------
        config_file : str
            YAML file to be loaded
        """
        with open(config_file, 'rt') as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

    @staticmethod
    def _join(loader, node):
        """Constructor to join placeholders and text"""
        return ''.join([str(i) for i in loader.construct_sequence(node)])

    def __getitem__(self, item):
        return self._config[item]

    def get(self, key):
        return self._config.get(key, None)
