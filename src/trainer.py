"""Trainer module"""

import logging as lg
import pandas as pd

from src.config_loader import ConfigLoader
from src.logging_config import setup_logging

setup_logging()
CONFIG = ConfigLoader()
CONFIG.load('config/config.yaml')


class Trainer:

    def __init__(self, config):

        self._config = config
        self.data = None
        # self.model_class = None
        self.model = None
        self.parameters_space = None
        self.train = None
        self.valid = None
        self.metadata = None

    @property
    def target_column(self):
        return self._config['target_column']

    def run(self):
        self.load_data()
        self.preprocessing()

    def load_data(self):
        # TODO: Implement loading process from S3
        logger = lg.getLogger(self.load_data.__name__)
        logger.info('Loading data')
        self.data = pd.read_csv(self._config['local_data_path'])
        logger.info(self.data.head())
        logger.info(self.data.columns)

    def preprocessing(self):
        """
        - Remove full NaN columns
        - Fillna NaN in categorical features as 'nan'
        """
        logger = lg.getLogger(self.preprocessing.__name__)
        logger.info('Preprocessing data')
        self.data.dropna(axis='columns', how='all', inplace=True)
        self.data.select_dtypes(exclude=['number', 'bool']).fillna('nan')
        logger.info(self.data.head())
        logger.info(self.data.columns)

    def qa_data(self):
        pass

    def split_data(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def save_artifacts(self):
        pass


if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.run()
