"""Trainer module"""

import logging as lg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.transformers import SelectDtypeColumns, CountThresholder, CategoricalEncoder
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
        self.val = None
        self.metadata = None

    @property
    def target_column(self):
        return self._config['target_column']

    def run(self):
        self.load_data()
        self.preprocessing()
        self.split_data()

    def load_data(self):
        # TODO: Implement loading process from S3
        logger = lg.getLogger(self.load_data.__name__)
        logger.info('Loading data')
        self.data = pd.read_csv(self._config['local_data_path'])
        logger.info('Data columns: %s', self.data.columns)
        logger.info('Data head: \n %s', self.data.head())

    def preprocessing(self):
        """
        - Remove full NaN columns
        - Fillna NaN in categorical features as 'nan'
        """
        logger = lg.getLogger(self.preprocessing.__name__)
        logger.info('Preprocessing data')
        self.data.dropna(axis='columns', how='all', inplace=True)
        self.data.select_dtypes(exclude=['number', 'bool']).fillna('nan')
        logger.info('Data head: \n %s', self.data.head())
        logger.info(self.data.columns)

    # def qa_data(self):
    # TODO: Add quality data analysis
    #     pass

    def split_data(self):
        logger = lg.getLogger(self.split_data.__name__)
        self.train, self.val = train_test_split(self.data, train_size=self._config['train_size'])
        logger.info('Training dataset shape: %s head: \n %s', self.train.shape,  self.train.head())
        logger.info('Validation dataset shape: %s head: \n %s', self.val.shape,  self.val.head())


    @staticmethod
    def _create_pipeline(search_space):
        # TODO: Parametrize the model defined
        pipeline = make_pipeline(make_union(make_pipeline(SelectDtypeColumns(exclude=['number', 'bool']),
                                                          CountThresholder(**search_space['countthresholder']),
                                                          CategoricalEncoder(**search_space['categoricalencoder'])),
                                            make_pipeline(SelectDtypeColumns(include=['number', 'bool']),
                                                          SimpleImputer(**search_space['simpleimputer']))),
                                 LogisticRegression(solver='liblinear', **search_space['logisticregression']))
        return pipeline

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def save_artifacts(self):
        pass


if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.run()
