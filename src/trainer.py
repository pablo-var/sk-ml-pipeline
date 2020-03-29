"""Trainer module"""

import logging as lg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.logging_config import setup_logging
from src.config_loader import ConfigLoader
from src.transformers import SelectDtypeColumns, CountThresholder, CategoricalEncoder
from src.bayes_hyperopt import BayesOpt

setup_logging()
CONFIG = ConfigLoader()
CONFIG.load('config/config.yaml')


class Trainer:

    def __init__(self, config, seed=0):

        self._config = config
        self._seed = seed
        np.random.seed(self._seed)
        self.data = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.best_pipeline = None
        self.parameters_space = None
        self.df_train = None
        self.df_test = None
        self.metadata = None

    @property
    def target_column(self):
        return self._config['target_column']

    def run(self):
        self.load_data()
        self.preprocessing()
        self.qa_data()
        self.split_data()
        self.train_model()
        self.evaluate_model()

    def load_data(self):
        # TODO: Implement loading process from S3
        logger = lg.getLogger(self.load_data.__name__)
        logger.info('Loading data')
        # https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
        self.data = pd.read_csv(self._config['local_data_path'])
        self.data.diagnosis.replace({'M': 1, 'B': 0}, inplace=True)
        logger.info('Data columns: %s', self.data.columns)
        logger.info('Data head: \n %s', self.data.head())

    def preprocessing(self):
        """
        - Remove full NaN columns
        - Fillna NaN in categorical features as 'nan'
        """
        logger = lg.getLogger(self.preprocessing.__name__)
        logger.info('Preprocessing data')
        self.data.drop(columns='id', inplace=True)
        self.data.dropna(axis='columns', how='all', inplace=True)
        # self.data['cat'] = np.random.choice(['a', 'b'], size=self.data.shape[0])
        self.data.select_dtypes(exclude=['number', 'bool']).fillna('nan')
        logger.info('Data head: \n %s', self.data.head())
        logger.info(self.data.columns)

    def qa_data(self):
        logger = lg.getLogger(self.qa_data.__name__)
        # TODO: Add quality data analysis
        self.categorical_columns = set(self.data.select_dtypes(exclude=['number', 'bool']).columns)
        logger.info('Categorical columns: [%s]', self.categorical_columns)
        self.numerical_columns = set(self.data.select_dtypes(include=['number', 'bool']).columns)
        logger.info('Numerical columns: [%s]', self.numerical_columns)

    def split_data(self):
        logger = lg.getLogger(self.split_data.__name__)
        self.df_train, self.df_test = train_test_split(self.data, train_size=self._config['train_size'])
        logger.info('Training dataset shape: %s head: \n %s', self.df_train.shape, self.df_train.head())
        logger.info('Train target distribution: \n %s', self.df_train[self.target_column].mean())
        logger.info('Test dataset shape: %s head: \n %s', self.df_test.shape, self.df_test.head())
        logger.info('Test target distribution: \n %s', self.df_test[self.target_column].mean())

    def _create_pipeline(self, search_space):
        # TODO: Parametrize the model defined
        numerical_encoder = make_pipeline(SelectDtypeColumns(include=['number', 'bool']),
                                          SimpleImputer(**search_space['simpleimputer']))
        categorical_encoder = make_pipeline(SelectDtypeColumns(exclude=['number', 'bool']),
                                            CountThresholder(**search_space['countthresholder']),
                                            CategoricalEncoder(**search_space['categoricalencoder']))
        if self.categorical_columns and self.numerical_columns:
            encoder = make_union(numerical_encoder, categorical_encoder)
        elif self.numerical_columns:
            encoder = numerical_encoder
        else:
            encoder = categorical_encoder

        model_object = LogisticRegression(solver='liblinear', **search_space['logisticregression'])
        pipeline = make_pipeline(encoder, model_object)
        return pipeline

    def train_model(self):
        logger = lg.getLogger(self.train_model.__name__)
        bayes = BayesOpt(self._config)
        x = self.df_train.drop(columns=self.target_column)
        y = self.df_train[self.target_column].values
        logger.info('X dataset shape: %s head: \n %s', x.shape, x)
        best_space = bayes.optimization(self._create_pipeline, x, y)
        logger.info(best_space)
        self.best_pipeline = self._create_pipeline(best_space)
        self.best_pipeline.fit(x, y)

    def evaluate_model(self):
        logger = lg.getLogger(self.evaluate_model.__name__)
        y_true = self.df_test[self.target_column].values
        y_pred = self.best_pipeline.predict(self.df_test.drop(columns=self.target_column))
        evaluation_report = classification_report(y_true, y_pred)
        logger.info('Evaluation report: `\n %s', evaluation_report)

    def save_artifacts(self):
        pass


if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.run()
