"""Trainer module"""

import os
from datetime import datetime
import logging as lg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from src.logging_config import setup_logging
from src.config_loader import ConfigLoader
from src.transformers import SelectDtypeColumns, CountThresholder, CategoricalEncoder
from src.bayes_hyperopt import BayesOpt
from src.utils import persist_local_artifact

setup_logging()
CONFIG = ConfigLoader()
CONFIG.load('config/config.yaml')


class Trainer:
    """Train machine learning models and save artifacts using MLflow"""
    # TODO: Log MLflow metrics
    def __init__(self, config, seed=0):
        self._config = config
        self._seed = seed
        np.random.seed(self._seed)
        self.data = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.df_train = None
        self.df_test = None
        self.best_pipeline = None
        self.metadata = None

    @property
    def target_column(self):
        """Target column name in the dataset"""
        return self._config['target_column']

    @property
    def target_replace_mapping(self):
        """Dictionary to decode the raw target values"""
        return self._config.get('target_replace_mapping')

    @property
    def mlflow_experiment_name(self):
        """MLflow experiment name"""
        return self._config['mlflow_experiment_name']

    def run(self):
        """Complete pipeline execution for model training and persistence"""
        logger = lg.getLogger(self.load_data.__name__)
        logger.info('Starting pipeline execution')
        self.load_data()
        self.preprocessing()
        self.qa_data()
        self.split_data()
        try:
            mlflow.create_experiment(self.mlflow_experiment_name)
            logger.info('The MLflow experiment %s has been created', self.mlflow_experiment_name)
        except MlflowException:
            logger.info('The MLflow experiment %s has been set', self.mlflow_experiment_name)
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run():
            self.train_model()
            self.evaluate_model()
            self.persist_artifacts()

    def load_data(self):
        """Load the dataset as pandas `DataFrame` object"""
        # TODO: Implement loading process from S3
        logger = lg.getLogger(self.load_data.__name__)
        logger.info('Loading data')
        self.data = pd.read_csv(self._config['local_data_path'])
        if self.target_replace_mapping:
            self.data[self.target_column].replace(self.target_replace_mapping, inplace=True)
            logger.info('The target has been encoded using: %s', self.target_replace_mapping)
        logger.info('Data columns: %s', self.data.columns)
        logger.info('Data head: \n %s', self.data.head())

    def preprocessing(self):
        """Clean the input dataset for the training process.

        The following steps are executed:
            - Drop the columns specified in the config file
            - Remove columns with only null values
            - Fill the null values of categorical columns as 'nan`
        """
        logger = lg.getLogger(self.preprocessing.__name__)
        logger.info('Preprocessing data:')
        self.data.drop(columns=self._config['drop_columns'], inplace=True)
        self.data.dropna(axis='columns', how='all', inplace=True)
        self.data.select_dtypes(exclude=['number', 'bool']).fillna('nan')
        logger.info('Data head: \n %s', self.data.head())
        logger.info(self.data.columns)

    def qa_data(self):
        """Analyze the quality of the data to detect bugs at the beginning"""
        logger = lg.getLogger(self.qa_data.__name__)
        self.categorical_columns = set(self.data.select_dtypes(exclude=['number', 'bool']).columns)
        logger.info('Categorical columns: [%s]', self.categorical_columns)
        self.numerical_columns = set(self.data.select_dtypes(include=['number', 'bool']).columns)
        logger.info('Numerical columns: [%s]', self.numerical_columns)

    def split_data(self):
        """Split the data into training and test set"""
        logger = lg.getLogger(self.split_data.__name__)
        self.df_train, self.df_test = train_test_split(self.data, train_size=self._config['train_size'])
        logger.info('Training dataset shape: %s head: \n %s', self.df_train.shape, self.df_train.head())
        logger.info('Train target distribution: \n %s', self.df_train[self.target_column].mean())
        logger.info('Test dataset shape: %s head: \n %s', self.df_test.shape, self.df_test.head())
        logger.info('Test target distribution: \n %s', self.df_test[self.target_column].mean())

    def _create_pipeline(self, search_space):
        """
        Create the sklearn model `pipeline` object.

        The pipeline consist in two main steps; encoding and scoring.
        The former is global and apply a transformation of numerical
        and categorical features if needed. The former is the model
        class that generates predictions.

        Parameters
        ----------
        search_space: dict
            Dictionary including the parameters space for optimization

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline for feature encoding and model scoring
        """
        # TODO: Parametrize for multiple and different input model classes
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
        """Train the model and select the better parameters.

        Bayesian hyperparameter tuning is applied to select the optimal
        parameters for both feature encoding and model scoring.
        """
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
        """Evaluate the model performance"""
        # TODO: Add a baseline to compare and sklearn metrics from config
        logger = lg.getLogger(self.evaluate_model.__name__)
        y_true = self.df_test[self.target_column].values
        y_pred = self.best_pipeline.predict(self.df_test.drop(columns=self.target_column))
        evaluation_report = classification_report(y_true, y_pred)
        logger.info('Evaluation report: `\n %s', evaluation_report)

    def persist_artifacts(self):
        """Persist the metadata and the model"""
        logger = lg.getLogger(self.evaluate_model.__name__)
        model_file = os.path.join(self._config['local_artifacts_path'],
                                  f"model_{datetime.now().strftime('%Y%m%d%H%M')}.pkl")
        # persist_local_artifact(self.best_pipeline, model_file)
        mlflow.sklearn.log_model(self.best_pipeline, 'sk_model', self._config['conda_environment_path'])
        logger.info('Model artifact stored in %s', model_file)


if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.run()
