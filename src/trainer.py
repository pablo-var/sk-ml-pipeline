"""Trainer module"""


class Trainer:

    def __init__(self):

        self._config = None
        self.model_class = None
        self.model = None
        self.parameters_space = None
        self.target_name = None
        self.data = None
        self.train = None
        self.valid = None
        self.metadata = None

    def run(self):
        pass

    def _load_data(self):
        pass

    def _basic_preprocessing(self):
        """Fillna NaN in categorical features as 'nan'
        """

    def _qa_data(self):
        pass

    def _split_data(self):
        pass

    def _train_model(self):
        pass

    def _evaluate_model(self):
        pass

    def _save_artifacts(self):
        pass
