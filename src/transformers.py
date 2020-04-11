"""Custom transformers for sklearn pipelines"""

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder


class SelectDtypeColumns(TransformerMixin):
    """Class to select columns of a specific type from a dataframe.

    Transformer that selects columns of a specific type from a dataframe
    and returns a numpy array. At least one of the parameters must be
    supplied. The implementation follow the scikit-learn structure.

    Parameters
    ----------
    include : scalar or list-like
        A selection of dtypes or strings to be included.
    exclude : scalar or list-like
        A selection of dtypes or strings to be excluded.
    """
    def __init__(self, include=None, exclude=None):
        self.include_ = include
        self.exclude_ = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.select_dtypes(include=self.include_, exclude=self.exclude_).values
        return X_transformed


class CountThresholder(TransformerMixin):
    """Encode low frequency classes as a single one

    Encode less frequent categorical variables to a new category. The input must be a numpy array with rows and columns.
    For each column, the transformed output is a column where the input category is retained whether:
    - it is in the top max_categories_ ordered by frequency in the training data
    - it has occurred at least a minimum relative frequency the training data.
    The default name for the new category is 'other'.
    Inspiration:
        - https://turi.com/products/create/docs/generated/graphlab.toolkits.feature_engineering.CountThresholder.html

    Parameters
    ----------
    max_categories : int
        If not None, build a encoding that only consider the top max categories ordered by term frequency.
    min_rel_freq : float in range [0.0, 1.0]
        Minimum relative frequency for categories.
    category_name : str (default='other')
        The value to use for the categories that do not satisfy the conditions.
    """

    def __init__(self, max_categories=None, min_rel_freq=0.0, category_name='other'):
        assert max_categories is None or max_categories >= 1, "max_categories must be greater than 1"
        assert isinstance(min_rel_freq, float), "min_rel_freq must be float"
        assert 0.0 <= min_rel_freq <= 1.0, "min_rel_freq should be between 0.0 and 1.0"
        assert isinstance(category_name, str), "category_name should be str"
        self.max_categories_ = max_categories
        self.min_rel_freq_ = min_rel_freq
        self.category_name_ = category_name
        self.top_categories = []
        self.n_features_ = None

    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray), "X must be a np.ndarray"
        assert X.ndim == 2, "X must have dimension 2"
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            feature = pd.Series(X[:, i])
            frequencies = feature.value_counts(normalize=True)
            categories = set(frequencies.index)
            if self.max_categories_ is not None:
                categories = categories.intersection(set(frequencies[:self.max_categories_].index))
            if self.min_rel_freq_ is not None:
                categories = categories.intersection(set(frequencies[frequencies >= self.min_rel_freq_].index))
            self.top_categories.append(categories)
        return self

    def transform(self, X, y=None):
        assert isinstance(X, np.ndarray), "X must be a np.ndarray"
        assert X.ndim == 2, "X must have dimension 2"
        assert X.shape[1] == self.n_features_, "the number of columns is not correct"
        X_transformed = X.copy()
        for i in range(self.n_features_):
            feature = pd.Series(X_transformed[:, i])
            feature.loc[~feature.isin(self.top_categories[i])] = self.category_name_
            X_transformed[:, i] = feature
        return X_transformed


class CategoricalEncoder(TransformerMixin):
    """Encode categorical features using different strategies

    The class is a wrapper of transformers than follow the structure
    of scikit-learn. The purpose is to apply hyperparameter optimization
    of the encoding by changing only the `encoder` name.

    Parameters
    ----------
    encoder : str
        Encoding strategy name. It must be `onehot` or `mean`

    """
    # TODO: Add OrdinalEncoder
    def __init__(self, encoder):
        self.encoder_ = encoder
        self._class_mapping = {'onehot': OneHotEncoder, 'mean': TargetEncoder}
        assert encoder in self._class_mapping.keys(), 'the encoder must be: {}'.format(list(self._class_mapping.keys()))
        self.encoder_class = self._class_mapping[encoder]()

    def fit(self, X, y=None):
        if self.encoder_ == 'mean':
            assert y is not None, 'the target is needed for the mean encoding'
        self.encoder_class.fit(X, y)
        return self

    def transform(self, X, y=None):
        if self.encoder_ == 'mean':
            return self.encoder_class.transform(X, y)
        return self.encoder_class.transform(X)
