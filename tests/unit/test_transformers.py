"""Module for unit testing the custom transformers"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.transformers import SelectDtypeColumns, CountThresholder, CategoricalEncoder
from category_encoders.target_encoder import TargetEncoder


@pytest.fixture(scope="module")
def df_test():
    df_test = pd.DataFrame({'int': [1, 2, 3], 'bool': [True, False, True], 'cat': ['a', 'b', 'c']})
    return df_test


@pytest.fixture(scope="module")
def columns():
    col_1 = ['one_1',
             'two_1', 'two_1',
             'three_1', 'three_1', 'three_1',
             'four_1', 'four_1', 'four_1', 'four_1',
             'five_1', 'five_1', 'five_1', 'five_1', 'five_1']
    col_2 = ['seven_2', 'seven_2', 'seven_2', 'seven_2', 'seven_2', 'seven_2', 'seven_2',
             'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2']
    columns = np.array([col_1, col_2], dtype='O').transpose()
    return columns


@pytest.fixture(scope="module")
def target():
    target = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    return target


def test__SelectDtypeColumns__init__correct_instantiation():
    transformer = SelectDtypeColumns(include=['number', 'bool'])
    assert transformer.include_ == ['number', 'bool']


def test__SelectDtypeColumns___include_numeric_bool_correct_response(df_test):
    transformer = SelectDtypeColumns(include=['number', 'bool'])
    assert np.array_equal(transformer.fit_transform(df_test), np.array([[1, True], [2, False], [3, True]]))


def test__SelectDtypeColumns___exclude_numeric_bool_correct_response(df_test):
    transformer = SelectDtypeColumns(exclude=['number', 'bool'])
    assert np.array_equal(transformer.fit_transform(df_test), np.array([['a'], ['b'], ['c']]))


def test__CountThresholder__init__correct_instantiation():
    transformer = CountThresholder(max_categories=5, min_rel_freq=0.01, category_name='test_name')
    assert transformer.max_categories_ == 5
    assert transformer.min_rel_freq_ == 0.01
    assert transformer.category_name_ == 'test_name'


def test__CountThresholder__init__max_categories_incorrect_instantiation():
    with pytest.raises(AssertionError, match='max_categories must be greater than 1'):
        CountThresholder(max_categories=-3)


def test__CountThresholder__init__min_rel_freq_incorrect_instantiation_invalid_type():
    with pytest.raises(AssertionError, match='min_rel_freq must be float'):
        CountThresholder(min_rel_freq='3')


def test__CountThresholder__init__min_rel_freq_incorrect_instantiation_negative_freq():
    with pytest.raises(AssertionError, match='min_rel_freq should be between 0.0 and 1.0'):
        CountThresholder(min_rel_freq=5.0)


def test__CountThresholder__init__category_name_incorrect_instantiation():
    with pytest.raises(AssertionError, match='category_name should be str'):
        CountThresholder(category_name=5)


def test__CountThresholder__fit__transform__no_array_input_assertion():
    transformer = CountThresholder()
    with pytest.raises(AssertionError, match='X must be a np.ndarray'):
        _ = transformer.fit(['two', 'two'])
    with pytest.raises(AssertionError, match='X must be a np.ndarray'):
        _ = transformer.fit(1)
    array = np.array(['two', 'two']).reshape(2, 1)
    transformer.fit(array)
    with pytest.raises(AssertionError, match='X must be a np.ndarray'):
        _ = transformer.transform(['two', 'two'])
    with pytest.raises(AssertionError, match='X must be a np.ndarray'):
        _ = transformer.transform(1)


def test__CountThresholder__fit__1dim_array_input_assertion():
    transformer = CountThresholder()
    with pytest.raises(AssertionError, match='X must have dimension 2'):
        transformer.fit(np.array(['two', 'two']))
    array = np.array(['two', 'two']).reshape(2, 1)
    transformer.fit(array)
    with pytest.raises(AssertionError, match='X must have dimension 2'):
        transformer.transform(np.array(['two', 'two']))


def test__CountThresholder__fit__transform__same_number_features():
    transformer = CountThresholder()
    array_1_features = np.array(['two', 'two']).reshape(2, 1)
    transformer.fit(array_1_features)
    array_2_features = np.array([['two', 'two'], ['two', 'two']]).reshape(2, 2)
    with pytest.raises(AssertionError, match='the number of columns is not correct'):
        transformer.transform(array_2_features)


def test__CountThresholder__fit__transform__correct_response_max_categories(columns):
    transformer = CountThresholder(max_categories=1)
    transformer.fit(columns)
    train = transformer.transform(columns)
    assert (list(train[:, 0]) == ['other', 'other', 'other', 'other', 'other', 'other',
                                  'other', 'other', 'other', 'other', 'five_1', 'five_1',
                                  'five_1', 'five_1', 'five_1'])
    assert (list(train[:, 1]) == ['other', 'other', 'other', 'other', 'other', 'other', 'other',
                                  'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2',
                                  'eight_2', 'eight_2'])


def test__CountThresholder__fit__transform__correct_response_min_rel_freq_float(columns):
    transformer = CountThresholder(min_rel_freq=0.2)
    transformer.fit(columns)
    train = transformer.transform(columns)
    assert (list(train[:, 0]) == ['other', 'other', 'other',
                                  'three_1', 'three_1', 'three_1',
                                  'four_1', 'four_1', 'four_1', 'four_1',
                                  'five_1', 'five_1', 'five_1', 'five_1', 'five_1'])
    assert (list(train[:, 1]) == ['seven_2', 'seven_2', 'seven_2', 'seven_2', 'seven_2',
                                  'seven_2', 'seven_2', 'eight_2', 'eight_2', 'eight_2',
                                  'eight_2', 'eight_2', 'eight_2', 'eight_2', 'eight_2'])


def test__CountThresholder__fit_transform__correct_response_new_categories(columns):
    transformer = CountThresholder()
    transformer.fit(columns)
    test = transformer.fit_transform(np.array(['new', 'new'], dtype='O').reshape(2, 1))
    assert list(test) == ['other', 'other']


def test__CountThresholder__fit_transform__method_correct_response():
    transformer = CountThresholder(min_rel_freq=0.1)
    array = np.array(['two', 'two']).reshape(2, 1)
    assert list(transformer.fit_transform(array)) == ['two', 'two']


@pytest.mark.parametrize("encoder_name, python_class", [('onehot', OneHotEncoder), ('mean', TargetEncoder)])
def test_CategoricalEncoder_correct_instantiation(encoder_name, python_class):
    transformer = CategoricalEncoder(encoder=encoder_name)
    assert isinstance(transformer.encoder_class, python_class)


def test_CategoricalEncoder_correct_instantiation_correct_assertion():
    with pytest.raises(AssertionError, match="the encoder must be: \['onehot', 'mean'\]"):
        CategoricalEncoder(encoder='non_valid_encoder')


@pytest.mark.parametrize("encoder_name, python_class", [('onehot', OneHotEncoder), ('mean', TargetEncoder)])
def test_CategoricalEncoder_correct_execution(encoder_name, python_class, columns, target):
    transformer = CategoricalEncoder(encoder=encoder_name)
    transformer.fit(columns, target)


def test_CategoricalEncoder_target_encoder_correct_assertion(columns):
    transformer = CategoricalEncoder(encoder='mean')
    with pytest.raises(AssertionError, match='the target is needed for the mean encoding'):
        transformer.fit(columns)
