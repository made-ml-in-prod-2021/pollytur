from heart.features import CustomTransformer, preprocess_dataset
from heart.parameters import FeatureTypes
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

DATASET_PATH = 'data/raw/heart.csv'

MOCK_DATAFRAME_TRAIN = pd.DataFrame.from_dict({'a': [1, 2, 3, 4, 5],
                                               'b': [1.2, 3.4, 5.6, 7.8, 9.0],
                                               'c': ['1', '2', '3', '1', '2'],
                                               'd': [0, 0, 1, 1, 1],
                                               'e': ['1', '2', '3', '1', '2'],
                                               'f': [0, 0, 1, 1, 1],
                                               })

MOCK_DATAFRAME_TEST = pd.DataFrame.from_dict({'a': [1, 2, 3, 4, 5],
                                              'b': [1.2, 3.4, 5.6, 7.8, 9.0],
                                              'c': ['1', '2', '3', '1', '2'],
                                              'd': [0, 0, 1, 1, 1],
                                              'e': ['1', '2', '3', '4', '5'],
                                              'f': [0, 0, 1, 1, 1],
                                              })


def test_categorical_features():
    cols = ['c', 'd']
    trans = CustomTransformer(cols, None, None)
    trans.fit(MOCK_DATAFRAME_TRAIN)
    x = trans.transform(MOCK_DATAFRAME_TRAIN.copy())
    for c in cols:
        assert is_categorical_dtype(x[c]), 'Incorrect categorical types'


def test_numerical_features():
    cols = ['a', 'b']
    trans = CustomTransformer(None, cols, None)
    trans.fit(MOCK_DATAFRAME_TRAIN.copy())
    x = trans.transform(MOCK_DATAFRAME_TRAIN)
    for c in cols:
        assert is_numeric_dtype(x[c]), 'Incorrect numeric types'


def test_onehot_features():
    cols = ['e', 'f']
    trans = CustomTransformer(None, None, cols)
    trans.fit(MOCK_DATAFRAME_TRAIN)
    x = trans.transform(MOCK_DATAFRAME_TRAIN.copy())
    for c in cols:
        assert c not in x.columns.tolist(), 'Column is not deleted after 1-hot encoding'
        unique_values = set(MOCK_DATAFRAME_TRAIN[c].tolist())
        new_columns = sum([1 for i in x.columns if c in i])
        assert len(unique_values) == new_columns, 'Incorrect number of columns in 1-hot encoding'


def test_new_element_in_test_one_hot():
    trans = CustomTransformer(None, None, ['e', 'f'])
    trans.fit(MOCK_DATAFRAME_TRAIN)
    x_train = trans.transform(MOCK_DATAFRAME_TRAIN)
    x_test = trans.transform(MOCK_DATAFRAME_TEST)
    assert set(x_test.columns) == set(x_train.columns), 'Incorrect 1-hot encoding for nw values in test'


def test_extract_target():
    df = pd.read_csv(DATASET_PATH)
    params = FeatureTypes(
        categorical=None,
        numerical=None,
        one_hot_encoding=None,
        target='target'
    )
    x, y = preprocess_dataset(df, params, False)
    assert isinstance(y, pd.Series), 'Incorrect y extraction'
    assert y.shape[0] == df.shape[0], 'Incorrect shape for y extraction'


def test_preprocess_dataset():
    cols_cat = ['c', 'd']
    cols_num = ['a', 'b']
    cols_onehot = ['e', 'f']
    params = FeatureTypes(
        categorical=cols_cat,
        numerical=cols_num,
        one_hot_encoding=cols_onehot,
        target=None
    )

    x, y = preprocess_dataset(MOCK_DATAFRAME_TRAIN.copy(), params, True)
    for c in cols_onehot:
        assert c not in x.columns, 'Column is not deleted after 1-hot encoding'
        unique_values = set(MOCK_DATAFRAME_TRAIN[c])
        new_columns = sum([1 for i in x.columns if c in i])
        assert len(unique_values) == new_columns, 'Incorrect number of columns in 1-hot encoding'
    for c in cols_num:
        assert is_numeric_dtype(x[c]), 'Incorrect numeric types'
    for c in cols_cat:
        assert is_categorical_dtype(x[c]), 'Incorrect categorical types'
