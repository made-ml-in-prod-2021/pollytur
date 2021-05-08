from heart.data import *
import numpy as np
import pandas as pd
from heart.parameters import FeatureTypes, TrainTestSplit

DATASET_PATH = 'data/raw/heart.csv'

MOCK_DATAFRAME_TRAIN_NAN = pd.DataFrame.from_dict({'a': [1, 2, 3, 4, 5],
                                                   'b': [1.2, 3.4, 5.6, 7.8, 9.0],
                                                   'c': ['1', '2', '3', '1', '2'],
                                                   'd': [np.nan, 0, 1, 1, 1],
                                                   'e': ['1', '2', '3', '1', '2'],
                                                   'f': [0, 0, 1, 1, 1],
                                                   })

MOCK_DATAFRAME_TRAIN = pd.DataFrame.from_dict({'a': [1, 2, 3, 4, 5],
                                               'b': [1.2, 3.4, 5.6, 7.8, 9.0],
                                               'c': ['1', '2', '3', '1', '2'],
                                               'd': [1, 0, 1, 1, 1],
                                               'e': ['1', '2', '3', '1', '2'],
                                               'f': [0, 0, 1, 1, 1],
                                               })

PARAMS_WITH_VAL = TrainTestSplit(val_needed = True)
PARAMS_WO_VAL = TrainTestSplit()
FEATURES = FeatureTypes(target='target',
                        categorical=[],
                        numerical=['age', 'trestbps', 'trestbps', 'thalach', 'oldpeak', 'ca'],
                        one_hot_encoding=['sex', 'thal', 'slope', 'cp', 'fbs', 'restecg', 'exang'])


def test_check_nans_correct():
    failed = False
    try:
        check_for_nans(MOCK_DATAFRAME_TRAIN)
    except:
        failed = True
    assert failed is not True, 'Check for nan fails on a correct file'


def test_check_nans_fail():
    failed = False
    try:
        check_for_nans(MOCK_DATAFRAME_TRAIN_NAN)
    except:
        failed = True
    assert failed, 'Check for nan does not fail on a broken file'


def test_read_data():
    df = read_data(DATASET_PATH)
    assert isinstance(df, pd.DataFrame), 'Incorrect reading'
    assert df.shape == (303, 14), 'Incorrect reading return shape'


def test_split_data_with_val():
    df = read_data(DATASET_PATH)
    train_data, test_data, val_data = split_data(df, PARAMS_WITH_VAL)
    print('\n MY SIZES ', train_data.shape, test_data.shape, val_data.shape)
    assert int(PARAMS_WITH_VAL.val_size * df.shape[0]) in [val_data.shape[0] - 1,
                                                           val_data.shape[0], val_data.shape[0] + 1], \
        'Incorrect validation size'
    assert int(PARAMS_WITH_VAL.test_size * df.shape[0]) in [test_data.shape[0] - 1,
                                                            test_data.shape[0], test_data.shape[0] + 1], \
        'Incorrect test size'


def test_split_data_without_val():
    df = read_data(DATASET_PATH)
    train_data, test_data, val_data = split_data(df, PARAMS_WO_VAL)
    print('\n MY SIZES ', train_data.shape, test_data.shape)
    assert val_data is None, 'Validation done should not be'
    assert int(PARAMS_WO_VAL.test_size * df.shape[0]) in [test_data.shape[0] - 1,
                                                            test_data.shape[0], test_data.shape[0] + 1], \
        'Incorrect test size'


def test_make_dataset():
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(DATASET_PATH, PARAMS_WO_VAL, FEATURES)
    assert val_x is None, 'Validation done should not be'
    assert val_y is None, 'Validation done should not be'
    assert train_x.shape[0] == train_y.shape[0], 'Train sizes do not match'
    assert test_x.shape[0] == test_y.shape[0], 'Test sizes do not match'


