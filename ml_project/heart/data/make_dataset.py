import pandas as pd
from sklearn.model_selection import train_test_split
from heart.parameters.dataset_params import *
from typing import Tuple, Optional
from heart.features.build_features import preprocess_dataset


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    check_for_nans(data)
    return data


def check_for_nans(df: pd.DataFrame):
    assert df.isnull().sum().sum() == 0, 'Nans in the data'


def split_data(data: pd.DataFrame, params: TrainTestSplit) \
        -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    train_data, test_data = train_test_split(
        data, test_size=params.test_size, random_state=params.random_state
    )
    val_data = None
    if params.val_needed:
        train_data, val_data = train_test_split(
            train_data, test_size=params.val_size * data.shape[0]/train_data.shape[0], random_state=params.random_state
        )

    return train_data, test_data, val_data


def make_dataset(path: str, params: TrainTestSplit, feature_params: FeatureTypes) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    data = read_data(path)
    train_data, test_data, val_data = split_data(data, params)
    val_x, val_y = None, None
    train_x, train_y = preprocess_dataset(train_data, feature_params, True)
    if val_data:
        val_x, val_y = preprocess_dataset(val_data, feature_params, False)
    test_x, test_y = preprocess_dataset(test_data, feature_params, False)
    return train_x, train_y, test_x, test_y, val_x, val_y
