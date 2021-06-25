from heart.parameters.dataset_params import FeatureTypes
import pandas as pd
from typing import Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Set
import pickle


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical: Optional[List[str]], numerical: Optional[List[str]],
                 one_hot_encoding: Optional[List[str]]):
        self.categorical = categorical if categorical else []
        self.numerical = numerical if numerical else []
        self.one_hot_encoding = one_hot_encoding if one_hot_encoding else []
        self.columns_allowed: Set[str] = set()

    def fit(self, df: pd.DataFrame):
        if self.one_hot_encoding:
            for i in self.one_hot_encoding:
                for col in pd.get_dummies(df[i], prefix=str(i)).columns.to_list():
                    self.columns_allowed.add(col)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical:
            df[col] = df[col].astype('category')

        for col in self.numerical:
            df[col] = pd.to_numeric(df[col])

        for col in self.one_hot_encoding:
            one_hot = pd.get_dummies(df[col], prefix=str(col))
            df.drop(col, axis=1, inplace=True)
            df = df.join(one_hot)
        to_drop = list(set(df.columns.to_list()).
                       difference(self.columns_allowed).
                       difference(set(self.numerical)).
                       difference(set(self.categorical)))
        df.drop(to_drop, inplace=True, axis=1)
        to_add = list(self.columns_allowed.difference(set(df.columns.to_list())))
        for i in to_add:
            df[i] = 0
        return df


GLOBAL_TRANSFORMER = CustomTransformer([], [], [])


def load_transformer(path:str) -> None:
    global GLOBAL_TRANSFORMER
    with open(path, 'rb') as fout:
        GLOBAL_TRANSFORMER = pickle.load(fout)


def save_transformer(path:str) -> None:
    with open(path, 'wb') as fout:
        pickle.dump(GLOBAL_TRANSFORMER, fout)


def preprocess_dataset(df: pd.DataFrame, params: FeatureTypes, fit: bool) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    global GLOBAL_TRANSFORMER
    y = None
    x = df
    if params.target:
        assert params.target in df.columns, 'Incorrect target column name'
        y = x[params.target]
        x.drop(params.target, axis=1, inplace=True)

    if params.numerical:
        for c in params.numerical:
            assert c in df.columns, 'Incorrect numerical column names'
    if params.categorical:
        for c in params.categorical:
            assert c in df.columns, 'Incorrect categorical column names'
    if params.one_hot_encoding:
        for c in params.one_hot_encoding:
            assert c in df.columns, 'Incorrect one_hot_encoding column name'

    if fit:
        transformer = CustomTransformer(params.categorical, params.numerical, params.one_hot_encoding)
        GLOBAL_TRANSFORMER = transformer
        transformer.fit(x)
    else:
        transformer = GLOBAL_TRANSFORMER
    x = transformer.transform(x)
    return x, y
