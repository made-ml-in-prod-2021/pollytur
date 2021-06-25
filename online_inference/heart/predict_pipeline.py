import pandas as pd
import joblib
import pickle
from heart.parameters.dataset_params import FeatureTypes
from typing import Tuple, Optional

params = FeatureTypes(numerical=['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
                      target=None,
                      one_hot_encoding=['sex', 'cp', 'fbs', 'slope', 'thal', 'restecg', 'exang'],
                      categorical=None)  # FeatureTypes
TRANSFORMER = None
MODEL = None


def load_transformer(path: str) -> None:
    with open(path, 'rb') as fout:
        transformer = pickle.load(fout)
    return transformer


def load_model(path: str) -> None:
    return joblib.load(path)


def setup_models(path_transformer: str, path_model: str):
    global TRANSFORMER, MODEL
    TRANSFORMER = load_transformer(path_transformer)
    MODEL = load_model(path_model)
    return MODEL


def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
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
    x = TRANSFORMER.transform(x)
    return x, y


def predict_pipeline(predict_x: pd.DataFrame):
    predict_x, _ = preprocess_dataset(predict_x)
    prediction = MODEL.predict(predict_x)
    return prediction
