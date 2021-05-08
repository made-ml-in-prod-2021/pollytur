import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Optional
from heart.parameters import TrainingParameters
import importlib


def make_model(module: str, method: str, model_parameters: Optional[dict]):
    clas_mod = importlib.import_module(module)
    clf = getattr(clas_mod, method)()
    if model_parameters:
        clf.set_params(**model_parameters)
    return clf


def train(features: pd.DataFrame, target: pd.Series, train_params: TrainingParameters):
    clf = make_model(train_params.module, train_params.method, train_params.model_parameters)
    clf.fit(features, target)
    return clf


def predict(clf, features: pd.DataFrame) -> np.ndarray:
    return clf.predict(features)


def evaluate(prediction: np.ndarray, targets: pd.Series) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(targets, prediction),
        'report': classification_report(targets, prediction)
    }
