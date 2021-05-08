from predict_pipeline import *
from heart.parameters import FeatureTypes, PredictionParameters
import logging.config
import os

FEATURES = FeatureTypes(target='target',
                        categorical=[],
                        numerical=['age', 'trestbps', 'trestbps', 'thalach', 'oldpeak', 'ca'],
                        one_hot_encoding=['sex', 'thal', 'slope', 'cp', 'fbs', 'restecg', 'exang'])

PREDICT_PARAMS = PredictionParameters(logging_config='configs/logging_config.yaml',
                                      data='data/raw/heart.csv',
                                      model='models/model_logreg.pkl',
                                      output='models/output.csv',
                                      transformer='models/transformer.pkl',
                                      dataset_features=FEATURES)


def test_predict_pipeline():
    with open(PREDICT_PARAMS.logging_config, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))
    predict_pipeline(PREDICT_PARAMS)
    assert os.path.exists(PREDICT_PARAMS.output), 'Predictions are not saved'


def test_run_predict_pipeline():
    run_predict_pipeline(PREDICT_PARAMS)
    assert os.path.exists(PREDICT_PARAMS.output), 'Predictions are not saved'
