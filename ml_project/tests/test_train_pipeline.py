import train_pipeline
import os
from heart.parameters import TrainingPipelineParameters, TrainingParameters, TrainTestSplit, FeatureTypes
from heart.models import train
from heart.data import make_dataset
import logging
import logging.config
import yaml

FEATURES = FeatureTypes(target='target',
                        categorical=[],
                        numerical=['age', 'trestbps', 'trestbps', 'thalach', 'oldpeak', 'ca'],
                        one_hot_encoding=['sex', 'thal', 'slope', 'cp', 'fbs', 'restecg', 'exang'])

PIPELINE_PARAMS = TrainingPipelineParameters(logging_config='configs/logging_config.yaml',
                                             input_data='data/raw/heart.csv',
                                             output_model='models/model_test_rf.pkl',
                                             metrics_test='models/metrics_testing_rf.json',
                                             metrics_valid='models/metrics_valid_rf.json',
                                             transformer='models/transformer.pkl',
                                             dataset_splitting=TrainTestSplit(),
                                             dataset_features=FEATURES,
                                             training=TrainingParameters())


def test_predict_with_logging():
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(PIPELINE_PARAMS.input_data,
                                                                  PIPELINE_PARAMS.dataset_splitting,
                                                                  PIPELINE_PARAMS.dataset_features)

    model = train(train_x, train_y, PIPELINE_PARAMS.training)
    train_pipeline.predict_with_loging(model, train_x, train_y, 'testing', PIPELINE_PARAMS.metrics_test)
    assert os.path.isfile(PIPELINE_PARAMS.metrics_test), 'Metric is not saved'


def test_training_pipeline():
    with open(PIPELINE_PARAMS.logging_config, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))
    train_pipeline.training_pipeline(PIPELINE_PARAMS)
    assert os.path.isfile(PIPELINE_PARAMS.metrics_test), 'Metric is not saved'
    assert os.path.isfile(PIPELINE_PARAMS.transformer), 'Transformer is not saved'
