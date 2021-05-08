import logging
import logging.config
import json
import hydra
from omegaconf import OmegaConf
import yaml
from heart.parameters import TrainingPipelineParameters
from heart.data import make_dataset
from heart.features import save_transformer
from heart.models import *
import pandas as pd
import joblib

LOGGING_CONFIG_PATH = 'configs/logging_config.yaml'
COMPRESS_MODE = 9


def predict_with_loging(model, x: pd.DataFrame, y: pd.Series, log: str, save_path: str):
    predicts = predict(model, x)
    metrics = evaluate(predicts, y)
    logging.info(f'{log} metrics: {metrics}')
    with open(save_path, 'w') as fout:
        json.dump(metrics, fout)


def training_pipeline(pipeline_params: TrainingPipelineParameters):
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(pipeline_params.input_data,
                                                                  pipeline_params.dataset_splitting,
                                                                  pipeline_params.dataset_features)

    logging.debug(f'Train dataset shape: {train_x.shape}, test dataset shape: {test_x.shape}')

    model = train(train_x, train_y, pipeline_params.training)
    _ = joblib.dump(model, pipeline_params.output_model, compress=COMPRESS_MODE)

    predict_with_loging(model, test_x, test_y, 'test', pipeline_params.metrics_test)
    if pipeline_params.dataset_splitting.val_needed:
        predict_with_loging(model, val_x, val_y, 'validation', pipeline_params.metrics_valid)
    save_transformer(pipeline_params.transformer)


@hydra.main(config_path='configs', config_name='train_config_random_forest')
def run_training_pipeline(config: TrainingPipelineParameters):
    with open(config.logging_config, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))
    logging.info(f'Train pipeline started. Config :\n{{\n{OmegaConf.to_yaml(config)}}}')
    training_pipeline(config)
    logging.info('Train pipeline successfully finished')


if __name__ == '__main__':
    run_training_pipeline()
