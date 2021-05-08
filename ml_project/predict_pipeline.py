import logging
import logging.config
import pandas as pd
import joblib
import hydra
from omegaconf import OmegaConf
import yaml
from heart.parameters import PredictionParameters
from heart.data import read_data, preprocess_dataset
from heart.models import *
from heart.features import load_transformer

LOGGING_CONFIG_PATH = 'configs/logging_config.yaml'


def predict_pipeline(pipeline_params: PredictionParameters):
    predict_x = read_data(pipeline_params.data)
    load_transformer(pipeline_params.transformer)
    logging.debug(f'Predict dataset shape after loading: {predict_x.shape}')
    predict_x, _ = preprocess_dataset(predict_x, pipeline_params.dataset_features, False)
    logging.debug(f'Predict dataset shape after preprocessing: {predict_x.shape}')
    model = joblib.load(pipeline_params.model)
    logging.debug(f'Predict Model successfully loaded')
    prediction = predict(model, predict_x)
    prediction_json = {'predictions: ':  prediction}
    pd.DataFrame.from_dict(prediction_json).to_csv(pipeline_params.output)


@hydra.main(config_path='configs', config_name='predict_config')
def run_predict_pipeline(config: PredictionParameters):
    with open(config.logging_config, 'r') as fin:
        logging.config.dictConfig(yaml.safe_load(fin))
    logging.info(f'Predict pipeline started. Config :\n{{\n{OmegaConf.to_yaml(config)}}}')
    predict_pipeline(config)
    logging.info('Predict pipeline successfully finished')


if __name__ == '__main__':
    run_predict_pipeline()
