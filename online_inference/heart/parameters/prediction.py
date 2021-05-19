from dataclasses import dataclass
from heart.parameters.dataset_params import *


@dataclass()
class PredictionParameters:
    logging_config: str
    data: str
    model: str
    output: str
    dataset_features: FeatureTypes
    transformer: str
