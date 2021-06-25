from dataclasses import dataclass, field
from heart.parameters.dataset_params import *


@dataclass()
class TrainingParameters:
    module: str = field(default='sklearn.ensemble')
    method: str = field(default='RandomForestClassifier')
    model_parameters: Optional[dict] = None
    random_state: int = field(default=42)


@dataclass()
class TrainingPipelineParameters:
    logging_config: str
    input_data: str
    output_model: str
    metrics_test: str
    metrics_valid: Optional[str]
    dataset_splitting: TrainTestSplit
    dataset_features: FeatureTypes
    training: TrainingParameters
    transformer: str
