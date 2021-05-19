from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class TrainTestSplit:
    val_size: float = 0.1
    test_size: float = 0.1
    val_needed: bool = False
    random_state: int = 42

@dataclass()
class FeatureTypes:
    categorical: Optional[List[str]]
    numerical: Optional[List[str]]
    target: Optional[str]
    one_hot_encoding: Optional[List[str]]
