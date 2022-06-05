from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    categorical_transformation: str
    numerical_features: List[str]
    numerical_transformation: str
    target_col: Optional[str]
