from dataclasses import dataclass

from ml_example.params.data_split_params import SplittingParams
from ml_example.params.training_params import TrainingParams
from ml_example.params.feature_params import FeatureParams


@dataclass()
class PipelineParams:
    input_data_path: str
    feature_params: FeatureParams
    output_model_path: str
    metric_report_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams