from typing import List
import sys
import numpy as np

sys.path.append(".")

from ml_example.features.feature_extraction import FeatureTransformer, IdentityTransformer
from ml_example.data.read_data import download_data
from ml_example.data.prepare_data import preprocess_data
from ml_example.params.data_split_params import SplittingParams
from ml_example.params.feature_params import FeatureParams


def test_transform_features_pipeline(
        synthetic_dataset_path: str,
        split_size: float, random_state: int,
        target_col: str,
        categorical_features: List[str],
        numerical_features: List[str],
):
    categorical_transformation = "OneHotEncoder"
    numerical_transformation = "StandardScaler"
    df = download_data(synthetic_dataset_path)
    split_params = SplittingParams(split_size=split_size, random_state=random_state, shuffle=True)
    df_train, df_test = preprocess_data(split_params, df)
    feature_params = FeatureParams(categorical_features, categorical_transformation,
                                   numerical_features, numerical_transformation,
                                   target_col)
    transformer = FeatureTransformer(feature_params)
    x_train, _ = transformer.fit_transform(df_train)
    x_test, _ = transformer.transform(df_test)
    assert np.allclose(x_train[:, :len(numerical_features)].mean(axis=0), 0, atol=1e-5), (
        "Numerical features should be Standartize")
    assert np.allclose(x_train[:, :len(numerical_features)].std(axis=0), 1, atol=1e-5), (
        "Numerical features should be Standartize")
