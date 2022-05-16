from typing import List
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.append(".")

from ml_example.features.feature_extraction import FeatureTransformer, IdentityTransformer
from ml_example.data.read_data import download_data
from ml_example.data.prepare_data import preprocess_data
from ml_example.params.data_split_params import SplittingParams
from ml_example.params.feature_params import FeatureParams
from ml_example.models.model import Model
from ml_example.params.training_params import TrainingParams


def test_train_model(
        synthetic_dataset_path: str, split_size: float, random_state: int,
        target_col: str, shuffle: bool,
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
    x_train, y_train = transformer.fit_transform(df_train)
    x_test, y_test = transformer.transform(df_test)

    train_params = TrainingParams(model="LogisticRegression", model_params={"C": 1.0, "fit_intercept": True,
                                                                                 "random_state": 42,
                                                                                 "solver": "liblinear"})
    model = Model(train_params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)
    assert isinstance(y_pred, pd.Series)
    assert isinstance(y_pred, pd.Series)
    assert y_pred_proba.shape == y_pred.shape == y_test.shape
    assert isinstance(model.model, LogisticRegression)
