import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from ml_example.params.feature_params import FeatureParams

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer for empty transformation"""

    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        """Dummy fit

        :param input_array: dummy array
        :param y: dummy target
        :return: self
        """
        return self

    def transform(self, input_array: pd.DataFrame,
                  y=None) -> pd.DataFrame:
        """Dummy transformation

        :param input_array: dummy array
        :param y: dummy target
        :return: pseudo-transformed array
        """
        return input_array * 1


class FeatureTransformer:
    """Feature transformer class"""

    def __init__(self, params: FeatureParams):
        self.target_col = params.target_col
        self.transformer = None
        self.is_fitted = False
        self.numerical_features = params.numerical_features
        if params.numerical_transformation == "StandardScaler":
            self.numeric_transformation = StandardScaler
        elif not params.numerical_transformation:
            self.numeric_transformation = IdentityTransformer
        else:
            raise NotImplementedError
        self.categorical_features = params.categorical_features
        if params.categorical_transformation == "OneHotEncoder":
            self.categorical_transformation = OneHotEncoder
        elif not params.categorical_transformation:
            self.categorical_transformation = IdentityTransformer
        else:
            raise NotImplementedError

    def fit(self, data: pd.DataFrame):
        """fitting transformer with given params

        :param data: data to fit transformer
        :return: None
        """
        target = data[self.target_col]
        train_data = data.drop(self.target_col, axis=1)

        self.transformer = self._build_transformer()
        self.transformer.fit(train_data, target)
        self.is_fitted = True

    def _build_transformer(self) -> ColumnTransformer:
        """building transformer with current params

        :return: ColumnTransformer class
        """
        numeric_transformer = Pipeline(
            steps=[("numeric_transformation", self.numeric_transformation())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("categorical_transformation", self.categorical_transformation())]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        return transformer

    def transform(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms data with built transformer

        :param data: data to transform
        :return: transformed data and target
        """
        target = data[self.target_col]
        data2transform = data.drop(self.target_col, axis=1)
        if self.is_fitted:
            transformed_data = self.transformer.transform(data2transform)
        else:
            logging.error(" Cannot transform features "
                          "with unfitted transformer. "
                          "Untransformed data has been returned.")
            return data2transform, target
        return transformed_data, target

    def fit_transform(self, data) -> tuple[pd.DataFrame, pd.Series]:
        """fit-transform operation

        :param data: data to fit and transform
        :return: transformed data and target
        """
        self.fit(data)
        return self.transform(data)
