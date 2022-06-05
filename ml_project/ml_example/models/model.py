"""Script for Model class"""
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from ml_example.params.training_params import TrainingParams


class Model:
    """ML model wrapper"""
    def __init__(self, params: TrainingParams):
        self.model_type = params.model
        self.model_params = params.model_params
        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression(**self.model_params)
        elif self.model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == "LGBMClassifier":
            self.model = LGBMClassifier(**self.model_params)
        else:
            raise NotImplementedError()

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fits model with given data

        :param features: data except target
        :param target: target
        :return: None
        """
        self.model.fit(features, target)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict target with fitted model

        :param data: data to predict
        :return: prediction
        """
        y_pred = pd.Series(self.model.predict(data))
        return y_pred

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Predict probabilities target with fitted model

        :param data: data to predict
        :return: prediction
        """
        y_pred = pd.Series(self.model.predict_proba(data)[:, 1])
        return y_pred

    def save_model(self, path: str):
        """Saves Model entity to given path

        :param path: path to save Model
        :return: None
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(path: str):
        """Loading Model entity from given path

        :param path: path to load model
        :return: Model
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def count_metrics(target: pd.Series, prediction: pd.Series):
        """Counting metrics for fitted model

        :param target: true values
        :param prediction: model prediction
        :return: dict with metrics
        """
        roc_auc = roc_auc_score(target, prediction)
        precision = precision_score(target, prediction)
        recall = recall_score(target, prediction)
        accuracy = accuracy_score(target, prediction)
        f1 = f1_score(target, prediction)

        return {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
