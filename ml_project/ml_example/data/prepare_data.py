"""Script for splitting data to train and test datasets"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_example.params.data_split_params import SplittingParams

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def preprocess_data(params: SplittingParams,
                    data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data with given params

    :param params: SplittingParams given with config file
    :param data: data to proceed
    :return: splitted data for training and testing
    """
    random_state = params.random_state
    split_size = params.split_size
    shuffle = params.shuffle
    train_data, test_data = train_test_split(data, random_state=random_state,
                                             test_size=split_size, shuffle=shuffle)
    logger.info(f"Train shape = {train_data.shape}, test shape = {test_data.shape}")
    return train_data, test_data
