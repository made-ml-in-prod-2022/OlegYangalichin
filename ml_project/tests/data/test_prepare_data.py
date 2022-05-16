import sys

sys.path.append(".")
import numpy as np
from ml_example.data.read_data import download_data
from ml_example.data.prepare_data import preprocess_data
from ml_example.params.data_split_params import SplittingParams


def test_split_train_test_data(synthetic_dataset_path: str,
                               split_size: float, random_state: int, shuffle: bool):
    df = download_data(synthetic_dataset_path)
    split_params = SplittingParams(split_size=split_size, random_state=random_state, shuffle=shuffle)
    df_train, df_test = preprocess_data(split_params, df)

    assert np.isclose(df_test.shape[0] / df.shape[0], split_size, atol=0.01), "Not valid data splitting"
