from typing import List
import sys
sys.path.append(".")
from ml_example.data.read_data import download_data


def test_load_dataset(
        synthetic_dataset_path: str,
        n_rows_in_synthetic_data: int,
        target_col: str,
        numerical_features: List[str],
        categorical_features: List[str],
):
    data = download_data(synthetic_dataset_path)

    assert data.shape[0] == n_rows_in_synthetic_data, "No valid number of rows in .csv"
    assert len(data.columns) == len(numerical_features) + len(categorical_features) + 1, \
        "Number of features should be equal number numerical + number categorial + target"
    assert target_col in data.columns, "Target is absent in dataframe"
