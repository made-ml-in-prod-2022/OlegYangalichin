from typing import List

import pytest

from fake_data_generator import generate_fake_data


@pytest.fixture()
def n_rows_in_synthetic_data() -> int:
    return 200


@pytest.fixture()
def split_size() -> float:
    return 0.2


@pytest.fixture()
def random_state() -> int:
    return 42

@pytest.fixture()
def shuffle() -> int:
    return True


@pytest.fixture()
def target_col() -> str:
    return "condition"

@pytest.fixture()
def config_path() -> str:
    return "tests/test_config.yaml"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]


@pytest.fixture()
def synthetic_dataset_path(tmpdir_factory, n_rows_in_synthetic_data):
    filename = str(tmpdir_factory.mktemp("data").join("syntetic_dataset.csv"))

    fake_dataset_df = generate_fake_data(n_rows_in_synthetic_data)
    fake_dataset_df.to_csv(filename, index=None)

    return filename