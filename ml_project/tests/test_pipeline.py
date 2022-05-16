import os
import sys

sys.path.append('.')
from train_and_predict_pipeline import parse_config, train_and_predict_pipeline


def test_run_train_pipeline(
        tmp_path, config_path,
):
    parsed_config = parse_config(config_path)
    assert parsed_config is not None, "Func doesn't return config"
    output_model_path = tmp_path.joinpath("model/model.pkl")
    train_and_predict_pipeline(config_path, "", save_model=False, save_results=False)
    assert not os.path.exists(output_model_path), "Model is saved but it shouldn't be."
