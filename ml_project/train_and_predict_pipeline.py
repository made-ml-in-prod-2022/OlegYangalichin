"""Full CLI pipeline to train and predict with given configurations"""
import logging
import json
import click
from marshmallow_dataclass import class_schema
import yaml

from ml_example.data.read_data import download_data
from ml_example.data.prepare_data import preprocess_data
from ml_example.features.feature_extraction import FeatureTransformer
from ml_example.params.pipeline_config import PipelineParams
from ml_example.models.model import Model

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def parse_config(config_path: str) -> class_schema:
    """Parsing YAML format config to class schema

    :param config_path: path to YAML config
    :return: class schema
    """
    with open(config_path, "r") as config:
        scheme = class_schema(PipelineParams)()
        return scheme.load(yaml.safe_load(config))


def train_and_predict_pipeline(config_path: str, model_path: str,
                               save_model: bool, save_results: bool):
    """Pipeline for training and predicting with given configuration

    :param config_path: YAML config path
    :param model_path: path to load a model
    :param save_model: whether to save model or not
    :param save_results: whether to save metric report or not
    :return: None
    """
    logger.info("Parsing config file...")
    try:
        pipeline_params = parse_config(config_path)
    except UnicodeDecodeError:
        logger.info("Wrong config file has been provided. Please make sure it's YAML!")
        raise UnicodeDecodeError

    train_data, test_data = preprocess_data(pipeline_params.splitting_params,
                                            download_data(pipeline_params.input_data_path))

    logger.info("Transforming features...")
    feature_transformer = FeatureTransformer(pipeline_params.feature_params)
    x_train, y_train = feature_transformer.fit_transform(train_data)
    x_test, y_test = feature_transformer.transform(test_data)
    logger.info("Transformation completed.")

    if model_path:
        logger.info("Loading model...")
        try:
            model = Model.load_model(model_path)
            logger.info("Model has been successfully downloaded.")
        except:
            logger.info("Model hasn't been downloaded. "
                        "Please consider wrapping your model with models/Model class.")
            raise AttributeError
    else:
        model = Model(pipeline_params.train_params)
        logger.info("Training model...")
        model.fit(x_train, y_train)
        logger.info("Training's done.")

    if save_model:
        logger.info(f"Saving model to {pipeline_params.output_model_path}")
        model.save_model(pipeline_params.output_model_path)

    logger.info("Building predictions...")
    metric_report = model.count_metrics(y_test,  model.predict(x_test))
    logger.info(f"Test metrics: {metric_report}")

    if save_results:
        logger.info(f"Saving results to {pipeline_params.metric_report_path}")
        with open(pipeline_params.metric_report_path, "w") as metric_file:
            json.dump(metric_report, metric_file)


@click.group()
def cli():
    """Group for CLI commands

    :return: None
    """
    pass


@cli.command("run_pipeline")
@click.option("--config_path")
@click.option("--save_model", default=False, type=bool)
@click.option("--save_results", default=True, type=bool)
@click.option("--model_path", default='')
def run_pipeline(config_path: str, save_model: bool, save_results: bool, model_path: str):
    """ Run full customized pipeline

    :param config_path: YAML config path
    :param model_path: path to load a model
    :param save_model: whether to save model or not
    :param save_results: whether to save metric report or not
    :return: None
    """
    train_and_predict_pipeline(config_path, model_path, save_model, save_results)


@cli.command("train_and_save")
@click.option("--config_path")
def train_and_save(config_path):
    """Just train and save model with test metrics

    :param config_path: YAML config path
    :return: None
    """
    train_and_predict_pipeline(config_path, '', True, False)


@cli.command("load_and_predict")
@click.option("--config_path")
@click.option("--model_path")
@click.option("--save_results", default=True, type=bool)
def load_and_predict(config_path, model_path, save_results):
    """Load model and predict with it

    :param config_path: YAML config path
    :param model_path: path to load a model
    :param save_results: whether to save metric report or not
    :return: None
    """
    train_and_predict_pipeline(config_path, model_path, False, save_results)


if __name__ == "__main__":
    cli()
