"""Script for data downloading"""
import os
import sys
import logging
import pandas as pd

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FMT)


def download_data(path: str) -> pd.DataFrame:
    """Downloding data from .csv file
    :param path: path to .csv dataset
    :return: downloaded data in pd.DataFrame format
    """
    try:
        logger.info("Downloading...")
        data = pd.read_csv(path, index_col=False)
    except FileNotFoundError:
        logger.error(f"File with path {path} doesn't exist."
                     f" Current path is {os.getcwd()}.")
    except:
        logger.error("There is something wrong "
                     "while reading .csv input file.")
        logger.error(sys.exc_info()[1])
    else:
        logger.info("Dataset download is completed.")
        return data
