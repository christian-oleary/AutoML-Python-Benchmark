"""
Code for initiating forecasting experiments
"""

import os

import pandas as pd

from src.util import Utils
from src.autogluon.models import AutoGluonForecaster


class Forecasting():
    """Functionality for applying forecasting libraries to existing datasets"""

    logger = Utils.logger

    forecaster_names = [ 'AutoGluon' ]

    @staticmethod
    def run_forecasting_libraries(forecaster_names, datasets_directory):
        """Run forecasting libraries on the davailable datasets

        :param forecaster_names: List of forecasting library names
        :param datasets_directory: Path to forecasting datasets directory (str)
        """

        csv_files = Utils.get_csv_datasets(datasets_directory)
        csv_files = [ csv_files[0] ]

        for csv_file in csv_files:
            dataset_path = os.path.join(datasets_directory, csv_file)
            Forecasting.logger.debug(f'Reading dataset {dataset_path}')
            df = pd.read_csv(dataset_path)

            for forecaster_name in forecaster_names:
                forecaster = Forecasting._init_forecaster(forecaster_name)
                Forecasting.logger.info(f'Applying {forecaster.name} to {dataset_path}')
                Forecasting.apply_forecaster(df, forecaster)


    @staticmethod
    def _init_forecaster(forecaster_name):
        """Create forecaster object from name (see Forecasting.get_forecaster_names())

        :param forecaster_name: Name of forecaster (str)
        :raises ValueError: Raised for unknown forecaster name
        :return: Forecaster object
        """
        if forecaster_name == AutoGluonForecaster.name:
            forecaster = AutoGluonForecaster()
        else:
            raise ValueError(f'Unknown forecaster {forecaster_name}. Options: {Forecasting.forecaster_names}')
        return forecaster


    @staticmethod
    def get_forecaster_names():
        """Return list of forecaster names

        :return: list of forecaster names (str)
        """
        return Forecasting.forecaster_names


    @staticmethod
    def apply_forecaster(df, forecaster):
        """Apply forecaster to a forecasting dataset

        :param df: DataFrame of time series data
        :param forecaster: Forecaster object
        """
