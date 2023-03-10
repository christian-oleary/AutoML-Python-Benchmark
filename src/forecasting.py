"""
Code for initiating forecasting experiments
"""

import os
import time

import pandas as pd

from src.util import Utils
from src.autogluon.models import AutoGluonForecaster


class Forecasting():
    """Functionality for applying forecasting libraries to existing datasets"""

    logger = Utils.logger

    forecaster_names = [ 'AutoGluon' ]

    @staticmethod
    def run_forecasting_libraries(forecaster_names, datasets_directory, time_limit=3600):
        """Intended entrypoint to run forecasting libraries on the davailable datasets

        :param forecaster_names: List of forecasting library names
        :param datasets_directory: Path to forecasting datasets directory (str)
        :param time_limit: Time limit in seconds (int)
        """

        Forecasting._validate_inputs(forecaster_names, datasets_directory, time_limit)

        csv_files = Utils.get_csv_datasets(datasets_directory)
        # csv_files = [ csv_files[0] ] # TODO: For development only. To be removed
        metadata = pd.read_csv(os.path.join(datasets_directory, '0_metadata.csv'))

        for csv_file in csv_files:
            dataset_path = os.path.join(datasets_directory, csv_file)
            Forecasting.logger.debug(f'Reading dataset {dataset_path}')
            df = pd.read_csv(dataset_path, index_col=0)

            # Holdout for model testing (80% training, 20% testing)
            train_df = df.head(int(len(df)* 0.8))
            test_df = df.tail(int(len(df)* 0.2))

            # Get dataset metadata
            data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]
            target_name = train_df.columns[0] # Target names not currently known
            frequency = data['frequency'].iloc[0]
            horizon = data['horizon'].iloc[0]
            if pd.isna(horizon):
                horizon = max([1, int(len(df) * .05)])

            # Run each forecaster on the dataset
            for forecaster_name in forecaster_names:
                # Initialize forecaster and estimate a time/iterations limit
                forecaster = Forecasting._init_forecaster(forecaster_name)
                limit = forecaster.estimate_initial_limit(time_limit)

                # Run forecaster and record total runtime
                Forecasting.logger.info(f'Applying {forecaster.name} to {dataset_path}')
                # simulation_valid = False
                # while not simulation_valid:
                start_time = time.perf_counter()
                forecaster.forecast(train_df, test_df, target_name, horizon, limit, frequency)
                duration = time.perf_counter() - start_time
                Utils.logger.debug(f'{forecaster.name} took {duration} seconds {csv_file}')

                    # # Only valid if time limit not exceeded
                    # if duration <= 3600:
                    #     simulation_valid = True
                        # Re-estimate time/iterations limit based on previous duration
                    #     limit = forecaster.estimate_new_limit(time_limit, limit, duration)


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


    def _validate_inputs(forecaster_names, datasets_directory, time_limit):
        """Validation inputs for entrypoint run_forecasting_libraries()"""

        if not isinstance(forecaster_names, list):
            raise TypeError(f'forecaster_names must be a list. Received: {type(forecaster_names)}')

        for name in forecaster_names:
            if name not in Forecasting.forecaster_names:
                raise ValueError(f'Unknown forecaster. Options: {forecaster_names}')

        try:
            _ = os.listdir(datasets_directory)
        except NotADirectoryError as e:
            raise NotADirectoryError(f'Unknown directory for datasets_directory. Received: {datasets_directory}') from e

        if not isinstance(time_limit, int):
            raise TypeError(f'time_limit must be an int. Received: {time_limit}')

        if time_limit <= 0:
            raise ValueError(f'time_limit must be > 0. Received: {time_limit}')


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
