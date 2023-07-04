"""
Code for initiating forecasting experiments
"""

import os
import time

import pandas as pd
from sklearn.experimental import enable_iterative_imputer # import needed for IterativeImputer
from sklearn.impute import IterativeImputer

from src.util import Utils


class Forecasting():
    """Functionality for applying forecasting libraries to existing datasets"""

    logger = Utils.logger

    forecaster_names = [ 'AutoGluon', 'AutoKeras', 'AutoTS', 'AutoPyTorch',
                        # 'ETNA', # Internal library errors
                        'EvalML', 'FEDOT', 'FLAML', 'Ludwig', 'PyCaret']

    # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021):
    # "we do not consider the London smart meters, wind farms, solar power, and wind power datasets
    # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
    # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation"
    omitted_datasets = [
        'kaggle_web_traffic_dataset_with_missing_values',
        'kaggle_web_traffic_dataset_without_missing_values',
        'kaggle_web_traffic_weekly_dataset',
        'london_smart_meters_dataset_with_missing_values',
        'london_smart_meters_dataset_without_missing_values',
        'solar_weekly_dataset',
        'solar_10_minutes_dataset',
        'solar_4_seconds_dataset',
        'web_traffic_extended_dataset_with_missing_values',
        'web_traffic_extended_dataset_without_missing_values',
        'wind_farms_minutely_dataset_without_missing_values',
        'wind_farms_minutely_dataset_with_missing_values',
        'wind_4_seconds_dataset',
        ]


    @staticmethod
    def run_forecasting_libraries(config):
        """Intended entrypoint to run forecasting libraries on the davailable datasets

        :param config: Program configuration
        """

        Forecasting._validate_inputs(config)

        csv_files = Utils.get_csv_datasets(config.forecasting_data_dir)
        # for i in range(len(csv_files)): print(i, csv_files[i])
        # csv_files = [ csv_files[0] ] # TODO: For development only. To be removed
        metadata = pd.read_csv(os.path.join(config.forecasting_data_dir, '0_metadata.csv'))

        for csv_file in csv_files:
            dataset_name = csv_file.split('.')[0]

            # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021)
            # we do not consider the London smart meters, wind farms, solar power, and wind power datasets
            # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
            # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation
            filter_forecast_datasets = True # TODO: make an env variable
            if filter_forecast_datasets and dataset_name in Forecasting.omitted_datasets:
                Forecasting.logger.debug(f'Skipping dataset {dataset_name}')

            # Read dataset
            dataset_path = os.path.join(config.forecasting_data_dir, csv_file)
            Forecasting.logger.debug(f'Reading dataset {dataset_path}')
            df = pd.read_csv(dataset_path, index_col=0)

            # Holdout for model testing (80% training, 20% testing). This seems to be used by Godahewa et al.:
            # https://github.com/rakshitha123/TSForecasting/blob/master/experiments/rolling_origin.R#L10
            train_df = df.head(int(len(df)* 0.8))
            test_df = df.tail(int(len(df)* 0.2))

            # Get dataset metadata
            data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]
            target_name = None
            # As target names not currently known:
            for col in test_df.columns:
                percentage_nan = test_df[col].isnull().sum() * 100 / len(test_df)
                print(col, percentage_nan, percentage_nan < 0.5)
                if percentage_nan < 0.5: # i.e. at least 50% values present
                    target_name = col
                    break

            if target_name is None:
                raise ValueError(f'Failed to find suitable forecasting target in {csv_file}')

            frequency = data['frequency'].iloc[0]
            horizon = data['horizon'].iloc[0]
            if pd.isna(horizon):
                raise ValueError(f'Missing horizon in 0_metadata.csv for {csv_file}')
            horizon = int(horizon)

            # TODO: revise frequencies, determine and data formatting stage
            if pd.isna(frequency) and 'm3_other_dataset.csv' in csv_file:
                frequency = 'yearly'

            # Interpolate any missing values in the test data
            if test_df[target_name].isnull().values.any():
                imputer = IterativeImputer(max_iter=10, random_state=0)
                # imputer = SimpleImputer()
                test_df[target_name] = imputer.fit_transform(test_df[[target_name]]).ravel()

            # Run each forecaster on the dataset
            for forecaster_name in config.libraries:
                results_subdir = os.path.join(config.results_dir, dataset_name)

                # Initialize forecaster and estimate a time/iterations limit
                forecaster = Forecasting._init_forecaster(forecaster_name)
                limit = forecaster.estimate_initial_limit(config.time_limit)

                # Run forecaster and record total runtime
                Forecasting.logger.info(f'Applying {forecaster.name} to {dataset_path}')
                # simulation_valid = False
                # attempts = 0
                # while not simulation_valid and attempts < 10:
                    # attempts += 1

                start_time = time.perf_counter()
                tmp_dir = os.path.join('tmp', dataset_name, forecaster.name)
                os.makedirs(tmp_dir, exist_ok=True)
                predictions = forecaster.forecast(train_df, test_df, target_name, horizon, limit, frequency, tmp_dir)
                duration = time.perf_counter() - start_time
                Utils.logger.debug(f'{forecaster.name} took {duration} seconds {csv_file}')

                # Check that model outputted enough predictions
                actual = test_df[target_name]
                if actual.shape[0] > predictions.shape[0]:
                    raise ValueError(f'Not enough predictions {predictions.shape[0]} for test set {actual.shape[0]}')

                # Truncate and flatten predictions if needed
                if actual.shape[0] < predictions.shape[0]:
                    predictions = predictions.head(actual.shape[0])
                predictions = predictions.flatten()

                # Save regression scores and plots
                scores = Utils.regression_scores(actual, predictions, results_subdir, forecaster_name,
                                                duration=duration)

                try: # If pandas Series
                    predictions = predictions.reset_index(drop=True)
                except: pass

                Utils.plot_forecast(actual.reset_index(drop=True).values, predictions, results_subdir,
                                    f'{forecaster_name}_{round(scores["R2"], 2)}')
                    # # Only valid if time limit not exceeded
                    # if duration <= limit:
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

        # Import statements included here to accomodate different/conflicting setups
        if forecaster_name == 'AutoGluon':
            from src.autogluon.models import AutoGluonForecaster
            forecaster = AutoGluonForecaster()
        elif forecaster_name == 'AutoKeras':
            from src.autokeras.models import AutoKerasForecaster
            forecaster = AutoKerasForecaster()
        elif forecaster_name == 'AutoTS':
            from src.autots.models import AutoTSForecaster
            forecaster = AutoTSForecaster()
        elif forecaster_name == 'AutoPyTorch':
            from src.autopytorch.models import AutoPyTorchForecaster
            forecaster = AutoPyTorchForecaster()
        elif forecaster_name == 'EvalML':
            from src.evalml.models import EvalMLForecaster
            forecaster = EvalMLForecaster()
        elif forecaster_name == 'ETNA':
            from src.etna.models import ETNAForecaster
            forecaster = ETNAForecaster()
        elif forecaster_name == 'FEDOT':
            from src.fedot.models import FEDOTForecaster
            forecaster = FEDOTForecaster()
        elif forecaster_name == 'FLAML':
            from src.flaml.models import FLAMLForecaster
            forecaster = FLAMLForecaster()
        elif forecaster_name == 'Ludwig':
            from src.ludwig.models import LudwigForecaster
            forecaster = LudwigForecaster()
        elif forecaster_name == 'PyCaret':
            from src.pycaret.models import PyCaretForecaster
            forecaster = PyCaretForecaster()
        else:
            raise ValueError(f'Unknown forecaster {forecaster_name}. Options: {Forecasting.forecaster_names}')
        return forecaster


    def _validate_inputs(config):
        """Validation inputs for entrypoint run_forecasting_libraries()"""

        if config.libraries == 'all':
            config.libraries = Forecasting.forecaster_names

        elif config.libraries == 'installed':
            config = Forecasting._check_installed(config)
        else:
            print('config.libraries', config.libraries)
            if not isinstance(config.libraries, list):
                raise TypeError(f'forecaster_names must be a list or "all". Received: {type(config.libraries)}')

            for name in config.libraries:
                if name not in Forecasting.forecaster_names:
                    raise ValueError(f'Unknown forecaster. Options: {Forecasting.forecaster_names}')

        try:
            _ = os.listdir(config.forecasting_data_dir)
        except NotADirectoryError as e:
            raise NotADirectoryError(f'Unknown directory for datasets_directory. Received: {config.forecasting_data_dir}') from e

        if not isinstance(config.time_limit, int):
            raise TypeError(f'time_limit must be an int. Received: {config.time_limit}')

        if config.time_limit <= 0:
            raise ValueError(f'time_limit must be > 0. Received: {config.time_limit}')


    @staticmethod
    def _check_installed(config):
        """Determine which libararies are installed.

        :param config: Argparser configuration
        :raises ValueError: If no AutoML library is installed
        :return: Updated argparser config object
        """
        # Try to import libraries to determine which are installed
        config.libraries = []
        try:
            from src.autogluon.models import AutoGluonForecaster
            config.libraries.append('AutoGluon')
        except:
            Forecasting.logger.debug('Not using AutoGluon')

        try:
            from src.autokeras.models import AutoKerasForecaster
            config.libraries.append('AutoKeras')
        except:
            Forecasting.logger.debug('Not using AutoKeras')

        try:
            from src.autots.models import AutoTSForecaster
            config.libraries.append('AutoTS')
        except:
            Forecasting.logger.debug('Not using AutoTS')

        try:
            from src.autopytorch.models import AutoPyTorchForecaster
            config.libraries.append('AutoPyTorch')
        except:
            Forecasting.logger.debug('Not using AutoPyTorch')

        try:
            from src.evalml.models import EvalMLForecaster
            config.libraries.append('EvalML')
        except:
            Forecasting.logger.debug('Not using EvalML')

        try:
            from src.etna.models import ETNAForecaster
            config.libraries.append('ETNA')
        except:
            Forecasting.logger.debug('Not using ETNA')

        try:
            from src.fedot.models import FEDOTForecaster
            config.libraries.append('FEDOT')
        except:
            Forecasting.logger.debug('Not using FEDOT')

        try:
            from src.flaml.models import FLAMLForecaster
            config.libraries.append('FLAML')
        except:
            Forecasting.logger.debug('Not using FLAML')

        try:
            from src.ludwig.models import LudwigForecaster
            config.libraries.append('Ludwig')
        except:
            Forecasting.logger.debug('Not using Ludwig')

        try:
            from src.pycaret.models import PyCaretForecaster
            config.libraries.append('PyCaret')
        except:
            Forecasting.logger.debug('Not using PyCaret')

        if len(config.libraries) == 0:
            raise ValueError('No AutoML libraries are available. Check installation!')

        Forecasting.logger.info(f'Using Libraries: {config.libraries}')
        return config


    @staticmethod
    def get_forecaster_names():
        """Return list of forecaster names

        :return: list of forecaster names (str)
        """
        return Forecasting.forecaster_names
