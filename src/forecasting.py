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
                        'ETNA', # Internal library errors
                        'EvalML', 'FEDOT', 'FLAML', 'Ludwig', 'PyCaret']

    @staticmethod
    def run_forecasting_libraries(forecaster_names, datasets_directory, time_limit=3600, results_dir='results'):
        """Intended entrypoint to run forecasting libraries on the davailable datasets

        :param forecaster_names: List of forecasting library names
        :param datasets_directory: Path to forecasting datasets directory (str)
        :param time_limit: Time limit in seconds (int)
        """

        Forecasting._validate_inputs(forecaster_names, datasets_directory, time_limit)

        csv_files = Utils.get_csv_datasets(datasets_directory)
        # for i in range(len(csv_files)): print(i, csv_files[i])
        csv_files = [ csv_files[0] ] # TODO: For development only. To be removed
        metadata = pd.read_csv(os.path.join(datasets_directory, '0_metadata.csv'))

        for csv_file in csv_files:
            # Read dataset
            dataset_path = os.path.join(datasets_directory, csv_file)
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
            for forecaster_name in forecaster_names:
                dataset_name = csv_file.split('.')[0]
                results_subdir = os.path.join(results_dir, dataset_name)

                # Initialize forecaster and estimate a time/iterations limit
                forecaster = Forecasting._init_forecaster(forecaster_name)
                limit = forecaster.estimate_initial_limit(time_limit)

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
