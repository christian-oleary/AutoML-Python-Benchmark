"""
Code for initiating forecasting experiments
"""

import os
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer

from src.base import Forecaster
from src.errors import AutomlLibraryError, DatasetTooSmallError
from src.logs import logger
from src.util import Utils


class Forecasting():
    """Functionality for applying forecasting libraries to existing datasets"""


    univariate_forecaster_names = [ 'autogluon', 'autokeras', 'autots', 'autopytorch',
                                    'evalml', 'fedot', 'flaml', 'ludwig', 'pycaret']

    global_forecaster_names = [ 'autogluon', 'autokeras', 'autots', 'autopytorch',
                                # 'etna', # Internal library errors
                                'evalml', 'fedot', 'flaml', 'ludwig', 'pycaret']

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
        'solar_10_minutes_dataset',
        'solar_4_seconds_dataset',
        'web_traffic_extended_dataset_with_missing_values',
        'web_traffic_extended_dataset_without_missing_values',
        'wind_farms_minutely_dataset_without_missing_values',
        'wind_farms_minutely_dataset_with_missing_values',
        'wind_4_seconds_dataset',
        ]


    def run_forecasting_libraries(self, data_dir, config, forecast_type):
        """Entrypoint to run forecasting libraries on available datasets

        :param str data_dir: Path to datasets directory
        :param argparse.Namespace config: arguments from command line
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        """

        self._validate_inputs(config, forecast_type)

        csv_files = Utils.get_csv_datasets(data_dir)
        metadata = pd.read_csv(os.path.join(data_dir, '0_metadata.csv'))

        for csv_file in csv_files:
            dataset_name = csv_file.split('.')[0]

            # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021)
            # we do not consider the London smart meters, wind farms, solar power, and wind power datasets
            # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
            # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation
            filter_forecast_datasets = True # TODO: make an env variable
            if filter_forecast_datasets and dataset_name in self.omitted_datasets:
                logger.debug(f'Skipping dataset {dataset_name}')
                continue

            # Read dataset
            dataset_path = os.path.join(data_dir, csv_file)
            logger.info(f'Reading dataset {dataset_path}')
            if forecast_type == 'global':
                df = pd.read_csv(dataset_path, index_col=0)
            elif forecast_type == 'univariate':
                if 'libra' in dataset_path:
                    df = pd.read_csv(dataset_path, header=None)
                else:
                    df = pd.read_csv(dataset_path)
                    df = df.set_index('applicable_date')
                df.columns = [ 'target' ]
            else:
                raise NotImplementedError()

            # Holdout for model testing (80% training, 20% testing).
            # This seems to be used by Godahewa et al. for global forecasting:
            # https://github.com/rakshitha123/TSForecasting/blob/master/experiments/rolling_origin.R#L10
            # Also used by Bauer 2021 for univariate forecasting
            if 'ISEM_prices' in dataset_name:
                train_df = df.loc[:'19/10/2020 23:00',:] # 293 days or ~80% of 2020
                test_df = df.loc['20/10/2020 00:00':,:] # 73 days or ~20% of 2020
            else:
                train_df = df.head(int(len(df)* 0.8))
                test_df = df.tail(int(len(df)* 0.2))

            # Get dataset metadata
            if forecast_type == 'global':
                raise NotImplementedError()
                data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]

                frequency = data['frequency'].iloc[0]
                horizon = data['horizon'].iloc[0]
                if pd.isna(horizon):
                    raise ValueError(f'Missing horizon in 0_metadata.csv for {csv_file}')
                horizon = int(horizon)

                # TODO: revise frequencies, determine and data formatting stage
                if pd.isna(frequency) and 'm3_other_dataset.csv' in csv_file:
                    frequency = 'yearly'
                actual = test_df.values

            else:
                data = metadata[metadata['file'] == csv_file]
                frequency = int(data['frequency'].iloc[0])
                horizon = int(data['horizon'].iloc[0])
                actual = test_df.copy().values.flatten()
                y_train = train_df.copy().values.flatten() # Required for MASE
                # Libra's custom rolling origin forecast:
                # kwargs = {
                #     'origin_index': int(data['origin_index'].iloc[0]),
                #     'step_size': int(data['step_size'].iloc[0])
                #     }

            # Run each forecaster on the dataset
            for forecaster_name in config.libraries:
                # Initialize forecaster and estimate a time/iterations limit
                forecaster = self._init_forecaster(forecaster_name)

                for preset in forecaster.presets:
                    limit = forecaster.estimate_initial_limit(config.time_limit, preset)

                    if config.results_dir != None:
                        results_subdir = os.path.join(config.results_dir, f'{forecast_type}_forecasting', dataset_name,
                                                    forecaster_name, f'preset-{preset}_proc-{config.nproc}_limit-{limit}')
                        # If results are invalid and need to be removed:
                        # if 'forecaster_name' in results_subdir and os.path.exists(results_subdir):
                        #     shutil.rmtree(results_subdir)
                    else:
                        results_subdir = None

                    # Option to skip training if completed previously
                    if not config.repeat_results and self.results_exist(results_subdir, forecaster_name):
                        logger.info(f'Results found for {results_subdir}. Skipping training')

                        # Summarize experiment results
                        if config.results_dir != None:
                            Utils.summarize_dataset_results(
                                os.path.join(config.results_dir, f'{forecast_type}_forecasting', dataset_name), plots=False)

                        continue

                    # Run forecaster and record total runtime
                    logger.info(f'Applying {forecaster_name} (preset: {preset}) to {dataset_path}')
                    start_time = time.perf_counter()
                    # Recreate temporary files directory to ensure libraries start from scratch
                    tmp_dir = os.path.join('tmp', dataset_name, forecaster_name)
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    os.makedirs(tmp_dir)
                    try:
                        predictions = forecaster.forecast(train_df.copy(), test_df.copy(), forecast_type, horizon,
                                                          limit, frequency, tmp_dir, nproc=config.nproc, preset=preset)

                        duration = round(time.perf_counter() - start_time, 2)
                        logger.debug(f'{forecaster_name} (preset: {preset}) took {duration} seconds for {csv_file}')

                        # Generate scores and plots
                        if config.results_dir != None:
                            self.evaluate_predictions(actual, predictions, y_train, results_subdir, forecaster_name, duration)

                    except DatasetTooSmallError as e1:
                        logger.error('Failed to fit. Dataset too small for library.')
                        self.record_failure(results_subdir, e1)
                    except AutomlLibraryError as e2:
                        logger.error(f'{forecaster_name} (preset: {preset}) failed to fit.')
                        self.record_failure(results_subdir, e2)
                    except Exception as e3:
                        logger.critical(f'{forecaster_name} (preset: {preset}) failed!')
                        logger.critical(e3, exc_info=True)
                        raise e3

                    # Summarize experiment results
                    if config.results_dir != None:
                        Utils.summarize_dataset_results(
                            os.path.join(config.results_dir, f'{forecast_type}_forecasting', dataset_name), plots=True)


    def results_exist(self, results_subdir, forecaster_name):
        """Check if results have been generated for a given experiment

        :param str results_subdir: Path to results directory
        :param str forecaster_name: Name of forecasting model
        :return bool: True if results exist, False otherwise
        """
        if results_subdir is not None:
            results_csv_exists = os.path.exists(os.path.join(results_subdir, f'{forecaster_name}.csv'))
            plots_exist = os.path.exists(os.path.join(results_subdir, 'plots'))

            if results_csv_exists and plots_exist:
                results_exist = True
            elif os.path.exists(os.path.join(results_subdir, 'failed.txt')):
                results_exist = True
            else:
                results_exist = False
        else:
            results_exist = False
        return results_exist


    def record_failure(self, results_subdir, error):
        os.makedirs(results_subdir, exist_ok=True)
        with open(os.path.join(results_subdir, 'failed.txt'), 'w') as fh:
            fh.write(str(error))


    def evaluate_predictions(self, actual, predictions, y_train, results_subdir, forecaster_name, duration):
        """Generate model scores and plots from predictions

        :param np.array actual: Original data
        :param np.array predictions: Predicted data
        :param np.array y_train: Training values (required for MASE)
        :param str results_subdir: Path to results subdirectory
        :param str forecaster_name: Name of library
        :param float duration: Duration of fitting/inference times
        :raises ValueError: If predictions < actual
        """
        # Check that model outputted enough predictions
        if actual.shape[0] > predictions.shape[0]:
            logger.error(f'Predictions: {predictions}')
            logger.error(f'Actual: {actual}')
            raise ValueError(f'Not enough predictions {predictions.shape[0]} for test set {actual.shape[0]}')

        # Truncate and flatten predictions if needed
        if actual.shape[0] < predictions.shape[0]:
            try:
                predictions = predictions.head(actual.shape[0])
            except:
                predictions = predictions[:len(actual)]
        predictions = predictions.flatten()

        # Save regression scores and plots
        scores = Utils.regression_scores(actual, predictions, y_train, results_subdir, forecaster_name,
                                        duration=duration)

        preds_path = os.path.join(results_subdir, 'predictions.csv')
        try:
            predictions.to_csv(preds_path)
        except:
            np.savetxt(preds_path, predictions, fmt='%s', delimiter=',')

        try: # If pandas Series
            predictions = predictions.reset_index(drop=True)
        except: pass

        if results_subdir != None:
            Utils.plot_forecast(actual, predictions, results_subdir, f'{forecaster_name}_{round(scores["R2"], 2)}')


    def analyse_results(self, config, forecast_type, plots=True):
        """Analyse the overall results of running AutoML libraries on datasets

        :param str data_dir: Path to datasets directory
        :param argparse.Namespace config: arguments from command line
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param bool plots: Save plots as images, defaults to True
        """
        logger.info(f'Analysing results ({forecast_type})')
        self._validate_inputs(config, forecast_type)

        if config.results_dir == None:
            logger.warning('No results directory specified. Skipping')

        elif not os.path.exists(config.results_dir):
            logger.error(f'Results directory not found: {config.results_dir} ({type(config.results_dir)})')

        else:
            Utils.summarize_overall_results(config.results_dir, forecast_type, plots=plots)


    def _init_forecaster(self, forecaster_name):
        """Create forecaster object from name (see Forecasting.get_forecaster_names())

        :param forecaster_name: Name of forecaster (str)
        :raises ValueError: Raised for unknown forecaster name
        :return: Forecaster object
        """

        # Import statements included here to accomodate different/conflicting setups
        if forecaster_name == 'test':
            forecaster = Forecaster()
        elif forecaster_name == 'autogluon':
            from src.autogluon.models import AutoGluonForecaster
            forecaster = AutoGluonForecaster()
        elif forecaster_name == 'autokeras':
            from src.autokeras.models import AutoKerasForecaster
            forecaster = AutoKerasForecaster()
        elif forecaster_name == 'autots':
            from src.autots.models import AutoTSForecaster
            forecaster = AutoTSForecaster()
        elif forecaster_name == 'autopytorch':
            from src.autopytorch.models import AutoPyTorchForecaster
            forecaster = AutoPyTorchForecaster()
        elif forecaster_name == 'etna':
            from src.etna.models import ETNAForecaster
            forecaster = ETNAForecaster()
        elif forecaster_name == 'evalml':
            from src.evalml.models import EvalMLForecaster
            forecaster = EvalMLForecaster()
        elif forecaster_name == 'fedot':
            from src.fedot.models import FEDOTForecaster
            forecaster = FEDOTForecaster()
        elif forecaster_name == 'flaml':
            from src.flaml.models import FLAMLForecaster
            forecaster = FLAMLForecaster()
        elif forecaster_name == 'ludwig':
            from src.ludwig.models import LudwigForecaster
            forecaster = LudwigForecaster()
        elif forecaster_name == 'pycaret':
            from src.pycaret.models import PyCaretForecaster
            forecaster = PyCaretForecaster()
        else:
            raise ValueError(f'Unknown forecaster {forecaster_name}. Options: {self.global_forecaster_names}')
        return forecaster


    def _validate_inputs(self, config, forecast_type):
        """Validation inputs for entrypoint run_forecasting_libraries()"""

        options = ['global', 'multivariate', 'univariate']
        if forecast_type not in options:
            raise ValueError(f'Unrecognised forecast type: {forecast_type}. Options: {options}')

        if forecast_type == 'multivariate':
            raise NotImplementedError('multivariate forecasting not implemented')

        if config.libraries == 'all':
            config.libraries = self.global_forecaster_names

        elif config.libraries == 'installed':
            config = self._check_installed(config)
        else:
            if not isinstance(config.libraries, list):
                raise TypeError(f'forecaster_names must be a list or "all". Received: {type(config.libraries)}')

            for name in config.libraries:
                if name not in [ 'None', 'test' ] and name not in self.global_forecaster_names:
                    raise ValueError(f'Unknown forecaster. Options: {self.global_forecaster_names}')

        if not isinstance(config.nproc, int):
            raise TypeError(f'nproc must be an int. Received {type(config.nproc)}')

        if config.nproc == 0 or config.nproc < -1:
            raise ValueError(f'nproc must be -1 or a positive int. Received {config.nproc}')

        is_none = config.results_dir in ['None', 'none', None]
        if not isinstance(config.results_dir, str) and not is_none:
            raise TypeError(f'results_dir must be str or None. Received: {config.results_dir} ({type(config.results_dir)})')
        elif is_none:
            config.results_dir = None

        try:
            _ = os.listdir(config.univariate_forecasting_data_dir)
        except NotADirectoryError as e:
            raise NotADirectoryError(f'Unknown directory for datasets_directory. Received: {config.forecasting_data_dir}') from e

        try:
            _ = os.listdir(config.global_forecasting_data_dir)
        except NotADirectoryError as e:
            raise NotADirectoryError(f'Unknown directory for datasets_directory. Received: {config.forecasting_data_dir}') from e

        if not isinstance(config.time_limit, int):
            raise TypeError(f'time_limit must be an int. Received: {config.time_limit}')

        if config.time_limit <= 0:
            raise ValueError(f'time_limit must be > 0. Received: {config.time_limit}')


    def _check_installed(self, config):
        """Determine which libararies are installed.

        :param config: Argparser configuration
        :raises ValueError: If no AutoML library is installed
        :return: Updated argparser config object
        """
        # Try to import libraries to determine which are installed
        config.libraries = []
        try:
            from src.autogluon.models import AutoGluonForecaster
            config.libraries.append('autogluon')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoGluon')

        try:
            from src.autokeras.models import AutoKerasForecaster
            config.libraries.append('autokeras')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoKeras')

        try:
            from src.autots.models import AutoTSForecaster
            config.libraries.append('autots')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoTS')

        try:
            from src.autopytorch.models import AutoPyTorchForecaster
            config.libraries.append('autopytorch')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoPyTorch')

        try:
            from src.evalml.models import EvalMLForecaster
            config.libraries.append('evalml')
        except ModuleNotFoundError as e:
            logger.debug('Not using EvalML')

        try:
            from src.etna.models import ETNAForecaster
            config.libraries.append('etna')
        except ModuleNotFoundError as e:
            logger.debug('Not using ETNA')

        try:
            from src.fedot.models import FEDOTForecaster
            config.libraries.append('fedot')
        except ModuleNotFoundError as e:
            logger.debug('Not using FEDOT')

        try:
            from src.flaml.models import FLAMLForecaster
            config.libraries.append('flaml')
        except:
            logger.debug('Not using FLAML')

        try:
            from src.ludwig.models import LudwigForecaster
            config.libraries.append('ludwig')
        except ModuleNotFoundError as e:
            logger.debug('Not using Ludwig')

        try:
            from src.pycaret.models import PyCaretForecaster
            config.libraries.append('pycaret')
        except ModuleNotFoundError as e:
            logger.debug('Not using PyCaret')

        if len(config.libraries) == 0:
            raise ModuleNotFoundError('No AutoML libraries can be imported. Are any installed?')

        logger.info(f'Using Libraries: {config.libraries}')
        return config


    def get_global_forecaster_names(self):
        """Return list of forecaster names

        :return: list of forecaster names (str)
        """
        return self.global_forecaster_names
