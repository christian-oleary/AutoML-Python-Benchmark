"""
Code for initiating forecasting experiments
"""

import os
from glob import glob
import shutil
import time

import numpy as np
import pandas as pd

from src.base import Forecaster
from src.errors import AutomlLibraryError, DatasetTooSmallError
from src.logs import logger
from src.util import Utils
from src.validation import Task


class Forecasting:
    """Functionality for applying forecasting libraries to existing datasets"""

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

    def run_forecasting_libraries(self, config):
        """Entrypoint to run forecasting libraries on available datasets

        :param argparse.Namespace config: arguments from command line
        """

        self.config = config
        self.data_dir = config.data_dir
        self.forecast_type = config.task

        csv_files = Utils.get_csv_datasets(self.data_dir)
        metadata = pd.read_csv(os.path.join(self.data_dir, '0_metadata.csv'))

        for csv_file in csv_files:
            self.dataset_name = csv_file.split('.')[0]

            # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021)
            # we do not consider the London smart meters, wind farms, solar power, and wind power datasets
            # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
            # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation
            filter_forecast_datasets = True  # TODO: make an env variable
            if filter_forecast_datasets and self.dataset_name in self.omitted_datasets:
                logger.debug(f'Skipping dataset {self.dataset_name}')
                continue

            # Read dataset
            self.dataset_path = os.path.join(self.data_dir, csv_file)
            logger.info(f'Reading dataset {self.dataset_path}')

            if self.forecast_type == Task.GLOBAL_FORECASTING.value:
                self.df = pd.read_csv(self.dataset_path, index_col=0)

            elif self.forecast_type == Task.UNIVARIATE_FORECASTING.value:
                if 'libra' in self.dataset_path:
                    self.df = pd.read_csv(self.dataset_path, header=None)
                else:
                    self.df = pd.read_csv(self.dataset_path)
                    self.df = self.df.set_index('applicable_date')
                self.df.columns = ['target']
            else:
                raise NotImplementedError()

            # I-SEM dataset
            if 'ISEM_prices' in self.dataset_name:
                self.train_df = self.df.loc[:'19/10/2020 23:00', :]  # 293 days or ~80% of 2020
                self.test_df = self.df.loc['20/10/2020 00:00':, :]  # 73 days or ~20% of 2020

            # Holdout for model testing (80% training, 20% testing).
            # This seems to be used by Godahewa et al. for global forecasting:
            # https://github.com/rakshitha123/TSForecasting/blob/master/experiments/rolling_origin.R#L10
            # Also used by Bauer 2021 for univariate forecasting
            else:
                self.train_df = self.df.head(int(len(self.df) * 0.8))
                self.test_df = self.df.tail(int(len(self.df) * 0.2))

            # Get dataset metadata
            if self.forecast_type == Task.GLOBAL_FORECASTING.value:
                raise NotImplementedError()
                self.data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]

                self.frequency = self.data['frequency'].iloc[0]
                self.horizon = self.data['horizon'].iloc[0]
                if pd.isna(self.horizon):
                    raise ValueError(f'Missing horizon in 0_metadata.csv for {csv_file}')
                self.horizon = int(self.horizon)

                # TODO: revise frequencies, determine and data formatting stage
                if pd.isna(self.frequency) and 'm3_other_dataset.csv' in csv_file:
                    self.frequency = 'yearly'
                self.actual = self.test_df.values

            elif self.forecast_type == Task.MULTIVARIATE_FORECASTING.value:
                raise NotImplementedError()

            elif self.forecast_type == Task.UNIVARIATE_FORECASTING.value:
                self.data = metadata[metadata['file'] == csv_file]
                self.frequency = int(self.data['frequency'].iloc[0])
                self.horizon = int(self.data['horizon'].iloc[0])
                self.actual = self.test_df.copy().values.flatten()
                self.y_train = self.train_df.copy().values.flatten()  # Required for MASE
                # Libra's custom rolling origin forecast:
                # kwargs = {
                #     'origin_index': int(self.data['origin_index'].iloc[0]),
                #     'step_size': int(self.data['step_size'].iloc[0])
                #     }
            else:
                raise ValueError(f'Unknown forecast_type: {self.forecast_type}')

            # Run each forecaster on the dataset
            for self.forecaster_name in config.libraries:
                # Initialize forecaster and estimate a time/iterations limit
                self.forecaster = self._init_forecaster(self.forecaster_name)

                for preset in self.forecaster.presets:
                    self.limit = self.forecaster.estimate_initial_limit(config.time_limit, preset)
                    self.evaluate_library_preset(preset, csv_file)

    def evaluate_library_preset(self, preset, csv_file):
        """Evaluate a specific library on a specific preset

        :param str preset: Library specific preset
        :param str csv_file: Name of dataset CSV file
        """
        if self.config.results_dir is not None:
            results_subdir = os.path.join(
                self.config.results_dir,
                f'{self.forecast_type}_forecasting',
                self.dataset_name,
                self.forecaster_name,
                f'preset-{preset}_proc-{self.config.nproc}_limit-{self.limit}',
            )
            # If results are invalid and need to be removed:
            # if 'forecaster_name' in results_subdir and os.path.exists(results_subdir):
            #     shutil.rmtree(results_subdir)
            #     return
        else:
            results_subdir = None

        max_trials = self.determine_num_trials(results_subdir)
        # Option to skip training if completed previously
        if max_trials == 0:
            logger.info(f'Results found for {results_subdir}. Skipping training')

            # Summarize experiment results
            if self.config.results_dir is not None:
                Utils.summarize_dataset_results(
                    os.path.join(
                        self.config.results_dir,
                        f'{self.forecast_type}_forecasting',
                        self.dataset_name,
                    ),
                    plots=False,
                )
            return

        # for iteration in range(min_trials):
        #     logger.info(f'Iteration {iteration+1} of {min_trials}')

        if max_trials > 0:
            # Run forecaster and record total runtime
            tmp_dir = self.delete_tmp_dirs()
            logger.info(
                f'Applying {self.forecaster_name} (preset: {preset}) to {self.dataset_path}'
            )
            start_time = time.perf_counter()

            try:
                predictions = self.forecaster.forecast(
                    self.train_df.copy(),
                    self.test_df.copy(),
                    self.forecast_type,
                    self.horizon,
                    self.limit,
                    self.frequency,
                    tmp_dir,
                    nproc=self.config.nproc,
                    preset=preset,
                )

                duration = round(time.perf_counter() - start_time, 2)
                logger.debug(
                    f'{self.forecaster_name} (preset: {preset}) took {duration} seconds for {csv_file}'
                )

                # Generate scores and plots
                if self.config.results_dir is not None:
                    self.evaluate_predictions(
                        self.actual,
                        predictions,
                        self.y_train,
                        results_subdir,
                        self.forecaster_name,
                        duration,
                    )

            except DatasetTooSmallError as e1:
                logger.error('Failed to fit. Dataset too small for library.')
                self.record_failure(results_subdir, e1)
            except AutomlLibraryError as e2:
                logger.error(f'{self.forecaster_name} (preset: {preset}) failed to fit.')
                self.record_failure(results_subdir, e2)
            except Exception as e3:
                logger.critical(f'{self.forecaster_name} (preset: {preset}) failed!')
                logger.critical(e3, exc_info=True)
                raise e3

            # Summarize experiment results
            if self.config.results_dir is not None:
                Utils.summarize_dataset_results(
                    os.path.join(
                        self.config.results_dir,
                        f'{self.forecast_type}_forecasting',
                        self.dataset_name,
                    ),
                    plots=True,
                )

    def determine_num_trials(self, results_subdir):
        """Determine how many experiments to run"""
        if self.config.repeat_results:
            num_iterations = 1

        elif results_subdir is not None:
            results_path = os.path.join(results_subdir, f'{self.forecaster_name}.csv')
            # How many results remaining
            if os.path.exists(results_path):
                results = pd.read_csv(results_path)
                num_iterations = max(self.config.max_results - len(results), 0)
            # Skip failing trials
            elif os.path.exists(os.path.join(results_subdir, 'failed.txt')):
                num_iterations = 0
            else:
                num_iterations = self.config.max_results

        return num_iterations

    def record_failure(self, results_subdir, error):
        """Record failed forecasting attempts"""
        os.makedirs(results_subdir, exist_ok=True)
        with open(os.path.join(results_subdir, 'failed.txt'), 'w', encoding='utf8') as fh:
            fh.write(str(error))

    def evaluate_predictions(
        self, actual, predictions, y_train, results_subdir, forecaster_name, duration
    ):
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
            raise ValueError(
                f'Not enough predictions {predictions.shape[0]} for test set {actual.shape[0]}'
            )

        # Truncate and flatten predictions if needed
        if actual.shape[0] < predictions.shape[0]:
            try:
                predictions = predictions.head(actual.shape[0])
            except AttributeError:
                predictions = predictions[: len(actual)]
        predictions = predictions.flatten()

        # Save regression scores and plots
        scores = Utils.regression_scores(
            actual, predictions, y_train, results_subdir, forecaster_name, duration=duration
        )

        preds_path = os.path.join(results_subdir, 'predictions.csv')
        try:
            predictions.to_csv(preds_path)
        except AttributeError:
            np.savetxt(preds_path, predictions, fmt='%s', delimiter=',')

        try:  # If pandas Series
            predictions = predictions.reset_index(drop=True)
        except AttributeError:
            pass

        if results_subdir is not None:
            Utils.plot_forecast(
                actual, predictions, results_subdir, f'{forecaster_name}_{round(scores["R2"], 2)}'
            )

    def analyse_results(self, config, plots=True):
        """Analyse the overall results of running AutoML libraries on datasets

        :param str data_dir: Path to datasets directory
        :param argparse.Namespace config: arguments from command line
        :param bool plots: Save plots as images, defaults to True
        """
        logger.info(f'Analyzing results ({config.task})')

        if config.results_dir is None:
            logger.warning('No results directory specified. Skipping')

        elif not os.path.exists(config.results_dir):
            logger.error(
                f'Results directory not found: {config.results_dir} ({type(config.results_dir)})'
            )

        else:
            Utils.summarize_overall_results(config.results_dir, config.task, plots=plots)

    def delete_tmp_dirs(self):
        """Delete old temporary files directory to ensure libraries start from scratch"""
        tmp_dir = os.path.join('tmp', self.dataset_name, self.forecaster_name)
        paths_to_delete = [
            tmp_dir,
            'checkpoints',
            'catboost_info',
            'time_series_forecaster',
            'etna-auto.db',
        ] + glob('.lr_find_*.ckpt')

        for folder in paths_to_delete:
            if os.path.exists(folder):
                if os.path.isfile(folder):
                    os.remove(folder)
                else:
                    shutil.rmtree(folder)
        os.makedirs(tmp_dir)
        return tmp_dir

    def _init_forecaster(self, forecaster_name):
        """Create forecaster object from name (see Forecasting.forecaster_names)

        :param forecaster_name: Name of forecaster (str)
        :raises ValueError: Raised for unknown forecaster name
        :return: Forecaster object
        """

        # Import statements included here to accommodate different/conflicting setups
        # pylint: disable=C0415:import-outside-toplevel
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
            raise ValueError(f'Unknown forecaster {forecaster_name}')
        return forecaster
