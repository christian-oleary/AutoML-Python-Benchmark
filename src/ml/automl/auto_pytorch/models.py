"""AutoPyTorch models."""

from __future__ import annotations
import copy
from pathlib import Path

import pandas as pd

from ml.base import Forecaster
from ml.TSForecasting.data_loader import FREQUENCY_MAP

try:
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask  # type: ignore
except ModuleNotFoundError as error:
    raise ModuleNotFoundError('AutoPyTorch not installed') from error


class AutoPyTorchForecaster(Forecaster):
    """AutoPyTorch Forecaster."""

    name = 'AutoPyTorch'

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95

    presets = ['20', '40', '60', '80', '100']

    def forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        forecast_type: str,
        horizon: int,
        limit: int,
        frequency: str | int,
        tmp_dir: str | Path,
        nproc: int = 1,
        preset: str = '20',
        target_name: str | None = None,
        verbose: int = 1,
    ):
        """Perform time series forecasting.

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Ensemble size, defaults to 20
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :param int verbose: Verbosity, defaults to 1
        :return predictions: Numpy array of predictions
        """
        if forecast_type == 'univariate':
            freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]
            lag = 1
            X_train, y_train, X_test, y_test = self.create_tabular_dataset(
                train_df, test_df, horizon, target_name, tabular_y=False, lag=lag
            )

            if 'ISEM_prices' in str(tmp_dir):
                train_df.index = pd.to_datetime(train_df.index, format='%d/%m/%Y %H:%M')
                test_df.index = pd.to_datetime(test_df.index, format='%d/%m/%Y %H:%M')
            else:
                train_df.index = pd.to_datetime(train_df.index, unit='D')
                test_df.index = pd.to_datetime(test_df.index, unit='D')
            y_train = pd.Series(y_train, index=X_train.index)
            y_test = pd.Series(y_test, index=X_test.index)
        else:
            freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
            y_train = y_train.reset_index(drop=True)
            X_train = X_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

        horizon = y_test.shape[0]  # Limitation of AutoPyTorch

        api = TimeSeriesForecastingTask(ensemble_size=int(preset))

        api.search(
            X_train=[X_train],
            y_train=[copy.deepcopy(y_train)],
            X_test=[X_test],
            y_test=[copy.deepcopy(y_test)],
            freq=freq,
            n_prediction_steps=y_test.shape[0],
            n_jobs=nproc,
            memory_limit=32 * 1024,
            total_walltime_limit=limit,
            # min_num_test_instances=100, # proxy validation sets for the tasks with more than 100 series
            optimize_metric='mean_MASE_forecasting',
        )

        # To forecast values value after the X_train, ask datamanager to generate a test set
        test_sets = api.dataset.generate_test_seqs()
        predictions = api.predict(test_sets)[0]
        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :param str preset: Model configuration to use
        :return: Estimated time limit (int)
        """
        return int(time_limit * self.initial_training_fraction)
