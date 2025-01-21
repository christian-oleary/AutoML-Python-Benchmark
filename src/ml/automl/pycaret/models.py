"""PyCaret models"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pycaret.time_series import TSForecastingExperiment  # type: ignore
except ModuleNotFoundError as error:
    raise ModuleNotFoundError('PyCaret not installed') from error

from ml.base import Forecaster
from ml.logs import logger
from ml.TSForecasting.data_loader import FREQUENCY_MAP


class PyCaretForecaster(Forecaster):
    """Forecasting using PyCaret"""

    name = 'PyCaret'

    initial_training_fraction = 0.95  # Use 95% of max. time for training in initial experiment
    tuning_fraction = 0.85  # Use 95% of max. time for training in initial experiment

    presets = ['']

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
        preset: str = '',
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
        :param str preset: Model configuration to use, defaults to ''
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """
        if forecast_type == 'global':
            freq = FREQUENCY_MAP[frequency].replace('1', '')
            train_df.index = pd.to_datetime(train_df.index).to_period(freq)
            test_df.index = pd.to_datetime(test_df.index).to_period(freq)
        elif 'ISEM_prices' in str(tmp_dir):
            freq = 'H'
            train_df.index = pd.to_datetime(train_df.index)  # .to_period(freq)
            train_df.index = pd.date_range(
                start=train_df.index.min(), freq='H', periods=len(train_df)
            ).to_period(freq)

            test_df.index = pd.to_datetime(test_df.index)  # .to_period(freq)
            test_df.index = pd.date_range(
                start=test_df.index.min(), freq='H', periods=len(test_df)
            ).to_period(freq)

            # Drop irrelevant rows
            if forecast_type == 'univariate' and 'ISEM_prices' in str(tmp_dir):
                test_df['pycaret_datetime'] = test_df.index
                # test_df['pycaret_datetime'] = pd.to_datetime(test_df['pycaret_datetime'], errors='coerce')
                test_df = test_df[test_df['pycaret_datetime'].dt.hour == 0]
                test_df = test_df.drop('pycaret_datetime', axis=1)

        exp = TSForecastingExperiment()
        exp.setup(
            train_df,
            target=target_name,
            fh=horizon,
            fold=2,  # Lower folds prevents errors with short time series
            n_jobs=nproc,
            numeric_imputation_target='ffill',
            numeric_imputation_exogenous='ffill',
            use_gpu=True,
        )

        logger.debug('Training models...')
        model = exp.compare_models(budget_time=limit)

        # Produces worse results:
        # logger.debug('Tuning...')
        # time_remaining = time.time() - start_time
        # if time_remaining < (limit * self.tuning_fraction * 60):
        #     logger.debug('Tuning best model')
        #     model = exp.tune_model(model, budget_time=time_remaining)

        logger.debug('Making predictions...')
        if forecast_type == 'global':
            raise NotImplementedError()
            predictions = exp.predict_model(model, X=test_df.drop(target_name, axis=1), fh=horizon)
            predictions = predictions['y_pred'].values
        else:
            predictions = self.rolling_origin_forecast(
                exp, model, train_df, test_df, horizon, freq, column='y_pred'
            )
        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models.

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in minutes (int)
        """
        return int((time_limit / 60) * self.initial_training_fraction)

    def rolling_origin_forecast(self, exp, model, X_train, X_test, horizon, freq, column=None):
        """Iteratively forecast over increasing dataset.

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """
        # Make predictions
        predicted = exp.predict_model(model, X=X_train, fh=horizon)
        if column is not None:
            predicted = predicted[column].values[-horizon:]
        # logger.debug(f'predicted.shape: {predicted.shape}')
        predictions = [predicted]

        data = X_train
        for s in X_test.iterrows():
            data.index = data.index.to_timestamp()
            new_index = pd.date_range(
                start=data.index.min(), freq='H', periods=len(data) + 1
            ).to_period(freq)
            data.loc[len(data.index)] = s[1].values
            # data = pd.concat([data, s])
            data.index = new_index
            # data.index = pd.to_datetime(data.index)#.to_period(freq)
            # data.index = data.index.to_timestamp()
            # data.index = pd.date_range(start=data.index.min(), freq='H', periods=len(data)).to_period(freq)

            # logger.debug('data', data, type(data), data.shape)
            predicted = exp.predict_model(model, X=data, fh=horizon)
            if column is not None:
                predicted = predicted[column].values[-horizon:]

            predictions.append(predicted)

        # Flatten predictions and truncate if needed
        # logger.debug('len(predictions)', len(predictions))
        try:
            predictions = np.concatenate([p.flatten() for p in predictions])
        except AttributeError:
            predictions = np.concatenate([p.values.flatten() for p in predictions])
        return predictions
