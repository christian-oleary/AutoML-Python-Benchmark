"""FLAML models."""

import os

import numpy as np
import pandas as pd

from ml.base import Forecaster
from ml.errors import DatasetTooSmallError
from ml.logs import logger
from ml.plots import Utils

try:
    from flaml import AutoML
except ModuleNotFoundError:
    raise ModuleNotFoundError('FLAML not installed')


class FLAMLForecaster(Forecaster):
    """FLAML Forecaster"""

    name = 'FLAML'

    # Use 99% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.99

    presets = ['auto']

    def forecast(
        self,
        train_df,
        test_df,
        forecast_type,
        horizon,
        limit,
        frequency,
        tmp_dir,
        nproc=1,
        preset='auto',
        target_name=None,
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
        :param str preset: Model configuration to use, defaults to 'auto'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """
        if len(test_df) <= horizon + 1:  # 4 = lags
            raise DatasetTooSmallError('Dataset too small for FLAML', ValueError())

        os.makedirs(tmp_dir, exist_ok=True)

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

            if 'ISEM_prices' in tmp_dir:
                train_df.index = pd.to_datetime(train_df.index, format='%d/%m/%Y %H:%M')
                train_df.index = pd.date_range(
                    start=train_df.index.min(), freq='H', periods=len(train_df)
                )
                test_df.index = pd.to_datetime(test_df.index, format='%d/%m/%Y %H:%M')

                # Not required as FLAML is using timestamps as features
                # test_df['flaml_datetime'] = test_df.index
                # test_df['flaml_datetime'] = pd.to_datetime(test_df['flaml_datetime'], errors='coerce')
                # test_df = test_df[test_df['flaml_datetime'].dt.hour == 0]
                # test_df = test_df.drop('flaml_datetime', axis=1)
                # test_df = test_df[test_df.index.dt.hour == 0]

            else:
                train_df.index = pd.to_datetime(train_df.index, unit='D')
                test_df.index = pd.to_datetime(test_df.index, unit='D')

            y_train = train_df[target_name]

        else:
            raise NotImplementedError()
            train_df.index = pd.to_datetime(train_df.index)
            test_df.index = pd.to_datetime(test_df.index)

        automl = AutoML()
        logger.debug('Training models...')
        automl.fit(
            X_train=train_df.index.to_series(name='ds').values,
            y_train=y_train,
            estimator_list=preset,
            eval_method='auto',
            log_file_name=os.path.join(tmp_dir, 'ts_forecast.log'),
            n_jobs=nproc,
            period=horizon,
            task='ts_forecast',
            time_budget=limit,  # seconds
            verbose=0,  # Higher = more messages
        )
        logger.debug('Training finished.')

        predictions = automl.predict(
            test_df.index.to_series(name='ds').values, period=horizon
        ).values

        # predictions = self.rolling_origin_forecast(automl, train_df.index.to_series(name='ds').values,
        #                                            test_df.index.to_series().to_frame(), horizon)
        # predictions = predictions[:len(test_df)].values
        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models.

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in seconds (int)
        """
        return int(time_limit * self.initial_training_fraction)

    def rolling_origin_forecast(self, model, X_train, X_test, horizon):
        """Iteratively forecast over increasing dataset.

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        preds = model.predict(X_train)[-horizon:]
        predictions = [preds]

        # predictions = []
        for s in test_splits:
            s = s.index.to_series(name='ds').values
            if len(s) < horizon:
                s = X_test.tail(horizon).index.to_series(name='ds').values
            preds = model.predict(s)[-horizon:]
            predictions.append(preds)

        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([p.flatten() for p in predictions])
        except AttributeError:
            predictions = np.concatenate([p.values.flatten() for p in predictions])
        predictions = predictions[: len(X_test)]
        return predictions
