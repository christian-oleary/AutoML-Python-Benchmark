import logging

import numpy as np
import pandas as pd
from autots import AutoTS, create_regressor

from src.base import Forecaster
from src.errors import DatasetTooSmallError
from src.logs import logger
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class AutoTSForecaster(Forecaster):

    name = 'AutoTS'

    # Training configurations ordered from slowest to fastest
    presets = [ 'all', 'default', 'fast_parallel', 'fast', 'superfast' ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='superfast',
                 target_name=None):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str preset: Model configuration to use, defaults to 'all'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """


        if forecast_type == 'global':
            raise NotImplementedError()
            freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
            train_regressors, regressor_forecast = create_regressor(
                train_df,
                forecast_length=horizon,
                frequency='infer',
                scale=True,
                summarize='auto',
                backfill='bfill',
                fill_na='spline',
            )
        else:
            train_df.index = pd.to_datetime(train_df.index, unit='D')
            test_df.index = pd.to_datetime(test_df.index, unit='D')
            # We need to pass future_regressor to be able to do rolling origin forecasting
            train_regressors = train_df

        min_allowed_train_percent = 0.1
        model = AutoTS(
            ensemble=['auto'],
            frequency='infer',
            forecast_length=horizon,
            max_generations=limit,
            model_list=preset,
            models_to_validate=0.2,
            n_jobs=1,
            prediction_interval=0.95,
            random_seed=limit,
            transformer_list='all',
            min_allowed_train_percent=min_allowed_train_percent,
            verbose=-1,
        )

        logger.debug('Training model...')

        if (horizon * min_allowed_train_percent) > int((train_df.shape[0]/4) - horizon):
            raise DatasetTooSmallError('Time series is too short for AutoTS', ValueError())

        if forecast_type == 'global':
            raise NotImplementedError()
        else:
            # Can randomly fail: https://github.com/winedarksea/AutoTS/issues/140
            try:
                model = model.fit(train_df, future_regressor=train_regressors)
            except:
                model = model.fit(train_df, future_regressor=train_regressors)

        logger.debug('Making predictions...')
        predictions = self.rolling_origin_forecast(model, train_df, test_df, horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return (time_limit / 600) # Estimate a generation takes 10 minutes
        return 1 # One GA generation


    def rolling_origin_forecast(self, model, train_X, test_X, horizon):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param train_X: Training feature data (pandas DataFrame)
        :param test_X: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(test_X, horizon)

        # Make predictions
        predictions = []
        for s in test_splits:
            # train_X = pd.concat([train_X, s])
            if len(s) < horizon:
                horizon = len(s)
            preds = model.predict(future_regressor=s, forecast_length=horizon).forecast.values
            predictions.append(preds)

        # Flatten predictions and truncate if needed
        predictions = np.concatenate([ p.flatten() for p in predictions ])
        predictions = predictions[:len(test_X)]
        return predictions
