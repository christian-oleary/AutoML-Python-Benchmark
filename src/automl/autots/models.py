"""AutoTS models"""

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from autots import AutoTS, create_regressor

from src.automl.base import Forecaster
from src.automl.errors import DatasetTooSmallError
from src.automl.logs import logger

cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.addHandler(logging.NullHandler())
cmdstanpy_logger.propagate = False
cmdstanpy_logger.setLevel(logging.CRITICAL)

# Presets are every combination of the following:
configs = ['superfast', 'fast', 'fast_parallel', 'default', 'all']
time_limits = ['60', '120', '240', '480', '960', '1920']  # Minutes: 1, 2, 4, 8, 16, 32
presets = list(itertools.product(configs, time_limits))
presets = ['__'.join(p) for p in presets]


class AutoTSForecaster(Forecaster):

    name = 'AutoTS'

    # Training configurations ordered from slowest to fastest
    presets = presets

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
        preset: str = 'superfast__60',
        target_name: str | None = None,
        verbose: int = 1,
    ):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Model configuration to use, defaults to 'superfast'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :param int verbose: Verbosity, defaults to 1
        :return np.ndarray predictions: Model predictions
        """

        logger.debug('Preparing data...')
        if forecast_type == 'global':
            raise NotImplementedError()
            # freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
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
            if target_name is None:
                target_name = 'target'

            if 'ISEM_prices' in tmp_dir:
                train_df.index = pd.to_datetime(train_df.index, format='%d/%m/%Y %H:%M')
                test_df.index = pd.to_datetime(test_df.index, format='%d/%m/%Y %H:%M')
            else:
                train_df.index = pd.to_datetime(train_df.index, unit='D')
                test_df.index = pd.to_datetime(test_df.index, unit='D')

            train_df.columns = [target_name]
            test_df.columns = [target_name]

            X_train, _, X_test, __ = self.create_tabular_dataset(
                train_df, test_df, horizon, target_name, tabular_y=False
            )
            train_regressors = X_train[f'{target_name}-{horizon}']
            test_regressors = X_test[f'{target_name}-{horizon}']
            assert np.isfinite(X_train).all().all()
            assert np.isfinite(X_test).all().all()
            assert np.isfinite(train_regressors).all()
            assert np.isfinite(test_regressors).all()

        min_allowed_train_percent = 0.1
        if (horizon * min_allowed_train_percent) > int((train_df.shape[0] / 4) - horizon):
            raise DatasetTooSmallError('Time series is too short for AutoTS', ValueError())

        limit = int(limit)
        # Max generations and generation timeout (if converted back to seconds) should approxiamtely equal the limit
        # It can be slightly off due to rounding and the 0.95 modifier that accounts for miscellaneous processing
        max_generations = int(round((limit / int(preset.split('__')[1])) * 0.95, 0))
        generation_timeout = int(round((limit / max_generations) / 60, 0))  # In minutes
        logger.debug(
            f'Max. generations: {max_generations}, Generation timeout: {generation_timeout}'
        )

        model = AutoTS(
            ensemble=['auto'],
            frequency='infer',
            forecast_length=horizon,
            max_generations=max_generations,
            generation_timeout=generation_timeout,
            model_list=preset.split('__')[0],
            models_to_validate=0.2,
            n_jobs=nproc,
            prediction_interval=0.95,
            random_seed=limit,
            transformer_list='all',
            min_allowed_train_percent=min_allowed_train_percent,
            verbose=-1,
        )

        logger.debug('Training AutoTS...')

        if forecast_type == 'global':
            raise NotImplementedError()
        else:
            # Can randomly fail: https://github.com/winedarksea/AutoTS/issues/140
            for _ in range(3):
                try:
                    logger.debug(f'Fitting ({preset})...')
                    model = model.fit(train_df, future_regressor=train_regressors)
                    logger.debug(f'Predicting ({preset})...')
                    predictions = model.predict(
                        future_regressor=test_regressors, forecast_length=len(test_regressors)
                    ).forecast.values
                    assert np.isfinite(predictions).all()
                    break
                except Exception as e:
                    logger.error(e)
                    logger.error(f'Failed on fit attempt ({preset})')
                    model = model.fit(train_df, future_regressor=train_regressors)
                    predictions = model.predict(
                        future_regressor=test_regressors, forecast_length=horizon
                    ).forecast.values
                    assert np.isfinite(predictions).all()
                    break

        predictions = np.concatenate([p.flatten()[:horizon] for p in predictions][::horizon])
        if forecast_type == 'univariate':
            if 'ISEM_prices' not in tmp_dir:
                predictions = predictions[: len(test_regressors)]

        # # predictions = self.rolling_origin_forecast(model, test_df, test_regressors, horizon)
        # # predictions = model.predict(future_regressor=test_regressors, forecast_length=horizon).forecast.values
        # # AutoTS will predict at every step, but we are using a step gap of length=horizon
        # predictions = np.array(predictions).T.tolist()
        # predictions = predictions[::horizon] # i.e. only keep predictions with an interval=horizon
        # predictions = list(itertools.chain(*predictions)) # Flatten to a 1D list
        # predictions = np.array(predictions[:len(X_test)]) # Drop any extra unused predictions
        # assert np.isfinite(np.array(predictions)).all()

        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Trials limit (int)
        """
        # Estimates of how long a generation will take in seconds
        # divisors = {
        #     'superfast': 60, 'fast': 60,
        #     'fast_parallel': 1200, 'default': 1200, 'all': 1200
        # }
        # return math.ceil(int(time_limit / divisors[preset]))

        # print('preset', preset)
        # print('preset.split("__")', preset.split('__'))
        # return round((time_limit / int(preset.split('__')[1])) * 0.95, 2)

        # Calculated in forecasting function
        return time_limit

    def rolling_origin_forecast(self, model, test_X, test_regressors, horizon):
        """DEPRECATED. Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param test_X: Test feature data (pandas DataFrame)
        :param test_regressors:
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """
        # # Split test set
        # test_splits = Utils.split_test_set(test_X, horizon)
        # regressor_splits = Utils.split_test_set(test_regressors, horizon)
        # print('\n\n\n\n')
        # print('test_X', test_X)
        # print('test_regressors', test_regressors)
        # print('test_splits', test_splits)

        # # Make predictions
        # predictions = []
        # for s in regressor_splits:
        #     # train_X = pd.concat([train_X, s])
        #     if len(s) < horizon:
        #         horizon = len(s)
        #     print('\n\n')
        #     print(s, s.shape, horizon)
        #     print('\n\n')
        #     preds = model.predict(future_regressor=s, forecast_length=horizon).forecast.values
        #     predictions.append(preds)

        # # Flatten predictions and truncate if needed
        # predictions = np.concatenate([ p.flatten() for p in predictions ])
        # predictions = predictions[:len(test_X)]
        # return predictions
