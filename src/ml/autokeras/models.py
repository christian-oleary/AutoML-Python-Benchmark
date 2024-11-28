"""AutoKeras models"""

import itertools
import os
from pathlib import Path
import shutil

import autokeras as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from src.ml.base import Forecaster
from src.ml.errors import AutomlLibraryError
from src.ml.logs import logger
from src.ml.util import Utils
from src.ml.validation import Task

# Presets are every combination of the following:
optimizers = ['hyperband', 'greedy', 'bayesian', 'random']
min_delta = ['0', '1', '2', '4', '8', '16', '32', '64', '128', '256']  # 0 = no early stopping
num_epochs = ['1000']  # default
num_trials = ['100']  # default
presets = list(itertools.product(num_trials, num_epochs, optimizers, min_delta))
presets = ['_'.join(p) for p in presets]


class AutoKerasForecaster(Forecaster):

    name = 'AutoKeras'

    # Training configurations (not ordered)
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
        preset: str = 'greedy_32_60',
        target_name: str = None,
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
        :param str preset: Model configuration to use
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :param int verbose: Verbosity, defaults to 1
        :return predictions: Numpy array of predictions
        """

        # Cannot use tmp_dir due to internal bugs with AutoKeras
        original_tmp_dir = tmp_dir
        tmp_dir = 'time_series_forecaster'
        shutil.rmtree(tmp_dir, ignore_errors=True)

        self.forecast_type = forecast_type
        if self.forecast_type == Task.UNIVARIATE_FORECASTING.value:
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]
            lag = 1  # AK has lookback
            X_train, y_train, X_test, _ = self.create_tabular_dataset(
                train_df, test_df, horizon, target_name, tabular_y=False, lag=lag
            )
        else:
            raise NotImplementedError()

        limit = int(limit)
        trials = int(preset.split('_')[0])
        epochs = int(preset.split('_')[1])
        optimizer = preset.split('_')[2]
        min_delta = int(preset.split('_')[3])
        tmp_dir = os.path.join(
            tmp_dir, f'{optimizer}_{min_delta}delta_{epochs}epochs_{trials}trials_{limit}'
        )

        # Initialise forecaster
        lookback = self.get_default_lag(horizon)
        params = {
            # 'directory': tmp_dir, # Internal errors with AutoKeras
            'lookback': lookback,
            'max_trials': trials,
            'objective': 'val_loss',
            'overwrite': False,
            'predict_from': 1,
            'predict_until': horizon,
            'seed': limit,
            'tuner': optimizer,
        }
        clf = ak.TimeseriesForecaster(**params)
        logger.debug(params)

        # "lookback" must be divisable by batch size due to library bug:
        # https://github.com/keras-team/autokeras/issues/1720
        # Start at 1024 as batch size and decrease until a factor is found
        # Counting down prevents unnecessarily small batch sizes being selected
        batch_size = None
        size = 1024  # Prospective batch size
        while batch_size is None:
            if (lookback / size).is_integer():  # i.e. is a factor
                batch_size = size
            else:
                size -= 1
        logger.debug(f'Calculated batch size as {batch_size}')

        # Create validation set
        x_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=int(limit)
        )

        # Callbacks
        callbacks = []
        if min_delta > 0:
            early_stopping = EarlyStopping(
                monitor='val_mean_squared_error',
                patience=3,
                min_delta=min_delta,
                verbose=1,
                mode='auto',
            )
            callbacks.append(early_stopping)

        # Train models
        logger.info(f'Fitting AutoKeras with preset {preset}...')
        clf.fit(
            x=x_train,
            y=y_train,
            # validation_split=0.2, # Internal errors
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs,
            verbose=0,
        )

        logger.info(f'Rolling origin forecast (preset: {preset})...')
        predictions = self.rolling_origin_forecast(
            clf, X_train, X_test, horizon, forecast_type, original_tmp_dir
        )
        if len(predictions) == 0:
            raise AutomlLibraryError('AutoKeras failed to produce predictions', ValueError())
        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Included for API compatibility

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Trials limit (int)
        """
        # return int(time_limit / int(preset.split('_')[0]))
        return time_limit

    def rolling_origin_forecast(self, model, X_train, X_test, horizon, forecast_type, tmp_dir):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """

        # Split test set
        # if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
        #     X_test['autokeras_datetime'] = X_test.index
        #     X_test['autokeras_datetime'] = pd.to_datetime(X_test['autokeras_datetime'], errors='coerce')
        #     X_test = X_test[X_test['autokeras_datetime'].dt.hour == 0]
        #     X_test = X_test.drop('autokeras_datetime', axis=1)
        #     test_splits = Utils.split_test_set(X_test, 1)
        # else:
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        data = X_train
        preds = model.predict(data)[-1]
        assert len(preds.flatten()) > 0
        predictions = [preds]

        for i, s in enumerate(test_splits):
            logger.debug(f'{i+1} of {len(test_splits)}')
            data = pd.concat([data, s])
            preds = model.predict(data, verbose=0)

            # AutoKeras can produce empty predictions on first inference (?)
            # Update: Only occurs trials = 1 and epochs = 1
            # if len(preds.flatten()) == 0:
            #     preds = model.predict(data)

            if len(preds) > horizon:
                preds = preds[-horizon:]

            predictions.append(preds)

        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([p.flatten() for p in predictions])
        except AttributeError:
            predictions = np.concatenate([p.values.flatten() for p in predictions])
        predictions = predictions[: len(X_test)]
        return predictions
