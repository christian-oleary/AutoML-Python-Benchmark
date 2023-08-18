import os
import shutil

import autokeras as ak
import pandas as pd
from sklearn.impute import IterativeImputer
import tensorflow as tf

from src.base import Forecaster


class AutoKerasForecaster(Forecaster):

    name = 'AutoKeras'

    # Training configurations (not ordered)
    presets = ['greedy', 'bayesian', 'hyperband', 'random']


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='greedy'):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param preset: Model configuration to use
        :return predictions: Numpy array of predictions
        """

        # Cannot use tmp_dir due to internal bugs with AutoKeras
        tmp_dir = 'time_series_forecaster'
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [ target_name ]
            test_df.columns = [ target_name ]
            lag = 1 # AK has lookback
            X_train, y_train, X_test, _ = self.create_tabular_dataset(train_df, test_df, horizon, target_name,
                                                                      tabular_y=False, lag=lag)
        else:
            raise NotImplementedError()

        epochs = 100 # AK default
        tmp_dir = os.path.join(tmp_dir, f'{preset}_{epochs}epochs')

        # Initialise forecaster
        params = {
            # 'directory': tmp_dir, # Internal errors with AutoKeras
            'lookback': self.get_default_lag(horizon),
            'max_trials': limit,
            'objective': 'val_loss',
            'overwrite': False,
            'predict_from': 1,
            'predict_until': horizon,
            'seed': limit,
            'tuner': preset,
        }
        clf = ak.TimeseriesForecaster(**params)

        # "lookback" must be divisable by batch size due to library bug:
        # https://github.com/keras-team/autokeras/issues/1720
        # Start at 512 as batch size and decrease until a factor is found
        # Counting down prevents unnecessarily small batch sizes being selected
        batch_size = None
        size = 512 # Prospective batch size
        while batch_size == None:
            if (horizon / size).is_integer(): # i.e. is a factor
                batch_size = size
            else:
                size -= 1

        # Train models
        clf.fit(
            x=X_train,
            y=y_train,
            # validation_split=0.2, # Internal errors
            validation_data=(X_train, y_train),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )

        predictions = self.rolling_origin_forecast(clf, X_train, X_test, horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return int(time_limit / 900) # Estimate a trial takes about 15 minutes
        return 1 # One trial
