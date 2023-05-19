import os

import autokeras as ak
import pandas as pd

from src.abstract import Forecaster
from src.util import Utils


class AutoKerasForecaster(Forecaster):

    name = 'AutoKeras'


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/autokeras'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        import warnings
        warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')


        # Split target from features
        train_y = train_df[target_name]
        train_X = train_df.drop(target_name, axis=1)
        test_X = test_df.drop(target_name, axis=1)

        # Split train data into train and validation
        val_split = int(len(train_df) * 0.1)
        val_y = train_y[:val_split]
        val_X = train_X[:val_split]
        train_y = train_y[val_split:]
        train_X = train_X[val_split:]

        # Initialise forecaster
        clf = ak.TimeseriesForecaster(
            lookback=horizon,
            predict_from=1,
            predict_until=horizon,
            max_trials=limit,
            objective='val_loss',
            overwrite=False,
            directory=tmp_dir
        )

        model_path = os.path.join(tmp_dir, 'time_series_forecaster', 'graph')
        if not os.path.exists(model_path):
            # lookback must be divisable by batch size due to library bug:
            # https://github.com/keras-team/autokeras/issues/1720
            batch_size = None
            size = 8 # initial batch size
            while batch_size == None:
                if size >= horizon:
                    size = 1

                if (horizon / size).is_integer():
                    batch_size = size
                else:
                    size += 1

            # Train models
            clf.fit(
                x=train_X,
                y=train_y,
                validation_data=(val_X, val_y),
                batch_size=batch_size,
                verbose=0
            )

        # Predict with the best model
        df = pd.concat([train_X, test_X])
        predictions = clf.predict(df)
        predictions = predictions.flatten()
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return int(time_limit / 900) # Estimate a trial takes about 15 minutes
        return 1 # One trial
