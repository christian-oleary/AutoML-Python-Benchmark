import os

from flaml import AutoML
import numpy as np
import pandas as pd

from src.abstract import Forecaster
from src.util import Utils


class FLAMLForecaster(Forecaster):

    name = 'FLAML'

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95

    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir, preset='auto'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        :param preset: Model configuration to use
        :return predictions: Numpy array of predictions
        """

        os.makedirs(tmp_dir, exist_ok=True)

        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        automl = AutoML()
        automl.fit(X_train=train_df.index.to_series().values,
                   y_train=train_df[target_name].values,
                   estimator_list=preset,
                   eval_method='auto',
                   log_file_name=os.path.join(tmp_dir, 'ts_forecast.log'),
                   period=horizon, # AssertionError: Model is optimized for horizon, length of X must be equal to `period`.
                   task='ts_forecast',
                   time_budget=limit, # 15
                   )

        predictions = self.rolling_origin_forecast(automl, train_df.index.to_series().to_frame(),
                                                   test_df.index.to_series().to_frame(), horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, X_train, X_test, horizon):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        preds = model.predict(X_train.tail(horizon))
        predictions = [ preds ]

        for s in test_splits:
            if len(s) < horizon:
                s = X_test.tail(horizon)
            preds = model.predict(s)
            predictions.append(preds)

        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([ p.flatten() for p in predictions ])
        except:
            predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        predictions = predictions[:len(X_test)]
        return predictions
