import os

from flaml import AutoML
import pandas as pd

from src.abstract import Forecaster


class FLAMLForecaster(Forecaster):

    name = 'FLAML'

    # Training configurations approximately ordered from slowest to fastest
    presets = [ 'best_quality', 'auto', 'gpu', 'stable', 'ts', 'fast_train' ]

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
        print('predictions', predictions.shape)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
