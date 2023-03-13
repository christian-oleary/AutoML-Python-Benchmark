"""
Abstract Classes
"""

from abc import ABC, abstractmethod

class Forecaster(ABC):
    """Abstract Forecaster"""

    @abstractmethod
    def forecast(self, train_df, test_df, target_name, horizon):
        """Perform forecasting using AutoGluon TimeSeriesPredictor

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Time or iterations limit (int)
        :param tmp_dir: Path to directory to store temporary files (str)
        """


    @abstractmethod
    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """


    def estimate_new_limit(self, time_limit, current_limit, duration, limit_type='time'):
        """Estimate what time/interations limit to use

        :param time_limit: Required time limit for valid experiment
        :param current_limit: The limit used for the previous experiment
        :param duration: Duration in seconds of previous experiment
        :param limit_type: Limit type ("time" or "iterations")
        :return new_limit: New time/iterations limit
        """

        if duration <= time_limit:
            raise ValueError(f'Invalid call as last experiment was within time limit: {duration} <= {time_limit}')

        if current_limit > time_limit:
            raise ValueError(f'current_limit is greater than time_limit: {duration} <= {time_limit}')

        if limit_type == 'time':
            new_limit = int(current_limit - (duration - current_limit)) # Subtract overtime from training time
        else:
            raise NotImplementedError()

        return new_limit
