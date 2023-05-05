from pycaret.time_series import *
from pycaret.time_series import TSForecastingExperiment
import pandas as pd

from src.abstract import Forecaster


class PyCaretForecaster(Forecaster):

    name = 'PyCaret'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/pycaret'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        train_df.index = pd.to_datetime(train_df.index)

        exp = TSForecastingExperiment()
        exp.setup(train_df,
                  target=target_name,
                  fh=horizon,
                  numeric_imputation_target='ffill',
                  numeric_imputation_exogenous='ffill',
                  )

        # exp.check_stats()

        # best is an empty list! Issue opened at 04/05/2023
        # WIP...
        # https://github.com/pycaret/pycaret/issues/3530
        best = exp.compare_models(budget_time=limit)
        print('best', best)

        predictions = exp.predict_model(best, fh=horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
