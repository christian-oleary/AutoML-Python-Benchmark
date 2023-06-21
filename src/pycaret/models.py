import numpy as np
import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from src.abstract import Forecaster
from src.util import Utils


class PyCaretForecaster(Forecaster):

    name = 'PyCaret'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        :return predictions: TODO
        """

        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        # Using the correct horizon causes a variety of internal library errors
        # during the forecasting stage (via predict_model())
        # TODO: Use the correct horizon if possible
        horizon = len(test_df)

        exp = TSForecastingExperiment()
        exp.setup(train_df,
                  target=target_name,
                  fh=horizon,
                  fold=2, # Lower folds prevents errors with short time series
                  numeric_imputation_target='ffill',
                  numeric_imputation_exogenous='ffill',
                  use_gpu=True,
                  )

        model = exp.compare_models(budget_time=limit)
        predictions = exp.predict_model(model, X=test_df.drop(target_name, axis=1), fh=horizon)
        predictions = predictions['y_pred'].values
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
