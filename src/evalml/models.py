import numpy as np
import pandas as pd
from evalml.automl import AutoMLSearch

from src.base import Forecaster
from src.util import Utils


class EvalMLForecaster(Forecaster):

    name = 'EvalML'

    # Training configurations ordered from slowest to fastest
    presets = [ 'default', 'iterative' ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='default',
                 target_name=None):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str preset: Model configuration to use, defaults to 'default'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        train_df['time_index'] = pd.to_datetime(train_df.index)
        test_df['time_index'] = pd.to_datetime(test_df.index)
        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        # Split target from features
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        problem_config = {
            'gap': 0,
            'max_delay': horizon, # for feature engineering
            'forecast_horizon': horizon,
            'time_index': 'time_index'
        }

        automl = AutoMLSearch(
            X_train,
            y_train,
            allowed_model_families='regression',
            automl_algorithm=preset,
            problem_type='time series regression',
            problem_configuration=problem_config,
            max_time=limit,
            verbose=False,
        )
        automl.search()

        model = automl.best_pipeline
        predictions = self.rolling_origin_forecast(model, X_train, X_test, y_train, horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, train_X, test_X, y_train, horizon):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param train_X: Training feature data (pandas DataFrame)
        :param test_X: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(test_X, horizon)

        predictions = []
        for s in test_splits:
            preds = model.predict(s, objective=None, X_train=train_X, y_train=y_train).values
            predictions.append(preds)
            train_X = pd.concat([train_X, s])

        # Flatten predictions and truncate if needed
        predictions = np.concatenate([ p.flatten() for p in predictions ])
        predictions = predictions[:len(test_X)]
        return predictions