import pandas as pd
from evalml.automl import AutoMLSearch

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class EvalMLForecaster(Forecaster):

    name = 'EvalML'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/evalml'):
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
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        freq = FREQUENCY_MAP[frequency].replace('1', '')
        problem_config = {
            'gap': 0,
            'max_delay': horizon, # for feature engineering
            'forecast_horizon': horizon,
            'time_index': frequency
        }

        automl = AutoMLSearch(
            X_train,
            y_train,
            problem_type='time series regression',
            max_batches=1,
            problem_configuration=problem_config,
            max_time=limit,
            automl_algorithm='iterative',
            # allowed_model_families=[
            #     'xgboost',
            #     'random_forest',
            #     'linear_model',
            #     'extra_trees',
            #     'decision_tree',
            # ],
            verbose=True
        )

        automl.search()

        pl = automl.best_pipeline
        predictions = pl.predict(X_test, objective=None, X_train=X_train, y_train=y_train)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
