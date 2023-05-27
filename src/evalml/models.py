import pandas as pd
from evalml.automl import AutoMLSearch

from src.abstract import Forecaster


class EvalMLForecaster(Forecaster):

    name = 'EvalML'

    # Training configurations ordered from slowest to fastest
    presets = [ 'default_long', 'iterative_fast', 'default_fast' ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, preset, target_name, horizon, limit, frequency, tmp_dir):
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

        import warnings
        warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')

        train_df['time_index'] = pd.to_datetime(train_df.index)
        test_df['time_index'] = pd.to_datetime(test_df.index)
        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        # Split target from features
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        problem_config = {
            'gap': 0,
            'max_delay': horizon, # for feature engineering
            'forecast_horizon': horizon,
            'time_index': 'time_index'
        }

        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        automl_algorithm = preset.split('_')[0]
        automl = AutoMLSearch(
            X_train,
            y_train,
            allowed_model_families='regression',
            automl_algorithm=automl_algorithm,
            problem_type='time series regression',
            problem_configuration=problem_config,
            max_time=limit,
            verbose=False,
        )

        mode = preset.split('_')[1]
        automl.search(mode=mode)

        pl = automl.best_pipeline
        predictions = pl.predict(X_test, objective=None, X_train=X_train, y_train=y_train)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
