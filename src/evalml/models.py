import numpy as np
import pandas as pd
from evalml.automl import AutoMLSearch

from src.base import Forecaster
from src.util import Utils
from src.errors import DatasetTooSmallError


class EvalMLForecaster(Forecaster):

    name = 'EvalML'

    # Training configurations ordered from slowest to fastest
    # presets = [ 'default', 'iterative' ]
    presets = [ 'default', ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
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
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Model configuration to use, defaults to 'default'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        # Prepare data
        if target_name == None:
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

        lag = 1
        X_train, y_train, X_test, _ = self.create_tabular_dataset(train_df, test_df, horizon, target_name,
                                                                  tabular_y=False, lag=lag)

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        if forecast_type == 'global':
            raise NotImplementedError()

        if 'price_ROI_DA' in tmp_dir:
            X_train['time_index'] = pd.to_datetime(X_train.index, format='%d/%m/%Y %H:%M')
            X_test['time_index'] = pd.to_datetime(X_test.index, format='%d/%m/%Y %H:%M')
            y_train = pd.Series(y_train, index=X_train.index)
        else:
            freq = 'D'
            X_train['time_index'] = pd.to_datetime(X_train.index, unit=freq)
            X_test['time_index'] = pd.to_datetime(X_test.index, unit=freq)
            y_train = pd.Series(y_train)

        problem_config = {
            'gap': 0,
            'max_delay': horizon, # for feature engineering
            'forecast_horizon': horizon,
            'time_index': 'time_index'
        }

        eval_size = horizon * 3 # as n_splits=3
        train_size = len(train_df) - eval_size
        window_size = problem_config['gap'] + problem_config['max_delay'] + horizon
        if train_size <= window_size:
            raise DatasetTooSmallError('Time series is too short for EvalML. Must be > 5*horizon',  e)

        automl = AutoMLSearch(
            X_train,
            y_train,
            allowed_model_families=[ 'regression' ],
            automl_algorithm=preset,
            problem_type='time series regression',
            problem_configuration=problem_config,
            max_time=limit,
            n_jobs=nproc,
            verbose=False,
        )

        automl.search()
        model = automl.best_pipeline
        predictions = self.rolling_origin_forecast(model, X_train, X_test, y_train, horizon, forecast_type)
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, train_X, test_X, y_train, horizon, forecast_type):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param train_X: Training feature data (pandas DataFrame)
        :param test_X: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(test_X, horizon)

        predictions = []
        for s in test_splits:
            if horizon > len(s): # Pad with zeros to prevent errors with ARIMA
                padding = horizon - len(s)
                s = pd.concat([s, pd.DataFrame([s.values[0].tolist()] * padding, columns=s.columns)])
                start_index = s.index.values[0]
                s.index = np.arange(start_index, start_index + len(s))
                if forecast_type == 'univariate':
                    s['time_index'] = pd.date_range(start=s['time_index'].values[0], periods=len(s))

                preds = model.predict(s, objective=None, X_train=train_X, y_train=y_train).values
                preds = preds[:len(s)] # Drop placeholder predictions
            else:
                preds = model.predict(s, objective=None, X_train=train_X, y_train=y_train).values
            predictions.append(preds)
            train_X = pd.concat([train_X, s])

        # Flatten predictions and truncate if needed
        predictions = np.concatenate([ p.flatten() for p in predictions ])
        predictions = predictions[:len(test_X)]
        return predictions