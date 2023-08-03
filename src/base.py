"""
Base Classes
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer

from src.util import Utils
from src.logs import logger


class Forecaster:
    """Base Forecaster"""

    presets = [ 'none' ]


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir, **kwargs):
        """Perform time series forecasting using a basic model

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param forecast_type: Type of forecasting, 'global', 'univariate' or 'multivariate'
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """
        if forecast_type == 'univariate':
            # Create tabular data

            target_col = 'target'
            train_df.columns = [ target_col ]
            test_df.columns = [ target_col ]
            X_train, y_train, X_test = self.create_tabular_dataset(train_df, test_df, horizon, target_col,
                                                                   frequency=frequency)

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)

            class Model:
                def __init__(self, model):
                    self.model = model

                def predict(self, X):
                    return self.model.predict(X)[-1]

            if forecast_type == 'univariate':
                model = Model(model)
                predictions = self.rolling_origin_forecast(model, X_train, X_test, horizon)
        else:
            raise NotImplementedError()
        return predictions


    def create_tabular_dataset(self, train_df, test_df, horizon, target_col, lag=None, frequency=None):
        """Prepare training and test sets for tabular regression

        :param pd.DataFrame train_df: Training data
        :param pd.DataFrame test_df: Test data
        :param int horizon: Forecast horizon
        :param str target_col: Name of target column
        :param int lag: Lag/window size, defaults to None
        :return tuple: Tuple containing training features, training labels, test features
        """
        if lag == None and frequency == None:
            raise ValueError('Either lag or frequency must be provided')

        if lag == None:
            lag = int(1.25*frequency) # If this fails, horizon may be used (Monash 2021). Libra does not specify.

        lag = 1

        train_df, X_train, y_train = self.create_tabular_data(train_df, lag, horizon, target_col)
        test_df, X_test, _ = self.create_tabular_data(test_df, lag, horizon, target_col)

        # Impute resulting missing values
        imputer = IterativeImputer() # ~ 1 min
        y_train = imputer.fit_transform(y_train)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
        return X_train, y_train, X_test


    def create_tabular_data(self, df, lag, horizon, target, drop_original=False):
        """Prepare time series data for tabular regression

        :param pd.DataFrame df: Time series data
        :param int lag: Lag/window size
        :param int horizon: Forecast horizon
        :param str target: Name of target column to forecast
        :return pd.DataFrame(s): Dataframe, features and targets
        """
        df, targets = self.create_future_values(df, horizon, target)
        df = self.create_lag_features(df, targets, window_size=lag)

        # Drop rows missing future target values
        # df = df[df[targets[-1]].notna()]

        y = df[targets]
        X = df.drop(targets, axis=1)
        return df, X, y


    def create_lag_features(self, df, targets, window_size, ignored=[]):
        """Create features based on historical feature values

        :param pd.DataFrame df: Input DataFrame
        :param list targets: List of names of target columns
        :param int window_size: Window/lag size
        :param list ignored: List of column names to ignore
        :return pd.DataFrame: DataFrame with columns replaced with lagged versions
        """
        for col in df.columns:
            if col not in ignored and col not in targets[1:]:
                for i in range(1, window_size+1):
                    df[f'{col}-{i}'] = df[col].shift(i)

                if col != targets[0]:
                    df = df.drop(col, axis=1)
        return df


    def create_future_values(self, df, horizon, target):
        """Create a window of future values for multioutput forecasting
        """
        targets = [ target ]
        for i in range(1, horizon):
            col_name = f'{target}+{i}'
            df[col_name] = df[target].shift(-i)
            targets.append(col_name)
        return df, targets


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """
        return time_limit


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


    def rolling_origin_forecast(self, model, X_train, X_test, horizon, column=None):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """

        # Split test set
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        preds = model.predict(X_train)
        if column != None:
            preds = preds[column].values
        predictions = [ preds ]

        for s in test_splits:
            X_train = pd.concat([X_train, s])

            preds = model.predict(X_train)
            if column != None:
                preds = preds[column].values

            predictions.append(preds)

        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([ p.flatten() for p in predictions ])
        except:
            predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        predictions = predictions[:len(X_test)]
        return predictions


    def libra_forecast(self, model, X_train, X_test, horizon, step_size):
        """DEPRECATED. Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param step_size: Step size, defaults to None
        :return: Predictions (numpy array)
        """
        raise NotImplementedError()
        df = pd.concat([X_train, X_test])
        # Split test set
        test_splits = Utils.split_test_set(X_test, step_size)
        # Make predictions
        preds = model.predict(X_train)
        # preds.to_csv('preds.csv', index=False)
        actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
        # actual.to_csv('actual.csv', index=False)
        predictions = [{ 'actual': actual, 'predicted': preds }]
        for s in test_splits:
            X_train = pd.concat([X_train, s])
            # Need to retrain model here
            preds = model.predict(X_train)
            actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
            predictions.append([{ 'actual': actual, 'predicted': preds }])
        return predictions
