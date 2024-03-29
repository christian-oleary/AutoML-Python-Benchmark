"""
Base Classes
"""

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from src.util import Utils
from src.logs import logger


class Forecaster:
    """Base Forecaster"""

    regression_models = {
        DummyRegressor.__name__: (DummyRegressor, {}),
        LGBMRegressor.__name__: (LGBMRegressor, {
            'verbosity': [ -1 ],
            # Based on Lynch et al. 2021:
            'learning_rate': [0.2, 0.1, 0.05, 0.025, 0.0125],
            'n_estimators': range(50, 501, 50),
            'max_depth': range(3, 14, 2),
            'subsample': np.arange(0.1, 1.1, 0.1),
            'colsample_bytree': np.arange(0.1, 1.1, 0.1),
            'num_leaves': [round(0.6*2**x) for x in range(3,14,2)],
            'min_child_samples': range(10, 71, 10)
        }),
        LinearRegression.__name__: (LinearRegression, {}),
        XGBRegressor.__name__: (XGBRegressor, {
            'verbosity': [ 0 ],
            # Based on Lynch et al. 2021:
            'learning_rate': [ 0.2, 0.1, 0.05, 0.025, 0.0125 ],
            'n_estimators': range(50, 500, 50),
            'max_depth': range(3, 14, 2),
            'subsample': np.arange(0.1, 1.1, 0.1),
            'colsample_bytree': np.arange(0.1, 1.1, 0.1)
        }),

    }

    presets = [ 'Naive', 'Constant' ] + list(regression_models.keys())

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset='LinearRegression',
                 target=None):
        """Perform time series forecasting

        :param pd.DataFrametrain_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, 'global', 'univariate' or 'multivariate'
        :param int horizon: Forecast horizon (how far ahead to predict) (int)
        :param int limit: Iterations limit (int)
        :param int frequency: Data frequency (str)
        :param str tmp_dir: Path to directory to store temporary files (str)
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Modelling presets
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        """

        if forecast_type == 'univariate':
            # Create tabular data
            if target is None:
                target = 'target'
            train_df.columns = [ target ]
            test_df.columns = [ target ]

            logger.debug('Formatting into tabular dataset...')
            lag = self.get_default_lag(horizon)
            X_train, y_train, X_test, y_test = self.create_tabular_dataset(train_df, test_df, horizon, target, lag=lag)

            # Fit model
            logger.debug(f'Training {preset} model...')
            if preset == 'Naive':
                predictions = X_test[f'{target}-24'].values
            elif preset == 'Constant':
                predictions = np.full(len(y_test), np.mean(y_train))
            else:
                X_train = X_train.tail(10000)
                y_train = y_train[-10000:]
                predictions = self.train_model(X_train, y_train, X_test, horizon, forecast_type, nproc, tmp_dir,
                                               model_name=preset)
        else:
            raise NotImplementedError('Base models not implemented for multivariate/global forecasting yet')
        return predictions


    def train_model(self, X_train, y_train, X_test, horizon, forecast_type, nproc, tmp_dir, model_name):
        # Scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        # Train model
        constructor, hyperparameters = self.regression_models[model_name]
        hyperparameters = { f'estimator__{k}': v for k, v in hyperparameters.items() }

        try:
            model = constructor(n_jobs=nproc)
        except: # If API does not have n_jobs
            model = constructor()

        model = MultiOutputRegressor(model)
        model = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_jobs=nproc, verbose=1,
                                   cv=10, scoring='neg_mean_absolute_error')
        model.fit(X_train, y_train)

        # Generate predictions
        if forecast_type == 'univariate':
            # model = Model(model)

            # Drop irrelevant rows
            if 'I-SEM' in tmp_dir or 'ISEM' in tmp_dir:
                X_test['base_datetime'] = X_test.index
                X_test['base_datetime'] = pd.to_datetime(X_test['base_datetime'], errors='coerce')
                X_test = X_test[X_test['base_datetime'].dt.hour == 0]
                X_test = X_test.drop('base_datetime', axis=1)

            X_test = scaler.transform(X_test)
            predictions = model.predict(X_test)
            predictions = predictions.flatten()
        else:
            raise NotImplementedError()
        return predictions


    def get_default_lag(self, horizon):
        """Get default lag/lookback for generating a sliding window of features

        :param int horizon: Prediction horizon
        :return int: Equals horizon
        """

        return horizon # horizon may be used (Monash 2021). Libra does not specify.


    def create_tabular_dataset(self, train_df, test_df, horizon, target_cols, tabular_y=True, lag=None):
        """Prepare training and test sets for tabular regression

        :param pd.DataFrame train_df: Training data
        :param pd.DataFrame test_df: Test data
        :param int horizon: Forecast horizon
        :param str or list target_cols: Name of target column(s)
        :param bool tabular_y: Target (y) returned with 'horizon' columns if true, as one column otherwise, defaults to False
        :param int lag: Lag/window size, defaults to None
        :return tuple: Tuple containing training features, training labels, test features
        """
        if lag == None and horizon == None:
            raise ValueError('Either lag or horizon must be provided')

        if lag == None:
            lag = self.get_default_lag(horizon)

        train_df, X_train, y_train = self.create_tabular_data(train_df, lag, horizon, target_cols, tabular_y)
        test_df, X_test, y_test = self.create_tabular_data(test_df, lag, horizon, target_cols, tabular_y)

        # Impute resulting missing values
        imputer = IterativeImputer(n_nearest_features=3, max_iter=5)
        y_train = imputer.fit_transform(y_train)
        y_test = imputer.transform(y_test)
        if not tabular_y:
            y_train = y_train.flatten()
            y_test = y_test.flatten()
        X_train = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
        return X_train, y_train, X_test, y_test


    def create_tabular_data(self, df, lag, horizon, targets, tabular_y, drop_original=False):
        """Prepare time series data for tabular regression

        :param pd.DataFrame df: Time series data
        :param int lag: Lag/window size
        :param int horizon: Forecast horizon
        :param str or list targets: Name of target column(s) to forecast
        :return pd.DataFrame(s): Dataframe, features and targets
        """
        if isinstance(targets, str):
            targets = [ targets ]

        if tabular_y:
            df, target_cols = self.create_future_values(df, horizon, targets)
        else:
            target_cols = targets

        df = self.create_lag_features(df, targets, target_cols, window_size=lag)

        # Drop rows missing future target values
        # df = df[df[targets[-1]].notna()]

        y = df[target_cols]
        X = df.drop(target_cols, axis=1)
        df.to_csv('df.csv')
        X.to_csv('X.csv')
        y.to_csv('y.csv')
        return df, X, y


    def create_lag_features(self, df, targets, target_cols, window_size, ignored=[]):
        """Create features based on historical feature values

        :param pd.DataFrame df: Input DataFrame
        :param list targets: List of names of target columns
        :param int window_size: Window/lag size
        :param list ignored: List of column names to ignore
        :return pd.DataFrame: DataFrame with columns replaced with lagged versions
        """
        for col in df.columns:
            if col not in ignored and (col not in target_cols or col in targets):
                for i in range(1, window_size+1):
                    df[f'{col}-{i}'] = df[col].shift(i)

                if col != targets[0]:
                    df = df.drop(col, axis=1)
        return df


    def create_future_values(self, df, horizon, targets):
        """Create a window of future values for multioutput forecasting
        """
        all_target_cols = []
        for target in targets:
            target_cols = [ target ]
            for i in range(1, horizon):
                col_name = f'{target}+{i}'
                df[col_name] = df[target].shift(-i)
                target_cols.append(col_name)
            all_target_cols = all_target_cols + target_cols
        return df, all_target_cols


    def estimate_initial_limit(self, time_limit, preset):
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
        raise NotImplementedError('Subclasses should implement rolling_origin_forecast()')


    def _rolling_origin_forecast(self, model, X_train, X_test, horizon, column=None):
        """DEPRECATED. Libraries should implement their own methods.

        Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """
        raise NotImplementedError()

        # Split test set
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        preds = model.predict(X_train)
        if column != None:
            preds = preds[column].values

        if len(preds.flatten()) > 0:
            preds = preds[-horizon:]

        predictions = [ preds ]

        for s in test_splits:
            X_train = pd.concat([X_train, s])

            preds = model.predict(X_train)
            if column != None:
                preds = preds[column].values

            if len(preds.flatten()) > 0:
                preds = preds[-horizon:]

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
        # df = pd.concat([X_train, X_test])
        # # Split test set
        # test_splits = Utils.split_test_set(X_test, step_size)
        # # Make predictions
        # preds = model.predict(X_train)
        # # preds.to_csv('preds.csv', index=False)
        # actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
        # # actual.to_csv('actual.csv', index=False)
        # predictions = [{ 'actual': actual, 'predicted': preds }]
        # for s in test_splits:
        #     X_train = pd.concat([X_train, s])
        #     # Need to retrain model here
        #     preds = model.predict(X_train)
        #     actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
        #     predictions.append([{ 'actual': actual, 'predicted': preds }])
        # return predictions


    def tabular_regression(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                           nproc=1,
                           preset='LinearRegression',
                           target_names=None):
        """Perform tabular regression

        :param pd.DataFrametrain_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, 'global', 'univariate' or 'multivariate'
        :param int horizon: Forecast horizon (how far ahead to predict) (int)
        :param int limit: Iterations limit (int)
        :param int frequency: Data frequency (str)
        :param str tmp_dir: Path to directory to store temporary files (str)
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Modelling presets
        :param list target_names: List of names of target variables, defaults to None which means all.
        """

        # Create tabular data
        if target_names is None:
            raise NotImplementedError()

        logger.debug('Formatting into tabular dataset...')
        lag = self.get_default_lag(horizon)
        X_train, y_train, X_test, y_test = self.create_tabular_dataset(train_df, test_df, horizon, target_names,
                                                                       lag=lag)

        # Fit model
        logger.debug(f'Training {preset} model...')
        if preset == 'Naive':
            predictions = X_test[[f'{t}-24' for t in target_names]].values
        elif preset == 'Constant':
            predictions = np.full(len(y_test), np.mean(y_train, axis=1))
        else:
            X_train = X_train.tail(10000)
            y_train = y_train[-10000:]
            predictions = self.train_model(X_train, y_train, X_test, horizon, forecast_type, nproc, tmp_dir,
                                            model_name=preset)
        return predictions

