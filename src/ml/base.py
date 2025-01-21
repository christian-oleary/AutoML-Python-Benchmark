"""Base Classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    PassiveAggressiveRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from ml.logs import logger

# Check for optional dependencies
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
    logger.warning('lightgbm not installed, ignoring...')

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    logger.warning('xgboost not installed, ignoring...')


# fmt: off
regression_models: dict[str, tuple[object, dict]] = {
    BayesianRidge.__name__: (BayesianRidge, {
        'tol': [1e-2, 1e-3, 1e-4],
        'alpha_1': [1e-5, 1e-6, 1e-7],
        'alpha_2': [1e-5, 1e-6, 1e-7],
        'lambda_1': [1e-5, 1e-6, 1e-7],
        'lambda_2': [1e-5, 1e-6, 1e-7],
    }),
    DecisionTreeRegressor.__name__: (DecisionTreeRegressor, {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [8, 16, 32, 64, 128, None],
        'max_features': [1.0, 'sqrt', 'log2'],
    }),
    DummyRegressor.__name__: (DummyRegressor, {}),
    ExtraTreeRegressor.__name__: (ExtraTreeRegressor, {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [8, 16, 32, 64, 128, None],
        'max_features': [1.0, 'sqrt', 'log2'],
    }),
    ElasticNet.__name__: (ElasticNet, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1],
        'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1],
        'tol': [1e-5, 1e-4, 1e-3],
        'selection': ['cyclic', 'random'],
    }),
    GaussianProcessRegressor.__name__: (GaussianProcessRegressor, {
        'alpha': [1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
        'n_restarts_optimizer': [0, 1, 2, 3],
        'normalize_y': [True, False],
    }),
    KernelRidge.__name__: (KernelRidge, {
        # KRR uses squared error loss while support vector regression uses epsilon-insensitive loss
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'kernel': [
            'additive_chi2', 'chi2', 'linear', 'rbf', 'laplacian', 'sigmoid', 'cosine',
        ],
        'coef0': [0.0, 0.5, 1.0],
    }),
    KNeighborsRegressor.__name__: (KNeighborsRegressor, {
        'n_neighbors': list(range(1, 50)),
        'weights': ['uniform', 'distance'],
        'p': [2, 3, 4],
    }),
    Lasso.__name__: (Lasso, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'tol': [1e-2, 1e-3, 1e-4],
        'selection': ['cyclic', 'random'],
    }),
    LinearRegression.__name__: (LinearRegression, {}),
    LinearSVR.__name__: (LinearSVR, {
        'epsilon': [0.0, 0.5, 1.0],
        'tol': [1e-3, 1e-4, 1e-5],
        'C': [0.01, 0.1, 1.0, 10],
        'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
        'intercept_scaling': [0.001, 0.1, 1, 10],
        'max_iter': [500, 1000, 1500],
    }),
    MLPRegressor.__name__: (MLPRegressor, {
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (1, 1,), (2, 2,), (3, 3,), (4, 4,),
        ],
        'activation': ['relu'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [1, 0.1, 0.01, 0.001],
    }),
    NuSVR.__name__: (NuSVR, {
        'nu': [0.2, 0.4, 0.6, 0.8, 1.0],
        'C': [0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'sigmoid', 'linear'],
        'degree': [2, 3, 4, 5, 6],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0],
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
    }),
    PassiveAggressiveRegressor.__name__: (PassiveAggressiveRegressor, {
        'C': [0.001, 0.01, 0.1],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'epsilon': [0.001, 0.01, 0.1, 1.0],
        # 'average': [True, 10],
        'early_stopping': [True, False],
    }),
    RandomForestRegressor.__name__: (RandomForestRegressor, {
        'n_estimators': [10, 50, 100, 150, 200],
        'criterion': ['absolute_error', 'poisson', 'squared_error'],
        'max_depth': [16, 32, 64, 128, None],
        'max_features': [1.0, 'sqrt', 'log2'],
        'bootstrap': [True, False],
    }),
    Ridge.__name__: (Ridge, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'tol': [1e-2, 1e-3, 1e-4],
        'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr'],
    }),
    SGDRegressor.__name__: (SGDRegressor, {
        'loss': [
            'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
        ],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [1e-3, 1e-4, 1e-5],
        'l1_ratio': [0.0, 0.15, 0.5, 1.0],
        'max_iter': [100, 1000, 10000],
        'tol': [1e-2, 1e-3, 1e-4],
        'epsilon': [0.001, 0.01, 0.1, 1.0],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'early_stopping': [True, False],
    }),
    SVR.__name__: (SVR, {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0],
        'tol': [1e-3, 1e-4, 1e-5],
        'C': [0.01, 0.1, 1.0, 10, 100],
        'epsilon': [0.0, 0.5, 1.0],
        'shrinking': [True, False],
        'max_iter': [50, 100, 150, 200, 250, 300],
    }),
}
# fmt: on

# Add LightGBM model if available
info = "Can install with: `pip install -e .[lightgbm]`. Continuing..."
if Utils.attempt_import('lightgbm', extra_text=info):
    from lightgbm import LGBMRegressor
    _search_space = {
        'verbosity': [-1],
        # Based on Lynch et al. 2021:
        'learning_rate': [0.2, 0.1, 0.05, 0.025, 0.0125],
        'n_estimators': range(50, 501, 50),
        'max_depth': range(3, 14, 2),
        'subsample': np.arange(0.1, 1.1, 0.1),
        'colsample_bytree': np.arange(0.1, 1.1, 0.1),
        'num_leaves': [round(0.6 * 2**x) for x in range(3, 14, 2)],
        'min_child_samples': range(10, 71, 10),
    }
    regression_models['LGBMRegressor'] = (LGBMRegressor, _search_space)

# Add XGBoost model if available
if Utils.attempt_import('xgboost', extra_text=info.replace('lightgbm', 'xgboost')):
    from xgboost import XGBRegressor
    _search_space = {
        'verbosity': [0],
        # Based on Lynch et al. 2021:
        'learning_rate': [0.2, 0.1, 0.05, 0.025, 0.0125],
        'n_estimators': range(50, 500, 50),
        'max_depth': range(3, 14, 2),
        'subsample': np.arange(0.1, 1.1, 0.1),
        'colsample_bytree': np.arange(0.1, 1.1, 0.1),
    }
    regression_models['XGBRegressor'] = (XGBRegressor, _search_space)


class Forecaster:
    """Base Forecaster."""

    presets = ['Naive', 'Constant'] + list(regression_models.keys())

    def forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        forecast_type: str,
        horizon: int,
        limit: int,
        frequency: int | str,  # pylint: disable=unused-argument
        tmp_dir: str | Path,
        nproc: int = 1,
        preset: str = 'LinearRegression',
        target: str | None = None,
        verbose: int = 1,
    ):
        """Perform time series forecasting

        :param pd.DataFrame train_df: DataFrame of training data
        :param pd.DataFrame test_df: DataFrame of test data
        :param str forecast_type: Type of forecasting, 'global', 'univariate' or 'multivariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit for forecasting
        :param int frequency: Data frequency (included for API compatibility)
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Modelling presets, defaults to 'LinearRegression'
        :param str | None target: Name of target variable for multivariate forecasting, defaults to None
        :param int verbose: Verbosity
        """
        self.verbose = verbose
        if forecast_type == 'univariate':
            # Create tabular data
            if target is None:
                target = 'target'
            train_df.columns = [target]
            test_df.columns = [target]

            logger.debug('Formatting into tabular dataset...')
            X_train, y_train, X_test, y_test = self.create_tabular_dataset(
                train_df=train_df,
                test_df=test_df,
                horizon=horizon,
                target_cols=[target],
                lag=self.get_default_lag(horizon),
            )

            # Fit model
            logger.debug(f'Training {preset} model...')
            if preset == 'Naive':
                predictions = X_test[f'{target}-{horizon}'].values
            elif preset == 'Constant':
                predictions = np.full(len(y_test), np.mean(y_train))
            else:
                predictions = self.train_model(
                    X_train=X_train.tail(10000),
                    y_train=y_train[-10000:],
                    X_test=X_test,
                    forecast_type=forecast_type,
                    nproc=nproc,
                    tmp_dir=tmp_dir,
                    model_name=preset,
                )
        else:
            raise NotImplementedError(
                'Base models not implemented for multivariate/global forecasting yet'
            )
        return predictions

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        forecast_type: str,
        nproc: int,
        tmp_dir: str | Path,
        model_name: str,
    ) -> np.ndarray:
        """Train machine learning model

        :param pd.DataFrame X_train: Training features
        :param np.ndarray y_train: Training target values
        :param pd.DataFrame X_test: Test features
        :param str forecast_type: Type of forecasting (univariate, multivariate or global)
        :param int nproc: Number of processes
        :param str tmp_dir: Dir to store any temporary files
        :param str model_name: Name of model
        :raises NotImplementedError: Unsupported forecasting methods
        :return np.ndarray: Forecast predictions
        """
        # Scale data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        # Train model
        constructor, hyperparameters = regression_models[model_name]
        hyperparameters = {f'estimator__{k}': v for k, v in hyperparameters.items()}

        try:
            model = constructor(n_jobs=nproc)
        except TypeError:
            model = constructor()

        model = MultiOutputRegressor(model)
        model = RandomizedSearchCV(
            estimator=model,
            param_distributions=hyperparameters,
            n_jobs=nproc,
            verbose=self.verbose,
            cv=10,
            random_state=1,
            scoring='neg_mean_absolute_error',
        )
        model.fit(X_train, y_train)

        # Generate predictions
        if forecast_type == 'univariate':
            # Drop irrelevant rows
            if 'I-SEM' in str(tmp_dir) or 'ISEM' in str(tmp_dir):
                X_test['base_datetime'] = X_test.index
                X_test['base_datetime'] = pd.to_datetime(X_test['base_datetime'], errors='coerce')
                X_test = X_test[X_test['base_datetime'].dt.hour == 0]
                X_test = X_test.drop('base_datetime', axis=1)

            X_test = scaler.transform(X_test)
            predictions = model.predict(X_test)
            predictions = predictions.flatten()

        elif forecast_type == 'multivariate':
            raise NotImplementedError()
        elif forecast_type == 'global':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return predictions

    def get_default_lag(self, horizon: int):
        """Get default lag/look-back for generating a sliding window of features

        :param int horizon: Prediction horizon
        :return int: Equals horizon
        """
        return horizon  # horizon may be used (Monash 2021). Libra does not specify.

    def create_tabular_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        horizon: int,
        target_cols: list,
        tabular_y: bool = True,
        lag: int | None = None,
    ):
        """Prepare training and test sets for tabular regression

        :param pd.DataFrame train_df: Training data
        :param pd.DataFrame test_df: Test data
        :param int horizon: Forecast horizon
        :param str or list target_cols: Name of target column(s)
        :param bool tabular_y: Target returned with 'horizon' columns if true, as one column otherwise, default is False
        :param int lag: Lag/window size, defaults to None
        :return tuple: Tuple containing training features, training labels, test features
        """
        if lag is None and horizon is None:
            raise ValueError('Either lag or horizon must be provided')

        if lag is None:
            lag = self.get_default_lag(horizon)

        train_df, X_train, y_train = self.create_tabular_data(
            train_df, lag, horizon, target_cols, tabular_y
        )
        test_df, X_test, y_test = self.create_tabular_data(
            test_df, lag, horizon, target_cols, tabular_y
        )

        # Impute resulting missing values
        imputer = IterativeImputer(n_nearest_features=3, max_iter=5, random_state=1)
        y_train = imputer.fit_transform(y_train)
        y_test = imputer.transform(y_test)
        if not tabular_y:
            y_train = y_train.flatten()
            y_test = y_test.flatten()
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
        )
        X_test = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)
        return X_train, y_train, X_test, y_test

    def create_tabular_data(
        self, df: pd.DataFrame, lag: int, horizon: int, targets: list | str, tabular_y: bool
    ):
        """Prepare time series data for tabular regression

        :param pd.DataFrame df: Time series data
        :param int lag: Lag/window size
        :param int horizon: Forecast horizon
        :param str | list targets: Name of target column(s) to forecast
        :return pd.DataFrame(s): Dataframe, features and targets
        """
        if isinstance(targets, str):
            targets = [targets]

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

    def create_lag_features(
        self,
        df: pd.DataFrame,
        targets: list,
        target_cols: list,
        window_size: int,
        ignored: list | None = None,
    ) -> pd.DataFrame:
        """Create features based on historical feature values

        :param pd.DataFrame df: Input DataFrame
        :param list targets: List of names of target columns
        :param int window_size: Window/lag size
        :param list | None ignored: List of column names to ignore, defaults to None
        :return pd.DataFrame: DataFrame with columns replaced with lagged versions
        """
        if ignored is None:
            ignored = []

        lagged_cols = {}
        for col in df.columns:
            if col not in ignored and (col not in target_cols or col in targets):
                for i in range(1, window_size + 1):
                    lagged_cols[f'{col}-{i}'] = df[col].shift(i)

                if col != targets[0]:
                    df = df.drop(col, axis=1)

        new_cols = pd.concat(lagged_cols, axis=1, ignore_index=False)
        df = pd.concat((df, new_cols), axis=1, ignore_index=False)
        return df

    def create_future_values(self, df: pd.DataFrame, horizon: int, targets: list):
        """Create a window of future values for multioutput forecasting

        :param pd.DataFrame df: Input DataFrame
        :param int horizon: Forecast horizon
        :param list targets: List of target column names
        :return pd.DataFrame: DataFrame with future values appended
        """
        all_target_cols: list[str] = []
        future_cols = {}
        for target in targets:
            target_cols = [target]
            for i in range(1, horizon):
                col_name = f'{target}+{i}'
                target_cols.append(col_name)
                future_cols[col_name] = df[target].shift(i)
            all_target_cols = all_target_cols + target_cols

        new_cols = pd.concat(future_cols, axis=1, ignore_index=False)
        df = pd.concat((df, new_cols), axis=1, ignore_index=False)

        return df, all_target_cols

    def estimate_initial_limit(self, time_limit: int, preset: str) -> int:
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param int time_limit: Maximum time allowed for AutoGluonForecaster.forecast()
        :param str preset: Library preset (unused in Forecaster class)
        :return int: Estimated time limit
        """
        logger.debug(f'Estimating initial time limit for {preset}...')
        return time_limit

    def estimate_new_limit(self, time_limit, current_limit, duration, limit_type='time'):
        """Estimate what time/iterations limit to use

        :param time_limit: Required time limit for valid experiment
        :param current_limit: The limit used for the previous experiment
        :param duration: Duration in seconds of previous experiment
        :param limit_type: Limit type ("time" or "iterations")
        :return new_limit: New time/iterations limit
        """
        if duration <= time_limit:
            raise ValueError(
                f'Invalid call as last experiment was within time limit: {duration} <= {time_limit}'
            )

        if current_limit > time_limit:
            raise ValueError(
                f'current_limit is greater than time_limit: {duration} <= {time_limit}'
            )

        if limit_type == 'time':
            # Subtract overtime from training time
            new_limit = int(current_limit - (duration - current_limit))
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

    # def _rolling_origin_forecast(self, model, X_train, X_test, horizon, column=None):
    #     """DEPRECATED. Libraries should implement their own methods.

    #     Iteratively forecast over increasing dataset

    #     :param model: Forecasting model, must have predict()
    #     :param X_train: Training feature data (pandas DataFrame)
    #     :param X_test: Test feature data (pandas DataFrame)
    #     :param horizon: Forecast horizon (int)
    #     :param column: Specifies forecast column if dataframe outputted, defaults to None
    #     :return: Predictions (numpy array)
    #     """
    #     raise NotImplementedError()
    #     # Split test set
    #     test_splits = Utils.split_test_set(X_test, horizon)
    #     # Make predictions
    #     predictions = model.predict(X_train)
    #     if column is not None:
    #         predictions = predictions[column].values
    #     if len(predictions.flatten()) > 0:
    #         predictions = predictions[-horizon:]
    #     predictions = [predictions]
    #     for s in test_splits:
    #         X_train = pd.concat([X_train, s])
    #         predictions = model.predict(X_train)
    #         if column is not None:
    #             predictions = predictions[column].values
    #         if len(predictions.flatten()) > 0:
    #             predictions = predictions[-horizon:]
    #         predictions.append(predictions)
    #     # Flatten predictions and truncate if needed
    #     try:
    #         predictions = np.concatenate([p.flatten() for p in predictions])
    #     except AttributeError():
    #         predictions = np.concatenate([p.values.flatten() for p in predictions])
    #     predictions = predictions[:len(X_test)]
    #     return predictions

    # def libra_forecast(self, model, X_train, X_test, horizon, step_size):
    #     """DEPRECATED. Iteratively forecast over increasing dataset

    #     :param model: Forecasting model, must have predict()
    #     :param X_train: Training feature data (pandas DataFrame)
    #     :param X_test: Test feature data (pandas DataFrame)
    #     :param horizon: Forecast horizon (int)
    #     :param step_size: Step size, defaults to None
    #     :return: Predictions (numpy array)
    #     """
    #     raise NotImplementedError()
    #     df = pd.concat([X_train, X_test])
    #     # Split test set
    #     test_splits = Utils.split_test_set(X_test, step_size)
    #     # Make predictions
    #     predictions = model.predict(X_train)
    #     # predictions.to_csv('predictions.csv', index=False)
    #     actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
    #     # actual.to_csv('actual.csv', index=False)
    #     predictions = [{ 'actual': actual, 'predicted': predictions }]
    #     for s in test_splits:
    #         X_train = pd.concat([X_train, s])
    #         # Need to retrain model here
    #         predictions = model.predict(X_train)
    #         actual = df.iloc[len(X_train):len(X_train)+horizon, 0]
    #         predictions.append([{ 'actual': actual, 'predicted': predictions }])
    #     return predictions

    # def tabular_regression(
    #     self,
    #     train_df: pd.DataFrame,
    #     test_df: pd.DataFrame,
    #     forecast_type: str,
    #     horizon: int,
    #     limit: int,
    #     frequency: str,
    #     tmp_dir: str | Path,
    #     nproc: int = 1,
    #     preset: str = 'LinearRegression',
    #     target_names: list = None
    # ):
    #     """Perform tabular regression

    #     :param pd.DataFrame train_df: Dataframe of training data
    #     :param pd.DataFrame test_df: Dataframe of test data
    #     :param str forecast_type: Type of forecasting, 'global', 'univariate' or 'multivariate'
    #     :param int horizon: Forecast horizon (how far ahead to predict) (int)
    #     :param int limit: Iterations limit (int)
    #     :param int frequency: Data frequency (str)
    #     :param str | Path tmp_dir: Path to directory to store temporary files (str)
    #     :param int nproc: Number of threads/processes allowed, defaults to 1
    #     :param str preset: Modelling presets
    #     :param list target_names: List of names of target variables, defaults to None which means all.
    #     """

    #     # Create tabular data
    #     if target_names is None:
    #         raise NotImplementedError()

    #     logger.debug('Formatting into tabular dataset...')
    #     lag = self.get_default_lag(horizon)
    #     X_train, y_train, X_test, y_test = self.create_tabular_dataset(
    #         train_df, test_df, horizon, target_names, lag=lag)

    #     # Fit model
    #     logger.debug(f'Training {preset} model...')
    #     if preset == 'Naive':
    #         predictions = X_test[[f'{t}-24' for t in target_names]].values
    #     elif preset == 'Constant':
    #         predictions = np.full(len(y_test), np.mean(y_train, axis=1))
    #     else:
    #         X_train = X_train.tail(10000)
    #         y_train = y_train[-10000:]
    #         predictions = self.train_model(
    #             X_train, y_train, X_test, forecast_type, nproc, tmp_dir, model_name=preset)
    #     return predictions
