from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import numpy as np
import pandas as pd
from networkx.exception import NetworkXError

from src.base import Forecaster
from src.errors import AutomlLibraryError
from src.logs import logger
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils


class AutoGluonForecaster(Forecaster):

    name = 'AutoGluon'

    # Training configurations ordered from slowest/"best" to fastest/"worst"
    presets = [
        'best_quality',
        # The following often result in NetworkX errors
        'high_quality',
        'medium_quality',
        'fast_training'
        ]

    # Use 90% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.9


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset='fast_training',
                 target_name=None):
        """Perform time series forecasting using AutoGluon TimeSeriesPredictor

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Model configuration to use, defaults to 'fast_training'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return np.array: Predictions
        """

        logger.debug('Formatting indices...')
        # Format index
        timestamp_column = 'timestamp'
        try:
            train_df = train_df.reset_index(names=[timestamp_column])
            test_df = test_df.reset_index(names=[timestamp_column])
        except TypeError as e: # Pandas < 1.5.0
            train_df = train_df.rename_axis(timestamp_column).reset_index()
            test_df = test_df.rename_axis(timestamp_column).reset_index()

        # Format univariate data
        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [ timestamp_column, target_name ]
            test_df.columns = [ timestamp_column, target_name ]
            if 'ISEM_prices' in tmp_dir:
                ignore_time_index = False
            else:
                ignore_time_index = True
        else:
            ignore_time_index = False
            raise NotImplementedError()

        if ignore_time_index:
            logger.warning('The value of "ignore_time_index" is True. This will result in slower predictions')

        # AutoGluon requires an ID column
        train_df['ID'] = 1
        test_df['ID'] = 1

        logger.debug('Converting to TimeSeriesDataFrame...')
        # TimeSeriesDataFrame inherits from pandas.DataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_df, id_column='ID', timestamp_column=timestamp_column)
        test_data = TimeSeriesDataFrame.from_data_frame(test_df, id_column='ID', timestamp_column=timestamp_column)

        if forecast_type == 'global':
            # If frequency detection failed, fill manually
            # SUPPORTED_FREQUENCIES = {"D", "W", "M", "Q", "A", "Y", "H", "T", "min", "S"}
            freq = FREQUENCY_MAP[frequency].replace('1', '')
        elif forecast_type == 'univariate':
            if ignore_time_index:
                freq = 'S' # based on Libra R repository.
            else:
                freq = 'H' # I-SEM

        logger.debug('Index processing and imputation...')
        if train_data.freq == None:
            train_data = train_data.to_regular_index(freq=freq)

        if test_data.freq == None:
            test_data = test_data.to_regular_index(freq=freq)

        # Attempt to fill missing values
        if train_data.isnull().values.any():
            train_data = train_data.fill_missing_values()

        if test_data.isnull().values.any():
            test_data = test_data.fill_missing_values()

        # If imputation (partially) failed, fill missing data with zeroes
        if train_data.isnull().values.any():
            train_data = train_data.fillna(0)
            logger.warning('Autogluon failed to impute some training data data. Filling with zeros')

        if test_data.isnull().values.any():
            test_data = test_data.fillna(0)
            logger.warning('Autogluon failed to impute some test data data. Filling with zeros')

        # Create Predictor
        predictor = TimeSeriesPredictor(prediction_length=horizon,
                                        path=tmp_dir,
                                        target=target_name,
                                        ignore_time_index=ignore_time_index,
                                        verbosity=0,
                                        cache_predictions=True,
                                        eval_metric='sMAPE')

        try:
            logger.debug('Training AutoGluon...')
            # Train models
            predictor.fit(train_data, presets=preset, random_seed=limit, time_limit=limit)
            # Get predictions
            logger.debug('Making predictions...')
            predictions = self.rolling_origin_forecast(predictor, train_data, test_data, horizon, column='mean')
        except NetworkXError as error:
            raise AutomlLibraryError('AutoGluon failed to fit/predict due to NetworkX', error)

        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :param str preset: Model configuration to use
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)


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

        for i, s in enumerate(test_splits):
            X_train = pd.concat([X_train, s])
            X_train = X_train.get_reindexed_view(freq=X_test.freq)

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