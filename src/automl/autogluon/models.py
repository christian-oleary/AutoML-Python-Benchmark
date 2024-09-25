"""AutoGluon models."""

from pathlib import Path

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except ModuleNotFoundError as error:
    raise ModuleNotFoundError('AutoGluon not installed') from error
import numpy as np
import pandas as pd
from networkx.exception import NetworkXError

from src.automl.base import Forecaster
from src.automl.errors import AutomlLibraryError
from src.automl.logs import logger
from src.automl.TSForecasting.data_loader import FREQUENCY_MAP


class AutoGluonForecaster(Forecaster):
    """Forecasting using AutoGluon."""

    name = 'AutoGluon'

    # Training configurations ordered from slowest/"best" to fastest/"worst"
    presets = [
        'fast_training',
        'medium_quality',
        'high_quality',
        'best_quality',
    ]

    # Use 90% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.9

    def forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        forecast_type: str,
        horizon: int,
        limit: int,
        frequency: str | int,
        tmp_dir: str | Path,
        nproc: int = 1,
        preset: str = 'fast_training',
        target_name: str | None = None,
        verbose: int = 1,
    ):
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

        # Done after from_data_frame() calls instead (below)
        # if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
        #     test_df['autogluon_datetime'] = test_df.index
        #     test_df['autogluon_datetime'] = pd.to_datetime(test_df['autogluon_datetime'], errors='coerce')
        #     test_df = test_df[test_df['autogluon_datetime'].dt.hour == 0]
        #     test_df = test_df.drop('autogluon_datetime', axis=1)

        # Format index
        timestamp_column = 'timestamp'
        try:
            train_df = train_df.reset_index(names=[timestamp_column])
            test_df = test_df.reset_index(names=[timestamp_column])
        except TypeError:  # Pandas < 1.5.0
            train_df = train_df.rename_axis(timestamp_column).reset_index()
            test_df = test_df.rename_axis(timestamp_column).reset_index()

        # Format univariate data
        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [timestamp_column, target_name]
            test_df.columns = [timestamp_column, target_name]
            if 'ISEM_prices' in str(tmp_dir):
                ignore_time_index = False
            else:
                ignore_time_index = True
        else:
            ignore_time_index = False
            raise NotImplementedError()

        if ignore_time_index:
            logger.warning(
                'The value of "ignore_time_index" is True. This will result in slower predictions'
            )

        # AutoGluon requires an ID column
        train_df['ID'] = 1
        test_df['ID'] = 1

        logger.debug('Converting to TimeSeriesDataFrame...')
        # TimeSeriesDataFrame inherits from pandas.DataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column='ID', timestamp_column=timestamp_column
        )
        test_data = TimeSeriesDataFrame.from_data_frame(
            test_df, id_column='ID', timestamp_column=timestamp_column
        )

        # Drop irrelevant rows. This happens after the from_data_frame() calls as AutoGluon tries to (badly) impute
        # the missing values otherwise.
        if forecast_type == 'univariate' and 'ISEM_prices' in str(tmp_dir):
            test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column], errors='coerce')
            test_df = test_df[test_df[timestamp_column].dt.hour == 0]

        if forecast_type == 'global':
            # If frequency detection failed, fill manually
            # SUPPORTED_FREQUENCIES = {"D", "W", "M", "Q", "A", "Y", "H", "T", "min", "S"}
            freq = FREQUENCY_MAP[frequency].replace('1', '')
        elif forecast_type == 'univariate':
            if ignore_time_index:
                freq = 'S'  # based on Libra R repository.
            else:
                freq = 'H'  # I-SEM

        logger.debug('Index processing and imputation...')
        if train_data.freq is None:
            train_data = train_data.to_regular_index(freq=freq)

        if test_data.freq is None:
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
        predictor = TimeSeriesPredictor(
            prediction_length=horizon,
            path=tmp_dir,
            target=target_name,
            ignore_time_index=ignore_time_index,
            verbosity=0,
            cache_predictions=True,
            eval_metric='sMAPE',
            # eval_metric='AutoMLBenchmarkScorer',
        )

        # For debugging only, i.e. all except Naive model
        self.excluded_model_types = [
            'RecursiveTabular',
            'SeasonalNaive',
            'Theta',
            'AutoETS',
            'DeepAR',
            'TemporalFusionTransformer',
            'PatchTST',
            'DirectTabular',
            'AutoARIMA',
        ]

        try:
            logger.debug('Training AutoGluon...')
            # Train models
            predictor.fit(
                train_data,
                presets=preset,
                random_seed=limit,
                time_limit=limit,
                #   time_limit=100, excluded_model_types=self.excluded_model_types, # debugging
            )
            # Get predictions
            logger.debug('Making predictions...')
            predictions = self.rolling_origin_forecast(
                predictor, train_data, test_data, horizon, tmp_dir, column='mean'
            )
        except NetworkXError as error:
            raise AutomlLibraryError('AutoGluon failed to fit/predict due to NetworkX', error)

        if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
            # Re-use test_data indices for date filtering
            predictions = np.array(predictions)
            for i in range(len(predictions[0])):
                test_data[f't+{i}'] = predictions[:, i]
            test_data = test_data.drop(target_name, axis=1)

            # Drop irrelevant predictions
            predictions = []
            for index in test_df[timestamp_column].values:
                start = pd.Timestamp(index)
                end = start + pd.to_timedelta(1, unit='h')
                row = test_data.slice_by_time(start, pd.Timestamp(end))
                predictions.append(row.values)

        # Drop irrelevant rows
        if forecast_type == 'univariate':
            # logger.debug('0 test_df.shape', test_df.shape)
            # logger.debug('1 test_data.shape', test_data.shape)
            # logger.debug('2 len(predictions)', len(predictions))
            # logger.debug('3 predictions[0].shape', predictions[0].shape)
            # logger.debug('4 flatten', np.concatenate([ p.flatten() for p in predictions ]).shape)
            predictions = np.concatenate([p.flatten() for p in predictions])
            # predictions = np.concatenate([ p.flatten() for p in predictions ][::horizon])
            # logger.debug('6 predictions.shape', predictions.shape)
            # predictions = predictions[:len(test_df)]
            # logger.debug('7 predictions.shape', predictions.shape)
        else:
            raise NotImplementedError()

        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :param str preset: Model configuration to use
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)

    def rolling_origin_forecast(self, model, X_train, X_test, horizon, tmp_dir, column=None):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (Autogluon DataFrame)
        :param X_test: Test feature data (Autogluon DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """

        # Make predictions
        data = X_train[-horizon:]
        preds = model.predict(data)
        if column is not None:
            preds = preds[column].values[-horizon:]
        assert len(preds) == horizon
        predictions = [preds]

        # Split test set
        # test_splits = Utils.split_test_set(X_test, horizon)
        # for s in test_splits:
        #     s = s.head(1)
        #     data = pd.concat([data, s])

        df = pd.concat([X_train, X_test])
        df = df.get_reindexed_view(freq=X_test.freq)
        predictions = []

        for i in range(len(X_test)):
            # s = X_test.iloc[[i]]
            # data = pd.concat([data, s])
            # data = data.get_reindexed_view(freq=X_test.freq)

            data = df.tail(len(df) - i)

            preds = model.predict(data[-horizon:])
            if column is not None:
                preds = preds[column].values[-horizon:]

            assert len(preds) == horizon
            # predictions.append(preds)
            predictions.insert(0, preds)

        return predictions
