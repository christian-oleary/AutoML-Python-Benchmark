import os
import itertools

from etna.auto import Auto
from etna.datasets.tsdataset import TSDataset
from etna.metrics import SMAPE
import numpy as np
import pandas as pd

from src.base import Forecaster
from src.errors import AutomlLibraryError
from src.logs import logger
from src.TSForecasting.data_loader import FREQUENCY_MAP

# Presets are every combination of the following:
# tune_size = ['1', '2', '3', '4', '5', '10', '20', '30', '40', '50', '100', '200', '300', '400', '500']
# n_trials = ['1', '2', '3', '4', '5', '10', '20', '30', '40', '50', '100', '200', '300', '400', '500']
# All produce the same result...
tune_size = ['5']
n_trials = ['5']
presets = list(itertools.product(tune_size, n_trials))
presets = [ '_'.join(p) for p in presets] + ['0_0']


class ETNAForecaster(Forecaster):

    name = 'ETNA'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment

    presets = presets

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset=presets[0],
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
        :param str preset: Unused. Included for API compatibility
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        logger.debug('Preprocessing...')
        # May be useful for global forecasting:
        # def format_dataframe(df):
        #     try:
        #         df = df.reset_index(names=['timestamp'])
        #     except TypeError as e: # Pandas < 1.5.0
        #         df = df.rename_axis('timestamp').reset_index()
        #     df['target'] = df[target_name]
        #     df = df.drop(target_name, axis=1)
        #     df['segment'] = 'segment_target'
        #     segments = []
        #     for i, col in enumerate(df.columns):
        #         if col not in ['target', 'segment', 'timestamp']:
        #             segment = df[[col]]
        #             segment.columns = ['target']
        #             segment['segment'] = f'segment_feature_{i}'
        #             segment['timestamp'] = df['timestamp']
        #             segments.append(segment)
        #             df = df.drop(col, axis=1)
        #             print(df.columns)
        #     df = pd.concat([df] + segments)
        #     return df
        # train_df = format_dataframe(train_df)
        # test_df = format_dataframe(test_df)
        # tail_steps = len(train_df)
        # future_steps = len(test_df)

        train_df = train_df.rename_axis('timestamp').reset_index()
        test_df = test_df.rename_axis('timestamp').reset_index()
        train_df["segment"] = 'main'
        test_df["segment"] = 'main'

        if frequency == 24:
            frequency = 'hourly'
        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
        df = TSDataset.to_dataset(train_df)
        ts = TSDataset(df, freq=freq)
        # ts.to_pandas(True).to_csv('ts.csv')

        # future_ts = ts.make_future(future_steps=test_df.shape[0], tail_steps=best_pipeline.context_size)
        # future_ts = ts.make_future(future_steps=future_steps, tail_steps=tail_steps)
        # future_ts = train_ts.make_future(future_steps=horizon, tail_steps=model.context_size)
        # ETNA creates gaps which results in errors due to missing values
        # if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir: # Drop all but one hour
        #     X_test['etna_datetime'] = pd.to_datetime(X_test['timestamp'], errors='coerce')
        #     X_test = X_test[X_test['etna_datetime'].dt.hour == 0]
        #     X_test = X_test.drop('etna_datetime', axis=1)

        # Shift data (lag) back by one period to prevent leakage
        X_test = pd.concat([train_df.tail(horizon), test_df.head(len(test_df)-horizon)], ignore_index=True)

        logger.debug(f'Training with ETNA ({preset})...')
        auto = Auto(target_metric=SMAPE(), horizon=horizon, experiment_folder=os.path.join(tmp_dir, preset))

        # Get best pipeline
        preset_parts = preset.split('_')
        tune_size = int(preset_parts[0])
        n_trials = int(preset_parts[1])

        best_pipeline = None
        predictions = None
        MAX_ATTEMPTS = 5
        for i in range(MAX_ATTEMPTS): # May take multiple attempts due to a variety of internal ETNA errors
            logger.warning(f'ATTEMPT {i+1} of {MAX_ATTEMPTS}')
            try:
                best_pipeline = auto.fit(ts, timeout=limit, tune_size=tune_size, n_trials=n_trials, n_jobs=nproc, catch=())
                assert best_pipeline is not None, f'ETNA training failed using preset: {preset}'
                logger.debug(f'Training finished, best_pipeline: {best_pipeline}')

                logger.debug('Rolling origin forecasting...')
                try:
                    predictions = self.rolling_origin_forecast(best_pipeline, X_test, horizon, freq)
                except Exception as e1:
                    logger.error('Rolling origin forecasting failed. Re-attempting..')
                    best_pipeline.fit(ts)
                    predictions = self.rolling_origin_forecast(best_pipeline, X_test, horizon, freq)

                assert predictions is not None
                logger.critical('SUCCESS. BREAKING')
                break
            except  Exception as e2:
                logger.debug('ETNA failed. Re-attempting training...')

        if predictions is None:
            raise AutomlLibraryError(f'ETNA failed with preset {preset}.', Exception)
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Unused. Included for API compatability
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, X_test, horizon, freq):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param ts_dataset: A TSDataset of test feature data
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """

        predictions = []
        for i in range(0, len(X_test), horizon):
            data = X_test.head(i) # returns pd.DataFrame
            ts = TSDataset(TSDataset.to_dataset(X_test), freq=freq)
            preds = model.forecast(ts).to_pandas().values
            predictions.append(preds)

        if len(data)+horizon < len(X_test): # May happen if horizon does not divide evenly into len(X_text)
            data = X_test.tail(horizon) # returns pd.DataFrame
            ts = TSDataset(TSDataset.to_dataset(X_test), freq=freq)
            preds = model.forecast(ts).to_pandas().values
            predictions.append(preds)

        # Flatten predictions
        # print('-> len(predictions)', len(predictions))
        try:
            predictions = np.concatenate([ p.flatten() for p in predictions ])
        except:
            predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        assert len(predictions) == len(X_test)
        # print('-> predictions.shape', predictions.shape)
        # predictions = predictions[:len(X_test)] # Truncate if needed
        # print('-> predictions.shape', predictions.shape)
        return predictions
