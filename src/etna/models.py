import os

from etna.auto import Auto
from etna.core import load
from etna.datasets.tsdataset import TSDataset
from etna.metrics import SMAPE
import pandas as pd

from src.base import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class ETNAForecaster(Forecaster):

    name = 'ETNA'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment

    presets = [ 'none' ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 target_name=None,
                 presets='none'):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

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

        # tail_steps = len(train_df)
        # future_steps = len(test_df)
        # train_df = format_dataframe(train_df)
        # test_df = format_dataframe(test_df)

        train_df = train_df.rename_axis('timestamp').reset_index()
        test_df = test_df.rename_axis('timestamp').reset_index()

        # train_df.to_csv('train_df.csv', index=False)
        train_df.to_csv('train_df_.csv')

        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
        # train_df.index = pd.DatetimeIndex(train_df.index).to_period(freq)
        df = TSDataset.to_dataset(train_df)
        ts = TSDataset(df, freq=freq)

        model_path = os.path.join(tmp_dir, 'pipeline.zip')
        # if not os.path.exists(model_path):
        auto = Auto(target_metric=SMAPE(), horizon=horizon, experiment_folder=tmp_dir)

        # Get best pipeline
        # _catboost.CatBoostError: C:/Go_Agent/pipelines/BuildMaster/catboost.git/catboost/libs/metrics/metric.cpp:6487: All train targets are equal
        # best_pipeline = auto.fit(ts, timeout=limit, catch=(Exception,))
        # best_pipeline = auto.fit(ts, timeout=limit, catch=())
        best_pipeline = auto.fit(ts, timeout=15, catch=())
        print('best_pipeline')
        print(best_pipeline)
        print(type(best_pipeline))
        print(dir(best_pipeline))
        print('best_pipeline.ts', best_pipeline.ts, type(best_pipeline.ts))

        best_pipeline.save(model_path)
        # else:
        #     best_pipeline = load(model_path, ts=ts)

        # future_ts = ts.make_future(future_steps=test_df.shape[0], tail_steps=best_pipeline.context_size)
        # future_ts = ts.make_future(future_steps=future_steps, tail_steps=tail_steps)
        # future_ts.to_pandas(True).to_csv('future_ts.csv')

        # ValueError: Pipeline is not fitted! Fit the Pipeline before calling forecast method.
        # predictions = best_pipeline.forecast(future_ts)
        predictions = best_pipeline.forecast()
        print('predictions')
        print(predictions)
        print(type(predictions))
        print(predictions.shape)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
