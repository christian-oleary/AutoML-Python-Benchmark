from etna.auto import Auto
from etna.datasets.tsdataset import TSDataset
from etna.metrics import SMAPE
import pandas as pd

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class ETNAForecaster(Forecaster):

    name = 'ETNA'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir):
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

        def format_dataframe(df):
            try:
                df = df.reset_index(names=['timestamp'])
            except TypeError as e: # Pandas < 1.5.0
                df = df.rename_axis('timestamp').reset_index()

            df['target'] = df[target_name]
            df = df.drop(target_name, axis=1)
            df['segment'] = 'segment_target'

            segments = []
            for i, col in enumerate(df.columns):
                if col not in ['target', 'segment', 'timestamp']:
                    segment = df[[col]]
                    segment.columns = ['target']
                    segment['segment'] = f'segment_feature_{i}'
                    segment['timestamp'] = df['timestamp']
                    segments.append(segment)
                    df = df.drop(col, axis=1)
                    print(df.columns)
            df = pd.concat([df] + segments)
            return df

        tail_steps = len(train_df)
        future_steps = len(test_df)
        train_df = format_dataframe(train_df)
        test_df = format_dataframe(test_df)

        train_df.to_csv('train_df.csv', index=False)

        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
        # train_df.index = pd.DatetimeIndex(train_df.index).to_period(freq)
        df = TSDataset.to_dataset(train_df)
        ts = TSDataset(df, freq=freq)

        auto = Auto(target_metric=SMAPE(), horizon=horizon, experiment_folder=tmp_dir)

        # Get best pipeline
        # best_pipeline = auto.fit(ts, timeout=limit, catch=(Exception,))
        best_pipeline = auto.fit(ts, timeout=limit, catch=())
        # Errors:
        # 1. CatBoost
        # _catboost.CatBoostError: C:/Go_Agent/pipelines/BuildMaster/catboost.git/catboost/libs/metrics/metric.cpp:6487: All train targets are equal
        # 2. statsmodels
        # File "C:\sw\AutoML-Python-Benchmark\env\lib\site-packages\statsmodels\tsa\holtwinters\model.py", line 257, in __init__
        # ValueError: seasonal_periods has not been provided and index does not have a known freq. You must provide seasonal_periods
        print('best_pipeline')
        print(best_pipeline)
        print(type(best_pipeline))

        # future_ts = ts.make_future(future_steps=test_df.shape[0], tail_steps=best_pipeline.context_size)
        # future_ts = ts.make_future(future_steps=future_steps, tail_steps=tail_steps)
        # future_ts.to_pandas(True).to_csv('future_ts.csv')

        # ValueError: Pipeline is not fitted! Fit the Pipeline before calling forecast method.
        predictions = best_pipeline.forecast()
        # predictions = best_pipeline.forecast(future_ts, prediction_size=horizon)
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
