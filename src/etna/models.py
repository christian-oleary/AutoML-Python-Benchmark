from etna.auto import Auto
from etna.datasets.tsdataset import TSDataset
from etna.metrics import SMAPE
import pandas as pd

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class ETNAForecaster(Forecaster):

    name = 'ETNA'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/etna'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        # import warnings
        # warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')

        try:
            train_df = train_df.reset_index(names=['timestamp'])
            test_df = test_df.reset_index(names=['timestamp'])
        except TypeError as e: # Pandas < 1.5.0
            train_df = train_df.rename_axis('timestamp').reset_index()
            test_df = test_df.rename_axis('timestamp').reset_index()

        train_df['target'] = train_df[target_name]
        test_df['target'] = test_df[target_name]

        train_df['segment'] = 'segment_1'

        # temp
        print(train_df.columns)
        train_df = train_df.drop('T1', axis=1)
        train_df = train_df.drop('T2', axis=1)
        train_df = train_df.drop('T3', axis=1)
        train_df = train_df.drop('T4', axis=1)
        train_df = train_df.drop('T5', axis=1)
        print(train_df.columns)

        freq = FREQUENCY_MAP[frequency].replace('1', '')
        df = TSDataset.to_dataset(train_df)
        ts = TSDataset(df, freq=freq) # KeyError: 'segment'

        ts.describe()

        auto = Auto(
            target_metric=SMAPE(),
            horizon=horizon,
            experiment_folder=tmp_dir,
        )

        # Get best pipeline
        best_pipeline = auto.fit(ts, catch=(Exception,))
        print(best_pipeline)

        future_ts = ts.make_future(future_steps=test_df.shape[0], tail_steps=best_pipeline.context_size)

        predictions = best_pipeline.forecast(future_ts, prediction_size=horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
