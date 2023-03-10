import warnings
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class AutoGluonForecaster(Forecaster):

    name = 'AutoGluon'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/autogluon'):
        """Perform forecasting using AutoGluon TimeSeriesPredictor

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Time limit in seconds (int)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        train_df = train_df.reset_index(names=['timestamp'])
        test_df = test_df.reset_index(names=['timestamp'])
        train_df['ID'] = 1
        test_df['ID'] = 1

        # TimeSeriesDataFrame inherits from pandas.DataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_df, id_column='ID', timestamp_column='timestamp')
        test_data = TimeSeriesDataFrame.from_data_frame(test_df, id_column='ID', timestamp_column='timestamp')

        # Fill missing values must be called manually first
        train_data = train_data.fill_missing_values()
        test_data = test_data.fill_missing_values()

        # train_data = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
        # test_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv")
        # target_name = "target"

        predictor = TimeSeriesPredictor(prediction_length=horizon, path=tmp_dir, target=target_name,
                                        ignore_time_index=True,
                                        eval_metric='sMAPE')

        # predictor.fit(train_data, presets='best_quality', time_limit=limit)
        # predictor.fit(train_data, presets='high_quality', time_limit=limit)
        # predictor.fit(train_data, presets='fast_training', time_limit=limit)
        predictor.fit(train_data, presets='fast_training', time_limit=10)

        predictions = predictor.predict(train_data) # forecast
        print('predictions', predictions, type(predictions))

        result = predictor.evaluate(test_data)
        print('result', result, type(result))

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
