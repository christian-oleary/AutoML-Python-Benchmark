import warnings
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils


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

        try:
            train_df = train_df.reset_index(names=['timestamp'])
            test_df = test_df.reset_index(names=['timestamp'])
        except TypeError as e: # Pandas < 1.5.0
            train_df = train_df.rename_axis('timestamp').reset_index()
            test_df = test_df.rename_axis('timestamp').reset_index()

        train_df['ID'] = 1
        test_df['ID'] = 1

        # TimeSeriesDataFrame inherits from pandas.DataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_df, id_column='ID', timestamp_column='timestamp')
        test_data = TimeSeriesDataFrame.from_data_frame(test_df, id_column='ID', timestamp_column='timestamp')

        # If frequency detection failed, fill manually
        # SUPPORTED_FREQUENCIES = {"D", "W", "M", "Q", "A", "Y", "H", "T", "min", "S"}
        freq = FREQUENCY_MAP[frequency].replace('1', '')
        if train_data.freq == None:
            train_data = train_data.to_regular_index(freq=freq)

        if test_data.freq == None:
            test_data = test_data.to_regular_index(freq=freq)

        # Attempt to fill missing values
        train_data = train_data.fill_missing_values()
        test_data = test_data.fill_missing_values()

        # If imputation (partially) failed, fill missing data with zeroes
        if train_data.isna().any().any():
            train_data = train_data.fillna(0)
            Utils.logger.warning('Autogluon failed to impute some training data data. Filling with zeros')

        if test_data.isna().any().any():
            test_data = test_data.fillna(0)
            Utils.logger.warning('Autogluon failed to impute some test data data. Filling with zeros')

        # Create Predictor
        predictor = TimeSeriesPredictor(prediction_length=horizon, path=tmp_dir, target=target_name,
                                        ignore_time_index=True,
                                        verbosity=0,
                                        eval_metric='sMAPE')

        # Train models
        # predictor.fit(train_data, presets='best_quality', time_limit=limit)
        # predictor.fit(train_data, presets='high_quality', time_limit=limit)
        # predictor.fit(train_data, presets='medium_quality', time_limit=limit)
        # predictor.fit(train_data, presets='fast_training', time_limit=limit)
        predictor.fit(train_data, presets='fast_training', time_limit=30)

        # Get predictions
        predictions = predictor.predict(train_data) # forecast
        predictions = predictions['mean'] # other values available for probabilistic forecast
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
