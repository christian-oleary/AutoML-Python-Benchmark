import warnings
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils


class AutoGluonForecaster(Forecaster):

    name = 'AutoGluon'

    # Training configurations ordered from slowest/"best" to fastest/"worst"
    presets = [ 'best_quality', 'high_quality', 'medium_quality', 'fast_training' ]

    # Use 90% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.9


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
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
        :param str preset: Model configuration to use, defaults to 'fast_training'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return np.array: Predictions
        """

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

        # AutoGluon requires an ID column
        train_df['ID'] = 1
        test_df['ID'] = 1

        # TimeSeriesDataFrame inherits from pandas.DataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(train_df, id_column='ID', timestamp_column=timestamp_column)
        test_data = TimeSeriesDataFrame.from_data_frame(test_df, id_column='ID', timestamp_column=timestamp_column)

        if forecast_type == 'global':
            # If frequency detection failed, fill manually
            # SUPPORTED_FREQUENCIES = {"D", "W", "M", "Q", "A", "Y", "H", "T", "min", "S"}
            freq = FREQUENCY_MAP[frequency].replace('1', '')
        elif forecast_type == 'univariate':
            freq = 'S' # based on Libra R repository.

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
        predictor = TimeSeriesPredictor(prediction_length=horizon,
                                        path=tmp_dir,
                                        target=target_name,
                                        ignore_time_index=True,
                                        verbosity=0,
                                        eval_metric='sMAPE')

        # Train models
        predictor.fit(train_data, presets=preset, random_seed=limit, time_limit=limit)

        # Get predictions
        # predictions = predictor.predict(train_data) # forecast
        # predictions = predictions['mean'].values # other values available for probabilistic forecast
        predictions = self.rolling_origin_forecast(predictor, train_data, test_data, horizon, column='mean')
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
