import copy

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask

from src.base import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class AutoPyTorchForecaster(Forecaster):

    name = 'AutoPyTorch'

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95

    presets = [ '' ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 target_name=None,
                 presets=''):
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

        y_train = train_df[[target_name]]
        X_train = train_df.drop(target_name, axis=1)
        y_test = test_df[[target_name]]
        X_test = test_df.drop(target_name, axis=1)

        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')

        y_train = y_train.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        api = TimeSeriesForecastingTask(ensemble_size=50)

        api.search(
            X_train=[X_train],
            y_train=[copy.deepcopy(y_train)],
            X_test=[X_test],
            y_test=[copy.deepcopy(y_test)],
            freq=freq,
            n_prediction_steps=y_test.shape[0],
            # n_prediction_steps=horizon,
            memory_limit=32 * 1024,
            total_walltime_limit=limit,
            # min_num_test_instances=100, # proxy validation sets for the tasks with more than 100 series
            optimize_metric='mean_MASE_forecasting',
        )

        # To forecast values value after the X_train, ask datamanager to generate a test set
        test_sets = api.dataset.generate_test_seqs()
        predictions = api.predict(test_sets)[0]
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
