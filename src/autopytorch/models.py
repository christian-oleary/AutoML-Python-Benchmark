import copy

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils


class AutoPyTorchForecaster(Forecaster):

    name = 'AutoGluon'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/autopytorch'):
        """Perform time series forecasting using AutoGluon TimeSeriesPredictor

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Time limit in seconds (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        api = TimeSeriesForecastingTask()

        api.search(
            X_train=X_train,
            y_train=copy.deepcopy(y_train),
            X_test=X_test,
            optimize_metric='mean_MASE_forecasting',
            n_prediction_steps=horizon,
            memory_limit=16 * 1024,
            freq=frequency,
            # start_times=start_times,
            func_eval_time_limit_secs=50,
            total_walltime_limit=limit,
            min_num_test_instances=1000, # proxy validation sets for the tasks with more than 1000 series
            # known_future_features=known_future_features,
        )

        # To forecast values value after the X_train, ask datamanager to generate a test set
        test_sets = api.dataset.generate_test_seqs()

        predictions = api.predict(test_sets)
        print('horizon, predictions', horizon, predictions.shape)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
