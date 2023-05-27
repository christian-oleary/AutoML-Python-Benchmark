import copy

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask

from src.abstract import Forecaster
from src.TSForecasting.data_loader import FREQUENCY_MAP


class AutoPyTorchForecaster(Forecaster):

    name = 'AutoPyTorch'

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95

    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir):
        """Perform time series forecasting using AutoPyTorch

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Time limit in seconds (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        :return predictions: TODO
        """

        raise NotImplementedError()
        target_name = 'humidity'
        y_train = train_df[[target_name]]
        X_train = train_df.drop(target_name, axis=1)
        y_test = test_df[[target_name]]
        X_test = test_df.drop(target_name, axis=1)

        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')

        y_train = y_train.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        print('X_train', X_train, X_train.shape)
        # X_train.to_csv('X_train.csv')
        print('X_test', X_test, X_test.shape)
        # X_test.to_csv('X_test.csv')
        print('y_train', y_train, y_train.shape)
        # y_train.to_csv('y_train.csv')
        print('y_test', y_test, y_test.shape)
        # y_test.to_csv('y_test.csv')

        # from sktime.datasets import load_longley
        # targets, features = load_longley()

        # forecasting_horizon = 1
        # horizon=1

        # print('targets', targets, type(targets), targets.shape)
        # targets.to_csv('targets.csv')
        # print('features', type(features), features)
        # features.to_csv('features.csv')

        # # Dataset optimized by APT-TS can be a list of np.ndarray/ pd.DataFrame where each series represents an element in the
        # # list, or a single pd.DataFrame that records the series
        # # index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
        # # column
        # # Within each series, we take the last forecasting_horizon as test targets. The items before that as training targets
        # # Normally the value to be forecasted should follow the training sets
        # y_train = [targets[: -forecasting_horizon]]
        # y_test = [targets[-forecasting_horizon:]]
        # print('y_train', y_train, type(y_train), len(y_train), type(y_train[0]))
        # print('y_test', y_test, type(y_test), len(y_test), type(y_test[0]))

        # # same for features. For uni-variant models, X_train, X_test can be omitted and set as None
        # X_train = [features[: -forecasting_horizon]]
        # # Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
        # # could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
        # # we could also omit X_test
        # known_future_features = list(features.columns)
        # X_test = [features[-forecasting_horizon:]]
        # print('X_train', X_train, type(X_train), len(X_train), type(X_train[0]))
        # print('X_test', X_test, type(X_test), len(X_test), type(X_test[0]))

        # start_times = [targets.index.to_timestamp()[0]]
        # freq = '1Y'
        # exit()

        api = TimeSeriesForecastingTask()
        api.search(
            X_train=[X_train],
            y_train=[copy.deepcopy(y_train)],
            X_test=[X_test],
            y_test=[copy.deepcopy(y_test)],
            optimize_metric='mean_MASE_forecasting',
            n_prediction_steps=y_test.shape[0],
            memory_limit=16 * 1024,
            freq=freq,
            # start_times=start_times,
            func_eval_time_limit_secs=60,
            total_walltime_limit=limit,
            min_num_test_instances=100, # proxy validation sets for the tasks with more than 100 series
        )

        # To forecast values value after the X_train, ask datamanager to generate a test set
        test_sets = api.dataset.generate_test_seqs()

        predictions = api.predict(test_sets)[0]
        print('predictions', predictions, type(predictions), predictions.shape)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial time limit to use for TimeSeriesPredictor fit()

        :param time_limit: Maximum time allowed for AutoGluonForecaster.forecast() (int)
        :return: Estimated time limit (int)
        """

        return int(time_limit * self.initial_training_fraction)
