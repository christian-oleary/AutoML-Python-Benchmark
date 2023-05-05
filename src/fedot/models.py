from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
import pandas as pd

from src.abstract import Forecaster


class FEDOTForecaster(Forecaster):

    name = 'FEDOT'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/fedot'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        # Specify the task and the forecast length
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))

        # Initialize for the time-series forecasting
        model = Fedot(problem='ts_forecasting', task_params=task.task_params,
                      use_input_preprocessing=True,
                    #   timeout=limit,
                      timeout=0.5,
                    #   preset='auto',
                      preset='fast_train',
                      seed=1,
                      )

        # Split target from features
        import warnings
        warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        # Fit models
        model.fit(X_train, y_train)

        # use model to obtain out-of-sample forecast with one step
        predictions = model.forecast(X_test)

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
